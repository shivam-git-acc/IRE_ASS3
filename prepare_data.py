#!/usr/bin/env python3
"""Stage 0 — download the datasets and compute exact ground truth.

Full mode  : SIFT-1M and GloVe-100 from ann-benchmarks, 10,000 queries each, plus nested
             100K/300K SIFT subsets for Part 3.
Quick mode : SIFT only, first 100,000 vectors, 1,000 queries, 25K/50K subsets.

GloVe is unit-normalised so squared-L2 ranking equals cosine ranking; the whole project then
uses one metric and one exact baseline. The neighbour ids stored in each HDF5 file are kept
only as an independent cross-check on the ground truth we compute ourselves.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annbench_core as C

URLS = {
    "sift": ("https://ann-benchmarks.com/sift-128-euclidean.hdf5", False),
    "glove": ("https://ann-benchmarks.com/glove-100-angular.hdf5", True),
}
GT_K = 100
SUBSET_SEED = 20261023


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  cached: {os.path.basename(dest)} ({C.human(os.path.getsize(dest))})")
        return dest
    print(f"  downloading {url}", flush=True)
    r = subprocess.run(["wget", "-c", "-q", "--show-progress", "--progress=dot:giga",
                        "-O", dest, url])
    if r.returncode != 0 or not os.path.exists(dest):
        raise SystemExit(f"download failed: {url}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import h5py
    C.ensure_dirs()
    env = C.env_fingerprint()
    datasets = ["sift"] if a.quick else ["sift", "glove"]
    n_cap = 100_000 if a.quick else None
    nq = 1_000 if a.quick else 10_000
    subsets = [25_000, 50_000] if a.quick else [100_000, 300_000]
    C.set_split(nq)
    print(f"\nmode: {'QUICK' if a.quick else 'FULL'} · datasets {datasets} · "
          f"n_cap {n_cap} · queries {nq} · subsets {subsets}")

    manifest = {"quick": a.quick, "datasets": datasets, "n_cap": n_cap, "n_queries": nq,
                "subsets": subsets, "subset_seed": SUBSET_SEED, "gt_k": GT_K,
                "split": {"tune": [C.TUNE.start, C.TUNE.stop],
                          "eval": [C.EVAL.start, C.EVAL.stop]},
                "env": env, "gt_check": {}, "relative_contrast": {}}

    for name in datasets:
        url, normalise = URLS[name]
        print(f"\n[{name}]")
        path = fetch(url, f"{C.TMP}/{os.path.basename(url)}")
        with h5py.File(path, "r") as f:
            base = np.ascontiguousarray(f["train"][:n_cap], dtype=np.float32)
            query = np.ascontiguousarray(f["test"][:nq], dtype=np.float32)
            stored = np.ascontiguousarray(f["neighbors"][:nq], dtype=np.int32)
        if normalise:
            base, query = C.normalise(base), C.normalise(query)
            print(f"  unit-normalised (norms "
                  f"{np.linalg.norm(base, axis=1).min():.6f}–"
                  f"{np.linalg.norm(base, axis=1).max():.6f})")
        print(f"  base {base.shape} · queries {query.shape}")

        t0 = time.perf_counter()
        gt_ids, gt_dists = C.exact_gt(base, query, k=GT_K)
        secs = time.perf_counter() - t0

        np.save(f"{C.TMP}/data/{name}_base.npy", base)
        np.save(f"{C.TMP}/data/{name}_query.npy", query)
        np.save(f"{C.TMP}/data/{name}_stored_neighbors.npy", stored)
        np.save(f"{C.TMP}/gt/{name}_top100_ids.npy", gt_ids)
        np.save(f"{C.TMP}/gt/{name}_top100_dists.npy", gt_dists)

        assert C.recall_from_ids(gt_ids[:, :10], base, query, gt_dists).mean() == 1.0
        if n_cap is None:
            # The file's neighbour ids index the FULL base, so this cross-check is only
            # defined when the base has not been truncated. In quick mode they point past
            # the end of the array, which is why it is skipped rather than approximated.
            rec_file = float(C.recall_from_ids(stored[:, :10], base, query, gt_dists).mean())
            print(f"  ground truth in {secs:.1f}s · "
                  f"recall@10 of the dataset's own neighbours vs ours: {rec_file:.6f}")
            assert rec_file > 0.999, "our ground truth disagrees with the dataset file"
        else:
            rec_file = None
            print(f"  ground truth in {secs:.1f}s · agreement with the dataset's own "
                  f"neighbours is not checked in quick mode (the base is truncated to "
                  f"{n_cap:,}, so those ids no longer exist)")

        msq = C.mean_sq_dist_to_random(base, query, n_sample=min(100_000, len(base)), seed=0)
        rc10 = float(C.relative_contrast(gt_dists, msq, k=10).mean())
        manifest["gt_check"][name] = dict(seconds=round(secs, 1), recall_file_vs_ours=rec_file)
        manifest["relative_contrast"][name] = dict(
            rc_at_1=float(C.relative_contrast(gt_dists, msq, k=1).mean()), rc_at_10=rc10)
        print(f"  relative contrast RC@10 {rc10:.3f}")
        del base, query, stored, gt_ids, gt_dists

    # Nested subsets for Part 3.
    print("\n[subsets for Part 3]")
    base = np.load(f"{C.TMP}/data/sift_base.npy")
    query = np.load(f"{C.TMP}/data/sift_query.npy")
    perm = np.random.default_rng(SUBSET_SEED).permutation(len(base))
    prev = None
    for n in subsets:
        tag = f"sift{n // 1000}k"
        ids = np.sort(perm[:n]).astype(np.int32)
        np.save(f"{C.TMP}/data/{tag}_ids.npy", ids)
        sub = np.ascontiguousarray(base[ids])
        gi, gd = C.exact_gt(sub, query, k=GT_K, verbose=False)
        np.save(f"{C.TMP}/gt/{tag}_top100_ids.npy", gi)
        np.save(f"{C.TMP}/gt/{tag}_top100_dists.npy", gd)
        assert gi.max() < n and C.recall_from_ids(gi[:, :10], sub, query, gd).mean() == 1.0
        cur = set(ids.tolist())
        assert prev is None or prev <= cur, "subsets are not nested"
        prev = cur
        print(f"  {tag}: n={n:,} · mean d10 {np.sqrt(gd[:, 9]).mean():.2f}")
        del sub, gi, gd

    # Record the fingerprint of the machine that produced the measurements, so the report
    # can attribute them correctly even when the figures are rendered somewhere else.
    os.makedirs(f"{C.OUT}/env", exist_ok=True)
    with open(f"{C.OUT}/env/{env['machine_id']}.json", "w") as f:
        json.dump(env, f, indent=2)

    with open(f"{C.OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {C.OUT}/manifest.json · data in {C.TMP}")

    if C.hf_enabled():
        tok = C.get_hf_token()
        for nm in datasets:
            for sub, fn in (("data", f"{nm}_base.npy"), ("data", f"{nm}_query.npy"),
                            ("data", f"{nm}_stored_neighbors.npy"),
                            ("gt", f"{nm}_top100_ids.npy"), ("gt", f"{nm}_top100_dists.npy")):
                C.hf_upload(f"{C.TMP}/{sub}/{fn}", f"{sub}/{fn}", token=tok, quiet=True)
        C.hf_upload(f"{C.OUT}/manifest.json", "manifest.json", token=tok, quiet=True)
        print("mirrored to Hugging Face")


if __name__ == "__main__":
    main()
