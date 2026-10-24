#!/usr/bin/env python3
"""Assemble nb00_setup_data.ipynb, embedding annbench_core.py into a %%writefile cell."""

import json
import os
import pathlib

SRC = pathlib.Path(__file__).parent
CORE = (SRC.parent.parent / "annbench_core.py").read_text()
TESTS = (SRC.parent.parent / "tests" / "test_recall.py").read_text()
OUT = SRC.parent / "nb00_setup_data.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


cells = [

md(r"""
# A3 · nb00 — environment & data preparation

**CS4.406 Assignment 3: Benchmarking Approximate Nearest Neighbour Search**

This notebook produces every input the later notebooks consume. It does no benchmarking.

1. fingerprints the machine and pins library versions
2. downloads **SIFT-1M** (128-d, Euclidean) and **GloVe-100** (1.18M, angular) from ann-benchmarks
3. unit-normalises GloVe so squared-L2 ordering equals cosine ordering
4. recomputes the exact top-100 ground truth on the T4, **in float64**, and cross-checks it
   against the neighbours stored in each HDF5 file
5. draws nested 100K / 300K SIFT subsets for the Part 3 scaling experiment and computes their GT
6. mirrors everything to `hf://donbosoc/ire-a3-ann`

### Before you run

- **Settings → Accelerator → GPU T4 x2**
- **Settings → Internet → On**
- **Add-ons → Secrets →** add your Hugging Face write token, labelled `HF_TOKEN`, and attach it

Runtime is about 35–50 minutes, most of it the two downloads and the ~1 GB upload.

### Why the GPU is here

It is *not* speeding up the benchmark. The assignment mandates single-thread,
one-query-at-a-time CPU timing, so every measurement in nb01–nb05 runs on `faiss-cpu`
pinned to one thread. The T4 only computes brute-force ground truth — for the normalised
GloVe vectors, the Part 3 subsets and the Part 4 synthetic sets — turning hours of CPU
work into seconds.
"""),

md("## 1 · Install and import"),

code(r"""
import subprocess, sys

def pip(*args):
    print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                         capture_output=True, text=True).stdout[-500:] or "ok")

try:
    import faiss
    print(f"faiss already present: {faiss.__version__}")
except ImportError:
    pip("faiss-cpu")
    import faiss
    print(f"faiss installed: {faiss.__version__}")

for mod, pkg in [("h5py", "h5py"), ("huggingface_hub", "huggingface_hub")]:
    try:
        __import__(mod)
    except ImportError:
        pip(pkg)

import h5py, huggingface_hub
print(f"h5py {h5py.__version__} · huggingface_hub {huggingface_hub.__version__}")
"""),

md("""
## 2 · The shared core

Every notebook in this project writes an identical copy of `annbench_core.py`, so each one
runs standalone with no cross-notebook imports. It holds the tie-aware recall metric, the
timing harness, the GPU ground-truth routine and the results-table writer.
"""),

code("%%writefile annbench_core.py\n" + CORE),

code("%%writefile test_recall.py\n" + TESTS),

md("""
## 3 · Machine fingerprint and Hugging Face auth

Kaggle VMs are shared and the CPU model can change between sessions, so the fingerprint
goes into every row of `results.csv` and the report states it plainly rather than claiming a
quiet dedicated machine. **Check that `machine_id` stays the same across nb01–nb05** — if it
changes, cross-dataset QPS comparisons need a re-run or an explicit caveat.
"""),

code(r"""
import os, sys, json, time, gc, hashlib, subprocess
import numpy as np

sys.path.insert(0, os.getcwd())
import annbench_core as C

C.ensure_dirs()
ENV = C.env_fingerprint()

with open(f"{C.OUT}/env/{ENV['machine_id']}.json", "w") as f:
    json.dump(ENV, f, indent=2)

TOKEN = C.get_hf_token()
C.hf_ensure_repo(TOKEN)
print(f"\nHF repo ready: https://huggingface.co/datasets/{C.HF_REPO}")
print(f"scratch: {C.TMP}   output: {C.OUT}")
"""),

md("""
### Sanity-check the recall metric right away

The full unit-test discussion belongs to Part 1 (nb01), but a broken recall function would
silently poison the ground-truth cross-check below, so the tests run here too.
"""),

code(r"""
print(subprocess.run([sys.executable, "-m", "pytest", "-q", "test_recall.py"],
                     capture_output=True, text=True).stdout[-2000:])
"""),

md("""
## 4 · Download the datasets

HF mirror first (fast, and immune to upstream going away), ann-benchmarks.com as the
fallback. `wget -c` resumes a partial file, so a dropped connection costs nothing.
"""),

code(r"""
DATASETS = {
    "sift":  dict(url="https://ann-benchmarks.com/sift-128-euclidean.hdf5",
                  file="sift-128-euclidean.hdf5", metric="euclidean", normalise=False),
    "glove": dict(url="https://ann-benchmarks.com/glove-100-angular.hdf5",
                  file="glove-100-angular.hdf5", metric="angular", normalise=True),
}

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  cached: {dest} ({C.human(os.path.getsize(dest))})")
        return dest
    print(f"  downloading {url}")
    r = subprocess.run(["wget", "-c", "-q", "--show-progress", "--progress=dot:giga",
                        "-O", dest, url], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError(f"download failed: {r.stderr[-800:]}")
    print(f"  got {C.human(os.path.getsize(dest))}")
    return dest

for name, spec in DATASETS.items():
    print(f"[{name}]")
    spec["path"] = fetch(spec["url"], f"{C.TMP}/{spec['file']}")

for name, spec in DATASETS.items():
    with h5py.File(spec["path"], "r") as f:
        print(f"\n[{name}] keys={list(f.keys())}")
        for k in f.keys():
            print(f"   {k:12s} {str(f[k].shape):22s} {f[k].dtype}")
        spec["n"], spec["d"] = f["train"].shape
"""),

md("""
## 5 · To `.npy`, with GloVe normalised

GloVe is distributed for angular distance. Unit-normalising every vector makes squared-L2
rank-equivalent to cosine, so the whole project can use one metric and one exact baseline —
which is what the assignment asks for. The file's own `distances` are in cosine units and we
deliberately never use their scale; only the neighbour **ids** are used, as an independent
check on our own ground truth.
"""),

code(r"""
def load_split(spec):
    with h5py.File(spec["path"], "r") as f:
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        query = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        stored_nb = np.ascontiguousarray(f["neighbors"][:], dtype=np.int32)
    if spec["normalise"]:
        nz = (np.linalg.norm(base, axis=1) == 0).sum()
        base, query = C.normalise(base), C.normalise(query)
        print(f"  normalised; {nz} zero-norm base vectors left as zeros")
        print(f"  base norms  min {np.linalg.norm(base, axis=1).min():.6f} "
              f"max {np.linalg.norm(base, axis=1).max():.6f}")
    return base, query, stored_nb

PREP = {}
for name, spec in DATASETS.items():
    print(f"[{name}]")
    base, query, stored_nb = load_split(spec)
    print(f"  base {base.shape} {base.dtype} · queries {query.shape} · stored GT {stored_nb.shape}")
    np.save(f"{C.TMP}/data/{name}_base.npy", base)
    np.save(f"{C.TMP}/data/{name}_query.npy", query)
    np.save(f"{C.TMP}/data/{name}_stored_neighbors.npy", stored_nb)
    PREP[name] = dict(n=base.shape[0], d=base.shape[1])
    del base, query, stored_nb
    gc.collect()

print("\nQuery split: tune = queries[0:5000], eval = queries[5000:10000].")
print("Every number in the report comes from the eval half only.")
"""),

md("""
## 6 · Exact ground truth, and the cross-check that matters

The GPU selects the top 100 by the `|q|²+|b|²−2q·b` expansion in float32 (fast), then the
returned distances are **recomputed by direct subtraction in float64** (exact). Selection
error from the expansion can only mis-order points that are equal to within a few ULPs —
i.e. ties, which the recall metric forgives by construction — while the *threshold* recall@10
depends on ends up exact.

Because the refine step re-sorts the top 100 exactly, the final **top-10 is exact** unless the
float32 pass dropped a true top-10 point out of its top *100*, which would take an error of
order (d100 − d10) — thousands of times larger than the ~1 ULP we actually see.

The check below asks the independent question: does the ground truth we computed agree with
the neighbours shipped in the HDF5 file? Scored with our tie-aware metric, this should be
≈ 1.000. Anything materially below that means our pipeline disagrees with the dataset authors
and nothing downstream is trustworthy.

The block size adapts to free VRAM: a (10000 × chunk) float32 distance block plus `topk`'s
own workspace is what sets the limit, not the dataset size.
""" ),

code(r"""
GT_K = 100
gt_report = {}

for name in DATASETS:
    print(f"[{name}]")
    base = np.load(f"{C.TMP}/data/{name}_base.npy")
    query = np.load(f"{C.TMP}/data/{name}_query.npy")

    t0 = time.perf_counter()
    gt_ids, gt_dists = C.exact_gt_gpu(base, query, k=GT_K)
    secs = time.perf_counter() - t0

    np.save(f"{C.TMP}/gt/{name}_top100_ids.npy", gt_ids)
    np.save(f"{C.TMP}/gt/{name}_top100_dists.npy", gt_dists)

    stored_nb = np.load(f"{C.TMP}/data/{name}_stored_neighbors.npy")
    agree_exact = float((stored_nb[:, :10] == gt_ids[:, :10]).all(axis=1).mean())
    rec_stored = C.recall_from_ids(stored_nb[:, :10], base, query, gt_dists)
    rec_self = C.recall_from_ids(gt_ids[:, :10], base, query, gt_dists)

    print(f"  computed in {secs:.1f}s")
    print(f"  d1  mean {np.sqrt(gt_dists[:, 0]).mean():.4f} · "
          f"d10 mean {np.sqrt(gt_dists[:, 9]).mean():.4f}  (true distances)")
    print(f"  queries whose stored top-10 is id-identical to ours : {agree_exact:.4f}")
    print(f"  recall@10 of the FILE's neighbours vs our GT        : {rec_stored.mean():.6f}")
    print(f"  recall@10 of our own GT vs itself (must be 1.0)     : {rec_self.mean():.6f}")
    assert rec_self.mean() == 1.0, "self-recall must be exactly 1.0"
    assert rec_stored.mean() > 0.999, "our ground truth disagrees with the dataset file"

    gt_report[name] = dict(seconds=round(secs, 1), id_identical=agree_exact,
                           recall_file_vs_ours=round(float(rec_stored.mean()), 6),
                           d1_mean=float(np.sqrt(gt_dists[:, 0]).mean()),
                           d10_mean=float(np.sqrt(gt_dists[:, 9]).mean()))
    del base, query, gt_ids, gt_dists, stored_nb
    gc.collect()

print("\nGround truth verified against both dataset files.")
"""),

md("""
### Relative contrast of each dataset

How much nearer is the 10th neighbour than a random point? This single number explains most
of the recall differences we expect between SIFT and GloVe, so it is worth measuring now
rather than asserting later. It is also the quantity the Part 4 experiment varies on purpose.
"""),

code(r"""
contrast = {}
for name in DATASETS:
    base = np.load(f"{C.TMP}/data/{name}_base.npy")
    query = np.load(f"{C.TMP}/data/{name}_query.npy")
    gt_dists = np.load(f"{C.TMP}/gt/{name}_top100_dists.npy")
    msq = C.mean_sq_dist_to_random(base, query, n_sample=100_000, seed=0)
    rc1 = C.relative_contrast(gt_dists, msq, k=1)
    rc10 = C.relative_contrast(gt_dists, msq, k=10)
    contrast[name] = dict(rc_at_1=float(rc1.mean()), rc_at_10=float(rc10.mean()))
    print(f"[{name}] relative contrast: RC@1 {rc1.mean():.3f} · RC@10 {rc10.mean():.3f}  "
          f"(higher = easier)")
    del base, query, gt_dists
    gc.collect()
"""),

md("""
## 7 · Nested subsets for the Part 3 scaling experiment

Part 3 re-runs the SIFT benchmark at 100K, 300K and 1M. The subsets are drawn with a fixed
seed and **nested** (100K ⊂ 300K ⊂ 1M), so a difference between sizes is attributable to size
alone and not to having sampled a different region of the space. We store only the chosen
**ids** plus their ground truth — the vectors themselves are recoverable from the 1M base by
fancy-indexing, which keeps ~200 MB off the upload.

Ground-truth ids for a subset are **local** (0..n−1), matching what an index built on that
subset will return.
"""),

code(r"""
SUBSET_SEED = 20261023
SUBSET_SIZES = [100_000, 300_000]

base = np.load(f"{C.TMP}/data/sift_base.npy")
query = np.load(f"{C.TMP}/data/sift_query.npy")
rng = np.random.default_rng(SUBSET_SEED)
perm = rng.permutation(len(base))

subset_report = {}
for n in SUBSET_SIZES:
    tag = f"sift{n // 1000}k"
    ids = np.sort(perm[:n]).astype(np.int32)      # nested: prefixes of one permutation
    np.save(f"{C.TMP}/data/{tag}_ids.npy", ids)
    sub = np.ascontiguousarray(base[ids])
    gt_ids, gt_dists = C.exact_gt_gpu(sub, query, k=GT_K)
    np.save(f"{C.TMP}/gt/{tag}_top100_ids.npy", gt_ids)
    np.save(f"{C.TMP}/gt/{tag}_top100_dists.npy", gt_dists)
    assert gt_ids.max() < n and gt_ids.min() >= 0, "subset GT ids must be local"
    assert C.recall_from_ids(gt_ids[:, :10], sub, query, gt_dists).mean() == 1.0
    full_d10 = np.load(f"{C.TMP}/gt/sift_top100_dists.npy")[:, 9]
    print(f"[{tag}] n={n:,}  d10 mean {np.sqrt(gt_dists[:, 9]).mean():.2f} "
          f"(1M: {np.sqrt(full_d10).mean():.2f})")
    subset_report[tag] = dict(n=n, d10_mean=float(np.sqrt(gt_dists[:, 9]).mean()))
    del sub, gt_ids, gt_dists
    gc.collect()

prev = None
for n in SUBSET_SIZES:
    cur = set(np.load(f"{C.TMP}/data/sift{n // 1000}k_ids.npy").tolist())
    if prev is not None:
        assert prev <= cur, "subsets are not nested"
    prev = cur
print("\nSubsets are nested and their ground truth checks out.")
del base, query
gc.collect()
"""),

md("""
## 8 · Mirror to Hugging Face

Bases and queries go up as `.npy` so later notebooks start in seconds instead of re-pulling
from ann-benchmarks.com. The raw HDF5 files are *not* mirrored — everything in them that we
use (vectors, queries, stored neighbour ids) is already captured above, so mirroring them
would double the upload for nothing.
"""),

code(r"""
UPLOAD_BASES = True       # ~1 GB; set False to upload only the small artifacts

manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": ENV,
            "gt_check": gt_report, "relative_contrast": contrast,
            "subsets": subset_report, "subset_seed": SUBSET_SEED,
            "gt_k": GT_K, "query_split": {"tune": [0, 5000], "eval": [5000, 10000]},
            "files": {}}

uploads = []
for name in DATASETS:
    if UPLOAD_BASES:
        uploads.append((f"{C.TMP}/data/{name}_base.npy", f"data/{name}_base.npy"))
    uploads += [(f"{C.TMP}/data/{name}_query.npy", f"data/{name}_query.npy"),
                (f"{C.TMP}/data/{name}_stored_neighbors.npy",
                 f"data/{name}_stored_neighbors.npy"),
                (f"{C.TMP}/gt/{name}_top100_ids.npy", f"gt/{name}_top100_ids.npy"),
                (f"{C.TMP}/gt/{name}_top100_dists.npy", f"gt/{name}_top100_dists.npy")]
for n in SUBSET_SIZES:
    tag = f"sift{n // 1000}k"
    uploads += [(f"{C.TMP}/data/{tag}_ids.npy", f"data/{tag}_ids.npy"),
                (f"{C.TMP}/gt/{tag}_top100_ids.npy", f"gt/{tag}_top100_ids.npy"),
                (f"{C.TMP}/gt/{tag}_top100_dists.npy", f"gt/{tag}_top100_dists.npy")]

for local, remote in uploads:
    a = np.load(local, mmap_mode="r")
    manifest["files"][remote] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                     bytes=os.path.getsize(local))
    del a

t0 = time.perf_counter()
for i, (local, remote) in enumerate(uploads, 1):
    print(f"[{i}/{len(uploads)}]", end=" ")
    C.hf_upload(local, remote, token=TOKEN, msg="nb00: prepared data")
print(f"\nuploaded {len(uploads)} files in {time.perf_counter() - t0:.0f}s")

with open(f"{C.OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
C.hf_upload(f"{C.OUT}/manifest.json", "manifest.json", token=TOKEN, msg="nb00: manifest")
C.hf_upload("annbench_core.py", "code/annbench_core.py", token=TOKEN, msg="nb00: core")
C.hf_upload("test_recall.py", "code/test_recall.py", token=TOKEN, msg="nb00: tests")
C.hf_upload(f"{C.OUT}/env/{ENV['machine_id']}.json",
            f"logs/env/{ENV['machine_id']}.json", token=TOKEN, msg="nb00: env")
print(f"\nhttps://huggingface.co/datasets/{C.HF_REPO}/tree/main")
"""),

md("## 9 · Summary — paste this back"),

code(r"""
print("=" * 72)
print("nb00 SUMMARY")
print("=" * 72)
print(f"machine_id   : {ENV['machine_id']}")
print(f"cpu          : {ENV['cpu_model']} ({ENV['cpu_logical']} logical), {ENV['mem_gb']} GB RAM")
print(f"gpu          : {ENV['gpu']}")
print(f"versions     : faiss {ENV['faiss']} · torch {ENV['torch']} · numpy {ENV['numpy']} "
      f"· h5py {ENV['h5py']} · python {ENV['python']}")
print()
for name in DATASETS:
    p, g, c = PREP[name], gt_report[name], contrast[name]
    print(f"{name:6s} n={p['n']:>9,}  d={p['d']:>4}  "
          f"GT {g['seconds']:>5.1f}s  file-vs-ours recall {g['recall_file_vs_ours']:.6f}  "
          f"id-identical {g['id_identical']:.4f}")
    print(f"{'':6s} d1 {g['d1_mean']:.4f} · d10 {g['d10_mean']:.4f} · "
          f"RC@1 {c['rc_at_1']:.3f} · RC@10 {c['rc_at_10']:.3f}")
for tag, s in subset_report.items():
    print(f"{tag:6s} n={s['n']:>9,}  d10 mean {s['d10_mean']:.2f}")
print()
print(f"free disk /kaggle/temp : {C.human(__import__('shutil').disk_usage(C.TMP).free)}")
print(f"HF repo                : https://huggingface.co/datasets/{C.HF_REPO}")
print("=" * 72)
"""),

md("""
### What happens next

`nb01` (Part 1) builds the exact FAISS baseline on top of these artifacts: it runs the recall
unit tests properly, confirms exact search reproduces the stored neighbours, and times exact
search one-query-at-a-time against all-queries-in-one-batch on both datasets.

Paste the summary above back before moving on — the measured GT timings, relative contrasts
and `machine_id` determine the sweep grids and time budget for nb02/nb03.
""" ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size / 1024:.0f} KB)")
