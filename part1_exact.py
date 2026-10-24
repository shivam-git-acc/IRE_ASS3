#!/usr/bin/env python3
"""Part 1 — exact FAISS baseline, the recall@10 metric, and one-at-a-time vs batch timing.

Three timing modes per dataset, all on the same IndexFlatL2:
  one_at_a_time  1 thread  — the protocol every ANN number in this project uses
  batch          1 thread  — isolates batching with the thread count held fixed
  batch_Nthreads N threads — context only; not comparable with the ANN measurements

Keeping the single-thread batch in the middle is the point: comparing one-at-a-time against a
multi-threaded batch would confound batching with parallelism.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annbench_core as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import faiss
    C.ensure_dirs()
    env = C.env_fingerprint()
    man = json.load(open(f"{C.OUT}/manifest.json"))
    C.set_split(man["n_queries"])
    datasets = man["datasets"]
    n_single = 200 if a.quick else 1000
    repeats = 2 if a.quick else 3
    log = C.ResultsLog(env, token=C.get_hf_token() if C.hf_enabled() else None)

    summary = {}
    for name in datasets:
        d = C.load_local(name, with_base=True, with_stored=True)
        base, query, gt_dists, stored = d["base"], d["query"], d["gt_dists"], d["stored"]
        n, dim = base.shape
        qe = np.ascontiguousarray(query[C.EVAL])
        gte = gt_dists[C.EVAL]

        C.set_single_thread()
        ix = faiss.IndexFlatL2(dim)
        t0 = time.perf_counter(); ix.add(base); add_s = time.perf_counter() - t0
        size = C.index_size_bytes(ix)

        # Correctness pass uses every core: recall does not depend on thread count.
        faiss.omp_set_num_threads(env["cpu_logical"])
        _, I = ix.search(qe, 10)
        faiss.omp_set_num_threads(1)
        rec_exact = C.recall_from_ids(I, base, qe, gte)
        assert rec_exact.mean() == 1.0, "exact search disagrees with our ground truth"

        # Tie structure of the ground truth itself -- always well defined.
        boundary = np.isclose(gt_dists[:, 9], gt_dists[:, 10], rtol=1e-9, atol=0).mean()
        anytie = np.isclose(gt_dists[:, 1:11], gt_dists[:, :10], rtol=1e-9, atol=0).any(1).mean()

        print(f"\n[{name}] n={n:,} d={dim} · IndexFlatL2 {C.human(size)} "
              f"({size / n:.0f} B/vector, add {add_s:.1f}s)")
        print(f"  exact vs our ground truth, tie-aware : {rec_exact.mean():.6f}")
        # The file's neighbour ids index the full base; with a truncated base they point
        # past the end of the array, so the comparison is skipped rather than faked.
        if man["n_cap"] is None:
            naive_file = np.array([len(set(x.tolist()) & set(y.tolist())) / 10
                                   for x, y in zip(stored[:, :10], d["gt_ids"][:, :10])])
            tie_file = C.recall_from_ids(stored[:, :10], base, query, gt_dists)
            nf, tf = float(naive_file.mean()), float(tie_file.mean())
            print(f"  file's neighbours vs ours, tie-aware : {tf:.6f}")
            print(f"  file's neighbours vs ours, naive     : {nf:.6f}")
            print(f"  -> the naive metric discards {(tf - nf) * 100:.2f} recall points "
                  f"for answers that are not wrong")
        else:
            nf = tf = None
            print(f"  file's neighbours vs ours            : not checked (base truncated "
                  f"to {man['n_cap']:,} in quick mode)")
        print(f"  queries with a tie at d10==d11       : {boundary * 100:.2f}%")
        print(f"  queries with any tie in the top 11   : {anytie * 100:.2f}%")

        timings = {}
        qs = qe[:n_single]
        rec_s = C.recall_from_ids(ix.search(qs, 10)[1], base, qs, gte[:n_single])
        t = C.bench_single_query(lambda x: ix.search(x, 10), qs, repeats=repeats, warmup=20)
        timings["one_at_a_time"] = t
        log.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
                index_kind="exact_flat", build_params={}, search_param_name="query_mode",
                search_param_value="one_at_a_time", recall=rec_s, timing=t,
                build_time_s=add_s, add_time_s=add_s, index_size_bytes=size, threads=1,
                notes=f"IndexFlatL2; first {n_single} eval queries")
        print(f"  one_at_a_time (1 thread, {n_single} q): {t['qps_median']:8.2f} qps · "
              f"p50 {t['lat_p50_ms']:6.2f} ms · p95 {t['lat_p95_ms']:6.2f} ms")

        tb = C.bench_batch(lambda x: ix.search(x, 10), qe, repeats=repeats, warmup=1)
        timings["batch"] = tb
        log.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
                index_kind="exact_flat", build_params={}, search_param_name="query_mode",
                search_param_value="batch", recall=rec_exact, timing=tb, build_time_s=add_s,
                add_time_s=add_s, index_size_bytes=size, threads=1,
                notes="IndexFlatL2; all eval queries in one call")
        print(f"  batch         (1 thread, {len(qe)} q): {tb['qps_median']:8.2f} qps "
              f"({tb['qps_median'] / t['qps_median']:.1f}x)")

        nt = env["cpu_logical"]
        faiss.omp_set_num_threads(nt)
        tb4 = C.bench_batch(lambda x: ix.search(x, 10), qe, repeats=repeats, warmup=1)
        faiss.omp_set_num_threads(1)
        timings["batch_threads"] = tb4
        log.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
                index_kind="exact_flat", build_params={}, search_param_name="query_mode",
                search_param_value=f"batch_{nt}threads", recall=rec_exact, timing=tb4,
                build_time_s=add_s, add_time_s=add_s, index_size_bytes=size, threads=nt,
                notes="context only; not comparable with the single-thread ANN numbers")
        print(f"  batch         ({nt} threads, {len(qe)} q): {tb4['qps_median']:8.2f} qps")

        summary[name] = dict(
            n=n, dim=dim, bytes_per_vector=size / n, add_s=add_s,
            exact_tie_aware=float(rec_exact.mean()), file_tie_aware=tf,
            file_naive=nf, boundary_tie=float(boundary),
            any_tie=float(anytie),
            qps_1q=t["qps_median"], qps_batch=tb["qps_median"], qps_batch_nt=tb4["qps_median"],
            batch_speedup=tb["qps_median"] / t["qps_median"])
        del ix, base, query, gt_dists, stored

    log.flush()
    with open(f"{C.OUT}/results/part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {C.OUT}/results/part1_summary.json")


if __name__ == "__main__":
    main()
