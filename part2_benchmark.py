#!/usr/bin/env python3
"""Part 2 — benchmark HNSW, IVF-Flat and IVF-PQ on one dataset.

    python part2_benchmark.py --dataset sift
    python part2_benchmark.py --dataset glove

A few build settings per index, and for each a sweep of the search-time parameter so recall
climbs from low up past 0.95. Query timing is single-thread, one query at a time, median of
three repeats; recall is measured in a separate multi-threaded batched pass because it does
not depend on thread count.

The coarse quantizer is trained once per `nlist` and shared by every index that uses it, so
the IVF-PQ configurations differ only in `m` — one variable at a time. The code asserts the
centroids really were reused rather than trusting FAISS to have skipped retraining.

Resumable: rows already present in results.csv are skipped.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annbench_core as C

K = 10
RECALL_STOP = 0.995


def divisors_for_pq(dim, wanted):
    """FAISS requires dim % m == 0. SIFT's 128 allows 8/16/32/64; GloVe's 100 does not, so
    each dataset uses divisors of its own dimension spanning a comparable byte range."""
    return [m for m in wanted if dim % m == 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["sift", "glove"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import faiss
    C.ensure_dirs()
    env = C.env_fingerprint()
    man = json.load(open(f"{C.OUT}/manifest.json"))
    C.set_split(man["n_queries"])
    ds = a.dataset
    bt = env["cpu_logical"]
    repeats = 1 if a.quick else 3
    budget = 180 if a.quick else (1800 if ds == "glove" else 900)

    d = C.load_local(ds, with_base=True)
    base, query, gt = d["base"], d["query"], d["gt_dists"]
    N, DIM = base.shape
    q_eval = np.ascontiguousarray(query[C.EVAL])
    gt_eval = gt[C.EVAL]
    log = C.ResultsLog(env, token=C.get_hf_token() if C.hf_enabled() else None)
    print(f"\n[{ds}] base {base.shape} · eval queries {q_eval.shape} · "
          f"{'QUICK' if a.quick else 'FULL'}")

    if a.quick:
        hnsw_m, ivf_nlist, pq_m = [16], [256, 1024], divisors_for_pq(DIM, [8, 16, 10, 20])[:2]
        ef_grid = [10, 32, 64, 128, 256]
        nprobe_grid = [1, 4, 16, 64, 256]
    else:
        hnsw_m = [8, 16, 32]
        ivf_nlist = [1024, 4096, 16384]
        pq_m = divisors_for_pq(DIM, [8, 16, 32, 64] if DIM % 8 == 0 else [10, 20, 25, 50])
        ef_grid = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768] + \
                  ([1024] if ds == "glove" else [])
        nprobe_grid = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                       768, 1024] + ([1536, 2048] if ds == "glove" else [])
    pq_nlist = ivf_nlist[-2] if len(ivf_nlist) > 1 else ivf_nlist[0]
    print(f"grid: HNSW M={hnsw_m} · IVF nlist={ivf_nlist} · "
          f"IVF-PQ m={pq_m} at nlist={pq_nlist}")

    def recall_of(ix):
        faiss.omp_set_num_threads(bt)
        _, I = ix.search(q_eval, K)
        faiss.omp_set_num_threads(1)
        return C.recall_from_ids(I, base, q_eval, gt_eval)

    def sweep(ix, kind, bp, pname, grid, setter, build_s, train_s, add_s, size):
        t_start = time.perf_counter()
        for v in grid:
            if log.done("2", ds, N, kind, bp, pname, v):
                continue
            setter(ix, v)
            rec = recall_of(ix)
            C.set_single_thread()
            t = C.bench_single_query(lambda x: ix.search(x, K), q_eval,
                                     repeats=repeats, warmup=30)
            log.add(part="2", dataset=ds, n_vectors=N, dim=DIM, metric="l2",
                    index_kind=kind, build_params=bp, search_param_name=pname,
                    search_param_value=v, recall=rec, timing=t, build_time_s=build_s,
                    train_time_s=train_s, add_time_s=add_s, index_size_bytes=size,
                    threads=1, notes=f"build_threads={bt}")
            r = float(rec.mean())
            print(f"    {pname}={v:>5}  recall {r:.4f}  {t['qps_median']:9.1f} qps  "
                  f"p95 {t['lat_p95_ms']:6.2f} ms")
            if r >= RECALL_STOP:
                print(f"    -> recall {r:.4f} >= {RECALL_STOP}, stopping sweep")
                break
            if time.perf_counter() - t_start > budget:
                print("    -> wall-clock budget exceeded, stopping sweep")
                log.add(part="2", dataset=ds, n_vectors=N, dim=DIM, metric="l2",
                        index_kind=kind, build_params=bp, search_param_name=pname,
                        search_param_value="BUDGET_STOP", recall=np.array([r]), timing=t,
                        notes="sweep truncated by time budget")
                break
        log.flush()

    # One coarse quantizer per nlist, shared by every index at that nlist.
    quant, qtrain = {}, {}
    for nlist in ivf_nlist:
        faiss.omp_set_num_threads(bt)
        t0 = time.perf_counter()
        probe = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, nlist, faiss.METRIC_L2)
        probe.train(base)
        qtrain[nlist] = time.perf_counter() - t0
        cent = probe.quantizer.reconstruct_n(0, nlist)
        qz = faiss.IndexFlatL2(DIM); qz.add(cent)
        quant[nlist] = (qz, cent)
        del probe
        print(f"coarse k-means nlist={nlist:>6,}: {qtrain[nlist]:7.1f}s "
              f"({N // nlist} vectors/list)")
    faiss.omp_set_num_threads(1)

    def ivf_flat(nlist):
        qz, cent = quant[nlist]
        faiss.omp_set_num_threads(bt)
        ix = faiss.IndexIVFFlat(qz, DIM, nlist, faiss.METRIC_L2); ix.own_fields = False
        t0 = time.perf_counter(); ix.train(base); tr = time.perf_counter() - t0
        assert np.allclose(ix.quantizer.reconstruct_n(0, nlist), cent), \
            "the coarse quantizer was retrained"
        t0 = time.perf_counter(); ix.add(base); ad = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        return ix, tr, ad

    for nlist in [n for n in ivf_nlist if n != ivf_nlist[-1]]:
        ix, tr, ad = ivf_flat(nlist)
        size = C.index_size_bytes(ix)
        print(f"\nIVF-Flat nlist={nlist}: add {ad:.1f}s · {C.human(size)} · {size / N:.0f} B/vec")
        sweep(ix, "ivf_flat", {"nlist": nlist}, "nprobe",
              [p for p in nprobe_grid if p <= nlist], lambda i, v: setattr(i, "nprobe", int(v)),
              qtrain[nlist] + tr + ad, qtrain[nlist] + tr, ad, size)
        del ix

    for m in pq_m:
        qz, cent = quant[pq_nlist]
        faiss.omp_set_num_threads(bt)
        ix = faiss.IndexIVFPQ(qz, DIM, pq_nlist, m, 8); ix.own_fields = False
        t0 = time.perf_counter(); ix.train(base); tr = time.perf_counter() - t0
        assert np.allclose(ix.quantizer.reconstruct_n(0, pq_nlist), cent), \
            "the coarse quantizer was retrained -- PQ configs would not be comparable"
        t0 = time.perf_counter(); ix.add(base); ad = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        size = C.index_size_bytes(ix)
        print(f"\nIVF-PQ m={m} ({m} B/code): PQ train {tr:.1f}s · {C.human(size)} · "
              f"{size / N:.1f} B/vec · {(DIM * 4) / (size / N):.1f}x smaller than flat")
        sweep(ix, "ivf_pq", {"nlist": pq_nlist, "m": m, "nbits": 8}, "nprobe",
              [p for p in nprobe_grid if p <= pq_nlist],
              lambda i, v: setattr(i, "nprobe", int(v)),
              qtrain[pq_nlist] + tr + ad, tr, ad, size)
        del ix

    for M in hnsw_m:
        faiss.omp_set_num_threads(bt)
        ix = faiss.IndexHNSWFlat(DIM, M); ix.hnsw.efConstruction = 200
        t0 = time.perf_counter(); ix.add(base); tb_ = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        size = C.index_size_bytes(ix)
        print(f"\nHNSW M={M} efC=200: build {tb_:.1f}s ({bt} threads) · {C.human(size)} · "
              f"{size / N:.0f} B/vec (graph {size / N - DIM * 4:.0f} B/vec)")
        sweep(ix, "hnsw", {"M": M, "efConstruction": 200}, "efSearch", ef_grid,
              lambda i, v: setattr(i.hnsw, "efSearch", int(v)), tb_, None, tb_, size)
        del ix

    nlist = ivf_nlist[-1]          # most expensive k-means, left until last
    ix, tr, ad = ivf_flat(nlist)
    size = C.index_size_bytes(ix)
    print(f"\nIVF-Flat nlist={nlist}: add {ad:.1f}s · {C.human(size)} · {size / N:.0f} B/vec")
    sweep(ix, "ivf_flat", {"nlist": nlist}, "nprobe",
          [p for p in nprobe_grid if p <= nlist], lambda i, v: setattr(i, "nprobe", int(v)),
          qtrain[nlist] + tr + ad, qtrain[nlist] + tr, ad, size)
    del ix
    log.flush()
    print(f"\nPart 2 ({ds}) complete · results.csv holds {len(log.df)} rows")


if __name__ == "__main__":
    main()
