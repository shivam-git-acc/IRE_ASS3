#!/usr/bin/env python3
"""Part 3 — how the cost of reaching recall@10 = 0.95 grows with n, on SIFT.

Nested, seeded subsets (100K subset of 300K subset of 1M) so a difference between sizes is
attributable to size rather than to having sampled a different region of the space.

The search parameter is chosen on the TUNE query half — the grid is walked measuring recall
only, stopping at the first value to reach 0.95 — and that value is then timed on the EVAL
half. The reported number therefore never saw the eval queries during selection.

Hypotheses H3.1–H3.6 are scored automatically against the conditions committed in
hypothesis.md before this was first run.
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
TARGET = 0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import faiss
    C.ensure_dirs()
    env = C.env_fingerprint()
    man = json.load(open(f"{C.OUT}/manifest.json"))
    C.set_split(man["n_queries"])
    bt = env["cpu_logical"]
    repeats = 1 if a.quick else 3
    n_single = 200 if a.quick else 1000

    d = C.load_local("sift", with_base=True)
    base_full, query = d["base"], d["query"]
    q_tune = np.ascontiguousarray(query[C.TUNE])
    q_eval = np.ascontiguousarray(query[C.EVAL])
    sizes = man["subsets"] + [len(base_full)]
    log = C.ResultsLog(env, token=C.get_hf_token() if C.hf_enabled() else None)

    SUB = {}
    for n in sizes:
        if n == len(base_full):
            sub, gt = base_full, d["gt_dists"]
        else:
            tag = f"sift{n // 1000}k"
            ids = np.load(f"{C.TMP}/data/{tag}_ids.npy")
            sub = np.ascontiguousarray(base_full[ids])
            gt = np.load(f"{C.TMP}/gt/{tag}_top100_dists.npy")
        SUB[n] = dict(base=sub, gt_tune=gt[C.TUNE], gt_eval=gt[C.EVAL])
        print(f"n={n:>9,}  {C.human(sub.nbytes)}")

    if a.quick:
        hnsw_m, ivf_nlist = [16], [256, 1024]
        ef_grid = [10, 32, 64, 128, 256]
        nprobe_grid = [1, 4, 16, 64, 256]
    else:
        hnsw_m, ivf_nlist = [8, 16, 32], [1024, 2048, 4096, 8192]
        ef_grid = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
        nprobe_grid = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                       768, 1024]

    exact_qps = {}
    for n in sizes:
        s = SUB[n]
        C.set_single_thread()
        ix = faiss.IndexFlatL2(s["base"].shape[1])
        t0 = time.perf_counter(); ix.add(s["base"]); add_s = time.perf_counter() - t0
        size = C.index_size_bytes(ix)
        qs = q_eval[:n_single]
        rec = C.recall_from_ids(ix.search(qs, K)[1], s["base"], qs, s["gt_eval"][:n_single])
        assert rec.mean() == 1.0
        t = C.bench_single_query(lambda x: ix.search(x, K), qs, repeats=repeats, warmup=20)
        log.add(part="3", dataset="sift", n_vectors=n, dim=s["base"].shape[1], metric="l2",
                index_kind="exact_flat", build_params={}, search_param_name="query_mode",
                search_param_value="one_at_a_time", recall=rec, timing=t, build_time_s=add_s,
                add_time_s=add_s, index_size_bytes=size, threads=1,
                notes=f"scaling reference; first {n_single} eval queries")
        exact_qps[n] = t["qps_median"]
        print(f"n={n:>9,}  exact {t['qps_median']:8.2f} qps · p50 {t['lat_p50_ms']:7.2f} ms")
        del ix
    log.flush()

    def recall_on(ix, qs, gtd, arr):
        faiss.omp_set_num_threads(bt)
        _, I = ix.search(qs, K)
        faiss.omp_set_num_threads(1)
        return C.recall_from_ids(I, arr, qs, gtd)

    SCAL = []

    def run(n, kind, bp, pname, grid, setter, build_s, size):
        s = SUB[n]
        chosen = None
        best_tune = 0.0
        for v in grid:
            setter(ix, v)
            r = float(recall_on(ix, q_tune, s["gt_tune"], s["base"]).mean())
            best_tune = max(best_tune, r)
            if r >= TARGET:
                chosen = v
                print(f"    tune screen: picked {pname}={v} (tune recall {r:.4f})")
                break
        if chosen is None:
            print(f"    tune screen: never reached {TARGET} (best {best_tune:.4f})")
            SCAL.append(dict(n=n, kind=kind, build_params=bp, param=None,
                             recall=best_tune, qps=None, build_s=build_s, size=size))
            return
        setter(ix, chosen)
        rec = recall_on(ix, q_eval, s["gt_eval"], s["base"])
        C.set_single_thread()
        t = C.bench_single_query(lambda x: ix.search(x, K), q_eval, repeats=repeats, warmup=30)
        frac = chosen / bp["nlist"] if kind == "ivf_flat" else None
        note = f"param chosen on tune split; build_threads={bt}"
        if frac and frac > 0.5:
            print(f"    ! nprobe={chosen} is {frac:.0%} of nlist -- effectively a full scan")
            note += " DEGENERATE(>50% of lists probed)"
        log.add(part="3", dataset="sift", n_vectors=n, dim=s["base"].shape[1], metric="l2",
                index_kind=kind, build_params=bp, search_param_name=pname,
                search_param_value=chosen, recall=rec, timing=t, build_time_s=build_s,
                add_time_s=build_s, index_size_bytes=size, threads=1, notes=note)
        print(f"    eval: recall {rec.mean():.4f} · {t['qps_median']:9.1f} qps")
        SCAL.append(dict(n=n, kind=kind, build_params=bp, param=chosen,
                         recall=float(rec.mean()), qps=t["qps_median"],
                         build_s=build_s, size=size, frac_lists=frac))

    for n in sizes:
        s = SUB[n]
        dim = s["base"].shape[1]
        print(f"\n--- HNSW · n={n:,} ---")
        for M in hnsw_m:
            faiss.omp_set_num_threads(bt)
            ix = faiss.IndexHNSWFlat(dim, M); ix.hnsw.efConstruction = 200
            t0 = time.perf_counter(); ix.add(s["base"]); tb_ = time.perf_counter() - t0
            faiss.omp_set_num_threads(1)
            size = C.index_size_bytes(ix)
            print(f"  M={M:<3d} build {tb_:7.1f}s ({tb_ / n * 1e6:5.1f} us/vector) · "
                  f"{size / n:.0f} B/vec")
            run(n, "hnsw", {"M": M, "efConstruction": 200}, "efSearch", ef_grid,
                lambda i, v: setattr(i.hnsw, "efSearch", int(v)), tb_, size)
            del ix
        print(f"\n--- IVF-Flat · n={n:,} ---")
        for nlist in ivf_nlist:
            faiss.omp_set_num_threads(bt)
            ix = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
            t0 = time.perf_counter(); ix.train(s["base"]); ix.add(s["base"])
            tb_ = time.perf_counter() - t0
            faiss.omp_set_num_threads(1)
            size = C.index_size_bytes(ix)
            print(f"  nlist={nlist:<5d} build {tb_:7.1f}s · {n // nlist} vectors/list")
            run(n, "ivf_flat", {"nlist": nlist}, "nprobe",
                [p for p in nprobe_grid if p <= nlist],
                lambda i, v: setattr(i, "nprobe", int(v)), tb_, size)
            del ix
        log.flush()

    # ---- scoring the pre-registered hypotheses ----
    import pandas as pd
    sc = pd.DataFrame([r for r in SCAL if r["qps"] is not None])
    best = sc.sort_values("qps", ascending=False).groupby(["kind", "n"], as_index=False).first()
    EXP = {}
    for kind in ("ivf_flat", "hnsw"):
        g = best[best.kind == kind].sort_values("n")
        if len(g) >= 2:
            EXP[kind] = dict(exponent=float(np.polyfit(np.log(g.n.astype(float)),
                                                       np.log(g.qps.astype(float)), 1)[0]),
                             ratio=float(g.qps.iloc[0] / g.qps.iloc[-1]))
    xs = np.array(sorted(exact_qps))
    EXP["exact_flat"] = dict(
        exponent=float(np.polyfit(np.log(xs.astype(float)),
                                  np.log([exact_qps[x] for x in xs]), 1)[0]),
        ratio=exact_qps[xs[0]] / exact_qps[xs[-1]])

    v = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    e, hx, ixp = (EXP["exact_flat"]["exponent"], EXP["hnsw"]["exponent"],
                  EXP["ivf_flat"]["exponent"])
    hr = EXP["hnsw"]["ratio"]
    gaps = {n: float(best[(best.kind == "hnsw") & (best.n == n)].qps.iloc[0])
               / float(best[(best.kind == "ivf_flat") & (best.n == n)].qps.iloc[0])
            for n in sizes
            if len(best[(best.kind == "hnsw") & (best.n == n)])
            and len(best[(best.kind == "ivf_flat") & (best.n == n)])}
    bpv = {n: float(best[(best.kind == "hnsw") & (best.n == n)].build_s.iloc[0]) / n * 1e6
           for n in sizes if len(best[(best.kind == "hnsw") & (best.n == n)])}
    adv = {n: float(best[(best.kind == "hnsw") & (best.n == n)].qps.iloc[0]) / exact_qps[n]
           for n in sizes if len(best[(best.kind == "hnsw") & (best.n == n)])}
    lo, hi = min(sizes), max(sizes)

    verdicts = {
        "H3_1": -1.15 <= e <= -0.85,
        "H3_2": hr < 4.0 and hx >= -0.5,
        "H3_3": e < ixp < hx,
        "H3_4": gaps.get(lo, 99) < 2.5 and gaps.get(lo, 0) < gaps.get(hi, 0),
        "H3_5": (bpv.get(hi, 0) / max(bpv.get(lo, 1), 1e-9)) >= 1.2,
        "H3_6": min(adv.values()) > 10,
    }
    print("\n" + "=" * 78 + "\nHYPOTHESIS VERDICTS (Part 3)\n" + "=" * 78)
    print(f"H3.1  exact ~ n^-1.0            exponent {e:+.3f}  -> {v(verdicts['H3_1'])}")
    print(f"H3.2  HNSW -0.2..-0.4, <4x      exponent {hx:+.3f}, ratio {hr:.2f}x  "
          f"-> {v(verdicts['H3_2'])}")
    print(f"H3.3  IVF strictly between      {e:+.3f} < {ixp:+.3f} < {hx:+.3f}  "
          f"-> {v(verdicts['H3_3'])}")
    print(f"H3.4  HNSW/IVF gap < 2.5x at n={lo:,}  "
          + " ".join(f"{n // 1000}K:{g:.2f}x" for n, g in sorted(gaps.items()))
          + f"  -> {v(verdicts['H3_4'])}")
    print(f"H3.5  HNSW build us/vec +>=20%  "
          + " ".join(f"{n // 1000}K:{x:.1f}" for n, x in sorted(bpv.items()))
          + f"  -> {v(verdicts['H3_5'])}")
    print(f"H3.6  ANN >10x exact everywhere "
          + " ".join(f"{n // 1000}K:{x:.0f}x" for n, x in sorted(adv.items()))
          + f"  -> {v(verdicts['H3_6'])}")
    print("=" * 78)

    with open(f"{C.OUT}/results/part3_summary.json", "w") as f:
        json.dump({"exact_qps": {str(k): x for k, x in exact_qps.items()},
                   "exponents": EXP, "gaps": {str(k): x for k, x in gaps.items()},
                   "build_us_per_vector": {str(k): x for k, x in bpv.items()},
                   "ann_advantage": {str(k): x for k, x in adv.items()},
                   "verdicts": verdicts,
                   "rows": [{k: x for k, x in r.items() if k != "size"} for r in SCAL]},
                  f, indent=2)
    print(f"wrote {C.OUT}/results/part3_summary.json")


if __name__ == "__main__":
    main()
