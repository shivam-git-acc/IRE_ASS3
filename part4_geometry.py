#!/usr/bin/env python3
"""Part 4 — distance concentration: making both indexes fail on purpose.

A 2 x 4 design from the committed generator: {uniform, clustered} x d in {16,64,256,1024},
n = 200,000, seed 20261023. The uniform row varies the property under study (contrast, which
collapses as d grows); the clustered row is the control — identical n, d, generator and search
budget, differing only in geometry.

A distance computation costs O(d), so QPS falls with dimension for reasons unrelated to search
difficulty. The primary measurements are therefore dimension-independent:
  * recall@10 at a fixed budget (efSearch=64, nprobe=16), identical everywhere;
  * the fraction of the dataset actually scanned, from FAISS's own distance counters.

Hypotheses H4.2–H4.5 are scored against the conditions committed before this first ran.
H4.1 is a design check that was measured while building the generator, and is not scored.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annbench_core as C
import gen_synthetic as G

K = 10
SCAN_STOP = 0.50
CONTROL_D = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import faiss
    C.ensure_dirs()
    env = C.env_fingerprint()
    bt = env["cpu_logical"]
    repeats = 1 if a.quick else 3
    N = 20_000 if a.quick else G.N_DEFAULT
    NQ = 400 if a.quick else G.NQ_DEFAULT
    DIMS = [16, 64, 256] if a.quick else list(G.DIMS)
    KINDS = list(G.KINDS)
    HNSW_M, EFC, IVF_NLIST = 16, 200, (128 if a.quick else 1024)
    FIXED = {"hnsw": 64, "ivf_flat": 16 if not a.quick else 2}
    sweep_grid = {"hnsw": [64, 128, 256, 512, 1024, 2048],
                  "ivf_flat": ([16, 32, 64, 128, 256, 512] if not a.quick
                               else [2, 8, 32, 128])}
    budget = 240 if a.quick else 900
    control_d = CONTROL_D if CONTROL_D in DIMS else DIMS[-1]

    log = C.ResultsLog(env, token=C.get_hf_token() if C.hf_enabled() else None)
    stats = {"hnsw": lambda: faiss.cvar.hnsw_stats,
             "ivf_flat": lambda: faiss.cvar.indexIVF_stats}
    print(f"{len(KINDS)} x {len(DIMS)} datasets of n={N:,}, {NQ:,} queries, seed {G.SEED}")
    print(f"fixed budget: efSearch={FIXED['hnsw']}, nprobe={FIXED['ivf_flat']} "
          f"(nlist={IVF_NLIST})")

    def measure(ix, kind, arr, qs, gtd, time_it):
        faiss.omp_set_num_threads(1)
        st = stats[kind]()
        st.reset()
        _, I = ix.search(qs, K)
        frac = float(st.ndis) / len(qs) / len(arr)
        rec = C.recall_from_ids(I, arr, qs, gtd)
        t = (C.bench_single_query(lambda x: ix.search(x, K), qs, repeats=repeats, warmup=20)
             if time_it else None)
        return rec, frac, t

    GEO, FIX, SW = {}, {}, {}
    t_start = time.perf_counter()
    for d in DIMS:
        for kind in KINDS:
            name = f"synth_{kind}_d{d}"
            print(f"\n{'=' * 70}\n{name}  (n={N:,}, d={d})   [t+{time.perf_counter() - t_start:.0f}s]")
            xb, xq = G.generate(kind, d, n=N, nq=NQ)
            gi, gd = C.exact_gt(xb, xq, k=max(16, K), verbose=False)
            assert C.recall_from_ids(gi[:, :K], xb, xq, gd).mean() == 1.0
            msq = C.mean_sq_dist_to_random(xb, xq, n_sample=min(50_000, N), seed=0)
            rc10 = float(C.relative_contrast(gd, msq, k=10).mean())
            GEO[name] = dict(kind=kind, d=d, rc_at_10=rc10,
                             rc_at_1=float(C.relative_contrast(gd, msq, k=1).mean()),
                             d10_mean=float(np.sqrt(gd[:, 9]).mean()))
            print(f"  RC@10 {rc10:.3f}")

            for ik in ("hnsw", "ivf_flat"):
                bp = ({"M": HNSW_M, "efConstruction": EFC} if ik == "hnsw"
                      else {"nlist": IVF_NLIST})
                pname = "efSearch" if ik == "hnsw" else "nprobe"
                faiss.omp_set_num_threads(bt)
                t0 = time.perf_counter()
                if ik == "hnsw":
                    ix = faiss.IndexHNSWFlat(d, HNSW_M); ix.hnsw.efConstruction = EFC
                    ix.add(xb)
                else:
                    ix = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, IVF_NLIST,
                                            faiss.METRIC_L2)
                    ix.train(xb); ix.add(xb)
                t_build = time.perf_counter() - t0
                faiss.omp_set_num_threads(1)
                size = C.index_size_bytes(ix)
                print(f"  {ik:9s} build {t_build:6.1f}s · {C.human(size)}")

                grid = sweep_grid[ik] if d == DIMS[-1] else [FIXED[ik]]
                rows, ts = [], time.perf_counter()
                for vpar in grid:
                    if log.done("4", name, N, ik, bp, pname, vpar):
                        continue
                    if ik == "hnsw":
                        ix.hnsw.efSearch = int(vpar)
                    else:
                        ix.nprobe = int(vpar)
                    is_fixed = vpar == grid[0]
                    rec, frac, t = measure(ix, ik, xb, xq, gd, time_it=is_fixed)
                    log.add(part="4", dataset=name, n_vectors=N, dim=d, metric="l2",
                            index_kind=ik, build_params=bp, search_param_name=pname,
                            search_param_value=vpar, recall=rec, timing=t,
                            build_time_s=t_build, add_time_s=t_build,
                            index_size_bytes=size, threads=1,
                            scanned_frac=round(frac, 6), rc_at_10=round(rc10, 4),
                            notes=f"{'fixed budget' if is_fixed else 'H4.4 sweep'}; "
                                  f"scanned_frac={frac:.6f}; rc10={rc10:.3f}")
                    rows.append(dict(param=vpar, recall=float(rec.mean()), frac=frac))
                    if is_fixed:
                        FIX[(name, ik)] = dict(name=name, kind=kind, d=d, index=ik,
                                               recall=float(rec.mean()), frac=frac,
                                               qps=t["qps_median"] if t else None,
                                               build_s=t_build, size=size, rc10=rc10)
                    print(f"    {pname}={vpar:>5}  recall {rec.mean():.4f}  "
                          f"scanned {frac * 100:6.2f}%"
                          + (f"  {t['qps_median']:8.1f} qps" if t else "  (untimed)"))
                    if rec.mean() >= 0.95 or frac >= SCAN_STOP:
                        break
                    if time.perf_counter() - ts > budget:
                        print("    -> wall-clock budget exceeded")
                        break
                if len(grid) > 1:
                    SW[(name, ik)] = rows
                del ix
                log.flush()
            del xb, xq, gi, gd, msq

    # ---- scoring ----
    v = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    u = {i: [FIX[(f"synth_uniform_d{d}", i)]["recall"] for d in DIMS]
         for i in ("hnsw", "ivf_flat")}
    mono = all(all(x >= y - 0.01 for x, y in zip(s, s[1:])) for s in u.values())
    under = all(s[-1] < 0.30 for s in u.values())
    h42 = mono and under
    h43_rows = [(i, FIX[(f"synth_uniform_d{control_d}", i)]["recall"],
                 FIX[(f"synth_clustered_d{control_d}", i)]["recall"])
                for i in ("hnsw", "ivf_flat")]
    h43 = all(c > 0.90 for _, _, c in h43_rows)
    h43_premise = all(uu < 0.60 for _, uu, _ in h43_rows)
    h44 = True
    for i in ("hnsw", "ivf_flat"):
        rr = SW.get((f"synth_uniform_d{DIMS[-1]}", i), [])
        hit = [r for r in rr if r["recall"] >= 0.95]
        if hit:
            h44 &= min(hit, key=lambda r: r["frac"])["frac"] >= 0.50
    h45 = (u["hnsw"][0] - u["hnsw"][-1]) >= (u["ivf_flat"][0] - u["ivf_flat"][-1])

    print("\n" + "=" * 80 + "\nHYPOTHESIS VERDICTS (Part 4)\n" + "=" * 80)
    print(f"H4.2  uniform recall collapses, both < 0.30 at d={DIMS[-1]}")
    for i, s in u.items():
        print(f"        {i:9s} " + " -> ".join(f"{x:.4f}" for x in s))
    print(f"        monotonic={mono}, both under 0.30={under}   -> {v(h42)}")
    print(f"\nH4.3  THE CONTROL: clustered d={control_d} > 0.90 where uniform < 0.60")
    for i, uu, cc in h43_rows:
        print(f"        {i:9s} uniform {uu:.4f}  clustered {cc:.4f}  (gap {cc - uu:+.4f})")
    print(f"        -> {v(h43)}" + ("" if h43_premise else
                                    "   (premise did not hold: weaker test than intended)"))
    print(f"\nH4.4  0.95 on uniform d={DIMS[-1]} needs >= 50% scanned  -> {v(h44)}")
    for i in ("hnsw", "ivf_flat"):
        rr = SW.get((f"synth_uniform_d{DIMS[-1]}", i), [])
        if rr:
            pk = max(rr, key=lambda r: r["recall"])
            print(f"        {i:9s} peak recall {pk['recall']:.4f} at "
                  f"{pk['frac'] * 100:.1f}% scanned")
    print(f"\nH4.5  HNSW degrades at least as fast as IVF-Flat (recorded as probably wrong)")
    print(f"        drop across the uniform row: hnsw "
          f"{u['hnsw'][0] - u['hnsw'][-1]:+.4f}, ivf_flat "
          f"{u['ivf_flat'][0] - u['ivf_flat'][-1]:+.4f}   -> {v(h45)}")
    print("        note: HNSW starts higher AND ends higher while scanning less, so the "
          "absolute-drop\n        criterion flatters IVF-Flat; see the report.")
    print("=" * 80)

    verdicts = dict(H4_2=h42, H4_3=h43, H4_4=h44, H4_5=h45)
    with open(f"{C.OUT}/results/part4_geometry.json", "w") as f:
        json.dump({"geometry": GEO, "verdicts": verdicts,
                   "fixed": {f"{k[0]}|{k[1]}": x for k, x in FIX.items()},
                   "sweeps": {f"{k[0]}|{k[1]}": x for k, x in SW.items()}}, f, indent=2)
    log.flush()
    print(f"wrote {C.OUT}/results/part4_geometry.json")


if __name__ == "__main__":
    main()
