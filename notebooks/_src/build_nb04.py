#!/usr/bin/env python3
"""Assemble nb04_part3_scaling.ipynb (Part 3: how the cost of recall 0.95 grows with n)."""

import json
import pathlib

SRC = pathlib.Path(__file__).parent
CORE = (SRC.parent.parent / "annbench_core.py").read_text()
TESTS = (SRC.parent.parent / "tests" / "test_recall.py").read_text()
OUT = SRC.parent / "nb04_part3_scaling.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [

md(r"""
# A3 · nb04 — Part 3: scaling on SIFT (100K / 300K / 1M)

**CS4.406 Assignment 3**  ·  prerequisites: `nb00`–`nb03`, and `hypothesis.md` committed

How does the cost of reaching **recall@10 = 0.95** grow with the number of vectors, for HNSW,
IVF-Flat and exact search?

Subsets are the nested, seeded ones nb00 built (100K ⊂ 300K ⊂ 1M, seed 20261023), so a
difference between sizes is attributable to size and not to having sampled a different region
of the space. Ground truth was recomputed per subset on the T4.

### Predictions on record (committed before this ran)

| | prediction | falsified if |
|---|---|---|
| **H3.1** | exact scales as n^-1.0; ~226 qps @100K, ~75 @300K | exponent outside [-1.15, -0.85] |
| **H3.2** | HNSW exponent -0.2 to -0.4; under 4x faster at 100K than at 1M | ratio > 4x or exponent steeper than -0.5 |
| **H3.3** | IVF-Flat strictly between HNSW and exact (-0.5 to -0.8) | not strictly between |
| **H3.4** | HNSW's 3.8x edge over IVF-Flat shrinks below 2.5x at 100K | gap >= 2.5x, or widens as n falls |
| **H3.5** | HNSW build seconds-per-vector rises >= 20% from 100K to 1M | it does not |
| **H3.6** | ANN still beats exact by >10x even at 100K | advantage under 10x |

### Protocol: the search parameter is picked on the TUNE split

This is the part the assignment's query split exists for. For every build setting we walk the
search-parameter grid **on the tune queries (0–4999), recall only**, and stop at the *first*
value reaching 0.95. That value — chosen without ever looking at the eval half — is then timed
on the **eval queries (5000–9999)**, and that timing is the reported number.

Selecting the parameter this way is also much cheaper than timing the whole grid, which is what
makes three sizes x three index types affordable in one session.

Config selection (which `M`, which `nlist`) still compares eval QPS across 3–4 candidates per
index per size. That residual is small: a 5000-query recall mean has a standard error near
0.0014, far below the gaps involved. The report states it.

Builds use all 4 cores; query timing is strictly 1 thread, one query at a time, 3 repeats,
median. Runtime ≈ 1 hour.
"""),

md("## 1 · Setup"),

code(r"""
import subprocess, sys

def pip(*a):
    print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a],
                         capture_output=True, text=True).stdout[-400:] or "ok")
try:
    import faiss
except ImportError:
    pip("faiss-cpu"); import faiss
for m, p in [("h5py", "h5py"), ("huggingface_hub", "huggingface_hub")]:
    try: __import__(m)
    except ImportError: pip(p)
print("faiss", faiss.__version__)
"""),

code("%%writefile annbench_core.py\n" + CORE),

code("%%writefile test_recall.py\n" + TESTS),

code(r"""
import os, sys, json, time, gc, math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
import annbench_core as C

C.ensure_dirs()
ENV = C.env_fingerprint()
TOKEN = C.get_hf_token()
LOG = C.ResultsLog(ENV, token=TOKEN)
BUILD_THREADS = ENV["cpu_logical"]
print(f"\nmachine_id {ENV['machine_id']}  (nb00-nb03 ran on dc469939)")
if ENV["machine_id"] != "dc469939":
    print("  ! different machine -- Part 3 QPS will not be comparable with Part 2; note it")
"""),

md("""
## 2 · Data and the nested subsets

Only the ids of each subset were stored, so the vectors come from fancy-indexing the 1M base.
Ground-truth ids for a subset are **local** (0..n-1), matching what an index built on that
subset returns.
"""),

code(r"""
D = C.load_prepared("sift", token=TOKEN, with_base=True)
base_full, query = D["base"], D["query"]
q_tune = np.ascontiguousarray(query[C.TUNE])
q_eval = np.ascontiguousarray(query[C.EVAL])

SIZES = [100_000, 300_000, 1_000_000]
SUB = {}
for n in SIZES:
    if n == len(base_full):
        sub, gt = base_full, D["gt_dists"]
    else:
        tag = f"sift{n // 1000}k"
        ids = np.load(C.fetch(f"data/{tag}_ids.npy", TOKEN))
        sub = np.ascontiguousarray(base_full[ids])
        gt = np.load(C.fetch(f"gt/{tag}_top100_dists.npy", TOKEN))
        assert len(ids) == n
    SUB[n] = dict(base=sub, gt_tune=gt[C.TUNE], gt_eval=gt[C.EVAL])
    print(f"n={n:>9,}  base {sub.shape}  {C.human(sub.nbytes)}")

# nesting sanity: the subsets really are prefixes of one permutation
i100 = set(np.load(C.fetch("data/sift100k_ids.npy", TOKEN, verbose=False)).tolist())
i300 = set(np.load(C.fetch("data/sift300k_ids.npy", TOKEN, verbose=False)).tolist())
assert i100 <= i300, "subsets are not nested"
print("\nsubsets nested: 100K subset of 300K subset of 1M")
"""),

md("""
## 3 · Exact search at each size

The reference line. Timed exactly as in nb01: one query at a time, one thread, 3 repeats,
median, on a 1000-query sample of the eval half (at ~22 qps the full 5000 would cost 45 min
per size for no extra precision).
"""),

code(r"""
K = 10
REPEATS = 3
N_SINGLE = 1000
exact_rows = {}

for n in SIZES:
    s = SUB[n]
    if LOG.done("3", "sift", n, "exact_flat", {}, "query_mode", "one_at_a_time"):
        prev = LOG.df[(LOG.df.part.astype(str) == "3") & (LOG.df.n_vectors.astype(str) == str(n))
                      & (LOG.df.index_kind == "exact_flat")]
        exact_rows[n] = float(prev.qps_median.iloc[0])
        print(f"n={n:>9,}  exact {exact_rows[n]:8.2f} qps  (from results.csv)")
        continue
    C.set_single_thread()
    ix = faiss.IndexFlatL2(ix_dim := s["base"].shape[1])
    t0 = time.perf_counter(); ix.add(s["base"]); t_add = time.perf_counter() - t0
    size = C.index_size_bytes(ix)
    qs = q_eval[:N_SINGLE]
    _, I = ix.search(qs, K)
    rec = C.recall_from_ids(I, s["base"], qs, s["gt_eval"][:N_SINGLE])
    assert rec.mean() == 1.0, "exact search must score 1.0 against its own ground truth"
    t = C.bench_single_query(lambda x: ix.search(x, K), qs, repeats=REPEATS, warmup=20)
    LOG.add(part="3", dataset="sift", n_vectors=n, dim=ix_dim, metric="l2",
            index_kind="exact_flat", build_params={}, search_param_name="query_mode",
            search_param_value="one_at_a_time", recall=rec, timing=t, build_time_s=t_add,
            add_time_s=t_add, index_size_bytes=size, threads=1,
            notes=f"scaling reference; first {N_SINGLE} eval queries")
    exact_rows[n] = t["qps_median"]
    print(f"n={n:>9,}  exact {t['qps_median']:8.2f} qps · p50 {t['lat_p50_ms']:7.2f} ms · "
          f"{C.human(size)}")
    del ix; gc.collect()
LOG.flush()
"""),

md("""
## 4 · Harness: screen on tune, time on eval

`cheapest_at_target` walks the grid on the tune queries and returns the first parameter value
whose **tune** recall reaches 0.95, or `None` if the grid never gets there. Only that single
point is then timed on eval. Screening is recall-only and multi-threaded, so it is cheap; the
one timed measurement is strictly single-threaded.
"""),

code(r"""
TARGET = 0.95

def recall_on(index, queries, gt_d, base_arr):
    faiss.omp_set_num_threads(BUILD_THREADS)
    _, I = index.search(queries, K)
    faiss.omp_set_num_threads(1)
    return C.recall_from_ids(I, base_arr, queries, gt_d)

def cheapest_at_target(index, n, grid, setter, label):
    '''First grid value whose TUNE recall reaches TARGET. Returns (value, tune_recall, trace).'''
    s = SUB[n]
    trace = []
    for v in grid:
        setter(index, v)
        r = float(recall_on(index, q_tune, s["gt_tune"], s["base"]).mean())
        trace.append((v, r))
        if r >= TARGET:
            print(f"    tune screen {label}: {len(trace)} points, picked {v} "
                  f"(tune recall {r:.4f})")
            return v, r, trace
    print(f"    tune screen {label}: NEVER reached {TARGET} "
          f"(best {max(t[1] for t in trace):.4f} at {max(trace, key=lambda t: t[1])[0]})")
    return None, max(t[1] for t in trace), trace

def time_point(index, n, kind, bp, pname, v, setter, build_s, size, notes=""):
    s = SUB[n]
    setter(index, v)
    rec = recall_on(index, q_eval, s["gt_eval"], s["base"])
    C.set_single_thread()
    t = C.bench_single_query(lambda x: index.search(x, K), q_eval, repeats=REPEATS, warmup=30)
    LOG.add(part="3", dataset="sift", n_vectors=n, dim=s["base"].shape[1], metric="l2",
            index_kind=kind, build_params=bp, search_param_name=pname, search_param_value=v,
            recall=rec, timing=t, build_time_s=build_s, add_time_s=build_s,
            index_size_bytes=size, threads=1,
            notes=notes or f"param chosen on tune split; build_threads={BUILD_THREADS}")
    print(f"    eval: recall {rec.mean():.4f} · {t['qps_median']:8.1f} qps · "
          f"p50 {t['lat_p50_ms']:.2f} ms · spread {t['qps_spread_pct']}%")
    return dict(n=n, kind=kind, build_params=bp, param=v, recall=float(rec.mean()),
                qps=t["qps_median"], build_s=build_s, size=size)

SCAL = []
print("harness ready")
"""),

md("""
## 5 · HNSW at each size

`M ∈ {8, 16, 32}` with `efConstruction = 200`, the same set at every size so the comparison is
about n alone. nb02 found `M=16` best on SIFT at 1M; whether that holds at 100K is one of the
things H3.4 is about.
"""),

code(r"""
HNSW_M = [8, 16, 32]
EFC = 200
EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]

def set_ef(ix, v):
    ix.hnsw.efSearch = int(v)

for n in SIZES:
    s = SUB[n]
    dim = s["base"].shape[1]
    print(f"\n--- HNSW · n={n:,} ---")
    for M in HNSW_M:
        bp = {"M": M, "efConstruction": EFC}
        faiss.omp_set_num_threads(BUILD_THREADS)
        ix = faiss.IndexHNSWFlat(dim, M); ix.hnsw.efConstruction = EFC
        t0 = time.perf_counter(); ix.add(s["base"]); t_build = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        size = C.index_size_bytes(ix)
        print(f"  M={M:<3d} build {t_build:7.1f}s ({t_build / n * 1e6:5.1f} us/vector) · "
              f"{C.human(size)} · {size / n:.0f} B/vec")
        v, r, _ = cheapest_at_target(ix, n, EF_GRID, set_ef, f"M={M}")
        if v is not None:
            SCAL.append(time_point(ix, n, "hnsw", bp, "efSearch", v, set_ef, t_build, size))
        else:
            SCAL.append(dict(n=n, kind="hnsw", build_params=bp, param=None, recall=r,
                             qps=None, build_s=t_build, size=size))
        del ix; gc.collect()
    LOG.flush()
"""),

md("""
## 6 · IVF-Flat at each size

`nlist ∈ {1024, 2048, 4096, 8192}`, again the same set at every size. At 100K, `nlist=8192`
leaves only ~12 vectors per list — a deliberately bad setting that the tune screen is free to
reject, rather than one excluded by hand.
"""),

code(r"""
IVF_NLIST = [1024, 2048, 4096, 8192]
NPROBE_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

def set_nprobe(ix, v):
    ix.nprobe = int(v)

for n in SIZES:
    s = SUB[n]
    dim = s["base"].shape[1]
    print(f"\n--- IVF-Flat · n={n:,} ---")
    for nlist in IVF_NLIST:
        bp = {"nlist": nlist}
        faiss.omp_set_num_threads(BUILD_THREADS)
        ix = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
        t0 = time.perf_counter(); ix.train(s["base"]); ix.add(s["base"])
        t_build = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        size = C.index_size_bytes(ix)
        print(f"  nlist={nlist:<5d} build {t_build:7.1f}s · {C.human(size)} · "
              f"{n // nlist} vectors/list")
        grid = [p for p in NPROBE_GRID if p <= nlist]
        v, r, _ = cheapest_at_target(ix, n, grid, set_nprobe, f"nlist={nlist}")
        if v is not None:
            frac = v / nlist
            note = (f"param chosen on tune split; build_threads={BUILD_THREADS}; "
                    f"nprobe/nlist={frac:.2f}")
            if frac > 0.5:
                print(f"    ! nprobe={v} is {frac:.0%} of nlist -- this IVF has effectively "
                      f"degenerated into a full scan, so its 'cost of 0.95' is really "
                      f"exact-search cost plus quantizer overhead")
                note += " DEGENERATE(>50% of lists probed)"
            SCAL.append(time_point(ix, n, "ivf_flat", bp, "nprobe", v, set_nprobe,
                                   t_build, size, notes=note))
            SCAL[-1]["frac_lists"] = frac
        else:
            SCAL.append(dict(n=n, kind="ivf_flat", build_params=bp, param=None, recall=r,
                             qps=None, build_s=t_build, size=size))
        del ix; gc.collect()
    LOG.flush()
"""),

md("## 7 · Scaling exponents and the verdict on each hypothesis"),

code(r"""
import pandas as pd

rows = [r for r in SCAL if r["qps"] is not None]
sc = pd.DataFrame(rows)
best = (sc.sort_values("qps", ascending=False)
          .groupby(["kind", "n"], as_index=False).first())
for n in SIZES:
    best = pd.concat([best, pd.DataFrame([dict(kind="exact_flat", n=n, qps=exact_rows[n],
                                               recall=1.0, param="full scan",
                                               build_params={}, build_s=None, size=None)])],
                     ignore_index=True)

print(f"{'index':11s}{'n':>11s}{'setting':>28s}{'recall':>9s}{'qps':>10s}"
      f"{'build s':>9s}{'B/vec':>8s}")
print("-" * 86)
for kind in ("exact_flat", "ivf_flat", "hnsw"):
    for n in SIZES:
        r = best[(best.kind == kind) & (best.n == n)]
        if not len(r):
            continue
        r = r.iloc[0]
        setting = (f"{json.dumps(r.build_params)} @ {r.param}"
                   if kind != "exact_flat" else "full scan")
        print(f"{kind:11s}{r.n:>11,}{setting:>28s}{r.recall:>9.4f}{r.qps:>10.1f}"
              f"{(f'{r.build_s:.0f}' if r.build_s else '-'):>9s}"
              f"{(f'{r.size / r.n:.0f}' if r.size else '-'):>8s}")

EXP = {}
print(f"\n{'index':11s}{'exponent a in qps ~ n^a':>26s}{'100K/1M ratio':>16s}")
print("-" * 53)
for kind in ("exact_flat", "ivf_flat", "hnsw"):
    g = best[best.kind == kind].sort_values("n")
    if len(g) < 2:
        continue
    a = float(np.polyfit(np.log(g.n.values.astype(float)), np.log(g.qps.values.astype(float)), 1)[0])
    ratio = float(g.qps.iloc[0] / g.qps.iloc[-1])
    EXP[kind] = dict(exponent=a, ratio=ratio)
    print(f"{kind:11s}{a:>26.3f}{ratio:>15.2f}x")
"""),

code(r"""
def verdict(ok):
    return "SUPPORTED" if ok else "NOT SUPPORTED"

print("=" * 86); print("HYPOTHESIS VERDICTS (Part 3)"); print("=" * 86)

e = EXP["exact_flat"]["exponent"]
h31 = -1.15 <= e <= -0.85
print(f"H3.1  exact scales as n^-1.0              exponent {e:+.3f}   -> {verdict(h31)}")
print(f"      H3.1 also predicted (for SIFT) ~226 qps at 100K and ~75 at 300K; "
      f"measured {exact_rows[SIZES[0]]:.1f} and {exact_rows[SIZES[1]]:.1f}")

hx, hr = EXP["hnsw"]["exponent"], EXP["hnsw"]["ratio"]
h32 = hr < 4.0 and hx >= -0.5
print(f"\nH3.2  HNSW exponent -0.2..-0.4, <4x       exponent {hx:+.3f}, ratio {hr:.2f}x"
      f"  -> {verdict(h32)}")

ix_ = EXP["ivf_flat"]["exponent"]
h33 = EXP["hnsw"]["exponent"] > ix_ > e
print(f"\nH3.3  IVF-Flat strictly between           {e:+.3f} < {ix_:+.3f} < {hx:+.3f}"
      f"  -> {verdict(h33)}")

gaps = {}
for n in SIZES:
    h = best[(best.kind == "hnsw") & (best.n == n)]
    i = best[(best.kind == "ivf_flat") & (best.n == n)]
    if len(h) and len(i):
        gaps[n] = float(h.qps.iloc[0]) / float(i.qps.iloc[0])
h34 = gaps.get(100_000, 99) < 2.5 and gaps.get(100_000, 0) < gaps.get(1_000_000, 0)
print(f"\nH3.4  HNSW/IVF gap shrinks below 2.5x@100K  "
      + " ".join(f"{n // 1000}K:{g:.2f}x" for n, g in sorted(gaps.items()))
      + f"  -> {verdict(h34)}")

bpv = {}
for n in SIZES:
    h = best[(best.kind == "hnsw") & (best.n == n)]
    if len(h) and h.build_s.iloc[0]:
        bpv[n] = float(h.build_s.iloc[0]) / n * 1e6
h35 = (bpv.get(1_000_000, 0) / bpv.get(100_000, 1)) >= 1.2 if len(bpv) >= 2 else False
print(f"\nH3.5  HNSW build us/vector rises >=20%      "
      + " ".join(f"{n // 1000}K:{v:.1f}us" for n, v in sorted(bpv.items()))
      + f"  ({bpv.get(1_000_000, 0) / max(bpv.get(100_000, 1), 1e-9):.2f}x)"
      + f"  -> {verdict(h35)}")

adv = {n: float(best[(best.kind == 'hnsw') & (best.n == n)].qps.iloc[0]) / exact_rows[n]
       for n in SIZES if len(best[(best.kind == 'hnsw') & (best.n == n)])}
h36 = min(adv.values()) > 10
print(f"\nH3.6  ANN >10x exact at every size         "
      + " ".join(f"{n // 1000}K:{a:.0f}x" for n, a in sorted(adv.items()))
      + f"  -> {verdict(h36)}")
print("=" * 86)

VERDICTS = dict(H3_1=h31, H3_2=h32, H3_3=h33, H3_4=h34, H3_5=h35, H3_6=h36)
"""),

md("## 8 · Figure"),

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
style = {"exact_flat": ("crimson", "d--", "exact (full scan)"),
         "ivf_flat": ("#059669", "s-", "IVF-Flat"),
         "hnsw": ("#2563eb", "o-", "HNSW")}

ax = axes[0]
for kind, (col, mk, lab) in style.items():
    g = best[best.kind == kind].sort_values("n")
    if not len(g):
        continue
    ax.plot(g.n, g.qps, mk, color=col, lw=1.8, ms=6,
            label=f"{lab}  (n^{EXP[kind]['exponent']:+.2f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of vectors"); ax.set_ylabel("queries/s at recall@10 = 0.95")
ax.set_title("Cost of reaching recall 0.95 vs dataset size\n(SIFT, 1 thread, one query at a time)",
             fontsize=10)
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
ax.set_xticks(SIZES); ax.set_xticklabels([f"{n // 1000}K" if n < 1e6 else "1M" for n in SIZES])

ax = axes[1]
for kind, (col, mk, lab) in style.items():
    g = best[(best.kind == kind) & best.build_s.notna()].sort_values("n")
    if not len(g):
        continue
    ax.plot(g.n, g.build_s.astype(float), mk, color=col, lw=1.8, ms=6, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of vectors"); ax.set_ylabel("build time (s, 4 threads)")
ax.set_title("Build cost vs dataset size", fontsize=10)
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
ax.set_xticks(SIZES); ax.set_xticklabels([f"{n // 1000}K" if n < 1e6 else "1M" for n in SIZES])

fig.tight_layout()
fig.savefig(f"{C.OUT}/figures/part3_scaling.png", dpi=150)
plt.show()
C.hf_upload(f"{C.OUT}/figures/part3_scaling.png", "figures/part3_scaling.png",
            token=TOKEN, msg="nb04 part3 figure")
"""),

md("## 9 · Summary — paste this back"),

code(r"""
print("=" * 86); print("nb04 SUMMARY (Part 3 · scaling on SIFT)"); print("=" * 86)
print(f"machine_id {ENV['machine_id']} · faiss {ENV['faiss']} · "
      f"results.csv now holds {len(LOG.df)} rows\n")
print(f"{'index':11s}{'100K':>22s}{'300K':>22s}{'1M':>22s}")
print("-" * 79)
for kind in ("exact_flat", "ivf_flat", "hnsw"):
    cells_ = []
    for n in SIZES:
        r = best[(best.kind == kind) & (best.n == n)]
        if not len(r):
            cells_.append("-"); continue
        r = r.iloc[0]
        p = "" if kind == "exact_flat" else f" @{r.param}"
        cells_.append(f"{r.qps:.1f} qps{p}")
    print(f"{kind:11s}" + "".join(c.rjust(22) for c in cells_))

print(f"\n{'index':11s}{'exponent':>12s}{'100K/1M':>10s}")
for kind, v in EXP.items():
    print(f"{kind:11s}{v['exponent']:>+12.3f}{v['ratio']:>9.2f}x")

print("\nbest setting per size")
for kind in ("ivf_flat", "hnsw"):
    for n in SIZES:
        r = best[(best.kind == kind) & (best.n == n)]
        if len(r):
            print(f"  {kind:9s} n={n:>9,}  {json.dumps(r.iloc[0].build_params)} "
                  f"@ {r.iloc[0].param}  (recall {r.iloc[0].recall:.4f})")

deg = [r for r in SCAL if r.get("frac_lists", 0) > 0.5]
if deg:
    print("\nIVF settings that degenerated into near-full scans (nprobe > 50% of nlist)")
    for r in deg:
        print(f"  n={r['n']:>9,} {json.dumps(r['build_params'])} @ nprobe={r['param']} "
              f"({r['frac_lists']:.0%} of lists)")

print("\nconfigs that never reached 0.95 on the tune split")
none_ = [r for r in SCAL if r["qps"] is None]
print("  none" if not none_ else
      "\n".join(f"  {r['kind']:9s} n={r['n']:>9,} {json.dumps(r['build_params'])} "
                f"peak tune recall {r['recall']:.4f}" for r in none_))

print("\nhypothesis verdicts: " + ", ".join(
    f"{k.replace('_', '.')}={'Y' if v else 'N'}" for k, v in VERDICTS.items()))
print("=" * 86)
"""),

md("""
### Next

`nb05` runs Part 4: the synthetic distance-concentration experiment from `gen_synthetic.py`,
whose hypotheses are already committed. Paste this summary back first — H3.4's measured gaps
tell me how much headroom the Part 4 budgets need, and any `never reached 0.95` rows change
which settings are worth testing there.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size / 1024:.0f} KB)")
