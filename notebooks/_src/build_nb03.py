#!/usr/bin/env python3
"""Assemble nb03_part2_glove.ipynb (Part 2: HNSW / IVF-Flat / IVF-PQ on GloVe-100)."""

import json
import pathlib

SRC = pathlib.Path(__file__).parent
CORE = (SRC.parent.parent / "annbench_core.py").read_text()
TESTS = (SRC.parent.parent / "tests" / "test_recall.py").read_text()
OUT = SRC.parent / "nb03_part2_glove.ipynb"

DATASET = "glove"
TITLE = "nb03 — Part 2: HNSW, IVF-Flat and IVF-PQ on GloVe-100"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [

md(r"""
# A3 · nb03 — Part 2: three indexes on GloVe-100

**CS4.406 Assignment 3**  ·  prerequisites: `nb00`, `nb01`, `nb02`

The same grid as nb02, on GloVe-100 (1,183,514 vectors, 100-d). Vectors were unit-normalised
in nb00, so squared-L2 ranking is identical to cosine ranking and one exact baseline covers
both.

### What nb02 found on SIFT, and what to watch here

| index | best QPS @ recall 0.95 | note |
|---|---|---|
| HNSW `M=16` | 5817 | 257x the exact baseline's 22.6 qps |
| IVF-Flat `nlist=4096` | 1534 | 3.8x slower than HNSW |
| IVF-PQ | **never reached 0.95** | peak 0.886 at `m=64`; 0.42 / 0.60 / 0.74 / 0.89 for m = 8/16/32/64 |

GloVe is the harder geometry: nb00 measured **RC@10 = 1.51** against SIFT's **2.63**, i.e. the
10th neighbour is only ~1.5x nearer than a random point. Both the graph and the cell structure
have less signal to exploit, so expect lower QPS at matched recall and a search parameter that
has to climb further. The grids below are extended accordingly (`nprobe` to 2048, `efSearch`
to 1024) and the per-sweep budget is raised to 1800 s.

### One forced difference from nb02

FAISS requires `d % m == 0` for product quantisation. SIFT's d=128 allowed m = 8/16/32/64;
**d=100 does not**, so GloVe uses `m = 10/20/25/50` (divisors of 100), giving 10-50 bytes per
vector. The two PQ sweeps therefore span comparable compression ratios but cannot use identical
m values -- the report compares them by bytes per vector, not by m.

### Protocol (unchanged from nb02, so the curves are comparable)

Query timing 1 thread / one query at a time / 3 repeats / median; recall measured in a separate
multi-threaded batched pass; builds on all 4 cores; one shared coarse quantizer per `nlist`;
resumable from `results.csv`.
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
import os, sys, json, time, gc
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
import annbench_core as C

C.ensure_dirs()
ENV = C.env_fingerprint()
TOKEN = C.get_hf_token()
LOG = C.ResultsLog(ENV, token=TOKEN)

BUILD_THREADS = ENV["cpu_logical"]
print(f"\nmachine_id {ENV['machine_id']}  (nb00-nb02 ran on dc469939)")
if ENV["machine_id"] != "dc469939":
    print("  ! different machine from nb00-nb02 -- cross-dataset QPS is not comparable; note it")
"""),

md("""
## 2 · Data

`base` is needed in RAM throughout: the recall metric recomputes the exact distance of every
returned id from it, which is what keeps IVF-PQ honest.
"""),

code(r"""
DATASET = "glove"
D = C.load_prepared(DATASET, token=TOKEN, with_base=True)
base, query, gt_dists = D["base"], D["query"], D["gt_dists"]
N, DIM = base.shape
q_eval = np.ascontiguousarray(query[C.EVAL])
q_tune = np.ascontiguousarray(query[C.TUNE])
gt_eval = gt_dists[C.EVAL]
_ex = LOG.df[(LOG.df.part.astype(str) == "1") & (LOG.df.dataset == DATASET)
             & (LOG.df.search_param_value == "one_at_a_time")]
EXACT_QPS = float(_ex.qps_median.iloc[0]) if len(_ex) else None
print(f"\n{DATASET}: base {base.shape} · eval queries {q_eval.shape}")
print(f"exact baseline from nb01: "
      f"{f'{EXACT_QPS:.2f} qps @ recall 1.0' if EXACT_QPS else 'NOT FOUND - run nb01 first'}")
"""),

md("""
## 3 · The benchmark harness

`sweep()` walks a search-parameter grid for one built index. Two guards keep the session
finite:

* **stop once recall ≥ 0.995** — the curve is monotone in the search parameter, so anything
  past that is spent well beyond the 0.95 the report needs;
* **a per-sweep wall-clock budget** — if a grid turns out more expensive than predicted the
  sweep stops and says so in `results.csv` rather than silently eating the session.

Note the asymmetry: a sweep that has *not* reached 0.95 keeps going to the end of its grid.
That is deliberate — IVF-PQ is expected to hit a ceiling below 0.95, and a plateau is only
evidence if we actually ran the rest of the grid to show it.
"""),

code(r"""
K = 10
REPEATS = 3
RECALL_STOP = 0.995
SWEEP_BUDGET_S = 1800         # per (index, build) sweep; raised from nb02 (GloVe is harder)

def measure_recall(index, queries, gt_d):
    '''Batched, multi-threaded: recall is thread-independent, so buy it cheaply.'''
    faiss.omp_set_num_threads(BUILD_THREADS)
    _, I = index.search(queries, K)
    faiss.omp_set_num_threads(1)
    return C.recall_from_ids(I, base, queries, gt_d)

def sweep(index, kind, build_params, param_name, grid, setter,
          build_time_s=None, train_time_s=None, add_time_s=None,
          size_bytes=None, part="2", notes=""):
    rows, t_start = [], time.perf_counter()
    print(f"  {'param':>8s} {'recall@10':>10s} {'qps':>9s} {'p50 ms':>8s} "
          f"{'p95 ms':>8s} {'spread':>7s}")
    for v in grid:
        if LOG.done(part, DATASET, N, kind, build_params, param_name, v):
            print(f"  {v:>8} {'(already in results.csv, skipped)':>45s}")
            continue
        setter(index, v)
        rec = measure_recall(index, q_eval, gt_eval)
        C.set_single_thread()
        t = C.bench_single_query(lambda x: index.search(x, K), q_eval,
                                 repeats=REPEATS, warmup=30)
        LOG.add(part=part, dataset=DATASET, n_vectors=N, dim=DIM, metric="l2",
                index_kind=kind, build_params=build_params, search_param_name=param_name,
                search_param_value=v, recall=rec, timing=t, build_time_s=build_time_s,
                train_time_s=train_time_s, add_time_s=add_time_s,
                index_size_bytes=size_bytes, threads=1,
                notes=notes or f"build_threads={BUILD_THREADS}")
        r = float(rec.mean())
        rows.append((v, r, t["qps_median"]))
        print(f"  {v:>8} {r:>10.4f} {t['qps_median']:>9.1f} {t['lat_p50_ms']:>8.2f} "
              f"{t['lat_p95_ms']:>8.2f} {str(t['qps_spread_pct']) + '%':>7s}")
        if r >= RECALL_STOP:
            print(f"  -> recall {r:.4f} >= {RECALL_STOP}, stopping sweep")
            break
        if time.perf_counter() - t_start > SWEEP_BUDGET_S:
            print(f"  -> wall-clock budget {SWEEP_BUDGET_S}s exceeded, stopping sweep")
            LOG.add(part=part, dataset=DATASET, n_vectors=N, dim=DIM, metric="l2",
                    index_kind=kind, build_params=build_params,
                    search_param_name=param_name, search_param_value="BUDGET_STOP",
                    recall=np.array([r]), timing=t, notes="sweep truncated by time budget")
            break
    LOG.flush()
    return rows

RESULTS = {}
print("harness ready")
"""),

md("""
## 4 · Shared coarse quantizers

k-means at `nlist = 16384` over a million vectors is the single most expensive build step
here, so each `nlist` is trained exactly once and the resulting quantizer is handed to every
index that uses it. `own_fields = False` stops FAISS from freeing a quantizer that Python
still owns, and the dict below keeps those references alive.
"""),

code(r"""
IVF_NLIST = [1024, 4096, 16384]
QUANT, QTRAIN = {}, {}

for nlist in IVF_NLIST:
    faiss.omp_set_num_threads(BUILD_THREADS)
    t0 = time.perf_counter()
    probe = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, nlist, faiss.METRIC_L2)
    probe.train(base)
    QTRAIN[nlist] = time.perf_counter() - t0
    cent = probe.quantizer.reconstruct_n(0, nlist)
    q = faiss.IndexFlatL2(DIM); q.add(cent)
    QUANT[nlist] = (q, cent)
    del probe
    gc.collect()
    print(f"nlist={nlist:>6,}: coarse k-means {QTRAIN[nlist]:7.1f}s  "
          f"({nlist * 100 / N:.2f}% of n, {N // nlist} vectors/list)")
faiss.omp_set_num_threads(1)
"""),

md("""
## 5 · IVF-Flat

`nprobe` is swept on a geometric grid capped at `nlist`. Expressed as *work*, probing `p`
lists of an `nlist`-way partition scans about `p/nlist` of the base, so the three `nlist`
settings are being compared at matched scan fractions rather than at matched `nprobe`.
"""),

code(r"""
NPROBE_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
               1536, 2048]

def set_nprobe(ix, v):
    ix.nprobe = int(v)

for nlist in [1024, 4096]:          # cheap first; 16384 is built last
    bp = {"nlist": nlist}
    q, cent = QUANT[nlist]
    faiss.omp_set_num_threads(BUILD_THREADS)
    ix = faiss.IndexIVFFlat(q, DIM, nlist, faiss.METRIC_L2); ix.own_fields = False
    t0 = time.perf_counter(); ix.train(base); t_train = time.perf_counter() - t0
    assert np.allclose(ix.quantizer.reconstruct_n(0, nlist), cent), "coarse quantizer changed"
    t0 = time.perf_counter(); ix.add(base); t_add = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    size = C.index_size_bytes(ix)
    print(f"\nIVF-Flat nlist={nlist}: train {QTRAIN[nlist] + t_train:.1f}s (shared k-means "
          f"reused) · add {t_add:.1f}s · {C.human(size)} · {size / N:.0f} B/vec")
    RESULTS[("ivf_flat", nlist)] = sweep(
        ix, "ivf_flat", bp, "nprobe", [p for p in NPROBE_GRID if p <= nlist], set_nprobe,
        build_time_s=QTRAIN[nlist] + t_train + t_add, train_time_s=QTRAIN[nlist] + t_train,
        add_time_s=t_add, size_bytes=size)
    del ix
    gc.collect()
"""),

md("""
## 6 · IVF-PQ

`nlist` is held at 4096 and only the number of sub-quantizers `m` varies: 8, 16, 32 and 64
bytes per vector against IVF-Flat's 512. All four share the 4096-way coarse quantizer, so any
difference between them is attributable to quantisation alone.

PQ is the one index expected to hit a recall ceiling. Its returned distances are
*approximate*, so the metric recomputes exact distances for the returned ids — without that,
PQ's recall comes out dramatically overstated.
"""),

code(r"""
PQ_NLIST = 4096
PQ_M = [10, 20, 25, 50]           # divisors of DIM=100; FAISS requires d % m == 0
PQ_NBITS = 8

for m in PQ_M:
    bp = {"nlist": PQ_NLIST, "m": m, "nbits": PQ_NBITS}
    q, cent = QUANT[PQ_NLIST]
    faiss.omp_set_num_threads(BUILD_THREADS)
    ix = faiss.IndexIVFPQ(q, DIM, PQ_NLIST, m, PQ_NBITS); ix.own_fields = False
    t0 = time.perf_counter(); ix.train(base); t_train = time.perf_counter() - t0
    assert np.allclose(ix.quantizer.reconstruct_n(0, PQ_NLIST), cent), \
        "coarse quantizer was retrained -- the PQ configs would not be comparable"
    t0 = time.perf_counter(); ix.add(base); t_add = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    size = C.index_size_bytes(ix)
    print(f"\nIVF-PQ m={m} ({m} bytes/code): PQ train {t_train:.1f}s · add {t_add:.1f}s · "
          f"{C.human(size)} · {size / N:.1f} B/vec · "
          f"{(DIM * 4) / (size / N):.1f}x smaller than flat ({DIM * 4} B/vec)")
    RESULTS[("ivf_pq", m)] = sweep(
        ix, "ivf_pq", bp, "nprobe", [p for p in NPROBE_GRID if p <= PQ_NLIST], set_nprobe,
        build_time_s=QTRAIN[PQ_NLIST] + t_train + t_add, train_time_s=t_train,
        add_time_s=t_add, size_bytes=size)
    del ix
    gc.collect()
"""),

md("""
### 6b · Why PQ stops short — the mechanism, measured

nb02 showed IVF-PQ plateauing below 0.95 on SIFT at every code size. The explanation has to be
quantitative, so this measures the two quantities that compete:

* **quantisation error** — how far a PQ-reconstructed vector sits from the real one. Any
  distance PQ computes is wrong by up to this much.
* **the rank-10 margin** — the true distance gap between the 10th and 11th neighbours. This is
  the difference PQ must resolve to get the boundary of the top 10 right.

If the error is much larger than the margin, the ordering around rank 10 is essentially noise
and no amount of `nprobe` can fix it: more probes widen the candidate set, but the ranking
inside that set stays wrong. That is a ceiling, not a tuning problem.
"""),

code(r'''
m_diag = PQ_M[-1]
qd, centd = QUANT[PQ_NLIST]
faiss.omp_set_num_threads(BUILD_THREADS)
ixd = faiss.IndexIVFPQ(qd, DIM, PQ_NLIST, m_diag, PQ_NBITS); ixd.own_fields = False
ixd.train(base); ixd.add(base)
faiss.omp_set_num_threads(1)

N_SAMP = 20000
try:
    ixd.make_direct_map()
    recon = ixd.reconstruct_n(0, N_SAMP)
except Exception as e:
    print(f"  reconstruct_n unavailable ({e}); falling back to per-vector reconstruct")
    recon = np.vstack([ixd.reconstruct(i) for i in range(N_SAMP)])

err = np.linalg.norm(base[:N_SAMP].astype(np.float64) - recon.astype(np.float64), axis=1)
d10 = np.sqrt(gt_eval[:, 9])
d11 = np.sqrt(gt_eval[:, 10])
margin = d11 - d10

print(f"IVF-PQ m={m_diag} ({m_diag} bytes/vector), {N_SAMP:,} sampled base vectors")
print(f"  quantisation error ||x - decode(encode(x))||")
print(f"      median {np.median(err):.4f} · p90 {np.percentile(err, 90):.4f}")
print(f"  true distance to the 10th neighbour")
print(f"      median {np.median(d10):.4f}")
print(f"  rank-10 margin  d11 - d10")
print(f"      median {np.median(margin):.6f} · p90 {np.percentile(margin, 90):.6f}")
print(f"\n  error / margin  =  {np.median(err) / np.median(margin):,.0f}x")
print(f"  error / d10     =  {np.median(err) / np.median(d10) * 100:.1f}% of the "
      f"distance being measured")
print(f"\nReading: a distance computed against a PQ reconstruction is uncertain by about "
      f"{np.median(err):.3f},\nwhile deciding the 10th-vs-11th boundary correctly requires "
      f"resolving about {np.median(margin):.5f}.\nThe ordering at that boundary is therefore "
      f"dominated by quantisation noise, which is why\nrecall saturates below 0.95 no matter "
      f"how many lists are probed.")

PQ_DIAG = dict(m=m_diag, err_median=float(np.median(err)),
               err_p90=float(np.percentile(err, 90)), d10_median=float(np.median(d10)),
               margin_median=float(np.median(margin)),
               ratio=float(np.median(err) / np.median(margin)))
del ixd
gc.collect()
'''),

md("""
## 7 · HNSW

`efConstruction` is held at 200 and the graph degree `M` varies: 8, 16, 32. `efSearch` is the
search-time knob. Build cost is the thing to watch here — HNSW pays its search speed up front.
"""),

code(r"""
HNSW_M = [8, 16, 32]
EFC = 200
EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

def set_ef(ix, v):
    ix.hnsw.efSearch = int(v)

for M in HNSW_M:
    bp = {"M": M, "efConstruction": EFC}
    faiss.omp_set_num_threads(BUILD_THREADS)
    ix = faiss.IndexHNSWFlat(DIM, M)
    ix.hnsw.efConstruction = EFC
    t0 = time.perf_counter(); ix.add(base); t_build = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    size = C.index_size_bytes(ix)
    print(f"\nHNSW M={M} efC={EFC}: build {t_build:.1f}s ({t_build / 60:.1f} min, "
          f"{BUILD_THREADS} threads) · {C.human(size)} · {size / N:.0f} B/vec "
          f"(graph overhead {size / N - DIM * 4:.0f} B/vec)")
    RESULTS[("hnsw", M)] = sweep(
        ix, "hnsw", bp, "efSearch", EF_GRID, set_ef,
        build_time_s=t_build, add_time_s=t_build, size_bytes=size)
    del ix
    gc.collect()
"""),

md("""
## 8 · IVF-Flat at nlist = 16384

Left until last because its k-means is the most expensive step in the notebook. If the session
dies before this finishes, everything above is already in `results.csv` and a re-run resumes
here.
"""),

code(r"""
nlist = 16384
bp = {"nlist": nlist}
q, cent = QUANT[nlist]
faiss.omp_set_num_threads(BUILD_THREADS)
ix = faiss.IndexIVFFlat(q, DIM, nlist, faiss.METRIC_L2); ix.own_fields = False
t0 = time.perf_counter(); ix.train(base); t_train = time.perf_counter() - t0
assert np.allclose(ix.quantizer.reconstruct_n(0, nlist), cent), "coarse quantizer changed"
t0 = time.perf_counter(); ix.add(base); t_add = time.perf_counter() - t0
faiss.omp_set_num_threads(1)
size = C.index_size_bytes(ix)
print(f"\nIVF-Flat nlist={nlist}: train {QTRAIN[nlist] + t_train:.1f}s · add {t_add:.1f}s · "
      f"{C.human(size)} · {size / N:.0f} B/vec")
RESULTS[("ivf_flat", nlist)] = sweep(
    ix, "ivf_flat", bp, "nprobe", [p for p in NPROBE_GRID if p <= nlist], set_nprobe,
    build_time_s=QTRAIN[nlist] + t_train + t_add, train_time_s=QTRAIN[nlist] + t_train,
    add_time_s=t_add, size_bytes=size)
del ix
gc.collect()
LOG.flush()
"""),

md("""
## 9 · Sanity check

Not the report figure — nb06 builds those from `results.csv`. This is here to catch an
obviously broken curve before the session ends, while the indexes could still be rebuilt.
"""),

code(r"""
df = LOG.df
d2 = df[(df.part.astype(str) == "2") & (df.dataset == DATASET)
        & (df.search_param_value != "BUDGET_STOP")].copy()
d2["recall_at_10"] = d2["recall_at_10"].astype(float)
d2["qps_median"] = d2["qps_median"].astype(float)

fig, ax = plt.subplots(figsize=(8, 5.5))
styles = {"hnsw": ("o-", "#2563eb"), "ivf_flat": ("s-", "#059669"), "ivf_pq": ("^-", "#d97706")}
for (kind, bpj), grp in d2.groupby(["index_kind", "build_params"]):
    grp = grp.sort_values("recall_at_10")
    mk, col = styles.get(kind, ("x-", "gray"))
    lbl = f"{kind} {json.loads(bpj)}"
    ax.plot(grp.recall_at_10, grp.qps_median, mk, color=col, alpha=0.75, ms=4, lw=1.2,
            label=lbl)

if EXACT_QPS:
    ax.axhline(EXACT_QPS, color="crimson", ls="--", lw=1.4,
               label=f"exact baseline ({EXACT_QPS:.1f} qps @ recall 1.0)")
    ax.plot([1.0], [EXACT_QPS], "*", color="crimson", ms=13)
ax.axvline(0.95, color="black", ls=":", lw=1, alpha=0.5)
ax.set_yscale("log"); ax.set_xlabel("recall@10 (tie-aware)")
ax.set_ylabel("queries per second (1 thread, one at a time)")
ax.set_title(f"{DATASET.upper()} — recall vs throughput (sanity check)")
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=6.5, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig(f"{C.OUT}/figures/nb03_{DATASET}_sanity.png", dpi=130)
plt.show()
C.hf_upload(f"{C.OUT}/figures/nb03_{DATASET}_sanity.png",
            f"figures/nb03_{DATASET}_sanity.png", token=TOKEN, msg="nb03 sanity plot")
"""),

md("## 10 · Summary — paste this back"),

code(r"""
import math

def at_recall(grp, target):
    '''Best QPS reaching `target` recall, linearly interpolated along the curve.'''
    g = grp.sort_values("recall_at_10")
    r, qp = g.recall_at_10.values, g.qps_median.values
    if len(r) == 0 or r.max() < target:
        return None
    if r.min() >= target:
        return float(qp.max())
    i = int(np.searchsorted(r, target))
    if i == 0 or r[i] == r[i - 1]:
        return float(qp[i])
    w = (target - r[i - 1]) / (r[i] - r[i - 1])
    return float(math.exp(math.log(qp[i - 1]) * (1 - w) + math.log(qp[i]) * w))

df = LOG.df
d2 = df[(df.part.astype(str) == "2") & (df.dataset == DATASET)
        & (df.search_param_value != "BUDGET_STOP")].copy()
d2["recall_at_10"] = d2["recall_at_10"].astype(float)
d2["qps_median"] = d2["qps_median"].astype(float)

print("=" * 96); print(f"nb03 SUMMARY (Part 2 · {DATASET.upper()})"); print("=" * 96)
print(f"machine_id {ENV['machine_id']} · faiss {ENV['faiss']} · build_threads {BUILD_THREADS}")
print(f"results.csv now holds {len(LOG.df)} rows\n")
print(f"{'index':10s}{'build':32s}{'peak rec':>9s}{'qps@0.90':>10s}{'qps@0.95':>10s}"
      f"{'build s':>9s}{'size':>11s}{'B/vec':>8s}")
print("-" * 96)
best = {}
for (kind, bpj), grp in d2.groupby(["index_kind", "build_params"]):
    g = grp.copy()
    q90, q95 = at_recall(g, 0.90), at_recall(g, 0.95)
    bt = g.build_time_s.dropna().astype(float)
    sz = g.index_size_bytes.dropna().astype(float)
    print(f"{kind:10s}{bpj:32s}{g.recall_at_10.max():>9.4f}"
          f"{(f'{q90:.1f}' if q90 else '-'):>10s}{(f'{q95:.1f}' if q95 else '-'):>10s}"
          f"{(f'{bt.iloc[0]:.0f}' if len(bt) else '-'):>9s}"
          f"{(C.human(sz.iloc[0]) if len(sz) else '-'):>11s}"
          f"{(f'{sz.iloc[0] / N:.0f}' if len(sz) else '-'):>8s}")
    if q95:
        best.setdefault(kind, []).append((q95, bpj))

print()
for kind, lst in sorted(best.items()):
    q, bpj = max(lst)
    print(f"best at recall 0.95 · {kind:9s} {q:9.1f} qps  with {bpj}")
if "PQ_DIAG" in dir():
    pd_ = PQ_DIAG
    print(f"\npq mechanism (m={pd_['m']}): quantisation error {pd_['err_median']:.4f} vs "
          f"rank-10 margin {pd_['margin_median']:.6f}  ->  {pd_['ratio']:,.0f}x")

missing = [k for k in ("hnsw", "ivf_flat", "ivf_pq") if k not in best]
for k in missing:
    sub = d2[d2.index_kind == k]
    print(f"NEVER REACHED 0.95 · {k:9s} peak recall {sub.recall_at_10.max():.4f} "
          f"at {sub.loc[sub.recall_at_10.idxmax(), 'build_params']}")
print("=" * 96)
"""),

md("""
### Next

`nb03` runs the same grid on GloVe-100. Expect it to be harder and slower: nb00 measured
GloVe's relative contrast at **RC@10 = 1.51** against SIFT's **2.63**, meaning the 10th
neighbour is barely nearer than a random point, so IVF will need far more probes for the same
recall.

Paste the summary back, with any `NEVER REACHED 0.95` lines — those decide how far the GloVe
grids need to extend and whether the IVF-PQ ceiling is a SIFT quirk or general.
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
