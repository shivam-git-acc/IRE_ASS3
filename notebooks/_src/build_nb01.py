#!/usr/bin/env python3
"""Assemble nb01_part1_exact.ipynb (Part 1: exact baseline, recall function, timing)."""

import json
import pathlib

SRC = pathlib.Path(__file__).parent
CORE = (SRC.parent.parent / "annbench_core.py").read_text()
TESTS = (SRC.parent.parent / "tests" / "test_recall.py").read_text()
OUT = SRC.parent / "nb01_part1_exact.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [

md(r"""
# A3 · nb01 — Part 1: exact baseline

**CS4.406 Assignment 3**  ·  prerequisite: `nb00_setup_data` must have run

Part 1 asks for four things, and this notebook does them in order:

1. **exact search with FAISS**, used as the reference for everything later
2. **our own recall@10**, written so that *ties at the 10th neighbour are not counted as
   mistakes*, with unit tests
3. a check that **exact results agree with the neighbours stored in the data file**
4. exact search timed **one query at a time** and **all queries in one batch**, with an
   explanation of why they differ so much

Runtime ≈ 20–25 minutes, most of it the ~1 GB pull from the HF mirror.

> Settings: **GPU T4 x2** (only for consistency with the other notebooks — Part 1 is pure
> CPU), **Internet On**, secret `HF_TOKEN` attached.
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
sys.path.insert(0, os.getcwd())
import annbench_core as C

C.ensure_dirs()
ENV = C.env_fingerprint()
TOKEN = C.get_hf_token()
LOG = C.ResultsLog(ENV, token=TOKEN)
print(f"\nmachine_id {ENV['machine_id']} — must match nb00's for the numbers to be comparable")
"""),

md("""
## 2 · Deliverable 2 — the recall@10 function and its tests

The metric lives in `annbench_core.recall_at_k`. The decision that matters:

> A returned id counts as correct when **its exact distance is ≤ the exact distance of the
> true 10th neighbour**, not when it appears in the stored top-10 id list.

If the true 10th and 11th neighbours sit at the same distance, an index returning either is
equally right, so an id-set test would punish a correct answer. `recall_from_ids` recomputes
the exact distances of the returned ids from the base vectors, which also makes it safe for
IVF-PQ — that index reports *approximate* distances, and feeding those to the metric inflates
its recall badly (we show this in nb02).

The tests cover: exact result, permuted top-10, one genuine miss, all wrong, **a tie at rank
10**, all-distances-identical, zero-distance ground truth, duplicate ids, `-1` padding,
PQ-style lying distances, and agreement with an independent naive implementation.
"""),

code(r"""
r = subprocess.run([sys.executable, "-m", "pytest", "-v", "--tb=short", "test_recall.py"],
                   capture_output=True, text=True)
print(r.stdout[-4000:])
assert r.returncode == 0, "recall unit tests failed"
"""),

md("""
## 3 · Load the prepared data

`base`/`query` are exactly the arrays nb00 wrote; `gt_*` is our exact top-100 ground truth;
`stored` is the neighbour list shipped inside the ann-benchmarks HDF5 file, kept so we can
check ourselves against the dataset authors.
"""),

code(r"""
DATASETS = ["sift", "glove"]
DATA = {}
for name in DATASETS:
    print(f"[{name}]")
    DATA[name] = C.load_prepared(name, token=TOKEN, with_base=True, with_stored=True)
    d = DATA[name]
    print(f"  base {d['base'].shape} {d['base'].dtype} · query {d['query'].shape} "
          f"· gt {d['gt_ids'].shape}")
"""),

md("""
## 4 · Deliverable 3 — do our exact results agree with the data file?

Two questions, deliberately kept apart:

* **Does FAISS exact search reproduce our ground truth?** Must be exactly 1.0, or our
  reference is broken.
* **Do the dataset's own stored neighbours agree with ours?** This is where ties show up.
  nb00 already found that only ~93% of SIFT queries have a *positionally identical* top-10,
  while the tie-aware recall is a clean 1.000000 — so the disagreements are ties, not errors.
  The table below separates the three ways of asking.
""" ),

code(r"""
TIE_TOL = 1e-9
agreement = {}

for name in DATASETS:
    d = DATA[name]
    base, query, gt_ids, gt_dists, stored = (d["base"], d["query"], d["gt_ids"],
                                             d["gt_dists"], d["stored"])
    n, dim = base.shape

    C.set_single_thread()
    index = faiss.IndexFlatL2(dim)
    t0 = time.perf_counter(); index.add(base); add_s = time.perf_counter() - t0
    size = C.index_size_bytes(index)
    DATA[name]["index"] = index
    DATA[name]["index_size"] = size
    DATA[name]["add_s"] = add_s

    # Verification search uses all threads -- we are checking correctness here, not timing.
    faiss.omp_set_num_threads(ENV["cpu_logical"])
    _, I = index.search(np.ascontiguousarray(query[C.EVAL]), 10)
    faiss.omp_set_num_threads(1)

    rec_exact = C.recall_from_ids(I, base, query[C.EVAL], gt_dists[C.EVAL])
    naive_exact = np.array([len(set(a.tolist()) & set(b.tolist())) / 10
                            for a, b in zip(I, gt_ids[C.EVAL, :10])])

    # how the FILE's neighbours compare with ours, over all 10k queries
    rec_file = C.recall_from_ids(stored[:, :10], base, query, gt_dists)
    naive_file = np.array([len(set(a.tolist()) & set(b.tolist())) / 10
                           for a, b in zip(stored[:, :10], gt_ids[:, :10])])
    positional = (stored[:, :10] == gt_ids[:, :10]).all(1).mean()

    # tie structure of the ground truth itself
    boundary_tie = np.isclose(gt_dists[:, 9], gt_dists[:, 10], rtol=TIE_TOL, atol=0)
    top10 = gt_dists[:, :11]
    any_tie = np.isclose(top10[:, 1:], top10[:, :-1], rtol=TIE_TOL, atol=0).any(1)

    agreement[name] = dict(
        n=n, dim=dim, add_s=add_s, index_size=size,
        exact_tie_aware=float(rec_exact.mean()), exact_naive=float(naive_exact.mean()),
        file_tie_aware=float(rec_file.mean()), file_naive=float(naive_file.mean()),
        file_positional=float(positional),
        boundary_tie=float(boundary_tie.mean()), any_tie=float(any_tie.mean()))

    print(f"[{name}] IndexFlatL2 add {add_s:.1f}s · {C.human(size)} · {size / n:.0f} B/vector")
    assert rec_exact.mean() == 1.0, "FAISS exact search disagrees with our ground truth"
    del I
    gc.collect()
"""),

code(r"""
hdr = f"{'':46s}" + "".join(f"{n:>12s}" for n in DATASETS)
print(hdr); print("-" * len(hdr))
rows = [
    ("FAISS exact vs our GT, tie-aware",      "exact_tie_aware", "{:.6f}"),
    ("FAISS exact vs our GT, naive id-overlap", "exact_naive",    "{:.6f}"),
    ("", None, None),
    ("file's neighbours vs ours, tie-aware",  "file_tie_aware",  "{:.6f}"),
    ("file's neighbours vs ours, naive id-overlap", "file_naive", "{:.6f}"),
    ("file's neighbours vs ours, positional match", "file_positional", "{:.4f}"),
    ("", None, None),
    ("queries where d10 == d11 (boundary tie)", "boundary_tie",  "{:.4f}"),
    ("queries with any tie inside the top 11",  "any_tie",       "{:.4f}"),
]
for label, key, fmt in rows:
    if key is None:
        print(); continue
    print(f"{label:46s}" + "".join(fmt.format(agreement[n][key]).rjust(12) for n in DATASETS))

print("\nReading:")
for name in DATASETS:
    a = agreement[name]
    lost = a["file_tie_aware"] - a["file_naive"]
    print(f"  {name}: {a['boundary_tie'] * 100:.2f}% of queries have a tie exactly at the 10th "
          f"neighbour, and {a['any_tie'] * 100:.2f}% have one somewhere in the top 11. A naive "
          f"id-overlap metric scores the dataset's own published neighbours "
          f"{a['file_naive']:.4f} rather than {a['file_tie_aware']:.4f} -- it throws away "
          f"{lost * 100:.2f} recall points for answers that are not wrong, and would put that "
          f"same floor under every index we benchmark.")
"""),

md("""
### Why this matters beyond bookkeeping

SIFT vectors are integer-valued, so exact distance ties are common; GloVe is continuous and
has almost none. A naive id-overlap metric therefore puts a **floor under SIFT recall that no
index can cross** — it would mark the dataset's own published ground truth as imperfect. Every
recall number in this project uses the distance-threshold definition instead.
"""),

md("""
## 5 · Deliverable 4 — timing, one query at a time vs one batch

Three measurements per dataset, all on the same `IndexFlatL2`:

| mode | threads | what it isolates |
|---|---|---|
| `one_at_a_time` | 1 | the protocol every ANN number in this project uses |
| `batch` | 1 | the effect of batching **alone**, threads held fixed |
| `batch_4threads` | 4 | what you would actually do in production |

Keeping the single-thread batch in the middle is the point: without it, the one-vs-batch gap
confounds batching with parallelism, and the "why" in the report would be wrong.

Exact search is slow enough one-at-a-time (tens of ms per query) that timing all 5000 would
burn ~45 min per dataset for no extra precision, so that mode uses the first
`N_SINGLE` = 1000 eval queries. The row records `n_queries` so the sample is visible in
`results.csv` rather than hidden.
"""),

code(r"""
N_SINGLE = 1000
REPEATS = 3
timings = {}

for name in DATASETS:
    d = DATA[name]
    index, base, query = d["index"], d["base"], d["query"]
    n, dim = base.shape
    qe = np.ascontiguousarray(query[C.EVAL])
    gte = d["gt_dists"][C.EVAL]
    timings[name] = {}

    print(f"\n[{name}] n={n:,} d={dim}")
    C.set_single_thread()

    # recall is identical in all three modes (same exact index); measure it once
    _, I = index.search(qe[:N_SINGLE], 10)
    rec_s = C.recall_from_ids(I, base, qe[:N_SINGLE], gte[:N_SINGLE])

    t = C.bench_single_query(lambda x: index.search(x, 10), qe[:N_SINGLE],
                             repeats=REPEATS, warmup=20)
    timings[name]["one_at_a_time"] = t
    print(f"  one_at_a_time (1 thread, {N_SINGLE} q): {t['qps_median']:8.2f} qps · "
          f"p50 {t['lat_p50_ms']:7.2f} ms · p95 {t['lat_p95_ms']:7.2f} ms · "
          f"spread {t['qps_spread_pct']}%")
    LOG.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
            index_kind="exact_flat", build_params={}, search_param_name="query_mode",
            search_param_value="one_at_a_time", recall=rec_s, timing=t,
            build_time_s=d["add_s"], add_time_s=d["add_s"],
            index_size_bytes=d["index_size"], threads=1,
            notes=f"IndexFlatL2; first {N_SINGLE} eval queries")

    tb = C.bench_batch(lambda x: index.search(x, 10), qe, repeats=REPEATS, warmup=1)
    timings[name]["batch"] = tb
    print(f"  batch         (1 thread, {len(qe)} q): {tb['qps_median']:8.2f} qps · "
          f"wall {tb['wall_s_median']:.2f} s")
    rec_b = C.recall_from_ids(index.search(qe, 10)[1], base, qe, gte)
    LOG.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
            index_kind="exact_flat", build_params={}, search_param_name="query_mode",
            search_param_value="batch", recall=rec_b, timing=tb,
            build_time_s=d["add_s"], add_time_s=d["add_s"],
            index_size_bytes=d["index_size"], threads=1,
            notes="IndexFlatL2; all 5000 eval queries in one call")

    nt = ENV["cpu_logical"]
    faiss.omp_set_num_threads(nt)
    tb4 = C.bench_batch(lambda x: index.search(x, 10), qe, repeats=REPEATS, warmup=1)
    faiss.omp_set_num_threads(1)
    timings[name]["batch_threads"] = tb4
    print(f"  batch         ({nt} threads, {len(qe)} q): {tb4['qps_median']:8.2f} qps · "
          f"wall {tb4['wall_s_median']:.2f} s")
    LOG.add(part="1", dataset=name, n_vectors=n, dim=dim, metric="l2",
            index_kind="exact_flat", build_params={}, search_param_name="query_mode",
            search_param_value=f"batch_{nt}threads", recall=rec_b, timing=tb4,
            build_time_s=d["add_s"], add_time_s=d["add_s"],
            index_size_bytes=d["index_size"], threads=nt,
            notes="context only; not comparable with the single-thread ANN numbers")
    del I
    gc.collect()

LOG.flush()
"""),

md("## 6 · The gap, quantified"),

code(r"""
print(f"{'dataset':8s}{'1q/1thr':>12s}{'batch/1thr':>12s}{'speedup':>10s}"
      f"{'batch/Nthr':>12s}{'extra':>8s}{'bytes/vec':>11s}")
print("-" * 73)
for name in DATASETS:
    t = timings[name]
    a = t["one_at_a_time"]["qps_median"]
    b = t["batch"]["qps_median"]
    c = t["batch_threads"]["qps_median"]
    bpv = agreement[name]["index_size"] / agreement[name]["n"]
    print(f"{name:8s}{a:12.2f}{b:12.2f}{b / a:9.1f}x{c:12.2f}{c / b:7.1f}x{bpv:11.0f}")

n  = agreement[DATASETS[0]]["n"]
nq = len(DATA[DATASETS[0]]["query"][C.EVAL])
nt = ENV["cpu_logical"]

print(f'''
Why the two differ so much
--------------------------
Both modes do the same arithmetic -- every query is compared against every base vector. What
changes is how many times the base matrix crosses the memory bus, and how the work is issued.

One query at a time, FAISS computes a matrix-VECTOR product. All {n:,} base vectors
({C.human(DATA[DATASETS[0]]["index_size"])} on SIFT) are streamed from RAM, each used for a
single multiply-accumulate against the one query, then evicted before the next query starts.
The kernel is bound by memory bandwidth rather than by the CPU, and the fixed per-call cost --
a Python call, argument validation, a k=10 result heap built and torn down -- is paid once per
query.

Batched, it becomes a matrix-MATRIX product that FAISS hands to BLAS. A tile of base vectors
is loaded into L2/L3 once and reused against every one of the {nq:,} queries before being
evicted, so each byte fetched from RAM does thousands of times more arithmetic. That converts
a bandwidth-bound kernel into a compute-bound one which can actually saturate the SIMD units,
and it amortises the per-call overhead across the whole batch. The measured speedup above is
therefore a memory-traffic result, not a parallelism one -- both rows use one thread.

The {nt}-thread row is context only. It is not comparable with the single-thread ANN
measurements in Parts 2-4, and the report says so.
''')
"""),

md("## 7 · Summary — paste this back"),

code(r"""
print("=" * 78); print("nb01 SUMMARY (Part 1)"); print("=" * 78)
print(f"machine_id {ENV['machine_id']} · cpu {ENV['cpu_model']} ({ENV['cpu_logical']} logical)"
      f" · faiss {ENV['faiss']}")
print(f"recall unit tests: 12 passed · results.csv now holds {len(LOG.df)} rows\n")

for name in DATASETS:
    a, t = agreement[name], timings[name]
    print(f"[{name}] n={a['n']:,} d={a['dim']} · IndexFlatL2 {C.human(a['index_size'])} "
          f"({a['index_size'] / a['n']:.0f} B/vec, add {a['add_s']:.1f}s)")
    print(f"   exact vs our GT      tie-aware {a['exact_tie_aware']:.6f} | "
          f"naive {a['exact_naive']:.6f}")
    print(f"   file's GT vs ours    tie-aware {a['file_tie_aware']:.6f} | "
          f"naive {a['file_naive']:.6f} | positional {a['file_positional']:.4f}")
    print(f"   ties                 d10==d11 {a['boundary_tie'] * 100:.2f}% | "
          f"any tie in top-11 {a['any_tie'] * 100:.2f}%")
    print(f"   qps  1q/1thr {t['one_at_a_time']['qps_median']:.2f}  "
          f"(p50 {t['one_at_a_time']['lat_p50_ms']:.2f}ms, "
          f"p95 {t['one_at_a_time']['lat_p95_ms']:.2f}ms, "
          f"spread {t['one_at_a_time']['qps_spread_pct']}%)")
    print(f"   qps  batch/1thr {t['batch']['qps_median']:.2f} "
          f"({t['batch']['qps_median'] / t['one_at_a_time']['qps_median']:.1f}x)  |  "
          f"batch/{ENV['cpu_logical']}thr {t['batch_threads']['qps_median']:.2f}")
print("=" * 78)
"""),

md("""
### Next

`nb02` benchmarks HNSW, IVF-Flat and IVF-PQ on SIFT. The one-query-at-a-time exact QPS above
is the baseline those recall-vs-QPS curves are plotted against, and its latency tells us how
much sweep we can afford per session.

Paste the summary back — the exact QPS and the measured per-query latency set the nb02 grid.
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
