# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction — Assignment 3.

Benchmarks **HNSW**, **IVF-Flat** and **IVF-PQ** against an exact FAISS baseline on **SIFT-1M**
and **GloVe-100**, plus two experiments: how the cost of recall 0.95 scales with dataset size
(Part 3), and what distance concentration does to both index families (Part 4).

---

## Running it

```bash
pip install -r requirements.txt

bash run_all.sh --quick     # ~10 min: SIFT only, first 100K vectors
bash run_all.sh             # full run: both datasets, all five parts
```

Both commands regenerate every number, figure and table in the report from the data files.
Output lands in `out/` (results, figures, logs) and downloaded data in `scratch/`; neither is
committed.

**No Hugging Face account is needed.** If `HF_TOKEN` is set, results are additionally mirrored
to a HF dataset repo (how the original runs survived across Kaggle sessions); without it that
step is silently skipped and everything stays local. Set `A3_NO_HF=1` to force local-only.

**No GPU is needed.** A GPU is used to compute brute-force ground truth when one is present;
otherwise the identical ground truth is computed through FAISS on the CPU. Either way the
returned distances are recomputed in float64, so the recall threshold is exact.

### What each mode does

| | `--quick` | full |
|---|---|---|
| datasets | SIFT, first 100,000 vectors | SIFT-1M + GloVe-100 (1.18M) |
| queries | 1,000 (500 tune / 500 eval) | 10,000 (5,000 / 5,000) |
| Part 2 grid | 1 HNSW, 2 IVF-Flat, 2 IVF-PQ settings | 3 / 3 / 4 settings |
| Part 3 sizes | 25K / 50K / 100K | 100K / 300K / 1M |
| Part 4 | 2 × 3 design, n = 20,000 | 2 × 4 design, n = 200,000 |
| repeats per timing | 1 | 3 (median reported) |
| runtime | ≈ 10 min + ~1 min download | ≈ 9 h |

Quick mode exercises every code path end to end — it is a smoke test of the pipeline, not a
small version of the results. Its absolute QPS numbers are not comparable with the report's,
which came from the full run on the machine recorded in `out/results/results.csv`.

### Runtime expectations for the full run

Roughly: data preparation 40 min (mostly downloading ~1 GB), Part 1 25 min, Part 2 on SIFT
1.5 h, Part 2 on GloVe 3 h, Part 3 1 h, Part 4 1 h, aggregation 2 min. The GloVe sweeps
dominate. Every stage appends to `out/results/results.csv` as it goes and skips rows already
present, so an interrupted run resumes rather than restarting.

---

## Measurement protocol

These are the choices the numbers depend on, so they are stated rather than buried:

* **Query timing is single-thread, one query at a time**, `k = 10`, one warm-up pass, three
  timed repeats, **median** reported, `time.perf_counter_ns`. FAISS is pinned with
  `omp_set_num_threads(1)` and the pin is asserted, not assumed.
* **Recall is measured in a separate multi-threaded batched pass.** Recall does not depend on
  thread count, and measuring it single-threaded would have multiplied the runtime for an
  identical answer.
* **Builds use all cores**, with the thread count recorded in every row. Build times are
  comparable with each other but not with a single-threaded world.
* **Queries 0–4999 are the tune half; 5000–9999 are the eval half.** Every reported number
  comes from the eval half. In Part 3 the search parameter is *chosen* on the tune half and
  then timed on eval, so the reported setting never saw the eval queries during selection.
* **GloVe is unit-normalised**, which makes squared-L2 ranking identical to cosine ranking, so
  one metric and one exact baseline cover both datasets.
* **All distances are squared L2 in float32 data, compared in float64.** The neighbour ids
  shipped inside the ann-benchmarks HDF5 files are used only as an independent cross-check on
  the ground truth computed here; their distance *scales* (Euclidean vs cosine) are never used.

### recall@10, and why ties matter

A returned id counts as correct when **its exact distance is ≤ the exact distance of the true
10th neighbour**, not when it appears in a stored top-10 id list. If the true 10th and 11th
neighbours sit at the same distance, returning either is equally right.

This is not pedantry. On SIFT — integer-valued vectors, so exact ties are common — 1.37% of
queries have a tie at exactly `d10 == d11` and 12.03% have one somewhere in the top 11. A naive
id-overlap metric scores the dataset's *own published neighbours* at 0.9994 rather than 1.0,
putting a floor under SIFT recall that no index can cross.

`recall_from_ids` also recomputes the exact distance of every returned id from the base
vectors rather than trusting what the index reports. That matters most for IVF-PQ, whose
returned distances are approximations: on one test index, believing them gave 0.76 recall where
the true figure was 0.37.

Tests in `tests/test_recall.py` cover the exact result, a permuted top-10, a genuine miss, all
wrong, **a tie at rank 10**, all-distances-identical, zero-distance ground truth, duplicate ids,
`-1` padding, PQ-style lying distances, and agreement with an independent naive implementation.

```bash
python -m pytest -q tests/
```

---

## Layout

```
run_all.sh              the one command; drives the stages below in order
prepare_data.py         download, normalise, exact ground truth, nested subsets
part1_exact.py          Part 1: exact baseline, recall metric, 1-at-a-time vs batch timing
part2_benchmark.py      Part 2: --dataset sift | glove
part3_scaling.py        Part 3: cost of recall 0.95 at 100K / 300K / 1M
part4_geometry.py       Part 4: distance concentration, synthetic
make_report_assets.py   figures and tables from results.csv (no benchmarking)
annbench_core.py        recall metric, timing harness, ground truth, results table
gen_synthetic.py        the Part 4 data generator (seeded; `--selftest` checks determinism)
hypothesis.md           Part 3 and Part 4 predictions, committed before those runs
tests/test_recall.py    unit tests for the recall metric
tools/                  regenerates make_report_assets.py from the notebook
build_report.py         renders report.pdf from ../results/ (numbers are read, not typed);
                        stamps page numbers, then checks the PDF for silently clipped content
report_body.html        the report's content template
report_style.css        its print stylesheet
diagrams.py             the report's inline-SVG measurement-protocol diagram
notebooks/              the Kaggle notebooks the experiments were actually run in
```

`results.csv` holds one row per benchmark run — every number behind the report — with the
build parameters, search parameter, recall, median QPS and its spread across repeats, p50/p95
latency, build time, index size, bytes per vector, thread count, query split, and the CPU model
and library versions it was measured on.

### Notebooks

The experiments were actually run on Kaggle (one T4 session per part), and those notebooks are
in `notebooks/`. Each is emitted by its builder in `notebooks/_src/`, which embeds this
directory's `annbench_core.py` verbatim — all seven notebooks carry one identical copy, so a
notebook and the scripts can never disagree about the recall metric or the timing harness. For
the aggregation stage the notebook and `make_report_assets.py` share a body too, generated by
`tools/sync_report_assets.py`; `--check` fails if they have drifted.

### Hypothesis timing — please read the timezones

The assignment requires each hypothesis to be committed **before** that experiment's results.
It was, but the two clocks differ and a direct comparison looks wrong:

* `git log` prints **IST (+0530)** — the hypothesis commit reads `2026-10-23 22:46:42 +0530`.
* `results.csv` timestamps come from the Kaggle container, which runs in **UTC**.

Converted to one clock:

| event | UTC |
|---|---|
| hypothesis commit (`6c5e3da`) | 2026-10-23 **17:16:42** |
| first Part 3 result | 2026-10-23 17:26:38 — 10 min after |
| first Part 4 result | 2026-10-23 19:19:01 — 2 h after |

Part 2 results predate the commit, which is expected: only Parts 3 and 4 have hypotheses.

```bash
git -C code log --date=iso --format='%h %ad %s'   # IST
head -2 results/results.csv                        # UTC
```

### Where results live

`run_all.sh` writes to `out/`, which is gitignored and not part of the submission. The
`results/` directory beside this one holds the **submitted** measurements — the 373 rows
produced by the Kaggle runs described below — and is what the report is built from. Re-running
the pipeline will not overwrite them.

`build_report.py` regenerates `report.pdf` from `../results/`:

```bash
python build_report.py --roll <your roll number>
```

---

## Environment the reported numbers came from

All measurements in the report were taken on a single machine:

```
Intel(R) Xeon(R) CPU @ 2.00GHz, 4 logical cores, 31.3 GB RAM   (Kaggle, GPU T4 x2)
Tesla T4 (ground truth only; no index is ever searched on the GPU)
Python 3.13.15 · FAISS 1.15.1 · NumPy 2.1.3 · h5py 3.16.0
machine_id dc469939
```

This is a shared cloud VM, not a quiet dedicated box. Run-to-run spread on the exact baseline
was 1.8–3.1%, which is small relative to the effects reported, but it is recorded per row
(`qps_spread_pct`) rather than glossed over. `machine_id` is a hash of the CPU model, core
count and memory size; it is identical across every row, so no cross-dataset comparison in the
report spans different hardware.

## Data

From [ann-benchmarks.com](https://ann-benchmarks.com): `sift-128-euclidean.hdf5` (1,000,000 ×
128, Euclidean) and `glove-100-angular.hdf5` (1,183,514 × 100, angular). Downloaded by
`prepare_data.py`; neither the datasets nor any index file is committed.
