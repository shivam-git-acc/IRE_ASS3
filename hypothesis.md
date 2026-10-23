# Hypotheses

Written and committed **before** running the Part 3 and Part 4 experiments, as the assignment
requires. Predictions are stated so they can be shown wrong; where I am genuinely unsure I say
so rather than hedging into unfalsifiability. A wrong prediction stays in this file and gets
discussed in the report.

Everything below is grounded in what Parts 1 and 2 already measured on this machine
(Intel Xeon @ 2.00 GHz, 4 logical cores, 31.3 GB RAM, FAISS 1.15.1, single-thread timing):

| measurement | SIFT-1M | GloVe-100 (1.18M) |
|---|---|---|
| exact search, 1 query at a time | 22.63 qps | 23.25 qps |
| relative contrast RC@10 | 2.63 | 1.51 |
| best HNSW @ recall 0.95 | 5817 qps (`M=16`) | 584 qps (`M=32`) |
| best IVF-Flat @ recall 0.95 | 1534 qps (`nlist=4096`) | 300 qps (`nlist=16384`) |
| IVF-PQ peak recall@10 | 0.886 (`m=64`) | 0.874 (`m=50`) |

---

## Part 3 — Scaling on SIFT (100K / 300K / 1M)

**Experiment.** Rebuild HNSW, IVF-Flat and exact search on nested random subsets of SIFT at
100K, 300K and 1M vectors, and measure the cost of reaching recall@10 = 0.95 at each size.
Subsets are nested (100K ⊂ 300K ⊂ 1M) and drawn with seed 20261023, so any difference is
attributable to size rather than to having sampled a different region. The cheapest setting
reaching 0.95 is chosen on the **tune** query split; the reported numbers come from the
**eval** split.

### H3.1 — Exact search scales as n^(-1.0)

A linear scan reads every vector once, so QPS should be inversely proportional to n with no
sub-linear relief. Concretely: **~226 qps at 100K and ~75 qps at 300K**, extrapolating from the
measured 22.63 qps at 1M, within ±15%.

*Falsified if* the fitted exponent is outside [-1.15, -0.85]. The most likely reason for that
would be cache effects: at 100K the base is 51 MB and may partly sit in L3, which would make
small-n exact search *faster* than the n^(-1) line predicts. I expect this to be a small
effect, but it is the one I would check first.

### H3.2 — HNSW degrades far more slowly, with exponent around -0.2 to -0.4

Greedy descent through a navigable small-world graph visits O(log n) nodes, but holding recall
fixed at 0.95 requires `efSearch` to grow with n as well, so the observed exponent will be
worse than pure log-n would suggest and still much shallower than exact's -1.0.

Concretely: **HNSW at 100K is less than 4× faster than HNSW at 1M** (against exact's 10×).

*Falsified if* HNSW's QPS ratio from 100K to 1M exceeds 4×, or if the fitted exponent is
steeper than -0.5.

### H3.3 — IVF-Flat sits between the two, around -0.5 to -0.8

IVF must scan a roughly constant *fraction* of the data to reach a fixed recall, and that
fraction times n grows with n; `nlist` tuning recovers some of it but not all. I expect an
exponent clearly steeper than HNSW's and clearly shallower than exact's.

*Falsified if* IVF-Flat's exponent is not strictly between HNSW's and exact's.

This is the prediction I hold most loosely. IVF's scaling depends on how `nlist` is chosen per
size, and with `nlist ∝ √n` the theory argues for something closer to -0.5 while a fixed
`nlist` argues for -1.0. I am sweeping `nlist` per size rather than fixing it, which should
land nearer -0.5.

### H3.4 — HNSW's advantage over IVF-Flat shrinks as n shrinks

At 1M on SIFT, HNSW beat IVF-Flat by 3.8× at recall 0.95. A graph's logarithmic advantage
needs scale to pay for itself, so at 100K the gap should be **smaller than 2.5×**.

*Falsified if* the gap at 100K is ≥ 2.5×, or widens as n decreases.

### H3.5 — HNSW build cost per vector rises with n; IVF-Flat's stays roughly flat

Graph construction is O(n log n), so seconds-per-vector should increase measurably from 100K to
1M. IVF's cost is dominated by k-means over a subsample capped at `256 × nlist` points, so its
per-vector cost should be much flatter.

*Falsified if* HNSW's build seconds-per-vector does not increase by at least 20% from 100K to
1M.

### H3.6 — ANN remains worth it at every size tested

Even at 100K, where exact search should reach ~226 qps, HNSW should still be more than an order
of magnitude faster. The honest cost of ANN at small n is therefore **build time and memory,
not query throughput** — which is the framing the report's question 4 needs.

*Falsified if* HNSW's advantage over exact at 100K is under 10×.

---

## Part 4 — Hard geometry: distance concentration (synthetic)

**Experiment.** Generate synthetic data where the only thing that changes is the geometry, and
watch HNSW and IVF-Flat break. A 2 × 4 design, all at n = 200,000 vectors with a fixed seed:

| | d = 16 | d = 64 | d = 256 | d = 1024 |
|---|---|---|---|---|
| **uniform** in [0,1]^d | | | | |
| **clustered** (100 Gaussian clusters, σ = 0.05) | | | | |

The uniform row varies the property under study (distance concentration, which rises with d).
The clustered row is the **control**: same n, same d, same search budget, same generator — only
the geometry differs. If degradation tracks the uniform row and not the clustered row, the
cause is contrast, not dimension or arithmetic cost.

Generator: `gen_synthetic.py`, seed 20261023, committed with this file.

### Separating geometry from arithmetic

A distance computation costs O(d), so QPS falls with d for reasons that have nothing to do with
search difficulty. To avoid mistaking arithmetic for geometry, the primary measurements are:

1. **recall@10 at a fixed candidate budget** (`efSearch = 64`, `nprobe = 16`), and
2. **the number of distance computations** FAISS actually performs
   (`faiss.cvar.hnsw_stats.ndis`, `faiss.cvar.indexIVF_stats.ndis`), reported as a *fraction of
   the dataset scanned*.

QPS is reported too, but the mechanism argument rests on (1) and (2), which are
dimension-independent units.

### H4.1 — Relative contrast collapses toward 1 in the uniform row *(design check, already measured)*

**This one is not a blind prediction and is not scored as one.** Relative contrast is a property
of the generator, not of any index, so it was measured while designing the manipulation — you
want to confirm that the knob moves the thing it is supposed to move before spending a session
on it. Measured at n = 200,000 over 500 queries by GPU brute force:

| d | uniform RC@10 | clustered RC@10 | ratio |
|---|---|---|---|
| 16 | 2.347 | 9.774 | 4.2x |
| 64 | 1.393 | 7.304 | 5.2x |
| 256 | 1.159 | 6.501 | 5.6x |
| 1024 | **1.073** | 6.148 | 5.7x |

The manipulation works: uniform contrast collapses monotonically toward 1 while the clustered
control holds above 6 at every d. For scale, SIFT measured 2.63 and GloVe 1.51 — so the uniform
row spans from roughly SIFT's difficulty at d = 16 to considerably harder than GloVe at
d = 1024, and the clustered row is easier than either real dataset throughout.

Everything from H4.2 onward concerns **index behaviour** and is a genuine prediction made
before running anything.

### H4.2 — Recall at a fixed budget collapses with d for uniform data

At `efSearch = 64` / `nprobe = 16`, recall@10 should fall monotonically along the uniform row,
and at d = 1024 **both indexes should be under 0.30**.

*Falsified if* either index holds recall above 0.30 at uniform d = 1024, or if recall is
non-monotonic in d by more than measurement noise (±0.01).

### H4.3 — The clustered control holds up at the same d — this is the key result

At d = 256 and a budget where uniform recall is below 0.60, the clustered dataset should
exceed **0.90**. Same dimension, same vector count, same index settings, same arithmetic cost
per distance — so any difference is geometric.

*Falsified if* the clustered control degrades along with the uniform row. That would mean
dimension itself, not contrast, is what breaks these indexes, and my whole mechanism story
would be wrong.

### H4.4 — Reaching 0.95 forces both indexes to degenerate into near-exhaustive scans

On uniform d = 1024, I predict neither index reaches recall 0.95 while scanning less than
**50% of the dataset**, i.e. both collapse to something barely better than a linear scan, and
their QPS advantage over exact search at that recall drops below 2×.

*Falsified if* either index reaches 0.95 while scanning under 50%.

### H4.5 — HNSW degrades at least as fast as IVF-Flat (held loosely)

Greedy graph descent needs a usable distance gradient to follow, and concentration removes it,
so I expect HNSW to lose more than IVF-Flat does. But IVF's k-means cells also become
uninformative when contrast vanishes, and I can argue it either way.

**I am genuinely unsure about this one and expect it is the prediction most likely to be
wrong.** I am recording it because an honest prior beats a retrofitted explanation.

*Falsified if* IVF-Flat's recall at fixed budget falls faster than HNSW's across the uniform row.

---

## What these experiments do not test

Stated up front so the report's limitations are not written with hindsight:

* Part 3 varies only n on SIFT. It says nothing about how scaling interacts with the harder
  GloVe geometry, where Part 2 already showed the index ranking compresses.
* Part 4 varies ambient dimension with iid coordinates, so intrinsic and ambient dimension move
  together. It does not separate them — low intrinsic dimension embedded in high ambient
  dimension (the realistic case for learned embeddings) is untested.
* Neither experiment covers IVF-PQ, whose recall ceiling Part 2 established separately.
* All timings come from one shared cloud VM. Run-to-run spread was 1.8–3.1% in Part 1, which is
  small relative to the effects predicted here, but it is not a quiet dedicated machine.
