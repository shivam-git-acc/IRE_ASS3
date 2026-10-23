"""Seeded generator for the Part 4 experiment (distance concentration).

Committed alongside hypothesis.md, before the experiment runs.

The 2 x 4 design: {uniform, clustered} x {d = 16, 64, 256, 1024}, all at n = 200,000.
The uniform row varies the property under study -- distance concentration, which rises with d.
The clustered row is the control: identical n, d, generator and search budget, differing only
in geometry. If recall tracks the uniform row and not the clustered one, the cause is contrast
rather than dimension or per-distance arithmetic cost.

Determinism: every (kind, d) pair gets its own stream from one root seed, so any single
dataset can be regenerated in isolation and is bit-identical across machines and across
runs. Verify with `python gen_synthetic.py --selftest`.
"""

import argparse

import numpy as np

SEED = 20261023
N_DEFAULT = 200_000          # per dataset; the assignment caps a synthetic set at 500K
NQ_DEFAULT = 2_000
DIMS = (16, 64, 256, 1024)
KINDS = ("uniform", "clustered")

# Clustered-row parameters. 100 clusters of sigma=0.05 inside the unit cube are far enough
# apart that finding the right cluster is easy, which is exactly the high-contrast regime we
# want as a foil to the uniform row.
N_CLUSTERS = 100
CLUSTER_SIGMA = 0.05

_KIND_ID = {k: i for i, k in enumerate(KINDS)}


def _rng(kind, d, role):
    """An independent, reproducible stream per (kind, d, role)."""
    if kind not in _KIND_ID:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return np.random.default_rng([SEED, _KIND_ID[kind], int(d), {"base": 0, "query": 1}[role]])


def generate(kind, d, n=N_DEFAULT, nq=NQ_DEFAULT):
    """Return (base, queries) as C-contiguous float32.

    uniform   -- iid U[0,1] coordinates. As d grows, every pairwise distance converges on the
                 same value, so the nearest neighbour stops being meaningfully nearer than
                 anything else. This is the failure mode under study.
    clustered -- 100 isotropic Gaussian clusters with tight sigma. Queries are drawn from the
                 same mixture, so a query sits inside a cluster whose members are far nearer
                 than any other cluster's. Contrast stays high at every d.
    """
    d = int(d)
    rb, rq = _rng(kind, d, "base"), _rng(kind, d, "query")

    if kind == "uniform":
        base = rb.random((n, d), dtype=np.float32)
        queries = rq.random((nq, d), dtype=np.float32)
        return np.ascontiguousarray(base), np.ascontiguousarray(queries)

    # Cluster centres come from the base stream so that base and queries share them.
    centres = _rng(kind, d, "base").random((N_CLUSTERS, d), dtype=np.float32)

    def draw(rng, count):
        who = rng.integers(0, N_CLUSTERS, size=count)
        noise = rng.standard_normal((count, d), dtype=np.float32) * np.float32(CLUSTER_SIGMA)
        return np.ascontiguousarray(centres[who] + noise)

    # draw() must not consume `centres`' own draws twice, so re-derive the streams here.
    rb2, rq2 = _rng(kind, d, "base"), _rng(kind, d, "query")
    rb2.random((N_CLUSTERS, d), dtype=np.float32)        # skip past the centres
    return draw(rb2, n), draw(rq2, nq)


def describe(kind, d, n=N_DEFAULT):
    return f"{kind}_d{d}_n{n}"


def _selftest():
    """Determinism and basic geometric sanity, on small n so it runs in seconds."""
    n, nq = 4000, 200
    ok = True

    for kind in KINDS:
        for d in (16, 256):
            a1, q1 = generate(kind, d, n, nq)
            a2, q2 = generate(kind, d, n, nq)
            same = np.array_equal(a1, a2) and np.array_equal(q1, q2)
            print(f"  {kind:10s} d={d:<5d} reproducible={same}  base={a1.shape} {a1.dtype}")
            ok &= same and a1.shape == (n, d) and q1.shape == (nq, d)
            ok &= a1.flags["C_CONTIGUOUS"] and a1.dtype == np.float32

    # different (kind, d) must not collide
    u, _ = generate("uniform", 64, n, nq)
    c, _ = generate("clustered", 64, n, nq)
    ok &= not np.array_equal(u, c)

    # contrast: clustered should be far easier than uniform at the same d
    print("\n  relative contrast RC@10 (mean distance to a random point / distance to the 10th NN)")
    for d in (16, 64, 256):
        for kind in KINDS:
            base, q = generate(kind, d, n, nq)
            full = ((q[:, None, :].astype(np.float64)
                     - base[None, :, :].astype(np.float64)) ** 2).sum(-1)
            d10 = np.sqrt(np.sort(full, axis=1)[:, 9])
            rc = np.sqrt(full.mean(1)) / np.maximum(d10, 1e-12)
            print(f"    {kind:10s} d={d:<5d} RC@10 = {rc.mean():6.3f}")

    print(f"\n  selftest {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--out", help="write <out>_base.npy and <out>_query.npy")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())
    if not (a.kind and a.dim):
        ap.error("need --kind and --dim, or --selftest")
    base, q = generate(a.kind, a.dim, a.n)
    print(f"{describe(a.kind, a.dim, a.n)}: base {base.shape}, queries {q.shape}")
    if a.out:
        np.save(f"{a.out}_base.npy", base)
        np.save(f"{a.out}_query.npy", q)
        print(f"wrote {a.out}_base.npy / {a.out}_query.npy")
