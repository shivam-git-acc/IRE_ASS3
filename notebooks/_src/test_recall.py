"""Unit tests for the tie-aware recall@10 metric (Part 1).

Run: pytest -q test_recall.py      (or: python test_recall.py)
"""

import numpy as np
import pytest

from annbench_core import (K, exact_dists_for_ids, recall_at_k, recall_from_ids)


# ---------------------------------------------------------------------------
# A tiny hand-built dataset where we know every distance exactly.
# 1-D points at 0,1,2,...  query at 0  ->  neighbours are 0,1,2,... in order.
# ---------------------------------------------------------------------------

@pytest.fixture
def line():
    base = np.arange(20, dtype=np.float32).reshape(-1, 1)
    queries = np.zeros((1, 1), dtype=np.float32)
    gt_ids = np.arange(K, dtype=np.int32)[None, :]
    gt_dists = exact_dists_for_ids(base, queries, gt_ids)
    return base, queries, gt_ids, gt_dists


def test_exact_result_is_one(line):
    base, queries, gt_ids, gt_dists = line
    assert recall_from_ids(gt_ids, base, queries, gt_dists)[0] == 1.0


def test_permuted_top10_is_one(line):
    """Recall@10 is a set measure: order within the top 10 must not matter."""
    base, queries, gt_ids, gt_dists = line
    perm = gt_ids[:, ::-1].copy()
    assert recall_from_ids(perm, base, queries, gt_dists)[0] == 1.0


def test_one_genuine_miss_is_point_nine(line):
    base, queries, gt_ids, gt_dists = line
    got = gt_ids.copy()
    got[0, 9] = 15                      # true distance 225 >> d10 = 81
    assert recall_from_ids(got, base, queries, gt_dists)[0] == pytest.approx(0.9)


def test_all_wrong_is_zero(line):
    base, queries, _, gt_dists = line
    got = np.arange(10, 20, dtype=np.int32)[None, :]
    assert recall_from_ids(got, base, queries, gt_dists)[0] == 0.0


def test_tie_at_tenth_neighbour_is_not_a_mistake():
    """THE case the assignment singles out.

    Points at +-1..+-5 give five exact distance ties. The true 10th neighbour is at
    distance 25, and so is the 11th: both +5 and -5 sit there. An index that returns -5
    where the stored list says +5 has made no error, and must still score 1.0.
    """
    base = np.array([[0.], [1.], [-1.], [2.], [-2.], [3.], [-3.], [4.], [-4.],
                     [5.], [-5.]], dtype=np.float32)
    queries = np.zeros((1, 1), dtype=np.float32)
    stored = np.arange(K, dtype=np.int32)[None, :]          # ids 0..9, i.e. +5 not -5
    gt_dists = exact_dists_for_ids(base, queries, stored)
    assert gt_dists[0, K - 1] == 25.0

    swapped = stored.copy()
    swapped[0, 9] = 10                                      # return -5 instead of +5
    assert recall_from_ids(swapped, base, queries, gt_dists)[0] == 1.0, \
        "a tie at the 10th neighbour was wrongly counted as a mistake"

    # And a naive id-set metric would have scored this 0.9 -- confirm the distinction
    # is real rather than vacuous.
    naive = len(set(swapped[0].tolist()) & set(stored[0].tolist())) / K
    assert naive == 0.9


def test_all_distances_identical_is_one():
    """Degenerate but legal: every point is equidistant, so any 10 ids are correct."""
    base = np.ones((50, 3), dtype=np.float32)
    queries = np.zeros((1, 3), dtype=np.float32)
    gt_dists = exact_dists_for_ids(base, queries, np.arange(K, dtype=np.int32)[None, :])
    got = np.arange(40, 50, dtype=np.int32)[None, :]
    assert recall_from_ids(got, base, queries, gt_dists)[0] == 1.0


def test_zero_distance_ground_truth():
    """d10 == 0 (duplicated vectors): the relative slack vanishes, so the absolute
    slack has to carry the comparison. Guards against a divide-by-zero style bug."""
    base = np.zeros((20, 4), dtype=np.float32)
    base[15:] = 7.0
    queries = np.zeros((1, 4), dtype=np.float32)
    gt_dists = exact_dists_for_ids(base, queries, np.arange(K, dtype=np.int32)[None, :])
    assert gt_dists[0, K - 1] == 0.0
    assert recall_from_ids(np.arange(5, 15, dtype=np.int32)[None, :],
                           base, queries, gt_dists)[0] == 1.0
    assert recall_from_ids(np.arange(10, 20, dtype=np.int32)[None, :],
                           base, queries, gt_dists)[0] == pytest.approx(0.5)


def test_duplicate_ids_are_credited_once(line):
    """An index returning the same vector ten times must score 0.1, not 1.0."""
    base, queries, _, gt_dists = line
    got = np.zeros((1, K), dtype=np.int32)
    assert recall_from_ids(got, base, queries, gt_dists)[0] == pytest.approx(0.1)


def test_missing_results_are_not_hits(line):
    """FAISS pads with -1 when an index finds fewer than k candidates."""
    base, queries, gt_ids, gt_dists = line
    got = gt_ids.copy()
    got[0, 5:] = -1
    assert recall_from_ids(got, base, queries, gt_dists)[0] == pytest.approx(0.5)


def test_pq_style_approximate_distances_do_not_corrupt_recall(line):
    """IVF-PQ returns approximate distances. recall_from_ids must ignore whatever the
    index reports and recompute from the base vectors."""
    base, queries, gt_ids, gt_dists = line
    lying = np.zeros((1, K), dtype=np.float64)              # index claims every hit is perfect
    wrong_ids = np.arange(10, 20, dtype=np.int32)[None, :]  # ...but they are all wrong
    assert recall_at_k(wrong_ids, lying, gt_dists)[0] == 1.0          # believing the lie
    assert recall_from_ids(wrong_ids, base, queries, gt_dists)[0] == 0.0   # recomputing


def test_matches_bruteforce_on_random_data():
    """Cross-check against an independent, deliberately naive implementation on random
    float data, where exact ties essentially never occur so the two must agree."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((400, 8), dtype=np.float32)
    queries = rng.standard_normal((25, 8), dtype=np.float32)

    full = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1).astype(np.float64)
    order = np.argsort(full, axis=1)
    gt_ids = order[:, :K].astype(np.int32)
    gt_dists = np.take_along_axis(full, order, axis=1)[:, :K]

    got = order[:, rng.permutation(30)[:K]].astype(np.int32)   # a sloppy "index"
    mine = recall_from_ids(got, base, queries, gt_dists)
    naive = np.array([len(set(g.tolist()) & set(t.tolist())) / K
                      for g, t in zip(got, gt_ids)])
    assert np.allclose(mine, naive)


def test_batch_shape_and_mean(line):
    base, queries, gt_ids, gt_dists = line
    q = np.repeat(queries, 4, axis=0)
    gtd = np.repeat(gt_dists, 4, axis=0)
    got = np.repeat(gt_ids, 4, axis=0).copy()
    got[2, 9] = 19
    r = recall_from_ids(got, base, q, gtd)
    assert r.shape == (4,)
    assert r.mean() == pytest.approx((1 + 1 + 0.9 + 1) / 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
