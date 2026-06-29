"""Tests for late-interaction MaxSim scoring."""

import numpy as np

from facsimile.retrieve.scorer import max_sim_score, rank_pages


def test_identical_vectors_give_nq_score() -> None:
    vecs = np.random.randn(10, 128).astype(np.float32)
    score = max_sim_score(vecs, vecs)
    assert abs(score - 10.0) < 1e-4


def test_orthogonal_vectors_give_zero() -> None:
    q = np.zeros((2, 4), dtype=np.float32)
    d = np.zeros((2, 4), dtype=np.float32)
    q[0, 0] = 1.0
    q[1, 1] = 1.0
    d[0, 2] = 1.0
    d[1, 3] = 1.0
    score = max_sim_score(q, d)
    assert abs(score) < 1e-6


def test_relevant_doc_scores_higher_than_random() -> None:
    rng = np.random.default_rng(42)
    q = rng.standard_normal((5, 64)).astype(np.float32)
    d_relevant = q + rng.standard_normal((5, 64)).astype(np.float32) * 0.1
    d_random = rng.standard_normal((5, 64)).astype(np.float32)
    assert max_sim_score(q, d_relevant) > max_sim_score(q, d_random)


def test_returns_float() -> None:
    q = np.random.randn(3, 16).astype(np.float32)
    d = np.random.randn(3, 16).astype(np.float32)
    assert isinstance(max_sim_score(q, d), float)


def test_score_is_non_negative_for_similar_vectors() -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, 32)).astype(np.float32)
    # positively correlated doc
    d = q + rng.standard_normal((4, 32)).astype(np.float32) * 0.05
    assert max_sim_score(q, d) > 0


def test_rank_pages_returns_sorted_descending() -> None:
    rng = np.random.default_rng(7)
    q = rng.standard_normal((4, 32)).astype(np.float32)
    page_vecs = {
        1: rng.standard_normal((4, 32)).astype(np.float32),
        2: q + rng.standard_normal((4, 32)).astype(np.float32) * 0.01,  # very relevant
        3: rng.standard_normal((4, 32)).astype(np.float32),
    }
    ranked = rank_pages(q, page_vecs, top_k=3)
    assert ranked[0][0] == 2  # most relevant page first
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_pages_respects_top_k() -> None:
    rng = np.random.default_rng(1)
    q = rng.standard_normal((3, 16)).astype(np.float32)
    page_vecs = {i: rng.standard_normal((3, 16)).astype(np.float32) for i in range(10)}
    ranked = rank_pages(q, page_vecs, top_k=3)
    assert len(ranked) == 3
