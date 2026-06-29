"""Late-interaction MaxSim scoring (ColPali / ColQwen style)."""

import numpy as np


def max_sim_score(
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
) -> float:
    """
    Compute the MaxSim late-interaction score.

    For each query token, find the maximum cosine similarity to any doc patch,
    then sum those maxima.  This is the ColBERT / ColPali scoring function that
    allows patch-level alignment without a full cross-encoder.

    query_vectors: (Nq, D) float array of query token embeddings.
    doc_vectors:   (Nd, D) float array of document patch embeddings.
    Returns a scalar score (higher = more relevant).
    """
    q = np.asarray(query_vectors, dtype=np.float32)
    d = np.asarray(doc_vectors, dtype=np.float32)

    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    d_norm = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)

    # (Nq, Nd) cosine similarity matrix
    sim = q_norm @ d_norm.T

    # Sum of per-query-token maximums
    return float(sim.max(axis=-1).sum())


def rank_pages(
    query_vectors: np.ndarray,
    page_vectors: dict[int, np.ndarray],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Rank pages by MaxSim score and return the top-k (page_number, score) pairs.

    page_vectors maps page_number → (Nd, D) patch embedding array.
    """
    scored = [
        (page_num, max_sim_score(query_vectors, pvecs))
        for page_num, pvecs in page_vectors.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
