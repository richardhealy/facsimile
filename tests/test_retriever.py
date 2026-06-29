"""Tests for M2 retrieval: text query → late-interaction page ranking."""

from __future__ import annotations

import numpy as np
import pytest

from facsimile.index.store import (
    COLLECTION_NAME,
    PageRecord,
    ensure_collection,
    get_memory_client,
    upsert_pages,
)
from facsimile.retrieve.retriever import (
    ColPaliTextEmbedder,
    RetrievalResult,
    Retriever,
    StubTextEmbedder,
    TextQueryEmbedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indexed_store(
    n_pages: int = 5,
    n_patches: int = 8,
    dim: int = 128,
    doc_id: str = "test_doc",
    seed: int = 99,
):
    """Return (client, page_vecs_dict) with n_pages random pages indexed."""
    client = get_memory_client()
    ensure_collection(client, vector_dim=dim)
    rng = np.random.default_rng(seed)
    records = []
    page_vecs: dict[int, list[list[float]]] = {}
    for i in range(n_pages):
        vecs = rng.standard_normal((n_patches, dim)).tolist()
        page_vecs[i + 1] = vecs
        records.append(
            PageRecord(
                doc_id=doc_id,
                page_num=i + 1,
                doc_path="/tmp/test.pdf",
                vectors=vecs,
            )
        )
    upsert_pages(client, records)
    return client, page_vecs


# ---------------------------------------------------------------------------
# StubTextEmbedder
# ---------------------------------------------------------------------------


def test_stub_embedder_shape() -> None:
    emb = StubTextEmbedder(n_tokens=8, dim=128)
    vecs = emb.embed_query("What is the revenue in Q3?")
    assert len(vecs) == 8
    assert len(vecs[0]) == 128


def test_stub_embedder_is_deterministic() -> None:
    emb = StubTextEmbedder()
    query = "total assets"
    assert emb.embed_query(query) == emb.embed_query(query)


def test_stub_embedder_differs_per_query_text() -> None:
    emb = StubTextEmbedder()
    a = emb.embed_query("question one")
    b = emb.embed_query("question two")
    assert a != b


def test_stub_embedder_satisfies_protocol() -> None:
    assert isinstance(StubTextEmbedder(), TextQueryEmbedder)


def test_stub_embedder_vectors_are_floats() -> None:
    emb = StubTextEmbedder(n_tokens=4, dim=16)
    vecs = emb.embed_query("test")
    assert all(isinstance(v, float) for row in vecs for v in row)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def test_retriever_returns_results() -> None:
    client, _ = _make_indexed_store(n_pages=5)
    retriever = Retriever(client, StubTextEmbedder())
    results = retriever.retrieve("What is the operating margin?", top_k=3)
    assert len(results) == 3


def test_retriever_results_are_retrieval_result_instances() -> None:
    client, _ = _make_indexed_store(n_pages=5)
    retriever = Retriever(client, StubTextEmbedder())
    results = retriever.retrieve("revenue breakdown")
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_retriever_result_fields_populated() -> None:
    client, _ = _make_indexed_store(n_pages=5)
    retriever = Retriever(client, StubTextEmbedder())
    results = retriever.retrieve("chart analysis", top_k=5)
    for r in results:
        assert isinstance(r.doc_id, str)
        assert isinstance(r.page_num, int)
        assert 1 <= r.page_num <= 5
        assert isinstance(r.doc_path, str)
        assert isinstance(r.score, float)


def test_retriever_respects_top_k() -> None:
    client, _ = _make_indexed_store(n_pages=10)
    retriever = Retriever(client, StubTextEmbedder())
    assert len(retriever.retrieve("query", top_k=4)) == 4
    assert len(retriever.retrieve("query", top_k=1)) == 1


def test_retriever_scores_descending() -> None:
    client, _ = _make_indexed_store(n_pages=5)
    retriever = Retriever(client, StubTextEmbedder())
    results = retriever.retrieve("operating expenses", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_doc_id_matches_ingested() -> None:
    client, _ = _make_indexed_store(n_pages=3, doc_id="annual_report_2023")
    retriever = Retriever(client, StubTextEmbedder())
    results = retriever.retrieve("net profit", top_k=3)
    assert all(r.doc_id == "annual_report_2023" for r in results)


def test_retriever_relevant_page_ranks_first() -> None:
    """A page whose patch vectors align with the query should rank first."""
    client = get_memory_client()
    ensure_collection(client)
    rng = np.random.default_rng(7)

    # Create a query-aligned page (page 3) and noise pages
    query_vecs = rng.standard_normal((8, 128)).astype(np.float32)
    records = []
    for page_num in range(1, 6):
        if page_num == 3:
            # Near-copy of query vectors — should score highest
            page_patches = (query_vecs + rng.standard_normal((8, 128)).astype(np.float32) * 0.01).tolist()
        else:
            page_patches = rng.standard_normal((8, 128)).tolist()
        records.append(
            PageRecord(
                doc_id="doc",
                page_num=page_num,
                doc_path="/tmp/doc.pdf",
                vectors=page_patches,
            )
        )
    upsert_pages(client, records)

    # Embedder that returns the known query vectors regardless of text
    class FixedEmbedder:
        def embed_query(self, text: str) -> list[list[float]]:
            return query_vecs.tolist()

    retriever = Retriever(client, FixedEmbedder())
    results = retriever.retrieve("any question", top_k=5)
    assert results[0].page_num == 3


def test_retriever_custom_collection() -> None:
    client = get_memory_client()
    ensure_collection(client, collection_name="custom_col")
    records = [
        PageRecord(doc_id="d", page_num=1, doc_path="/f.pdf", vectors=[[0.1] * 128])
    ]
    upsert_pages(client, records, collection_name="custom_col")
    retriever = Retriever(client, StubTextEmbedder(), collection_name="custom_col")
    results = retriever.retrieve("test", top_k=1)
    assert len(results) == 1


def test_colpali_text_embedder_raises_without_deps() -> None:
    """ColPaliTextEmbedder raises ImportError when colpali-engine is absent."""
    with pytest.raises(ImportError, match="colpali-engine"):
        ColPaliTextEmbedder()
