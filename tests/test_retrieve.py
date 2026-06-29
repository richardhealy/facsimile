"""Tests for M2: Retriever — text query → top-k page retrieval via Qdrant MaxSim."""

from __future__ import annotations

import numpy as np
import pytest

from facsimile.index.store import ensure_collection, get_memory_client, upsert_pages, PageRecord
from facsimile.ingest.embed import StubEmbedder
from facsimile.retrieve import RetrievedPage, Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_and_embedder():
    """Return an in-memory Qdrant client pre-loaded with three page records."""
    embedder = StubEmbedder(n_patches=8, dim=128, seed=99)
    client = get_memory_client()
    ensure_collection(client)

    # Craft three pages: page 2 uses the same RNG state as the query so its
    # embedding will be strongly correlated with what embed_queries returns.
    rng = np.random.default_rng(99)

    # Page 1 — random noise
    page1_vecs = rng.standard_normal((8, 128)).tolist()
    # Page 2 — strongly correlated with the query (nearby in vector space)
    base = rng.standard_normal((8, 128)).astype(np.float32)
    page2_vecs = (base + rng.standard_normal((8, 128)).astype(np.float32) * 0.01).tolist()
    # Page 3 — random noise
    page3_vecs = rng.standard_normal((8, 128)).tolist()

    records = [
        PageRecord(doc_id="test_doc", page_num=1, doc_path="/doc.pdf", vectors=page1_vecs),
        PageRecord(doc_id="test_doc", page_num=2, doc_path="/doc.pdf", vectors=page2_vecs),
        PageRecord(doc_id="test_doc", page_num=3, doc_path="/doc.pdf", vectors=page3_vecs),
    ]
    upsert_pages(client, records)
    return client, embedder


# ---------------------------------------------------------------------------
# embed_queries tests
# ---------------------------------------------------------------------------

def test_embed_queries_returns_list_per_text():
    embedder = StubEmbedder(n_patches=8, dim=128)
    result = embedder.embed_queries(["hello", "world"])
    assert len(result) == 2


def test_embed_queries_shape():
    embedder = StubEmbedder(n_patches=8, dim=128)
    result = embedder.embed_queries(["what is revenue?"])
    assert len(result) == 1
    assert len(result[0]) == 8       # n_patches
    assert len(result[0][0]) == 128  # dim


def test_embed_queries_returns_floats():
    embedder = StubEmbedder(n_patches=4, dim=32)
    result = embedder.embed_queries(["test query"])
    assert isinstance(result[0][0][0], float)


def test_embed_queries_different_texts_differ():
    embedder = StubEmbedder(n_patches=8, dim=128, seed=0)
    r1 = embedder.embed_queries(["query one"])
    r2 = embedder.embed_queries(["query two"])
    arr1 = np.array(r1[0])
    arr2 = np.array(r2[0])
    assert not np.allclose(arr1, arr2)


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

def test_retriever_returns_retrieved_page_objects(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("some query", top_k=2)
    assert all(isinstance(r, RetrievedPage) for r in results)


def test_retriever_respects_top_k(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("some query", top_k=2)
    assert len(results) <= 2


def test_retriever_returns_all_when_top_k_exceeds_pages(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("some query", top_k=10)
    assert len(results) == 3  # only 3 pages were inserted


def test_retriever_results_sorted_descending(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("some query", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_payload_fields_present(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("test query", top_k=1)
    assert len(results) == 1
    page = results[0]
    assert page.doc_id == "test_doc"
    assert page.doc_path == "/doc.pdf"
    assert page.page_num in {1, 2, 3}
    assert isinstance(page.score, float)


def test_retriever_score_is_positive(client_and_embedder):
    client, embedder = client_and_embedder
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("some query", top_k=3)
    # MaxSim with cosine similarity on realistic vectors produces positive scores
    assert all(r.score > 0 for r in results)


def test_retriever_uses_custom_collection():
    """Retriever should query the collection it was initialised with."""
    embedder = StubEmbedder(n_patches=4, dim=128, seed=0)
    client = get_memory_client()
    custom_col = "custom_col"
    ensure_collection(client, collection_name=custom_col)
    upsert_pages(
        client,
        [PageRecord(doc_id="d", page_num=1, doc_path="/f.pdf",
                    vectors=np.random.randn(4, 128).tolist())],
        collection_name=custom_col,
    )
    retriever = Retriever(client, embedder, collection_name=custom_col)
    results = retriever.retrieve("query", top_k=1)
    assert len(results) == 1
