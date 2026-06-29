"""Tests for the end-to-end late-interaction Retriever."""

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
from facsimile.ingest.embed import StubEmbedder
from facsimile.retrieve.retriever import RetrievedPage, Retriever


def _make_index(n_pages: int = 5, n_patches: int = 8, dim: int = 128):
    """In-memory Qdrant index pre-loaded with *n_pages* stub records."""
    client = get_memory_client()
    ensure_collection(client, vector_dim=dim)
    rng = np.random.default_rng(0)
    records = [
        PageRecord(
            doc_id="test_doc",
            page_num=i + 1,
            doc_path="/docs/test.pdf",
            vectors=rng.standard_normal((n_patches, dim)).tolist(),
        )
        for i in range(n_pages)
    ]
    upsert_pages(client, records)
    embedder = StubEmbedder(n_patches=n_patches, dim=dim, seed=1)
    return client, embedder


def test_retrieve_returns_retrieved_page_instances():
    client, embedder = _make_index()
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("What is the revenue?", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievedPage) for r in results)


def test_retrieve_respects_top_k():
    client, embedder = _make_index(n_pages=10)
    retriever = Retriever(client, embedder)
    for k in (1, 3, 5):
        results = retriever.retrieve("test query", top_k=k)
        assert len(results) == k


def test_retrieve_scores_descending():
    client, embedder = _make_index()
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("test query", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieved_page_has_correct_fields():
    client, embedder = _make_index()
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("test query", top_k=1)
    assert len(results) == 1
    page = results[0]
    assert page.doc_id == "test_doc"
    assert isinstance(page.page_num, int)
    assert 1 <= page.page_num <= 5
    assert page.doc_path == "/docs/test.pdf"
    assert isinstance(page.score, float)


def test_retrieve_top_k_larger_than_index():
    """Should return as many pages as exist rather than raising."""
    client, embedder = _make_index(n_pages=3)
    retriever = Retriever(client, embedder)
    results = retriever.retrieve("test query", top_k=10)
    assert len(results) == 3


def test_retrieve_uses_collection_name():
    """Retriever should work with a custom collection name."""
    dim = 128
    client = get_memory_client()
    custom_name = "custom_pages"
    ensure_collection(client, collection_name=custom_name, vector_dim=dim)
    rng = np.random.default_rng(7)
    records = [
        PageRecord(
            doc_id="doc_a",
            page_num=1,
            doc_path="/a.pdf",
            vectors=rng.standard_normal((4, dim)).tolist(),
        )
    ]
    upsert_pages(client, records, collection_name=custom_name)
    embedder = StubEmbedder(n_patches=4, dim=dim)
    retriever = Retriever(client, embedder, collection_name=custom_name)
    results = retriever.retrieve("query", top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "doc_a"


def test_stub_embedder_embed_queries_returns_correct_shape():
    embedder = StubEmbedder(n_patches=12, dim=64)
    result = embedder.embed_queries(["query one", "query two"])
    assert len(result) == 2
    for vec_list in result:
        assert len(vec_list) == 12
        assert len(vec_list[0]) == 64
