"""Tests for M2 retrieval: late-interaction scoring and top-page selection."""

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
from facsimile.retrieve.retriever import RetrievedPage, retrieve, retrieve_for_query


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_client():
    """In-memory Qdrant client pre-loaded with 5 pages from 2 documents."""
    client = get_memory_client()
    ensure_collection(client)
    rng = np.random.default_rng(0)
    records = [
        PageRecord(
            doc_id="doc1",
            page_num=i + 1,
            doc_path="/data/doc1.pdf",
            vectors=rng.standard_normal((8, 128)).tolist(),
        )
        for i in range(3)
    ] + [
        PageRecord(
            doc_id="doc2",
            page_num=i + 1,
            doc_path="/data/doc2.pdf",
            vectors=rng.standard_normal((8, 128)).tolist(),
        )
        for i in range(2)
    ]
    upsert_pages(client, records)
    return client


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_returns_retrieved_page_objects(self, populated_client):
        query = np.random.default_rng(99).standard_normal((8, 128)).tolist()
        results = retrieve(query, populated_client)
        assert all(isinstance(r, RetrievedPage) for r in results)

    def test_respects_top_k(self, populated_client):
        query = np.random.default_rng(1).standard_normal((8, 128)).tolist()
        assert len(retrieve(query, populated_client, top_k=3)) == 3
        assert len(retrieve(query, populated_client, top_k=1)) == 1

    def test_scores_are_descending(self, populated_client):
        query = np.random.default_rng(2).standard_normal((8, 128)).tolist()
        results = retrieve(query, populated_client)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_payload_fields_present(self, populated_client):
        query = np.random.default_rng(3).standard_normal((8, 128)).tolist()
        r = retrieve(query, populated_client, top_k=1)[0]
        assert r.doc_id in {"doc1", "doc2"}
        assert isinstance(r.page_num, int)
        assert r.doc_path.endswith(".pdf")
        assert isinstance(r.score, float)

    def test_top_k_clamps_to_collection_size(self, populated_client):
        query = np.random.default_rng(4).standard_normal((8, 128)).tolist()
        results = retrieve(query, populated_client, top_k=100)
        assert len(results) == 5

    def test_most_similar_page_ranks_first(self):
        """A page with vectors near the query must be ranked first."""
        client = get_memory_client()
        ensure_collection(client)

        rng = np.random.default_rng(77)
        query_vecs = rng.standard_normal((4, 128)).astype(np.float32)

        # Page 1: nearly identical to query — should rank first
        relevant_vecs = (query_vecs + rng.standard_normal((4, 128)).astype(np.float32) * 0.01).tolist()
        # Pages 2 and 3: unrelated random vectors
        irrelevant_vecs1 = rng.standard_normal((4, 128)).tolist()
        irrelevant_vecs2 = rng.standard_normal((4, 128)).tolist()

        upsert_pages(client, [
            PageRecord(doc_id="doc", page_num=1, doc_path="/p.pdf", vectors=relevant_vecs),
            PageRecord(doc_id="doc", page_num=2, doc_path="/p.pdf", vectors=irrelevant_vecs1),
            PageRecord(doc_id="doc", page_num=3, doc_path="/p.pdf", vectors=irrelevant_vecs2),
        ])

        results = retrieve(query_vecs.tolist(), client, top_k=3)
        assert results[0].page_num == 1

    def test_empty_collection_returns_empty_list(self):
        client = get_memory_client()
        ensure_collection(client)
        query = np.random.default_rng(5).standard_normal((8, 128)).tolist()
        assert retrieve(query, client) == []


# ---------------------------------------------------------------------------
# retrieve_for_query()
# ---------------------------------------------------------------------------


class TestRetrieveForQuery:
    def test_returns_retrieved_page_objects(self, populated_client):
        embedder = StubEmbedder()
        results = retrieve_for_query("what is the revenue?", embedder, populated_client)
        assert all(isinstance(r, RetrievedPage) for r in results)

    def test_same_query_same_results(self, populated_client):
        embedder = StubEmbedder()
        r1 = retrieve_for_query("operating profit by region", embedder, populated_client)
        r2 = retrieve_for_query("operating profit by region", embedder, populated_client)
        assert [p.page_num for p in r1] == [p.page_num for p in r2]

    def test_respects_top_k(self, populated_client):
        embedder = StubEmbedder()
        assert len(retrieve_for_query("chart", embedder, populated_client, top_k=2)) == 2


# ---------------------------------------------------------------------------
# StubEmbedder.embed_query
# ---------------------------------------------------------------------------


class TestStubEmbedderQuery:
    def test_returns_list_of_vectors(self):
        vecs = StubEmbedder(n_patches=8, dim=128).embed_query("hello")
        assert len(vecs) == 8
        assert len(vecs[0]) == 128

    def test_deterministic_across_instances(self):
        a = StubEmbedder().embed_query("what is EBITDA?")
        b = StubEmbedder().embed_query("what is EBITDA?")
        assert a == b

    def test_different_texts_give_different_vectors(self):
        emb = StubEmbedder()
        assert emb.embed_query("revenue") != emb.embed_query("operating profit")

    def test_vectors_are_floats(self):
        vecs = StubEmbedder().embed_query("test")
        assert all(isinstance(v, float) for v in vecs[0])
