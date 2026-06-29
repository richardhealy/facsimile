"""Tests for M2 retrieval: Retriever and embed_queries."""

import numpy as np
import pytest

from facsimile.index.store import (
    COLLECTION_NAME,
    PageRecord,
    ensure_collection,
    get_memory_client,
    upsert_pages,
)
from facsimile.ingest.embed import COLPALI_DIM, Embedder, StubEmbedder
from facsimile.retrieve.retriever import RetrievedPage, Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_populated_client(n_pages: int = 5, n_patches: int = 8, seed: int = 0):
    """Return (client, embedder) with n_pages indexed."""
    client = get_memory_client()
    ensure_collection(client)
    embedder = StubEmbedder(n_patches=n_patches, dim=COLPALI_DIM, seed=seed)
    records = [
        PageRecord(
            doc_id="test_doc",
            page_num=i + 1,
            doc_path=f"/tmp/test_doc.pdf",
            vectors=embedder.embed_images([None])[0],  # type: ignore[arg-type]
        )
        for i in range(n_pages)
    ]
    upsert_pages(client, records)
    return client, embedder


# ---------------------------------------------------------------------------
# StubEmbedder.embed_queries
# ---------------------------------------------------------------------------

class TestStubEmbedderQueries:
    def test_returns_one_result_per_query(self):
        emb = StubEmbedder(n_patches=4, dim=16)
        result = emb.embed_queries(["what is the revenue?", "show total costs"])
        assert len(result) == 2

    def test_shape_per_query(self):
        emb = StubEmbedder(n_patches=6, dim=32)
        result = emb.embed_queries(["single query"])
        assert len(result[0]) == 6
        assert len(result[0][0]) == 32

    def test_empty_list_returns_empty(self):
        emb = StubEmbedder()
        assert emb.embed_queries([]) == []

    def test_different_seeds_differ(self):
        emb1 = StubEmbedder(seed=1)
        emb2 = StubEmbedder(seed=2)
        v1 = emb1.embed_queries(["hello"])
        v2 = emb2.embed_queries(["hello"])
        assert v1 != v2


# ---------------------------------------------------------------------------
# Embedder protocol compliance
# ---------------------------------------------------------------------------

def test_stub_embedder_satisfies_protocol():
    assert isinstance(StubEmbedder(), Embedder)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_retrieve_returns_list(self):
        client, embedder = _make_populated_client()
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("what is total revenue?", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_returns_retrieved_page_objects(self):
        client, embedder = _make_populated_client()
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("test query", top_k=2)
        assert all(isinstance(r, RetrievedPage) for r in results)

    def test_retrieve_respects_top_k(self):
        client, embedder = _make_populated_client(n_pages=10)
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("revenue", top_k=4)
        assert len(results) == 4

    def test_retrieve_top_k_larger_than_index(self):
        client, embedder = _make_populated_client(n_pages=3)
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("revenue", top_k=10)
        assert len(results) == 3

    def test_retrieve_scores_descending(self):
        client, embedder = _make_populated_client(n_pages=5)
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("show me the data", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_payload_populated(self):
        client, embedder = _make_populated_client()
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("test", top_k=1)
        page = results[0]
        assert page.doc_id == "test_doc"
        assert isinstance(page.page_num, int)
        assert page.page_num >= 1
        assert page.doc_path == "/tmp/test_doc.pdf"
        assert isinstance(page.score, float)

    def test_retrieve_empty_index_returns_empty(self):
        client = get_memory_client()
        ensure_collection(client)
        embedder = StubEmbedder()
        retriever = Retriever(client, embedder)
        results = retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_uses_custom_collection(self):
        client = get_memory_client()
        custom_name = "custom_collection"
        from facsimile.index.store import VECTOR_DIM, ensure_collection as ec, upsert_pages
        ec(client, collection_name=custom_name, vector_dim=VECTOR_DIM)
        embedder = StubEmbedder(n_patches=4, dim=VECTOR_DIM)
        record = PageRecord(
            doc_id="doc1",
            page_num=1,
            doc_path="/tmp/doc1.pdf",
            vectors=embedder.embed_images([None])[0],  # type: ignore[arg-type]
        )
        upsert_pages(client, [record], collection_name=custom_name)
        retriever = Retriever(client, embedder, collection_name=custom_name)
        results = retriever.retrieve("test", top_k=1)
        assert len(results) == 1

    def test_relevant_page_ranked_first(self):
        """A page whose vectors correlate with the query should rank higher."""
        rng = np.random.default_rng(99)
        client = get_memory_client()
        ensure_collection(client)

        # Build a deterministic query vector set.
        query_vecs = rng.standard_normal((8, COLPALI_DIM)).astype(np.float32)
        # Relevant page: nearly identical to query vectors.
        relevant_vecs = (query_vecs + rng.standard_normal((8, COLPALI_DIM)) * 0.05).tolist()
        # Noise pages.
        noise_pages = [
            rng.standard_normal((8, COLPALI_DIM)).tolist()
            for _ in range(4)
        ]

        records = [
            PageRecord(doc_id="doc", page_num=1, doc_path="/f.pdf", vectors=relevant_vecs),
            *[
                PageRecord(doc_id="doc", page_num=i + 2, doc_path="/f.pdf", vectors=v)
                for i, v in enumerate(noise_pages)
            ],
        ]
        upsert_pages(client, records)

        class FixedQueryEmbedder:
            """Returns predetermined query vectors regardless of the query string."""
            def embed_images(self, images):
                return []

            def embed_queries(self, texts):
                return [query_vecs.tolist() for _ in texts]

        retriever = Retriever(client, FixedQueryEmbedder())
        results = retriever.retrieve("any text", top_k=5)
        assert results[0].page_num == 1
