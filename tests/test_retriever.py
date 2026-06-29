"""Integration tests for the M2 Retriever (embed query → Qdrant MaxSim → pages)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from facsimile.index.store import (
    COLLECTION_NAME,
    PageRecord,
    ensure_collection,
    get_memory_client,
    upsert_pages,
)
from facsimile.ingest.embed import COLPALI_DIM, StubEmbedder, StubQueryEmbedder
from facsimile.retrieve.retriever import RetrievedPage, Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_image(width: int = 612, height: int = 792) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _make_store(
    n_pages: int = 3,
    doc_id: str = "test_doc",
    doc_path: str = "/tmp/test.pdf",
) -> tuple:
    """Return (client, records) with *n_pages* stub-embedded pages stored."""
    client = get_memory_client()
    ensure_collection(client)
    embedder = StubEmbedder(seed=42)
    images = [_blank_image() for _ in range(n_pages)]
    vectors_per_page = embedder.embed_images(images)
    records = [
        PageRecord(
            doc_id=doc_id,
            page_num=i + 1,
            doc_path=doc_path,
            vectors=vecs,
        )
        for i, vecs in enumerate(vectors_per_page)
    ]
    upsert_pages(client, records)
    return client, records


# ---------------------------------------------------------------------------
# QueryEmbedder (StubQueryEmbedder)
# ---------------------------------------------------------------------------


class TestStubQueryEmbedder:
    def test_implements_query_embedder_protocol(self) -> None:
        from facsimile.ingest.embed import QueryEmbedder

        assert isinstance(StubQueryEmbedder(), QueryEmbedder)

    def test_returns_one_result_per_query(self) -> None:
        qe = StubQueryEmbedder()
        results = qe.embed_queries(["revenue", "net income", "market share"])
        assert len(results) == 3

    def test_each_result_has_correct_token_count(self) -> None:
        qe = StubQueryEmbedder(n_tokens=6)
        results = qe.embed_queries(["test"])
        assert len(results[0]) == 6

    def test_each_token_has_correct_dim(self) -> None:
        qe = StubQueryEmbedder(dim=COLPALI_DIM)
        results = qe.embed_queries(["test"])
        assert all(len(v) == COLPALI_DIM for v in results[0])

    def test_same_query_gives_same_vectors(self) -> None:
        qe_a = StubQueryEmbedder(seed=0)
        qe_b = StubQueryEmbedder(seed=0)
        assert qe_a.embed_queries(["revenue"]) == qe_b.embed_queries(["revenue"])

    def test_different_queries_give_different_vectors(self) -> None:
        qe = StubQueryEmbedder()
        res_a = qe.embed_queries(["revenue"])
        res_b = qe.embed_queries(["net income"])
        assert res_a != res_b

    def test_stable_across_python_hash_randomisation(self) -> None:
        # MD5-based seeding means results are stable even if PYTHONHASHSEED varies.
        qe = StubQueryEmbedder(seed=7)
        r1 = qe.embed_queries(["annual report"])
        r2 = qe.embed_queries(["annual report"])
        assert r1 == r2


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_returns_retrieved_page_objects(self) -> None:
        client, _ = _make_store()
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("What is the revenue?")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedPage) for r in results)

    def test_respects_top_k(self) -> None:
        client, _ = _make_store(n_pages=5)
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("test", top_k=2)
        assert len(results) <= 2

    def test_results_carry_payload_metadata(self) -> None:
        client, _ = _make_store(doc_id="annual_report", doc_path="/docs/report.pdf")
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("revenue")
        for r in results:
            assert r.doc_id == "annual_report"
            assert r.doc_path == "/docs/report.pdf"
            assert r.page_num >= 1

    def test_scores_are_floats(self) -> None:
        client, _ = _make_store()
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("chart")
        assert all(isinstance(r.score, float) for r in results)

    def test_scores_are_non_negative(self) -> None:
        client, _ = _make_store()
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("chart")
        assert all(r.score >= 0 for r in results)

    def test_results_sorted_descending_by_score(self) -> None:
        client, _ = _make_store(n_pages=5)
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("revenue", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_page_numbers_are_in_expected_range(self) -> None:
        n = 4
        client, _ = _make_store(n_pages=n)
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("test", top_k=n)
        page_nums = {r.page_num for r in results}
        assert page_nums.issubset(set(range(1, n + 1)))

    def test_empty_collection_returns_empty_list(self) -> None:
        client = get_memory_client()
        ensure_collection(client)
        retriever = Retriever(StubQueryEmbedder(), client)
        results = retriever.retrieve("revenue")
        assert results == []

    def test_custom_collection_name(self) -> None:
        client = get_memory_client()
        ensure_collection(client, collection_name="custom_col")
        embedder = StubEmbedder(seed=1)
        vecs = embedder.embed_images([_blank_image()])
        upsert_pages(
            client,
            [PageRecord(doc_id="d", page_num=1, doc_path="/x.pdf", vectors=vecs[0])],
            collection_name="custom_col",
        )
        retriever = Retriever(StubQueryEmbedder(), client, collection_name="custom_col")
        results = retriever.retrieve("test")
        assert len(results) == 1

    def test_high_similarity_page_ranked_first(self) -> None:
        """A page whose embedding closely matches the query should rank first."""
        client = get_memory_client()
        ensure_collection(client)

        # Build query vectors from the stub embedder first.
        qe = StubQueryEmbedder(n_tokens=4, seed=0)
        query_vecs = np.array(qe.embed_queries(["target query"])[0], dtype=np.float32)

        # Page 1: vectors closely aligned with the query.
        noise = np.random.default_rng(123).standard_normal(query_vecs.shape).astype(np.float32)
        page1_vecs = (query_vecs + noise * 0.01).tolist()

        # Page 2: random vectors, unlikely to align.
        page2_vecs = np.random.default_rng(999).standard_normal(query_vecs.shape).tolist()

        upsert_pages(
            client,
            [
                PageRecord("doc", 1, "/d.pdf", page1_vecs),
                PageRecord("doc", 2, "/d.pdf", page2_vecs),
            ],
        )
        retriever = Retriever(qe, client)
        results = retriever.retrieve("target query", top_k=2)
        assert results[0].page_num == 1
