"""Tests for facsimile.index.store (in-memory Qdrant)."""

import numpy as np
import pytest

from facsimile.index.store import (
    COLLECTION_NAME,
    VECTOR_DIM,
    PageRecord,
    _page_id,
    ensure_collection,
    get_memory_client,
    query_pages,
    upsert_pages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vectors(n_patches: int = 4, dim: int = VECTOR_DIM) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_patches, dim)).tolist()


def _make_client():
    client = get_memory_client()
    ensure_collection(client)
    return client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def test_ensure_collection_creates_collection():
    client = get_memory_client()
    ensure_collection(client)
    names = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in names


def test_ensure_collection_idempotent():
    client = get_memory_client()
    ensure_collection(client)
    ensure_collection(client)  # should not raise
    names = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in names


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def test_upsert_single_page():
    client = _make_client()
    record = PageRecord(
        doc_id="doc1",
        page_num=1,
        doc_path="/tmp/doc1.pdf",
        vectors=_random_vectors(),
    )
    upsert_pages(client, [record])
    count = client.count(collection_name=COLLECTION_NAME).count
    assert count == 1


def test_upsert_multiple_pages():
    client = _make_client()
    records = [
        PageRecord(
            doc_id="doc1",
            page_num=i,
            doc_path="/tmp/doc1.pdf",
            vectors=_random_vectors(n_patches=6),
        )
        for i in range(1, 4)
    ]
    upsert_pages(client, records)
    count = client.count(collection_name=COLLECTION_NAME).count
    assert count == 3


def test_upsert_idempotent():
    """Re-upserting the same pages does not create duplicates."""
    client = _make_client()
    record = PageRecord(
        doc_id="doc1",
        page_num=1,
        doc_path="/tmp/doc1.pdf",
        vectors=_random_vectors(),
    )
    upsert_pages(client, [record])
    upsert_pages(client, [record])
    count = client.count(collection_name=COLLECTION_NAME).count
    assert count == 1


def test_payload_stored_correctly():
    client = _make_client()
    record = PageRecord(
        doc_id="annual_report_2023",
        page_num=7,
        doc_path="/data/annual_report_2023.pdf",
        vectors=_random_vectors(),
    )
    upsert_pages(client, [record])
    results = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[_page_id("annual_report_2023", 7)],
        with_payload=True,
    )
    assert len(results) == 1
    payload = results[0].payload
    assert payload["doc_id"] == "annual_report_2023"
    assert payload["page_num"] == 7
    assert payload["doc_path"] == "/data/annual_report_2023.pdf"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def test_query_returns_scored_results():
    client = _make_client()
    records = [
        PageRecord(
            doc_id="doc1",
            page_num=i,
            doc_path="/tmp/doc1.pdf",
            vectors=_random_vectors(n_patches=4),
        )
        for i in range(1, 6)
    ]
    upsert_pages(client, records)
    query_vecs = _random_vectors(n_patches=3)
    results = query_pages(client, query_vecs, top_k=3)
    assert len(results) == 3
    # Scores should be in descending order.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_top_k_respected():
    client = _make_client()
    records = [
        PageRecord(
            doc_id="doc1",
            page_num=i,
            doc_path="/tmp/doc1.pdf",
            vectors=_random_vectors(),
        )
        for i in range(1, 11)
    ]
    upsert_pages(client, records)
    results = query_pages(client, _random_vectors(), top_k=4)
    assert len(results) == 4


# ---------------------------------------------------------------------------
# _page_id
# ---------------------------------------------------------------------------

def test_page_id_is_deterministic():
    a = _page_id("doc1", 3)
    b = _page_id("doc1", 3)
    assert a == b


def test_page_id_differs_by_page():
    assert _page_id("doc1", 1) != _page_id("doc1", 2)


def test_page_id_differs_by_doc():
    assert _page_id("doc1", 1) != _page_id("doc2", 1)


def test_page_id_is_positive_int():
    pid = _page_id("doc1", 1)
    assert isinstance(pid, int)
    assert pid >= 0
