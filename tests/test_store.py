"""Tests for facsimile.index.store (in-memory Qdrant)."""

import os
import subprocess
import sys

import numpy as np
import pytest

from facsimile.index.store import (
    COLLECTION_NAME,
    VECTOR_DIM,
    PageRecord,
    _page_id,
    delete_other_pages,
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


def _doc_pages(client, doc_id: str, page_nums) -> None:
    upsert_pages(
        client,
        [PageRecord(doc_id, n, f"/tmp/{doc_id}.pdf", _random_vectors()) for n in page_nums],
    )


def _stored(client) -> list[tuple[str, int]]:
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
    return sorted((p.payload["doc_id"], p.payload["page_num"]) for p in points)


def test_delete_other_pages_keeps_listed_pages_only():
    client = _make_client()
    _doc_pages(client, "doc1", [1, 2, 3])
    _doc_pages(client, "doc2", [1, 2, 3])
    delete_other_pages(client, "doc1", [1, 2])
    assert _stored(client) == [
        ("doc1", 1), ("doc1", 2), ("doc2", 1), ("doc2", 2), ("doc2", 3),
    ]


def test_delete_other_pages_with_nothing_to_keep_removes_the_document():
    client = _make_client()
    _doc_pages(client, "doc1", [1, 2])
    _doc_pages(client, "doc2", [1])
    delete_other_pages(client, "doc1", [])
    assert _stored(client) == [("doc2", 1)]


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


def test_page_id_fits_in_unsigned_64_bits():
    for doc_id, page_num in [("doc1", 1), ("annual_report", 7), ("x" * 500, 9999)]:
        assert 0 <= _page_id(doc_id, page_num) < 2**64


def test_page_id_is_stable_across_processes():
    """IDs must not depend on the per-process string hash salt.

    Python salts ``hash()`` of strings per process, so an ID built on it
    changes on every restart and re-ingesting into a persistent Qdrant adds
    a second copy of each page instead of overwriting the first.
    """
    code = "from facsimile.index.store import _page_id; print(_page_id('annual_report', 7))"
    ids = set()
    for seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        ids.add(int(out.stdout.strip()))
    assert ids == {_page_id("annual_report", 7)}


def test_page_id_scheme_is_pinned():
    """Changing the ID scheme orphans every point already stored, so pin it."""
    assert _page_id("annual_report", 7) == 0xF5F882AD78F53697
