"""Tests for the Qdrant multivector store."""

import numpy as np
import pytest

from facsimile.index.store import PageVectorStore, _page_uuid


VECTOR_SIZE = 16
NUM_PATCHES = 8


def _rand_vectors(num: int = NUM_PATCHES, size: int = VECTOR_SIZE) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.standard_normal((num, size)).tolist()


@pytest.fixture()
def store() -> PageVectorStore:
    s = PageVectorStore(location=":memory:")
    s.create_collection(vector_size=VECTOR_SIZE)
    return s


def test_collection_exists(store: PageVectorStore) -> None:
    assert store.collection_exists()


def test_count_empty(store: PageVectorStore) -> None:
    assert store.count() == 0


def test_upsert_increments_count(store: PageVectorStore) -> None:
    store.upsert("doc1", 0, _rand_vectors(), {"source": "test.pdf"})
    assert store.count() == 1


def test_upsert_multiple_pages(store: PageVectorStore) -> None:
    for i in range(3):
        store.upsert("doc1", i, _rand_vectors())
    assert store.count() == 3


def test_query_returns_top_k(store: PageVectorStore) -> None:
    for i in range(5):
        store.upsert("doc_q", i, _rand_vectors())
    results = store.query(_rand_vectors(), top_k=3)
    assert len(results) == 3


def test_query_payload_contains_page(store: PageVectorStore) -> None:
    store.upsert("mydoc", 2, _rand_vectors(), {"title": "annual report"})
    results = store.query(_rand_vectors(), top_k=1)
    assert results[0].payload["doc_id"] == "mydoc"
    assert results[0].payload["page"] == 2
    assert results[0].payload["title"] == "annual report"


def test_upsert_idempotent(store: PageVectorStore) -> None:
    vecs = _rand_vectors()
    store.upsert("doc_idem", 0, vecs)
    store.upsert("doc_idem", 0, vecs)
    assert store.count() == 1


def test_page_uuid_stable() -> None:
    a = _page_uuid("report.pdf", 3)
    b = _page_uuid("report.pdf", 3)
    assert a == b


def test_page_uuid_distinct_pages() -> None:
    assert _page_uuid("doc", 0) != _page_uuid("doc", 1)


def test_page_uuid_distinct_docs() -> None:
    assert _page_uuid("a.pdf", 0) != _page_uuid("b.pdf", 0)
