"""Tests for facsimile.index.store."""

import pytest

from facsimile.index.store import (
    COLLECTION_NAME,
    PageRecord,
    _page_id,
    ensure_collection,
    make_memory_client,
    upsert_pages,
)

PATCH_DIM = 4  # tiny dimension so tests run fast


@pytest.fixture()
def client():
    c = make_memory_client()
    ensure_collection(c, patch_dim=PATCH_DIM)
    return c


class TestEnsureCollection:
    def test_creates_collection(self):
        c = make_memory_client()
        ensure_collection(c, patch_dim=PATCH_DIM)
        names = {col.name for col in c.get_collections().collections}
        assert COLLECTION_NAME in names

    def test_idempotent(self):
        c = make_memory_client()
        ensure_collection(c, patch_dim=PATCH_DIM)
        ensure_collection(c, patch_dim=PATCH_DIM)  # second call must not raise
        names = {col.name for col in c.get_collections().collections}
        assert COLLECTION_NAME in names


class TestUpsertPages:
    def test_upsert_and_count(self, client):
        records = [
            PageRecord(
                doc_id="doc1",
                page_number=1,
                vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            ),
            PageRecord(
                doc_id="doc1",
                page_number=2,
                vectors=[[0.9, 0.1, 0.2, 0.3]],
            ),
        ]
        upsert_pages(client, records)
        info = client.get_collection(COLLECTION_NAME)
        assert info.points_count == 2

    def test_upsert_overwrites_existing(self, client):
        r = PageRecord(doc_id="doc2", page_number=1, vectors=[[0.1, 0.2, 0.3, 0.4]])
        upsert_pages(client, [r])
        upsert_pages(client, [r])  # re-upsert same page
        info = client.get_collection(COLLECTION_NAME)
        assert info.points_count == 1

    def test_payload_stored(self, client):
        r = PageRecord(
            doc_id="doc3",
            page_number=1,
            vectors=[[0.1, 0.2, 0.3, 0.4]],
            payload={"source": "annual_report.pdf"},
        )
        upsert_pages(client, [r])
        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_page_id("doc3", 1)],
            with_payload=True,
        )
        assert result[0].payload["source"] == "annual_report.pdf"
        assert result[0].payload["page_number"] == 1


class TestPageId:
    def test_stable_across_calls(self):
        assert _page_id("doc", 1) == _page_id("doc", 1)

    def test_different_pages_differ(self):
        assert _page_id("doc", 1) != _page_id("doc", 2)

    def test_different_docs_differ(self):
        assert _page_id("docA", 1) != _page_id("docB", 1)
