"""Tests for the Qdrant multivector page store (Qdrant client mocked)."""

from unittest.mock import MagicMock, patch

from facsimile.index.store import COLLECTION_NAME, VECTOR_DIM, PageStore, _point_id

# ── helpers ──────────────────────────────────────────────────────────────────


def _mock_store() -> tuple[PageStore, MagicMock]:
    """Return (store, mock_qdrant_instance) with collections empty by default."""
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        mock.get_collections.return_value.collections = []
        store = PageStore(url="http://localhost:6333")
        return store, mock


# ── _point_id ────────────────────────────────────────────────────────────────


def test_point_id_is_deterministic() -> None:
    assert _point_id("doc1", 1) == _point_id("doc1", 1)


def test_point_id_differs_by_page() -> None:
    assert _point_id("doc1", 1) != _point_id("doc1", 2)


def test_point_id_differs_by_doc() -> None:
    assert _point_id("doc1", 1) != _point_id("doc2", 1)


def test_point_id_is_valid_uuid_string() -> None:
    import uuid

    uid = _point_id("doc1", 1)
    uuid.UUID(uid)  # raises ValueError if invalid


# ── ensure_collection ────────────────────────────────────────────────────────


def test_ensure_collection_creates_when_missing() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        mock.get_collections.return_value.collections = []

        store = PageStore()
        store.ensure_collection()

        mock.create_collection.assert_called_once()
        kwargs = mock.create_collection.call_args[1]
        assert kwargs["collection_name"] == COLLECTION_NAME


def test_ensure_collection_skips_when_exists() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        existing = MagicMock()
        existing.name = COLLECTION_NAME
        mock.get_collections.return_value.collections = [existing]

        store = PageStore()
        store.ensure_collection()

        mock.create_collection.assert_not_called()


# ── upsert_page ───────────────────────────────────────────────────────────────


def test_upsert_page_calls_qdrant_upsert() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        store = PageStore()

        vectors = [[0.1] * VECTOR_DIM, [0.2] * VECTOR_DIM]
        store.upsert_page("doc1", 1, vectors)

        mock.upsert.assert_called_once()


def test_upsert_page_payload_metadata() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        store = PageStore()

        vectors = [[0.5] * VECTOR_DIM]
        store.upsert_page("docX", 7, vectors, image_path="/img/p7.png")

        points = mock.upsert.call_args[1]["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["doc_id"] == "docX"
        assert payload["page_number"] == 7
        assert payload["image_path"] == "/img/p7.png"


def test_upsert_page_id_is_deterministic() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        store = PageStore()

        vectors = [[0.1] * VECTOR_DIM]
        store.upsert_page("doc1", 3, vectors)
        first_id = mock.upsert.call_args[1]["points"][0].id

        mock.upsert.reset_mock()
        store.upsert_page("doc1", 3, vectors)
        second_id = mock.upsert.call_args[1]["points"][0].id

        assert first_id == second_id


# ── search ────────────────────────────────────────────────────────────────────


def test_search_calls_query_points() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        mock.query_points.return_value.points = []
        store = PageStore()

        query = [[0.5] * VECTOR_DIM]
        results = store.search(query, top_k=3)

        mock.query_points.assert_called_once()
        kwargs = mock.query_points.call_args[1]
        assert kwargs["limit"] == 3
        assert results == []


def test_search_passes_collection_name() -> None:
    with patch("facsimile.index.store.QdrantClient") as MockCls:
        mock = MagicMock()
        MockCls.return_value = mock
        mock.query_points.return_value.points = []
        store = PageStore()

        store.search([[0.1] * VECTOR_DIM])

        kwargs = mock.query_points.call_args[1]
        assert kwargs["collection_name"] == COLLECTION_NAME
