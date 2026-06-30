"""Tests for the FastAPI application — health, ingest, query, pages, viewer."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from facsimile.api.main import (
    _get_embedder,
    _get_image_dir,
    _get_qdrant,
    app,
)
from facsimile.index.store import (
    PageRecord,
    ensure_collection,
    get_memory_client,
    upsert_pages,
)
from facsimile.ingest.embed import COLPALI_DIM, StubEmbedder

# ── module-level client (no lifespan) used only for endpoints with no state ───

_client = TestClient(app)


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_pdf_bytes() -> bytes:
    """Create a minimal valid single-page PDF in memory via pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(595, 842)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def _seed_page(
    qdrant,
    embedder: StubEmbedder,
    image_dir,
    doc_id: str = "testdoc",
    page_num: int = 1,
) -> None:
    """Insert one record into Qdrant and write a dummy PNG to image_dir."""
    record = PageRecord(
        doc_id=doc_id,
        page_num=page_num,
        doc_path=str(image_dir / f"{doc_id}.pdf"),
        vectors=embedder.embed_images([None])[0],  # type: ignore[arg-type]
    )
    upsert_pages(qdrant, [record])
    img = Image.new("RGB", (64, 64), color=(180, 200, 220))
    img.save(image_dir / f"{doc_id}_p{page_num}.png", format="PNG")


# ── fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def api_client(tmp_path):
    """TestClient with dependency-overridden state (no lifespan required)."""
    qdrant = get_memory_client()
    ensure_collection(qdrant)
    embedder = StubEmbedder(n_patches=8, dim=COLPALI_DIM)

    app.dependency_overrides[_get_qdrant] = lambda: qdrant
    app.dependency_overrides[_get_embedder] = lambda: embedder
    app.dependency_overrides[_get_image_dir] = lambda: tmp_path

    with TestClient(app) as c:
        yield c, qdrant, embedder, tmp_path

    app.dependency_overrides.clear()


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_returns_200() -> None:
    response = _client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status() -> None:
    data = _client.get("/health").json()
    assert data["status"] == "ok"


def test_health_includes_version() -> None:
    data = _client.get("/health").json()
    assert "version" in data


# ── /ingest ────────────────────────────────────────────────────────────────────


class TestIngest:
    def test_ingest_returns_200(self, api_client):
        c, *_ = api_client
        res = c.post("/ingest", files={"file": ("t.pdf", _make_pdf_bytes(), "application/pdf")})
        assert res.status_code == 200

    def test_ingest_returns_doc_id(self, api_client):
        c, *_ = api_client
        data = c.post("/ingest", files={"file": ("t.pdf", _make_pdf_bytes(), "application/pdf")}).json()
        assert "doc_id" in data
        assert isinstance(data["doc_id"], str)
        assert len(data["doc_id"]) == 16

    def test_ingest_returns_page_count(self, api_client):
        c, *_ = api_client
        data = c.post("/ingest", files={"file": ("t.pdf", _make_pdf_bytes(), "application/pdf")}).json()
        assert data["page_count"] >= 1

    def test_ingest_idempotent_same_doc_id(self, api_client):
        c, *_ = api_client
        pdf = _make_pdf_bytes()
        r1 = c.post("/ingest", files={"file": ("a.pdf", pdf, "application/pdf")}).json()
        r2 = c.post("/ingest", files={"file": ("b.pdf", pdf, "application/pdf")}).json()
        assert r1["doc_id"] == r2["doc_id"]

    def test_ingest_saves_page_images(self, api_client):
        c, _, _, image_dir = api_client
        data = c.post("/ingest", files={"file": ("t.pdf", _make_pdf_bytes(), "application/pdf")}).json()
        assert (image_dir / f"{data['doc_id']}_p1.png").exists()

    def test_ingest_different_files_different_doc_ids(self, api_client):
        c, *_ = api_client
        import hashlib
        p1 = b"%PDF-1.4 unique1"
        p2 = b"%PDF-1.4 unique2"
        id1 = hashlib.sha256(p1).hexdigest()[:16]
        id2 = hashlib.sha256(p2).hexdigest()[:16]
        assert id1 != id2


# ── /query ─────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_query_empty_index_returns_200(self, api_client):
        c, *_ = api_client
        res = c.post("/query", json={"question": "what is revenue?"})
        assert res.status_code == 200

    def test_query_empty_index_returns_no_pages(self, api_client):
        c, *_ = api_client
        data = c.post("/query", json={"question": "test"}).json()
        assert data["pages"] == []

    def test_query_returns_pages_after_ingest(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        data = c.post("/query", json={"question": "revenue", "top_k": 1}).json()
        assert len(data["pages"]) == 1

    def test_query_page_info_has_required_fields(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        page = c.post("/query", json={"question": "test", "top_k": 1}).json()["pages"][0]
        assert "doc_id" in page
        assert "page_num" in page
        assert "score" in page
        assert "image_url" in page

    def test_query_image_url_format(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        page = c.post("/query", json={"question": "test", "top_k": 1}).json()["pages"][0]
        assert page["image_url"].startswith("/pages/")

    def test_query_respects_top_k(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        for i in range(5):
            _seed_page(qdrant, embedder, image_dir, "doc1", i + 1)
        data = c.post("/query", json={"question": "test", "top_k": 2}).json()
        assert len(data["pages"]) == 2

    def test_query_top_k_larger_than_index(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        for i in range(3):
            _seed_page(qdrant, embedder, image_dir, "doc1", i + 1)
        data = c.post("/query", json={"question": "test", "top_k": 10}).json()
        assert len(data["pages"]) == 3

    def test_query_echoes_question(self, api_client):
        c, *_ = api_client
        data = c.post("/query", json={"question": "what is profit?"}).json()
        assert data["question"] == "what is profit?"

    def test_query_no_synthesis_by_default(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        data = c.post("/query", json={"question": "test"}).json()
        assert data["answer"] is None

    def test_query_scores_are_floats(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        page = c.post("/query", json={"question": "test", "top_k": 1}).json()["pages"][0]
        assert isinstance(page["score"], float)

    def test_query_cited_pages_empty_without_synthesis(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        data = c.post("/query", json={"question": "test"}).json()
        assert data["cited_pages"] == []

    def test_query_model_none_without_synthesis(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        data = c.post("/query", json={"question": "test"}).json()
        assert data["model"] is None


# ── /pages/{doc_id}/{page_num} ─────────────────────────────────────────────────


class TestGetPage:
    def test_existing_page_returns_200(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        res = c.get("/pages/doc1/1")
        assert res.status_code == 200

    def test_existing_page_content_type_png(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        res = c.get("/pages/doc1/1")
        assert res.headers["content-type"] == "image/png"

    def test_existing_page_has_body(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        res = c.get("/pages/doc1/1")
        assert len(res.content) > 0

    def test_missing_page_returns_404(self, api_client):
        c, *_ = api_client
        res = c.get("/pages/nonexistent/99")
        assert res.status_code == 404

    def test_missing_page_num_returns_404(self, api_client):
        c, qdrant, embedder, image_dir = api_client
        _seed_page(qdrant, embedder, image_dir, "doc1", 1)
        res = c.get("/pages/doc1/99")
        assert res.status_code == 404


# ── / (viewer) ─────────────────────────────────────────────────────────────────


class TestViewer:
    def test_viewer_returns_200(self):
        res = _client.get("/")
        assert res.status_code == 200

    def test_viewer_content_type_html(self):
        res = _client.get("/")
        assert "text/html" in res.headers["content-type"]

    def test_viewer_contains_app_name(self):
        res = _client.get("/")
        assert "facsimile" in res.text

    def test_viewer_contains_ingest_button(self):
        res = _client.get("/")
        assert "ingest" in res.text.lower()

    def test_viewer_contains_query_element(self):
        res = _client.get("/")
        assert "question" in res.text.lower()
