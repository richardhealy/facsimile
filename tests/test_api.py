"""Tests for the FastAPI application (M5)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facsimile.api.main import (
    app,
    get_embedder,
    get_pdf_store,
    get_qdrant_client,
)
from facsimile.index.store import (
    PageRecord,
    ensure_collection,
    get_memory_client,
    upsert_pages,
)
from facsimile.ingest.embed import COLPALI_DIM, StubEmbedder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(n_pages: int = 1) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(200, 200, 200))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_client():
    """TestClient with a fresh, empty index — no documents ingested."""
    qdrant = get_memory_client()
    ensure_collection(qdrant)
    embedder = StubEmbedder(n_patches=4, dim=COLPALI_DIM)
    pdf_store: dict[str, bytes] = {}

    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_pdf_store] = lambda: pdf_store

    with TestClient(app) as c:
        c._pdf_store = pdf_store  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def populated_client():
    """TestClient with two pages of 'testdoc' already in the index and pdf_store."""
    qdrant = get_memory_client()
    ensure_collection(qdrant)
    embedder = StubEmbedder(n_patches=4, dim=COLPALI_DIM)
    pdf_bytes = _make_pdf(2)
    pdf_store: dict[str, bytes] = {"testdoc": pdf_bytes}

    dummy = _dummy_image()
    records = [
        PageRecord(
            doc_id="testdoc",
            page_num=i + 1,
            doc_path="/tmp/testdoc.pdf",
            vectors=embedder.embed_images([dummy])[0],
        )
        for i in range(2)
    ]
    upsert_pages(qdrant, records)

    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_pdf_store] = lambda: pdf_store

    with TestClient(app) as c:
        c._pdf_store = pdf_store  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, empty_client: TestClient) -> None:
        assert empty_client.get("/health").status_code == 200

    def test_status_ok(self, empty_client: TestClient) -> None:
        assert empty_client.get("/health").json()["status"] == "ok"

    def test_includes_version(self, empty_client: TestClient) -> None:
        assert "version" in empty_client.get("/health").json()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_returns_200(self, empty_client: TestClient) -> None:
        pdf = _make_pdf(1)
        resp = empty_client.post(
            "/ingest",
            files={"file": ("report.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 200

    def test_ingest_returns_doc_id(self, empty_client: TestClient) -> None:
        pdf = _make_pdf(1)
        data = empty_client.post(
            "/ingest",
            files={"file": ("myreport.pdf", pdf, "application/pdf")},
        ).json()
        assert data["doc_id"] == "myreport"

    def test_ingest_returns_page_count(self, empty_client: TestClient) -> None:
        pdf = _make_pdf(3)
        data = empty_client.post(
            "/ingest",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        ).json()
        assert data["page_count"] == 3

    def test_ingest_patches_per_page_length_matches_page_count(
        self, empty_client: TestClient
    ) -> None:
        pdf = _make_pdf(2)
        data = empty_client.post(
            "/ingest",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        ).json()
        assert len(data["patches_per_page"]) == data["page_count"]

    def test_ingest_custom_doc_id(self, empty_client: TestClient) -> None:
        pdf = _make_pdf(1)
        data = empty_client.post(
            "/ingest",
            files={"file": ("file.pdf", pdf, "application/pdf")},
            data={"doc_id": "annual_report_2024"},
        ).json()
        assert data["doc_id"] == "annual_report_2024"

    def test_ingest_stores_pdf_for_page_serving(self, empty_client: TestClient) -> None:
        pdf = _make_pdf(1)
        empty_client.post(
            "/ingest",
            files={"file": ("stored.pdf", pdf, "application/pdf")},
        )
        assert "stored" in empty_client._pdf_store  # type: ignore[attr-defined]

    def test_ingest_rejects_non_pdf(self, empty_client: TestClient) -> None:
        resp = empty_client.post(
            "/ingest",
            files={"file": ("data.csv", b"col1,col2", "text/csv")},
        )
        assert resp.status_code == 422

    def test_ingest_rejects_html_file(self, empty_client: TestClient) -> None:
        resp = empty_client.post(
            "/ingest",
            files={"file": ("page.html", b"<html/>", "text/html")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_empty_index_returns_empty(self, empty_client: TestClient) -> None:
        resp = empty_client.post("/query", json={"query": "revenue"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_query_returns_results_when_populated(
        self, populated_client: TestClient
    ) -> None:
        resp = populated_client.post("/query", json={"query": "revenue"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0

    def test_query_result_has_expected_fields(
        self, populated_client: TestClient
    ) -> None:
        result = populated_client.post(
            "/query", json={"query": "anything"}
        ).json()["results"][0]
        assert "doc_id" in result
        assert "page_num" in result
        assert "score" in result
        assert "doc_path" in result

    def test_query_respects_top_k(self, populated_client: TestClient) -> None:
        resp = populated_client.post(
            "/query", json={"query": "data", "top_k": 1}
        )
        assert len(resp.json()["results"]) == 1

    def test_query_top_k_larger_than_index_returns_all(
        self, populated_client: TestClient
    ) -> None:
        resp = populated_client.post(
            "/query", json={"query": "data", "top_k": 100}
        )
        assert len(resp.json()["results"]) == 2  # 2 pages in testdoc

    def test_query_scores_descending(self, populated_client: TestClient) -> None:
        results = populated_client.post(
            "/query", json={"query": "total revenue"}
        ).json()["results"]
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_answer_empty_index_returns_404(self, empty_client: TestClient) -> None:
        resp = empty_client.post("/answer", json={"query": "what is revenue?"})
        assert resp.status_code == 404

    def test_answer_with_pages_calls_synthesize(
        self, populated_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from facsimile.answer.synthesizer import Answer

        mock_answer = Answer(
            text="Revenue was $100M [Page 1].", cited_pages=[1], model="claude-test"
        )

        monkeypatch.setattr(
            "facsimile.answer.synthesizer.synthesize", lambda *a, **kw: mock_answer
        )

        resp = populated_client.post("/answer", json={"query": "what is revenue?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == mock_answer.text
        assert data["cited_pages"] == [1]
        assert data["model"] == "claude-test"

    def test_answer_no_pdf_in_store_returns_422(
        self, empty_client: TestClient
    ) -> None:
        # Add pages to the index but NOT to the pdf_store
        from facsimile.index.store import PageRecord, upsert_pages

        qdrant = empty_client.app.dependency_overrides[get_qdrant_client]()  # type: ignore[attr-defined]
        embedder = empty_client.app.dependency_overrides[get_embedder]()  # type: ignore[attr-defined]
        dummy = _dummy_image()
        records = [
            PageRecord(
                doc_id="ghost",
                page_num=1,
                doc_path="/tmp/ghost.pdf",
                vectors=embedder.embed_images([dummy])[0],
            )
        ]
        upsert_pages(qdrant, records)
        # pdf_store is empty — no bytes stored for "ghost"
        resp = empty_client.post("/answer", json={"query": "anything"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Page image
# ---------------------------------------------------------------------------


class TestPageImage:
    def test_page_returns_png(self, populated_client: TestClient) -> None:
        resp = populated_client.get("/pages/testdoc/1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_page_is_valid_image(self, populated_client: TestClient) -> None:
        resp = populated_client.get("/pages/testdoc/1")
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "PNG"

    def test_page_second_page(self, populated_client: TestClient) -> None:
        resp = populated_client.get("/pages/testdoc/2")
        assert resp.status_code == 200

    def test_page_unknown_doc_returns_404(self, populated_client: TestClient) -> None:
        assert populated_client.get("/pages/nope/1").status_code == 404

    def test_page_out_of_range_returns_404(self, populated_client: TestClient) -> None:
        assert populated_client.get("/pages/testdoc/99").status_code == 404

    def test_page_zero_index_returns_404(self, populated_client: TestClient) -> None:
        assert populated_client.get("/pages/testdoc/0").status_code == 404


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class TestViewer:
    def test_viewer_returns_200(self, empty_client: TestClient) -> None:
        assert empty_client.get("/viewer").status_code == 200

    def test_viewer_returns_html(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/viewer")
        assert "text/html" in resp.headers["content-type"]

    def test_viewer_contains_form(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/viewer")
        assert "<form" in resp.text

    def test_viewer_with_query_empty_index_shows_no_results(
        self, empty_client: TestClient
    ) -> None:
        resp = empty_client.get("/viewer?query=revenue")
        assert resp.status_code == 200
        assert "No pages found" in resp.text

    def test_viewer_with_query_shows_results(
        self, populated_client: TestClient
    ) -> None:
        resp = populated_client.get("/viewer?query=revenue")
        assert resp.status_code == 200
        assert "testdoc" in resp.text

    def test_viewer_query_preserved_in_input(
        self, empty_client: TestClient
    ) -> None:
        resp = empty_client.get("/viewer?query=total+revenue")
        assert "total revenue" in resp.text

    def test_viewer_best_match_highlighted(
        self, populated_client: TestClient
    ) -> None:
        resp = populated_client.get("/viewer?query=revenue")
        assert "best" in resp.text

    def test_viewer_escapes_query_xss(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/viewer?query=<script>alert(1)</script>")
        assert "<script>" not in resp.text
