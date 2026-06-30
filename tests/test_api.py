"""Tests for the FastAPI application (M5 — full endpoint suite)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facsimile.answer.synthesizer import Answer as SynthAnswer
from facsimile.api.main import _state, app
from facsimile.index.store import ensure_collection, get_memory_client
from facsimile.ingest.embed import StubEmbedder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(n_pages: int = 2) -> bytes:
    """Return bytes of a minimal blank PDF with *n_pages* pages."""
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


_STUB_ANSWER = SynthAnswer(
    text="Revenue was $10M [Page 1].", cited_pages=[1], model="stub"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state(tmp_path):
    """Before each test, wire _state to a clean in-memory client and tmp dir."""
    client = get_memory_client()
    ensure_collection(client)
    _state.initialise(client, StubEmbedder(), tmp_path)
    yield


client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status() -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_includes_version() -> None:
    data = client.get("/health").json()
    assert "version" in data


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


def test_ingest_pdf_returns_200():
    pdf = _make_pdf(1)
    res = client.post("/ingest", files={"file": ("report.pdf", pdf, "application/pdf")})
    assert res.status_code == 200


def test_ingest_returns_doc_id_from_filename():
    pdf = _make_pdf(1)
    data = client.post(
        "/ingest", files={"file": ("annual_report.pdf", pdf, "application/pdf")}
    ).json()
    assert data["doc_id"] == "annual_report"


def test_ingest_returns_correct_page_count():
    pdf = _make_pdf(3)
    data = client.post(
        "/ingest", files={"file": ("deck.pdf", pdf, "application/pdf")}
    ).json()
    assert data["pages"] == 3


def test_ingest_single_page_pdf():
    pdf = _make_pdf(1)
    data = client.post(
        "/ingest", files={"file": ("one.pdf", pdf, "application/pdf")}
    ).json()
    assert data["pages"] == 1


def test_ingest_non_pdf_returns_422():
    res = client.post(
        "/ingest", files={"file": ("doc.txt", b"not a pdf", "text/plain")}
    )
    assert res.status_code == 422


def test_ingest_saves_page_images(tmp_path):
    pdf = _make_pdf(2)
    client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    assert (tmp_path / "doc" / "page_0001.png").exists()
    assert (tmp_path / "doc" / "page_0002.png").exists()


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------


def test_retrieve_empty_store_returns_empty():
    data = client.post("/retrieve", json={"query": "revenue", "top_k": 3}).json()
    assert data["pages"] == []


def test_retrieve_after_ingest_returns_pages():
    pdf = _make_pdf(2)
    client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    pages = client.post("/retrieve", json={"query": "test query", "top_k": 2}).json()[
        "pages"
    ]
    assert len(pages) == 2


def test_retrieve_page_fields():
    pdf = _make_pdf(1)
    client.post("/ingest", files={"file": ("myreport.pdf", pdf, "application/pdf")})
    pages = client.post("/retrieve", json={"query": "anything", "top_k": 1}).json()[
        "pages"
    ]
    assert len(pages) == 1
    p = pages[0]
    assert p["doc_id"] == "myreport"
    assert isinstance(p["page_num"], int)
    assert isinstance(p["score"], float)


def test_retrieve_respects_top_k():
    pdf = _make_pdf(5)
    client.post("/ingest", files={"file": ("big.pdf", pdf, "application/pdf")})
    pages = client.post("/retrieve", json={"query": "anything", "top_k": 3}).json()[
        "pages"
    ]
    assert len(pages) == 3


def test_retrieve_scores_descending():
    pdf = _make_pdf(4)
    client.post("/ingest", files={"file": ("scored.pdf", pdf, "application/pdf")})
    pages = client.post("/retrieve", json={"query": "data", "top_k": 4}).json()["pages"]
    scores = [p["score"] for p in pages]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


def test_ask_empty_store_returns_no_pages():
    data = client.post("/ask", json={"query": "revenue?", "top_k": 3}).json()
    assert data["retrieved_pages"] == []
    assert "index" in data["answer"].lower() or "ingest" in data["answer"].lower()


def test_ask_after_ingest_returns_retrieved_pages():
    with patch(
        "facsimile.answer.synthesizer.synthesize", return_value=_STUB_ANSWER
    ):
        pdf = _make_pdf(2)
        client.post("/ingest", files={"file": ("annual.pdf", pdf, "application/pdf")})
        data = client.post("/ask", json={"query": "revenue", "top_k": 2}).json()
    assert len(data["retrieved_pages"]) == 2


def test_ask_returns_answer_text():
    with patch(
        "facsimile.answer.synthesizer.synthesize", return_value=_STUB_ANSWER
    ):
        pdf = _make_pdf(1)
        client.post("/ingest", files={"file": ("q.pdf", pdf, "application/pdf")})
        data = client.post("/ask", json={"query": "anything", "top_k": 1}).json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_ask_returns_cited_pages():
    stub = SynthAnswer(text="[Page 1] has the answer.", cited_pages=[1], model="stub")
    with patch("facsimile.answer.synthesizer.synthesize", return_value=stub):
        pdf = _make_pdf(1)
        client.post("/ingest", files={"file": ("x.pdf", pdf, "application/pdf")})
        data = client.post("/ask", json={"query": "data", "top_k": 1}).json()
    assert 1 in data["cited_pages"]


def test_ask_returns_model_field():
    with patch(
        "facsimile.answer.synthesizer.synthesize", return_value=_STUB_ANSWER
    ):
        pdf = _make_pdf(1)
        client.post("/ingest", files={"file": ("m.pdf", pdf, "application/pdf")})
        data = client.post("/ask", json={"query": "anything", "top_k": 1}).json()
    assert data["model"] == "stub"


def test_ask_retrieved_pages_have_score():
    with patch(
        "facsimile.answer.synthesizer.synthesize", return_value=_STUB_ANSWER
    ):
        pdf = _make_pdf(2)
        client.post("/ingest", files={"file": ("s.pdf", pdf, "application/pdf")})
        data = client.post("/ask", json={"query": "anything", "top_k": 2}).json()
    assert all(isinstance(p["score"], float) for p in data["retrieved_pages"])


# ---------------------------------------------------------------------------
# GET /pages/{doc_id}/{page_num}
# ---------------------------------------------------------------------------


def test_pages_missing_doc_returns_404():
    res = client.get("/pages/nonexistent_doc/1")
    assert res.status_code == 404


def test_pages_missing_page_num_returns_404():
    pdf = _make_pdf(1)
    client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    res = client.get("/pages/doc/99")
    assert res.status_code == 404


def test_pages_returns_200():
    pdf = _make_pdf(1)
    client.post("/ingest", files={"file": ("mydoc.pdf", pdf, "application/pdf")})
    res = client.get("/pages/mydoc/1")
    assert res.status_code == 200


def test_pages_content_type_is_png():
    pdf = _make_pdf(1)
    client.post("/ingest", files={"file": ("mydoc.pdf", pdf, "application/pdf")})
    res = client.get("/pages/mydoc/1")
    assert res.headers["content-type"] == "image/png"


def test_pages_content_is_valid_image():
    pdf = _make_pdf(1)
    client.post("/ingest", files={"file": ("img.pdf", pdf, "application/pdf")})
    res = client.get("/pages/img/1")
    img = Image.open(io.BytesIO(res.content))
    assert img.size[0] > 0 and img.size[1] > 0


# ---------------------------------------------------------------------------
# GET /viewer
# ---------------------------------------------------------------------------


def test_viewer_returns_200():
    res = client.get("/viewer")
    assert res.status_code == 200


def test_viewer_content_type_is_html():
    res = client.get("/viewer")
    assert "text/html" in res.headers["content-type"]


def test_viewer_contains_ask_endpoint_reference():
    res = client.get("/viewer")
    assert "/ask" in res.text


def test_viewer_contains_pages_endpoint_reference():
    res = client.get("/viewer")
    assert "/pages" in res.text


def test_viewer_html_has_query_input():
    res = client.get("/viewer")
    assert "query-input" in res.text
