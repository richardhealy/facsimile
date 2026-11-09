"""Tests for the FastAPI application (M5 — full endpoint suite)."""

from __future__ import annotations

import io
import logging
from unittest.mock import patch

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import facsimile.ingest.embed as embed_module
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


def _ingest(name: str, n_pages: int):
    return client.post(
        "/ingest", files={"file": (name, _make_pdf(n_pages), "application/pdf")}
    )


def test_reingest_shorter_pdf_removes_the_old_page_images(tmp_path):
    assert _ingest("doc.pdf", 3).status_code == 200
    assert _ingest("doc.pdf", 1).json()["pages"] == 1
    page_dir = tmp_path / "doc"
    assert sorted(p.name for p in page_dir.iterdir()) == ["page_0001.png"]


def test_reingest_shorter_pdf_stops_serving_the_removed_pages():
    _ingest("doc.pdf", 3)
    _ingest("doc.pdf", 1)
    assert client.get("/pages/doc/1").status_code == 200
    assert client.get("/pages/doc/2").status_code == 404
    assert client.get("/pages/doc/3").status_code == 404


def test_reingest_leaves_other_files_in_the_page_folder(tmp_path):
    _ingest("doc.pdf", 2)
    notes = tmp_path / "doc" / "notes.txt"
    notes.write_text("kept")
    cover = tmp_path / "doc" / "page_cover.png"
    cover.write_bytes(b"kept")
    _ingest("doc.pdf", 1)
    assert notes.read_text() == "kept"
    assert cover.read_bytes() == b"kept"


def test_reingest_leaves_other_documents_alone(tmp_path):
    _ingest("long.pdf", 3)
    _ingest("short.pdf", 1)
    _ingest("short.pdf", 1)
    assert len(list((tmp_path / "long").glob("page_*.png"))) == 3
    assert client.get("/pages/long/3").status_code == 200


def test_failed_reingest_keeps_the_pages_the_index_still_has(tmp_path):
    _ingest("doc.pdf", 3)
    with (
        patch(
            "facsimile.api.main.ingest_and_store",
            side_effect=RuntimeError("embed failed"),
        ),
        pytest.raises(RuntimeError),
    ):
        _ingest("doc.pdf", 1)
    # The index still holds pages 2 and 3, so their images must stay servable.
    assert (tmp_path / "doc" / "page_0003.png").exists()
    assert client.get("/pages/doc/3").status_code == 200


@pytest.mark.parametrize("name", ["...pdf", "..pdf", "...PDF"])
def test_ingest_rejects_a_filename_whose_stem_is_dots(tmp_path, name):
    # Path("...pdf").stem is "..", which would put the page images (and the
    # stale-page cleanup) outside DATA_DIR.
    res = _ingest(name, 1)
    assert res.status_code == 422
    assert "name" in res.json()["detail"]
    assert not list(tmp_path.parent.glob("page_*.png"))
    assert not list(tmp_path.glob("page_*.png"))


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


@pytest.mark.parametrize("doc_id", ["%2E%2E", "%2E"])
def test_pages_does_not_serve_images_outside_a_document_folder(tmp_path, doc_id):
    # A PNG named like a page, above DATA_DIR and in DATA_DIR itself.
    (tmp_path.parent / "page_0001.png").write_bytes(b"outside")
    (tmp_path / "page_0001.png").write_bytes(b"outside")
    res = client.get(f"/pages/{doc_id}/1")
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


# ---------------------------------------------------------------------------
# Startup: picking the embedder from the environment
# ---------------------------------------------------------------------------


class _FakeModelEmbedder:
    """Stands in for ColPaliEmbedder / ColQwenEmbedder without torch or a download."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_images(self, images):
        return [[[0.0] * 128] for _ in images]

    def embed_queries(self, texts):
        return [[[0.0] * 128] for _ in texts]


@pytest.fixture
def startup_env(monkeypatch, tmp_path):
    """Clean environment for running the real lifespan (in-memory Qdrant, tmp DATA_DIR)."""
    for name in ("QDRANT_URL", "FACSIMILE_EMBEDDER", "FACSIMILE_EMBED_MODEL", "FACSIMILE_EMBED_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return monkeypatch


def test_startup_uses_stub_and_warns_when_no_embedder_is_set(startup_env, caplog):
    with caplog.at_level(logging.WARNING, logger="facsimile.api"), TestClient(app):
        assert isinstance(_state.embedder, StubEmbedder)
    assert "FACSIMILE_EMBEDDER is not set" in caplog.text
    assert "vectors are random" in caplog.text


def test_startup_warns_when_the_stub_is_chosen_explicitly(startup_env, caplog):
    startup_env.setenv("FACSIMILE_EMBEDDER", "stub")
    with caplog.at_level(logging.WARNING, logger="facsimile.api"), TestClient(app):
        assert isinstance(_state.embedder, StubEmbedder)
    assert "FACSIMILE_EMBEDDER is 'stub'" in caplog.text


def test_startup_builds_the_embedder_named_in_the_environment(startup_env, caplog):
    startup_env.setattr(embed_module, "ColQwenEmbedder", _FakeModelEmbedder)
    startup_env.setenv("FACSIMILE_EMBEDDER", "colqwen")
    startup_env.setenv("FACSIMILE_EMBED_MODEL", "vidore/colqwen2.5-v0.2")
    startup_env.setenv("FACSIMILE_EMBED_DEVICE", "cuda")
    with caplog.at_level(logging.WARNING, logger="facsimile.api"), TestClient(app) as c:
        assert isinstance(_state.embedder, _FakeModelEmbedder)
        assert _state.embedder.kwargs == {
            "model_name": "vidore/colqwen2.5-v0.2",
            "device": "cuda",
        }
        assert _state.retriever._embedder is _state.embedder
        assert c.get("/health").status_code == 200
    assert "vectors are random" not in caplog.text


def test_startup_fails_on_an_unknown_embedder(startup_env):
    startup_env.setenv("FACSIMILE_EMBEDDER", "colbert")
    with pytest.raises(ValueError, match="FACSIMILE_EMBEDDER"), TestClient(app):
        pass
