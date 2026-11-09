"""FastAPI application for facsimile.

Endpoints
---------
``GET  /health``                    Liveness check.
``POST /ingest``                    Upload a PDF and index its pages visually.
``POST /retrieve``                  Query the index; return top matching pages.
``POST /ask``                       Retrieve pages and synthesize a grounded answer.
``GET  /pages/{doc_id}/{page_num}`` Serve a rasterized page image as PNG.
``GET  /viewer``                    Interactive page-highlight viewer (HTML).

Configuration
-------------
Read from the environment at startup:

``QDRANT_URL``              Qdrant server URL; unset means in-memory Qdrant.
``DATA_DIR``                Where page PNGs are written (``facsimile_data``).
``FACSIMILE_EMBEDDER``      ``stub`` (default), ``colpali`` or ``colqwen``.
``FACSIMILE_EMBED_MODEL``   Hugging Face checkpoint; unset keeps the default.
``FACSIMILE_EMBED_DEVICE``  Torch device for the model (``cpu``).

With the default ``stub`` embedder every vector is random, so retrieval
returns arbitrary pages; startup logs a warning saying so.  An unknown
embedder name stops startup with a ``ValueError``, and ``colpali`` or
``colqwen`` without ``pip install facsimile[embed]`` with an ``ImportError``.

State
-----
The application holds a module-level :class:`_State` object that is
populated on startup via :func:`_State.initialise`.  Tests can call
``_state.initialise(...)`` directly to inject an in-memory Qdrant client and
a :class:`~facsimile.ingest.embed.StubEmbedder` without triggering the
lifespan.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from qdrant_client import QdrantClient

from facsimile.answer import synthesizer as _synthesizer
from facsimile.index.store import ensure_collection, get_client, get_memory_client
from facsimile.ingest.embed import Embedder, StubEmbedder, make_embedder
from facsimile.ingest.pipeline import ingest_and_store
from facsimile.ingest.rasterize import rasterize_bytes
from facsimile.retrieve.retriever import Retriever
from facsimile.viewer.app import viewer_html

_DEFAULT_DATA_DIR = Path("facsimile_data")

_log = logging.getLogger("facsimile.api")


class _State:
    """Mutable application state shared across requests."""

    def __init__(self) -> None:
        self.client: QdrantClient | None = None
        self.embedder: Embedder | None = None
        self.data_dir: Path = _DEFAULT_DATA_DIR
        self.retriever: Retriever | None = None

    def initialise(
        self,
        client: QdrantClient,
        embedder: Embedder,
        data_dir: Path,
    ) -> None:
        """Wire up all components and ensure the Qdrant collection exists."""
        self.client = client
        self.embedder = embedder
        self.data_dir = data_dir
        ensure_collection(client)
        self.retriever = Retriever(client, embedder)


_state = _State()


def _embedder_from_env() -> Embedder:
    """Build the embedder named by ``FACSIMILE_EMBEDDER`` (``stub`` when unset)."""
    name = os.environ.get("FACSIMILE_EMBEDDER") or "stub"
    try:
        embedder = make_embedder(
            name,
            model_name=os.environ.get("FACSIMILE_EMBED_MODEL") or None,
            device=os.environ.get("FACSIMILE_EMBED_DEVICE") or "cpu",
        )
    except ValueError as exc:
        raise ValueError(
            f"Bad embedder setting: {exc}. Check FACSIMILE_EMBEDDER and "
            "FACSIMILE_EMBED_MODEL."
        ) from exc
    if isinstance(embedder, StubEmbedder):
        setting = (
            f"FACSIMILE_EMBEDDER is {name!r}"
            if os.environ.get("FACSIMILE_EMBEDDER")
            else "FACSIMILE_EMBEDDER is not set"
        )
        _log.warning(
            "%s, so the API is using StubEmbedder: page and query vectors are "
            "random, and /retrieve and /ask return arbitrary pages. Set "
            "FACSIMILE_EMBEDDER=colpali or colqwen (after pip install -e '.[embed]') "
            "to retrieve by content.",
            setting,
        )
    return embedder


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    qdrant_url = os.environ.get("QDRANT_URL")
    client = get_client(qdrant_url) if qdrant_url else get_memory_client()
    embedder = _embedder_from_env()
    data_dir = Path(os.environ.get("DATA_DIR", str(_DEFAULT_DATA_DIR)))
    data_dir.mkdir(parents=True, exist_ok=True)
    _state.initialise(client, embedder, data_dir)
    yield


app = FastAPI(
    title="facsimile",
    description="Visual document RAG with late-interaction page embeddings",
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Response returned by ``POST /ingest``."""

    doc_id: str
    """Stable document identifier derived from the uploaded filename stem."""
    pages: int
    """Number of pages rasterized, embedded, and indexed."""


class PageInfo(BaseModel):
    """Metadata for one retrieved page, included in retrieve and ask responses."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float
    """MaxSim late-interaction relevance score (higher = more relevant)."""


class RetrieveRequest(BaseModel):
    """Request body for ``POST /retrieve``."""

    query: str
    """Natural-language question or search string."""
    top_k: int = 5
    """Maximum number of pages to return."""


class RetrieveResponse(BaseModel):
    """Response returned by ``POST /retrieve``."""

    pages: list[PageInfo]
    """Retrieved pages sorted by descending relevance score."""


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    query: str
    """Natural-language question to answer from the indexed documents."""
    top_k: int = 3
    """Number of pages to retrieve and pass to the vision-LLM."""


class AskResponse(BaseModel):
    """Response returned by ``POST /ask``."""

    answer: str
    """Vision-LLM response text with ``[Page N]`` citation markers."""
    cited_pages: list[int]
    """Page numbers cited in the answer, in ascending order."""
    model: str
    """Claude model ID used for synthesis."""
    retrieved_pages: list[PageInfo]
    """Pages passed to the vision-LLM, for display in the viewer."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _page_image_path(data_dir: Path, doc_id: str, page_num: int) -> Path:
    """Canonical filesystem path for a rasterized page PNG."""
    return data_dir / doc_id / f"page_{page_num:04d}.png"


def _is_dots(doc_id: str) -> bool:
    """True for ``.`` and ``..``, which name DATA_DIR or its parent, not a document."""
    return doc_id in {".", ".."}


_PAGE_IMAGE_NAME = re.compile(r"page_(\d+)\.png")


def _remove_stale_page_images(page_dir: Path, page_count: int) -> None:
    """Delete page PNGs in *page_dir* numbered above *page_count*.

    These are left over from an earlier, longer version of the document.
    Only files named ``page_<number>.png`` are touched.
    """
    for path in page_dir.glob("page_*.png"):
        match = _PAGE_IMAGE_NAME.fullmatch(path.name)
        if match and int(match.group(1)) > page_count:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF; rasterize, embed, and index every page.

    The page images are saved under ``{DATA_DIR}/{doc_id}/`` so they can be
    served by ``GET /pages/{doc_id}/{page_num}`` and displayed in the viewer.
    Re-ingesting a ``doc_id`` replaces its pages: once the new version is
    indexed, page images numbered beyond its last page are deleted.

    Returns the ``doc_id`` (derived from the filename stem) and the number of
    pages indexed.
    """
    if _state.client is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    doc_id = Path(filename).stem
    if _is_dots(doc_id):
        # Path("...pdf").stem is ".." (and "..pdf" gives "."), which would put
        # the page folder outside DATA_DIR or make DATA_DIR itself the folder.
        raise HTTPException(
            status_code=422,
            detail="The filename needs a name before .pdf, not only dots",
        )

    pdf_bytes = await file.read()

    page_dir = _state.data_dir / doc_id
    page_dir.mkdir(parents=True, exist_ok=True)

    page_images = rasterize_bytes(pdf_bytes)
    for i, img in enumerate(page_images):
        img.save(page_dir / f"page_{i + 1:04d}.png", format="PNG")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        ingest_and_store(tmp_path, _state.embedder, _state.client, doc_id=doc_id)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Only after the index has dropped the extra pages, so a failed re-ingest
    # never leaves indexed pages without their images.
    _remove_stale_page_images(page_dir, len(page_images))

    return IngestResponse(doc_id=doc_id, pages=len(page_images))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Embed *query* and return the *top_k* most relevant indexed pages."""
    if _state.retriever is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    pages = _state.retriever.retrieve(req.query, top_k=req.top_k)
    return RetrieveResponse(
        pages=[
            PageInfo(
                doc_id=p.doc_id,
                page_num=p.page_num,
                doc_path=p.doc_path,
                score=p.score,
            )
            for p in pages
        ]
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Retrieve relevant pages and synthesize a vision-LLM grounded answer.

    Retrieved page images are loaded from disk (written by ``POST /ingest``)
    and passed to :func:`~facsimile.answer.synthesizer.synthesize`.  The
    response includes the answer text, the page numbers cited by the model,
    and the full list of retrieved pages for display in the viewer.
    """
    if _state.retriever is None:
        raise HTTPException(status_code=503, detail="Store not initialised")

    pages = _state.retriever.retrieve(req.query, top_k=req.top_k)
    page_infos = [
        PageInfo(doc_id=p.doc_id, page_num=p.page_num, doc_path=p.doc_path, score=p.score)
        for p in pages
    ]

    if not pages:
        return AskResponse(
            answer="No pages are indexed. Ingest a document first.",
            cited_pages=[],
            model="none",
            retrieved_pages=[],
        )

    page_images = [
        (p.page_num, _page_image_path(_state.data_dir, p.doc_id, p.page_num))
        for p in pages
        if _page_image_path(_state.data_dir, p.doc_id, p.page_num).exists()
    ]

    if not page_images:
        return AskResponse(
            answer="Page images not found on disk — please re-ingest the document.",
            cited_pages=[],
            model="none",
            retrieved_pages=page_infos,
        )

    answer = _synthesizer.synthesize(req.query, page_images)
    return AskResponse(
        answer=answer.text,
        cited_pages=answer.cited_pages,
        model=answer.model,
        retrieved_pages=page_infos,
    )


@app.get("/pages/{doc_id}/{page_num}")
def get_page(doc_id: str, page_num: int) -> Response:
    """Serve a rasterized page image as PNG."""
    if _state.data_dir is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    img_path = _page_image_path(_state.data_dir, doc_id, page_num)
    if _is_dots(doc_id) or not img_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} of '{doc_id}' not found",
        )
    return Response(content=img_path.read_bytes(), media_type="image/png")


@app.get("/viewer", response_class=HTMLResponse)
def viewer() -> HTMLResponse:
    """Serve the interactive page-highlight viewer."""
    return HTMLResponse(content=viewer_html())
