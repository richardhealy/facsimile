"""FastAPI application for facsimile — visual document RAG.

Endpoints
---------
GET  /health                        liveness check
POST /ingest                        upload a PDF; rasterize, embed, and store
POST /query                         retrieve top-k pages; optionally synthesize
GET  /pages/{doc_id}/{page_num}     serve the rasterized PNG for a stored page
GET  /                              single-page viewer UI
"""

from __future__ import annotations

import hashlib
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from facsimile.index.store import ensure_collection, get_memory_client, upsert_pages
from facsimile.ingest.embed import Embedder, StubEmbedder
from facsimile.ingest.pipeline import ingest_pdf
from facsimile.ingest.rasterize import rasterize_pdf
from facsimile.retrieve.retriever import Retriever


# ── application state ──────────────────────────────────────────────────────────

_qdrant: QdrantClient | None = None
_embedder: Embedder | None = None
_image_dir: Path | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant, _embedder, _image_dir
    _qdrant = get_memory_client()
    ensure_collection(_qdrant)
    _embedder = StubEmbedder()
    _image_dir = Path(tempfile.mkdtemp(prefix="facsimile_"))
    yield
    _qdrant = None
    _embedder = None
    _image_dir = None


def _get_qdrant() -> QdrantClient:
    if _qdrant is None:
        raise HTTPException(status_code=503, detail="Service starting")
    return _qdrant


def _get_embedder() -> Embedder:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Service starting")
    return _embedder


def _get_image_dir() -> Path:
    if _image_dir is None:
        raise HTTPException(status_code=503, detail="Service starting")
    return _image_dir


# ── response models ────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    doc_id: str
    page_count: int


class PageInfo(BaseModel):
    doc_id: str
    page_num: int
    score: float
    image_url: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    synthesize: bool = False


class QueryResponse(BaseModel):
    question: str
    pages: list[PageInfo]
    answer: str | None = None
    cited_pages: list[int] = []
    model: str | None = None


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="facsimile",
    description="Visual document RAG with late-interaction page embeddings",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    client: QdrantClient = Depends(_get_qdrant),
    embedder: Embedder = Depends(_get_embedder),
    image_dir: Path = Depends(_get_image_dir),
) -> IngestResponse:
    """Upload a PDF, rasterize each page, embed with the visual encoder, and store in Qdrant.

    The doc_id is a 16-character SHA-256 prefix of the file content, so
    re-ingesting the same document is idempotent.
    """
    data = await file.read()
    doc_id = hashlib.sha256(data).hexdigest()[:16]

    pdf_path = image_dir / f"{doc_id}.pdf"
    pdf_path.write_bytes(data)

    records = ingest_pdf(pdf_path, embedder, doc_id=doc_id)
    upsert_pages(client, records)

    images = rasterize_pdf(pdf_path)
    for i, img in enumerate(images):
        img.save(image_dir / f"{doc_id}_p{i + 1}.png", format="PNG")

    return IngestResponse(doc_id=doc_id, page_count=len(records))


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    client: QdrantClient = Depends(_get_qdrant),
    embedder: Embedder = Depends(_get_embedder),
    image_dir: Path = Depends(_get_image_dir),
) -> QueryResponse:
    """Retrieve the top-k most relevant pages for a question.

    Set ``synthesize=true`` to also feed the page images to a Claude vision
    model and receive a grounded answer with ``[Page N]`` citations.  Requires
    ``ANTHROPIC_API_KEY`` to be set in the environment.
    """
    retriever = Retriever(client, embedder)
    pages = retriever.retrieve(req.question, top_k=req.top_k)

    page_infos = [
        PageInfo(
            doc_id=p.doc_id,
            page_num=p.page_num,
            score=p.score,
            image_url=f"/pages/{p.doc_id}/{p.page_num}",
        )
        for p in pages
    ]

    answer_text: str | None = None
    cited: list[int] = []
    model_used: str | None = None

    if req.synthesize and pages:
        from facsimile.answer.synthesizer import synthesize

        page_images = [
            (p.page_num, image_dir / f"{p.doc_id}_p{p.page_num}.png")
            for p in pages
        ]
        page_images = [(n, path) for n, path in page_images if path.exists()]
        if page_images:
            ans = synthesize(req.question, page_images)
            answer_text = ans.text
            cited = ans.cited_pages
            model_used = ans.model

    return QueryResponse(
        question=req.question,
        pages=page_infos,
        answer=answer_text,
        cited_pages=cited,
        model=model_used,
    )


@app.get("/pages/{doc_id}/{page_num}")
def get_page(
    doc_id: str,
    page_num: int,
    image_dir: Path = Depends(_get_image_dir),
) -> FileResponse:
    """Serve the rasterized PNG image for a stored page."""
    page_path = image_dir / f"{doc_id}_p{page_num}.png"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(page_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def viewer() -> str:
    """Serve the single-page viewer UI."""
    from facsimile.viewer.app import viewer_html
    return viewer_html()
