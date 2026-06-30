"""FastAPI application for facsimile — visual document RAG."""

from __future__ import annotations

import base64
import io
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from facsimile.index.store import ensure_collection, get_memory_client
from facsimile.ingest.embed import StubEmbedder
from facsimile.ingest.pipeline import ingest_and_store
from facsimile.ingest.rasterize import rasterize_bytes
from facsimile.retrieve.retriever import Retriever

_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Singleton state — tests override via app.dependency_overrides
# ---------------------------------------------------------------------------

_default_client = None
_default_embedder = None
_default_pdf_store: dict[str, bytes] = {}


def get_qdrant_client():
    """Return (and lazily create) the shared Qdrant client."""
    global _default_client
    if _default_client is None:
        _default_client = get_memory_client()
        ensure_collection(_default_client)
    return _default_client


def get_embedder():
    """Return (and lazily create) the shared Embedder instance."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = StubEmbedder()
    return _default_embedder


def get_retriever(
    client=Depends(get_qdrant_client),
    embedder=Depends(get_embedder),
) -> Retriever:
    """Build a Retriever wired to the current client and embedder."""
    return Retriever(client, embedder)


def get_pdf_store() -> dict[str, bytes]:
    """Return the in-process PDF content store (doc_id → raw bytes)."""
    return _default_pdf_store


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="facsimile",
    description=(
        "Visual document RAG with late-interaction page-image embeddings. "
        "Retrieve over page images — tables, charts, and layout survive."
    ),
    version=_VERSION,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health() -> dict:
    """Return service liveness and version."""
    return {"status": "ok", "version": _VERSION}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    doc_id: str
    page_count: int
    patches_per_page: list[int]


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest(
    file: Annotated[UploadFile, File(description="PDF document to ingest")],
    doc_id: Annotated[
        str | None,
        Form(description="Stable document identifier; defaults to the filename stem"),
    ] = None,
    dpi: Annotated[
        int,
        Form(description="Rasterization DPI — higher means larger patches, slower embedding"),
    ] = 150,
    client=Depends(get_qdrant_client),
    embedder=Depends(get_embedder),
    pdf_store: dict = Depends(get_pdf_store),
) -> IngestResponse:
    """Rasterize every page of a PDF, embed with the visual encoder, and store in Qdrant."""
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    resolved_doc_id = doc_id or Path(filename).stem

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        records = ingest_and_store(
            tmp_path, embedder, client, doc_id=resolved_doc_id, dpi=dpi
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    pdf_store[resolved_doc_id] = pdf_bytes

    return IngestResponse(
        doc_id=resolved_doc_id,
        page_count=len(records),
        patches_per_page=[len(r.vectors) for r in records],
    )


# ---------------------------------------------------------------------------
# Query (retrieve pages)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class PageResult(BaseModel):
    doc_id: str
    page_num: int
    doc_path: str
    score: float


class QueryResponse(BaseModel):
    results: list[PageResult]


@app.post("/query", response_model=QueryResponse, tags=["search"])
def query(
    req: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
) -> QueryResponse:
    """Retrieve the top-k most relevant pages for a natural-language query using late-interaction MaxSim."""
    pages = retriever.retrieve(req.query, top_k=req.top_k)
    return QueryResponse(
        results=[
            PageResult(
                doc_id=p.doc_id,
                page_num=p.page_num,
                doc_path=p.doc_path,
                score=p.score,
            )
            for p in pages
        ]
    )


# ---------------------------------------------------------------------------
# Answer (vision synthesis)
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    model: str = "claude-opus-4-8"


class AnswerResponse(BaseModel):
    text: str
    cited_pages: list[int]
    model: str


@app.post("/answer", response_model=AnswerResponse, tags=["answer"])
def answer_endpoint(
    req: AnswerRequest,
    retriever: Retriever = Depends(get_retriever),
    pdf_store: dict = Depends(get_pdf_store),
) -> AnswerResponse:
    """Retrieve top-k pages and synthesize a vision-grounded answer via Claude, with page citations."""
    pages = retriever.retrieve(req.query, top_k=req.top_k)
    if not pages:
        raise HTTPException(
            status_code=404,
            detail="No pages indexed. Ingest a document first via POST /ingest.",
        )

    pages_by_doc: dict[str, list] = defaultdict(list)
    for p in pages:
        pages_by_doc[p.doc_id].append(p)

    page_image_pairs = []
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for doc_id, doc_pages in pages_by_doc.items():
            pdf_bytes = pdf_store.get(doc_id)
            if pdf_bytes is None:
                continue
            images = rasterize_bytes(pdf_bytes)
            for p in doc_pages:
                idx = p.page_num - 1
                if 0 <= idx < len(images):
                    img_path = tmpdir / f"{doc_id}_{p.page_num}.png"
                    images[idx].save(img_path, format="PNG")
                    page_image_pairs.append((p.page_num, img_path))

        if not page_image_pairs:
            raise HTTPException(
                status_code=422,
                detail="No page images available. Re-ingest via POST /ingest.",
            )

        from facsimile.answer.synthesizer import synthesize

        try:
            ans = synthesize(
                question=req.query,
                page_images=page_image_pairs,
                model=req.model,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    return AnswerResponse(text=ans.text, cited_pages=ans.cited_pages, model=ans.model)


# ---------------------------------------------------------------------------
# Page image
# ---------------------------------------------------------------------------

@app.get(
    "/pages/{doc_id}/{page_num}",
    tags=["documents"],
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def page_image(
    doc_id: str,
    page_num: int,
    dpi: int = 150,
    pdf_store: dict = Depends(get_pdf_store),
) -> Response:
    """Return the rasterized PNG for a single page."""
    pdf_bytes = pdf_store.get(doc_id)
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    images = rasterize_bytes(pdf_bytes, dpi=dpi)
    if page_num < 1 or page_num > len(images):
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} not found in '{doc_id}' ({len(images)} pages)",
        )
    buf = io.BytesIO()
    images[page_num - 1].save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

_VIEWER_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 28px 24px;
    background: #f0f2f5; color: #1a1a2e; min-height: 100vh;
}
header { margin-bottom: 28px; }
h1 { font-size: 2rem; letter-spacing: -0.03em; margin-bottom: 6px; }
.tagline { color: #666; font-size: 0.95rem; line-height: 1.5; }
.search-row { display: flex; gap: 12px; margin-bottom: 28px; }
.search-row input {
    flex: 1; padding: 13px 18px;
    border: 1.5px solid #d1d5db; border-radius: 10px;
    font-size: 16px; outline: none; background: white;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.search-row input:focus {
    border-color: #3d5a80;
    box-shadow: 0 0 0 3px rgba(61,90,128,0.12);
}
.search-row button {
    padding: 13px 30px; background: #3d5a80; color: white;
    border: none; border-radius: 10px; font-size: 15px;
    cursor: pointer; font-weight: 600; white-space: nowrap;
    transition: background 0.15s;
}
.search-row button:hover { background: #2c4a70; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px; margin-bottom: 28px;
}
.card {
    background: white; border-radius: 12px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    transition: transform 0.15s, box-shadow 0.15s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.card.best {
    border: 3px solid #e07a5f;
    box-shadow: 0 4px 16px rgba(224,122,95,0.22);
}
.card img { width: 100%; display: block; border-bottom: 1px solid #f0f0f0; }
.card-info { padding: 13px 16px; }
.badge {
    display: inline-block;
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 2px 8px; border-radius: 20px;
    margin-bottom: 6px;
}
.badge.best { background: #fde8e3; color: #c0392b; }
.badge.other { background: #eef2f7; color: #5a7499; }
.card-title { font-size: 15px; font-weight: 600; margin-bottom: 3px; }
.card-score { font-size: 13px; color: #888; }
.no-results {
    text-align: center; color: #999; padding: 64px 24px;
    background: white; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.no-results code {
    background: #f0f2f5; padding: 2px 7px;
    border-radius: 5px; font-size: 0.9em; color: #555;
}
footer {
    margin-top: 36px; font-size: 13px; color: #bbb;
    border-top: 1px solid #e5e7eb; padding-top: 16px;
}
footer a { color: #3d5a80; text-decoration: none; }
footer code {
    background: #e8ecf0; padding: 1px 6px;
    border-radius: 4px; font-size: 0.88em; color: #444;
}
"""


@app.get("/viewer", response_class=HTMLResponse, tags=["viewer"])
def viewer(
    query: str = "",
    top_k: int = 5,
    retriever: Retriever = Depends(get_retriever),
    pdf_store: dict = Depends(get_pdf_store),
) -> str:
    """Browser UI — enter a query, see the top matching pages with the best result highlighted."""
    body_html = ""

    if query:
        pages = retriever.retrieve(query, top_k=top_k)
        if pages:
            cards_html = ""
            for i, page in enumerate(pages):
                is_best = i == 0
                img_html = ""
                pdf_bytes = pdf_store.get(page.doc_id)
                if pdf_bytes:
                    imgs = rasterize_bytes(pdf_bytes)
                    idx = page.page_num - 1
                    if 0 <= idx < len(imgs):
                        buf = io.BytesIO()
                        imgs[idx].save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        img_html = (
                            f'<img src="data:image/png;base64,{b64}" '
                            f'alt="Page {page.page_num}" loading="lazy">'
                        )
                badge = (
                    '<span class="badge best">#1 best match</span>'
                    if is_best
                    else f'<span class="badge other">#{i+1}</span>'
                )
                card_class = "card best" if is_best else "card"
                cards_html += f"""
<div class="{card_class}">
  {img_html}
  <div class="card-info">
    {badge}
    <div class="card-title">{page.doc_id} — Page {page.page_num}</div>
    <div class="card-score">Score: {page.score:.4f}</div>
  </div>
</div>"""
            body_html = f'<div class="grid">{cards_html}</div>'
        else:
            body_html = (
                '<div class="no-results">'
                "No pages found. Ingest a document first via "
                "<code>POST /ingest</code>."
                "</div>"
            )

    escaped_q = (
        query.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>facsimile — Visual Document Search</title>
<style>{_VIEWER_CSS}</style>
</head>
<body>
<header>
  <h1>facsimile</h1>
  <p class="tagline">
    Visual document RAG &mdash; retrieves over page images so tables and
    charts are never lost to text extraction.
  </p>
</header>
<form method="get" action="/viewer">
  <div class="search-row">
    <input
      type="text" name="query" value="{escaped_q}"
      placeholder="Ask a question about your documents&hellip;"
      autofocus>
    <button type="submit">Search</button>
  </div>
</form>
{body_html}
<footer>
  Ingest: <code>POST /ingest</code> &middot;
  Query: <code>POST /query</code> &middot;
  Answer: <code>POST /answer</code> &middot;
  API docs: <a href="/docs">/docs</a>
</footer>
</body>
</html>"""
