# facsimile

Visual document RAG using late-interaction page-image embeddings.

**Problem:** Text extraction destroys exactly what carries meaning in real business documents — a table's structure, a chart, a form's layout. `facsimile` retrieves on the visual document directly, so tables and charts survive end to end.

**Approach:**
- Each PDF page is rasterized to an image (no OCR, no text extraction).
- A ColPali/ColQwen-style visual encoder embeds each page as a set of patch vectors.
- A Qdrant multivector index stores the patch vectors; MAX_SIM late-interaction scoring ranks pages at query time.
- Retrieved page images are fed to a vision LLM, which reads the table or chart as rendered and cites the page.

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Rasterization | PyMuPDF |
| Visual encoder | ColPali / ColQwen2 (`colpali-engine`) |
| Vector index | Qdrant (multivector, MAX_SIM) |
| API | FastAPI |

## Status

M0 scaffold complete: rasterization and multivector store implemented, CI green.
See [PROGRESS.md](PROGRESS.md) for the full milestone roadmap.

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest
```

To run the API:

```bash
uvicorn facsimile.api.main:app --reload
```

## Benchmarks

_Coming in M4: visual vs text pipeline on table/chart questions._
