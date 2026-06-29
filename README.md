# facsimile

Visual document RAG with late-interaction page embeddings.

**The problem it solves:** Text extraction destroys exactly what carries meaning in real business documents — table structure, chart data, form layout. `facsimile` retrieves over page images directly using ColPali-style late-interaction visual embeddings, so nothing is lost between ingestion and synthesis.

## Stack

- **Python 3.11+**
- **Rasterization:** PyMuPDF (no external OCR)
- **Visual embeddings:** ColPali / ColQwen2 (late-interaction, patch-level)
- **Vector store:** Qdrant (multivector, MaxSim)
- **Synthesis:** Vision LLM (Claude) over retrieved page images
- **API:** FastAPI

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

```
facsimile/
  ingest/    # PDF → page images (PyMuPDF)
  index/     # Qdrant multivector store
  retrieve/  # MaxSim late-interaction scoring
  answer/    # Vision-LLM synthesis + page citations
  bench/     # Table/chart benchmark vs text pipeline
  api/       # FastAPI service
  viewer/    # Page-image viewer with retrieval highlights
```

## Milestones

| Milestone | Status |
|-----------|--------|
| M0: Scaffold — rasterization, multivector store, CI | ✅ Done |
| M1: Visual embedding — ColPali page embeddings indexed | ◐ Next |
| M2: Retrieval — late-interaction scoring, top pages | ☐ |
| M3: Vision synthesis — answers + page citations | ☐ |
| M4: Benchmark — visual vs text on tables/charts | ☐ |
| M5: Ship — viewer, full README, release | ☐ |

## Why this matters

A question whose answer is in a table or chart cannot be answered correctly by a text pipeline — the table's structure is destroyed the moment it is flattened into tokens. `facsimile` treats the rendered page as the unit of retrieval, so a vision LLM can read the table or chart exactly as it appears, and cite the page.
