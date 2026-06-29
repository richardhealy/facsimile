# facsimile

**Visual document RAG using late-interaction page-image embeddings.**

facsimile retrieves over pages as images — tables, charts, and form layouts survive intact instead of being flattened by text extraction. It uses a ColPali-style multi-vector late-interaction approach: each page is embedded as a set of patch vectors, and retrieval scores queries against each patch (MaxSim), achieving cross-encoder-grade accuracy without a separate OCR pipeline.

## The problem it solves

Text-extraction pipelines mangle exactly what carries meaning in real business documents: a table's row/column structure, a bar chart's axis labels, a form's visual layout. facsimile embeds the rendered page image directly, so the answer is read off pixels — not reconstructed from broken plaintext.

## Architecture

```
PDF → rasterize → page images → visual embeddings (ColPali) → Qdrant multivector index
Query → query embeddings → late-interaction scoring → top pages → vision-LLM synthesis → answer + citations
```

```
facsimile/
  ingest/    # PDF → page images (pypdfium2)
  index/     # Qdrant multivector store, per-page vectors
  retrieve/  # late-interaction scoring, top-page selection
  answer/    # vision-LLM synthesis over page images + citations
  bench/     # table/chart question set, visual-vs-text comparison
  api/       # FastAPI service
viewer/      # page-highlight UI
```

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Scaffold: rasterization, multivector store, CI green | ✅ Done |
| M1 | Visual embedding: late-interaction page embeddings indexed | ☐ |
| M2 | Retrieval: late-interaction scoring, top-page selection | ☐ |
| M3 | Vision synthesis: vision-LLM answers over pages + citations | ☐ |
| M4 | Benchmark: table/chart set, visual-vs-text numbers | ☐ |
| M5 | Ship: viewer with page highlight, README, release | ☐ |

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Requires Python ≥ 3.11.

## Stack

- **PDF rasterization:** pypdfium2 (zero system dependencies, pure pdfium bindings)
- **Vector store:** Qdrant with multivector (MaxSim) collections
- **Embeddings:** ColPali / ColQwen (planned — M1)
- **Synthesis:** vision-capable LLM via API (planned — M3)
- **API:** FastAPI + uvicorn
