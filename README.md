# facsimile

Visual document RAG using late-interaction page-image embeddings.

**Problem:** Text extraction destroys exactly what carries meaning in real business documents — a table's structure, a chart's values, a form's layout. `facsimile` skips the OCR step entirely and retrieves over page images directly.

**How it works:**
1. Each PDF page is rasterized to an image.
2. A visual encoder (ColPali/ColQwen) turns each page into a set of patch-level vectors.
3. Patch vectors are stored in Qdrant with multivector/MaxSim support — late-interaction scoring at query time.
4. The top-matching pages are fed to a vision LLM which reads the table or chart as rendered and cites the page.

## Status

M0 (Scaffold), M1 (Visual embedding), and M2 (Retrieval) complete. See [PROGRESS.md](PROGRESS.md) for the roadmap.

## Requirements

- Python ≥ 3.11
- [Qdrant](https://qdrant.tech/) (Docker: `docker run -p 6333:6333 qdrant/qdrant`)

## Install

```bash
# Core + tests
pip install -e ".[dev]"

# With real ColPali/ColQwen inference (requires GPU for practical speed)
pip install -e ".[embed]"
```

## Run tests

```bash
pytest
```

## Architecture

```
facsimile/
  ingest/    # PDF → page images (rasterize.py)
  index/     # Qdrant multivector store (store.py)
  retrieve/  # late-interaction MaxSim scoring
  answer/    # vision-LLM synthesis + citations
  bench/     # table/chart benchmark vs text pipeline
  api/       # FastAPI application
  viewer/    # page-highlight viewer
```
