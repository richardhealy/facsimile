# facsimile

Visual document RAG using late-interaction page-image embeddings.

**Problem:** Text extraction destroys exactly what carries meaning in real business documents — a table's structure, a chart's values, a form's layout. `facsimile` skips the OCR step entirely and retrieves over page images directly.

**How it works:**
1. Each PDF page is rasterized to an image.
2. A visual encoder (ColPali/ColQwen) turns each page into a set of patch-level vectors.
3. Patch vectors are stored in Qdrant with multivector/MaxSim support — late-interaction scoring at query time.
4. The top-matching pages are fed to a vision LLM which reads the table or chart as rendered and cites the page.

## Status

M0–M5 complete: Scaffold, Visual embedding, Retrieval, Vision synthesis, Benchmark, and Ship. See [PROGRESS.md](PROGRESS.md) for the full roadmap.

## Quickstart

```bash
pip install -e ".[dev]"

# Start the API (in-memory Qdrant; add QDRANT_URL for a persistent server)
uvicorn facsimile.api.main:app --reload

# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@annual_report.pdf"
# → {"doc_id": "annual_report", "pages": 42}

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was total revenue?", "top_k": 3}'
# → {"answer": "Total revenue was $4.2B [Page 7].", "cited_pages": [7], ...}

# Open the visual viewer
open http://localhost:8000/viewer
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/ingest` | Upload a PDF; rasterize, embed, and index every page |
| `POST` | `/retrieve` | Query the index; return top matching pages with scores |
| `POST` | `/ask` | Retrieve pages and synthesize a vision-LLM answer |
| `GET`  | `/pages/{doc_id}/{page_num}` | Serve a rasterized page image as PNG |
| `GET`  | `/viewer` | Interactive page-highlight viewer (HTML) |

### POST /ingest

```json
// multipart/form-data: file=<PDF bytes>
// Response:
{ "doc_id": "annual_report", "pages": 42 }
```

### POST /retrieve

```json
// Request:
{ "query": "What was EBITDA?", "top_k": 5 }
// Response:
{ "pages": [{ "doc_id": "annual_report", "page_num": 7, "score": 0.821, "doc_path": "..." }] }
```

### POST /ask

```json
// Request:
{ "query": "What was total revenue?", "top_k": 3 }
// Response:
{
  "answer": "Total revenue was $4.2B as shown in the income statement [Page 7].",
  "cited_pages": [7],
  "model": "claude-opus-4-8",
  "retrieved_pages": [{ "doc_id": "annual_report", "page_num": 7, "score": 0.821, "doc_path": "..." }]
}
```

## Benchmark

`facsimile` includes a built-in benchmark comparing the visual pipeline against a text-extraction baseline on table and chart questions.

| Metric | Visual pipeline | Text pipeline |
|--------|----------------|---------------|
| Answer accuracy | Expected substring in vision-LLM answer | Expected substring in extracted page text |
| Page accuracy | Correct page cited by vision LLM | Correct page in keyword-ranked top-k |

The text pipeline uses keyword-overlap scoring over raw `pypdfium2` text extraction — the approach most text RAG systems use. The visual pipeline uses late-interaction MaxSim retrieval and vision-LLM synthesis, so it reads tables and charts as rendered rather than as mangled text.

```python
from facsimile.bench import BenchQuestion, run_benchmark, summarize_results
from facsimile.retrieve.retriever import Retriever
from facsimile.ingest.embed import StubEmbedder  # replace with ColPaliEmbedder for real inference
from facsimile.index.store import get_memory_client, ensure_collection

client = get_memory_client()
ensure_collection(client)
retriever = Retriever(client, StubEmbedder())

questions = [
    BenchQuestion("What was total revenue?", "revenue", correct_page=4, question_type="table"),
]
results = run_benchmark(questions, "annual_report.pdf", retriever)
summary = summarize_results(results)
print(f"Visual answer accuracy: {summary.visual_answer_accuracy:.0%}")
print(f"Text answer accuracy:   {summary.text_answer_accuracy:.0%}")
```

A ready-to-use question template is at [`bench/sample_questions.json`](bench/sample_questions.json).

## Requirements

- Python ≥ 3.11
- [Qdrant](https://qdrant.tech/) (Docker: `docker run -p 6333:6333 qdrant/qdrant`) — optional; defaults to in-memory mode

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

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | *(in-memory)* | Qdrant server URL, e.g. `http://localhost:6333` |
| `DATA_DIR` | `facsimile_data` | Directory where rasterized page images are stored |
| `ANTHROPIC_API_KEY` | *(required for /ask)* | API key for the Claude vision model |

## Architecture

```
facsimile/
  ingest/    # PDF → page images (rasterize.py) + visual embeddings (embed.py)
  index/     # Qdrant multivector store (store.py)
  retrieve/  # late-interaction MaxSim scoring (retriever.py, scorer.py)
  answer/    # vision-LLM synthesis + citations (synthesizer.py)
  bench/     # table/chart benchmark vs text pipeline
  api/       # FastAPI application (main.py)
  viewer/    # page-highlight viewer HTML template (app.py)
```
