# facsimile

Visual document RAG using late-interaction page-image embeddings.

**Problem:** Text extraction destroys exactly what carries meaning in real business documents — a table's structure, a chart's values, a form's layout. `facsimile` skips the OCR step entirely and retrieves over page images directly.

**How it works:**
1. Each PDF page is rasterized to an image.
2. A visual encoder (ColPali/ColQwen) turns each page into a set of patch-level vectors.
3. Patch vectors are stored in Qdrant with multivector/MaxSim support — late-interaction scoring at query time.
4. The top-matching pages are fed to a vision LLM which reads the table or chart as rendered and cites the page.

## Status

M0–M5 complete: all milestones implemented, 156 tests passing. See [PROGRESS.md](PROGRESS.md) for the full roadmap.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Start the API + viewer
python -c "from facsimile.viewer.app import launch_viewer; launch_viewer(port=8000)"

# Ingest a PDF
curl -F "file=@annual_report.pdf" http://localhost:8000/ingest

# Query (retrieve top pages)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was total revenue?", "top_k": 5}'

# Answer (retrieval + vision synthesis, requires ANTHROPIC_API_KEY)
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What was total revenue?", "top_k": 3}'

# Browser viewer with page-highlight
open http://localhost:8000/viewer
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `POST /ingest` | Upload a PDF — rasterize, embed, index |
| `POST /query` | Retrieve top-k pages by late-interaction MaxSim |
| `POST /answer` | Retrieve + vision-LLM synthesis with page citations |
| `GET /pages/{doc_id}/{page_num}` | Serve a rasterized page as PNG |
| `GET /viewer` | Browser UI with page-highlight |

Interactive docs: `http://localhost:8000/docs`

## Benchmark

`facsimile` includes a built-in benchmark comparing the visual pipeline against a text-extraction baseline on table/chart questions.

| Metric | Visual pipeline | Text pipeline |
|--------|----------------|---------------|
| Answer accuracy | Expected substring in vision-LLM answer | Expected substring in extracted page text |
| Page accuracy | Correct page cited by vision LLM | Correct page in keyword-ranked top-k |

The text pipeline uses keyword-overlap scoring over raw pypdfium2 text extraction — the approach most text RAG systems use. The visual pipeline uses late-interaction MaxSim retrieval and vision-LLM synthesis, so it reads tables and charts as rendered rather than as mangled text.

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
- [Qdrant](https://qdrant.tech/) (Docker: `docker run -p 6333:6333 qdrant/qdrant`), or the built-in in-memory client for demos/tests

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
  ingest/    # PDF → page images (rasterize.py) + embeddings (embed.py, pipeline.py)
  index/     # Qdrant multivector store with MaxSim (store.py)
  retrieve/  # Late-interaction MaxSim retrieval (retriever.py, scorer.py)
  answer/    # Vision-LLM synthesis + citations (synthesizer.py)
  bench/     # Table/chart benchmark vs text pipeline
  api/       # FastAPI application (main.py) — all REST endpoints
  viewer/    # launch_viewer() helper for the browser UI
```

The page image is the atomic unit end to end — nothing about a table or chart is lost between ingestion and synthesis.
