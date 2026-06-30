# facsimile

Visual document RAG using late-interaction page-image embeddings.

**Problem:** Text extraction destroys exactly what carries meaning in real business documents — a table's structure, a chart's values, a form's layout. `facsimile` skips the OCR step entirely and retrieves over page images directly.

**How it works:**
1. Each PDF page is rasterized to an image.
2. A visual encoder (ColPali/ColQwen) turns each page into a set of patch-level vectors.
3. Patch vectors are stored in Qdrant with multivector/MaxSim support — late-interaction scoring at query time.
4. The top-matching pages are fed to a vision LLM which reads the table or chart as rendered and cites the page.

## Status

M0–M5 complete: Scaffold, Visual embedding, Retrieval, Vision synthesis, Benchmark, and Ship (FastAPI + viewer). See [PROGRESS.md](PROGRESS.md) for the full roadmap.

## Benchmark

`facsimile` includes a built-in benchmark that compares the visual pipeline against a text-extraction baseline on table/chart questions.

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

## API

Start the server:

```bash
uvicorn facsimile.api.main:app --reload
```

Open `http://localhost:8000` for the page-highlight viewer, or call the API directly:

```bash
# Ingest a PDF
curl -F file=@annual_report.pdf http://localhost:8000/ingest
# → {"doc_id": "abc123...", "page_count": 42}

# Retrieve top-3 pages
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was total revenue?", "top_k": 3}' | python -m json.tool

# Retrieve + synthesize (needs ANTHROPIC_API_KEY)
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was total revenue?", "top_k": 3, "synthesize": true}' | python -m json.tool

# Fetch a page image
curl http://localhost:8000/pages/<doc_id>/1 -o page1.png
```

Full OpenAPI docs are at `http://localhost:8000/docs`.

## Requirements

- Python ≥ 3.11
- Qdrant runs in-memory by default (no server needed for the demo); use `docker run -p 6333:6333 qdrant/qdrant` for a persistent deployment

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
