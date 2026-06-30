# Progress

## Implementation

- [x] **M0 — Scaffold**: project structure, PDF rasterization (`ingest/rasterize.py`), Qdrant multivector store (`index/store.py`), CI (GitHub Actions), 18 tests passing.
- [x] **M1 — Visual embedding**: `Embedder` protocol, `StubEmbedder` (tests/CI), `ColPaliEmbedder` + `ColQwenEmbedder` (optional `embed` extra); `ingest_pdf` / `ingest_and_store` pipeline; 35 tests passing.
- [x] **M2 — Retrieval**: `embed_queries` on Embedder protocol; `Retriever` class (query → embed → Qdrant MaxSim → `RetrievedPage` list); 14 new tests, 59 total.
- [x] **M3 — Vision synthesis**: `synthesize()` feeds retrieved page images to a Claude vision model; returns `Answer` with grounded text and `[Page N]`-cited page numbers; 21 new tests, 80 total.
- [x] **M4 — Benchmark**: `BenchQuestion` schema + `load_questions`; `text_retrieve` (keyword-overlap text pipeline as baseline); `run_benchmark` (visual vs text, per-question results); `summarize_results` (accuracy by pipeline and question type); `bench/sample_questions.json`; 45 new tests, 125 total.
- [x] **M5 — Ship**: FastAPI endpoints (`/ingest`, `/retrieve`, `/ask`, `/pages`, `/viewer`), page-highlight viewer (HTML SPA with cited-page border), `python-multipart` dependency; 27 new tests, 152 total.

## Documentation

- [ ] Inline doc comments across the public surface.
- [ ] API reference (OpenAPI spec + `docs/api.md`).
- [ ] Architecture dossier (`docs/architecture.md`).
- [ ] Integration guide (`docs/integration.md`).
- [ ] Usage and how-to guides; `docs/` index; final README pass.
