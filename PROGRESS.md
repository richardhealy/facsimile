# Progress

## Implementation

- [x] **M0 — Scaffold**: project structure, PDF rasterization (`ingest/rasterize.py`), Qdrant multivector store (`index/store.py`), CI (GitHub Actions), 18 tests passing.
- [x] **M1 — Visual embedding**: `Embedder` protocol, `StubEmbedder` (tests/CI), `ColPaliEmbedder` + `ColQwenEmbedder` (optional `embed` extra); `ingest_pdf` / `ingest_and_store` pipeline; 35 tests passing.
- [x] **M2 — Retrieval**: late-interaction MaxSim scoring, top-page selection via stored embeddings. `embed_queries` on all embedders; `Retriever` class ties query embedding → Qdrant MaxSim → ranked `RetrievedPage` list; 56 tests passing.
- [ ] **M3 — Vision synthesis**: vision-LLM answers from retrieved page images with page citations.
- [ ] **M4 — Benchmark**: table/chart question set, visual-vs-text comparison numbers.
- [ ] **M5 — Ship**: FastAPI endpoints, page-highlight viewer, README benchmark, release.

## Documentation

- [ ] Inline doc comments across the public surface.
- [ ] API reference (OpenAPI spec + `docs/api.md`).
- [ ] Architecture dossier (`docs/architecture.md`).
- [ ] Integration guide (`docs/integration.md`).
- [ ] Usage and how-to guides; `docs/` index; final README pass.
