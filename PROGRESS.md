# Progress

## Implementation

- [x] **M0 — Scaffold**: project structure, PDF rasterization (`ingest/rasterize.py`), Qdrant multivector store (`index/store.py`), CI (GitHub Actions), 18 tests passing.
- [x] **M1 — Visual embedding**: `Embedder` protocol, `StubEmbedder` (tests/CI), `ColPaliEmbedder` + `ColQwenEmbedder` (optional `embed` extra); `ingest_pdf` / `ingest_and_store` pipeline; 35 tests passing.
- [x] **M2 — Retrieval**: `embed_queries` on Embedder protocol; `Retriever` class (query → embed → Qdrant MaxSim → `RetrievedPage` list); 14 new tests, 59 total.
- [x] **M3 — Vision synthesis**: `Synthesizer` protocol, `StubSynthesizer` (CI-safe), `ClaudeSynthesizer` (Anthropic vision API, `facsimile[vision]`); `Answer` dataclass with text and cited pages; 16 new tests, 75 total.
- [ ] **M4 — Benchmark**: table/chart question set, visual-vs-text comparison numbers.
- [ ] **M5 — Ship**: FastAPI endpoints, page-highlight viewer, README benchmark, release.

## Documentation

- [ ] Inline doc comments across the public surface.
- [ ] API reference (OpenAPI spec + `docs/api.md`).
- [ ] Architecture dossier (`docs/architecture.md`).
- [ ] Integration guide (`docs/integration.md`).
- [ ] Usage and how-to guides; `docs/` index; final README pass.
