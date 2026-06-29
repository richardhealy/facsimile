# Progress

## Implementation

- [x] **M0 — Scaffold**: project structure, PDF rasterization (`ingest/rasterize.py`), Qdrant multivector store (`index/store.py`), CI (GitHub Actions), 18 tests passing.
- [ ] **M1 — Visual embedding**: late-interaction page embeddings (ColPali/ColQwen) indexed per page.
- [ ] **M2 — Retrieval**: late-interaction MaxSim scoring, top-page selection via stored embeddings.
- [ ] **M3 — Vision synthesis**: vision-LLM answers from retrieved page images with page citations.
- [ ] **M4 — Benchmark**: table/chart question set, visual-vs-text comparison numbers.
- [ ] **M5 — Ship**: FastAPI endpoints, page-highlight viewer, README benchmark, release.

## Documentation

- [ ] Inline doc comments across the public surface.
- [ ] API reference (OpenAPI spec + `docs/api.md`).
- [ ] Architecture dossier (`docs/architecture.md`).
- [ ] Integration guide (`docs/integration.md`).
- [ ] Usage and how-to guides; `docs/` index; final README pass.
