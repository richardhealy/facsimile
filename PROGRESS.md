# facsimile — Progress

## Implementation

- [x] **M0: Scaffold** — rasterization, multivector store, CI green
  - [x] PDF → page images (PyMuPDF, `facsimile/ingest/rasterize.py`)
  - [x] Qdrant multivector store with MaxSim config (`facsimile/index/store.py`)
  - [x] Late-interaction MaxSim scorer (`facsimile/retrieve/scorer.py`)
  - [x] FastAPI health endpoint (`facsimile/api/main.py`)
  - [x] Full test suite (28 tests, 100% pass)
  - [x] GitHub Actions CI
- [ ] **M1: Visual embedding** — ColPali page embeddings indexed
- [ ] **M2: Retrieval** — late-interaction scoring, top-page selection
- [ ] **M3: Vision synthesis** — vision-LLM answers over pages + citations
- [ ] **M4: Benchmark** — table/chart set, visual-vs-text numbers
- [ ] **M5: Ship** — viewer with page highlight, README, release

## Documentation

- [ ] Inline doc comments across public surface
- [ ] API reference (`docs/api.md`)
- [ ] Architecture dossier (`docs/architecture.md`)
- [ ] Integration guide (`docs/integration.md`)
- [ ] Usage/how-to guides and `docs/` index
- [ ] Final README pass
