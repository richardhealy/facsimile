# Progress

## Implementation

- [x] **M0 — Scaffold**: project structure, PDF rasterization (`ingest/rasterize.py`), Qdrant multivector store (`index/store.py`), CI (GitHub Actions), 18 tests passing.
- [x] **M1 — Visual embedding**: `Embedder` protocol, `StubEmbedder` (tests/CI), `ColPaliEmbedder` + `ColQwenEmbedder` (optional `embed` extra); `ingest_pdf` / `ingest_and_store` pipeline; 35 tests passing.
- [x] **M2 — Retrieval**: `embed_queries` on Embedder protocol; `Retriever` class (query → embed → Qdrant MaxSim → `RetrievedPage` list); 14 new tests, 59 total.
- [x] **M3 — Vision synthesis**: `synthesize()` feeds retrieved page images to a Claude vision model; returns `Answer` with grounded text and `[Page N]`-cited page numbers; 21 new tests, 80 total.
- [x] **M4 — Benchmark**: `BenchQuestion` schema + `load_questions`; `text_retrieve` (keyword-overlap text pipeline as baseline); `run_benchmark` (visual vs text, per-question results); `summarize_results` (accuracy by pipeline and question type); `bench/sample_questions.json`; 45 new tests, 125 total.
- [x] **M5 — Ship**: FastAPI endpoints (`/ingest`, `/retrieve`, `/ask`, `/pages`, `/viewer`), page-highlight viewer (HTML SPA with cited-page border), `python-multipart` dependency; 27 new tests, 152 total.

## Documentation

- [x] Inline doc comments across the public surface.
- [x] API reference (OpenAPI spec + `docs/api.md`).
- [x] Architecture dossier (`docs/architecture.md`).
- [ ] Integration guide (`docs/integration.md`).
- [ ] Usage and how-to guides; `docs/` index; final README pass.

## Maintenance

- [x] 2026-10-06: page point IDs no longer depend on Python's per-process string hash salt, and re-ingesting a document replaces its points. Before this, each re-ingest after a restart added another copy of every page to a persistent Qdrant, and a document that got shorter kept its old pages. 9 regression tests, 161 total.
- [x] 2026-11-08: the API picks its embedder from `FACSIMILE_EMBEDDER` (`stub`, `colpali` or `colqwen`), with `FACSIMILE_EMBED_MODEL` and `FACSIMILE_EMBED_DEVICE`, through the new `make_embedder()`. Before this the server always used the stub, whatever was set, and nothing said its retrieval was random. An unknown name or a missing `embed` extra now stops startup, and the stub logs a warning. 13 new tests, 174 total.

## Known issues

- The API still defaults to `StubEmbedder` (random vectors) when `FACSIMILE_EMBEDDER` is not set, so a default server returns arbitrary pages from `/retrieve` and `/ask`. Startup now warns about it. Whether the default should become a real model (torch and a multi-GB download in the default install) is an open decision.
- Citations are bare page numbers. With more than one document indexed, two retrieved pages can share a number (page 7 of two reports), and `cited_pages` cannot say which document the answer came from.
- Re-ingesting a shorter document through `POST /ingest` leaves the old page images in `DATA_DIR`, so `GET /pages/{doc_id}/{n}` still serves pages the document no longer has. They are no longer retrieved.
