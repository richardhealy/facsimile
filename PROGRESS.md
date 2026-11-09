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
- [x] 2026-11-09: re-ingesting a shorter document through `POST /ingest` now deletes the page images beyond its new last page, once the index update has succeeded, so `GET /pages` stops serving pages the document no longer has. Before this, a 3-page `report.pdf` re-ingested as 1 page still served pages 2 and 3. The same change refuses an upload whose filename stem is only dots: `...pdf` had the doc_id `..`, so its page images were written to the folder above `DATA_DIR`, and `GET /pages/%2E%2E/1` served a `page_0001.png` from there. 10 new tests, 184 total.

## Known issues

- The API still defaults to `StubEmbedder` (random vectors) when `FACSIMILE_EMBEDDER` is not set, so a default server returns arbitrary pages from `/retrieve` and `/ask`. Startup now warns about it. Whether the default should become a real model (torch and a multi-GB download in the default install) is an open decision.
- Citations are bare page numbers. With more than one document indexed, two retrieved pages can share a number (page 7 of two reports), and `cited_pages` cannot say which document the answer came from.
- A re-ingest that fails while embedding or indexing has already overwritten the page images it shares with the old version, so until a successful re-ingest those pages show the new images while the index still holds the old vectors. The extra old pages are kept, since the index still has them.
- `POST /ingest` rasterizes each PDF twice: once in the endpoint for the page images and again inside `ingest_and_store()` for embedding.
