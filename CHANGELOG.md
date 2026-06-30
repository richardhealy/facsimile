# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-30 (M3)
- M3 vision synthesis: `facsimile/answer/synthesizer.py` — `synthesize()` encodes retrieved page images as base64 PNG, sends them with a grounded-answer system prompt to a Claude vision model, extracts `[Page N]` citations from the response, and returns an `Answer(text, cited_pages, model)` dataclass.
- `facsimile/answer/__init__.py` now exports `Answer` and `synthesize`.
- `anthropic>=0.40` added to core dependencies.
- 21 new tests covering citation extraction, multi-page batching, error handling, and mock-API round-trips; 80 total.

### Added — 2026-06-29 (M2)
- M2 retrieval: `embed_queries` method added to `Embedder` protocol, `StubEmbedder`, `ColPaliEmbedder`, and `ColQwenEmbedder`.
- `facsimile/retrieve/retriever.py`: `Retriever` class and `RetrievedPage` dataclass — query text → Qdrant MaxSim → scored page list.
- `facsimile/retrieve/__init__.py` now exports the public retrieval surface.
- 14 new tests covering query embedding, protocol compliance, and end-to-end retrieval; 59 total.

### Added — 2026-06-29
- M1 visual embedding: `Embedder` protocol; `StubEmbedder` (deterministic, no model required); `ColPaliEmbedder` and `ColQwenEmbedder` wrappers (optional `pip install facsimile[embed]`).
- `facsimile/ingest/pipeline.py`: `ingest_pdf` and `ingest_and_store` — single-call PDF → embeddings → Qdrant pipeline.
- 17 new tests covering embedding shape, determinism, and end-to-end ingest round-trips; 35 total.
- M0 scaffold: project layout (`ingest/`, `index/`, `retrieve/`, `answer/`, `bench/`, `api/`, `viewer/`).
- `facsimile/ingest/rasterize.py`: PDF → PIL Images via pypdfium2 (`rasterize_pdf`, `rasterize_bytes`, `page_count`).
- `facsimile/index/store.py`: Qdrant multivector collection setup with MaxSim comparator, `upsert_pages`, `query_pages`.
- 18 tests covering rasterization and the multivector store (in-memory Qdrant, no server required).
- GitHub Actions CI workflow.
