# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-29 (M2)
- M2 retrieval: `embed_queries` method added to `Embedder` protocol and all implementations (`StubEmbedder`, `ColPaliEmbedder`, `ColQwenEmbedder`).
- `facsimile/retrieve/retriever.py`: `RetrievedPage` dataclass and `Retriever` class — text query → Qdrant MaxSim → ranked page list.
- `facsimile/retrieve/__init__.py` now exports `RetrievedPage`, `Retriever`, `max_sim_score`, `rank_pages`.
- 11 new tests in `tests/test_retrieve.py` covering query embedding shape, retrieval ordering, payload fields, and custom collections; 56 total.

### Added — 2026-06-29
- M1 visual embedding: `Embedder` protocol; `StubEmbedder` (deterministic, no model required); `ColPaliEmbedder` and `ColQwenEmbedder` wrappers (optional `pip install facsimile[embed]`).
- `facsimile/ingest/pipeline.py`: `ingest_pdf` and `ingest_and_store` — single-call PDF → embeddings → Qdrant pipeline.
- 17 new tests covering embedding shape, determinism, and end-to-end ingest round-trips; 35 total.
- M0 scaffold: project layout (`ingest/`, `index/`, `retrieve/`, `answer/`, `bench/`, `api/`, `viewer/`).
- `facsimile/ingest/rasterize.py`: PDF → PIL Images via pypdfium2 (`rasterize_pdf`, `rasterize_bytes`, `page_count`).
- `facsimile/index/store.py`: Qdrant multivector collection setup with MaxSim comparator, `upsert_pages`, `query_pages`.
- 18 tests covering rasterization and the multivector store (in-memory Qdrant, no server required).
- GitHub Actions CI workflow.
