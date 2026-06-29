# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-29
- M2 retrieval: `retrieve()` and `retrieve_for_query()` in `facsimile/retrieve/retriever.py` — queries Qdrant MaxSim index and returns `RetrievedPage` results sorted by score.
- `embed_query(text)` method added to the `Embedder` protocol, `StubEmbedder` (deterministic, hashlib-seeded), `ColPaliEmbedder`, and `ColQwenEmbedder` — maps text into the same patch-vector space as page images.
- 14 new tests covering retrieval structure, score ordering, top-k clamping, empty collection, relevance ranking, and query embedding determinism; 59 total.

### Added — 2026-06-29
- M1 visual embedding: `Embedder` protocol; `StubEmbedder` (deterministic, no model required); `ColPaliEmbedder` and `ColQwenEmbedder` wrappers (optional `pip install facsimile[embed]`).
- `facsimile/ingest/pipeline.py`: `ingest_pdf` and `ingest_and_store` — single-call PDF → embeddings → Qdrant pipeline.
- 17 new tests covering embedding shape, determinism, and end-to-end ingest round-trips; 35 total.
- M0 scaffold: project layout (`ingest/`, `index/`, `retrieve/`, `answer/`, `bench/`, `api/`, `viewer/`).
- `facsimile/ingest/rasterize.py`: PDF → PIL Images via pypdfium2 (`rasterize_pdf`, `rasterize_bytes`, `page_count`).
- `facsimile/index/store.py`: Qdrant multivector collection setup with MaxSim comparator, `upsert_pages`, `query_pages`.
- 18 tests covering rasterization and the multivector store (in-memory Qdrant, no server required).
- GitHub Actions CI workflow.
