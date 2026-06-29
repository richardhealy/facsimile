# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-29
- M0 scaffold: project structure (`ingest/`, `index/`, `retrieve/`, `answer/`, `bench/`, `api/`), `pyproject.toml`, and `.gitignore`.
- `facsimile.ingest.rasterize`: PDF-to-PIL rasterization via PyMuPDF with configurable DPI and `page_size_bytes` helper.
- `facsimile.index.store`: `PageVectorStore` backed by Qdrant multivector (MAX_SIM comparator); deterministic UUID point IDs; in-memory mode for tests.
- `facsimile.api.main`: minimal FastAPI app with `/health` endpoint.
- 15 passing tests covering rasterization and the multivector store.
- GitHub Actions CI workflow (`ci.yml`).
