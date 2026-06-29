# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-29
- M0 scaffold: Python package layout (`facsimile/{ingest,index,retrieve,answer,bench,api}`), `pyproject.toml` with full dependency manifest, `.gitignore`, and GitHub Actions CI workflow.
- `facsimile/ingest/rasterize.py`: PDF-to-page-image rasterization via pypdfium2 at configurable DPI, accepting either file paths or raw bytes.
- `facsimile/index/store.py`: Qdrant multivector collection management — `ensure_collection`, `upsert_pages`, `make_memory_client` — using MaxSim late-interaction scoring (ColPali-compatible).
- Test suite with 13 tests covering rasterization (single-page, multi-page, DPI scaling, PIL output type) and store (collection creation, idempotency, upsert, payload round-trip, ID stability).
