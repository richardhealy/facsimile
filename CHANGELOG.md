# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added — 2026-06-29
- M0 scaffold: PDF rasterization via PyMuPDF (`facsimile/ingest/rasterize.py`)
- Qdrant multivector page store with MaxSim late-interaction config (`facsimile/index/store.py`)
- Pure-NumPy MaxSim scorer and page ranker (`facsimile/retrieve/scorer.py`)
- FastAPI application with health endpoint (`facsimile/api/main.py`)
- Stub modules for answer synthesis, benchmark runner, and viewer
- Full test suite (28 tests covering rasterization, store, scorer, API)
- GitHub Actions CI workflow
