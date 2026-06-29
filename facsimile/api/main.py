"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from facsimile import __version__

app = FastAPI(
    title="facsimile",
    description=(
        "Visual document RAG: retrieve over page images using "
        "late-interaction visual embeddings, answer with a vision LLM."
    ),
    version=__version__,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
