"""FastAPI application for facsimile."""

from fastapi import FastAPI

app = FastAPI(
    title="facsimile",
    description="Visual document RAG with late-interaction page embeddings",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
