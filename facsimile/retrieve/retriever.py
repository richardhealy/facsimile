"""High-level retriever: text query → top-k page results via Qdrant MaxSim."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import QueryEmbedder


@dataclass
class RetrievedPage:
    """A single page returned by retrieval, with its score and provenance."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


class Retriever:
    """End-to-end retriever: embed a text query and score pages via Qdrant.

    Uses the Qdrant multivector/MaxSim index created by
    :func:`facsimile.index.store.ensure_collection`, so scoring is handled
    inside the index — no in-process matrix math at query time.
    """

    def __init__(
        self,
        query_embedder: QueryEmbedder,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._query_embedder = query_embedder
        self._client = client
        self._collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPage]:
        """Embed *query* and return the top-*k* pages ranked by MaxSim score.

        Args:
            query: Natural-language question or keyword string.
            top_k: Maximum number of pages to return.

        Returns:
            Pages sorted descending by relevance score.
        """
        all_vecs = self._query_embedder.embed_queries([query])
        query_vectors: list[list[float]] = all_vecs[0]

        scored = query_pages(
            self._client,
            query_vectors=query_vectors,
            top_k=top_k,
            collection_name=self._collection_name,
        )
        return [
            RetrievedPage(
                doc_id=p.payload["doc_id"],
                page_num=p.payload["page_num"],
                doc_path=p.payload["doc_path"],
                score=p.score,
            )
            for p in scored
        ]
