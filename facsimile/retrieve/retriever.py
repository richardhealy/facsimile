"""Late-interaction retriever: text query → ranked page results."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import Embedder


@dataclass
class RetrievedPage:
    """A page returned by the retriever with its MaxSim relevance score."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


class Retriever:
    """End-to-end late-interaction retriever over a Qdrant page index.

    Embeds the text query via the configured embedder (which must implement
    ``embed_queries``), then delegates MaxSim scoring to Qdrant's built-in
    multivector comparator and returns ranked ``RetrievedPage`` results.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPage]:
        """Return the *top_k* pages most relevant to *query*.

        Embeds *query* with the configured embedder, issues a MaxSim query
        against the Qdrant collection, and maps results to ``RetrievedPage``
        records ordered by descending relevance score.
        """
        query_vectors = self._embedder.embed_queries([query])[0]
        scored_points = query_pages(
            self._client,
            query_vectors,
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
            for p in scored_points
        ]
