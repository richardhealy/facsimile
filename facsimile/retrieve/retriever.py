"""Late-interaction retriever: text query → top-k pages from Qdrant.

``Retriever`` is the public entry point for M2.  It wires together:

- the ``Embedder`` protocol (query-side) from :mod:`facsimile.ingest.embed`,
- the Qdrant multivector store from :mod:`facsimile.index.store`.

The Qdrant index runs MaxSim natively via the ``MAX_SIM`` comparator, so
retrieval is a single network round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import Embedder


@dataclass
class RetrievedPage:
    """One search result returned by :class:`Retriever`."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


class Retriever:
    """Query the facsimile index and return scored page results.

    Args:
        client: Connected :class:`~qdrant_client.QdrantClient`.
        embedder: Any :class:`~facsimile.ingest.embed.Embedder` that also
            implements ``embed_queries``.
        collection_name: Qdrant collection to search (defaults to
            :data:`~facsimile.index.store.COLLECTION_NAME`).
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
        """Embed *query* and return the *top_k* most relevant pages.

        Args:
            query: Natural-language question or search string.
            top_k: Maximum number of pages to return, sorted by descending
                MaxSim score.

        Returns:
            List of :class:`RetrievedPage` objects in score-descending order.
            May be shorter than *top_k* if fewer pages are indexed.
        """
        query_vectors = self._embedder.embed_queries([query])[0]
        scored = query_pages(
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
            for p in scored
        ]
