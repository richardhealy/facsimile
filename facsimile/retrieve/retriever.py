"""High-level retrieval: text query → top-k document pages via Qdrant MaxSim."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import Embedder


@dataclass
class RetrievedPage:
    """A single page returned by retrieval, with its provenance and score."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


class Retriever:
    """Combines a query embedder with a Qdrant index to answer retrieval queries.

    Usage::

        retriever = Retriever(client, embedder)
        pages = retriever.retrieve("What was the revenue in Q3?", top_k=3)
        for page in pages:
            print(page.doc_id, page.page_num, page.score)
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
        """Embed *query* and return the top-k most relevant pages.

        The query is embedded with the same multi-vector space used for page
        images, then scored against the index using Qdrant's built-in MaxSim
        comparator.  Results are returned in descending score order.

        Args:
            query: Natural-language question or keyword string.
            top_k: Maximum number of pages to return.

        Returns:
            List of :class:`RetrievedPage` sorted by score descending.
        """
        query_vecs = self._embedder.embed_queries([query])[0]
        scored_points = query_pages(
            self._client,
            query_vecs,
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
