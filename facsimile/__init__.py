"""facsimile — visual document RAG via late-interaction page-image embeddings.

facsimile retrieves over PDF *pages as images*, using a ColPali/ColQwen
multi-vector model that produces one patch-vector per image region.  At query
time, Qdrant's built-in ``MAX_SIM`` comparator implements the late-interaction
scoring (sum of per-query-token maxima over doc patches), returning the most
relevant pages in a single round-trip.  A Claude vision model then reads the
page images and synthesizes a grounded answer that cites the exact pages it
drew from.

Package layout
--------------
``facsimile.ingest``
    PDF → page images (pypdfium2) → patch-vector embeddings (ColPali/ColQwen).
``facsimile.index``
    Qdrant multivector collection; upsert and query pages.
``facsimile.retrieve``
    :class:`~facsimile.retrieve.retriever.Retriever` — query → embed → MaxSim
    → :class:`~facsimile.retrieve.retriever.RetrievedPage` list.
``facsimile.answer``
    :func:`~facsimile.answer.synthesizer.synthesize` — retrieved page images
    → Claude vision model → :class:`~facsimile.answer.synthesizer.Answer` with
    citations.
``facsimile.bench``
    Question set loader, text-pipeline baseline, and benchmark runner for
    visual-vs-text comparison on table/chart questions.
``facsimile.api``
    FastAPI application (``/ingest``, ``/retrieve``, ``/ask``, ``/pages``,
    ``/viewer``).
``facsimile.viewer``
    Single-page HTML viewer that highlights cited pages with a gold border.
"""

__version__ = "0.1.0"
