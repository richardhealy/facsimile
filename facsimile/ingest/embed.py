"""Late-interaction visual embedding: page images → per-patch vectors.

The public surface is the ``Embedder`` protocol for indexing and the
``QueryEmbedder`` protocol for text-query embedding. Two implementations ship
for each:

- ``StubEmbedder`` / ``StubQueryEmbedder`` — deterministic random vectors, no
  model download.  Used in tests and CI.
- ``ColPaliEmbedder`` / ``ColQwenEmbedder`` — wraps ``colpali-engine``.
  Requires ``pip install facsimile[embed]``.

Both protocols return ``list[list[list[float]]]``: one inner list per
image/query, each containing *n_patches* (or *n_tokens*) vectors of length
*dim* (128 for ColPali).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

COLPALI_DIM = 128


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn page images into patch-vector lists."""

    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        """Return patch vectors for each image.

        Shape per image: ``(n_patches, dim)``.  ``n_patches`` may vary
        between images (depends on image size and the model's patch stride).
        """
        ...


class StubEmbedder:
    """Deterministic random-vector embedder — no model required.

    Useful for integration tests and CI where downloading a multi-GB
    model checkpoint is not feasible.
    """

    def __init__(
        self,
        n_patches: int = 16,
        dim: int = COLPALI_DIM,
        seed: int = 0,
    ) -> None:
        self._n_patches = n_patches
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        return [
            self._rng.standard_normal((self._n_patches, self._dim)).tolist()
            for _ in images
        ]


class ColPaliEmbedder:
    """ColPali / ColQwen2 embedder backed by ``colpali-engine``.

    Requires ``pip install facsimile[embed]``.  Model weights are downloaded
    from the Hugging Face Hub on first instantiation.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as exc:
            raise ImportError(
                "colpali-engine and torch are required for ColPaliEmbedder. "
                "Run: pip install facsimile[embed]"
            ) from exc

        self._torch = torch
        self._device = device
        self._processor = ColPaliProcessor.from_pretrained(model_name)
        self._model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        batch = self._processor.process_images(images).to(self._device)
        with self._torch.no_grad():
            embeddings = self._model(**batch)  # (batch, n_patches, dim)
        return [emb.float().cpu().tolist() for emb in embeddings]


@runtime_checkable
class QueryEmbedder(Protocol):
    """Anything that can turn text queries into token-vector lists."""

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        """Return token vectors for each query string.

        Shape per query: ``(n_tokens, dim)``.  ``n_tokens`` may vary across
        queries depending on the model's tokenizer.
        """
        ...


class StubQueryEmbedder:
    """Deterministic random-vector query embedder — no model required.

    Produces the same vectors for the same query string regardless of run
    (uses MD5 of the query text to seed the RNG, avoiding Python's
    randomised ``hash()``).
    """

    def __init__(
        self,
        n_tokens: int = 8,
        dim: int = COLPALI_DIM,
        seed: int = 0,
    ) -> None:
        self._n_tokens = n_tokens
        self._dim = dim
        self._seed = seed

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        results = []
        for query in queries:
            digest = int(hashlib.md5(query.encode()).hexdigest(), 16)
            rng = np.random.default_rng((digest + self._seed) % (2**32))
            results.append(rng.standard_normal((self._n_tokens, self._dim)).tolist())
        return results


class ColQwenEmbedder:
    """ColQwen2 embedder backed by ``colpali-engine``.

    Requires ``pip install facsimile[embed]``.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise ImportError(
                "colpali-engine and torch are required for ColQwenEmbedder. "
                "Run: pip install facsimile[embed]"
            ) from exc

        self._torch = torch
        self._device = device
        self._processor = ColQwen2Processor.from_pretrained(model_name)
        self._model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        batch = self._processor.process_images(images).to(self._device)
        with self._torch.no_grad():
            embeddings = self._model(**batch)
        return [emb.float().cpu().tolist() for emb in embeddings]
