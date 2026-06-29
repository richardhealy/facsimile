"""Late-interaction visual embedding: page images → per-patch vectors.

The public surface is the ``Embedder`` protocol. Two implementations ship:

- ``StubEmbedder`` — deterministic random vectors, no model download.  Used
  in tests and as a placeholder when the heavy ML deps are not installed.
- ``ColPaliEmbedder`` — wraps ``colpali-engine`` (ColPali / ColQwen2).
  Requires ``pip install facsimile[embed]``.

Both return the same shape: ``list[list[list[float]]]`` — one inner list per
image, each containing *n_patches* vectors of length *dim* (128 for ColPali).

``embed_query`` maps a text string to the same patch-vector space so that
late-interaction MaxSim scoring can be applied between query and page.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

COLPALI_DIM = 128


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn page images and text queries into patch-vector lists."""

    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        """Return patch vectors for each image.

        Shape per image: ``(n_patches, dim)``.  ``n_patches`` may vary
        between images (depends on image size and the model's patch stride).
        """
        ...

    def embed_query(self, text: str) -> list[list[float]]:
        """Return patch vectors for a text query.

        Shape: ``(n_query_patches, dim)``.  Vectors live in the same space
        as image patches so MaxSim can score query vs page directly.
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

    def embed_query(self, text: str) -> list[list[float]]:
        """Return deterministic random query vectors seeded from the query text."""
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self._n_patches, self._dim)).tolist()


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

    def embed_query(self, text: str) -> list[list[float]]:
        batch = self._processor.process_queries([text]).to(self._device)
        with self._torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings[0].float().cpu().tolist()


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

    def embed_query(self, text: str) -> list[list[float]]:
        batch = self._processor.process_queries([text]).to(self._device)
        with self._torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings[0].float().cpu().tolist()
