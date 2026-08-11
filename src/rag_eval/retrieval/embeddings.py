"""
Embedding model wrappers for the benchmark matrix.

Three embedding arms:
  - OpenAIEmbedder   → text-embedding-ada-002 (remote, requires OPENAI_API_KEY)
  - BGEEmbedder      → BAAI/bge-large-en-v1.5 (local, no API key needed)
  - E5Embedder       → intfloat/e5-large-v2 (local, no API key needed)

Design: All embedders expose the same interface so the retrieval layer
is agnostic to which embedding model is in use.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used in benchmark configs and leaderboard."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings."""
        ...


class OpenAIEmbedder(BaseEmbedder):
    """Embed text using OpenAI's text-embedding-ada-002.

    Requires OPENAI_API_KEY environment variable.

    Args:
        model: OpenAI model ID. Defaults to text-embedding-ada-002.
        batch_size: Number of texts per API call (OpenAI max: 2048).
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 512,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Required for the ada-002 embedding arm."
            )
        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return "ada-002"

    @property
    def embedding_dim(self) -> int:
        return 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)
            logger.debug("OpenAI embedder: %d/%d texts embedded", i + len(batch), len(texts))
        return np.array(all_embeddings, dtype=np.float32)


class BGEEmbedder(BaseEmbedder):
    """Embed text using BAAI/bge-large-en-v1.5 (runs locally via sentence-transformers).

    No API key required. Downloads model on first use (~1.3 GB).

    BGE models are optimized for retrieval tasks and consistently outperform
    ada-002 on retrieval benchmarks (BEIR, MTEB) while being free to run.
    """

    _HF_MODEL_ID = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        self._model = None  # Lazy-loaded

    @property
    def model_name(self) -> str:
        return "bge-large"

    @property
    def embedding_dim(self) -> int:
        return 1024

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            logger.info("Loading BGE embedding model: %s", self._HF_MODEL_ID)
            self._model = SentenceTransformer(self._HF_MODEL_ID)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        # BGE models expect a query prefix for retrieval queries
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class E5Embedder(BaseEmbedder):
    """Embed text using intfloat/e5-large-v2 (runs locally via sentence-transformers).

    No API key required. Downloads model on first use (~1.3 GB).

    E5 models use prefixed inputs: "query: ..." for queries and "passage: ..."
    for documents. This wrapper handles the prefix automatically.
    """

    _HF_MODEL_ID = "intfloat/e5-large-v2"

    def __init__(self) -> None:
        self._model = None  # Lazy-loaded

    @property
    def model_name(self) -> str:
        return "e5-large"

    @property
    def embedding_dim(self) -> int:
        return 1024

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            logger.info("Loading E5 embedding model: %s", self._HF_MODEL_ID)
            self._model = SentenceTransformer(self._HF_MODEL_ID)

    def embed(self, texts: list[str], prefix: str = "passage") -> np.ndarray:
        """Embed texts with E5's required prefix.

        Args:
            texts: Text strings to embed.
            prefix: "passage" for document chunks, "query" for queries.
        """
        self._load()
        prefixed = [f"{prefix}: {t}" for t in texts]
        return self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def get_embedder(model_name: str) -> BaseEmbedder:
    """Factory: return the appropriate embedder for a model name string.

    Args:
        model_name: One of "ada-002", "bge-large", "e5-large".

    Returns:
        Instantiated BaseEmbedder subclass.

    Raises:
        ValueError: For unknown model names.
    """
    if model_name == "ada-002":
        return OpenAIEmbedder()
    elif model_name == "bge-large":
        return BGEEmbedder()
    elif model_name == "e5-large":
        return E5Embedder()
    else:
        raise ValueError(
            f"Unknown embedding model '{model_name}'. "
            "Valid options: 'ada-002', 'bge-large', 'e5-large'"
        )
