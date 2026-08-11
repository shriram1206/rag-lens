"""
Text chunking strategies for the benchmark retrieval layer.

Three strategies are implemented to match the benchmark matrix:
  - SentenceChunker   → splits on sentence boundaries
  - ParagraphChunker  → splits on double-newline paragraph breaks
  - SemanticChunker   → groups sentences by semantic similarity using embeddings

Each chunker takes a raw document string and returns a list of text chunks.
All strategies produce non-overlapping, non-empty chunks.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# Simple sentence boundary regex — handles periods, exclamation marks, question marks
# followed by whitespace and an uppercase letter, or end of string.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class BaseChunker(ABC):
    """Abstract chunker interface."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into non-overlapping, non-empty chunks.

        Args:
            text: Raw document text.

        Returns:
            List of text chunks (never empty strings).
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Identifier used in benchmark config and leaderboard."""
        ...

    @staticmethod
    def _clean(chunks: list[str]) -> list[str]:
        """Remove empty or whitespace-only chunks."""
        return [c.strip() for c in chunks if c.strip()]


class SentenceChunker(BaseChunker):
    """Split text into individual sentence chunks.

    Args:
        max_sentences: Group this many sentences into a single chunk.
                       Default 1 = one sentence per chunk.
                       Increase to reduce chunk count and add more context per chunk.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(self, text: str) -> list[str]:
        sentences = _SENTENCE_BOUNDARY.split(text.strip())
        sentences = self._clean(sentences)

        if self._max_sentences == 1:
            return sentences

        # Group into windows of max_sentences
        groups = []
        for i in range(0, len(sentences), self._max_sentences):
            group = " ".join(sentences[i : i + self._max_sentences])
            if group.strip():
                groups.append(group.strip())
        return groups


class ParagraphChunker(BaseChunker):
    """Split text on double-newline paragraph boundaries.

    This is the simplest and fastest chunking strategy. Best suited for
    documents with clear paragraph structure (articles, documentation).
    """

    @property
    def strategy_name(self) -> str:
        return "paragraph"

    def chunk(self, text: str) -> list[str]:
        raw_chunks = re.split(r"\n\s*\n", text.strip())
        return self._clean(raw_chunks)


class SemanticChunker(BaseChunker):
    """Split text by detecting semantic breakpoints using cosine similarity.

    Sentences are embedded; a chunk boundary is inserted where cosine
    similarity between adjacent sentence groups drops below a threshold.

    Args:
        model_name: sentence-transformers model to use for embeddings.
        breakpoint_threshold: Similarity below this value triggers a chunk split.
                              Range [0, 1]; higher = more splits (smaller chunks).
        window_size: Number of sentences to compare at each boundary check.

    Note:
        This chunker loads a sentence-transformers model on first call.
        It is significantly slower than Sentence/ParagraphChunker but produces
        the most semantically coherent chunks.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_threshold: float = 0.3,
        window_size: int = 2,
    ) -> None:
        self._model_name = model_name
        self._threshold = breakpoint_threshold
        self._window = window_size
        self._model = None  # Lazy-loaded

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            logger.info("Loading semantic chunker model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def chunk(self, text: str) -> list[str]:
        self._load_model()
        sentences = _SENTENCE_BOUNDARY.split(text.strip())
        sentences = self._clean(sentences)

        if len(sentences) <= 2:
            return sentences

        # Embed all sentences at once (batched — efficient)
        embeddings = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        chunks: list[str] = []
        current_chunk: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            # Compare centroid of current window to next sentence
            window_start = max(0, i - self._window)
            current_centroid = embeddings[window_start:i].mean(axis=0)
            next_emb = embeddings[i]

            similarity = float(
                np.dot(current_centroid, next_emb)
                / (np.linalg.norm(current_centroid) * np.linalg.norm(next_emb) + 1e-9)
            )

            if similarity < (1.0 - self._threshold):
                # Semantic break detected — flush current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return self._clean(chunks)


def get_chunker(strategy: str) -> BaseChunker:
    """Factory: return the appropriate chunker for a strategy name.

    Args:
        strategy: One of "sentence", "paragraph", "semantic".

    Returns:
        Instantiated BaseChunker subclass.

    Raises:
        ValueError: For unknown strategy names.
    """
    if strategy == "sentence":
        return SentenceChunker()
    if strategy == "paragraph":
        return ParagraphChunker()
    if strategy == "semantic":
        return SemanticChunker()
    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Valid options: 'sentence', 'paragraph', 'semantic'"
    )
