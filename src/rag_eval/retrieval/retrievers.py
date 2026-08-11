"""
Retrieval implementations: Dense (ChromaDB) and Hybrid (Dense + BM25).

Both retrievers implement the same interface so the pipeline runner is
agnostic to which retrieval method is configured.

Dense retrieval uses ChromaDB for ANN vector search.
Hybrid retrieval fuses dense scores with BM25 keyword scores using
Reciprocal Rank Fusion (RRF) — a parameter-free late-fusion method.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from rag_eval.retrieval.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def index(self, chunks: list[str]) -> None:
        """Index a list of text chunks for subsequent retrieval.

        Args:
            chunks: Text chunks produced by a chunker. Must be non-empty.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to return.

        Returns:
            List of chunk strings, ordered by relevance (most relevant first).
        """
        ...

    @property
    @abstractmethod
    def retrieval_method(self) -> str:
        """Identifier used in benchmark configs and leaderboard ("dense" or "hybrid")."""
        ...


class DenseRetriever(BaseRetriever):
    """Vector-only retrieval using ChromaDB.

    Chunks are embedded and stored in an in-memory ChromaDB collection.
    Queries are embedded with the same model and retrieved by cosine similarity.

    Args:
        embedder: Any BaseEmbedder implementation.
        collection_name: Optional ChromaDB collection name.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._client = chromadb.Client()  # In-memory; no disk persistence
        self._collection_name = collection_name or f"rag_eval_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: list[str] = []

    @property
    def retrieval_method(self) -> str:
        return "dense"

    def index(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")
        self._chunks = chunks
        embeddings = self._embedder.embed(chunks)
        ids = [hashlib.md5(c.encode()).hexdigest()[:16] for c in chunks]
        self._collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
        )
        logger.debug("DenseRetriever: indexed %d chunks", len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query_emb = self._embedder.embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, len(self._chunks)),
        )
        documents = results.get("documents", [[]])[0]
        return documents


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval fusing dense vectors with BM25 keyword scores via RRF.

    Reciprocal Rank Fusion (RRF):
        score(d) = Σ_r 1 / (k + rank_r(d))
    where k=60 is the standard smoothing constant. No hyperparameter tuning needed.

    This method consistently outperforms pure dense retrieval on documents
    with rare or domain-specific terminology (the BM25 arm catches exact keyword matches
    that embeddings may summarize away).

    Args:
        embedder: Any BaseEmbedder implementation.
        rrf_k: RRF smoothing constant (default: 60, the standard value).
    """

    def __init__(self, embedder: BaseEmbedder, rrf_k: int = 60) -> None:
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._dense = DenseRetriever(embedder=embedder)
        self._bm25: BM25Okapi | None = None
        self._chunks: list[str] = []

    @property
    def retrieval_method(self) -> str:
        return "hybrid"

    def index(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")
        self._chunks = chunks
        # Index for both retrieval arms
        self._dense.index(chunks)
        tokenized = [c.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("HybridRetriever: indexed %d chunks (dense + BM25)", len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if self._bm25 is None:
            raise RuntimeError("HybridRetriever.index() must be called before retrieve()")

        # --- Dense arm ---
        dense_results = self._dense.retrieve(query, top_k=len(self._chunks))
        dense_rank: dict[str, int] = {doc: rank for rank, doc in enumerate(dense_results)}

        # --- BM25 arm ---
        query_tokens = query.lower().split()
        bm25_scores: np.ndarray = self._bm25.get_scores(query_tokens)
        bm25_order = np.argsort(bm25_scores)[::-1].tolist()
        bm25_rank: dict[str, int] = {
            self._chunks[idx]: rank for rank, idx in enumerate(bm25_order)
        }

        # --- RRF fusion ---
        all_docs = set(dense_rank.keys()) | set(bm25_rank.keys())
        rrf_scores: dict[str, float] = {}
        for doc in all_docs:
            dr = dense_rank.get(doc, len(self._chunks))  # Penalize missing
            br = bm25_rank.get(doc, len(self._chunks))
            rrf_scores[doc] = (1 / (self._rrf_k + dr)) + (1 / (self._rrf_k + br))

        ranked = sorted(all_docs, key=lambda d: rrf_scores[d], reverse=True)
        return ranked[:top_k]


def get_retriever(method: str, embedder: BaseEmbedder) -> BaseRetriever:
    """Factory: return the appropriate retriever for a method string.

    Args:
        method: "dense" or "hybrid".
        embedder: Embedder instance to use for vectorization.

    Returns:
        Instantiated BaseRetriever subclass.
    """
    if method == "dense":
        return DenseRetriever(embedder=embedder)
    elif method == "hybrid":
        return HybridRetriever(embedder=embedder)
    else:
        raise ValueError(
            f"Unknown retrieval method '{method}'. Valid options: 'dense', 'hybrid'"
        )
