"""
RunConfig: Declarative configuration for a single benchmark run.

Each of the 18 benchmark cells is represented as one RunConfig.
The config is the single source of truth for a run — given only the config
and the dataset, any run can be reproduced exactly.

Design: RunConfig is a frozen dataclass so it can be used as a dict key
and its config_id can be safely computed from its fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChunkingStrategy = Literal["sentence", "paragraph", "semantic"]
EmbeddingModel = Literal["ada-002", "bge-large", "e5-large"]
RetrievalMethod = Literal["dense", "hybrid"]


@dataclass(frozen=True)
class RunConfig:
    """Full configuration for a single benchmark pipeline run.

    Args:
        chunking_strategy: How documents are split into chunks.
        embedding_model: Which embedding model vectorizes chunks and queries.
        retrieval_method: Dense (vector-only) or Hybrid (vector + BM25).
        top_k: Number of chunks to retrieve per query.
        judge_model: Override the judge LLM (defaults to env var JUDGE_MODEL).
        run_id: Optional human-readable label; auto-generated if omitted.

    Example:
        config = RunConfig(
            chunking_strategy="semantic",
            embedding_model="bge-large",
            retrieval_method="hybrid",
        )
        print(config.config_id)  # → "semantic__bge-large__hybrid"
    """

    chunking_strategy: ChunkingStrategy
    embedding_model: EmbeddingModel
    retrieval_method: RetrievalMethod
    top_k: int = 5
    judge_model: str | None = None
    run_id: str = field(default="")

    def __post_init__(self) -> None:
        # Workaround for frozen dataclass: set computed field via object.__setattr__
        if not self.run_id:
            computed = self.config_id
            object.__setattr__(self, "run_id", computed)

    @property
    def config_id(self) -> str:
        """Stable, human-readable identifier derived from the configuration.

        Format: "<chunking>__<embedding>__<retrieval>"
        Example: "semantic__bge-large__hybrid"
        """
        return f"{self.chunking_strategy}__{self.embedding_model}__{self.retrieval_method}"

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON logging."""
        return {
            "config_id": self.config_id,
            "chunking_strategy": self.chunking_strategy,
            "embedding_model": self.embedding_model,
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
            "judge_model": self.judge_model,
            "run_id": self.run_id,
        }


def benchmark_matrix() -> list[RunConfig]:
    """Generate the configurations for the benchmark matrix.

    Returns:
        List of RunConfig objects. Runs 18 configs if OPENAI_API_KEY is set,
        otherwise gracefully falls back to 12 configs (local embedders only).
    """
    import os

    configs = []

    # Check if OpenAI key is available
    available_embedders = ["bge-large"]
    if os.environ.get("OPENAI_API_KEY"):
        available_embedders.insert(0, "ada-002")

    for chunking in ("sentence", "paragraph", "semantic"):
        for embedding in available_embedders:
            for retrieval in ("dense", "hybrid"):
                configs.append(
                    RunConfig(
                        chunking_strategy=chunking,  # type: ignore[arg-type]
                        embedding_model=embedding,  # type: ignore[arg-type]
                        retrieval_method=retrieval,  # type: ignore[arg-type]
                    )
                )
    return configs
