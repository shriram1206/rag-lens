"""
Pydantic data models for all data crossing module boundaries.

Design principle: Never pass bare dicts between modules.
Every piece of data the pipeline touches is validated here at ingestion time,
so downstream code can assume data integrity.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Ground-truth models (input to the framework)
# ---------------------------------------------------------------------------


class QAItem(BaseModel):
    """A single ground-truth question-answer-context triple.

    Args:
        item_id: Unique identifier; auto-generated from question hash if omitted.
        question: The question posed to the RAG system.
        ground_truth_answer: The correct answer to the question.
        ground_truth_context: The source passage(s) that contain the answer.
    """

    item_id: str = Field(default="", description="Unique ID; auto-generated if blank")
    question: str
    ground_truth_answer: str
    ground_truth_context: str

    @field_validator("question", "ground_truth_answer", "ground_truth_context", mode="before")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must not be empty or whitespace-only")
        return v.strip()

    @model_validator(mode="after")
    def _auto_generate_id(self) -> QAItem:
        if not self.item_id:
            digest = hashlib.sha256(
                f"{self.question}{self.ground_truth_answer}".encode()
            ).hexdigest()[:12]
            self.item_id = digest
        return self


class RAGOutput(BaseModel):
    """The output produced by a RAG pipeline for a single question.

    Args:
        item_id: Must match the corresponding QAItem.item_id.
        question: The original query.
        retrieved_context: Ordered list of text chunks returned by the retriever.
        generated_answer: The LLM answer generated from retrieved_context.
    """

    item_id: str
    question: str
    retrieved_context: list[str] = Field(min_length=1)
    generated_answer: str

    @field_validator("retrieved_context", mode="before")
    @classmethod
    def _chunks_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("retrieved_context must contain at least one chunk")
        cleaned = [c.strip() for c in v]
        if any(not c for c in cleaned):
            raise ValueError("retrieved_context must not contain empty chunks")
        return cleaned

    @field_validator("generated_answer", mode="before")
    @classmethod
    def _answer_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("generated_answer must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Evaluation result models (output from the framework)
# ---------------------------------------------------------------------------


class MetricScore(BaseModel):
    """A single metric score with the judge's rationale for auditability.

    Args:
        score: Numeric score in [0.0, 1.0].
        rationale: The judge LLM's natural-language explanation for the score.
        raw_response: The unmodified JSON string returned by the judge.
    """

    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    raw_response: str = Field(default="", description="Unmodified judge response for audit")

    @field_validator("score", mode="after")
    @classmethod
    def _round_score(cls, v: float) -> float:
        return round(v, 4)


class EvalResult(BaseModel):
    """Per-item evaluation result across all four metrics.

    Any metric may be None if evaluation failed for that item;
    the `error` field will explain why.
    """

    item_id: str
    question: str
    faithfulness: MetricScore | None = None
    answer_relevance: MetricScore | None = None
    context_precision: MetricScore | None = None
    context_recall: MetricScore | None = None
    error: str | None = Field(
        default=None,
        description="Populated when evaluation failed; do not silently ignore",
    )

    @property
    def composite_score(self) -> float | None:
        """Arithmetic mean of all non-None metric scores."""
        scores = [
            m.score
            for m in [
                self.faithfulness,
                self.answer_relevance,
                self.context_precision,
                self.context_recall,
            ]
            if m is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else None

    @property
    def is_valid(self) -> bool:
        """True if at least one metric was successfully scored and no error occurred."""
        return self.error is None and self.composite_score is not None


class RunSummary(BaseModel):
    """Aggregated evaluation summary for a single benchmark run configuration."""

    config_id: str
    chunking_strategy: str
    embedding_model: str
    retrieval_method: str
    n_items: int
    n_errors: int

    faithfulness_mean: float | None = None
    faithfulness_std: float | None = None
    answer_relevance_mean: float | None = None
    answer_relevance_std: float | None = None
    context_precision_mean: float | None = None
    context_precision_std: float | None = None
    context_recall_mean: float | None = None
    context_recall_std: float | None = None
    composite_score: float | None = None

    @property
    def error_rate(self) -> float:
        return round(self.n_errors / self.n_items, 4) if self.n_items > 0 else 0.0
