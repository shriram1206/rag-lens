"""
Unit tests for Pydantic schema validation (ingestion/schema.py).

Tests confirm that invalid data fails fast with clear errors rather than
propagating None values downstream and silently corrupting benchmark results.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_lens.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput


class TestQAItem:
    def test_valid_item_creates_successfully(self) -> None:
        item = QAItem(
            question="What is a transformer model?",
            ground_truth_answer="A neural network architecture using self-attention.",
            ground_truth_context="The Transformer model was introduced in the paper 'Attention is All You Need'.",
        )
        assert item.item_id != ""
        assert len(item.item_id) == 12

    def test_auto_id_is_deterministic(self) -> None:
        kwargs = dict(
            question="Q", ground_truth_answer="A", ground_truth_context="C"
        )
        id1 = QAItem(**kwargs).item_id
        id2 = QAItem(**kwargs).item_id
        assert id1 == id2

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            QAItem(question="  ", ground_truth_answer="A", ground_truth_context="C")

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValidationError):
            QAItem(question="Q", ground_truth_answer="", ground_truth_context="C")

    def test_explicit_id_preserved(self) -> None:
        item = QAItem(
            item_id="custom_id",
            question="Q",
            ground_truth_answer="A",
            ground_truth_context="C",
        )
        assert item.item_id == "custom_id"

    def test_whitespace_stripped(self) -> None:
        item = QAItem(
            question="  What is AI?  ",
            ground_truth_answer="  Artificial Intelligence.  ",
            ground_truth_context="  The study of intelligent machines.  ",
        )
        assert item.question == "What is AI?"
        assert item.ground_truth_answer == "Artificial Intelligence."


class TestRAGOutput:
    def test_valid_output_creates_successfully(self) -> None:
        out = RAGOutput(
            item_id="abc123",
            question="What is RAG?",
            retrieved_context=["RAG combines retrieval and generation.", "It uses vector search."],
            generated_answer="RAG is a technique that retrieves documents before generating answers.",
        )
        assert out.item_id == "abc123"
        assert len(out.retrieved_context) == 2

    def test_empty_retrieved_context_raises(self) -> None:
        with pytest.raises(ValidationError):
            RAGOutput(
                item_id="x",
                question="Q",
                retrieved_context=[],
                generated_answer="A",
            )

    def test_empty_chunk_in_context_raises(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            RAGOutput(
                item_id="x",
                question="Q",
                retrieved_context=["Valid chunk", "   ", "Another chunk"],
                generated_answer="A",
            )

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValidationError):
            RAGOutput(
                item_id="x",
                question="Q",
                retrieved_context=["ctx"],
                generated_answer="",
            )


class TestMetricScore:
    def test_score_in_range_succeeds(self) -> None:
        s = MetricScore(score=0.75, rationale="test", raw_response="")
        assert s.score == pytest.approx(0.75)

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricScore(score=-0.1, rationale="r", raw_response="")

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricScore(score=1.1, rationale="r", raw_response="")

    def test_score_rounded_to_four_decimals(self) -> None:
        s = MetricScore(score=0.333333333, rationale="r", raw_response="")
        assert s.score == 0.3333


class TestEvalResult:
    def test_is_valid_with_all_metrics(self) -> None:
        ms = MetricScore(score=0.8, rationale="r", raw_response="")
        r = EvalResult(
            item_id="x",
            question="Q",
            faithfulness=ms,
            answer_relevance=ms,
            context_precision=ms,
            context_recall=ms,
        )
        assert r.is_valid is True
        assert r.composite_score == pytest.approx(0.8)

    def test_is_invalid_with_error(self) -> None:
        r = EvalResult(item_id="x", question="Q", error="Something failed")
        assert r.is_valid is False
        assert r.composite_score is None

    def test_is_invalid_with_no_metrics(self) -> None:
        r = EvalResult(item_id="x", question="Q")
        assert r.is_valid is False
