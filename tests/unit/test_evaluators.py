"""
Unit tests for Faithfulness, AnswerRelevance, ContextPrecision, ContextRecall.

All tests use a MockJudge — NEVER call the live Groq API in unit tests.
See docs/qa-testing.md §2 for the testing philosophy.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_lens.evaluators.answer_relevance import AnswerRelevance
from rag_lens.evaluators.context_precision import ContextPrecision
from rag_lens.evaluators.context_recall import ContextRecall
from rag_lens.evaluators.faithfulness import Faithfulness
from rag_lens.ingestion.schema import MetricScore, QAItem, RAGOutput
from rag_lens.judge.base import BaseJudge, JudgeCallError, JudgeParseError, JudgeResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_judge(score: float = 0.85, rationale: str = "Test rationale") -> BaseJudge:
    """Return a mock judge that returns the given score."""
    judge = MagicMock(spec=BaseJudge)
    judge.judge.return_value = JudgeResponse(
        score=score,
        rationale=rationale,
        raw_response=f'{{"score": {score}, "rationale": "{rationale}"}}',
        prompt_version="v1.0.0",
    )
    return judge


def _make_qa_item(
    question: str = "What is RAG?",
    answer: str = "RAG stands for Retrieval-Augmented Generation.",
    context: str = "RAG (Retrieval-Augmented Generation) is an AI technique that retrieves relevant documents before generating an answer.",
) -> QAItem:
    return QAItem(question=question, ground_truth_answer=answer, ground_truth_context=context)


def _make_rag_output(
    item_id: str = "abc123",
    question: str = "What is RAG?",
    retrieved_context: list[str] | None = None,
    generated_answer: str = "RAG is a technique combining retrieval and generation.",
) -> RAGOutput:
    return RAGOutput(
        item_id=item_id,
        question=question,
        retrieved_context=retrieved_context or ["RAG combines retrieval and generation."],
        generated_answer=generated_answer,
    )


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_score_returns_metric_score_on_success(self) -> None:
        judge = _make_judge(score=0.9)
        evaluator = Faithfulness(judge=judge)
        output = _make_rag_output(item_id="test001")
        result = evaluator.score(output)
        assert result is not None
        assert result.score == pytest.approx(0.9, abs=0.001)
        assert isinstance(result, MetricScore)

    def test_score_returns_none_on_judge_call_error(self) -> None:
        judge = MagicMock(spec=BaseJudge)
        judge.judge.side_effect = JudgeCallError("API timeout")
        evaluator = Faithfulness(judge=judge)
        output = _make_rag_output(item_id="test002")
        result = evaluator.score(output)
        assert result is None

    def test_score_returns_none_on_judge_parse_error(self) -> None:
        judge = MagicMock(spec=BaseJudge)
        judge.judge.side_effect = JudgeParseError("Malformed JSON")
        evaluator = Faithfulness(judge=judge)
        output = _make_rag_output(item_id="test003")
        result = evaluator.score(output)
        assert result is None

    def test_evaluate_sets_error_on_failure(self) -> None:
        judge = MagicMock(spec=BaseJudge)
        judge.judge.side_effect = JudgeCallError("fail")
        evaluator = Faithfulness(judge=judge)
        output = _make_rag_output(item_id="test004")
        qa = _make_qa_item()
        qa = QAItem(item_id="test004", question=qa.question, ground_truth_answer=qa.ground_truth_answer, ground_truth_context=qa.ground_truth_context)
        result = evaluator.evaluate(output, qa)
        assert result.error is not None
        assert result.faithfulness is None

    def test_evaluate_populates_faithfulness_on_success(self) -> None:
        judge = _make_judge(score=0.75)
        evaluator = Faithfulness(judge=judge)
        output = _make_rag_output(item_id="test005")
        qa = QAItem(item_id="test005", question="What is RAG?", ground_truth_answer="RAG stands for Retrieval-Augmented Generation.", ground_truth_context="RAG is a technique.")
        result = evaluator.evaluate(output, qa)
        assert result.faithfulness is not None
        assert result.faithfulness.score == pytest.approx(0.75, abs=0.001)
        assert result.error is None

    def test_judge_called_with_combined_context(self) -> None:
        """Judge prompt must combine all retrieved chunks into a single context string."""
        judge = _make_judge()
        evaluator = Faithfulness(judge=judge)
        output = RAGOutput(
            item_id="ctx_test",
            question="Q?",
            retrieved_context=["Chunk A.", "Chunk B."],
            generated_answer="Answer.",
        )
        evaluator.score(output)
        call_args = judge.judge.call_args[1]["user_prompt"]
        assert "Chunk A." in call_args
        assert "Chunk B." in call_args


# ---------------------------------------------------------------------------
# AnswerRelevance
# ---------------------------------------------------------------------------


class TestAnswerRelevance:
    def test_score_success(self) -> None:
        judge = _make_judge(score=0.8)
        evaluator = AnswerRelevance(judge=judge)
        result = evaluator.score(_make_rag_output(item_id="rel001"))
        assert result is not None
        assert result.score == pytest.approx(0.8, abs=0.001)

    def test_score_none_on_error(self) -> None:
        judge = MagicMock(spec=BaseJudge)
        judge.judge.side_effect = JudgeCallError("rate limit")
        result = AnswerRelevance(judge=judge).score(_make_rag_output(item_id="rel002"))
        assert result is None

    def test_evaluate_sets_answer_relevance_field(self) -> None:
        judge = _make_judge(score=0.6)
        result = AnswerRelevance(judge=judge).evaluate(_make_rag_output(item_id="rel003"), _make_qa_item())
        assert result.answer_relevance is not None
        assert result.faithfulness is None  # Not set by this evaluator


# ---------------------------------------------------------------------------
# ContextPrecision
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_score_success(self) -> None:
        judge = _make_judge(score=0.7)
        result = ContextPrecision(judge=judge).score(_make_rag_output(item_id="prec001"))
        assert result is not None
        assert result.score == pytest.approx(0.7, abs=0.001)

    def test_score_none_on_error(self) -> None:
        judge = MagicMock(spec=BaseJudge)
        judge.judge.side_effect = JudgeParseError("bad")
        result = ContextPrecision(judge=judge).score(_make_rag_output(item_id="prec002"))
        assert result is None


# ---------------------------------------------------------------------------
# ContextRecall
# ---------------------------------------------------------------------------


class TestContextRecall:
    def test_score_success(self) -> None:
        judge = _make_judge(score=0.95)
        result = ContextRecall(judge=judge).score(_make_rag_output(item_id="rec001"), _make_qa_item())
        assert result is not None
        assert result.score == pytest.approx(0.95, abs=0.001)

    def test_composite_score_computed(self) -> None:
        """EvalResult.composite_score should be the mean of all non-None metric scores."""
        from rag_lens.ingestion.schema import EvalResult, MetricScore
        result = EvalResult(
            item_id="comp001",
            question="Q",
            faithfulness=MetricScore(score=0.8, rationale="r", raw_response=""),
            answer_relevance=MetricScore(score=0.6, rationale="r", raw_response=""),
            context_precision=MetricScore(score=0.9, rationale="r", raw_response=""),
            context_recall=MetricScore(score=0.7, rationale="r", raw_response=""),
        )
        assert result.composite_score == pytest.approx(0.75, abs=0.001)

    def test_composite_score_partial(self) -> None:
        """composite_score works even if only some metrics are populated."""
        from rag_lens.ingestion.schema import EvalResult, MetricScore
        result = EvalResult(
            item_id="comp002",
            question="Q",
            faithfulness=MetricScore(score=1.0, rationale="r", raw_response=""),
        )
        assert result.composite_score == pytest.approx(1.0, abs=0.001)
