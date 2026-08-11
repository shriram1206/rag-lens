"""
Answer Relevance evaluator.

Measures: Does the generated answer actually address the user's question?
A high-faithfulness but low-relevance answer is one where the model correctly
cited the context but answered a different question entirely (topic drift).
"""

from __future__ import annotations

import logging

from rag_lens.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput
from rag_lens.judge.base import BaseJudge, JudgeCallError, JudgeParseError
from rag_lens.judge.prompts import build_answer_relevance_prompt

logger = logging.getLogger(__name__)


class AnswerRelevance:
    """Evaluate whether the generated answer directly addresses the question.

    Args:
        judge: Any BaseJudge implementation.

    Example:
        evaluator = AnswerRelevance(judge=GroqJudge())
        result = evaluator.evaluate(rag_output=output, qa_item=item)
    """

    def __init__(self, judge: BaseJudge) -> None:
        self._judge = judge

    def score(self, rag_output: RAGOutput) -> MetricScore | None:
        """Score a single RAGOutput for answer relevance.

        Args:
            rag_output: The pipeline output to evaluate.

        Returns:
            MetricScore if successful, None on judge failure.
        """
        prompt = build_answer_relevance_prompt(
            question=rag_output.question,
            answer=rag_output.generated_answer,
        )
        try:
            response = self._judge.judge(user_prompt=prompt)
        except (JudgeCallError, JudgeParseError) as exc:
            logger.warning(
                "AnswerRelevance judge failed for item_id=%s: %s",
                rag_output.item_id,
                exc,
            )
            return None

        return MetricScore(
            score=response.score,
            rationale=response.rationale,
            raw_response=response.raw_response,
        )

    def evaluate(self, rag_output: RAGOutput, qa_item: QAItem) -> EvalResult:
        """Run answer relevance evaluation and return a populated EvalResult.

        Args:
            rag_output: The pipeline output to evaluate.
            qa_item: The ground-truth item.

        Returns:
            EvalResult with answer_relevance populated; error field set on failure.
        """
        metric = self.score(rag_output)
        return EvalResult(
            item_id=rag_output.item_id,
            question=rag_output.question,
            answer_relevance=metric,
            error=None if metric is not None else "Judge call failed — see logs",
        )
