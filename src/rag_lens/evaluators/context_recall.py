"""
Context Recall evaluator.

Measures: Did the retriever surface the chunks that actually contain the
information needed to produce the ground-truth answer?

This targets the retrieval layer specifically. A low recall score means the
right information exists in the corpus but the retriever failed to find it —
a different failure mode from low faithfulness (which is a generation problem).
"""

from __future__ import annotations

import logging

from rag_lens.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput
from rag_lens.judge.base import BaseJudge, JudgeCallError, JudgeParseError
from rag_lens.judge.prompts import build_context_recall_prompt

logger = logging.getLogger(__name__)


class ContextRecall:
    """Evaluate whether retrieval surfaces the required information.

    Args:
        judge: Any BaseJudge implementation.

    Example:
        evaluator = ContextRecall(judge=GroqJudge())
        result = evaluator.evaluate(rag_output=output, qa_item=item)
    """

    def __init__(self, judge: BaseJudge) -> None:
        self._judge = judge

    def score(self, rag_output: RAGOutput, qa_item: QAItem) -> MetricScore | None:
        """Score a single RAGOutput for context recall.

        Unlike other evaluators, recall requires the ground-truth answer from
        qa_item to determine what information "should" have been retrieved.

        Args:
            rag_output: The pipeline output to evaluate.
            qa_item: The ground-truth item (provides ground_truth_answer).

        Returns:
            MetricScore if successful, None on judge failure.
        """
        prompt = build_context_recall_prompt(
            question=rag_output.question,
            ground_truth_answer=qa_item.ground_truth_answer,
            retrieved_chunks=rag_output.retrieved_context,
        )
        try:
            response = self._judge.judge(user_prompt=prompt)
        except (JudgeCallError, JudgeParseError) as exc:
            logger.warning(
                "ContextRecall judge failed for item_id=%s: %s",
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
        """Run context recall evaluation and return a populated EvalResult.

        Args:
            rag_output: The pipeline output to evaluate.
            qa_item: The ground-truth item (required for recall — provides ground truth).

        Returns:
            EvalResult with context_recall populated.
        """
        metric = self.score(rag_output, qa_item)
        return EvalResult(
            item_id=rag_output.item_id,
            question=rag_output.question,
            context_recall=metric,
            error=None if metric is not None else "Judge call failed — see logs",
        )
