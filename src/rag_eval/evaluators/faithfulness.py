"""
Faithfulness evaluator.

Measures: Does the generated answer contain only claims that are grounded in
the retrieved context? A low score means the model hallucinated.

This is the most important metric for production RAG systems — hallucinated
answers are the primary failure mode that causes user distrust.
"""

from __future__ import annotations

import logging

from rag_eval.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput
from rag_eval.judge.base import BaseJudge, JudgeCallError, JudgeParseError
from rag_eval.judge.prompts import build_faithfulness_prompt

logger = logging.getLogger(__name__)


class Faithfulness:
    """Evaluate whether a RAG pipeline's answer is grounded in the retrieved context.

    Args:
        judge: Any BaseJudge implementation (GroqJudge, MockJudge, etc.).

    Example:
        evaluator = Faithfulness(judge=GroqJudge())
        result = evaluator.evaluate(rag_output=output, qa_item=item)
        print(result.faithfulness.score)  # → 0.92
    """

    def __init__(self, judge: BaseJudge) -> None:
        self._judge = judge

    def score(self, rag_output: RAGOutput) -> MetricScore | None:
        """Score a single RAGOutput for faithfulness.

        Args:
            rag_output: The pipeline output to evaluate.

        Returns:
            MetricScore if successful, None on judge failure (logged as warning).
        """
        combined_context = "\n\n".join(rag_output.retrieved_context)
        prompt = build_faithfulness_prompt(
            context=combined_context,
            answer=rag_output.generated_answer,
        )
        try:
            response = self._judge.judge(user_prompt=prompt)
        except (JudgeCallError, JudgeParseError) as exc:
            logger.warning(
                "Faithfulness judge failed for item_id=%s: %s",
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
        """Run faithfulness evaluation and return a populated EvalResult.

        Args:
            rag_output: The pipeline output to evaluate.
            qa_item: The ground-truth item (used only for item_id/question alignment).

        Returns:
            EvalResult with faithfulness populated; error field set on failure.
        """
        if rag_output.item_id != qa_item.item_id:
            logger.warning(
                "item_id mismatch: rag_output.item_id=%s, qa_item.item_id=%s",
                rag_output.item_id,
                qa_item.item_id,
            )

        metric = self.score(rag_output)
        return EvalResult(
            item_id=rag_output.item_id,
            question=rag_output.question,
            faithfulness=metric,
            error=None if metric is not None else "Judge call failed — see logs",
        )
