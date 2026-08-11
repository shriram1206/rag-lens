"""
Context Precision evaluator.

Measures: Of the chunks that were retrieved, what fraction were actually
relevant to the question? Low precision = the retriever is surfacing a lot
of noise along with the useful information.

Compare to Context Recall (which asks: "did we retrieve the right stuff?").
Precision asks: "of what we retrieved, was it all useful?"
High recall + low precision → retriever is conservative, casting a wide net.
Low recall + high precision → retriever is narrow, missing important chunks.
"""

from __future__ import annotations

import logging

from rag_lens.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput
from rag_lens.judge.base import BaseJudge, JudgeCallError, JudgeParseError
from rag_lens.judge.prompts import build_context_precision_prompt

logger = logging.getLogger(__name__)


class ContextPrecision:
    """Evaluate the signal-to-noise ratio of the retrieved context.

    Args:
        judge: Any BaseJudge implementation.

    Example:
        evaluator = ContextPrecision(judge=GroqJudge())
        result = evaluator.evaluate(rag_output=output, qa_item=item)
    """

    def __init__(self, judge: BaseJudge) -> None:
        self._judge = judge

    def score(self, rag_output: RAGOutput) -> MetricScore | None:
        """Score a single RAGOutput for context precision.

        Args:
            rag_output: The pipeline output to evaluate.

        Returns:
            MetricScore if successful, None on judge failure.
        """
        prompt = build_context_precision_prompt(
            question=rag_output.question,
            chunks=rag_output.retrieved_context,
        )
        try:
            response = self._judge.judge(user_prompt=prompt)
        except (JudgeCallError, JudgeParseError) as exc:
            logger.warning(
                "ContextPrecision judge failed for item_id=%s: %s",
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
        """Run context precision evaluation and return a populated EvalResult.

        Args:
            rag_output: The pipeline output to evaluate.
            qa_item: The ground-truth item (used for alignment only).

        Returns:
            EvalResult with context_precision populated.
        """
        metric = self.score(rag_output)
        return EvalResult(
            item_id=rag_output.item_id,
            question=rag_output.question,
            context_precision=metric,
            error=None if metric is not None else "Judge call failed — see logs",
        )
