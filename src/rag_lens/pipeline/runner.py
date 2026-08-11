"""
Pipeline runner: orchestrates retrieve → generate → judge for a RunConfig.

Architecture:
  1. Load dataset (QA items only; no pre-existing RAG outputs mode)
  2. Build the retrieval stack from RunConfig (chunker + embedder + retriever)
  3. For each QA item: retrieve top_k chunks → (optionally) generate an answer
     using a stub or live LLM → score with all 4 evaluators
  4. Persist raw EvalResult objects to JSONL BEFORE aggregation
  5. Aggregate and return RunSummary

Persistence-before-aggregation is critical: raw results are never overwritten,
so aggregation logic can be rerun, new statistics computed, or individual scores
audited without repeating the (expensive) judge API calls.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_lens.evaluators.answer_relevance import AnswerRelevance
from rag_lens.evaluators.context_precision import ContextPrecision
from rag_lens.evaluators.context_recall import ContextRecall
from rag_lens.evaluators.faithfulness import Faithfulness
from rag_lens.ingestion.schema import EvalResult, QAItem, RAGOutput, RunSummary
from rag_lens.judge.base import BaseJudge
from rag_lens.judge.groq_judge import GroqJudge
from rag_lens.pipeline.config import RunConfig
from rag_lens.retrieval.chunking import get_chunker
from rag_lens.retrieval.embeddings import get_embedder
from rag_lens.retrieval.retrievers import get_retriever

logger = logging.getLogger(__name__)

_RESULTS_RAW_DIR = Path("results/raw")


def run_pipeline(
    config: RunConfig,
    dataset: list[QAItem],
    corpus: list[str],
    answer_generator: Callable[[str, list[str]], str] | None = None,
    judge: BaseJudge | None = None,
    output_dir: Path | None = None,
    resume: bool = True,
) -> tuple[list[EvalResult], RunSummary]:
    """Run the full RAG eval pipeline for a single RunConfig.

    Args:
        config: The benchmark run configuration (chunking × embedding × retrieval).
        dataset: List of ground-truth QAItem objects.
        corpus: Raw document strings to index (the knowledge base).
        answer_generator: Optional callable (question, chunks) → answer string.
                          If None, uses the joined retrieved chunks as the "answer"
                          (retrieval-only evaluation mode).
        judge: BaseJudge instance. Defaults to GroqJudge() if None.
        output_dir: Directory to write raw JSONL results. Defaults to results/raw/.
        resume: If True and a partial results file exists for this config, skip
                already-evaluated items rather than re-running them.

    Returns:
        Tuple of (list[EvalResult], RunSummary).

    Side effects:
        Writes/appends per-item results to
        {output_dir}/{config.config_id}.jsonl as they are produced,
        before any aggregation occurs.
    """
    out_dir = output_dir or _RESULTS_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{config.config_id}.jsonl"

    judge = judge or GroqJudge()

    # --- Build retrieval stack ---
    logger.info("[%s] Building retrieval stack...", config.config_id)
    chunker = get_chunker(config.chunking_strategy)
    embedder = get_embedder(config.embedding_model)
    retriever = get_retriever(config.retrieval_method, embedder)

    all_chunks: list[str] = []
    for doc in corpus:
        all_chunks.extend(chunker.chunk(doc))
    logger.info("[%s] Indexed %d chunks from %d documents", config.config_id, len(all_chunks), len(corpus))
    retriever.index(all_chunks)

    # --- Resume: load already-evaluated item IDs ---
    evaluated_ids: set[str] = set()
    completed_results: list[EvalResult] = []
    if resume and results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw = json.loads(line)
                        er = EvalResult.model_validate(raw)
                        completed_results.append(er)
                        evaluated_ids.add(er.item_id)
                    except Exception:
                        pass  # Partial/corrupt line — ignore and re-evaluate
        if evaluated_ids:
            logger.info("[%s] Resuming: %d items already evaluated", config.config_id, len(evaluated_ids))

    # --- Evaluators ---
    faithfulness_eval = Faithfulness(judge=judge)
    relevance_eval = AnswerRelevance(judge=judge)
    precision_eval = ContextPrecision(judge=judge)
    recall_eval = ContextRecall(judge=judge)

    new_results: list[EvalResult] = []
    pending = [item for item in dataset if item.item_id not in evaluated_ids]

    logger.info("[%s] Evaluating %d items...", config.config_id, len(pending))

    with results_path.open("a", encoding="utf-8") as out_f:
        for i, qa_item in enumerate(pending):
            t0 = time.perf_counter()
            try:
                # Step 1: Retrieve
                retrieved_chunks = retriever.retrieve(qa_item.question, top_k=config.top_k)

                # Step 2: Generate answer (or use stub)
                if answer_generator:
                    answer = answer_generator(qa_item.question, retrieved_chunks)
                else:
                    # Retrieval-only mode: join chunks as the "answer"
                    answer = " ".join(retrieved_chunks[:2])

                # Step 3: Build RAGOutput
                rag_output = RAGOutput(
                    item_id=qa_item.item_id,
                    question=qa_item.question,
                    retrieved_context=retrieved_chunks,
                    generated_answer=answer,
                )

                # Step 4: Score all 4 metrics
                faith = faithfulness_eval.score(rag_output)
                relevance = relevance_eval.score(rag_output)
                precision = precision_eval.score(rag_output)
                recall = recall_eval.score(rag_output, qa_item)

                result = EvalResult(
                    item_id=qa_item.item_id,
                    question=qa_item.question,
                    faithfulness=faith,
                    answer_relevance=relevance,
                    context_precision=precision,
                    context_recall=recall,
                )

            except Exception as exc:
                logger.error("[%s] Item %s failed: %s", config.config_id, qa_item.item_id, exc)
                result = EvalResult(
                    item_id=qa_item.item_id,
                    question=qa_item.question,
                    error=f"Pipeline error: {exc}",
                )

            # Persist BEFORE aggregation (append-only)
            out_f.write(result.model_dump_json() + "\n")
            out_f.flush()
            new_results.append(result)

            elapsed = time.perf_counter() - t0
            logger.info(
                "[%s] %d/%d item_id=%s composite=%.3f elapsed=%.1fs",
                config.config_id,
                i + 1,
                len(pending),
                result.item_id,
                result.composite_score or -1,
                elapsed,
            )

    all_results = completed_results + new_results
    summary = _aggregate(config, all_results)
    logger.info(
        "[%s] Done. composite=%.3f n_errors=%d",
        config.config_id,
        summary.composite_score or -1,
        summary.n_errors,
    )
    return all_results, summary


def _aggregate(config: RunConfig, results: list[EvalResult]) -> RunSummary:
    """Aggregate per-item EvalResults into a RunSummary with mean + std."""

    def _stats(scores: list[float]) -> tuple[float | None, float | None]:
        if not scores:
            return None, None
        arr = np.array(scores)
        return round(float(arr.mean()), 4), round(float(arr.std()), 4)

    faith_scores = [r.faithfulness.score for r in results if r.faithfulness]
    rel_scores = [r.answer_relevance.score for r in results if r.answer_relevance]
    prec_scores = [r.context_precision.score for r in results if r.context_precision]
    rec_scores = [r.context_recall.score for r in results if r.context_recall]

    f_mean, f_std = _stats(faith_scores)
    r_mean, r_std = _stats(rel_scores)
    p_mean, p_std = _stats(prec_scores)
    rc_mean, rc_std = _stats(rec_scores)

    all_composites = [r.composite_score for r in results if r.composite_score is not None]
    composite, _ = _stats(all_composites)

    return RunSummary(
        config_id=config.config_id,
        chunking_strategy=config.chunking_strategy,
        embedding_model=config.embedding_model,
        retrieval_method=config.retrieval_method,
        n_items=len(results),
        n_errors=sum(1 for r in results if r.error),
        faithfulness_mean=f_mean,
        faithfulness_std=f_std,
        answer_relevance_mean=r_mean,
        answer_relevance_std=r_std,
        context_precision_mean=p_mean,
        context_precision_std=p_std,
        context_recall_mean=rc_mean,
        context_recall_std=rc_std,
        composite_score=composite,
    )
