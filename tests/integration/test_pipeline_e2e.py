"""
Integration test: full pipeline end-to-end with a mocked judge.

Validates the RunConfig → retrieve → generate → score → JSONL write path
using a small synthetic fixture dataset and a deterministic mock judge.
No live API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_eval.ingestion.schema import QAItem, EvalResult
from rag_eval.judge.base import BaseJudge, JudgeResponse
from rag_eval.pipeline.config import RunConfig
from rag_eval.pipeline.runner import run_pipeline


MINI_CORPUS = [
    "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
    "It emphasizes code readability and simplicity. Python supports multiple programming paradigms.",
    "Machine learning is a subset of artificial intelligence. It allows computers to learn from data "
    "without being explicitly programmed. Supervised learning uses labeled training data.",
    "FastAPI is a modern Python web framework for building APIs. It is based on standard Python type "
    "hints and automatically generates OpenAPI documentation. FastAPI is known for high performance.",
]

MINI_DATASET = [
    QAItem(
        item_id="item_001",
        question="Who created Python?",
        ground_truth_answer="Python was created by Guido van Rossum in 1991.",
        ground_truth_context="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
    ),
    QAItem(
        item_id="item_002",
        question="What is machine learning?",
        ground_truth_answer="Machine learning is a subset of AI that allows computers to learn from data.",
        ground_truth_context="Machine learning is a subset of artificial intelligence. It allows computers to learn from data.",
    ),
    QAItem(
        item_id="item_003",
        question="What is FastAPI?",
        ground_truth_answer="FastAPI is a modern Python web framework for building high-performance APIs.",
        ground_truth_context="FastAPI is a modern Python web framework for building APIs known for high performance.",
    ),
]


def _make_deterministic_judge(score: float = 0.8) -> BaseJudge:
    judge = MagicMock(spec=BaseJudge)
    judge.judge.return_value = JudgeResponse(
        score=score,
        rationale="Mock evaluation",
        raw_response=f'{{"score": {score}, "rationale": "Mock evaluation"}}',
        prompt_version="v1.0.0",
    )
    return judge


class TestPipelineE2E:
    def test_run_pipeline_produces_correct_number_of_results(self, tmp_path: Path) -> None:
        config = RunConfig(
            chunking_strategy="sentence",
            embedding_model="bge-large",
            retrieval_method="dense",
        )
        judge = _make_deterministic_judge(score=0.85)

        results, summary = run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
        )

        assert len(results) == 3
        assert summary.n_items == 3
        assert summary.n_errors == 0

    def test_run_pipeline_writes_jsonl_file(self, tmp_path: Path) -> None:
        config = RunConfig(
            chunking_strategy="paragraph",
            embedding_model="bge-large",
            retrieval_method="dense",
        )
        judge = _make_deterministic_judge()

        run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
        )

        results_file = tmp_path / f"{config.config_id}.jsonl"
        assert results_file.exists()
        lines = [l for l in results_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_jsonl_lines_are_valid_eval_results(self, tmp_path: Path) -> None:
        config = RunConfig(
            chunking_strategy="sentence",
            embedding_model="bge-large",
            retrieval_method="hybrid",
        )
        judge = _make_deterministic_judge(score=0.7)

        run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
        )

        results_file = tmp_path / f"{config.config_id}.jsonl"
        for line in results_file.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            er = EvalResult.model_validate(data)
            # Every item should have a valid composite score
            assert er.composite_score is not None or er.error is not None

    def test_run_pipeline_summary_composite_score_in_range(self, tmp_path: Path) -> None:
        config = RunConfig(
            chunking_strategy="sentence",
            embedding_model="bge-large",
            retrieval_method="dense",
        )
        judge = _make_deterministic_judge(score=0.9)

        _, summary = run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
        )

        assert summary.composite_score is not None
        assert 0.0 <= summary.composite_score <= 1.0

    def test_resume_skips_already_evaluated_items(self, tmp_path: Path) -> None:
        """Second run should not re-evaluate items already in the JSONL file."""
        config = RunConfig(
            chunking_strategy="sentence",
            embedding_model="bge-large",
            retrieval_method="dense",
        )
        judge = _make_deterministic_judge()

        # First run
        run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
            resume=True,
        )
        first_call_count = judge.judge.call_count

        # Second run — should skip all items
        run_pipeline(
            config=config,
            dataset=MINI_DATASET,
            corpus=MINI_CORPUS,
            judge=judge,
            output_dir=tmp_path,
            resume=True,
        )
        # No additional judge calls should be made
        assert judge.judge.call_count == first_call_count
