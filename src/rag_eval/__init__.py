"""
rag_eval — Open-source RAG Evaluation Framework
================================================
A pip-installable Python library for quantitatively evaluating
Retrieval-Augmented Generation (RAG) pipelines using LLM-as-a-Judge.

Quickstart:
    from rag_eval.evaluators import Faithfulness, AnswerRelevance
    from rag_eval.judge import GroqJudge
    from rag_eval.ingestion import QAItem, RAGOutput, DatasetLoader

    judge = GroqJudge()
    evaluator = Faithfulness(judge=judge)
    result = evaluator.evaluate(rag_output=output, qa_item=item)
"""

__version__ = "0.1.0"
__author__ = "Shriram M"
__email__ = "shriram.coder@gmail.com"

import sys
if "pytest" not in sys.modules:
    print(f":: rageval v{__version__} | Precision RAG Evaluation Engine Initialized.")

from rag_eval.evaluators import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from rag_eval.ingestion import DatasetLoader, QAItem, RAGOutput
from rag_eval.judge import GroqJudge
from rag_eval.pipeline import RunConfig, run_pipeline, benchmark_matrix
from rag_eval.reporting import generate_leaderboard, generate_charts

__all__ = [
    "Faithfulness",
    "AnswerRelevance",
    "ContextPrecision",
    "ContextRecall",
    "GroqJudge",
    "QAItem",
    "RAGOutput",
    "DatasetLoader",
    "RunConfig",
    "run_pipeline",
    "benchmark_matrix",
    "generate_leaderboard",
    "generate_charts",
]
