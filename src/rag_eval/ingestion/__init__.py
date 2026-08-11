"""Ingestion module: data loading and schema validation."""
from rag_eval.ingestion.schema import QAItem, RAGOutput, EvalResult, MetricScore, RunSummary
from rag_eval.ingestion.dataset_loader import DatasetLoader

__all__ = ["QAItem", "RAGOutput", "EvalResult", "MetricScore", "RunSummary", "DatasetLoader"]
