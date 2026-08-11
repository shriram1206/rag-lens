"""Ingestion module: data loading and schema validation."""
from rag_lens.ingestion.schema import QAItem, RAGOutput, EvalResult, MetricScore, RunSummary
from rag_lens.ingestion.dataset_loader import DatasetLoader

__all__ = ["QAItem", "RAGOutput", "EvalResult", "MetricScore", "RunSummary", "DatasetLoader"]
