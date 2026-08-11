"""Ingestion module: data loading and schema validation."""
from rag_lens.ingestion.dataset_loader import DatasetLoader
from rag_lens.ingestion.schema import EvalResult, MetricScore, QAItem, RAGOutput, RunSummary

__all__ = ["QAItem", "RAGOutput", "EvalResult", "MetricScore", "RunSummary", "DatasetLoader"]
