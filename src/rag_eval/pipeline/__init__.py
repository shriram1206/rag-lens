"""Pipeline module: RunConfig and pipeline runner."""
from rag_eval.pipeline.config import RunConfig, benchmark_matrix
from rag_eval.pipeline.runner import run_pipeline

__all__ = ["RunConfig", "benchmark_matrix", "run_pipeline"]
