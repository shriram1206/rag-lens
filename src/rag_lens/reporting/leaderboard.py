"""
Leaderboard generation: aggregates raw per-item JSONL results into a ranked CSV.

Design:
  - Reads only from results/raw/*.jsonl (never overwrites those files)
  - Outputs a composite-score-ranked leaderboard.csv
  - Reports mean ± std per metric, not single-point estimates
  - Any config with too many errors is flagged in the output with a warning
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rag_lens.ingestion.schema import EvalResult, RunSummary
from rag_lens.pipeline.config import RunConfig

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("results/raw")
_DEFAULT_OUTPUT = Path("results/leaderboard.csv")
_ERROR_RATE_WARN_THRESHOLD = 0.10  # Warn if >10% of items had errors


def generate_leaderboard(
    raw_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Read all raw JSONL result files and generate a ranked leaderboard.

    Args:
        raw_dir: Directory containing per-config .jsonl files.
                 Defaults to results/raw/.
        output_path: Where to write the CSV output.
                     Defaults to results/leaderboard.csv.

    Returns:
        pandas DataFrame with one row per config, sorted descending by composite score.

    Raises:
        FileNotFoundError: If raw_dir does not exist or contains no .jsonl files.
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR
    output_path = output_path or _DEFAULT_OUTPUT

    if not raw_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {raw_dir}")

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl result files found in {raw_dir}")

    rows = []
    for jsonl_path in jsonl_files:
        config_id = jsonl_path.stem
        results = _load_results(jsonl_path)
        if not results:
            logger.warning("Skipping empty results file: %s", jsonl_path)
            continue

        summary = _compute_summary(config_id, results)
        row = _summary_to_row(summary)
        rows.append(row)

        if summary.error_rate > _ERROR_RATE_WARN_THRESHOLD:
            logger.warning(
                "Config '%s' has a high error rate: %.1f%% — results may be unreliable",
                config_id,
                summary.error_rate * 100,
            )

    if not rows:
        raise ValueError("No valid result files could be loaded from %s" % raw_dir)

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info("Leaderboard written to %s (%d configs)", output_path, len(df))

    return df


def _load_results(path: Path) -> list[EvalResult]:
    """Load and parse a single .jsonl results file."""
    results = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(EvalResult.model_validate(data))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return results


def _compute_summary(config_id: str, results: list[EvalResult]) -> RunSummary:
    """Compute mean ± std for each metric from a list of EvalResults."""

    def _stats(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        arr = np.array(vals)
        return round(float(arr.mean()), 4), round(float(arr.std()), 4)

    parts = config_id.split("__")
    chunking, embedding, retrieval = (parts + ["", "", ""])[:3]

    faith = [r.faithfulness.score for r in results if r.faithfulness]
    rel = [r.answer_relevance.score for r in results if r.answer_relevance]
    prec = [r.context_precision.score for r in results if r.context_precision]
    rec = [r.context_recall.score for r in results if r.context_recall]
    composites = [r.composite_score for r in results if r.composite_score is not None]

    f_m, f_s = _stats(faith)
    r_m, r_s = _stats(rel)
    p_m, p_s = _stats(prec)
    rc_m, rc_s = _stats(rec)
    comp, _ = _stats(composites)

    return RunSummary(
        config_id=config_id,
        chunking_strategy=chunking,
        embedding_model=embedding,
        retrieval_method=retrieval,
        n_items=len(results),
        n_errors=sum(1 for r in results if r.error),
        faithfulness_mean=f_m,
        faithfulness_std=f_s,
        answer_relevance_mean=r_m,
        answer_relevance_std=r_s,
        context_precision_mean=p_m,
        context_precision_std=p_s,
        context_recall_mean=rc_m,
        context_recall_std=rc_s,
        composite_score=comp,
    )


def _summary_to_row(s: RunSummary) -> dict:
    """Flatten RunSummary to a dict for the DataFrame."""
    return {
        "config_id": s.config_id,
        "chunking_strategy": s.chunking_strategy,
        "embedding_model": s.embedding_model,
        "retrieval_method": s.retrieval_method,
        "n_items": s.n_items,
        "n_errors": s.n_errors,
        "error_rate": s.error_rate,
        "composite_score": s.composite_score,
        "faithfulness_mean": s.faithfulness_mean,
        "faithfulness_std": s.faithfulness_std,
        "answer_relevance_mean": s.answer_relevance_mean,
        "answer_relevance_std": s.answer_relevance_std,
        "context_precision_mean": s.context_precision_mean,
        "context_precision_std": s.context_precision_std,
        "context_recall_mean": s.context_recall_mean,
        "context_recall_std": s.context_recall_std,
    }
