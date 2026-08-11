"""
Benchmark visualization: matplotlib/seaborn charts for the leaderboard.

Generated charts:
  1. Metric comparison by chunking strategy (grouped bar chart)
  2. Metric comparison by embedding model (grouped bar chart)
  3. Metric comparison by retrieval method (grouped bar chart)
  4. Composite score heatmap (config_id × metric)

All charts are written to results/charts/ as high-resolution PNGs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_METRICS = ["faithfulness_mean", "answer_relevance_mean", "context_precision_mean", "context_recall_mean"]
_METRIC_LABELS = ["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"]
_CHART_DIR = Path("results/charts")

_PALETTE = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585"]


def generate_charts(
    leaderboard_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate all benchmark comparison charts from the leaderboard DataFrame.

    Args:
        leaderboard_df: Output of generate_leaderboard() — one row per config.
        output_dir: Directory to save PNG charts. Defaults to results/charts/.

    Returns:
        List of Path objects pointing to the generated chart files.
    """
    out_dir = output_dir or _CHART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.append(_grouped_bar_by_dimension(leaderboard_df, "chunking_strategy", out_dir))
    generated.append(_grouped_bar_by_dimension(leaderboard_df, "embedding_model", out_dir))
    generated.append(_grouped_bar_by_dimension(leaderboard_df, "retrieval_method", out_dir))
    generated.append(_composite_heatmap(leaderboard_df, out_dir))

    logger.info("Generated %d charts in %s", len(generated), out_dir)
    return generated


def _grouped_bar_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    out_dir: Path,
) -> Path:
    """Grouped bar chart: mean metric score per group in `dimension`.

    Args:
        df: Leaderboard DataFrame.
        dimension: Column to group by ("chunking_strategy", "embedding_model", "retrieval_method").
        out_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    grouped = df.groupby(dimension)[_METRICS].mean().reset_index()
    groups = grouped[dimension].tolist()
    n_groups = len(groups)
    n_metrics = len(_METRICS)

    x = np.arange(n_groups)
    bar_width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.5), 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    for i, (metric, label, color) in enumerate(zip(_METRICS, _METRIC_LABELS, _PALETTE, strict=False)):
        vals = grouped[metric].fillna(0).tolist()
        bars = ax.bar(x + offsets[i], vals, width=bar_width, label=label, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="white",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, color="white", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.set_xlabel(dimension.replace("_", " ").title(), color="white", fontsize=11)
    ax.set_ylabel("Mean Score", color="white", fontsize=11)
    ax.set_title(f"RAG Metrics by {dimension.replace('_', ' ').title()}", color="white", fontsize=13, fontweight="bold")
    ax.legend(
        frameon=True,
        framealpha=0.3,
        labelcolor="white",
        facecolor="#1a1a2e",
        fontsize=9,
    )
    ax.grid(axis="y", color="#333355", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / f"metrics_by_{dimension}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("Saved chart: %s", out_path)
    return out_path


def _composite_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of per-config scores across all 4 metrics.

    Args:
        df: Leaderboard DataFrame.
        out_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    heat_df = df.set_index("config_id")[_METRICS].copy()
    heat_df.columns = _METRIC_LABELS  # type: ignore[assignment]
    heat_df = heat_df.sort_values("Faithfulness", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(heat_df) * 0.45)))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    data = heat_df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(_METRIC_LABELS)))
    ax.set_xticklabels(_METRIC_LABELS, color="white", fontsize=10)
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df.index.tolist(), color="white", fontsize=8)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.3 < val < 0.7 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Score", color="white")

    ax.set_title(f"{len(heat_df)}-Config Benchmark Heatmap", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "benchmark_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("Saved heatmap: %s", out_path)
    return out_path
