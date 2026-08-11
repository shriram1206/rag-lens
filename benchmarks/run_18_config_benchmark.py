#!/usr/bin/env python3
"""
Run the full 18-configuration benchmark sweep.

Usage:
    python benchmarks/run_18_config_benchmark.py

Prerequisites:
  - GROQ_API_KEY set in .env
  - Package installed: pip install -e .
  - data/qa_dataset.json must exist
  - data/corpus.json must exist (list of raw document strings)

Results are written to results/raw/<config_id>.jsonl
After this completes, run: rag-lens report --results results/raw/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

load_dotenv()

from rag_lens.ingestion.dataset_loader import DatasetLoader
from rag_lens.judge.groq_judge import GroqJudge
from rag_lens.pipeline.config import RunConfig, benchmark_matrix
from rag_lens.pipeline.runner import run_pipeline
from rag_lens.reporting.leaderboard import generate_leaderboard
from rag_lens.reporting.charts import generate_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

DATASET_PATH = Path("data/qa_dataset.json")
CORPUS_PATH = Path("data/corpus.json")
RESULTS_DIR = Path("results/raw")


def build_corpus_from_dataset(dataset_path: Path) -> list[str]:
    """Build a corpus from the ground_truth_context fields of the QA dataset.

    In this benchmark, each QA item's ground_truth_context IS the relevant
    document. We treat the full context collection as the corpus to index,
    making this a self-contained benchmark that doesn't require external documents.
    """
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = data["data"] if isinstance(data, dict) else data
    return [item["ground_truth_context"] for item in items if item.get("ground_truth_context")]

def main() -> None:
    # --- Load dataset ---
    loader = DatasetLoader()
    try:
        qa_items = loader.load_qa_dataset(DATASET_PATH)
        # Limit to 2 items to minimize API usage for testing purposes
        qa_items = qa_items[:2]
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Dataset error:[/red] {exc}")
        sys.exit(1)

    # --- Build benchmark matrix ---
    configs: list[RunConfig] = benchmark_matrix()
    
    console.print(
        Panel(
            "[bold blue]RAG-Eval Framework[/bold blue] — Benchmark Sweep\n"
            f"Running {len(configs)} configurations across {len(qa_items)} items.\n"
            "Results are written to results/raw/ after each configuration.",
            title="Starting Benchmark",
        )
    )
    console.print(f"[green]✓[/green] Loaded {len(qa_items)} QA items from {DATASET_PATH}")
    
    # --- Build or load corpus ---
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open(encoding="utf-8") as f:
            corpus = json.load(f)
        console.print(f"[green]✓[/green] Loaded {len(corpus)} corpus documents from {CORPUS_PATH}")
    else:
        corpus = build_corpus_from_dataset(DATASET_PATH)
        # Save for reproducibility
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS_PATH.open("w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        console.print(f"[yellow]→[/yellow] Built corpus from dataset contexts: {len(corpus)} docs → saved to {CORPUS_PATH}")

    # --- Initialize judge ---
    try:
        judge = GroqJudge()
        console.print(f"[green]✓[/green] Judge: {judge.model_name}")
    except EnvironmentError as exc:
        console.print(f"[red]Judge init failed:[/red] {exc}")
        sys.exit(1)

    console.print(f"[green]✓[/green] {len(configs)} configurations to run\n")

    # --- Execute ---
    total_start = time.perf_counter()
    results_summary = []

    for i, config in enumerate(configs, 1):
        console.print(f"[bold cyan][{i:02d}/{len(configs)}] {config.config_id}[/bold cyan]")
        t0 = time.perf_counter()
        try:
            _, summary = run_pipeline(
                config=config,
                dataset=qa_items,
                corpus=corpus,
                judge=judge,
                output_dir=RESULTS_DIR,
                resume=True,
            )
            elapsed = time.perf_counter() - t0
            results_summary.append(summary)
            console.print(
                f"     composite={summary.composite_score or 'N/A':>6}  "
                f"faithfulness={summary.faithfulness_mean or 'N/A'}  "
                f"errors={summary.n_errors}/{summary.n_items}  "
                f"time={elapsed:.0f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            console.print(f"     [red]FAILED in {elapsed:.0f}s:[/red] {exc}")
            logger.error("Config %s failed", config.config_id, exc_info=True)

    total_elapsed = time.perf_counter() - total_start
    console.print(f"\n[green]All configurations complete in {total_elapsed/60:.1f} minutes.[/green]")

    # --- Generate reports ---
    console.print("\n[bold]Generating leaderboard and charts...[/bold]")
    try:
        df = generate_leaderboard(raw_dir=RESULTS_DIR)
        console.print(f"[green]✓[/green] Leaderboard: results/leaderboard.csv ({len(df)} configs)")
        chart_paths = generate_charts(df)
        for cp in chart_paths:
            console.print(f"[green]✓[/green] Chart: {cp}")
    except Exception as exc:
        console.print(f"[yellow]Report generation failed:[/yellow] {exc}")

    # --- Print top 5 ---
    if results_summary:
        sorted_results = sorted(
            results_summary,
            key=lambda s: s.composite_score or 0,
            reverse=True
        )
        console.print("\n[bold yellow]Top 5 Configurations:[/bold yellow]")
        for rank, s in enumerate(sorted_results[:5], 1):
            console.print(
                f"  #{rank} {s.config_id:<45} composite={s.composite_score:.4f}"
            )


if __name__ == "__main__":
    main()
