"""
CLI entrypoint for rag-lens.

Commands:
    rag-lens evaluate   → Score a single RAG output log against the 4 metrics
    rag-lens benchmark  → Run the full 18-config matrix sweep
    rag-lens report     → Regenerate leaderboard + charts from existing raw results

All commands print a rich-formatted summary to stdout.
Do NOT expose API keys in any log output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="rag-lens",
    help="Open-source RAG Evaluation Framework — quantitatively score your RAG pipeline.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.command()
def evaluate(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to QA ground-truth dataset (.json/.jsonl/.csv)",
        exists=True,
        readable=True,
    ),
    outputs: Path = typer.Option(
        ...,
        "--outputs",
        "-o",
        help="Path to RAG pipeline output log (.json/.jsonl)",
        exists=True,
        readable=True,
    ),
    judge_model: str = typer.Option(
        "llama-3.3-70b-versatile",
        "--judge-model",
        help="Groq model to use as the judge LLM",
    ),
) -> None:
    """[bold green]Evaluate[/bold green] a RAG output log against the 4 metrics.

    Loads the ground-truth dataset and the RAG output log, runs all 4 evaluators,
    and prints a per-item and aggregate score table.
    """
    from rag_lens.evaluators.answer_relevance import AnswerRelevance
    from rag_lens.evaluators.context_precision import ContextPrecision
    from rag_lens.evaluators.context_recall import ContextRecall
    from rag_lens.evaluators.faithfulness import Faithfulness
    from rag_lens.ingestion.dataset_loader import DatasetLoader
    from rag_lens.ingestion.schema import EvalResult
    from rag_lens.judge.groq_judge import GroqJudge

    loader = DatasetLoader()

    with console.status("Loading dataset..."):
        try:
            qa_items = loader.load_qa_dataset(dataset)
            rag_outputs = loader.load_rag_outputs(outputs)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Dataset error:[/red] {exc}")
            raise typer.Exit(1)

    qa_map = {item.item_id: item for item in qa_items}
    output_map = {o.item_id: o for o in rag_outputs}
    common_ids = set(qa_map) & set(output_map)

    if not common_ids:
        console.print("[red]No matching item_ids between dataset and outputs.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {len(qa_items)} items | "
            f"[bold]Outputs:[/bold] {len(rag_outputs)} items | "
            f"[bold]Matched:[/bold] {len(common_ids)} pairs",
            title="[bold blue]RAG-Eval[/bold blue]",
        )
    )

    try:
        judge = GroqJudge(model=judge_model)
    except OSError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    faith_eval = Faithfulness(judge=judge)
    rel_eval = AnswerRelevance(judge=judge)
    prec_eval = ContextRecall(judge=judge)
    rec_eval = ContextPrecision(judge=judge)

    results: list[EvalResult] = []
    with console.status("Running evaluations..."):
        for item_id in common_ids:
            qa = qa_map[item_id]
            out = output_map[item_id]
            faith = faith_eval.score(out)
            rel = rel_eval.score(out)
            prec = prec_eval.score(out, qa)
            rec = rec_eval.score(out)
            results.append(
                EvalResult(
                    item_id=item_id,
                    question=qa.question,
                    faithfulness=faith,
                    answer_relevance=rel,
                    context_recall=prec,
                    context_precision=rec,
                )
            )

    _print_results_table(results)


@app.command()
def benchmark(
    matrix: Path = typer.Option(
        Path("data/configs/benchmark_matrix.yaml"),
        "--matrix",
        help="YAML file declaring the 18 benchmark configurations",
    ),
    dataset: Path = typer.Option(
        Path("data/qa_dataset.json"),
        "--dataset",
        help="QA ground-truth dataset",
    ),
    corpus: Path = typer.Option(
        Path("data/corpus.json"),
        "--corpus",
        help="JSON list of raw document strings to index",
    ),
    output_dir: Path = typer.Option(
        Path("results/raw"),
        "--output-dir",
        help="Directory to write raw JSONL results",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume incomplete runs"),
) -> None:
    """[bold yellow]Benchmark[/bold yellow] — run the full 18-config matrix sweep.

    Executes all configurations declared in the matrix YAML, writing raw results
    to output_dir. Use [bold]rag-lens report[/bold] to generate the leaderboard after.
    """
    from rag_lens.ingestion.dataset_loader import DatasetLoader
    from rag_lens.pipeline.config import benchmark_matrix
    from rag_lens.pipeline.runner import run_pipeline

    loader = DatasetLoader()

    with console.status("Loading dataset and corpus..."):
        try:
            qa_items = loader.load_qa_dataset(dataset)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Dataset error:[/red] {exc}")
            raise typer.Exit(1)

        if not corpus.exists():
            console.print(f"[red]Corpus file not found:[/red] {corpus}")
            raise typer.Exit(1)
        with corpus.open(encoding="utf-8") as f:
            raw_corpus: list[str] = json.load(f)

    configs = benchmark_matrix()
    console.print(
        Panel(
            f"[bold]Configs:[/bold] {len(configs)} | [bold]Items:[/bold] {len(qa_items)} | "
            f"[bold]Corpus docs:[/bold] {len(raw_corpus)}",
            title="[bold yellow]Benchmark Sweep[/bold yellow]",
        )
    )

    for i, config in enumerate(configs, 1):
        console.print(f"\n[bold cyan][{i}/{len(configs)}] {config.config_id}[/bold cyan]")
        try:
            _, summary = run_pipeline(
                config=config,
                dataset=qa_items,
                corpus=raw_corpus,
                output_dir=output_dir,
                resume=resume,
            )
            console.print(
                f"  composite={summary.composite_score:.3f} | "
                f"errors={summary.n_errors}/{summary.n_items}"
            )
        except Exception as exc:
            console.print(f"  [red]FAILED:[/red] {exc}")
            logger.error("Config %s failed: %s", config.config_id, exc, exc_info=True)

    console.print("\n[green]Benchmark sweep complete.[/green] Run [bold]rag-lens report[/bold] to generate leaderboard.")


@app.command()
def report(
    results_dir: Path = typer.Option(
        Path("results/raw"),
        "--results",
        help="Directory containing raw .jsonl result files",
    ),
    output: Path = typer.Option(
        Path("results/leaderboard.csv"),
        "--output",
        help="Output path for leaderboard CSV",
    ),
    charts_dir: Path = typer.Option(
        Path("results/charts"),
        "--charts-dir",
        help="Directory to write comparison charts",
    ),
) -> None:
    """[bold magenta]Report[/bold magenta] — generate leaderboard and charts from existing results.

    Does not make any API calls — purely aggregates and visualizes saved data.
    Safe to run multiple times; charts and CSV are overwritten (not appended).
    """
    from rag_lens.reporting.charts import generate_charts
    from rag_lens.reporting.leaderboard import generate_leaderboard

    with console.status("Aggregating results..."):
        try:
            df = generate_leaderboard(raw_dir=results_dir, output_path=output)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Report error:[/red] {exc}")
            raise typer.Exit(1)

    _print_leaderboard_table(df)

    with console.status("Generating charts..."):
        chart_paths = generate_charts(df, output_dir=charts_dir)

    console.print(f"\n[green]✓[/green] Leaderboard: [bold]{output}[/bold]")
    for cp in chart_paths:
        console.print(f"[green]✓[/green] Chart: [bold]{cp}[/bold]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_results_table(results) -> None:
    table = Table(title="Evaluation Results", show_lines=True, style="bold")
    table.add_column("item_id", style="cyan", no_wrap=True)
    table.add_column("Faithfulness", justify="center")
    table.add_column("Relevance", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("Composite", justify="center", style="bold yellow")

    for r in results:
        table.add_row(
            r.item_id[:10],
            _fmt(r.faithfulness),
            _fmt(r.answer_relevance),
            _fmt(r.context_precision),
            _fmt(r.context_recall),
            f"{r.composite_score:.3f}" if r.composite_score is not None else "N/A",
        )

    console.print(table)


def _print_leaderboard_table(df) -> None:
    table = Table(title="Benchmark Leaderboard", show_lines=True)
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Config", style="cyan")
    table.add_column("Composite", justify="center", style="bold green")
    table.add_column("Faithfulness", justify="center")
    table.add_column("Relevance", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("Errors", justify="center", style="red")

    for _, row in df.iterrows():
        table.add_row(
            str(int(row["rank"])),
            row["config_id"],
            f"{row['composite_score']:.3f}" if row["composite_score"] else "N/A",
            f"{row['faithfulness_mean']:.3f}" if row["faithfulness_mean"] else "N/A",
            f"{row['answer_relevance_mean']:.3f}" if row["answer_relevance_mean"] else "N/A",
            f"{row['context_precision_mean']:.3f}" if row["context_precision_mean"] else "N/A",
            f"{row['context_recall_mean']:.3f}" if row["context_recall_mean"] else "N/A",
            f"{int(row['n_errors'])}/{int(row['n_items'])}",
        )

    console.print(table)


def _fmt(metric) -> str:
    return f"{metric.score:.3f}" if metric else "ERR"


if __name__ == "__main__":
    app()
