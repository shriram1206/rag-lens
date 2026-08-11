# rag-lens

**An open-source, pip-installable Python library for quantitatively evaluating RAG pipelines using LLM-as-a-Judge.**

> *"I don't just build RAG systems — I built the tool that measures whether RAG systems work."*

[![CI](https://github.com/shriram1206/rag-eval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/shriram1206/rag-eval-framework/actions)
[![PyPI version](https://badge.fury.io/py/rag-lens.svg)](https://badge.fury.io/py/rag-lens)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why This Exists

RAG systems are easy to prototype and hard to trust. Most teams evaluate them by manually reading a handful of outputs and eyeballing whether the answers "look right." This breaks down at scale, doesn't catch regressions, and can't tell you *why* a pipeline underperforms.

**RAG-Eval Framework** replaces vibes-based evaluation with a scored, reproducible pipeline that separates retrieval quality from generation quality.

---

## Quickstart

```bash
pip install rag-lens
cp .env.example .env  # Add your GROQ_API_KEY
```

```python
from rag_eval.evaluators import Faithfulness, AnswerRelevance
from rag_eval.judge import GroqJudge
from rag_eval.ingestion import QAItem, RAGOutput

judge = GroqJudge()

output = RAGOutput(
    item_id="test_001",
    question="What is RAG?",
    retrieved_context=["RAG combines retrieval with generation to reduce hallucinations."],
    generated_answer="RAG is a technique that retrieves documents before generating an answer.",
)

score = Faithfulness(judge=judge).score(output)
print(score.score)      # → 0.95
print(score.rationale)  # → "The answer is fully grounded in the retrieved context."
```

---

## The 4 Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are all claims in the answer grounded in the retrieved context? (anti-hallucination) |
| **Answer Relevance** | Does the answer directly address the question? |
| **Context Precision** | Of the retrieved chunks, what fraction were actually relevant? (signal-to-noise) |
| **Context Recall** | Did retrieval surface the information needed to answer the question? |

All scores in **[0.0, 1.0]**. All scored by an LLM judge (Llama-3 via Groq).

---

## CLI Usage

```bash
# Evaluate a single RAG output log
rag-eval evaluate --dataset data/qa_dataset.json --outputs my_rag_outputs.jsonl

# Run the full 18-configuration benchmark sweep
rag-eval benchmark --matrix data/configs/benchmark_matrix.yaml

# Regenerate leaderboard + charts from existing results (no API calls)
rag-eval report --results results/raw/
```

---

## 18-Configuration Benchmark

The framework ships a benchmark that tests **18 RAG configurations** against a 100-question dataset:

| Dimension | Options |
|---|---|
| Chunking | Sentence / Paragraph / Semantic |
| Embedding | ada-002 (OpenAI) / BGE-large (local) / E5-large (local) |
| Retrieval | Dense (ChromaDB) / Hybrid (Dense + BM25 via RRF) |

**Run it:**
```bash
python benchmarks/run_18_config_benchmark.py
```

---

## Judge Validation

LLM-as-a-judge scores are estimates, not ground truth. Before trusting benchmark conclusions, we validate the judge against human labels on a 20–30 item subset. See `docs/qa-testing.md §4` for the methodology. **Agreement rates are documented honestly — not hidden.**

> ⚠️ **Data notice:** This tool sends dataset and RAG output text to the Groq API to power the judge. Do not use with sensitive or confidential data.

---

## Project Structure

See `docs/project-structure.md` for a full annotated layout.

```
src/rag_eval/
├── judge/        → BaseJudge + GroqJudge (swappable LLM backend)
├── evaluators/   → Faithfulness, AnswerRelevance, ContextPrecision, ContextRecall
├── ingestion/    → Pydantic schemas + dataset loader
├── retrieval/    → 3 chunkers + 3 embedding wrappers + 2 retrievers
├── pipeline/     → RunConfig + benchmark runner (with resume support)
└── reporting/    → Leaderboard CSV + 4 matplotlib charts
```

---

## Installation (from source)

```bash
git clone https://github.com/shriram1206/rag-eval-framework.git
cd rag-eval-framework
pip install -e ".[dev]"
cp .env.example .env  # Fill in GROQ_API_KEY
pytest tests/unit tests/integration  # All tests use mocked judge — free to run
```

---

## Tech Stack

Python 3.11+ · Groq (Llama-3.1-70B) · ChromaDB · sentence-transformers · rank_bm25 · Pydantic v2 · pandas · matplotlib · typer · tenacity

---

## Author

**Shriram M** · [GitHub](https://github.com/shriram1206) · [LinkedIn](https://linkedin.com/in/shriram-m-sde) · shriram.coder@gmail.com

MIT License
