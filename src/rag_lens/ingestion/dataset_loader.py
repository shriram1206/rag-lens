"""
Dataset loader: ingests QA triples and RAG output logs into validated Pydantic models.

Supports JSON, JSONL, and CSV formats. Fails fast with a clear per-row error
rather than silently dropping malformed records.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from rag_lens.ingestion.schema import QAItem, RAGOutput

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and validate QA datasets and RAG output logs.

    All methods raise ValueError with a descriptive message on the first
    schema violation rather than silently skipping rows — silent drops would
    corrupt benchmark results without any visible signal.

    Example:
        loader = DatasetLoader()
        items = loader.load_qa_dataset("data/qa_dataset.json")
        outputs = loader.load_rag_outputs("results/run_001.jsonl")
    """

    def load_qa_dataset(self, path: str | Path) -> list[QAItem]:
        """Load and validate a ground-truth QA dataset.

        Args:
            path: Path to a JSON, JSONL, or CSV file. JSON may be a list of
                  objects or an object with a "data" key wrapping a list.

        Returns:
            List of validated QAItem objects.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If any row fails schema validation (includes row index).
        """
        path = Path(path)
        self._assert_exists(path)
        records = self._read_file(path)
        return self._validate_records(records, QAItem, "qa_dataset")

    def load_rag_outputs(self, path: str | Path) -> list[RAGOutput]:
        """Load and validate a RAG pipeline output log.

        Args:
            path: Path to a JSON or JSONL file.

        Returns:
            List of validated RAGOutput objects.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If any row fails schema validation.
        """
        path = Path(path)
        self._assert_exists(path)
        records = self._read_file(path)
        return self._validate_records(records, RAGOutput, "rag_outputs")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    def _read_file(self, path: Path) -> list[dict]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._read_json(path)
        if suffix == ".jsonl":
            return self._read_jsonl(path)
        if suffix == ".csv":
            return self._read_csv(path)
        raise ValueError(
            f"Unsupported file format '{suffix}'. Accepted: .json, .jsonl, .csv"
        )

    @staticmethod
    def _read_json(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        # Accept either a bare list or {"data": [...]}
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        raise ValueError(
            f"JSON dataset at {path} must be a list or a dict with a 'data' key"
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL parse error on line {i}: {exc}") from exc
        return records

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def _validate_records(
        records: list[dict],
        model_class: type[QAItem] | type[RAGOutput],
        context: str,
    ) -> list[QAItem] | list[RAGOutput]:
        validated = []
        for i, record in enumerate(records):
            try:
                validated.append(model_class.model_validate(record))
            except ValidationError as exc:
                raise ValueError(
                    f"{context}: row {i} failed validation — {exc.error_count()} error(s):\n"
                    + "\n".join(
                        f"  [{e['loc']}] {e['msg']}" for e in exc.errors()
                    )
                ) from exc
        logger.info("Loaded %d %s records from dataset", len(validated), context)
        return validated
