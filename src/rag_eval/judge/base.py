"""
Abstract base interface for judge LLMs.

Evaluators call BaseJudge — not GroqJudge directly.
This decouples metric logic from provider-specific code, keeping providers
swappable without touching the evaluation layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeResponse:
    """Structured response from a judge LLM call.

    Attributes:
        score: Float in [0.0, 1.0].
        rationale: The judge's natural-language explanation.
        raw_response: The unmodified string returned by the provider API.
        prompt_version: Version tag from prompts.py at call time.
    """

    score: float
    rationale: str
    raw_response: str
    prompt_version: str


class BaseJudge(ABC):
    """Abstract judge interface.

    All provider-specific implementations (GroqJudge, etc.) must inherit this
    and implement `judge`. Evaluators must only accept BaseJudge, never a
    concrete implementation directly.
    """

    @abstractmethod
    def judge(self, user_prompt: str) -> JudgeResponse:
        """Call the judge LLM with the given prompt.

        Args:
            user_prompt: The complete, formatted evaluation prompt (built by
                         the relevant function in judge/prompts.py).

        Returns:
            JudgeResponse with score, rationale, and raw provider output.

        Raises:
            JudgeCallError: If the API call fails after all retries.
            JudgeParseError: If the response cannot be parsed into a valid score.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier logged with each benchmark run."""
        ...


class JudgeCallError(RuntimeError):
    """Raised when the judge API fails after all retry attempts."""


class JudgeParseError(ValueError):
    """Raised when the judge response cannot be parsed into a valid score in [0, 1]."""
