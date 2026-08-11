"""
Groq/Llama-3 implementation of the BaseJudge interface.

Handles:
  - Exponential backoff retries via tenacity (rate limits, transient 5xx)
  - Strict JSON parsing of the judge response
  - Score bounds enforcement (fails loudly, never silently clamps)
  - Logging of every call for cost/latency visibility
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from rag_lens.judge.base import BaseJudge, JudgeCallError, JudgeParseError, JudgeResponse
from rag_lens.judge.prompts import PROMPT_VERSION, SYSTEM_MESSAGE

load_dotenv()

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIStatusError)


class GroqJudge(BaseJudge):
    """Judge LLM backed by Groq's Llama-3 inference API.

    Args:
        model: The Groq model name to use. Defaults to the env var
               JUDGE_MODEL or "llama-3.3-70b-versatile".
        temperature: Sampling temperature. Keep low (≤0.1) for deterministic
                     scoring; higher values increase score variance.
        max_retries: Maximum number of API retry attempts before raising
                     JudgeCallError.

    Example:
        judge = GroqJudge()
        response = judge.judge(prompt)
        print(response.score, response.rationale)
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Copy .env.example → .env and add your key."
            )
        self._client = Groq(api_key=api_key)
        self._model = model or os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")
        self._temperature = temperature
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    def judge(self, user_prompt: str) -> JudgeResponse:
        """Call the Groq API with retry logic and strict response parsing.

        Args:
            user_prompt: Fully formatted evaluation prompt from judge/prompts.py.

        Returns:
            JudgeResponse with validated score in [0.0, 1.0].

        Raises:
            JudgeCallError: If all retries are exhausted.
            JudgeParseError: If the response JSON is malformed or score is out of range.
        """
        raw = self._call_with_retry(user_prompt)
        return self._parse_response(raw)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    def _call_with_retry(self, user_prompt: str) -> str:
        """Inner call to Groq with tenacity retry. Returns raw response string."""
        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=256,
            )
        except _RETRYABLE:
            raise  # Let tenacity handle retryable errors
        except Exception as exc:
            raise JudgeCallError(f"Non-retryable error calling Groq API: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        raw = completion.choices[0].message.content or ""
        tokens_used = completion.usage.total_tokens if completion.usage else -1
        logger.debug(
            "Judge call: model=%s tokens=%d latency_ms=%.1f",
            self._model,
            tokens_used,
            elapsed_ms,
        )
        return raw

    def _parse_response(self, raw: str) -> JudgeResponse:
        """Parse and validate the judge's JSON response.

        Fails loudly on:
          - Non-parseable JSON
          - Missing 'score' or 'rationale' keys
          - Score outside [0.0, 1.0]

        Does NOT silently clamp scores — a score outside bounds is a bug in
        the prompt or response that must be surfaced, not hidden.
        """
        # Strip markdown code fences if the model wrapped its JSON
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeParseError(
                f"Judge returned non-JSON response. raw={raw!r}"
            ) from exc

        if "score" not in data or "rationale" not in data:
            raise JudgeParseError(
                f"Judge response missing required keys. Got keys: {list(data.keys())}. raw={raw!r}"
            )

        score = float(data["score"])
        if not (0.0 <= score <= 1.0):
            raise JudgeParseError(
                f"Judge returned score outside [0, 1]: {score}. This indicates a prompt bug."
            )

        return JudgeResponse(
            score=round(score, 4),
            rationale=str(data["rationale"]),
            raw_response=raw,
            prompt_version=PROMPT_VERSION,
        )
