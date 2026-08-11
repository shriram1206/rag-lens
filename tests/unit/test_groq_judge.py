"""
Unit tests for the GroqJudge response parser.

Only parses — never calls the live Groq API. Judge API calls are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rag_eval.judge.base import JudgeParseError
from rag_eval.judge.groq_judge import GroqJudge


class TestGroqJudgeParser:
    """Tests for _parse_response — no live API calls."""

    @pytest.fixture()
    def judge(self, monkeypatch: pytest.MonkeyPatch) -> GroqJudge:
        monkeypatch.setenv("GROQ_API_KEY", "fake_key_for_testing")
        with patch("rag_eval.judge.groq_judge.Groq"):
            return GroqJudge()

    def test_valid_json_parsed_correctly(self, judge: GroqJudge) -> None:
        raw = '{"score": 0.85, "rationale": "All claims are grounded."}'
        response = judge._parse_response(raw)
        assert response.score == pytest.approx(0.85)
        assert response.rationale == "All claims are grounded."

    def test_score_rounded_to_4_decimal_places(self, judge: GroqJudge) -> None:
        raw = '{"score": 0.666667, "rationale": "Partial grounding."}'
        response = judge._parse_response(raw)
        assert response.score == pytest.approx(0.6667, abs=0.0001)

    def test_markdown_code_fence_stripped(self, judge: GroqJudge) -> None:
        raw = '```json\n{"score": 0.9, "rationale": "Strong grounding."}\n```'
        response = judge._parse_response(raw)
        assert response.score == pytest.approx(0.9)

    def test_non_json_raises_parse_error(self, judge: GroqJudge) -> None:
        with pytest.raises(JudgeParseError, match="non-JSON"):
            judge._parse_response("The answer seems good, I'd give it a 0.8.")

    def test_missing_score_key_raises_parse_error(self, judge: GroqJudge) -> None:
        with pytest.raises(JudgeParseError, match="missing required keys"):
            judge._parse_response('{"rationale": "Missing score key."}')

    def test_score_above_1_raises_parse_error(self, judge: GroqJudge) -> None:
        with pytest.raises(JudgeParseError, match="outside \\[0, 1\\]"):
            judge._parse_response('{"score": 1.5, "rationale": "Out of range."}')

    def test_score_below_0_raises_parse_error(self, judge: GroqJudge) -> None:
        with pytest.raises(JudgeParseError, match="outside \\[0, 1\\]"):
            judge._parse_response('{"score": -0.1, "rationale": "Negative score."}')

    def test_score_exactly_0_is_valid(self, judge: GroqJudge) -> None:
        raw = '{"score": 0.0, "rationale": "Completely hallucinated."}'
        response = judge._parse_response(raw)
        assert response.score == pytest.approx(0.0)

    def test_score_exactly_1_is_valid(self, judge: GroqJudge) -> None:
        raw = '{"score": 1.0, "rationale": "Perfectly grounded."}'
        response = judge._parse_response(raw)
        assert response.score == pytest.approx(1.0)

    def test_prompt_version_attached(self, judge: GroqJudge) -> None:
        from rag_eval.judge.prompts import PROMPT_VERSION
        raw = '{"score": 0.7, "rationale": "Mostly grounded."}'
        response = judge._parse_response(raw)
        assert response.prompt_version == PROMPT_VERSION
