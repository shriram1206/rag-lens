"""
Unit tests for the three chunking strategies.

Tests confirm:
  - All chunkers produce non-empty chunks
  - No whitespace-only chunks
  - Edge cases: single sentence, empty-ish text, short documents
"""

from __future__ import annotations

import pytest

from rag_eval.retrieval.chunking import (
    SentenceChunker,
    ParagraphChunker,
    SemanticChunker,
    get_chunker,
)

_SAMPLE_TEXT = (
    "The Transformer architecture was introduced in 2017. "
    "It uses self-attention mechanisms to process sequential data. "
    "Unlike RNNs, Transformers process all tokens in parallel. "
    "This makes them significantly faster to train on modern hardware.\n\n"
    "BERT is a bidirectional Transformer model pre-trained by Google. "
    "It learns deep contextual representations of language. "
    "GPT models, developed by OpenAI, use unidirectional Transformers."
)

_SHORT_TEXT = "Only one sentence here."
_TWO_SENTENCE_TEXT = "First sentence. Second sentence."


class TestSentenceChunker:
    def test_produces_non_empty_chunks(self) -> None:
        chunks = SentenceChunker().chunk(_SAMPLE_TEXT)
        assert all(c.strip() for c in chunks)
        assert len(chunks) > 0

    def test_no_whitespace_only_chunks(self) -> None:
        chunks = SentenceChunker().chunk(_SAMPLE_TEXT)
        for c in chunks:
            assert c.strip() != ""

    def test_short_text_returns_at_least_one_chunk(self) -> None:
        chunks = SentenceChunker().chunk(_SHORT_TEXT)
        assert len(chunks) >= 1

    def test_max_sentences_groups_correctly(self) -> None:
        chunker = SentenceChunker(max_sentences=2)
        chunks = chunker.chunk(_TWO_SENTENCE_TEXT)
        # 2 sentences with max_sentences=2 → 1 combined chunk
        assert len(chunks) == 1
        assert "First sentence" in chunks[0]
        assert "Second sentence" in chunks[0]

    def test_strategy_name(self) -> None:
        assert SentenceChunker().strategy_name == "sentence"


class TestParagraphChunker:
    def test_splits_on_double_newline(self) -> None:
        chunks = ParagraphChunker().chunk(_SAMPLE_TEXT)
        assert len(chunks) == 2  # _SAMPLE_TEXT has 2 paragraphs

    def test_no_empty_chunks(self) -> None:
        chunks = ParagraphChunker().chunk(_SAMPLE_TEXT)
        assert all(c.strip() for c in chunks)

    def test_single_paragraph_text(self) -> None:
        chunks = ParagraphChunker().chunk("This is one paragraph with no breaks.")
        assert len(chunks) == 1

    def test_strategy_name(self) -> None:
        assert ParagraphChunker().strategy_name == "paragraph"


class TestSemanticChunker:
    """SemanticChunker requires sentence-transformers. Tests are skipped if
    the library is not installed in the test environment."""

    @pytest.fixture(autouse=True)
    def skip_if_no_transformers(self):
        pytest.importorskip("sentence_transformers")

    def test_produces_non_empty_chunks(self) -> None:
        chunks = SemanticChunker().chunk(_SAMPLE_TEXT)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_short_text_handled_gracefully(self) -> None:
        chunks = SemanticChunker().chunk(_SHORT_TEXT)
        assert len(chunks) >= 1

    def test_strategy_name(self) -> None:
        assert SemanticChunker().strategy_name == "semantic"


class TestGetChunker:
    def test_returns_sentence_chunker(self) -> None:
        assert isinstance(get_chunker("sentence"), SentenceChunker)

    def test_returns_paragraph_chunker(self) -> None:
        assert isinstance(get_chunker("paragraph"), ParagraphChunker)

    def test_returns_semantic_chunker(self) -> None:
        pytest.importorskip("sentence_transformers")
        assert isinstance(get_chunker("semantic"), SemanticChunker)

    def test_unknown_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("invalid_strategy")
