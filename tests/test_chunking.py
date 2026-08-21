from chunking import _split_sentences, chunk_text


class TestSplitSentences:
    def test_splits_on_period(self):
        result = _split_sentences("First sentence. Second sentence.")
        assert result == ["First sentence.", "Second sentence."]

    def test_splits_on_exclamation(self):
        result = _split_sentences("Hello! How are you?")
        assert result == ["Hello!", "How are you?"]

    def test_splits_on_newline(self):
        result = _split_sentences("Line one\nLine two\nLine three")
        assert result == ["Line one", "Line two", "Line three"]

    def test_strips_whitespace(self):
        result = _split_sentences("  First.   Second.  ")
        assert result == ["First.", "Second."]

    def test_skips_empty_parts(self):
        result = _split_sentences("First.\n\n\nSecond.")
        assert result == ["First.", "Second."]


class TestChunkText:
    def test_empty_string_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   ") == []

    def test_single_sentence_returns_one_chunk(self):
        result = chunk_text("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_sentences_under_limit_in_one_chunk(self):
        text = "Short sentence. Another short one. And one more."
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1

    def test_long_text_splits_at_sentence_boundary(self):
        sentences = [f"Sentence number {i} with some words." for i in range(20)]
        text = " ".join(sentences)
        result = chunk_text(text, chunk_size=100)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 150  # some tolerance for overlap

    def test_no_mid_word_cuts(self):
        sentences = [f"The quick brown fox jumps over the lazy dog. " for _ in range(20)]
        text = "".join(sentences)
        original_words = set(text.split())
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) > 1  # force multiple chunks
        for chunk in chunks:
            for word in chunk.split():
                # Split on periods — overlap may concatenate words at chunk boundaries
                for part in word.split("."):
                    clean = part.strip(".,!?;:")
                    if clean:
                        assert clean in {w.strip(".,!?;:") for w in original_words}, (
                            f"'{word}' is not in original text"
                        )

    def test_overlap_prepend(self):
        sentences = [f"This is sentence {i}." for i in range(10)]
        text = " ".join(sentences)
        result = chunk_text(text, chunk_size=50, overlap=20)
        assert len(result) > 1
        for i in range(1, len(result)):
            tail = result[i - 1][-20:]
            # .strip() may remove leading spaces from the tail
            assert result[i].startswith(tail.strip())

    def test_overlap_zero(self):
        sentences = [f"Sentence {i}." for i in range(5)]
        text = " ".join(sentences)
        result = chunk_text(text, chunk_size=20, overlap=0)
        assert len(result) > 1
        # With no overlap, no chunk should start with text from the previous chunk
        for i in range(1, len(result)):
            prev_tail = result[i - 1][-10:]
            assert not result[i].startswith(prev_tail)
