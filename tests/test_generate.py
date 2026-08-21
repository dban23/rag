from unittest.mock import patch, MagicMock

from generate import resolve_citations, generate
from retrieval import Hit


class TestResolveCitations:
    def test_replaces_valid_citation(self):
        hits = [Hit(text="t", source="readme.md", chunk_index=0, distance=0.1)]
        result = resolve_citations("As stated in [1]...", hits)
        assert result == "As stated in readme.md..."

    def test_ignores_out_of_range_citation(self):
        hits = [Hit(text="t", source="readme.md", chunk_index=0, distance=0.1)]
        result = resolve_citations("See [99] for details", hits)
        assert result == "See [99] for details"

    def test_no_citations_passes_through(self):
        hits = [Hit(text="t", source="readme.md", chunk_index=0, distance=0.1)]
        result = resolve_citations("No citations here.", hits)
        assert result == "No citations here."

    def test_multiple_citations(self):
        hits = [
            Hit(text="a", source="doc1.txt", chunk_index=0, distance=0.1),
            Hit(text="b", source="doc2.txt", chunk_index=1, distance=0.2),
        ]
        result = resolve_citations("From [1] and [2] we see...", hits)
        assert result == "From doc1.txt and doc2.txt we see..."

    def test_citation_zero_unchanged(self):
        hits = [Hit(text="t", source="readme.md", chunk_index=0, distance=0.1)]
        result = resolve_citations("See [0] for more", hits)
        assert result == "See [0] for more"


class TestGenerate:
    def test_calls_ollama_with_correct_model(self):
        hits = [Hit(text="Some context", source="doc.txt", chunk_index=0, distance=0.1)]
        mock_response = {"message": {"content": "The answer is [1]."}}

        with patch("generate.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = generate("What is this?", hits)

        mock_ollama.chat.assert_called_once()
        call_args = mock_ollama.chat.call_args
        assert call_args.kwargs["model"] == "llama3.2:3b"

    def test_builds_context_from_hits(self):
        hits = [
            Hit(text="First passage", source="a.txt", chunk_index=0, distance=0.1),
            Hit(text="Second passage", source="b.txt", chunk_index=1, distance=0.2),
        ]
        mock_response = {"message": {"content": "Answer [1] [2]"}}

        with patch("generate.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            generate("Question?", hits)

        call_args = mock_ollama.chat.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "First passage" in user_msg
        assert "Second passage" in user_msg
        assert "[1] (a.txt, chunk 0)" in user_msg
        assert "[2] (b.txt, chunk 1)" in user_msg

    def test_applies_citations_in_output(self):
        hits = [Hit(text="ctx", source="manual.md", chunk_index=2, distance=0.05)]
        mock_response = {"message": {"content": "According to [1], it works."}}

        with patch("generate.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = generate("How does it work?", hits)

        assert result == "According to manual.md, it works."

    def test_uses_temperature_02(self):
        hits = [Hit(text="ctx", source="doc.txt", chunk_index=0, distance=0.1)]
        mock_response = {"message": {"content": "answer"}}

        with patch("generate.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            generate("Q?", hits)

        call_args = mock_ollama.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.2
