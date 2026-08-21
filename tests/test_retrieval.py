from unittest.mock import patch, MagicMock

from retrieval import Hit, retrieve


def _make_mock_collection(query_result):
    """Create a mock ChromaDB collection that returns the given query result."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = query_result
    return mock_collection


def _patch_retrieval(mock_collection):
    """Context manager that patches chromadb so retrieve() uses our mock collection."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    return patch("chromadb.PersistentClient", return_value=mock_client)


class TestRetrieve:
    def test_empty_query_returns_empty(self):
        result = retrieve("")
        assert result == []

    def test_whitespace_query_returns_empty(self):
        result = retrieve("   ")
        assert result == []

    def test_returns_hit_objects(self):
        mock_collection = _make_mock_collection({
            "documents": [["Some text"]],
            "metadatas": [[{"source": "doc.txt", "chunk_index": 0}]],
            "distances": [[0.5]],
        })

        with _patch_retrieval(mock_collection):
            results = retrieve("test query")

        assert len(results) == 1
        assert isinstance(results[0], Hit)

    def test_hit_fields_populated(self):
        mock_collection = _make_mock_collection({
            "documents": [["Hello world"]],
            "metadatas": [[{"source": "readme.md", "chunk_index": 3}]],
            "distances": [[0.42]],
        })

        with _patch_retrieval(mock_collection):
            results = retrieve("query")

        hit = results[0]
        assert hit.text == "Hello world"
        assert hit.source == "readme.md"
        assert hit.chunk_index == 3
        assert hit.distance == 0.42

    def test_empty_collection_returns_empty(self):
        mock_collection = _make_mock_collection({
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        })

        with _patch_retrieval(mock_collection):
            results = retrieve("query")

        assert results == []

    def test_multiple_results(self):
        mock_collection = _make_mock_collection({
            "documents": [["Text A", "Text B", "Text C"]],
            "metadatas": [
                [
                    {"source": "a.txt", "chunk_index": 0},
                    {"source": "b.txt", "chunk_index": 1},
                    {"source": "c.txt", "chunk_index": 2},
                ]
            ],
            "distances": [[0.1, 0.3, 0.5]],
        })

        with _patch_retrieval(mock_collection):
            results = retrieve("query")

        assert len(results) == 3
        assert results[0].source == "a.txt"
        assert results[2].source == "c.txt"
