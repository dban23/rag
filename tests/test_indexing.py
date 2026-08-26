from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import redis

from indexing import OllamaEmbedder


def _fake_redis():
    """Create a mock Redis client that behaves like a real one."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None  # default: cache miss
    return mock


def _make_embedder(redis_url="redis://localhost:6379", mock_redis=None):
    """Helper to create an OllamaEmbedder with a mocked Redis connection."""
    if mock_redis is None:
        mock_redis = _fake_redis()
    with patch("indexing.redis.Redis") as MockRedis:
        MockRedis.from_url.return_value = mock_redis
        embedder = OllamaEmbedder(redis_url=redis_url)
    return embedder


def _fake_embedding(dim=768):
    """Return a fake embedding vector."""
    return [0.1] * dim


class TestOllamaEmbedderInit:
    def test_creates_redis_connection(self):
        mock_redis = _fake_redis()
        embedder = _make_embedder(mock_redis=mock_redis)

        mock_redis.ping.assert_called_once()
        assert embedder._redis is mock_redis

    def test_no_redis_when_url_none(self):
        with patch.dict("os.environ", {}, clear=True):
            embedder = OllamaEmbedder(redis_url=None)

        assert embedder._redis is None

    def test_no_redis_when_env_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            embedder = OllamaEmbedder()

        assert embedder._redis is None

    def test_graceful_on_connection_error(self):
        with patch("indexing.redis.Redis") as MockRedis:
            MockRedis.from_url.side_effect = redis.ConnectionError("refused")
            embedder = OllamaEmbedder(redis_url="redis://localhost:6379")

        assert embedder._redis is None

    def test_graceful_on_ping_failure(self):
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis.ConnectionError("timeout")

        with patch("indexing.redis.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis
            embedder = OllamaEmbedder(redis_url="redis://localhost:6379")

        assert embedder._redis is None

    def test_graceful_on_timeout_error(self):
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis.TimeoutError("slow")

        with patch("indexing.redis.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis
            embedder = OllamaEmbedder(redis_url="redis://localhost:6379")

        assert embedder._redis is None


class TestCacheHit:
    def test_returns_cached_embedding_without_calling_ollama(self):
        cached_vec = _fake_embedding()
        mock_redis = _fake_redis()
        mock_redis.get.return_value = np.array(cached_vec, dtype=np.float32).tobytes()
        embedder = _make_embedder(mock_redis=mock_redis)

        with patch("indexing.ollama") as mock_ollama:
            result = embedder(["hello world"])

        mock_ollama.embed.assert_not_called()
        assert len(result) == 1
        assert result[0] == pytest.approx(cached_vec)

    def test_multiple_texts_all_cached(self):
        vec1 = _fake_embedding()
        vec2 = [0.2] * 768
        mock_redis = _fake_redis()
        mock_redis.get.side_effect = [
            np.array(vec1, dtype=np.float32).tobytes(),
            np.array(vec2, dtype=np.float32).tobytes(),
        ]
        embedder = _make_embedder(mock_redis=mock_redis)

        with patch("indexing.ollama") as mock_ollama:
            result = embedder(["text one", "text two"])

        mock_ollama.embed.assert_not_called()
        assert len(result) == 2
        assert result[0] == pytest.approx(vec1)
        assert result[1] == pytest.approx(vec2)


class TestCacheMiss:
    def test_calls_ollama_and_stores_in_redis(self):
        new_vec = _fake_embedding()
        mock_redis = _fake_redis()
        embedder = _make_embedder(mock_redis=mock_redis)

        mock_response = MagicMock()
        mock_response.embeddings = [new_vec]

        with patch("indexing.ollama") as mock_ollama:
            mock_ollama.embed.return_value = mock_response
            result = embedder(["new text"])

        mock_ollama.embed.assert_called_once()
        assert result[0] == pytest.approx(new_vec)
        mock_redis.set.assert_called_once()
        # Verify the stored value is numpy bytes
        stored_value = mock_redis.set.call_args[0][1]
        assert isinstance(stored_value, bytes)

    def test_partial_cache_hit(self):
        """Only misses get embedded, hits come from cache."""
        cached_vec = _fake_embedding()
        new_vec = [0.2] * 768
        mock_redis = _fake_redis()
        # Use side_effect: first call returns cached bytes, second returns None (miss)
        mock_redis.get.side_effect = [
            np.array(cached_vec, dtype=np.float32).tobytes(),
            None,
        ]
        embedder = _make_embedder(mock_redis=mock_redis)

        mock_response = MagicMock()
        mock_response.embeddings = [new_vec]

        with patch("indexing.ollama") as mock_ollama:
            mock_ollama.embed.return_value = mock_response
            result = embedder(["cached text", "new text"])

        # Only the miss should be sent to ollama
        mock_ollama.embed.assert_called_once()
        call_args = mock_ollama.embed.call_args
        assert call_args.kwargs["input"] == ["search_document: new text"]
        assert len(result) == 2
        assert result[0] == pytest.approx(cached_vec)
        assert result[1] == pytest.approx(new_vec)


class TestGracefulDegradation:
    def test_works_without_redis(self):
        new_vec = _fake_embedding()
        embedder = OllamaEmbedder()  # no redis_url, no REDIS_URL env

        mock_response = MagicMock()
        mock_response.embeddings = [new_vec]

        with patch("indexing.ollama") as mock_ollama:
            mock_ollama.embed.return_value = mock_response
            result = embedder(["hello"])

        assert result[0] == pytest.approx(new_vec)

    def test_cache_get_returns_none_on_error(self):
        mock_redis = _fake_redis()
        mock_redis.get.side_effect = redis.TimeoutError("slow")
        embedder = _make_embedder(mock_redis=mock_redis)

        key = embedder._cache_key("search_document: test")
        assert embedder._cache_get(key) is None

    def test_cache_set_ignores_error(self):
        mock_redis = _fake_redis()
        mock_redis.set.side_effect = redis.TimeoutError("slow")
        embedder = _make_embedder(mock_redis=mock_redis)

        key = embedder._cache_key("search_document: test")
        # Should not raise
        embedder._cache_set(key, [0.1] * 768)


class TestCacheKey:
    def test_key_includes_prefix(self):
        embedder_doc = OllamaEmbedder(prefix="search_document: ")
        embedder_query = OllamaEmbedder(prefix="search_query: ")

        key_doc = embedder_doc._cache_key("search_document: hello")
        key_query = embedder_query._cache_key("search_query: hello")

        # Same text but different prefixes → different cache keys
        assert key_doc != key_query

    def test_key_starts_with_embed_prefix(self):
        embedder = OllamaEmbedder()
        key = embedder._cache_key("search_document: test")
        assert key.startswith(b"embed:")

    def test_same_text_same_key(self):
        embedder = OllamaEmbedder()
        key1 = embedder._cache_key("search_document: hello")
        key2 = embedder._cache_key("search_document: hello")
        assert key1 == key2
