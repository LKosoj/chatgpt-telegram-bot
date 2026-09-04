"""Tests for MovieInfoPlugin's timeout/to_thread fix (T07).

All four ``requests.get`` calls (in ``_get_new_movies``, ``_get_movie_details``,
``_get_movie_reviews``, ``_discover_movies``) used to have no ``timeout`` and were
invoked synchronously from ``async execute``, blocking the event loop.
"""

import pytest

import bot.plugins.movie_info as movie_info
from bot.plugins.movie_info import MovieInfoPlugin


class FakeHelper:
    """Records ``ask()`` calls and replies with a canned response."""

    def __init__(self, reply="Рекомендую первый фильм."):
        self.reply = reply
        self.calls = []

    async def ask(self, prompt, chat_id):
        self.calls.append({"prompt": prompt, "chat_id": chat_id})
        return self.reply, 42


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeRequests:
    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return FakeResponse(self._payload)


to_thread_calls = []


async def fake_to_thread(func, *args, **kwargs):
    to_thread_calls.append(getattr(func, "__name__", repr(func)))
    return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_to_thread_calls():
    to_thread_calls.clear()
    yield
    to_thread_calls.clear()


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    return MovieInfoPlugin()


@pytest.mark.asyncio
async def test_get_new_movies_passes_timeout_and_uses_to_thread(plugin, monkeypatch):
    fake_requests = FakeRequests({"results": []})
    monkeypatch.setattr(movie_info, "requests", fake_requests)
    monkeypatch.setattr(movie_info.asyncio, "to_thread", fake_to_thread)

    result = await plugin.execute("get_new_movies", helper=None, genre=None, count=5)

    assert result["movies"] == []
    assert len(fake_requests.calls) == 1
    assert fake_requests.calls[0]["kwargs"]["timeout"] == 10
    assert to_thread_calls == ["_get_new_movies"]


@pytest.mark.asyncio
async def test_get_movie_recommendations_uses_to_thread_for_all_sync_calls(plugin, monkeypatch):
    movie_payload = {
        "results": [
            {"id": 1, "title": "Test Movie", "genre_ids": []},
        ]
    }
    fake_requests = FakeRequests(movie_payload)
    monkeypatch.setattr(movie_info, "requests", fake_requests)
    monkeypatch.setattr(movie_info.asyncio, "to_thread", fake_to_thread)

    helper = FakeHelper()
    result = await plugin.execute(
        "get_movie_recommendations", helper=helper, genre=None, count=1, chat_id=123
    )

    assert "recommendations" in result
    assert set(to_thread_calls) == {
        "_get_new_movies",
        "_discover_movies",
        "_get_movie_details",
        "_get_movie_reviews",
    }
    assert all(call["kwargs"]["timeout"] == 10 for call in fake_requests.calls)
