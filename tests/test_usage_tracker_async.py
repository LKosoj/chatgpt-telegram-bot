import asyncio
import importlib.machinery
import importlib.util
import sqlite3
import sys
import types

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

import bot.usage_tracker as usage_tracker
from bot.usage_tracker import UsageTracker


def _make_tracker(tmp_path, user_id=42, name="user"):
    return UsageTracker(user_id, name, logs_dir=str(tmp_path))


def _usage_db_rows(tmp_path, query, params=()):
    conn = sqlite3.connect(tmp_path / "usage.sqlite3")
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_add_chat_tokens_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    await tracker.add_chat_tokens_async(100, cost=0.5, metadata={"model": "gpt-test"})

    assert calls == ["add_chat_tokens"]

    reference = _make_tracker(tmp_path, user_id=43, name="reference")
    reference.add_chat_tokens(100, cost=0.5, metadata={"model": "gpt-test"})

    assert sum(tracker.usage["usage_history"]["chat_tokens"].values()) == \
        sum(reference.usage["usage_history"]["chat_tokens"].values()) == 100


@pytest.mark.asyncio
async def test_add_image_request_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    await tracker.add_image_request_async("1024x1024")

    assert calls == ["add_image_request"]
    today_day, today_month = tracker.get_current_image_count()
    assert today_day == today_month == 1


@pytest.mark.asyncio
async def test_add_vision_tokens_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    await tracker.add_vision_tokens_async(200)

    assert calls == ["add_vision_tokens"]
    assert sum(tracker.usage["usage_history"]["vision_tokens"].values()) == 200


@pytest.mark.asyncio
async def test_add_tts_request_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    await tracker.add_tts_request_async(500, "tts-1")

    assert calls == ["add_tts_request"]
    today_day, today_month = tracker.get_current_tts_usage()
    assert today_day == today_month == 500


@pytest.mark.asyncio
async def test_add_transcription_seconds_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    await tracker.add_transcription_seconds_async(61)

    assert calls == ["add_transcription_seconds"]
    assert sum(tracker.usage["usage_history"]["transcription_seconds"].values()) == 61


@pytest.mark.asyncio
async def test_get_current_cost_async_routes_through_to_thread(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", fake_to_thread)

    tracker = _make_tracker(tmp_path)
    tracker.add_chat_tokens(1000, cost=2.0)

    result = await tracker.get_current_cost_async()

    assert calls == ["get_current_cost"]
    assert result == tracker.get_current_cost()


@pytest.mark.asyncio
async def test_concurrent_add_chat_tokens_async_does_not_lose_updates(tmp_path):
    tracker = _make_tracker(tmp_path)

    await asyncio.gather(*[tracker.add_chat_tokens_async(100) for _ in range(20)])

    assert sum(tracker.usage["usage_history"]["chat_tokens"].values()) == 2000

    rows = _usage_db_rows(
        tmp_path,
        "SELECT SUM(amount) AS total FROM usage_events WHERE event_type='chat_tokens'",
    )
    assert rows[0]["total"] == 2000
