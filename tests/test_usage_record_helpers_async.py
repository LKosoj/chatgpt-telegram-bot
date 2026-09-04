import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.usage_tracker import UsageTracker
import bot.utils as utils
from bot.utils import (
    get_remaining_budget_async,
    is_within_budget_async,
    record_chat_tokens_async,
    record_image_request_async,
    record_tts_request_async,
    record_transcription_seconds_async,
    record_vision_tokens_async,
)


def _make_config(allowed="42"):
    return {
        "allowed_user_ids": allowed,
        "token_price": 0.002,
        "image_prices": [0.016, 0.018, 0.02],
        "vision_token_price": 0.01,
        "tts_prices": [0.015, 0.030],
        "transcription_price": 0.006,
    }


def _make_tracker(tmp_path, user_id, name):
    return UsageTracker(user_id, name, logs_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_record_chat_tokens_async_charges_guest_for_non_allowed(tmp_path):
    user = _make_tracker(tmp_path, 99, "stranger")
    guests = _make_tracker(tmp_path, "guests", "guests")
    usage = {99: user, "guests": guests}

    assert await record_chat_tokens_async(usage, _make_config(allowed="42"), 99, 1000) is True

    assert sum(user.usage["usage_history"]["chat_tokens"].values()) == 1000
    assert sum(guests.usage["usage_history"]["chat_tokens"].values()) == 1000


@pytest.mark.asyncio
async def test_record_image_request_async_uses_init_image_prices(tmp_path):
    tracker = UsageTracker(
        42, "user", logs_dir=str(tmp_path),
        image_prices=[1.0, 2.0, 3.0],
    )
    usage = {42: tracker}
    assert await record_image_request_async(usage, _make_config(allowed="42"), 42, "256x256") is True
    assert tracker.usage["current_cost"]["day"] == 1.0


@pytest.mark.asyncio
async def test_record_vision_tokens_async_charges_user_and_guest(tmp_path):
    user = UsageTracker(99, "stranger", logs_dir=str(tmp_path), vision_token_price=0.5)
    guests = UsageTracker("guests", "guests", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {99: user, "guests": guests}
    assert await record_vision_tokens_async(usage, _make_config(allowed="42"), 99, 2000) is True
    assert sum(user.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert sum(guests.usage["usage_history"]["vision_tokens"].values()) == 2000


@pytest.mark.asyncio
async def test_record_vision_tokens_async_rejects_zero_and_negative(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert await record_vision_tokens_async(usage, _make_config(allowed="42"), 42, 0) is False
    assert await record_vision_tokens_async(usage, _make_config(allowed="42"), 42, -1) is False
    assert user.usage["usage_history"]["vision_tokens"] == {}


@pytest.mark.asyncio
async def test_record_tts_request_async_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), tts_prices=[0.5, 0.75])
    usage = {42: user}
    assert await record_tts_request_async(usage, _make_config(allowed="42"), 42, 1000, "tts-1-hd") is True
    model_history = user.usage["usage_history"]["tts_characters"]["tts-1-hd"]
    assert sum(model_history.values()) == 1000


@pytest.mark.asyncio
async def test_record_transcription_seconds_async_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), transcription_price=0.5)
    usage = {42: user}
    assert await record_transcription_seconds_async(usage, _make_config(allowed="42"), 42, 60) is True
    assert sum(user.usage["usage_history"]["transcription_seconds"].values()) == 60


@pytest.mark.asyncio
async def test_get_remaining_budget_async_uses_week_cost(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path))
    tracker.add_current_costs(2.5)
    usage = {42: tracker}
    update = SimpleNamespace(
        message=SimpleNamespace(from_user=SimpleNamespace(id=42, name="user")),
        callback_query=None,
        inline_query=None,
        effective_user=SimpleNamespace(id=42, name="user"),
    )
    config = {
        "admin_user_ids": "-",
        "allowed_user_ids": "42",
        "user_budgets": "10",
        "budget_period": "weekly",
        "guest_budget": 0,
    }

    assert await get_remaining_budget_async(config, usage, update) == 7.5


@pytest.mark.asyncio
async def test_is_within_budget_async_true_when_remaining_positive(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path))
    usage = {42: tracker}
    update = SimpleNamespace(
        message=SimpleNamespace(from_user=SimpleNamespace(id=42, name="user")),
        callback_query=None,
        inline_query=None,
        effective_user=SimpleNamespace(id=42, name="user"),
    )
    config = {
        "admin_user_ids": "-",
        "allowed_user_ids": "42",
        "user_budgets": "10",
        "budget_period": "monthly",
        "guest_budget": 0,
    }

    assert await is_within_budget_async(config, usage, update) is True


@pytest.mark.asyncio
async def test_remaining_budget_async_initializes_user_and_guest_trackers(tmp_path, monkeypatch):
    # Async-зеркало test_remaining_budget_initializes_user_and_guest_trackers_with_config_prices
    # (tests/test_usage_budget.py): пользователь вне allowed_user_ids считается гостем,
    # оба трекера создаются с ценами из config.
    usage = {}
    original_make_usage_tracker = utils.make_usage_tracker
    monkeypatch.setattr(
        utils,
        "make_usage_tracker",
        lambda config, user_id, user_name, logs_dir="usage_logs": original_make_usage_tracker(
            config, user_id, user_name, logs_dir=str(tmp_path)
        ),
    )
    update = SimpleNamespace(
        message=SimpleNamespace(from_user=SimpleNamespace(id=99, name="guest")),
        callback_query=None,
        inline_query=None,
        effective_user=SimpleNamespace(id=99, name="guest"),
    )
    config = {
        "admin_user_ids": "-",
        "allowed_user_ids": "42",
        "user_budgets": "10",
        "budget_period": "monthly",
        "guest_budget": 5,
        "token_price": 0.123,
        "image_prices": [1.0, 2.0, 3.0],
        "vision_token_price": 0.456,
        "tts_prices": [0.7, 0.8],
        "transcription_price": 0.321,
    }

    assert await get_remaining_budget_async(config, usage, update) == 5
    assert usage[99].prices == usage["guests"].prices == {
        "token_price": 0.123,
        "image_prices": [1.0, 2.0, 3.0],
        "vision_token_price": 0.456,
        "tts_prices": [0.7, 0.8],
        "transcription_price": 0.321,
    }
