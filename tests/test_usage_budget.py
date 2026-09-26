import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

if importlib.util.find_spec("tiktoken") is None:
    _tiktoken = types.ModuleType("tiktoken")
    _tiktoken.__spec__ = importlib.machinery.ModuleSpec("tiktoken", loader=None)
    _tiktoken.encoding_for_model = lambda _model: SimpleNamespace(encode=lambda value: list(value))
    _tiktoken.get_encoding = lambda _name: SimpleNamespace(encode=lambda value: list(value))
    sys.modules["tiktoken"] = _tiktoken

if importlib.util.find_spec("tenacity") is None:
    def _retry(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

    _tenacity = types.ModuleType("tenacity")
    _tenacity.__spec__ = importlib.machinery.ModuleSpec("tenacity", loader=None)
    _tenacity.retry = _retry
    _tenacity.stop_after_attempt = lambda *args, **kwargs: None
    _tenacity.wait_fixed = lambda *args, **kwargs: None
    _tenacity.retry_if_exception_type = lambda *args, **kwargs: None
    sys.modules["tenacity"] = _tenacity

import pytest

from bot.usage_tracker import UsageTracker
import bot.utils as utils
from bot.utils import _budget_user_and_name, _charge_user_and_guest, _charge_user_and_guest_async, get_remaining_budget
from bot.telegram_bot import ChatGPTTelegramBot


def test_weekly_budget_uses_week_cost(tmp_path):
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

    assert get_remaining_budget(config, usage, update) == 7.5


def test_remaining_budget_initializes_user_and_guest_trackers_with_config_prices(tmp_path, monkeypatch):
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

    assert get_remaining_budget(config, usage, update) == 5
    assert usage[99].prices == usage["guests"].prices == {
        "token_price": 0.123,
        "image_prices": [1.0, 2.0, 3.0],
        "vision_token_price": 0.456,
        "tts_prices": [0.7, 0.8],
        "transcription_price": 0.321,
    }


def test_charge_user_and_guest_strips_whitespace_in_allowed_ids():
    """"1, 2" должно парситься так же, как в is_allowed: до фикса ' 2' (с пробелом)
    не совпадает с str(user_id), и явно разрешённый user_id=2 всё равно списывается
    с общего гостевого бюджета."""
    calls = []
    usage = {2: SimpleNamespace(name="user"), "guests": SimpleNamespace(name="guests")}
    config = {"allowed_user_ids": "1, 2"}

    charged = _charge_user_and_guest(usage, config, 2, lambda tracker: calls.append(tracker))

    assert charged is True
    assert calls == [usage[2]]  # guests-трекер не должен быть тронут


async def test_charge_user_and_guest_async_strips_whitespace_in_allowed_ids():
    calls = []

    async def charge_fn(tracker):
        calls.append(tracker)

    usage = {2: SimpleNamespace(name="user"), "guests": SimpleNamespace(name="guests")}
    config = {"allowed_user_ids": "1, 2"}

    charged = await _charge_user_and_guest_async(usage, config, 2, charge_fn)

    assert charged is True
    assert calls == [usage[2]]


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"allowed_user_ids": "*"}
    bot.usage = {}
    return bot


async def _false(*_args, **_kwargs):
    return False


@pytest.mark.parametrize(
    "make_update,is_inline,expected_id,expected_name",
    [
        (
            lambda: SimpleNamespace(
                inline_query=SimpleNamespace(from_user=SimpleNamespace(id=1, name="inline-user")),
                callback_query=None, message=None, effective_user=None,
            ),
            True, 1, "inline-user",
        ),
        (
            lambda: SimpleNamespace(
                inline_query=None,
                callback_query=SimpleNamespace(from_user=SimpleNamespace(id=2, name="callback-user")),
                message=None, effective_user=None,
            ),
            False, 2, "callback-user",
        ),
        (
            lambda: SimpleNamespace(
                inline_query=None, callback_query=None,
                message=SimpleNamespace(from_user=SimpleNamespace(id=3, name="message-user")),
                effective_user=None,
            ),
            False, 3, "message-user",
        ),
        (
            lambda: SimpleNamespace(
                inline_query=None, callback_query=None, message=None,
                effective_user=SimpleNamespace(id=4, name="fallback-user"),
            ),
            False, 4, "fallback-user",
        ),
    ],
)
async def test_check_allowed_and_within_budget_resolves_same_user_as_budget_user_and_name(
    monkeypatch, caplog, make_update, is_inline, expected_id, expected_name,
):
    """Regression for the B5 extraction: check_allowed_and_within_budget's
    "who sent this" resolution across all 4 update shapes (inline query,
    callback query, message, bare effective_user fallback) must match
    utils._budget_user_and_name for the same update -- both before and
    after the duplicate-core block is replaced by a call to it."""
    import bot.telegram_bot as telegram_bot_module

    bot = _make_bot()

    async def _fake_send_disallowed(update, context, is_inline):
        return None

    bot.send_disallowed_message = _fake_send_disallowed
    monkeypatch.setattr(telegram_bot_module, "is_allowed", _false)
    update = make_update()

    with caplog.at_level("WARNING"):
        result = await bot.check_allowed_and_within_budget(update, SimpleNamespace(), is_inline=is_inline)

    assert result is False
    assert _budget_user_and_name(update, is_inline) == (expected_id, expected_name)
    assert any(
        f"User {expected_name} (id: {expected_id}) is not allowed" in record.message
        for record in caplog.records
    )
