"""
Test for T12b cluster B3: the ``BusyStatusMessage(...)`` construction
(preceded by ``self._build_plan_status_provider(chat_id, user_id)``)
duplicated across three call sites in ``bot/telegram_bot.py`` (media-group
vision, single-image vision, and the non-streaming chat reply). Extracted
into ``_build_busy_status(update, context, chat_id, user_id)``.
"""
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


class _FakeEncoding:
    def encode(self, value):
        return list(value)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: _FakeEncoding()
_tiktoken.get_encoding = lambda _name: _FakeEncoding()
_install_module_if_missing("tiktoken", _tiktoken)

_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)


def _retry(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_tenacity = types.ModuleType("tenacity")
_tenacity.retry = _retry
_tenacity.stop_after_attempt = lambda *args, **kwargs: None
_tenacity.wait_fixed = lambda *args, **kwargs: None
_tenacity.retry_if_exception_type = lambda *args, **kwargs: None
_install_module_if_missing("tenacity", _tenacity)

from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402
from bot.utils import BusyStatusMessage  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _make_bot(plan_tasks_provider=None):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"bot_language": "en"}
    agent_tools_plugin = (
        SimpleNamespace(get_plan_tasks=plan_tasks_provider) if plan_tasks_provider else None
    )
    bot.openai = SimpleNamespace(
        plugin_manager=SimpleNamespace(get_plugin=lambda name: agent_tools_plugin if name == "agent_tools" else None)
    )
    return bot


def test_build_busy_status_without_agent_tools_plugin():
    bot = _make_bot()
    update = SimpleNamespace()
    context = SimpleNamespace()

    status = bot._build_busy_status(update, context, chat_id=1, user_id=2)

    assert isinstance(status, BusyStatusMessage)
    assert status.update is update
    assert status.context is context
    assert status.config is bot.config
    assert status.plan_provider is None
    assert status.interval == 30.0


def test_build_busy_status_with_agent_tools_plugin_uses_plan_provider():
    bot = _make_bot(plan_tasks_provider=lambda chat_id, user_id: [{"id": "t1"}])
    update = SimpleNamespace()
    context = SimpleNamespace()

    status = bot._build_busy_status(update, context, chat_id=1, user_id=2)

    assert status.plan_provider is not None
    assert status.plan_provider() == [{"id": "t1"}]
    assert status.interval == 5.0


@pytest.mark.asyncio
async def test_all_three_call_sites_agree_via_shared_builder(monkeypatch):
    """Regression guard: all three original BusyStatusMessage construction
    sites now route through the shared builder (no residual inline copy)."""
    import inspect

    from bot import telegram_bot

    source = inspect.getsource(telegram_bot)
    assert source.count("BusyStatusMessage(") == 1
    assert source.count("self._build_busy_status(") == 3
