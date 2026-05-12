from __future__ import annotations

from unittest.mock import AsyncMock

from bot.plugin_manager import PluginManager
from bot.plugins.plugin import Plugin


class _FakePlugin(Plugin):
    plugin_id = "fake"

    def get_source_name(self) -> str:
        return "fake"

    def get_spec(self):
        return []

    async def execute(self, function_name, helper, **kwargs):
        return {}


def _make_pm() -> PluginManager:
    """Build a PluginManager without running load_plugins() side effects."""
    pm = PluginManager.__new__(PluginManager)
    pm.plugin_instances = {}
    return pm


async def test_default_close_async_is_no_op():
    plugin = _FakePlugin()
    result = await plugin.close_async()
    assert result is None


async def test_close_all_async_calls_each_plugin():
    pm = _make_pm()
    plugin_a = _FakePlugin()
    plugin_b = _FakePlugin()
    plugin_a.close_async = AsyncMock()
    plugin_b.close_async = AsyncMock()
    pm.plugin_instances = {"a": plugin_a, "b": plugin_b}

    await pm.close_all_async()

    plugin_a.close_async.assert_awaited_once()
    plugin_b.close_async.assert_awaited_once()


async def test_close_all_async_continues_after_exception():
    pm = _make_pm()
    bad = _FakePlugin()
    good = _FakePlugin()
    bad.close_async = AsyncMock(side_effect=RuntimeError("boom"))
    good.close_async = AsyncMock()
    pm.plugin_instances = {"bad": bad, "good": good}

    # Should NOT raise
    await pm.close_all_async()

    bad.close_async.assert_awaited_once()
    good.close_async.assert_awaited_once()


async def test_close_all_async_with_no_plugins():
    pm = _make_pm()
    pm.plugin_instances = {}

    # Should NOT raise
    await pm.close_all_async()
