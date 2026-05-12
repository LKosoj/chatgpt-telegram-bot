from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.plugin import Plugin


class _StubPlugin(Plugin):
    plugin_id = "stub"

    def __init__(self, name, return_value=None, raise_exc=None):
        super().__init__()
        self._name = name
        self._return = return_value
        self._raise = raise_exc

    def get_source_name(self):
        return self._name

    def get_spec(self):
        return []

    async def execute(self, function_name, helper, **kwargs):
        return {}

    async def contribute_prompt_fragment(self, slot, payload):
        if self._raise:
            raise self._raise
        return self._return


def _make_pm():
    pm = PluginManager.__new__(PluginManager)
    pm.plugin_instances = {}
    pm.plugins = {}
    pm.openai = None
    pm.bot = None
    pm.db = None
    pm.db_handle = None
    pm.storage_root = None
    pm.config = {}
    return pm


def _attach(pm, name, instance):
    pm.plugin_instances[name] = instance
    pm.plugins[name] = instance


async def test_collect_objects_returns_non_none_in_order(monkeypatch):
    pm = _make_pm()
    p_a = _StubPlugin("a", return_value=[["btn_a"]])
    p_b = _StubPlugin("b", return_value=None)  # opts out
    p_c = _StubPlugin("c", return_value=[["btn_c"]])
    _attach(pm, "a_plugin", p_a)
    _attach(pm, "b_plugin", p_b)
    _attach(pm, "c_plugin", p_c)
    monkeypatch.setattr(pm, "disabled_plugins_for_user", lambda uid: set())

    result = await pm.collect_objects("settings_menu_buttons", SimpleNamespace(user_id=None))

    assert result == [[["btn_a"]], [["btn_c"]]]


async def test_collect_objects_keeps_falsy_non_none(monkeypatch):
    pm = _make_pm()
    p = _StubPlugin("x", return_value=[])  # empty list — kept (not None)
    _attach(pm, "x_plugin", p)
    monkeypatch.setattr(pm, "disabled_plugins_for_user", lambda uid: set())

    result = await pm.collect_objects("some_slot", SimpleNamespace())
    assert result == [[]]


async def test_collect_objects_swallows_exceptions(monkeypatch):
    pm = _make_pm()
    bad = _StubPlugin("bad", raise_exc=RuntimeError("kaboom"))
    good = _StubPlugin("good", return_value="kept")
    _attach(pm, "a_bad", bad)
    _attach(pm, "z_good", good)
    monkeypatch.setattr(pm, "disabled_plugins_for_user", lambda uid: set())

    result = await pm.collect_objects("s", SimpleNamespace())
    assert result == ["kept"]


async def test_collect_objects_respects_per_user_disable(monkeypatch):
    pm = _make_pm()
    p = _StubPlugin("d", return_value="should-not-appear")
    _attach(pm, "d_plugin", p)
    monkeypatch.setattr(pm, "disabled_plugins_for_user", lambda uid: {"d_plugin"})

    result = await pm.collect_objects("s", SimpleNamespace(), user_id=42)
    assert result == []
