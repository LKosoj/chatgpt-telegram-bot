"""Tests for the Stage 0 plugin hooks framework in ``PluginManager``."""

import asyncio
import logging
from pathlib import Path

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.background import BackgroundTask
from bot.plugins.plugin import Plugin


class FakePlugin(Plugin):
    """Minimal plugin used to wire fake hook behaviour into the dispatcher."""

    def __init__(self, plugin_id: str):
        self.plugin_id = plugin_id
        self.function_prefix = plugin_id

    def get_source_name(self) -> str:
        return self.plugin_id

    def get_spec(self):
        return []

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "ok"}


@pytest.fixture()
def pm(tmp_path):
    """A ``PluginManager`` pointed at an empty plugin directory.

    We populate ``pm.plugins`` and ``pm.plugin_instances`` directly in tests
    instead of going through filesystem discovery — the dispatcher operates
    on those dicts.
    """
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    return PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))


def _register(pm: PluginManager, plugin: FakePlugin) -> None:
    pm.plugins[plugin.plugin_id] = type(plugin)
    pm.plugin_instances[plugin.plugin_id] = plugin


# ---------------------------------------------------------------- dispatch_observe


async def test_dispatch_observe_runs_concurrently(pm):
    """All observers run inside the same gather — verify with an Event barrier."""
    barrier = asyncio.Event()
    calls = []

    class A(FakePlugin):
        async def on_user_message(self, payload):
            calls.append(("a-start", payload))
            # Wait for B to signal; if execution were serial we'd deadlock.
            await asyncio.wait_for(barrier.wait(), timeout=1.0)
            calls.append(("a-end", payload))

    class B(FakePlugin):
        async def on_user_message(self, payload):
            calls.append(("b", payload))
            barrier.set()

    a = A("a")
    b = B("b")
    _register(pm, a)
    _register(pm, b)

    await pm.dispatch_observe("on_user_message", {"text": "hi"})

    assert ("a-start", {"text": "hi"}) in calls
    assert ("a-end", {"text": "hi"}) in calls
    assert ("b", {"text": "hi"}) in calls


async def test_dispatch_observe_isolates_exceptions(pm, caplog):
    other_called = []

    class Broken(FakePlugin):
        async def on_user_message(self, payload):
            raise RuntimeError("broken")

    class Ok(FakePlugin):
        async def on_user_message(self, payload):
            other_called.append(payload)

    _register(pm, Broken("broken"))
    _register(pm, Ok("ok"))

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        await pm.dispatch_observe("on_user_message", "payload")

    assert other_called == ["payload"]
    assert any("plugin_hook_error" in r.message for r in caplog.records)


async def test_dispatch_observe_skips_plugins_that_fail_to_construct(pm, caplog):
    called = []

    class Broken(FakePlugin):
        def __init__(self):
            raise RuntimeError("cannot construct")

    class Ok(FakePlugin):
        def __init__(self):
            super().__init__("bbb")

        async def on_user_message(self, payload):
            called.append(payload)

    pm.plugins["aaa"] = Broken
    pm.plugins["bbb"] = Ok

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        await pm.dispatch_observe("on_user_message", "payload")

    assert called == ["payload"]
    assert any("plugin_hook_error" in r.message for r in caplog.records)


# ---------------------------------------------------------------- dispatch_blocking


async def test_dispatch_blocking_sequential_order(pm):
    timeline = []

    class A(FakePlugin):
        async def on_session_before_delete(self, payload):
            timeline.append("a-start")
            await asyncio.sleep(0.02)
            timeline.append("a-end")

    class B(FakePlugin):
        async def on_session_before_delete(self, payload):
            timeline.append("b-start")
            timeline.append("b-end")

    _register(pm, A("aaa"))  # sorted order: aaa, bbb
    _register(pm, B("bbb"))

    await pm.dispatch_blocking("on_session_before_delete", "payload")

    assert timeline == ["a-start", "a-end", "b-start", "b-end"]


async def test_dispatch_blocking_first_raises_second_runs(pm, caplog):
    """Policy A: a failing plugin must NOT short-circuit the chain."""
    second = []

    class Broken(FakePlugin):
        async def on_session_before_delete(self, payload):
            raise RuntimeError("nope")

    class Ok(FakePlugin):
        async def on_session_before_delete(self, payload):
            second.append(payload)

    _register(pm, Broken("aaa"))  # comes first in sorted order
    _register(pm, Ok("bbb"))

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        try:
            # returns normally even though first plugin raised
            await pm.dispatch_blocking("on_session_before_delete", "p")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"dispatch_blocking should not propagate plugin exceptions: {exc!r}")

    assert second == ["p"]
    assert any("plugin_hook_error" in r.message for r in caplog.records)


# ---------------------------------------------------------------- collect_fragments


async def test_collect_fragments_sorted_by_plugin_id(pm):
    class Frag(FakePlugin):
        def __init__(self, plugin_id, text):
            super().__init__(plugin_id)
            self._text = text

        async def contribute_prompt_fragment(self, slot, payload):
            return f"{self._text}/{slot}"

    _register(pm, Frag("b", "B"))
    _register(pm, Frag("a", "A"))
    _register(pm, Frag("c", "C"))

    result = await pm.collect_fragments("slotX", {})

    assert result == ["A/slotX", "B/slotX", "C/slotX"]


async def test_collect_fragments_skips_none_and_raises(pm, caplog):
    class Good(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            return "good"

    class NoneRet(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            return None

    class Bad(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            raise RuntimeError("nope")

    class Empty(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            return ""

    _register(pm, Good("a"))
    _register(pm, NoneRet("b"))
    _register(pm, Bad("c"))
    _register(pm, Empty("d"))
    _register(pm, Good("e"))

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        result = await pm.collect_fragments("slot", {})

    # 'a' and 'e' both return 'good'; 'b' None, 'c' raised, 'd' empty -> skipped
    assert result == ["good", "good"]


# ---------------------------------------------------------------- apply_mutators


async def test_apply_mutators_chain_in_order(pm):
    class Prepender(FakePlugin):
        def __init__(self, plugin_id, label):
            super().__init__(plugin_id)
            self._label = label

        async def on_before_chat_request(self, messages, payload):
            return [self._label, *messages]

    _register(pm, Prepender("a", "A"))
    _register(pm, Prepender("b", "B"))
    _register(pm, Prepender("c", "C"))

    final = await pm.apply_mutators(
        "on_before_chat_request", payload={}, value=["base"]
    )

    # a runs first → [A, base]; b → [B, A, base]; c → [C, B, A, base]
    assert final == ["C", "B", "A", "base"]


async def test_apply_mutators_middle_raises_keeps_previous(pm, caplog):
    class Prepend(FakePlugin):
        def __init__(self, plugin_id, label):
            super().__init__(plugin_id)
            self._label = label

        async def on_before_chat_request(self, messages, payload):
            return [self._label, *messages]

    class Broken(FakePlugin):
        async def on_before_chat_request(self, messages, payload):
            raise RuntimeError("nope")

    _register(pm, Prepend("a", "A"))
    _register(pm, Broken("b"))
    _register(pm, Prepend("c", "C"))

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        final = await pm.apply_mutators(
            "on_before_chat_request", payload={}, value=["base"]
        )

    # a → [A, base]; b raises → still [A, base]; c → [C, A, base]
    assert final == ["C", "A", "base"]


async def test_apply_mutators_none_keeps_previous_value(pm):
    class Replace(FakePlugin):
        async def on_before_chat_request(self, messages, payload):
            return ["replaced"]

    class NoneRet(FakePlugin):
        async def on_before_chat_request(self, messages, payload):
            return None

    _register(pm, Replace("a"))
    _register(pm, NoneRet("b"))

    final = await pm.apply_mutators(
        "on_before_chat_request", payload={}, value=["base"]
    )

    # a replaces with ["replaced"]; b returns None → keep ["replaced"]
    assert final == ["replaced"]


# ---------------------------------------------------------------- background tasks


async def test_background_task_ticks_and_stops(pm):
    counter = {"n": 0}

    async def tick(*, application):
        counter["n"] += 1

    class TaskPlugin(FakePlugin):
        def get_background_tasks(self):
            return [BackgroundTask(name="tick", interval_seconds=0.01, coroutine_factory=tick)]

    _register(pm, TaskPlugin("tp"))

    await pm.start_background_tasks(application=object())
    # Give the loop a chance to run the factory at least once.
    await asyncio.sleep(0.05)
    await pm.stop_background_tasks(timeout=1.0)

    assert counter["n"] >= 1
    assert pm._background_tasks == {}


async def test_background_task_retries_after_exception(pm, caplog):
    counter = {"n": 0}
    second_call = asyncio.Event()

    async def flaky(*, application):
        counter["n"] += 1
        if counter["n"] >= 2:
            second_call.set()
        if counter["n"] == 1:
            raise RuntimeError("first call boom")

    class TaskPlugin(FakePlugin):
        def get_background_tasks(self):
            return [BackgroundTask(name="flaky", interval_seconds=0.01, coroutine_factory=flaky)]

    _register(pm, TaskPlugin("tp"))

    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        await pm.start_background_tasks(application=object())
        await asyncio.wait_for(second_call.wait(), timeout=2.0)
        await pm.stop_background_tasks(timeout=1.0)

    # At least two attempts: the failing one + at least one retry.
    assert counter["n"] >= 2
    assert any("background_task_error" in r.message for r in caplog.records)


# ---------------------------------------------------------------- per-user disabled


async def test_dispatch_observe_honors_per_user_disabled(pm, monkeypatch):
    called = []

    class A(FakePlugin):
        async def on_user_message(self, payload):
            called.append("a")

    class B(FakePlugin):
        async def on_user_message(self, payload):
            called.append("b")

    _register(pm, A("a"))
    _register(pm, B("b"))

    def fake_disabled(user_id):
        if user_id == 42:
            return {"a"}
        return set()

    monkeypatch.setattr(pm, "disabled_plugins_for_user", fake_disabled)

    await pm.dispatch_observe("on_user_message", "p", user_id=42)
    assert called == ["b"]

    called.clear()
    await pm.dispatch_observe("on_user_message", "p", user_id=43)
    assert sorted(called) == ["a", "b"]


async def test_dispatch_blocking_honors_per_user_disabled(pm, monkeypatch):
    called = []

    class A(FakePlugin):
        async def on_session_before_delete(self, payload):
            called.append("a")

    class B(FakePlugin):
        async def on_session_before_delete(self, payload):
            called.append("b")

    _register(pm, A("a"))
    _register(pm, B("b"))

    monkeypatch.setattr(
        pm, "disabled_plugins_for_user", lambda user_id: {"a"} if user_id == 42 else set()
    )

    await pm.dispatch_blocking("on_session_before_delete", "p", user_id=42)
    assert called == ["b"]


async def test_collect_fragments_honors_per_user_disabled(pm, monkeypatch):
    class A(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            return "A"

    class B(FakePlugin):
        async def contribute_prompt_fragment(self, slot, payload):
            return "B"

    _register(pm, A("a"))
    _register(pm, B("b"))

    monkeypatch.setattr(
        pm, "disabled_plugins_for_user", lambda user_id: {"a"} if user_id == 42 else set()
    )

    result = await pm.collect_fragments("slot", {}, user_id=42)
    assert result == ["B"]


async def test_apply_mutators_honors_per_user_disabled(pm, monkeypatch):
    class A(FakePlugin):
        async def on_before_chat_request(self, messages, payload):
            return ["A", *messages]

    class B(FakePlugin):
        async def on_before_chat_request(self, messages, payload):
            return ["B", *messages]

    _register(pm, A("a"))
    _register(pm, B("b"))

    monkeypatch.setattr(
        pm, "disabled_plugins_for_user", lambda user_id: {"a"} if user_id == 42 else set()
    )

    final = await pm.apply_mutators(
        "on_before_chat_request", payload={}, value=["base"], user_id=42
    )
    # Only B runs; A is disabled.
    assert final == ["B", "base"]


# ---------------------------------------------------------------- _call_initialize


def test_call_initialize_filters_legacy_signature(pm):
    """Plugins still using ``initialize(openai, bot, storage_root)`` must not see new kwargs."""

    captured = {}

    class Legacy(FakePlugin):
        def initialize(self, openai=None, bot=None, storage_root=None):
            captured["openai"] = openai
            captured["bot"] = bot
            captured["storage_root"] = storage_root

    plugin = Legacy("legacy")
    pm._call_initialize(
        plugin,
        openai="OAI",
        bot="BOT",
        storage_root="ROOT",
        db="DBHANDLE",
        plugin_config={"key": "val"},
    )

    assert captured == {"openai": "OAI", "bot": "BOT", "storage_root": "ROOT"}


def test_call_initialize_passes_db_and_config_when_accepted(pm):
    captured = {}

    class Modern(FakePlugin):
        def initialize(self, openai=None, bot=None, storage_root=None, db=None, plugin_config=None):
            captured.update(
                openai=openai, bot=bot, storage_root=storage_root, db=db, plugin_config=plugin_config
            )

    plugin = Modern("modern")
    pm._call_initialize(
        plugin,
        openai="OAI",
        bot="BOT",
        storage_root="ROOT",
        db="DBHANDLE",
        plugin_config={"k": "v"},
    )

    assert captured == {
        "openai": "OAI",
        "bot": "BOT",
        "storage_root": "ROOT",
        "db": "DBHANDLE",
        "plugin_config": {"k": "v"},
    }


def test_call_initialize_does_not_mask_internal_typeerror(pm):
    calls = []

    class Broken(FakePlugin):
        def initialize(self, openai=None):
            calls.append(openai)
            raise TypeError("internal init bug")

    with pytest.raises(TypeError, match="internal init bug"):
        pm._call_initialize(Broken("broken"), openai="OAI")

    assert calls == ["OAI"]


def test_set_openai_is_idempotent_for_same_helper_and_db(pm):
    calls = []

    class Modern(FakePlugin):
        def __init__(self):
            super().__init__("modern")

        def initialize(self, openai=None, bot=None, storage_root=None, db=None, plugin_config=None):
            self.openai = openai
            self.db_handle = db
            calls.append(db)

    plugin = Modern()
    _register(pm, plugin)
    pm.db_handle = object()
    helper = object()

    pm.set_openai(helper)
    pm.set_openai(helper)

    assert calls == [pm.db_handle]


def test_set_db_reinitializes_existing_plugins_with_db_handle(pm):
    calls = []

    class Modern(FakePlugin):
        def __init__(self):
            super().__init__("modern")

        def initialize(self, openai=None, bot=None, storage_root=None, db=None, plugin_config=None):
            self.openai = openai
            self.db_handle = db
            calls.append(db)

    plugin = Modern()
    _register(pm, plugin)
    helper = object()

    pm.set_openai(helper)
    pm.set_db(_FakeDb())

    assert calls == [None, pm.db_handle]


class _FakeConn:
    def __init__(self, executed):
        self._executed = executed

    def execute(self, sql, *args):
        self._executed.append(sql)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDb:
    def __init__(self):
        self.executed = []

    def get_connection(self):
        return _FakeConn(self.executed)


def test_register_plugin_schemas_does_not_invoke_initialize(pm):
    """Schema registry must not call ``initialize`` — ``set_openai`` does that exactly once."""
    init_calls = []

    class SchemaPlugin(FakePlugin):
        def __init__(self):
            super().__init__("schema_plugin")

        def initialize(self, openai=None, bot=None, storage_root=None, db=None, plugin_config=None):
            init_calls.append({"openai": openai, "db": db})

        def register_schema(self):
            return ["CREATE TABLE IF NOT EXISTS schema_plugin_t (id INTEGER)"]

    pm.plugins["schema_plugin"] = SchemaPlugin
    pm.plugin_instances.clear()
    pm.db = _FakeDb()

    pm.register_plugin_schemas()

    assert init_calls == []
    assert pm.db.executed == ["CREATE TABLE IF NOT EXISTS schema_plugin_t (id INTEGER)"]
    assert "schema_plugin" in pm.plugin_instances


def test_register_plugin_schemas_then_set_openai_initializes_once(pm):
    """End-to-end: schema first, then ``set_openai`` runs ``initialize`` exactly once per plugin."""
    init_calls = []

    class SchemaPlugin(FakePlugin):
        def __init__(self):
            super().__init__("p1")

        def initialize(self, openai=None, bot=None, storage_root=None, db=None, plugin_config=None):
            init_calls.append(openai)

        def register_schema(self):
            return ["CREATE TABLE IF NOT EXISTS p1_t (id INTEGER)"]

    pm.plugins["p1"] = SchemaPlugin
    pm.plugin_instances.clear()
    pm.db = _FakeDb()
    pm.storage_root = "/tmp"  # would normally cause get_plugin() to initialize eagerly

    pm.register_plugin_schemas()
    assert init_calls == []  # not yet

    pm.set_openai(object())  # fake openai
    assert len(init_calls) == 1
    assert init_calls[0] is not None


def test_register_plugin_schemas_raises_when_ddl_fails(pm):
    class BadSchemaPlugin(FakePlugin):
        def __init__(self):
            super().__init__("bad_schema")

        def register_schema(self):
            return ["CREATE TABLE bad (id INTEGER)"]

    class FailingConn:
        def execute(self, sql, *args):
            raise RuntimeError("ddl failed")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FailingDb:
        def get_connection(self):
            return FailingConn()

    pm.plugins["bad_schema"] = BadSchemaPlugin
    pm.plugin_instances.clear()
    pm.db = FailingDb()

    with pytest.raises(RuntimeError, match="bad_schema: RuntimeError: ddl failed"):
        pm.register_plugin_schemas()


def test_plugin_config_segment_preserves_prefix(pm):
    pm.config = {
        "hindsight_base_url": "https://h",
        "hindsight_token": "t",
        "hindsight_namespace": "ns",
        "model": "gpt",
        "image_size": "512x512",
        "tts_voice": "ru",
        "stream": True,
        "max_history_size": 15,
        "temperature": 1.0,
        "bot_language": "ru",
    }

    class P(FakePlugin):
        def get_config_prefix(self) -> str | None:
            return "hindsight_"

    plugin = P("h")
    slice_ = pm._plugin_config_segment(plugin)

    assert slice_ == {
        "hindsight_base_url": "https://h",
        "hindsight_token": "t",
        "hindsight_namespace": "ns",
    }
    # Keys preserve prefix; plugin slices its own.
    assert all(k.startswith("hindsight_") for k in slice_)
