import asyncio
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.plugin_manager import PluginManager
from bot.request_context import RequestContext


def _write_plugin(path: Path, class_name: str, func_name: str):
    code = f"""
from bot.plugins.plugin import Plugin

class {class_name}(Plugin):
    def get_source_name(self) -> str:
        return "{class_name}"

    def get_spec(self):
        return [{{"name": "{func_name}", "description": "x", "parameters": {{"type": "object", "properties": {{}}, "required": []}}}}]

    async def execute(self, function_name, helper, **kwargs):
        return {{"result": "ok"}}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_context_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class ContextPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Context"

    def get_spec(self):
        return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    async def execute(self, function_name, helper, **kwargs):
        request_context = kwargs["request_context"]
        return {
            "chat_id": kwargs["chat_id"],
            "user_id": kwargs["user_id"],
            "message_id": kwargs["message_id"],
            "context_user_id": request_context.user_id,
            "context_message_id": request_context.message_id,
        }
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_ok_false_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class OkFalsePlugin(Plugin):
    def get_source_name(self) -> str:
        return "OkFalse"

    def get_spec(self):
        return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    async def execute(self, function_name, helper, **kwargs):
        return {"ok": False, "code": "REMOTE_BLOCKED", "message": "blocked"}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_guard_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class GuardPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Guard"

    def get_spec(self):
        return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    def guard_tool_call(self, *, function_name, arguments, request_context=None):
        if function_name == "guard.do":
            return {"success": False, "error": "blocked"}
        return None

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "should-not-run"}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_prompt_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class PromptPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Prompt"

    def get_spec(self):
        return []

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "ok"}

    def get_prompt_handlers(self):
        return [{"handler": self.handle_prompt, "chat_action": "typing"}]

    def get_help_text(self):
        return "Prompt plugin help"

    async def handle_prompt(self, **kwargs):
        return False
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_bad_spec_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class BadSpecPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Bad"

    def get_spec(self):
        raise RuntimeError("broken spec")

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "bad"}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_cancel_plugin(path: Path):
    code = """
import asyncio

from bot.plugins.plugin import Plugin

class CancelPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Cancel"

    def get_spec(self):
        return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    async def execute(self, function_name, helper, **kwargs):
        raise asyncio.CancelledError()
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_request_context_probe_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class RequestContextProbePlugin(Plugin):
    def get_source_name(self) -> str:
        return "Probe"

    def get_spec(self):
        return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    async def execute(self, function_name, helper, **kwargs):
        return {
            "has_request_context": "request_context" in kwargs,
            "chat_id": kwargs.get("chat_id"),
            "user_id": kwargs.get("user_id"),
        }
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_model_name_collision_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class RawNamePlugin(Plugin):
    def get_source_name(self) -> str:
        return "Raw"

    def get_function_prefix(self) -> str:
        return ""

    def get_spec(self):
        return [{"name": "alpha_do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

    async def execute(self, function_name, helper, **kwargs):
        return {"plugin": "raw", "function_name": function_name}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_imported_class_plugin(tmp_path: Path, plugin_dir: Path):
    (tmp_path / "foreign_plugin.py").write_text(textwrap.dedent("""
        from bot.plugins.plugin import Plugin

        class AImportedPlugin(Plugin):
            def get_source_name(self) -> str:
                return "Imported"

            def get_spec(self):
                return [{"name": "imported", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

            async def execute(self, function_name, helper, **kwargs):
                return {"result": "imported"}
    """), encoding="utf-8")
    (plugin_dir / "local.py").write_text(textwrap.dedent("""
        from foreign_plugin import AImportedPlugin
        from bot.plugins.plugin import Plugin

        class LocalPlugin(Plugin):
            def get_source_name(self) -> str:
                return "Local"

            def get_spec(self):
                return [{"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]

            async def execute(self, function_name, helper, **kwargs):
                return {"result": "local"}
    """), encoding="utf-8")


def test_config_none_uses_empty_config(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    pm = PluginManager(config=None, plugins_directory=str(plugin_dir))

    assert pm.config == {}
    assert pm.enabled_plugins == []


def test_config_none_loads_all_plugins_by_default(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")

    pm = PluginManager(config=None, plugins_directory=str(plugin_dir))

    assert "alpha" in pm.plugins
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])
    assert [spec["function"]["name"] for spec in specs] == ["alpha_do"]


def test_register_plugin_ignores_imported_plugin_classes(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_imported_class_plugin(tmp_path, plugin_dir)
    monkeypatch.syspath_prepend(str(tmp_path))

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    assert pm.plugins["local"].__name__ == "LocalPlugin"


def test_namespacing_and_collision(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "deepl.py", "DeeplPlugin", "translate")
    _write_plugin(plugin_dir / "ddg_translate.py", "DDGTranslatePlugin", "translate")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])
    names = [s["function"]["name"] for s in specs]
    assert "deepl_translate" in names
    assert "ddg_translate_translate" in names
    assert pm.to_canonical_function_name("deepl_translate") == "deepl.translate"
    assert pm.to_canonical_function_name("ddg_translate_translate") == "ddg_translate.translate"


@pytest.mark.asyncio
async def test_model_safe_function_name_collision_round_trips_to_correct_plugin(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    _write_model_name_collision_plugin(plugin_dir / "raw.py")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])
    model_names = [spec["function"]["name"] for spec in specs]

    raw_model_name = pm.to_model_function_name(".alpha_do")
    assert "alpha_do" in model_names
    assert raw_model_name in model_names
    assert raw_model_name != "alpha_do"
    assert pm.to_canonical_function_name("alpha_do") == "alpha.do"
    assert pm.to_canonical_function_name(raw_model_name) == ".alpha_do"

    result = await pm.call_function(raw_model_name, helper=None, arguments="{}")

    assert json.loads(result) == {"plugin": "raw", "function_name": "alpha_do"}


@pytest.mark.asyncio
async def test_call_function_lookup_skips_unrelated_broken_plugin(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_bad_spec_plugin(plugin_dir / "aaa_bad.py")
    _write_plugin(plugin_dir / "good.py", "GoodPlugin", "do")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    result = await pm.call_function("good.do", None, "{}")

    assert json.loads(result) == {"result": "ok"}


@pytest.mark.asyncio
async def test_call_function_accepts_model_safe_function_name(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "good.py", "GoodPlugin", "do")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])
    assert specs[0]["function"]["name"] == "good_do"

    result = await pm.call_function("good_do", None, "{}")

    assert json.loads(result) == {"result": "ok"}


def test_filter_allowed_plugins(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    filtered = pm.filter_allowed_plugins(["alpha", "missing"])
    assert filtered == ["alpha"]


def test_function_allowlist_uses_plugin_ownership(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    _write_plugin(plugin_dir / "beta.py", "BetaPlugin", "run")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    assert pm.get_plugin_name_by_function_name("alpha.do") == "alpha"
    assert pm.is_function_allowed("alpha.do", ["alpha"]) is True
    assert pm.get_plugin_name_by_function_name("alpha.missing") is None
    assert pm.is_function_allowed("alpha.missing", ["alpha"]) is False
    assert pm.is_function_allowed("beta.run", ["alpha"]) is False
    assert pm.is_function_allowed("beta.run", ["All"]) is True


def test_prompt_handlers_include_plugin_name(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_prompt_plugin(plugin_dir / "prompt_plugin.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    handlers = pm.get_prompt_handlers()

    assert len(handlers) == 1
    assert handlers[0]["plugin_name"] == "prompt_plugin"
    assert handlers[0]["chat_action"] == "typing"


def test_plugin_help_texts_include_plugin_name(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_prompt_plugin(plugin_dir / "prompt_plugin.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    assert pm.get_plugin_help_texts() == [{
        "plugin_name": "prompt_plugin",
        "text": "Prompt plugin help",
    }]


def test_strict_validation_raises_on_duplicate_function_names(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "shared.do")
    _write_plugin(plugin_dir / "beta.py", "BetaPlugin", "shared.do")
    monkeypatch.setenv("PLUGIN_STRICT_VALIDATION", "true")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    with pytest.raises(ValueError, match="Duplicate function name"):
        pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])


@pytest.mark.asyncio
async def test_call_function_passes_request_context_to_plugin_execute(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_context_plugin(plugin_dir / "context.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    request_context = RequestContext(
        chat_id=77,
        user_id=42,
        message_id=123,
        session_id="session-1",
    )

    result = await pm.call_function(
        "context.do",
        helper=None,
        arguments=json.dumps({
            "chat_id": 999,
            "user_id": 999,
            "message_id": 999,
            "request_context": {"user_id": 999},
        }),
        request_context=request_context,
    )

    payload = json.loads(result)
    assert payload["chat_id"] == 77
    assert payload["user_id"] == 42
    assert payload["message_id"] == 123
    assert payload["context_user_id"] == 42
    assert payload["context_message_id"] == 123


@pytest.mark.asyncio
async def test_call_function_removes_model_supplied_request_context_without_request_context(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_request_context_probe_plugin(plugin_dir / "probe.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    result = await pm.call_function(
        "probe.do",
        helper=None,
        arguments=json.dumps({
            "chat_id": 77,
            "user_id": 42,
            "request_context": {"user_id": 999},
        }),
    )

    assert json.loads(result) == {
        "has_request_context": False,
        "chat_id": 77,
        "user_id": 42,
    }


@pytest.mark.asyncio
async def test_call_function_propagates_cancelled_error_from_plugin_execute(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_cancel_plugin(plugin_dir / "cancel.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    with pytest.raises(asyncio.CancelledError):
        await pm.call_function("cancel.do", helper=None, arguments="{}")


@pytest.mark.asyncio
async def test_call_function_returns_error_when_spec_missing_and_does_not_execute(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    plugin = pm.get_plugin("alpha")
    plugin.execute = AsyncMock(return_value={"result": "should-not-run"})
    monkeypatch.setattr(pm, "get_spec_by_function_name", lambda _function_name: None)

    result = await pm.call_function("alpha.do", helper=None, arguments="{}")

    assert "Function spec for alpha.do not found" in json.loads(result)["error"]
    plugin.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_function_records_missing_spec_as_error_telemetry(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    plugin = pm.get_plugin("alpha")
    plugin.execute = AsyncMock(return_value={"result": "should-not-run"})
    monkeypatch.setattr(pm, "get_spec_by_function_name", lambda _function_name: None)
    events = []

    class FakeDB:
        def record_tool_call_event(self, **kwargs):
            events.append(kwargs)

    result = await pm.call_function(
        "alpha.do",
        helper=SimpleNamespace(db=FakeDB()),
        arguments=json.dumps({"chat_id": 77, "user_id": 42}),
    )

    assert "Function spec for alpha.do not found" in json.loads(result)["error"]
    plugin.execute.assert_not_awaited()
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert "Function spec for alpha.do not found" in events[0]["error"]
    assert events[0]["chat_id"] == 77
    assert events[0]["user_id"] == 42


@pytest.mark.asyncio
async def test_call_function_respects_plugin_guard(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_guard_plugin(plugin_dir / "guard.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    result = await pm.call_function("guard.do", helper=None, arguments="{}")

    assert json.loads(result) == {"success": False, "error": "blocked"}


def test_disabled_plugins_for_user_without_db_returns_empty(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    assert pm.disabled_plugins_for_user(42) == set()
    assert pm.disabled_plugins_for_user(None) == set()


def test_disabled_plugins_for_user_with_db(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def get_user_settings(self, user_id):
            if user_id == 42:
                return {"disabled_plugins": ["weather", "time"]}
            return {}

    pm.set_db(FakeDB())
    assert pm.disabled_plugins_for_user(42) == {"weather", "time"}
    assert pm.disabled_plugins_for_user(7) == set()
    assert pm.disabled_plugins_for_user(None) == set()


def test_is_plugin_disabled_for_user_handles_none_inputs(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def get_user_settings(self, user_id):
            return {"disabled_plugins": ["weather"]}

    pm.set_db(FakeDB())
    assert pm.is_plugin_disabled_for_user(None, 42) is False
    assert pm.is_plugin_disabled_for_user("weather", None) is False
    assert pm.is_plugin_disabled_for_user("weather", 42) is True
    assert pm.is_plugin_disabled_for_user("", 42) is False


def test_disabled_plugins_for_user_normalizes_list(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def get_user_settings(self, user_id):
            return {
                "disabled_plugins": [
                    "  weather  ",
                    "weather",
                    "",
                    "   ",
                    "time",
                    "time",
                ],
            }

    pm.set_db(FakeDB())
    assert pm.disabled_plugins_for_user(42) == {"weather", "time"}


def test_disabled_plugins_for_user_invalid_settings_shape(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def get_user_settings(self, user_id):
            return None

    pm.set_db(FakeDB())
    assert pm.disabled_plugins_for_user(42) == set()


def test_user_settings_scope_reuses_disabled_plugin_and_skill_settings(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def __init__(self):
            self.calls = []

        def get_user_settings(self, user_id):
            self.calls.append(user_id)
            return {
                "disabled_plugins": ["weather"],
                "disabled_skills": ["demo"],
            }

    db = FakeDB()
    pm.set_db(db)

    with pm.user_settings_scope(42):
        assert pm.disabled_plugins_for_user(42) == {"weather"}
        assert pm.is_plugin_disabled_for_user("weather", 42) is True
        assert pm.disabled_skills_for_user(42) == {"demo"}
        assert db.calls == [42]

    assert pm.disabled_plugins_for_user(42) == {"weather"}
    assert db.calls == [42, 42]


@pytest.mark.asyncio
async def test_user_settings_scope_reuses_settings_across_plugin_phases(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    class FakeDB:
        def __init__(self):
            self.calls = []

        def get_user_settings(self, user_id):
            self.calls.append(user_id)
            return {
                "disabled_plugins": ["blocked"],
                "disabled_skills": ["demo"],
            }

    class AllowedPlugin:
        plugin_id = "allowed"

        def get_plugin_id(self):
            return self.plugin_id

        async def on_user_message(self, payload):
            return None

        async def contribute_prompt_fragment(self, slot, payload):
            return "allowed"

        async def on_before_chat_request(self, messages, payload):
            return messages

    class BlockedPlugin(AllowedPlugin):
        plugin_id = "blocked"

    pm.set_db(FakeDB())
    pm.plugins.update({"allowed": AllowedPlugin, "blocked": BlockedPlugin})
    pm.plugin_instances.update({
        "allowed": AllowedPlugin(),
        "blocked": BlockedPlugin(),
    })

    with pm.user_settings_scope(42):
        await pm.dispatch_observe("on_user_message", "payload", user_id=42)
        assert await pm.collect_fragments("slot", "payload", user_id=42) == ["allowed"]
        assert await pm.apply_mutators("on_before_chat_request", "payload", [], user_id=42) == []
        assert pm.disabled_skills_for_user(42) == {"demo"}

    assert pm.db.calls == [42]


@pytest.mark.asyncio
async def test_call_function_records_tool_telemetry(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    events = []

    class FakeDB:
        def record_tool_call_event(self, **kwargs):
            events.append(kwargs)

    request_context = RequestContext(
        chat_id=77,
        user_id=42,
        message_id=123,
        session_id="session-1",
        request_id="req-1",
    )

    result = await pm.call_function(
        "alpha.do",
        helper=SimpleNamespace(db=FakeDB()),
        arguments=json.dumps({"chat_id": 77, "user_id": 42}),
        request_context=request_context,
    )

    assert json.loads(result) == {"result": "ok"}
    assert len(events) == 1
    assert events[0]["function_name"] == "alpha.do"
    assert events[0]["plugin_name"] == "alpha"
    assert events[0]["status"] == "success"
    assert events[0]["chat_id"] == 77
    assert events[0]["user_id"] == 42
    assert events[0]["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_call_function_records_ok_false_as_error(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_ok_false_plugin(plugin_dir / "ok_false.py")
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    events = []

    class FakeDB:
        def record_tool_call_event(self, **kwargs):
            events.append(kwargs)

    result = await pm.call_function(
        "ok_false.do",
        helper=SimpleNamespace(db=FakeDB()),
        arguments=json.dumps({"chat_id": 77, "user_id": 42}),
    )

    assert json.loads(result) == {"ok": False, "code": "REMOTE_BLOCKED", "message": "blocked"}
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["error"] == "blocked"
