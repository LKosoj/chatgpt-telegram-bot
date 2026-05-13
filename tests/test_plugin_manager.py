import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

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
            "context_user_id": request_context.user_id,
            "context_message_id": request_context.message_id,
        }
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


def test_namespacing_and_collision(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "deepl.py", "DeeplPlugin", "translate")
    _write_plugin(plugin_dir / "ddg_translate.py", "DDGTranslatePlugin", "translate")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["All"])
    names = [s["function"]["name"] for s in specs]
    assert "deepl.translate" in names
    assert "ddg_translate.translate" in names


@pytest.mark.asyncio
async def test_call_function_lookup_skips_unrelated_broken_plugin(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_bad_spec_plugin(plugin_dir / "aaa_bad.py")
    _write_plugin(plugin_dir / "good.py", "GoodPlugin", "do")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

    result = await pm.call_function("good.do", None, "{}")

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
        arguments=json.dumps({"chat_id": request_context.plugin_chat_id}),
        request_context=request_context,
    )

    payload = json.loads(result)
    assert payload["chat_id"] == 77
    assert payload["context_user_id"] == 42
    assert payload["context_message_id"] == 123


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
