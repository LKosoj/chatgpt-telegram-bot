import json
import textwrap
from pathlib import Path

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
    assert payload["chat_id"] == "77"
    assert payload["context_user_id"] == 42
    assert payload["context_message_id"] == 123
