import textwrap
from pathlib import Path

import pytest

from bot.plugin_manager import PluginManager


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


def test_namespacing_and_collision(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    _write_plugin(plugin_dir / "deepl.py", "DeeplPlugin", "translate")
    _write_plugin(plugin_dir / "ddg_translate.py", "DDGTranslatePlugin", "translate")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    specs = pm.get_functions_specs(helper=None, model_to_use="openai/gpt-4.1", allowed_plugins=["All"])
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
