import json
import textwrap
from pathlib import Path

import asyncio
import pytest

from bot.plugin_manager import PluginManager


def _write_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class DemoPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Demo"

    def get_spec(self):
        return [{
            "name": "do",
            "description": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"}
                },
                "required": ["text"]
            }
        }]

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "ok"}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def test_plugin_arg_validation(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "demo.py")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    bad_args = json.dumps({"count": 1})
    result = asyncio.run(pm.call_function("demo.do", None, bad_args))
    assert "Missing required arg" in result
