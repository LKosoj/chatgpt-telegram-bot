import textwrap
from pathlib import Path

from bot.plugin_manager import PluginManager


def _write_plugin(path: Path, class_name: str, command_name: str):
    code = f"""
from bot.plugins.plugin import Plugin

class {class_name}(Plugin):
    def get_source_name(self) -> str:
        return "{class_name}"

    def get_spec(self):
        return []

    async def execute(self, function_name, helper, **kwargs):
        return {{"result": "ok"}}

    def get_commands(self):
        return [{{\"command\": \"{command_name}\", \"description\": \"desc\", \"handler\": self.execute}}]
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def test_duplicate_plugin_commands(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "one.py", "OnePlugin", "dup")
    _write_plugin(plugin_dir / "two.py", "TwoPlugin", "dup")

    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    commands = pm.get_plugin_commands()
    names = [c["command"] for c in commands if c.get("command")]
    assert names.count("dup") == 1
