import logging
from pathlib import Path

from bot.chat_modes_registry import ChatModesRegistry


class DummyPluginManager:
    def __init__(self, available):
        self.available = set(available)

    def has_plugin(self, name):
        return name in self.available


def test_chat_modes_validate_tools_logs_error(tmp_path, caplog):
    yaml_path = tmp_path / "chat_modes.yml"
    yaml_path.write_text(
        """
assistant:
  tools:
    - missing_tool
  prompt_start: "hi"
  welcome_message: "hello"
""",
        encoding="utf-8",
    )

    registry = ChatModesRegistry(str(yaml_path))
    caplog.set_level(logging.ERROR)
    registry.validate_tools(DummyPluginManager(available={"ok"}))

    assert any("missing_tool" in rec.message for rec in caplog.records)
