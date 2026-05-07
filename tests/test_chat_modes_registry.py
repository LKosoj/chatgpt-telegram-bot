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


def test_skills_agent_mode_is_registered():
    yaml_path = Path(__file__).resolve().parents[1] / "bot" / "chat_modes.yml"
    registry = ChatModesRegistry(str(yaml_path))

    mode = registry.get_mode_by_key("skills_agent")

    assert mode is not None
    assert mode["tools"] == ["All"]
    assert "skills.list_skills" in mode["prompt_start"]
    assert "skills.run_skill_script" in mode["prompt_start"]
    assert "можно активировать несколько подходящих skills" in mode["prompt_start"]
    assert "обязательно верните непустой ответ" in mode["prompt_start"]
