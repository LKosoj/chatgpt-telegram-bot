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
    assert mode["defer_direct_results"] is True
    assert "skills.list_skills" in mode["prompt_start"]
    assert "skills.run_skill_script" in mode["prompt_start"]
    assert "skills.record_skill_reflection" in mode["prompt_start"]
    assert "skills.find_installable_skills" in mode["prompt_start"]
    assert "skills.install_skill" in mode["prompt_start"]
    assert "terminal.terminal" in mode["prompt_start"]
    assert "больше двух шагов" in mode["prompt_start"]
    assert "agent_tools.manage_plan_tasks" in mode["prompt_start"]
    assert "agent_tools.deliver_to_user" in mode["prompt_start"]
    assert "skills.publish_result" not in mode["prompt_start"]
    assert "skills.publish_artifact" not in mode["prompt_start"]
    assert "agent_tools.run_subagents" in mode["prompt_start"]
    assert "можно активировать несколько подходящих skills" in mode["prompt_start"]
    assert "status=completed" in mode["prompt_start"]
    assert "verification_summary" in mode["prompt_start"]
    assert "выберите ровно одно действие" in mode["prompt_start"]
    assert "Если tool нужен, вызовите его сразу" in mode["prompt_start"]
    assert "Никогда не выводите служебные reasoning-теги" in mode["prompt_start"]
    assert "Никогда не выводите сырые результаты tools" in mode["prompt_start"]
    assert "Не выдумывайте абсолютные пути" in mode["prompt_start"]
    assert "не повторяйте тот же вызов" in mode["prompt_start"]


def test_skills_agent_has_force_non_stream_first_turn_flag():
    yaml_path = Path(__file__).resolve().parents[1] / "bot" / "chat_modes.yml"
    registry = ChatModesRegistry(str(yaml_path))

    mode = registry.get_mode_by_key("skills_agent")

    assert mode is not None
    assert mode.get("force_non_stream_first_turn") is True


def test_skills_agent_mode_is_detected_by_prompt_markers():
    yaml_path = Path(__file__).resolve().parents[1] / "bot" / "chat_modes.yml"
    registry = ChatModesRegistry(str(yaml_path))

    mode = registry.get_mode_by_system_prompt(
        "Вы - агент, который умеет использовать локальные skills через инструменты skills.\n"
        "Старая версия prompt без mode_key."
    )

    assert mode is not None
    assert mode["defer_direct_results"] is True
