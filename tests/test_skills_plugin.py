import logging

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.skills import SkillsPlugin


def _write_skill(root, name="demo", *, allow_scripts=True):
    skill_dir = root / name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    allow_scripts_line = "allow_scripts: false\n" if not allow_scripts else ""
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Demo skill for tests\n"
            f"{allow_scripts_line}"
            "---\n"
            "# Demo\n\n"
            "Use the script when needed.\n"
        ),
        encoding="utf-8",
    )
    (scripts_dir / "echo.py").write_text(
        (
            "import json\n"
            "import sys\n"
            "print('args=' + json.dumps(sys.argv[1:]))\n"
        ),
        encoding="utf-8",
    )
    return skill_dir


def _make_plugin(tmp_path, monkeypatch, *, allow_scripts=False, admin_ids=""):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    if allow_scripts:
        monkeypatch.setenv("SKILLS_ALLOW_SCRIPTS", "true")
    else:
        monkeypatch.delenv("SKILLS_ALLOW_SCRIPTS", raising=False)
    if admin_ids:
        monkeypatch.setenv("SKILLS_SCRIPT_ADMIN_USER_IDS", admin_ids)
    else:
        monkeypatch.delenv("SKILLS_SCRIPT_ADMIN_USER_IDS", raising=False)

    _write_skill(skills_dir)
    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))
    return plugin


def test_skills_plugin_registers_specs(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill(skills_dir)
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("PLUGIN_STORAGE_ROOT", str(storage_dir))

    pm = PluginManager(config={"plugins": ["skills"]})
    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["skills"])
    names = {spec["function"]["name"] for spec in specs}

    assert names == {
        "skills.list_skills",
        "skills.get_skill",
        "skills.activate_skill",
        "skills.get_skill_status",
        "skills.update_skill_progress",
        "skills.run_skill_script",
    }


def test_skill_scan_accepts_frontmatter_description_with_colon(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    skill_dir = skills_dir / "workflow"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: workflow\n"
            "description: Full workflow: extract -> analyze -> report\n"
            "allow_scripts: false\n"
            "---\n"
            "Follow the workflow.\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["workflow"]["description"] == "Full workflow: extract -> analyze -> report"
    assert plugin.available_skills["workflow"]["metadata"]["allow_scripts"] is False


def test_initialize_logs_available_skills(tmp_path, monkeypatch, caplog):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill(skills_dir)
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    caplog.set_level(logging.INFO, logger="bot.plugins.skills")

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert "Available skills in" in caplog.text
    assert "demo" in caplog.text


@pytest.mark.asyncio
async def test_skill_scan_activation_and_progress(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)
    assert listed["success"] is True
    assert listed["skills"][0]["id"] == "demo"
    assert listed["skills"][0]["scripts"] == ["echo.py"]

    details = await plugin.execute("get_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)
    assert details["success"] is True
    assert "Use the script" in details["skill"]["body"]

    activated = await plugin.execute(
        "activate_skill",
        helper=None,
        skill_name="demo",
        initial_context='{"task": "inspect"}',
        chat_id=10,
        user_id=42,
    )
    assert activated["success"] is True
    assert activated["scope"] == "chat:10"

    updated = await plugin.execute(
        "update_skill_progress",
        helper=None,
        skill_name="demo",
        step=2,
        context_update='{"result": "ok"}',
        chat_id=10,
        user_id=42,
    )
    assert updated["success"] is True
    assert updated["state"]["current_step"] == 2
    assert updated["state"]["context"] == {"task": "inspect", "result": "ok"}

    status = await plugin.execute("get_skill_status", helper=None, skill_name="demo", chat_id=10, user_id=42)
    assert status["success"] is True
    assert status["state"]["current_step"] == 2


@pytest.mark.asyncio
async def test_list_skills_refreshes_by_default(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    _write_skill(tmp_path / "skills", name="later")

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)

    assert {skill["id"] for skill in listed["skills"]} == {"demo", "later"}


@pytest.mark.asyncio
async def test_skill_state_persists_between_instances(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    await plugin.execute(
        "activate_skill",
        helper=None,
        skill_name="demo",
        initial_context='{"task": "persist"}',
        chat_id=10,
        user_id=42,
    )

    restored = SkillsPlugin()
    restored.initialize(storage_root=str(tmp_path / "storage"))

    status = await restored.execute("get_skill_status", helper=None, skill_name="demo", chat_id=10, user_id=42)

    assert status["success"] is True
    assert status["state"]["context"] == {"task": "persist"}


@pytest.mark.asyncio
async def test_skill_script_execution_requires_explicit_enablement(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.py",
        args_json='["one"]',
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "disabled" in result["error"]


@pytest.mark.asyncio
async def test_skill_script_execution_uses_allowlist_and_json_args(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    denied = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="../echo.py",
        args_json='["one"]',
        chat_id=10,
        user_id=42,
    )
    assert denied["success"] is False

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.py",
        args_json='["one", {"two": 2}]',
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["returncode"] == 0
    assert '["one", "{\\"two\\": 2}"]' in result["stdout"]


@pytest.mark.asyncio
async def test_skill_script_execution_is_admin_restricted(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=99)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.py",
        args_json='["one"]',
        chat_id=10,
        user_id=99,
    )

    assert result["success"] is False
    assert "restricted" in result["error"]


@pytest.mark.asyncio
async def test_skill_script_timeout_kills_process(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    plugin.script_timeout = 1
    skill_dir = tmp_path / "skills" / "demo"
    (skill_dir / "scripts" / "slow.py").write_text(
        "import time\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    plugin.available_skills = plugin._scan_skills()
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="slow.py",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "timed out" in result["error"]
