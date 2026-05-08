import json
import logging
import shutil
from types import SimpleNamespace
from pathlib import Path

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


class FakeSettingsDB:
    def __init__(self, settings):
        self.settings = settings

    def get_user_settings(self, user_id):
        return self.settings.get(user_id)


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
        "skills.find_installable_skills",
        "skills.install_skill",
        "skills.activate_skill",
        "skills.get_skill_status",
        "skills.list_active_skills",
        "skills.update_skill_progress",
        "skills.deactivate_skill",
        "skills.run_skill_script",
        "skills.record_skill_reflection",
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
    assert activated["skill"]["script_tool_calls"] == [{
        "tool_name": "skills.run_skill_script",
        "arguments": {"skill_name": "demo", "script_name": "echo.py"},
    }]

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
    assert status["script_tool_calls"] == [{
        "tool_name": "skills.run_skill_script",
        "arguments": {"skill_name": "demo", "script_name": "echo.py"},
    }]

    all_status = await plugin.execute("get_skill_status", helper=None, chat_id=10, user_id=42)
    assert all_status["active_script_tool_calls"] == [{
        "tool_name": "skills.run_skill_script",
        "arguments": {"skill_name": "demo", "script_name": "echo.py"},
    }]


@pytest.mark.asyncio
async def test_list_skills_refreshes_by_default(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    _write_skill(tmp_path / "skills", name="later")

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)

    assert {skill["id"] for skill in listed["skills"]} == {"demo", "later"}


@pytest.mark.asyncio
async def test_user_disabled_skills_are_hidden_and_rejected(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    helper = SimpleNamespace(db=FakeSettingsDB({42: {"disabled_skills": ["demo"]}}))

    listed = await plugin.execute("list_skills", helper=helper, chat_id=10, user_id=42)
    details = await plugin.execute("get_skill", helper=helper, skill_name="demo", chat_id=10, user_id=42)
    activated = await plugin.execute("activate_skill", helper=helper, skill_name="demo", chat_id=10, user_id=42)

    assert listed["skills"] == []
    assert details["success"] is False
    assert "disabled" in details["error"]
    assert activated["success"] is False
    assert "disabled" in activated["error"]


@pytest.mark.asyncio
async def test_list_skills_marks_allow_scripts_per_skill(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    _write_skill(tmp_path / "skills", name="no_scripts", allow_scripts=False)

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)
    by_id = {skill["id"]: skill for skill in listed["skills"]}

    assert by_id["demo"]["allow_scripts"] is True
    assert by_id["no_scripts"]["allow_scripts"] is False


@pytest.mark.asyncio
async def test_list_skills_disables_allow_scripts_when_global_off(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)
    assert listed["skills"][0]["allow_scripts"] is False
    assert listed["scripts_enabled"] is False


def test_skill_scan_lists_nested_and_non_python_scripts(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    scripts_dir = tmp_path / "skills" / "demo" / "scripts"
    tools_dir = scripts_dir / "tools"
    tools_dir.mkdir()
    (tools_dir / "build.js").write_text("console.log('ok')\n", encoding="utf-8")
    (scripts_dir / "run.sh").write_text("echo ok\n", encoding="utf-8")
    (scripts_dir / "__init__.py").write_text("", encoding="utf-8")
    (scripts_dir / "schema.xsd").write_text("<schema />\n", encoding="utf-8")
    (scripts_dir / "library.jar").write_bytes(b"jar")
    (scripts_dir / ".hidden.py").write_text("print('hidden')\n", encoding="utf-8")
    cache_dir = scripts_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "skip.pyc").write_bytes(b"skip")

    plugin.available_skills = plugin._scan_skills()

    assert plugin.available_skills["demo"]["scripts"] == ["echo.py", "run.sh", "tools/build.js"]


def test_skill_scan_returns_cached_result_when_unchanged(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    first = plugin._scan_skills()
    second = plugin._scan_skills()

    assert second is first


def test_skill_scan_invalidates_cache_when_script_added(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    first = plugin._scan_skills()

    new_script = tmp_path / "skills" / "demo" / "scripts" / "extra.py"
    new_script.write_text("print('extra')\n", encoding="utf-8")

    second = plugin._scan_skills()
    assert second is not first
    assert "extra.py" in second["demo"]["scripts"]


@pytest.mark.asyncio
async def test_skill_reflection_applies_repeated_proposal_after_threshold(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    proposal = "If a referenced helper is missing, inspect available scripts before choosing a fallback."

    for count in range(1, 4):
        result = await plugin.execute(
            "record_skill_reflection",
            helper=None,
            skill_name="demo",
            proposal=proposal,
            failure_mode="missing helper",
            evidence=f"attempt {count}",
            chat_id=10,
            user_id=42,
        )
        assert result["success"] is True
        assert result["count"] == count
        assert result["threshold_reached"] is False
        assert result["applied"] is False

    skill_md = tmp_path / "skills" / "demo" / "SKILL.md"
    assert "Learned Clarifications" not in skill_md.read_text(encoding="utf-8")

    applied = await plugin.execute(
        "record_skill_reflection",
        helper=None,
        skill_name="demo",
        proposal=proposal,
        failure_mode="missing helper",
        evidence="attempt 4",
        chat_id=10,
        user_id=42,
    )

    assert applied["success"] is True
    assert applied["count"] == 4
    assert applied["threshold"] == 3
    assert applied["threshold_reached"] is True
    assert applied["applied"] is True

    content = skill_md.read_text(encoding="utf-8")
    assert "## Learned Clarifications" in content
    assert f"- {proposal}" in content

    details = await plugin.execute("get_skill", helper=None, skill_name="demo")
    assert proposal in details["skill"]["body"]

    stored = json.loads((tmp_path / "storage" / "skill_reflections.json").read_text(encoding="utf-8"))
    [entry] = stored["demo"].values()
    assert entry["count"] == 4
    assert entry["applied"] is True
    assert len(entry["examples"]) == 3


@pytest.mark.asyncio
async def test_find_installable_skills_parses_cli_results(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    async def fake_run(args, *, timeout):
        assert args == ["find", "testing"]
        return {
            "success": True,
            "returncode": 0,
            "stdout": (
                "Install with npx skills add <owner/repo@skill>\n\n"
                "anthropics/skills@webapp-testing 63K installs\n"
                "└ https://skills.sh/anthropics/skills/webapp-testing\n"
            ),
            "stderr": "",
        }

    plugin._run_skills_cli = fake_run

    result = await plugin.execute("find_installable_skills", helper=None, query="testing")

    assert result["success"] is True
    assert result["results"] == [{
        "package": "anthropics/skills@webapp-testing",
        "skill_name": "webapp-testing",
        "summary": "63K installs",
        "url": "https://skills.sh/anthropics/skills/webapp-testing",
    }]


@pytest.mark.asyncio
async def test_install_skill_can_be_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLS_ALLOW_INSTALLS", "false")
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="anthropics/skills@webapp-testing",
        confirmed=True,
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "disabled" in result["error"]


@pytest.mark.asyncio
async def test_install_skill_allows_all_users_by_default(tmp_path, monkeypatch):
    installed_dir = tmp_path / "global" / "webapp-testing"
    installed_dir.mkdir(parents=True)
    (installed_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: webapp-testing\n"
            "description: Browser testing skill\n"
            "---\n"
            "# Webapp Testing\n"
        ),
        encoding="utf-8",
    )
    plugin = _make_plugin(tmp_path, monkeypatch)

    async def fake_run(args, *, timeout):
        if args[:2] == ["add", "anthropics/skills@webapp-testing"]:
            return {"success": True, "returncode": 0, "stdout": "installed\n", "stderr": ""}
        if args == ["ls", "-g", "--json"]:
            return {
                "success": True,
                "returncode": 0,
                "stdout": json.dumps([{"name": "webapp-testing", "path": str(installed_dir)}]),
                "stderr": "",
            }
        raise AssertionError(f"Unexpected skills CLI args: {args}")

    plugin._run_skills_cli = fake_run

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="anthropics/skills@webapp-testing",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["skill"] == "webapp-testing"
    assert result["sync"] == "copied_to_skills_dir"
    assert "webapp-testing" in plugin.available_skills
    assert (tmp_path / "skills" / "webapp-testing" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_install_skill_restricts_to_configured_users(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLS_INSTALL_ADMIN_USER_IDS", "42")
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="owner/repo@restricted-skill",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "allow-list" in result["error"]


@pytest.mark.asyncio
async def test_install_skill_syncs_existing_cli_install_when_add_fails(tmp_path, monkeypatch):
    installed_dir = tmp_path / "global" / "existing-skill"
    installed_dir.mkdir(parents=True)
    (installed_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: existing-skill\n"
            "description: Already installed globally\n"
            "---\n"
            "# Existing Skill\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLS_ALLOW_INSTALLS", "true")
    monkeypatch.setenv("SKILLS_INSTALL_ADMIN_USER_IDS", "42")
    plugin = _make_plugin(tmp_path, monkeypatch)

    async def fake_run(args, *, timeout):
        if args[:2] == ["add", "owner/repo@existing-skill"]:
            return {"success": False, "returncode": 1, "stdout": "", "stderr": "already installed\n"}
        if args == ["ls", "-g", "--json"]:
            return {
                "success": True,
                "returncode": 0,
                "stdout": json.dumps([{"name": "existing-skill", "path": str(installed_dir)}]),
                "stderr": "",
            }
        raise AssertionError(f"Unexpected skills CLI args: {args}")

    plugin._run_skills_cli = fake_run

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="owner/repo@existing-skill",
        confirmed=True,
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["sync"] == "copied_to_skills_dir"
    assert "warning" in result
    assert "existing-skill" in plugin.available_skills


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
    assert result["runtime"] == "python"
    assert '["one", "{\\"two\\": 2}"]' in result["stdout"]


@pytest.mark.asyncio
async def test_skill_script_execution_supports_nested_python_scripts(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    tools_dir = tmp_path / "skills" / "demo" / "scripts" / "tools"
    tools_dir.mkdir()
    (tools_dir / "nested.py").write_text(
        (
            "import json\n"
            "import sys\n"
            "print('nested=' + json.dumps(sys.argv[1:]))\n"
        ),
        encoding="utf-8",
    )
    plugin.available_skills = plugin._scan_skills()
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="tools/nested.py",
        args_json='["one"]',
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["runtime"] == "python"
    assert 'nested=["one"]' in result["stdout"]


@pytest.mark.asyncio
async def test_skill_script_execution_supports_node_runtime(tmp_path, monkeypatch):
    if shutil.which("node") is None:
        pytest.skip("node runtime is not installed")

    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    script_path = tmp_path / "skills" / "demo" / "scripts" / "echo.js"
    script_path.write_text(
        "console.log('node_args=' + JSON.stringify(process.argv.slice(2)))\n",
        encoding="utf-8",
    )
    plugin.available_skills = plugin._scan_skills()
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.js",
        args_json='["one"]',
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["runtime"] == "node"
    assert 'node_args=["one"]' in result["stdout"]


@pytest.mark.asyncio
async def test_skill_script_execution_reports_missing_runtime(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    script_path = tmp_path / "skills" / "demo" / "scripts" / "echo.js"
    script_path.write_text("console.log('ok')\n", encoding="utf-8")
    plugin.available_skills = plugin._scan_skills()
    monkeypatch.setattr("bot.plugins.skills.shutil.which", lambda _name: None)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.js",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "node" in result["error"]


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
async def test_deactivate_skill_clears_state(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    deactivated = await plugin.execute(
        "deactivate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42,
    )
    assert deactivated["success"] is True
    assert deactivated["deactivated"] is True

    status = await plugin.execute(
        "get_skill_status", helper=None, skill_name="demo", chat_id=10, user_id=42,
    )
    assert status["success"] is False
    assert "is not active" in status["error"]


@pytest.mark.asyncio
async def test_cleanup_after_delivery_removes_active_skill(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    status = await plugin.execute(
        "get_skill_status", helper=None, skill_name="demo", chat_id=10, user_id=42,
    )
    assert status["success"] is True

    assert plugin.cleanup_after_delivery(
        {"plugin_id": "skills", "scope": "chat:10", "skill_id": "demo"}
    ) is True

    status_after = await plugin.execute(
        "get_skill_status", helper=None, skill_name="demo", chat_id=10, user_id=42,
    )
    assert status_after["success"] is False


@pytest.mark.asyncio
async def test_cleanup_after_delivery_idempotent_for_unknown_skill(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    assert plugin.cleanup_after_delivery({"scope": "chat:10", "skill_id": "missing"}) is False
    assert plugin.cleanup_after_delivery({}) is False
    assert plugin.cleanup_after_delivery(None) is False


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


@pytest.mark.asyncio
async def test_list_active_skills_returns_compact_summary(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    await plugin.execute(
        "activate_skill",
        helper=None,
        skill_name="demo",
        initial_context='{"task": "x"}',
        chat_id=10,
        user_id=42,
    )

    result = await plugin.execute(
        "list_active_skills",
        helper=None,
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["count"] == 1
    [entry] = result["active_skills"]
    assert entry["id"] == "demo"
    assert "current_step" in entry
    assert "context" not in entry


@pytest.mark.asyncio
async def test_audit_log_records_lifecycle_events(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="*")
    await plugin.execute(
        "activate_skill",
        helper=None,
        skill_name="demo",
        chat_id=10,
        user_id=42,
    )
    await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="echo.py",
        chat_id=10,
        user_id=42,
    )
    await plugin.execute(
        "deactivate_skill",
        helper=None,
        skill_name="demo",
        chat_id=10,
        user_id=42,
    )

    audit_path = tmp_path / "storage" / "skills_audit.jsonl"
    assert audit_path.exists()
    actions = [
        json.loads(line)["action"]
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "activate_skill" in actions
    assert "run_skill_script" in actions
    assert "deactivate_skill" in actions


@pytest.mark.asyncio
async def test_skill_script_receives_workdir_env(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="*")
    scripts_dir = tmp_path / "skills" / "demo" / "scripts"
    (scripts_dir / "show_workdir.py").write_text(
        (
            "import os\n"
            "print('SKILL_WORKDIR=' + os.environ.get('SKILL_WORKDIR', ''))\n"
            "print('SKILL_ID=' + os.environ.get('SKILL_ID', ''))\n"
            "print('SKILL_SCOPE=' + os.environ.get('SKILL_SCOPE', ''))\n"
        ),
        encoding="utf-8",
    )
    plugin.available_skills = plugin._scan_skills()
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "run_skill_script",
        helper=None,
        skill_name="demo",
        script_name="show_workdir.py",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["workdir"] is not None
    workdir_path = Path(result["workdir"])
    assert workdir_path.exists()
    assert "SKILL_WORKDIR=" in result["stdout"]
    assert str(workdir_path) in result["stdout"]
    assert "SKILL_ID=demo" in result["stdout"]
    assert "SKILL_SCOPE=chat:10" in result["stdout"]


def _write_skill_with_entrypoints(root, *, name="ep", entrypoints_block: str):
    """Helper for entrypoint tests: skill with two scripts and configurable frontmatter."""
    skill_dir = root / name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Skill with explicit entrypoints\n"
            f"{entrypoints_block}"
            "---\n"
            "Body.\n"
        ),
        encoding="utf-8",
    )
    (scripts_dir / "main.py").write_text("print('main')\n", encoding="utf-8")
    (scripts_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    return skill_dir


def test_explicit_entrypoints_filter_helpers(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(
        skills_dir, entrypoints_block="entrypoints:\n  - main.py\n"
    )
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == ["main.py"]


def test_no_entrypoints_falls_back_to_heuristic(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(skills_dir, entrypoints_block="")
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == ["helper.py", "main.py"]


def test_empty_entrypoints_list_yields_no_scripts(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(skills_dir, entrypoints_block="entrypoints: []\n")
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == []


def test_entrypoint_string_treated_as_single_item(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(
        skills_dir, entrypoints_block="entrypoints: main.py\n"
    )
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == ["main.py"]


def test_entrypoint_traversal_is_rejected(tmp_path, monkeypatch, caplog):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(
        skills_dir, entrypoints_block="entrypoints:\n  - ../../etc/passwd\n  - main.py\n"
    )
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    caplog.set_level(logging.WARNING, logger="bot.plugins.skills")

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == ["main.py"]
    assert "traversal" in caplog.text or "invalid" in caplog.text.lower()


def test_entrypoint_missing_file_is_skipped(tmp_path, monkeypatch, caplog):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    _write_skill_with_entrypoints(
        skills_dir, entrypoints_block="entrypoints:\n  - missing.py\n  - main.py\n"
    )
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    caplog.set_level(logging.WARNING, logger="bot.plugins.skills")

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["ep"]["scripts"] == ["main.py"]
    assert "does not exist" in caplog.text


def test_entrypoint_subpath_is_allowed(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    skill_dir = skills_dir / "nested"
    scripts_dir = skill_dir / "scripts"
    (scripts_dir / "tools").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: nested\n"
            "description: nested entrypoint\n"
            "entrypoints:\n"
            "  - tools/run.py\n"
            "---\n"
            "Body.\n"
        ),
        encoding="utf-8",
    )
    (scripts_dir / "tools" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (scripts_dir / "lib.py").write_text("# library\n", encoding="utf-8")

    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    assert plugin.available_skills["nested"]["scripts"] == ["tools/run.py"]
