import logging
import shutil

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
        "skills.publish_artifact",
        "skills.publish_result",
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


def test_skill_scan_lists_nested_and_non_python_scripts(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    scripts_dir = tmp_path / "skills" / "demo" / "scripts"
    tools_dir = scripts_dir / "tools"
    tools_dir.mkdir()
    (tools_dir / "build.js").write_text("console.log('ok')\n", encoding="utf-8")
    (scripts_dir / "run.sh").write_text("echo ok\n", encoding="utf-8")
    (scripts_dir / ".hidden.py").write_text("print('hidden')\n", encoding="utf-8")
    cache_dir = scripts_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "skip.pyc").write_bytes(b"skip")

    plugin.available_skills = plugin._scan_skills()

    assert plugin.available_skills["demo"]["scripts"] == ["echo.py", "run.sh", "tools/build.js"]


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
async def test_publish_artifact_returns_final_direct_result(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    artifact_path = tmp_path / "storage" / "silver_analysis.pptx"
    artifact_path.write_bytes(b"pptx-data")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_artifact",
        helper=None,
        skill_name="demo",
        file_path=str(artifact_path),
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["file_path"] == str(artifact_path)
    assert result["direct_result"] == {
        "kind": "file",
        "format": "path",
        "value": str(artifact_path),
        "defer": False,
    }


@pytest.mark.asyncio
async def test_publish_result_returns_text_and_multiple_artifacts(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    first_artifact = tmp_path / "storage" / "silver_analysis.pptx"
    second_artifact = tmp_path / "storage" / "silver_notes.docx"
    first_artifact.write_bytes(b"pptx-data")
    second_artifact.write_bytes(b"docx-data")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_result",
        helper=None,
        skill_name="demo",
        text="Итоговая сводка",
        artifacts=[
            {"file_path": str(first_artifact)},
            {"file_path": str(second_artifact)},
        ],
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["direct_result"]["kind"] == "final"
    assert result["direct_result"]["defer"] is False
    assert result["direct_result"]["text"] == "Итоговая сводка"
    assert [item["value"] for item in result["direct_result"]["artifacts"]] == [
        str(first_artifact),
        str(second_artifact),
    ]


@pytest.mark.asyncio
async def test_publish_result_artifacts_do_not_require_script_enablement(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    artifact_path = tmp_path / "storage" / "egg_presentation.pptx"
    artifact_path.write_bytes(b"pptx-data")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_result",
        helper=None,
        skill_name="demo",
        text="Готово",
        artifacts=[{"file_path": str(artifact_path)}],
        chat_id=10,
        user_id=99,
    )

    assert result["success"] is True
    assert result["direct_result"]["text"] == "Готово"
    assert result["direct_result"]["artifacts"][0]["value"] == str(artifact_path)


@pytest.mark.asyncio
async def test_publish_result_requires_text_or_artifacts(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=True, admin_ids="42")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_result",
        helper=None,
        skill_name="demo",
        text="",
        artifacts=[],
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "text, artifacts" in result["error"]


@pytest.mark.asyncio
async def test_publish_result_text_only_does_not_require_script_enablement(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=99)

    result = await plugin.execute(
        "publish_result",
        helper=None,
        skill_name="demo",
        text="Короткий итог",
        chat_id=10,
        user_id=99,
    )

    assert result["success"] is True
    assert result["direct_result"]["kind"] == "final"
    assert result["direct_result"]["text"] == "Короткий итог"
    assert result["direct_result"]["artifacts"] == []


@pytest.mark.asyncio
async def test_publish_artifact_requires_active_skill_not_script_admin(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    artifact_path = tmp_path / "storage" / "silver_analysis.pptx"
    artifact_path.write_bytes(b"pptx-data")

    inactive = await plugin.execute(
        "publish_artifact",
        helper=None,
        skill_name="demo",
        file_path=str(artifact_path),
        chat_id=10,
        user_id=42,
    )
    assert inactive["success"] is False
    assert "active" in inactive["error"]

    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)
    result = await plugin.execute(
        "publish_artifact",
        helper=None,
        skill_name="demo",
        file_path=str(artifact_path),
        chat_id=10,
        user_id=99,
    )
    assert result["success"] is True
    assert result["direct_result"]["value"] == str(artifact_path)


@pytest.mark.asyncio
async def test_publish_artifact_rejects_path_outside_controlled_storage(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir(exist_ok=True)
    outside_artifact = outside_dir / "secret.txt"
    outside_artifact.write_text("secret", encoding="utf-8")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_artifact",
        helper=None,
        skill_name="demo",
        file_path=str(outside_artifact),
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "controlled skill storage" in result["error"]


@pytest.mark.asyncio
async def test_publish_result_rejects_artifact_outside_controlled_storage(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch, allow_scripts=False)
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside_result"
    outside_dir.mkdir(exist_ok=True)
    outside_artifact = outside_dir / "secret.txt"
    outside_artifact.write_text("secret", encoding="utf-8")
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    result = await plugin.execute(
        "publish_result",
        helper=None,
        skill_name="demo",
        text="Done",
        artifacts=[{"file_path": str(outside_artifact)}],
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "controlled skill storage" in result["error"]


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
