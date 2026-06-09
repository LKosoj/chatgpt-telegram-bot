import importlib.machinery
import importlib.util
import json
import logging
import shutil
import sys
import tarfile
import types
import zipfile
from types import SimpleNamespace
from pathlib import Path

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

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


def _write_skill_agent(skill_dir, name="openai"):
    agents_dir = skill_dir / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / f"{name}.yaml").write_text(
        (
            "interface:\n"
            "  display_name: \"Novel Writing\"\n"
            "  short_description: \"Plan, draft, and review fiction\"\n"
            "  default_prompt: \"Use $novel-writing to plan, draft, or review a fiction chapter.\"\n"
        ),
        encoding="utf-8",
    )


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


class FakeAgentTools:
    def __init__(self):
        self.calls = []

    async def execute(self, function_name, helper, **kwargs):
        self.calls.append((function_name, kwargs))
        return {
            "success": True,
            "subagents": [{
                "id": kwargs["subagents"][0]["id"],
                "role": kwargs["subagents"][0]["role"],
                "status": "completed",
                "result": "drafted",
            }],
        }


class FakePluginManager:
    def __init__(self, agent_tools):
        self.agent_tools = agent_tools

    def get_plugin(self, name):
        return self.agent_tools if name == "agent_tools" else None


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
        "skills_list_skills",
        "skills_get_skill",
        "skills_get_skill_reference",
        "skills_get_skill_resource",
        "skills_find_installable_skills",
        "skills_install_skill",
        "skills_create_skill",
        "skills_activate_skill",
        "skills_get_skill_status",
        "skills_list_active_skills",
        "skills_update_skill_progress",
        "skills_deactivate_skill",
        "skills_run_skill_script",
        "skills_run_skill_agent",
        "skills_record_skill_reflection",
    }
    assert pm.to_canonical_function_name("skills_run_skill_agent") == "skills.run_skill_agent"


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


def test_external_project_capability_extraction_example_skill_scans(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "skills"
        / "external-project-capability-extraction"
    )
    target = skills_dir / "external-project-capability-extraction"
    shutil.copytree(source, target)
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    info = plugin.available_skills["external-project-capability-extraction"]
    assert "Read-only by default" in info["description"]
    assert {
        reference["reference_path"] for reference in info["references"]
    } == {
        "references/extraction-protocol.md",
        "references/output-schema.md",
        "references/security-checklist.md",
    }
    assert info["agents"] == [{
        "id": "openai",
        "file": "agents/openai.yaml",
        "display_name": "Capability Extractor",
        "short_description": "Extract reusable skills from projects",
        "default_prompt": "Use $external-project-capability-extraction to inspect an external project read-only and propose a local skill or SOP.",
    }]


@pytest.mark.asyncio
async def test_skill_scan_discovers_nested_meta_skills_and_references(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    skill_dir = skills_dir / "META-SKILLS" / "belief-examination"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "templates").mkdir(parents=True)
    (skill_dir / "data").mkdir(parents=True)
    (skills_dir / "META-SKILLS" / "_shared").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: belief-examination\n"
            "description: Nested meta skill\n"
            "---\n"
            "# Belief Examination\n\n"
            "Use `references/factual.md`, `META-SKILLS/_shared/core-principles.md`, "
            "and `templates/prompt.txt` when needed.\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "references" / "factual.md").write_text(
        "See `../../_shared/output-template.md` and `../data/schema.json`.\n",
        encoding="utf-8",
    )
    (skills_dir / "META-SKILLS" / "_shared" / "core-principles.md").write_text("# Core\n", encoding="utf-8")
    (skills_dir / "META-SKILLS" / "_shared" / "output-template.md").write_text("# Output\n", encoding="utf-8")
    (skill_dir / "templates" / "prompt.txt").write_text("Prompt template\n", encoding="utf-8")
    (skill_dir / "data" / "schema.json").write_text('{"type": "object"}\n', encoding="utf-8")
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))

    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)
    assert [skill["id"] for skill in listed["skills"]] == ["META-SKILLS/belief-examination"]

    details = await plugin.execute(
        "get_skill",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        chat_id=10,
        user_id=42,
    )
    assert details["success"] is True
    assert details["skill"]["references"] == [
        {
            "reference_path": "references/factual.md",
            "path": "META-SKILLS/belief-examination/references/factual.md",
        },
        {
            "reference_path": "META-SKILLS/_shared/core-principles.md",
            "path": "META-SKILLS/_shared/core-principles.md",
        },
        {
            "reference_path": "../../_shared/output-template.md",
            "path": "META-SKILLS/_shared/output-template.md",
        },
    ]
    assert details["skill"]["resources"] == [
        {
            "resource_path": "templates/prompt.txt",
            "path": "META-SKILLS/belief-examination/templates/prompt.txt",
            "size": len("Prompt template\n"),
            "encoding": "utf-8",
        },
        {
            "resource_path": "../data/schema.json",
            "path": "META-SKILLS/belief-examination/data/schema.json",
            "size": len('{"type": "object"}\n'),
            "encoding": "utf-8",
        },
    ]

    reference = await plugin.execute(
        "get_skill_reference",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        reference_path="META-SKILLS/_shared/core-principles.md",
        chat_id=10,
        user_id=42,
    )
    assert reference == {
        "success": True,
        "skill": "META-SKILLS/belief-examination",
        "reference_path": "META-SKILLS/_shared/core-principles.md",
        "path": "META-SKILLS/_shared/core-principles.md",
        "content": "# Core\n",
    }

    local_reference = await plugin.execute(
        "get_skill_reference",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        reference_path="references/factual.md",
        chat_id=10,
        user_id=42,
    )
    assert local_reference["success"] is True
    assert local_reference["path"] == "META-SKILLS/belief-examination/references/factual.md"
    assert local_reference["content"] == "See `../../_shared/output-template.md` and `../data/schema.json`.\n"

    recursive_reference = await plugin.execute(
        "get_skill_reference",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        reference_path="META-SKILLS/_shared/output-template.md",
        chat_id=10,
        user_id=42,
    )
    assert recursive_reference["success"] is True
    assert recursive_reference["content"] == "# Output\n"

    recursive_reference_by_original_path = await plugin.execute(
        "get_skill_reference",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        reference_path="../../_shared/output-template.md",
        chat_id=10,
        user_id=42,
    )
    assert recursive_reference_by_original_path["success"] is True
    assert recursive_reference_by_original_path["path"] == "META-SKILLS/_shared/output-template.md"

    resource = await plugin.execute(
        "get_skill_resource",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        resource_path="META-SKILLS/belief-examination/data/schema.json",
        chat_id=10,
        user_id=42,
    )
    assert resource == {
        "success": True,
        "skill": "META-SKILLS/belief-examination",
        "resource_path": "META-SKILLS/belief-examination/data/schema.json",
        "path": "META-SKILLS/belief-examination/data/schema.json",
        "encoding": "utf-8",
        "content": '{"type": "object"}\n',
    }

    resource_by_original_path = await plugin.execute(
        "get_skill_resource",
        helper=None,
        skill_name="META-SKILLS/belief-examination",
        resource_path="../data/schema.json",
        chat_id=10,
        user_id=42,
    )
    assert resource_by_original_path["success"] is True
    assert resource_by_original_path["path"] == "META-SKILLS/belief-examination/data/schema.json"


@pytest.mark.asyncio
async def test_skill_reference_rejects_unlisted_paths(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "get_skill_reference",
        helper=None,
        skill_name="demo",
        reference_path="../../outside.md",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "not listed" in result["error"]


@pytest.mark.asyncio
async def test_skill_resource_rejects_unlisted_paths(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "get_skill_resource",
        helper=None,
        skill_name="demo",
        resource_path="templates/prompt.txt",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "not listed" in result["error"]


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
async def test_skill_scan_exposes_agents_folder(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    skill_dir = tmp_path / "skills" / "demo"
    _write_skill_agent(skill_dir)
    plugin.available_skills = plugin._scan_skills()

    listed = await plugin.execute("list_skills", helper=None, chat_id=10, user_id=42)
    details = await plugin.execute("get_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    assert listed["skills"][0]["agents"] == ["openai"]
    assert details["skill"]["agents"] == [{
        "id": "openai",
        "file": "agents/openai.yaml",
        "display_name": "Novel Writing",
        "short_description": "Plan, draft, and review fiction",
        "default_prompt": "Use $novel-writing to plan, draft, or review a fiction chapter.",
    }]


@pytest.mark.asyncio
async def test_skill_scan_exposes_nested_agent_files(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    agents_dir = tmp_path / "skills" / "demo" / "agents" / "writing"
    agents_dir.mkdir(parents=True)
    (agents_dir / "openai.yaml").write_text(
        (
            "interface:\n"
            "  display_name: \"Writing Reviewer\"\n"
            "  short_description: \"Review long-form drafts\"\n"
            "  default_prompt: \"Use nested writing review instructions.\"\n"
        ),
        encoding="utf-8",
    )
    plugin.available_skills = plugin._scan_skills()

    details = await plugin.execute("get_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    assert details["skill"]["agents"] == [{
        "id": "writing/openai",
        "file": "agents/writing/openai.yaml",
        "display_name": "Writing Reviewer",
        "short_description": "Review long-form drafts",
        "default_prompt": "Use nested writing review instructions.",
    }]


@pytest.mark.asyncio
async def test_run_skill_agent_delegates_to_agent_tools(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    _write_skill_agent(tmp_path / "skills" / "demo")
    plugin.available_skills = plugin._scan_skills()
    agent_tools = FakeAgentTools()
    helper = SimpleNamespace(plugin_manager=FakePluginManager(agent_tools))
    plugin._activate_skill(
        "demo",
        {"delegation": {"chapter": "Draft chapter one.", "style": "Keep it concise."}},
        chat_id=10,
        user_id=42,
    )

    result = await plugin.execute(
        "run_skill_agent",
        helper=helper,
        skill_name="demo",
        agent_name="openai",
        context_ref="delegation",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is True
    assert result["subagent"]["result"] == "drafted"
    [(function_name, kwargs)] = agent_tools.calls
    assert function_name == "run_subagents"
    assert kwargs["chat_id"] == 10
    assert kwargs["user_id"] == 42
    [subagent] = kwargs["subagents"]
    assert subagent["id"] == "demo_openai"
    assert subagent["role"] == "Novel Writing"
    assert "Use $novel-writing" in subagent["task"]
    assert "Complete the delegated subtask" in subagent["task"]
    assert "Draft chapter one." in subagent["context"]
    assert "Keep it concise." in subagent["context"]


@pytest.mark.asyncio
async def test_run_skill_agent_rejects_inline_task_payload(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    _write_skill_agent(tmp_path / "skills" / "demo")
    plugin.available_skills = plugin._scan_skills()
    helper = SimpleNamespace(plugin_manager=FakePluginManager(FakeAgentTools()))

    result = await plugin.execute(
        "run_skill_agent",
        helper=helper,
        skill_name="demo",
        agent_name="openai",
        task="Draft chapter one.",
        chat_id=10,
        user_id=42,
    )

    assert result["success"] is False
    assert "no longer accepts task/context" in result["error"]


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
async def test_auto_mode_fragment_skips_user_disabled_skills(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    plugin.available_skills = {
        "demo": {"description": "Disabled demo skill"},
        "other": {"description": "Available skill"},
    }

    class FakePluginManager:
        def __init__(self):
            self.calls = []

        def disabled_skills_for_user(self, user_id):
            self.calls.append(user_id)
            return {"demo"}

    plugin_manager = FakePluginManager()
    plugin.openai = SimpleNamespace(plugin_manager=plugin_manager)

    fragment = await plugin.contribute_prompt_fragment(
        "auto_mode_priority",
        SimpleNamespace(user_id=42),
    )

    assert plugin_manager.calls == [42]
    assert "other: Available skill" in fragment
    assert "demo: Disabled demo skill" not in fragment


@pytest.mark.asyncio
async def test_auto_mode_fragment_marks_direct_skill_matches():
    plugin = SkillsPlugin()
    plugin.available_skills = {
        "powerpoint": {
            "name": "powerpoint",
            "description": "Create PPTX presentations, презентации, and слайды.",
        },
        "other": {
            "name": "other",
            "description": "Available skill",
        },
    }

    fragment = await plugin.contribute_prompt_fragment(
        "auto_mode_priority",
        SimpleNamespace(user_id=42, query="Создай презентацию pptx по рецепту"),
    )

    assert "прямое совпадение" in fragment
    assert "powerpoint" in fragment
    assert "pptx" in fragment


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


def test_skill_scan_invalidates_cache_when_reference_added(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    skill_md = tmp_path / "skills" / "demo" / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nSee `references/new.md`.\n",
        encoding="utf-8",
    )
    first = plugin._scan_skills()

    references_dir = tmp_path / "skills" / "demo" / "references"
    references_dir.mkdir()
    (references_dir / "new.md").write_text("# New\n", encoding="utf-8")

    second = plugin._scan_skills()
    assert second is not first
    assert second["demo"]["references"] == [{
        "reference_path": "references/new.md",
        "path": "demo/references/new.md",
    }]


def test_skill_scan_invalidates_cache_when_resource_added(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    skill_md = tmp_path / "skills" / "demo" / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nUse `templates/prompt.txt`.\n",
        encoding="utf-8",
    )
    first = plugin._scan_skills()

    templates_dir = tmp_path / "skills" / "demo" / "templates"
    templates_dir.mkdir()
    (templates_dir / "prompt.txt").write_text("Prompt\n", encoding="utf-8")

    second = plugin._scan_skills()
    assert second is not first
    assert second["demo"]["resources"] == [{
        "resource_path": "templates/prompt.txt",
        "path": "demo/templates/prompt.txt",
        "size": len("Prompt\n"),
        "encoding": "utf-8",
    }]


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
async def test_install_skill_supports_nested_target_path(tmp_path, monkeypatch):
    installed_dir = tmp_path / "global" / "existing-skill"
    installed_dir.mkdir(parents=True)
    (installed_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: existing-skill\n"
            "description: Existing skill installed into nested target\n"
            "---\n"
            "# Existing Skill\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLS_ALLOW_INSTALLS", "true")
    plugin = _make_plugin(tmp_path, monkeypatch)

    async def fake_run(args, *, timeout):
        if args[:2] == ["add", "owner/repo@existing-skill"]:
            return {"success": True, "returncode": 0, "stdout": "installed\n", "stderr": ""}
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
        skill_name="META-SKILLS/existing-skill",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["skill"] == "META-SKILLS/existing-skill"
    assert result["source_skill"] == "existing-skill"
    assert "META-SKILLS/existing-skill" in plugin.available_skills
    assert (tmp_path / "skills" / "META-SKILLS" / "existing-skill" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_install_skill_supports_local_directory_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "source skills" / "local-skill"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: local-skill\n"
            "description: Local directory skill\n"
            "---\n"
            "# Local Skill\n"
        ),
        encoding="utf-8",
    )
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package=str(source_dir),
        skill_name="META-SKILLS/local-skill",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "local"
    assert result["skill"] == "META-SKILLS/local-skill"
    assert "META-SKILLS/local-skill" in plugin.available_skills
    assert (tmp_path / "skills" / "META-SKILLS" / "local-skill" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_install_skill_supports_zip_archive_source(tmp_path, monkeypatch):
    archive_path = tmp_path / "archived-skill.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "repo-main/archived-skill/SKILL.md",
            (
                "---\n"
                "name: archived-skill\n"
                "description: Archived skill\n"
                "---\n"
                "# Archived Skill\n"
            ),
        )
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package=str(archive_path),
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "local"
    assert result["skill"] == "archived-skill"
    assert "archived-skill" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_supports_tar_archive_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "tar-src" / "tar-skill"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: tar-skill\n"
            "description: Tar skill\n"
            "---\n"
            "# Tar Skill\n"
        ),
        encoding="utf-8",
    )
    archive_path = tmp_path / "tar-skill.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir, arcname="bundle/tar-skill")
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package=str(archive_path),
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "local"
    assert result["skill"] == "tar-skill"
    assert "tar-skill" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_supports_markdown_file_source(tmp_path, monkeypatch):
    source_file = tmp_path / "single-skill.md"
    source_file.write_text(
        (
            "---\n"
            "name: single-skill\n"
            "description: Single file skill\n"
            "---\n"
            "# Single Skill\n"
        ),
        encoding="utf-8",
    )
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package=str(source_file),
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "local"
    assert result["skill"] == "single-skill"
    assert (tmp_path / "skills" / "single-skill" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_install_skill_supports_url_archive_source(tmp_path, monkeypatch):
    archive_path = tmp_path / "remote-skill.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "remote-skill/SKILL.md",
            (
                "---\n"
                "name: remote-skill\n"
                "description: Remote archive skill\n"
                "---\n"
                "# Remote Skill\n"
            ),
        )
    plugin = _make_plugin(tmp_path, monkeypatch)

    def fake_download(url, temp_dir):
        assert url == "https://example.test/remote-skill.zip"
        target = temp_dir / "remote-skill.zip"
        shutil.copy2(archive_path, target)
        return target, None

    plugin._download_url_to_path = fake_download

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://example.test/remote-skill.zip",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "url"
    assert result["skill"] == "remote-skill"
    assert "remote-skill" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_supports_github_repo_url_install_all(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / "research").mkdir(parents=True)
    (repo_dir / "META-SKILLS" / "belief-check").mkdir(parents=True)
    (repo_dir / "research" / "SKILL.md").write_text(
        (
            "---\n"
            "name: research\n"
            "description: Research skill\n"
            "---\n"
            "# Research\n"
        ),
        encoding="utf-8",
    )
    (repo_dir / "META-SKILLS" / "belief-check" / "SKILL.md").write_text(
        (
            "---\n"
            "name: belief-check\n"
            "description: Belief check skill\n"
            "---\n"
            "# Belief Check\n"
        ),
        encoding="utf-8",
    )
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else real_which(name),
    )

    def fake_clone(source, temp_dir):
        assert source == "https://github.com/LKosoj/skills"
        target = temp_dir / "git-source"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source = fake_clone

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/LKosoj/skills",
        install_all=True,
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "git"
    assert result["count"] == 2
    assert {entry["skill"] for entry in result["installed"]} == {
        "META-SKILLS/belief-check",
        "research",
    }
    assert "META-SKILLS/belief-check" in plugin.available_skills
    assert "research" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_github_repo_url_falls_back_to_archive_without_git(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / "fallback").mkdir(parents=True)
    (repo_dir / "fallback" / "SKILL.md").write_text(
        (
            "---\n"
            "name: fallback\n"
            "description: Fallback skill\n"
            "---\n"
            "# Fallback\n"
        ),
        encoding="utf-8",
    )
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: None if name == "git" else real_which(name),
    )

    def fail_clone(source, temp_dir):
        raise AssertionError("git clone should not be used when git is unavailable")

    def fake_download_github(source, temp_dir):
        assert source == "https://github.com/LKosoj/skills"
        target = temp_dir / "github-archive"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source = fail_clone
    plugin._download_github_repo_source = fake_download_github

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/LKosoj/skills",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["source_kind"] == "git"
    assert result["skill"] == "fallback"
    assert "fallback" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_lists_candidates_for_multi_skill_repo_without_install_all(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / "one").mkdir(parents=True)
    (repo_dir / "two").mkdir(parents=True)
    (repo_dir / "one" / "SKILL.md").write_text("# One\n", encoding="utf-8")
    (repo_dir / "two" / "SKILL.md").write_text("# Two\n", encoding="utf-8")
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else real_which(name),
    )

    def fake_clone(source, temp_dir):
        target = temp_dir / "git-source"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source = fake_clone

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/LKosoj/skills",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert result["source_kind"] == "git"
    assert "multiple skills" in result["error"]
    assert result["available_skills"] == [
        {"skill_name": "one", "path": "one"},
        {"skill_name": "two", "path": "two"},
    ]


def test_github_repo_info_parses_tree_url_with_subpath():
    plugin = SkillsPlugin()
    info = plugin._github_repo_info(
        "https://github.com/wgwtest/novel-writing/tree/main/novel-writing"
    )
    assert info == ("wgwtest", "novel-writing", "main", "novel-writing")


def test_github_repo_info_parses_tree_url_with_nested_subpath():
    plugin = SkillsPlugin()
    info = plugin._github_repo_info(
        "https://github.com/owner/repo/tree/main/path/to/skill"
    )
    assert info == ("owner", "repo", "main", "path/to/skill")


def test_is_git_repo_url_accepts_tree_subpath_url():
    plugin = SkillsPlugin()
    assert plugin._is_git_repo_url(
        "https://github.com/wgwtest/novel-writing/tree/main/novel-writing"
    ) is True
    # blob/raw/archive/releases should still be rejected
    assert plugin._is_git_repo_url(
        "https://github.com/owner/repo/blob/main/README.md"
    ) is False


def test_github_blob_info_parses_skill_md_url():
    plugin = SkillsPlugin()
    info = plugin._github_blob_info(
        "https://github.com/wgwtest/novel-writing/blob/main/novel-writing/SKILL.md"
    )
    assert info == ("wgwtest", "novel-writing", "main", ["novel-writing", "SKILL.md"])


def test_github_blob_info_returns_none_for_non_blob_url():
    plugin = SkillsPlugin()
    assert plugin._github_blob_info("https://github.com/owner/repo") is None
    assert (
        plugin._github_blob_info("https://github.com/owner/repo/tree/main/dir") is None
    )


def test_source_kind_for_blob_skill_md_is_git():
    plugin = SkillsPlugin()
    assert (
        plugin._skill_install_source_kind(
            "https://github.com/wgwtest/novel-writing/blob/main/novel-writing/SKILL.md"
        )
        == "git"
    )
    assert (
        plugin._skill_install_source_kind(
            "https://github.com/wgwtest/novel-writing/blob/main/CONTRIBUTING.md"
        )
        == "url"
    )


def test_rewrite_github_blob_to_raw_for_markdown():
    plugin = SkillsPlugin()
    info = plugin._github_blob_info(
        "https://github.com/owner/repo/blob/main/docs/skill.md"
    )
    rewritten, err = plugin._rewrite_github_blob_to_raw(info)
    assert err is None
    assert rewritten == "https://raw.githubusercontent.com/owner/repo/main/docs/skill.md"


def test_rewrite_github_blob_to_raw_rejects_non_markdown():
    plugin = SkillsPlugin()
    info = plugin._github_blob_info("https://github.com/owner/repo/blob/main/LICENSE")
    rewritten, err = plugin._rewrite_github_blob_to_raw(info)
    assert rewritten is None
    assert "SKILL.md" in err


@pytest.mark.asyncio
async def test_install_skill_supports_github_blob_url_pointing_at_skill_md(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    (repo_dir / "novel-writing" / "agents").mkdir(parents=True)
    (repo_dir / "novel-writing" / "SKILL.md").write_text(
        "---\nname: novel-writing\ndescription: Fiction\n---\n# Novel\n",
        encoding="utf-8",
    )
    (repo_dir / "novel-writing" / "agents" / "openai.yaml").write_text(
        "interface:\n  display_name: Novel\n", encoding="utf-8"
    )
    (repo_dir / "README.md").write_text("# Repo README\n", encoding="utf-8")
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else real_which(name),
    )

    captured = {}

    def fake_clone_branch(source, temp_dir, branch):
        captured["source"] = source
        captured["branch"] = branch
        target = temp_dir / "git-source"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source_branch = fake_clone_branch

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/wgwtest/novel-writing/blob/main/novel-writing/SKILL.md",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True, result
    assert result["source_kind"] == "git"
    assert result["skill"] == "novel-writing"
    assert captured["branch"] == "main"
    assert captured["source"] == "https://github.com/wgwtest/novel-writing"
    installed = tmp_path / "skills" / "novel-writing"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "agents" / "openai.yaml").is_file()
    assert not (installed / "README.md").exists()


@pytest.mark.asyncio
async def test_install_skill_supports_github_blob_url_pointing_at_markdown_file(
    tmp_path, monkeypatch
):
    plugin = _make_plugin(tmp_path, monkeypatch)
    captured = {}

    def fake_download(url, temp_dir):
        captured["url"] = url
        target = temp_dir / "helper.md"
        target.write_text(
            "---\nname: helper\ndescription: One-off helper\n---\n# Helper\n",
            encoding="utf-8",
        )
        return target, None

    plugin._download_url_to_path = fake_download

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/owner/repo/blob/main/docs/helper.md",
        skill_name="helper",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True, result
    assert result["source_kind"] == "url"
    assert captured["url"] == (
        "https://raw.githubusercontent.com/owner/repo/main/docs/helper.md"
    )
    assert (tmp_path / "skills" / "helper" / "SKILL.md").is_file()


@pytest.mark.asyncio
async def test_install_skill_rejects_github_blob_url_for_non_markdown(
    tmp_path, monkeypatch
):
    plugin = _make_plugin(tmp_path, monkeypatch)

    def fail_download(url, temp_dir):
        raise AssertionError("must reject before download")

    plugin._download_url_to_path = fail_download

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/owner/repo/blob/main/LICENSE",
        skill_name="bad",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "SKILL.md" in result["error"]


@pytest.mark.asyncio
async def test_install_skill_rejects_html_disguised_as_markdown(
    tmp_path, monkeypatch
):
    plugin = _make_plugin(tmp_path, monkeypatch)

    def fake_download(url, temp_dir):
        target = temp_dir / "page.md"
        target.write_text(
            "<!DOCTYPE html>\n<html><body>not markdown</body></html>",
            encoding="utf-8",
        )
        return target, None

    plugin._download_url_to_path = fake_download

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://example.com/page.md",
        skill_name="page",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "HTML" in result["error"]


@pytest.mark.asyncio
async def test_install_skill_supports_github_tree_url_with_subpath(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / "novel-writing" / "agents").mkdir(parents=True)
    (repo_dir / "novel-writing" / "SKILL.md").write_text(
        "---\nname: novel-writing\ndescription: Fiction\n---\n# Novel\n",
        encoding="utf-8",
    )
    (repo_dir / "novel-writing" / "agents" / "openai.yaml").write_text(
        "interface:\n  display_name: Novel\n", encoding="utf-8"
    )
    (repo_dir / "README.md").write_text("# Repo README\n", encoding="utf-8")
    (repo_dir / "other" / "stuff.txt").parent.mkdir(parents=True)
    (repo_dir / "other" / "stuff.txt").write_text("noise", encoding="utf-8")
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else real_which(name),
    )

    captured = {}

    def fake_clone_branch(source, temp_dir, branch):
        captured["source"] = source
        captured["branch"] = branch
        target = temp_dir / "git-source"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source_branch = fake_clone_branch

    def reject_plain_clone(source, temp_dir):
        raise AssertionError("plain clone should not be used for tree-URL with branch")

    plugin._clone_git_source = reject_plain_clone

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/wgwtest/novel-writing/tree/main/novel-writing",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True, result
    assert result["source_kind"] == "git"
    assert result["skill"] == "novel-writing"
    assert captured["branch"] == "main"
    assert captured["source"] == "https://github.com/wgwtest/novel-writing"
    installed = tmp_path / "skills" / "novel-writing"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "agents" / "openai.yaml").is_file()
    # subpath isolation: files outside novel-writing/ must not leak in
    assert not (installed / "README.md").exists()
    assert not (installed / "other").exists()
    assert "novel-writing" in plugin.available_skills


@pytest.mark.asyncio
async def test_install_skill_github_tree_url_rejects_missing_subpath(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / "novel-writing").mkdir(parents=True)
    (repo_dir / "novel-writing" / "SKILL.md").write_text("# X\n", encoding="utf-8")
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else real_which(name),
    )

    def fake_clone_branch(source, temp_dir, branch):
        target = temp_dir / "git-source"
        shutil.copytree(repo_dir, target)
        return target, None

    plugin._clone_git_source_branch = fake_clone_branch

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/wgwtest/novel-writing/tree/main/does-not-exist",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "does-not-exist" in result["error"]


@pytest.mark.asyncio
async def test_install_skill_github_tree_url_falls_back_to_archive_without_git(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    (repo_dir / "novel-writing").mkdir(parents=True)
    (repo_dir / "novel-writing" / "SKILL.md").write_text(
        "---\nname: novel-writing\ndescription: Fiction\n---\n# Novel\n",
        encoding="utf-8",
    )
    (repo_dir / "other.md").write_text("noise", encoding="utf-8")
    plugin = _make_plugin(tmp_path, monkeypatch)
    real_which = shutil.which
    monkeypatch.setattr(
        "bot.plugins.skills.shutil.which",
        lambda name: None if name == "git" else real_which(name),
    )

    captured = {}

    def fake_download(source, temp_dir):
        captured["source"] = source
        target = temp_dir / "archive-root"
        shutil.copytree(repo_dir, target)
        return target, None

    # Patch _materialize_local_skill_source indirectly via _download_github_repo_source:
    # we still want the subpath logic inside _download_github_repo_source to run.
    plugin._download_url_to_path = lambda url, temp_dir: (None, "skip-real-download")

    def fake_download_github(source, temp_dir):
        # mimic full pipeline: materialize archive then enter subpath
        captured["source"] = source
        info = plugin._github_repo_info(source)
        assert info is not None
        _, _, _, subpath = info
        target = temp_dir / "archive-root"
        shutil.copytree(repo_dir, target)
        if subpath:
            return plugin._enter_repo_subpath(target, subpath)
        return target, None

    plugin._download_github_repo_source = fake_download_github

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package="https://github.com/wgwtest/novel-writing/tree/main/novel-writing",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True, result
    assert result["skill"] == "novel-writing"
    installed = tmp_path / "skills" / "novel-writing"
    assert (installed / "SKILL.md").is_file()
    assert not (installed / "other.md").exists()


@pytest.mark.asyncio
async def test_install_skill_rejects_archive_path_traversal(tmp_path, monkeypatch):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape/SKILL.md", "# Escape\n")
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "install_skill",
        helper=None,
        package=str(archive_path),
        skill_name="bad",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "escapes target directory" in result["error"]


@pytest.mark.asyncio
async def test_create_skill_creates_skill_markdown_and_refreshes_registry(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "create_skill",
        helper=None,
        skill_name="custom/reviewer",
        name="Reviewer",
        description="Review generated text",
        instructions="# Reviewer\n\nCheck factual claims.\n",
        confirmed=True,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is True
    assert result["skill"] == "custom/reviewer"
    assert "custom/reviewer" in plugin.available_skills
    skill_md = tmp_path / "skills" / "custom" / "reviewer" / "SKILL.md"
    assert skill_md.exists()
    details = await plugin.execute(
        "get_skill",
        helper=None,
        skill_name="custom/reviewer",
        chat_id=10,
        user_id=999,
    )
    assert details["skill"]["name"] == "Reviewer"
    assert "Check factual claims" in details["skill"]["body"]


@pytest.mark.asyncio
async def test_create_skill_requires_confirmation(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)

    result = await plugin.execute(
        "create_skill",
        helper=None,
        skill_name="custom/reviewer",
        instructions="# Reviewer\n",
        confirmed=False,
        chat_id=10,
        user_id=999,
    )

    assert result["success"] is False
    assert "confirmed" in result["error"]
    assert "custom/reviewer" not in plugin.available_skills


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


# ----------------- on_before_chat_request: skills catalog injection -----------------


from bot.plugins.hooks import BeforeChatRequestPayload


def _skills_agent_system_message(content: str = "skills_agent prompt") -> dict:
    return {"role": "system", "content": content, "mode_key": "skills_agent"}


@pytest.mark.asyncio
async def test_on_before_chat_request_injects_catalog_in_skills_agent(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    messages = [
        _skills_agent_system_message(),
        {"role": "user", "content": "do something with PPTX"},
    ]
    payload = BeforeChatRequestPayload(chat_id=10, user_id=42, request_id=None)

    new_messages = await plugin.on_before_chat_request(messages, payload)

    assert new_messages is not None
    assert len(new_messages) == len(messages) + 1
    injected = new_messages[1]
    assert injected["role"] == "system"
    assert "Доступные локальные skills" in injected["content"]
    assert "- demo: Demo skill for tests" in injected["content"]
    # No active skills in this scope → no scripts section.
    assert "Активные skills в этой сессии" not in injected["content"]


@pytest.mark.asyncio
async def test_on_before_chat_request_skips_other_modes(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    messages = [
        {"role": "system", "content": "assistant prompt", "mode_key": "assistant"},
        {"role": "user", "content": "hi"},
    ]
    payload = BeforeChatRequestPayload(chat_id=10, user_id=42, request_id=None)

    assert await plugin.on_before_chat_request(messages, payload) is None


@pytest.mark.asyncio
async def test_on_before_chat_request_lists_scripts_for_active_skills(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    await plugin.execute("activate_skill", helper=None, skill_name="demo", chat_id=10, user_id=42)

    messages = [_skills_agent_system_message(), {"role": "user", "content": "go"}]
    payload = BeforeChatRequestPayload(chat_id=10, user_id=42, request_id=None)
    new_messages = await plugin.on_before_chat_request(messages, payload)

    assert new_messages is not None
    content = new_messages[1]["content"]
    assert "Активные skills в этой сессии" in content
    assert "demo:" in content
    expected_abs = str((tmp_path / "skills" / "demo" / "scripts" / "echo.py").resolve())
    # as_posix() output is what we render
    assert (tmp_path / "skills" / "demo" / "scripts" / "echo.py").as_posix() in content
    # Active-skill scripts section is irrelevant to other scopes
    other_payload = BeforeChatRequestPayload(chat_id=999, user_id=999, request_id=None)
    other_messages = await plugin.on_before_chat_request(messages, other_payload)
    assert other_messages is not None
    assert "Активные skills в этой сессии" not in other_messages[1]["content"]


@pytest.mark.asyncio
async def test_on_before_chat_request_no_system_message_returns_none(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    payload = BeforeChatRequestPayload(chat_id=10, user_id=42, request_id=None)
    assert await plugin.on_before_chat_request([{"role": "user", "content": "hi"}], payload) is None
    assert await plugin.on_before_chat_request([], payload) is None


@pytest.mark.asyncio
async def test_on_before_chat_request_does_not_mutate_input(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path, monkeypatch)
    messages = [_skills_agent_system_message(), {"role": "user", "content": "go"}]
    original_len = len(messages)
    payload = BeforeChatRequestPayload(chat_id=10, user_id=42, request_id=None)

    new_messages = await plugin.on_before_chat_request(messages, payload)

    assert new_messages is not None
    assert new_messages is not messages
    assert len(messages) == original_len  # input list untouched
