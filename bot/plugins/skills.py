from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..utils import compute_scope_key
from ..user_settings import USER_DISABLED_SKILLS_SETTING, get_user_settings, normalize_string_list
from .plugin import Plugin


logger = logging.getLogger(__name__)

TRUE_VALUES = {"1", "true", "yes", "on"}
SCRIPT_SUFFIXES = {".py", ".js", ".mjs", ".cjs", ".sh"}
AGENT_SUFFIXES = {".yaml", ".yml"}
REFLECTION_REPEAT_THRESHOLD = 3
REFLECTION_MAX_TEXT_CHARS = 1200
SKILLS_CLI_TIMEOUT_SECONDS = 180
SKILLS_CLI_AGENT = "codex"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MARKDOWN_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:\.\./|[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.md)"
    r"(?![A-Za-z0-9_./-])"
)


class SkillsPlugin(Plugin):
    """
    Exposes local Codex-style skills as tools for the existing function-calling loop.
    """

    plugin_id = "skills"
    function_prefix = "skills"

    def __init__(self):
        self.skills_dir: Path | None = None
        self.state_file: Path | None = None
        self.reflections_file: Path | None = None
        self.workdir_root: Path | None = None
        self.audit_file: Path | None = None
        self.available_skills: Dict[str, Dict[str, Any]] = {}
        self.active_skills: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.skill_reflections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.allow_scripts = False
        self.allow_installs = False
        self.script_admin_ids: set[int] = set()
        self.script_admin_all = False
        self.install_admin_ids: set[int] = set()
        self.install_admin_all = False
        self.script_timeout = 120
        self.install_timeout = SKILLS_CLI_TIMEOUT_SECONDS
        self.output_max_chars = 12000
        self.interim_after_seconds = 20
        self._state_lock = threading.RLock()
        self._reflection_lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._scan_cache: tuple[tuple, Dict[str, Dict[str, Any]]] | None = None

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        storage_path = Path(storage_root).resolve() if storage_root else Path.cwd().resolve()
        skills_dir = os.getenv("SKILLS_DIR")
        self.skills_dir = Path(skills_dir).expanduser().resolve() if skills_dir else storage_path / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = storage_path / "skills_state.json"
        self.reflections_file = storage_path / "skill_reflections.json"
        workdir_env = os.getenv("SKILLS_WORKDIR")
        self.workdir_root = (
            Path(workdir_env).expanduser().resolve()
            if workdir_env
            else storage_path / "skill_workdir"
        )
        self.workdir_root.mkdir(parents=True, exist_ok=True)
        self.audit_file = storage_path / "skills_audit.jsonl"
        self.allow_scripts = self._env_flag("SKILLS_ALLOW_SCRIPTS", default=False)
        self.script_timeout = self._env_int("SKILLS_SCRIPT_TIMEOUT", default=120, minimum=1)
        self.install_timeout = self._env_int(
            "SKILLS_INSTALL_TIMEOUT",
            default=SKILLS_CLI_TIMEOUT_SECONDS,
            minimum=10,
        )
        self.output_max_chars = self._env_int("SKILLS_SCRIPT_OUTPUT_MAX_CHARS", default=12000, minimum=1000)
        self.interim_after_seconds = self._env_int(
            "SKILLS_SCRIPT_INTERIM_AFTER_SECONDS", default=20, minimum=5
        )
        self.script_admin_ids, self.script_admin_all = self._parse_admin_ids(
            os.getenv("SKILLS_SCRIPT_ADMIN_USER_IDS", ""),
            env_name="SKILLS_SCRIPT_ADMIN_USER_IDS",
        )
        self.allow_installs = self._env_flag("SKILLS_ALLOW_INSTALLS", default=True)
        self.install_admin_ids, self.install_admin_all = self._parse_admin_ids(
            os.getenv("SKILLS_INSTALL_ADMIN_USER_IDS", "*"),
            env_name="SKILLS_INSTALL_ADMIN_USER_IDS",
        )

        self.available_skills = self._scan_skills()
        self._log_available_skills()
        self._load_state()
        self._load_reflections()

    def get_source_name(self) -> str:
        return "Skills"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "list_skills",
                "description": "List local skills available to the current agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refresh": {
                            "type": "boolean",
                            "description": "Whether to rescan the skills directory before listing. Defaults to true.",
                            "default": True,
                        }
                    },
                },
            },
            {
                "name": "get_skill",
                "description": "Return the full instruction body and metadata for one skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        }
                    },
                    "required": ["skill_name"],
                },
            },
            {
                "name": "get_skill_reference",
                "description": (
                    "Return one markdown reference file used by a skill, such as references/*.md "
                    "or META-SKILLS/_shared/*.md. The path must come from skills.get_skill."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "reference_path": {
                            "type": "string",
                            "description": "Reference path returned by skills.get_skill.",
                        },
                    },
                    "required": ["skill_name", "reference_path"],
                },
            },
            {
                "name": "find_installable_skills",
                "description": (
                    "Search skills.sh through the `skills` CLI for new skills that can be installed. "
                    "Returns package ids like owner/repo@skill for skills.install_skill."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query, for example 'testing', 'pptx', or 'github review'.",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "install_skill",
                "description": (
                    "Install a skill package found by skills.find_installable_skills, then sync it into "
                    "SKILLS_DIR and refresh the local skill registry. Enabled by default unless "
                    "SKILLS_ALLOW_INSTALLS=false. SKILLS_INSTALL_ADMIN_USER_IDS is a user allow-list "
                    "that defaults to '*'. confirmed=true is required after explicit user approval."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "package": {
                            "type": "string",
                            "description": "Install package id, usually owner/repo@skill from skills.find_installable_skills.",
                        },
                        "skill_name": {
                            "type": "string",
                            "description": "Optional target skill directory name. Required when package does not include @skill.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Must be true only after the user explicitly approved this exact package install.",
                        },
                    },
                    "required": ["package", "confirmed"],
                },
            },
            {
                "name": "activate_skill",
                "description": (
                    "Activate a skill for the current Telegram chat/user and store initial task context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "initial_context": {
                            "type": "string",
                            "description": "Task context as plain text or a JSON object string.",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
            {
                "name": "get_skill_status",
                "description": "Show active skill state for the current Telegram chat/user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Optional skill directory id or metadata name.",
                        }
                    },
                },
            },
            {
                "name": "list_active_skills",
                "description": (
                    "List skills currently active for this Telegram chat/user. "
                    "Returns a compact summary without internal context details."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "update_skill_progress",
                "description": "Update active skill progress and merge additional JSON/text context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "step": {
                            "type": "integer",
                            "description": "Current step number from the skill workflow.",
                        },
                        "context_update": {
                            "type": "string",
                            "description": "Progress update as plain text or a JSON object string.",
                        },
                    },
                    "required": ["skill_name", "step"],
                },
            },
            {
                "name": "run_skill_script",
                "description": (
                    "Run an allowed script from an active skill scripts directory. Scripts must be enabled by "
                    "SKILLS_ALLOW_SCRIPTS and the caller must be allowed by SKILLS_SCRIPT_ADMIN_USER_IDS."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "script_name": {
                            "type": "string",
                            "description": "Script path returned by the skill metadata, relative to the skill scripts directory.",
                        },
                        "args_json": {
                            "type": "string",
                            "description": (
                                "Optional JSON array of CLI args, JSON object converted to --key value, "
                                "or plain text passed as one argument."
                            ),
                        },
                    },
                    "required": ["skill_name", "script_name"],
                },
            },
            {
                "name": "run_skill_agent",
                "description": (
                    "Run a tool-capable subagent declared by a skill under agents/*.yaml. "
                    "Use when a skill exposes a specialist agent profile for the task. "
                    "The skill agent is executed through agent_tools.run_subagents, so agent_tools must be enabled."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Agent id from the skill's agents directory, for example openai.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Concrete bounded task for this skill agent.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional task-specific context to pass to the skill agent.",
                        },
                        "max_rounds": {
                            "type": "integer",
                            "description": "Optional subagent tool-call rounds budget.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override accepted by agent_tools.run_subagents.",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Optional subagent temperature.",
                        },
                    },
                    "required": ["skill_name", "agent_name", "task"],
                },
            },
            {
                "name": "record_skill_reflection",
                "description": (
                    "Record a reflection proposal after a skill failure. "
                    "Repeated identical proposals are accumulated; when the repeat count is greater "
                    "than three, the proposal is appended to that skill's SKILL.md as a learned clarification."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        },
                        "proposal": {
                            "type": "string",
                            "description": "One concise instruction that should be added to SKILL.md if it repeats.",
                        },
                        "failure_mode": {
                            "type": "string",
                            "description": "Optional short description of the failure that motivated this proposal.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Optional short evidence from the failed run.",
                        },
                    },
                    "required": ["skill_name", "proposal"],
                },
            },
            {
                "name": "deactivate_skill",
                "description": (
                    "Mark an active skill as completed and free its in-memory state. "
                    "Use when the skill workflow is finished and you do not need its state anymore. "
                    "Final delivery to the user is done separately via agent_tools.deliver_to_user, "
                    "which auto-deactivates any still-active skills, so explicit deactivation is "
                    "only needed mid-workflow."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill directory id or metadata name.",
                        }
                    },
                    "required": ["skill_name"],
                },
            },
        ]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        self._ensure_ready()
        request_context = kwargs.pop("request_context", None)
        user_id = kwargs.get("user_id")
        disabled_skills = self._disabled_skills_for_user(helper, user_id)
        if function_name == "list_skills":
            return self._list_skills(
                refresh=self._bool_arg(kwargs.get("refresh"), default=True),
                disabled_skills=disabled_skills,
            )
        if function_name == "get_skill":
            skill_name = str(kwargs.get("skill_name") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            return self._get_skill(skill_name)
        if function_name == "get_skill_reference":
            skill_name = str(kwargs.get("skill_name") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            return self._get_skill_reference(skill_name, str(kwargs.get("reference_path") or ""))
        if function_name == "find_installable_skills":
            return await self._find_installable_skills(str(kwargs.get("query") or ""))
        if function_name == "install_skill":
            call_kwargs = dict(kwargs)
            package = str(call_kwargs.pop("package", "") or "")
            skill_name = call_kwargs.pop("skill_name", None)
            confirmed = self._bool_arg(call_kwargs.pop("confirmed", False), default=False)
            return await self._install_skill(
                package,
                skill_name=str(skill_name or "") if skill_name is not None else None,
                confirmed=confirmed,
                **call_kwargs,
            )
        if function_name == "activate_skill":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            initial_context = call_kwargs.pop("initial_context", None)
            return self._activate_skill(
                skill_name,
                initial_context,
                **call_kwargs,
            )
        if function_name == "get_skill_status":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills) if skill_name else None
            if disabled_error:
                return disabled_error
            return self._get_skill_status(skill_name, disabled_skills=disabled_skills, **call_kwargs)
        if function_name == "list_active_skills":
            return self._list_active_skills(disabled_skills=disabled_skills, **kwargs)
        if function_name == "update_skill_progress":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            step = int(call_kwargs.pop("step", 0) or 0)
            context_update = call_kwargs.pop("context_update", None)
            return self._update_skill_progress(
                skill_name,
                step,
                context_update,
                **call_kwargs,
            )
        if function_name == "deactivate_skill":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            return self._deactivate_skill(skill_name, **call_kwargs)
        if function_name == "run_skill_script":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            script_name = str(call_kwargs.pop("script_name", "") or "")
            args_json = call_kwargs.pop("args_json", None)
            return await self._run_skill_script(
                skill_name,
                script_name,
                args_json,
                **call_kwargs,
            )
        if function_name == "run_skill_agent":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            disabled_error = self._disabled_skill_error(skill_name, disabled_skills)
            if disabled_error:
                return disabled_error
            agent_name = str(call_kwargs.pop("agent_name", "") or "")
            task = str(call_kwargs.pop("task", "") or "")
            context = call_kwargs.pop("context", None)
            return await self._run_skill_agent(
                helper,
                skill_name,
                agent_name,
                task,
                context=context,
                request_context=request_context,
                **call_kwargs,
            )
        if function_name == "record_skill_reflection":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            proposal = str(call_kwargs.pop("proposal", "") or "")
            failure_mode = call_kwargs.pop("failure_mode", None)
            evidence = call_kwargs.pop("evidence", None)
            return self._record_skill_reflection(
                skill_name,
                proposal,
                failure_mode=failure_mode,
                evidence=evidence,
                **call_kwargs,
            )
        return {"success": False, "error": f"Unknown skills tool: {function_name}"}

    def _ensure_ready(self) -> None:
        if self.skills_dir is None or self.state_file is None:
            self.initialize(
                openai=getattr(self, "openai", None),
                bot=getattr(self, "bot", None),
                storage_root=getattr(self, "storage_root", None),
            )

    def _disabled_skills_for_user(self, helper, user_id: int | None) -> set[str]:
        if user_id is None or helper is None or not getattr(helper, "db", None):
            return set()
        settings = get_user_settings(helper.db, user_id)
        return set(normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING)))

    def _disabled_skill_error(self, skill_name: str, disabled_skills: set[str]) -> Dict[str, Any] | None:
        skill_id = self._resolve_skill_id(skill_name)
        if skill_id and skill_id in disabled_skills:
            return {
                "success": False,
                "error": f"Skill '{skill_id}' is disabled in your user settings.",
            }
        return None

    def _scan_skills(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_paths()
        signature = self._scan_signature()
        if self._scan_cache is not None and self._scan_cache[0] == signature:
            return self._scan_cache[1]

        skills: Dict[str, Dict[str, Any]] = {}
        for skill_path in self._iter_skill_paths():
            md_file = skill_path / "SKILL.md"

            try:
                content = md_file.read_text(encoding="utf-8")
                metadata, body = self._parse_skill_markdown(content)
                scripts = self._list_skill_scripts(skill_path, metadata)
                agents = self._list_skill_agents(skill_path)
                references = self._list_skill_references(skill_path, body)
                skill_id = self._skill_id_for_path(skill_path)
                skills[skill_id] = {
                    "id": skill_id,
                    "name": str(metadata.get("name") or skill_id),
                    "description": str(metadata.get("description") or ""),
                    "metadata": metadata,
                    "body": body,
                    "scripts": scripts,
                    "agents": agents,
                    "references": references,
                    "path": str(skill_path),
                }
            except Exception as exc:
                logger.warning("Failed to parse skill %s: %s", md_file, exc)
        self._scan_cache = (signature, skills)
        return skills

    def _scan_signature(self) -> tuple:
        if self.skills_dir is None or not self.skills_dir.exists():
            return ()
        entries: List[tuple] = []
        markdown_entries: List[tuple] = []
        try:
            for skill_path in self._iter_skill_paths():
                md_file = skill_path / "SKILL.md"
                try:
                    md_mtime = md_file.stat().st_mtime_ns
                except OSError:
                    continue
                scripts_dir = skill_path / "scripts"
                script_entries: List[tuple] = []
                if scripts_dir.exists() and scripts_dir.is_dir():
                    for path in sorted(scripts_dir.rglob("*")):
                        try:
                            if not path.is_file() or not self._is_script_entrypoint(path, scripts_dir):
                                continue
                        except OSError:
                            continue
                        try:
                            script_entries.append((str(path.relative_to(scripts_dir)), path.stat().st_mtime_ns))
                        except OSError:
                            continue
                agents_dir = skill_path / "agents"
                agent_entries: List[tuple] = []
                if agents_dir.exists() and agents_dir.is_dir():
                    for path in sorted(agents_dir.rglob("*")):
                        try:
                            if not path.is_file() or path.suffix.lower() not in AGENT_SUFFIXES:
                                continue
                        except OSError:
                            continue
                        try:
                            agent_entries.append((str(path.relative_to(agents_dir)), path.stat().st_mtime_ns))
                        except OSError:
                            continue
                entries.append((
                    self._skill_id_for_path(skill_path),
                    md_mtime,
                    tuple(script_entries),
                    tuple(agent_entries),
                ))
            for path in sorted(self.skills_dir.rglob("*.md")):
                try:
                    if not path.is_file():
                        continue
                    relative_path = path.relative_to(self.skills_dir)
                    if any(part.startswith(".") or part == "__pycache__" for part in relative_path.parts):
                        continue
                    markdown_entries.append((relative_path.as_posix(), path.stat().st_mtime_ns))
                except OSError:
                    continue
        except OSError:
            return ()
        return tuple(entries), tuple(markdown_entries)

    def _iter_skill_paths(self) -> List[Path]:
        if self.skills_dir is None or not self.skills_dir.exists():
            return []
        paths: List[Path] = []
        for md_file in self.skills_dir.rglob("SKILL.md"):
            skill_path = md_file.parent
            try:
                relative = skill_path.relative_to(self.skills_dir)
            except ValueError:
                continue
            if not relative.parts:
                continue
            if any(
                part.startswith(".") or part in {"__pycache__", "_shared", "scripts", "agents"}
                for part in relative.parts
            ):
                continue
            paths.append(skill_path)
        return sorted(
            paths,
            key=lambda path: (
                len(path.relative_to(self.skills_dir).parts),
                path.relative_to(self.skills_dir).as_posix().lower(),
            ),
        )

    def _skill_id_for_path(self, skill_path: Path) -> str:
        if self.skills_dir is None:
            return skill_path.name
        try:
            return skill_path.relative_to(self.skills_dir).as_posix()
        except ValueError:
            return skill_path.name

    def _parse_skill_markdown(self, content: str) -> tuple[Dict[str, Any], str]:
        if content.startswith("﻿"):
            content = content[1:]
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3 and not parts[0].strip():
                frontmatter = parts[1].strip()
                try:
                    metadata = yaml.safe_load(frontmatter) or {}
                except yaml.YAMLError:
                    metadata = self._parse_flat_frontmatter(frontmatter)
                if not isinstance(metadata, dict):
                    metadata = {}
                return metadata, parts[2].strip()
        return {}, content.strip()

    def _parse_flat_frontmatter(self, frontmatter: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        current_key = None
        for raw_line in frontmatter.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if raw_line[0].isspace() and current_key:
                metadata[current_key] = f"{metadata[current_key]} {line}".strip()
                continue

            key, separator, value = raw_line.partition(":")
            if not separator:
                continue
            key = key.strip()
            if not key:
                continue
            metadata[key] = self._parse_flat_scalar(value.strip())
            current_key = key
        return metadata

    def _parse_flat_scalar(self, value: str) -> Any:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        lowered = value.lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return value

    def _list_skill_scripts(
        self, skill_path: Path, metadata: Dict[str, Any] | None = None
    ) -> List[str]:
        scripts_dir = skill_path / "scripts"
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            return []

        explicit = self._resolve_explicit_entrypoints(metadata or {}, scripts_dir)
        if explicit is not None:
            return explicit

        scripts: List[str] = []
        for path in scripts_dir.rglob("*"):
            if not path.is_file():
                continue
            if not self._is_script_entrypoint(path, scripts_dir):
                continue
            scripts.append(path.relative_to(scripts_dir).as_posix())
        return sorted(scripts)

    def _list_skill_agents(self, skill_path: Path) -> List[Dict[str, Any]]:
        agents_dir = skill_path / "agents"
        if not agents_dir.exists() or not agents_dir.is_dir():
            return []

        agents: List[Dict[str, Any]] = []
        try:
            paths = sorted(agents_dir.iterdir(), key=lambda path: path.name)
        except OSError:
            return []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in AGENT_SUFFIXES:
                continue
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Failed to parse skill agent %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                payload = {}
            interface = payload.get("interface")
            if not isinstance(interface, dict):
                interface = {}
            agent_id = path.stem
            display_name = str(
                interface.get("display_name")
                or payload.get("display_name")
                or payload.get("name")
                or agent_id
            )
            agents.append({
                "id": agent_id,
                "file": path.relative_to(skill_path).as_posix(),
                "display_name": display_name,
                "short_description": str(
                    interface.get("short_description")
                    or payload.get("short_description")
                    or payload.get("description")
                    or ""
                ),
                "default_prompt": str(
                    interface.get("default_prompt")
                    or payload.get("default_prompt")
                    or payload.get("prompt")
                    or ""
                ),
            })
        return agents

    def _list_skill_references(self, skill_path: Path, body: str) -> List[Dict[str, str]]:
        references: List[Dict[str, str]] = []
        seen: set[str] = set()
        for match in MARKDOWN_REFERENCE_RE.finditer(body or ""):
            reference_path = match.group(1)
            path, relative_path, error = self._resolve_skill_reference_path(skill_path, reference_path)
            if error or path is None or relative_path is None or relative_path in seen:
                continue
            seen.add(relative_path)
            references.append({
                "reference_path": reference_path,
                "path": relative_path,
            })
        return references

    def _resolve_skill_reference_path(
        self,
        skill_path: Path,
        reference_path: str,
    ) -> tuple[Path | None, str | None, str | None]:
        self._ensure_paths()
        normalized = str(reference_path or "").strip().strip("`")
        if not normalized:
            return None, None, "reference_path must be non-empty"
        raw_path = Path(normalized)
        if raw_path.is_absolute():
            return None, None, "reference_path must be relative"
        if raw_path.suffix.lower() != ".md":
            return None, None, "reference_path must point to a markdown file"

        try:
            skills_dir = self.skills_dir.resolve() if self.skills_dir is not None else skill_path.resolve()
        except OSError:
            return None, None, f"Reference '{reference_path}' not found"

        base_paths = [skill_path]
        if self.skills_dir is not None:
            base_paths.append(self.skills_dir)
        for base_path in base_paths:
            try:
                candidate = (base_path / raw_path).resolve()
            except OSError:
                continue
            if not self._is_relative_to(candidate, skills_dir):
                continue
            if candidate.is_file():
                return candidate, candidate.relative_to(skills_dir).as_posix(), None
        return None, None, f"Reference '{reference_path}' not found"

    def _resolve_explicit_entrypoints(
        self, metadata: Dict[str, Any], scripts_dir: Path
    ) -> List[str] | None:
        """
        If SKILL.md frontmatter declares `entrypoints`, treat it as the
        authoritative list of runnable scripts. Returns ``None`` when no
        explicit declaration is present, so the caller falls back to the
        suffix/executable heuristic.
        """
        if "entrypoints" not in metadata:
            return None

        raw = metadata["entrypoints"]
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            logger.warning(
                "Skill %s has invalid 'entrypoints' (expected list, got %s); "
                "falling back to heuristic",
                scripts_dir.parent.name,
                type(raw).__name__,
            )
            return None

        try:
            scripts_dir_resolved = scripts_dir.resolve()
        except OSError:
            return []

        resolved: List[str] = []
        seen: set[str] = set()
        for entry in raw:
            if not isinstance(entry, str):
                logger.warning(
                    "Skill %s entrypoint must be a string, skipping %r",
                    scripts_dir.parent.name, entry,
                )
                continue
            normalized = entry.strip().lstrip("/")
            if not normalized or ".." in Path(normalized).parts:
                logger.warning(
                    "Skill %s entrypoint %r is invalid (empty or traversal), skipping",
                    scripts_dir.parent.name, entry,
                )
                continue
            candidate = (scripts_dir / normalized).resolve()
            if not self._is_relative_to(candidate, scripts_dir_resolved):
                logger.warning(
                    "Skill %s entrypoint %r escapes scripts/, skipping",
                    scripts_dir.parent.name, entry,
                )
                continue
            if not candidate.is_file():
                logger.warning(
                    "Skill %s entrypoint %r does not exist under scripts/, skipping",
                    scripts_dir.parent.name, entry,
                )
                continue
            posix = candidate.relative_to(scripts_dir_resolved).as_posix()
            if posix in seen:
                continue
            seen.add(posix)
            resolved.append(posix)
        return resolved

    def _is_script_entrypoint(self, path: Path, scripts_dir: Path) -> bool:
        relative_path = path.relative_to(scripts_dir)
        if "__pycache__" in relative_path.parts:
            return False
        if any(part.startswith(".") for part in relative_path.parts):
            return False
        if path.name == "__init__.py" or path.suffix == ".pyc":
            return False
        if path.suffix.lower() in SCRIPT_SUFFIXES:
            return True
        return os.access(path, os.X_OK)

    def _list_skills(self, *, refresh: bool = False, disabled_skills: set[str] | None = None) -> Dict[str, Any]:
        if refresh:
            self.available_skills = self._scan_skills()
        disabled_skills = disabled_skills or set()
        return {
            "success": True,
            "skills": [
                {
                    "id": skill_id,
                    "name": info["name"],
                    "description": info["description"],
                    "scripts": info["scripts"],
                    "agents": [agent["id"] for agent in info.get("agents", [])],
                    "references": info.get("references", []),
                    "allow_scripts": self._skill_scripts_allowed(info),
                }
                for skill_id, info in self.available_skills.items()
                if skill_id not in disabled_skills
            ],
            "scripts_enabled": self.allow_scripts,
            "scripts_admin_restricted": not self.script_admin_all,
        }

    async def _find_installable_skills(self, query: str) -> Dict[str, Any]:
        search_query = " ".join((query or "").split())
        if not search_query:
            return {"success": False, "error": "query must be a non-empty string"}

        result = await self._run_skills_cli(
            ["find", search_query],
            timeout=self.install_timeout,
        )
        stdout = self._strip_ansi(result.get("stdout", ""))
        stderr = self._strip_ansi(result.get("stderr", ""))
        return {
            "success": bool(result.get("success")),
            "query": search_query,
            "results": self._parse_installable_skill_results(stdout),
            "stdout": self._truncate(stdout),
            "stderr": self._truncate(stderr),
            "returncode": result.get("returncode"),
        }

    async def _install_skill(
        self,
        package: str,
        *,
        skill_name: str | None,
        confirmed: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not self.allow_installs:
            return {
                "success": False,
                "error": "Skill installation is disabled by SKILLS_ALLOW_INSTALLS=false. Set it to true or unset it to enable installs.",
            }
        if not self._is_install_admin(user_id):
            return {
                "success": False,
                "error": "Skill installation is restricted by the SKILLS_INSTALL_ADMIN_USER_IDS user allow-list.",
            }
        if not confirmed:
            return {
                "success": False,
                "error": "confirmed must be true after explicit user approval for this package.",
            }

        package_id = (package or "").strip()
        if not package_id or any(char.isspace() for char in package_id):
            return {"success": False, "error": "package must be a non-empty skills package id without whitespace"}

        target_name = (skill_name or "").strip() or self._infer_skill_name_from_package(package_id)
        if not target_name:
            return {
                "success": False,
                "error": "skill_name is required when package does not include an @skill suffix.",
            }
        name_error = self._validate_skill_dir_name(target_name)
        if name_error:
            return {"success": False, "error": name_error}

        self._ensure_paths()
        target_path = (self.skills_dir / target_name).resolve()
        if target_path.exists():
            return {
                "success": False,
                "error": f"Skill '{target_name}' already exists in SKILLS_DIR.",
                "path": str(target_path),
            }

        install_result = await self._run_skills_cli(
            ["add", package_id, "-g", "--agent", SKILLS_CLI_AGENT, "--copy", "-y"],
            timeout=self.install_timeout,
        )
        install_stdout = self._strip_ansi(install_result.get("stdout", ""))
        install_stderr = self._strip_ansi(install_result.get("stderr", ""))
        if not install_result.get("success"):
            source_path, list_result = await self._find_cli_installed_skill_path(target_name)
            if source_path is not None:
                sync_mode, sync_error = self._sync_installed_skill(source_path, target_path)
                if sync_error is None:
                    self._scan_cache = None
                    self.available_skills = self._scan_skills()
                    self._audit(
                        "install_skill",
                        skill=target_name,
                        package=package_id,
                        user_id=user_id,
                        source_path=str(source_path),
                        path=str(target_path),
                        reused_existing=True,
                    )
                    return {
                        "success": True,
                        "package": package_id,
                        "skill": target_name,
                        "source_path": str(source_path),
                        "path": str(target_path),
                        "sync": sync_mode,
                        "warning": "skills CLI did not install a new copy, but an existing CLI install was synced into SKILLS_DIR.",
                        "stdout": self._truncate(install_stdout),
                        "stderr": self._truncate(install_stderr),
                        "available": target_name in self.available_skills,
                    }
            return {
                "success": False,
                "package": package_id,
                "skill": target_name,
                "returncode": install_result.get("returncode"),
                "stdout": self._truncate(install_stdout),
                "stderr": self._truncate(install_stderr),
                "error": "skills CLI failed to install the package.",
                "list_result": list_result,
            }

        source_path, list_result = await self._find_cli_installed_skill_path(target_name)
        if source_path is None:
            return {
                "success": False,
                "package": package_id,
                "skill": target_name,
                "stdout": self._truncate(install_stdout),
                "stderr": self._truncate(install_stderr),
                "error": "skills CLI reported success, but the installed skill path was not found.",
                "list_result": list_result,
            }

        sync_mode, sync_error = self._sync_installed_skill(source_path, target_path)
        if sync_error is not None:
            return {
                "success": False,
                "package": package_id,
                "skill": target_name,
                "source_path": str(source_path),
                "path": str(target_path),
                "error": sync_error,
            }

        self._scan_cache = None
        self.available_skills = self._scan_skills()
        self._audit(
            "install_skill",
            skill=target_name,
            package=package_id,
            user_id=user_id,
            source_path=str(source_path),
            path=str(target_path),
        )
        return {
            "success": True,
            "package": package_id,
            "skill": target_name,
            "source_path": str(source_path),
            "path": str(target_path),
            "sync": sync_mode,
            "stdout": self._truncate(install_stdout),
            "stderr": self._truncate(install_stderr),
            "available": target_name in self.available_skills,
        }

    def _sync_installed_skill(self, source_path: Path, target_path: Path) -> tuple[str, str | None]:
        try:
            if source_path.resolve() == target_path:
                return "already_in_skills_dir", None
            shutil.copytree(source_path, target_path, symlinks=True)
            return "copied_to_skills_dir", None
        except OSError as exc:
            return "", f"Failed to sync installed skill into SKILLS_DIR: {exc}"

    async def _run_skills_cli(self, args: List[str], *, timeout: int) -> Dict[str, Any]:
        npx_path = shutil.which("npx")
        if not npx_path:
            return {"success": False, "returncode": None, "stdout": "", "stderr": "npx executable not found"}

        self._ensure_paths()
        command = [npx_path, "-y", "skills", *args]
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["FORCE_COLOR"] = "0"
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.skills_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as exc:
            return {"success": False, "returncode": None, "stdout": "", "stderr": f"Failed to start skills CLI: {exc}"}

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"skills CLI timed out after {timeout}s",
            }

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _parse_installable_skill_results(self, stdout: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        lines = [line.strip() for line in stdout.splitlines()]
        for index, line in enumerate(lines):
            match = re.match(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+)(?:\s+(.*))?$", line)
            if not match:
                continue
            package = match.group(1)
            url = ""
            for next_line in lines[index + 1:index + 4]:
                if "http://" in next_line or "https://" in next_line:
                    url = next_line.lstrip("└- ").strip()
                    break
            results.append({
                "package": package,
                "skill_name": self._infer_skill_name_from_package(package),
                "summary": (match.group(2) or "").strip(),
                "url": url,
            })
        return results

    async def _find_cli_installed_skill_path(self, skill_name: str) -> tuple[Path | None, Dict[str, Any]]:
        list_result = await self._run_skills_cli(["ls", "-g", "--json"], timeout=self.install_timeout)
        stdout = self._strip_ansi(list_result.get("stdout", ""))
        if list_result.get("success"):
            try:
                entries = json.loads(stdout or "[]")
            except json.JSONDecodeError:
                entries = []
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict) or entry.get("name") != skill_name:
                        continue
                    path_text = entry.get("path")
                    if not path_text:
                        continue
                    path = Path(str(path_text)).expanduser().resolve()
                    if path.exists():
                        return path, {
                            "success": True,
                            "returncode": list_result.get("returncode"),
                        }

        for path in self._installed_skill_path_candidates(skill_name):
            if path.exists():
                return path, {
                    "success": bool(list_result.get("success")),
                    "returncode": list_result.get("returncode"),
                    "fallback": True,
                }

        return None, {
            "success": bool(list_result.get("success")),
            "returncode": list_result.get("returncode"),
            "stdout": self._truncate(stdout),
            "stderr": self._truncate(self._strip_ansi(list_result.get("stderr", ""))),
        }

    def _installed_skill_path_candidates(self, skill_name: str) -> List[Path]:
        candidates: List[Path] = []
        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(Path(codex_home).expanduser() / "skills" / skill_name)
        candidates.append(Path.home() / ".codex" / "skills" / skill_name)
        candidates.append(Path.home() / ".agents" / "skills" / skill_name)
        return [candidate.resolve() for candidate in candidates]

    def _infer_skill_name_from_package(self, package: str) -> str:
        if "@" not in package:
            return ""
        return package.rsplit("@", 1)[-1].strip()

    def _validate_skill_dir_name(self, value: str) -> str | None:
        if not value or value in {".", ".."}:
            return "skill_name must be a non-empty directory name"
        altsep = os.path.altsep
        if os.path.sep in value or (altsep and altsep in value):
            return "skill_name must be a single directory name, not a path"
        if any(char.isspace() for char in value):
            return "skill_name must not contain whitespace"
        return None

    def _strip_ansi(self, value: str) -> str:
        return ANSI_ESCAPE_RE.sub("", value or "")

    def _skill_scripts_allowed(self, info: Dict[str, Any]) -> bool:
        if not self.allow_scripts:
            return False
        if not info.get("scripts"):
            return False
        return info.get("metadata", {}).get("allow_scripts") is not False

    def _script_tool_calls(self, skill_id: str) -> List[Dict[str, Any]]:
        info = self.available_skills.get(skill_id)
        if not info:
            return []
        return [
            {
                "tool_name": "skills.run_skill_script",
                "arguments": {
                    "skill_name": skill_id,
                    "script_name": script_name,
                },
            }
            for script_name in info["scripts"]
        ]

    def _active_script_tool_calls(
        self,
        scope_state: Dict[str, Dict[str, Any]],
        disabled_skills: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        disabled_skills = disabled_skills or set()
        calls: List[Dict[str, Any]] = []
        for skill_id in sorted(scope_state):
            if skill_id in disabled_skills:
                continue
            calls.extend(self._script_tool_calls(skill_id))
        return calls

    def _log_available_skills(self) -> None:
        if not self.available_skills:
            logger.info("No skills found in %s", self.skills_dir)
            return
        skills_summary = ", ".join(
            f"{skill_id} ({info['name']})" if info["name"] != skill_id else skill_id
            for skill_id, info in self.available_skills.items()
        )
        logger.info("Available skills in %s (%s): %s", self.skills_dir, len(self.available_skills), skills_summary)

    def _bool_arg(self, value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in TRUE_VALUES:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _get_skill(self, skill_name: str) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        info = self.available_skills[skill_id]
        return {
            "success": True,
            "skill": {
                "id": skill_id,
                "name": info["name"],
                "description": info["description"],
                "metadata": info["metadata"],
                "body": info["body"],
                "scripts": info["scripts"],
                "agents": info.get("agents", []),
                "references": info.get("references", []),
            },
        }

    def _get_skill_reference(self, skill_name: str, reference_path: str) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        info = self.available_skills[skill_id]
        normalized_reference_path = str(reference_path or "").strip().strip("`")
        allowed_paths = {
            path
            for reference in info.get("references", [])
            for key in ("reference_path", "path")
            for path in (str(reference.get(key) or ""),)
            if path
        }
        if normalized_reference_path not in allowed_paths:
            return {
                "success": False,
                "error": f"Reference '{normalized_reference_path}' is not listed by skill '{skill_id}'",
            }
        skill_path = Path(info["path"]).resolve()
        path, relative_path, error = self._resolve_skill_reference_path(skill_path, normalized_reference_path)
        if error or path is None or relative_path is None:
            return {"success": False, "error": error or f"Reference '{normalized_reference_path}' not found"}
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"success": False, "error": f"Failed to read reference '{normalized_reference_path}': {exc}"}
        return {
            "success": True,
            "skill": skill_id,
            "reference_path": normalized_reference_path,
            "path": relative_path,
            "content": content,
        }

    def _activate_skill(self, skill_name: str, initial_context: Any = None, **kwargs) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        now = int(time.time())
        with self._state_lock:
            scope_state = self.active_skills.setdefault(scope, {})
            scope_state[skill_id] = {
                "current_step": 0,
                "context": self._decode_context(initial_context),
                "activated_at": now,
                "updated_at": now,
            }
            self._save_state()

        info = self.available_skills[skill_id]
        self._audit("activate_skill", skill=skill_id, scope=scope, user_id=kwargs.get("user_id"))
        return {
            "success": True,
            "skill": {
                "id": skill_id,
                "name": info["name"],
                "description": info["description"],
                "body": info["body"],
                "scripts": info["scripts"],
                "agents": info.get("agents", []),
                "references": info.get("references", []),
                "script_tool_calls": self._script_tool_calls(skill_id),
            },
            "scope": scope,
        }

    def _list_active_skills(self, disabled_skills: set[str] | None = None, **kwargs) -> Dict[str, Any]:
        disabled_skills = disabled_skills or set()
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            entries = []
            for skill_id, state in scope_state.items():
                if skill_id in disabled_skills:
                    continue
                info = self.available_skills.get(skill_id, {})
                entries.append({
                    "id": skill_id,
                    "name": info.get("name", skill_id),
                    "current_step": state.get("current_step", 0),
                    "activated_at": state.get("activated_at"),
                    "updated_at": state.get("updated_at"),
                })
        return {
            "success": True,
            "scope": scope,
            "active_skills": entries,
            "count": len(entries),
        }

    def _get_skill_status(self, skill_name: str = "", disabled_skills: set[str] | None = None, **kwargs) -> Dict[str, Any]:
        disabled_skills = disabled_skills or set()
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            if not skill_name:
                visible_scope_state = {
                    skill_id: state
                    for skill_id, state in scope_state.items()
                    if skill_id not in disabled_skills
                }
                return {
                    "success": True,
                    "scope": scope,
                    "active_skills": visible_scope_state,
                    "active_script_tool_calls": self._active_script_tool_calls(scope_state, disabled_skills),
                }

            skill_id = self._resolve_skill_id(skill_name)
            if not skill_id or skill_id not in scope_state:
                return {"success": False, "error": f"Skill '{skill_name}' is not active", "scope": scope}
            return {
                "success": True,
                "scope": scope,
                "skill": skill_id,
                "state": scope_state[skill_id],
                "script_tool_calls": self._script_tool_calls(skill_id),
            }

    async def _run_skill_agent(
        self,
        helper,
        skill_name: str,
        agent_name: str,
        task: str,
        *,
        context: Any = None,
        request_context=None,
        **kwargs,
    ) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        if not task.strip():
            return {"success": False, "error": "task must be a non-empty string"}

        info = self.available_skills[skill_id]
        agent = self._resolve_skill_agent(info, agent_name)
        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found in skill '{skill_id}'",
                "available_agents": [item["id"] for item in info.get("agents", [])],
            }

        plugin_manager = getattr(helper, "plugin_manager", None) if helper is not None else None
        agent_tools = None
        if plugin_manager is not None:
            resolver = getattr(helper, "resolve_allowed_plugins", None)
            is_function_allowed = getattr(plugin_manager, "is_function_allowed", None)
            if callable(resolver) and callable(is_function_allowed):
                chat_id = getattr(request_context, "chat_id", None) if request_context is not None else kwargs.get("chat_id")
                session_id = getattr(request_context, "session_id", None) if request_context is not None else None
                user_id = getattr(request_context, "user_id", None) if request_context is not None else kwargs.get("user_id")
                allowed_plugins = resolver(chat_id, session_id, user_id) if chat_id is not None else ["All"]
                if not is_function_allowed("agent_tools.run_subagents", allowed_plugins):
                    return {
                        "success": False,
                        "error": "agent_tools.run_subagents is not available in the current chat mode or user settings.",
                    }
            try:
                agent_tools = plugin_manager.get_plugin("agent_tools")
            except Exception:
                agent_tools = None
        if agent_tools is None or not hasattr(agent_tools, "execute"):
            return {
                "success": False,
                "error": "agent_tools plugin is required for skills.run_skill_agent.",
            }

        subagent = self._skill_agent_subagent_spec(skill_id, info, agent, task, context, kwargs)
        result = await agent_tools.execute(
            "run_subagents",
            helper,
            subagents=[subagent],
            max_rounds=kwargs.get("max_rounds"),
            request_context=request_context,
            chat_id=kwargs.get("chat_id"),
            user_id=kwargs.get("user_id"),
        )
        if not result.get("success"):
            return {
                "success": False,
                "skill": skill_id,
                "agent": agent["id"],
                "error": result.get("error") or "skill agent failed to start",
                "subagent_result": result,
            }
        subagents = result.get("subagents") or []
        return {
            "success": True,
            "skill": skill_id,
            "agent": agent["id"],
            "subagent": subagents[0] if subagents else None,
        }

    @staticmethod
    def _resolve_skill_agent(info: Dict[str, Any], agent_name: str) -> Dict[str, Any] | None:
        wanted = str(agent_name or "").strip().lower()
        if not wanted:
            return None
        for agent in info.get("agents", []):
            if wanted in {
                str(agent.get("id") or "").lower(),
                str(agent.get("display_name") or "").lower(),
            }:
                return agent
        return None

    def _skill_agent_subagent_spec(
        self,
        skill_id: str,
        info: Dict[str, Any],
        agent: Dict[str, Any],
        task: str,
        context: Any,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_id = str(agent.get("id") or "agent")
        display_name = str(agent.get("display_name") or agent_id)
        default_prompt = str(agent.get("default_prompt") or "").strip()
        short_description = str(agent.get("short_description") or "").strip()
        task_parts = [
            f"Use the skill `{skill_id}` as the governing instruction set.",
            f"Skill name: {info.get('name') or skill_id}",
        ]
        if short_description:
            task_parts.append(f"Skill agent purpose: {short_description}")
        if default_prompt:
            task_parts.append(f"Agent default prompt:\n{default_prompt}")
        task_parts.extend([
            f"Parent task:\n{task.strip()}",
            f"Skill instructions:\n{self._truncate(str(info.get('body') or ''))}",
        ])
        references = info.get("references") or []
        if references:
            task_parts.append(
                "Available skill references:\n"
                + "\n".join(
                    f"- {reference.get('reference_path')}"
                    for reference in references
                    if reference.get("reference_path")
                )
            )

        context_parts = [
            f"Skill path: {info.get('path')}",
            f"Agent definition: {agent.get('file')}",
        ]
        if context is not None:
            context_parts.append(f"Task context:\n{context}")

        subagent: Dict[str, Any] = {
            "id": self._safe_subagent_id(f"{skill_id}_{agent_id}"),
            "role": display_name,
            "task": "\n\n".join(task_parts),
            "context": "\n\n".join(context_parts),
        }
        for optional in ("max_rounds", "model", "temperature"):
            if kwargs.get(optional) is not None:
                subagent[optional] = kwargs[optional]
        return subagent

    @staticmethod
    def _safe_subagent_id(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        normalized = normalized.strip("._-")
        return (normalized or "skill_agent")[:80]

    def _update_skill_progress(
        self,
        skill_name: str,
        step: int,
        context_update: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        with self._state_lock:
            if skill_id not in self.active_skills.get(scope, {}):
                return {"success": False, "error": f"Skill '{skill_name}' is not active", "scope": scope}

            state = self.active_skills[scope][skill_id]
            state["current_step"] = step
            update = self._decode_context(context_update)
            if isinstance(update, dict):
                state.setdefault("context", {}).update(update)
            else:
                state.setdefault("context", {})["last_update"] = str(update)
            state["updated_at"] = int(time.time())
            self._save_state()
            return {"success": True, "scope": scope, "skill": skill_id, "state": state}

    def _deactivate_skill(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            removed = scope_state.pop(skill_id, None)
            if removed is None:
                return {"success": False, "error": f"Skill '{skill_name}' is not active", "scope": scope}
            if not scope_state:
                self.active_skills.pop(scope, None)
            self._save_state()
        self._audit("deactivate_skill", skill=skill_id, scope=scope, user_id=kwargs.get("user_id"))
        return {"success": True, "scope": scope, "skill": skill_id, "deactivated": True}

    async def _run_skill_script(
        self,
        skill_name: str,
        script_name: str,
        args_json: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not self.allow_scripts:
            return {
                "success": False,
                "error": "Skill scripts are disabled. Set SKILLS_ALLOW_SCRIPTS=true to enable them.",
            }
        if not self._is_script_admin(user_id):
            return {
                "success": False,
                "error": "Skill script execution is restricted by SKILLS_SCRIPT_ADMIN_USER_IDS.",
            }

        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        with self._state_lock:
            if skill_id not in self.active_skills.get(scope, {}):
                return {"success": False, "error": f"Skill '{skill_name}' is not active", "scope": scope}

        info = self.available_skills[skill_id]
        if info.get("metadata", {}).get("allow_scripts") is False:
            return {"success": False, "error": f"Skill '{skill_name}' does not allow scripts"}
        if script_name not in info["scripts"]:
            return {"success": False, "error": f"Script '{script_name}' not found in skill '{skill_id}'"}

        skill_path = Path(info["path"]).resolve()
        scripts_dir = (skill_path / "scripts").resolve()
        script_path = (scripts_dir / script_name).resolve()
        if not self._is_relative_to(script_path, scripts_dir):
            return {"success": False, "error": "Invalid script path"}
        try:
            mode = script_path.lstat().st_mode
        except OSError as exc:
            return {"success": False, "error": f"Cannot stat script '{script_name}': {exc}"}
        if not stat.S_ISREG(mode):
            return {"success": False, "error": f"Script '{script_name}' is not a regular file"}

        try:
            argv = self._parse_script_args(args_json)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        command, runtime, runtime_error = self._script_command(script_path)
        if runtime_error:
            return {
                "success": False,
                "error": runtime_error,
                "skill": skill_id,
                "script": script_name,
            }

        started = time.monotonic()
        logger.info(
            "Running skill script skill=%s script=%s runtime=%s user_id=%s scope=%s",
            skill_id,
            script_name,
            runtime,
            user_id,
            scope,
        )
        workdir = self._ensure_skill_workdir(skill_id, scope)
        chat_id = kwargs.get("chat_id")
        interim_task = self._spawn_interim_notice(chat_id, skill_id, script_name)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                *argv,
                cwd=str(skill_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._script_env(skill_id=skill_id, scope=scope, workdir=workdir),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.script_timeout,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            self._audit(
                "run_skill_script",
                skill=skill_id,
                script=script_name,
                runtime=runtime,
                scope=scope,
                user_id=user_id,
                returncode=None,
                error="timeout",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return {"success": False, "error": f"Script '{script_name}' timed out"}
        except Exception as exc:
            self._audit(
                "run_skill_script",
                skill=skill_id,
                script=script_name,
                runtime=runtime,
                scope=scope,
                user_id=user_id,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return {"success": False, "error": f"Failed to run script '{script_name}': {exc}"}
        finally:
            if interim_task is not None:
                interim_task.cancel()

        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = self._truncate(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = self._truncate(stderr_bytes.decode("utf-8", errors="replace"))
        logger.info(
            "Skill script finished skill=%s script=%s runtime=%s returncode=%s duration_ms=%s",
            skill_id,
            script_name,
            runtime,
            process.returncode,
            duration_ms,
        )
        self._audit(
            "run_skill_script",
            skill=skill_id,
            script=script_name,
            runtime=runtime,
            scope=scope,
            user_id=user_id,
            returncode=process.returncode,
            duration_ms=duration_ms,
        )
        return {
            "success": process.returncode == 0,
            "skill": skill_id,
            "script": script_name,
            "runtime": runtime,
            "returncode": process.returncode,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "workdir": str(workdir) if workdir else None,
        }

    def _record_skill_reflection(
        self,
        skill_name: str,
        proposal: str,
        *,
        failure_mode: Any = None,
        evidence: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        proposal_text = self._normalize_reflection_text(proposal)
        if not proposal_text:
            return {"success": False, "error": "Reflection proposal must be non-empty"}

        now = int(time.time())
        proposal_id = self._reflection_id(proposal_text)
        edit_result = {"applied": False, "already_present": False}
        with self._reflection_lock:
            skill_entries = self.skill_reflections.setdefault(skill_id, {})
            entry = skill_entries.setdefault(proposal_id, {
                "proposal": proposal_text,
                "count": 0,
                "created_at": now,
                "examples": [],
            })
            entry["proposal"] = proposal_text
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["updated_at"] = now
            failure_text = self._normalize_reflection_text(failure_mode)
            if failure_text:
                entry["failure_mode"] = failure_text
            evidence_text = self._normalize_reflection_text(evidence)
            if evidence_text:
                examples = entry.setdefault("examples", [])
                examples.append({
                    "ts": now,
                    "scope": compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id")),
                    "evidence": evidence_text,
                })
                del examples[:-3]

            if entry["count"] > REFLECTION_REPEAT_THRESHOLD and not entry.get("applied_at"):
                edit_result = self._append_skill_reflection(skill_id, proposal_text)
                if edit_result.get("error"):
                    entry["last_apply_error"] = edit_result["error"]
                else:
                    entry["applied_at"] = now
                    entry["applied"] = bool(edit_result.get("applied"))
                    if edit_result.get("already_present"):
                        entry["already_present"] = True
                    entry.pop("last_apply_error", None)
            self._save_reflections()

        self._audit(
            "record_skill_reflection",
            skill=skill_id,
            proposal_id=proposal_id,
            count=entry["count"],
            applied=edit_result.get("applied"),
            user_id=kwargs.get("user_id"),
        )
        result = {
            "success": not bool(edit_result.get("error")),
            "skill": skill_id,
            "proposal_id": proposal_id,
            "count": entry["count"],
            "threshold": REFLECTION_REPEAT_THRESHOLD,
            "threshold_reached": entry["count"] > REFLECTION_REPEAT_THRESHOLD,
            "applied": bool(edit_result.get("applied")),
            "already_present": bool(edit_result.get("already_present")),
        }
        if edit_result.get("error"):
            result["error"] = edit_result["error"]
        return result

    def _append_skill_reflection(self, skill_id: str, proposal_text: str) -> Dict[str, Any]:
        info = self.available_skills.get(skill_id)
        if not info:
            return {"applied": False, "error": f"Skill '{skill_id}' not found"}
        md_path = Path(info["path"]) / "SKILL.md"
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"applied": False, "error": f"Cannot read {md_path}: {exc}"}
        if proposal_text in content:
            return {"applied": False, "already_present": True}

        updated = content.rstrip()
        if "## Learned Clarifications" not in updated:
            updated += "\n\n## Learned Clarifications\n"
        elif not updated.endswith("\n"):
            updated += "\n"
        updated += f"- {proposal_text}\n"
        try:
            tmp_path = md_path.with_suffix(md_path.suffix + ".tmp")
            tmp_path.write_text(updated, encoding="utf-8")
            os.replace(tmp_path, md_path)
        except OSError as exc:
            return {"applied": False, "error": f"Cannot update {md_path}: {exc}"}

        self._scan_cache = None
        self.available_skills = self._scan_skills()
        return {"applied": True, "already_present": False}

    def cleanup_after_delivery(self, cleanup: Dict[str, Any]) -> bool:
        if not isinstance(cleanup, dict):
            return False
        scope = cleanup.get("scope")
        skill_id = cleanup.get("skill_id")
        if not scope or not skill_id:
            return False
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            removed = scope_state.pop(skill_id, None)
            if removed is None:
                return False
            if not scope_state:
                self.active_skills.pop(scope, None)
            self._save_state()
        logger.info(
            "Skill state cleaned up after delivery skill=%s scope=%s",
            skill_id,
            scope,
        )
        self._audit("cleanup_after_delivery", skill=skill_id, scope=scope)
        return True

    def _script_command(self, script_path: Path) -> tuple[List[str] | None, str | None, str | None]:
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            return [sys.executable, str(script_path)], "python", None
        if suffix in {".js", ".mjs", ".cjs"}:
            node_path = shutil.which("node")
            if not node_path:
                return None, "node", "Runtime 'node' is not available for this skill script"
            return [node_path, str(script_path)], "node", None
        if suffix == ".sh":
            shell_path = shutil.which("bash") or shutil.which("sh")
            if not shell_path:
                return None, "shell", "Runtime 'bash' or 'sh' is not available for this skill script"
            return [shell_path, str(script_path)], "shell", None
        if os.access(script_path, os.X_OK):
            return [str(script_path)], "executable", None
        return (
            None,
            None,
            (
                f"Unsupported script runtime for '{script_path.name}'. "
                "Supported scripts: .py, .js/.mjs/.cjs, .sh, or executable files."
            ),
        )

    def _resolve_skill_id(self, skill_name: str) -> str | None:
        normalized = (skill_name or "").strip()
        if not normalized:
            return None
        if normalized in self.available_skills:
            return normalized
        lowered = normalized.lower()
        for skill_id, info in self.available_skills.items():
            if str(info.get("name", "")).lower() == lowered:
                return skill_id
        return None

    def _decode_context(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        text = str(value).strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {"value": data}
        except json.JSONDecodeError:
            return {"task": text}

    def _parse_script_args(self, args_json: Any) -> List[str]:
        if args_json is None:
            return []
        text = str(args_json).strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(data, list):
            return [self._arg_to_string(item) for item in data]
        if isinstance(data, dict):
            args: List[str] = []
            for key, value in data.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Script args object keys must be non-empty strings")
                args.append(f"--{key.replace('_', '-')}")
                if isinstance(value, bool):
                    args.append("true" if value else "false")
                else:
                    args.append(self._arg_to_string(value))
            return args
        return [self._arg_to_string(data)]

    def _arg_to_string(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _normalize_reflection_text(self, value: Any) -> str:
        text = " ".join(str(value or "").split())
        return text[:REFLECTION_MAX_TEXT_CHARS]

    def _reflection_id(self, proposal_text: str) -> str:
        normalized = proposal_text.casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _load_reflections(self) -> None:
        if not self.reflections_file or not self.reflections_file.exists():
            self.skill_reflections = {}
            return
        with self._reflection_lock:
            try:
                data = json.loads(self.reflections_file.read_text(encoding="utf-8"))
            except Exception as exc:
                broken_path = self.reflections_file.with_suffix(self.reflections_file.suffix + ".broken")
                logger.error(
                    "Failed to load skill reflections from %s: %s. Quarantining to %s.",
                    self.reflections_file, exc, broken_path,
                )
                try:
                    os.replace(self.reflections_file, broken_path)
                except Exception:
                    logger.exception("Could not move corrupted skill reflections to %s", broken_path)
                self.skill_reflections = {}
                return
            self.skill_reflections = data if isinstance(data, dict) else {}

    def _save_reflections(self) -> None:
        if not self.reflections_file:
            return
        with self._reflection_lock:
            try:
                self.reflections_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.reflections_file.with_suffix(self.reflections_file.suffix + ".tmp")
                tmp_path.write_text(
                    json.dumps(self.skill_reflections, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp_path, self.reflections_file)
            except Exception as exc:
                logger.warning("Failed to save skill reflections: %s", exc)

    def _load_state(self) -> None:
        if not self.state_file or not self.state_file.exists():
            self.active_skills = {}
            return
        with self._state_lock:
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                broken_path = self.state_file.with_suffix(self.state_file.suffix + ".broken")
                logger.error(
                    "Failed to load skills state from %s: %s. Quarantining to %s and refusing to overwrite.",
                    self.state_file, exc, broken_path,
                )
                try:
                    os.replace(self.state_file, broken_path)
                except Exception:
                    logger.exception("Could not move corrupted skills state to %s", broken_path)
                self.active_skills = {}
                return
            self.active_skills = data if isinstance(data, dict) else {}

    def _save_state(self) -> None:
        if not self.state_file:
            return
        with self._state_lock:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
                tmp_path.write_text(
                    json.dumps(self.active_skills, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp_path, self.state_file)
            except Exception as exc:
                logger.warning("Failed to save skills state: %s", exc)

    def _ensure_paths(self) -> None:
        if self.skills_dir is None:
            self.initialize(
                openai=getattr(self, "openai", None),
                bot=getattr(self, "bot", None),
                storage_root=getattr(self, "storage_root", None),
            )
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _env_flag(self, name: str, *, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in TRUE_VALUES

    def _env_int(self, name: str, *, default: int, minimum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            return default
        return max(minimum, value)

    def _parse_admin_ids(self, raw: str, *, env_name: str = "SKILLS_SCRIPT_ADMIN_USER_IDS") -> tuple[set[int], bool]:
        value = (raw or "").strip()
        if value == "*":
            return set(), True
        ids = set()
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.add(int(item))
            except ValueError:
                logger.warning("Ignoring invalid %s entry: %s", env_name, item)
        return ids, False

    def _is_script_admin(self, user_id: Any) -> bool:
        if self.script_admin_all:
            return True
        try:
            return int(user_id) in self.script_admin_ids
        except (TypeError, ValueError):
            return False

    def _is_install_admin(self, user_id: Any) -> bool:
        if self.install_admin_all:
            return True
        try:
            return int(user_id) in self.install_admin_ids
        except (TypeError, ValueError):
            return False

    def _script_env(
        self,
        skill_id: str | None = None,
        scope: str | None = None,
        workdir: Path | None = None,
    ) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if workdir is not None:
            env["SKILL_WORKDIR"] = str(workdir)
        if skill_id:
            env["SKILL_ID"] = skill_id
        if scope:
            env["SKILL_SCOPE"] = scope
        return env

    def _ensure_skill_workdir(self, skill_id: str, scope: str) -> Path | None:
        if self.workdir_root is None:
            return None
        safe_scope = scope.replace(":", "_").replace("/", "_") or "global"
        workdir = (self.workdir_root / skill_id / safe_scope).resolve()
        try:
            workdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create skill workdir %s: %s", workdir, exc)
            return None
        return workdir

    def _spawn_interim_notice(
        self,
        chat_id: Any,
        skill_id: str,
        script_name: str,
    ) -> "asyncio.Task | None":
        bot = getattr(self, "bot", None) or getattr(getattr(self, "openai", None), "bot", None)
        if bot is None or chat_id is None or self.interim_after_seconds <= 0:
            return None
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return None
        first_delay = self.interim_after_seconds
        repeat_delay = max(first_delay, 60)
        typing_refresh_seconds = 4

        async def _watchdog():
            try:
                # Refresh the typing indicator every ~4s (Telegram clears it after 5s)
                # while the skill script runs. After the configured first delay, post
                # a textual interim notice and continue refreshing typing. Repeat the
                # textual notice every repeat_delay seconds for very long runs.
                elapsed = 0
                first_notice_sent = False
                last_notice_at = 0
                while True:
                    try:
                        await bot.send_chat_action(chat_id=chat_id_int, action="typing")
                    except Exception:
                        logger.debug("Failed to refresh typing indicator", exc_info=True)
                    await asyncio.sleep(typing_refresh_seconds)
                    elapsed += typing_refresh_seconds
                    if not first_notice_sent and elapsed >= first_delay:
                        try:
                            await bot.send_message(
                                chat_id=chat_id_int,
                                text=(
                                    f"⏳ Скрипт «{script_name}» скила «{skill_id}» всё ещё выполняется. "
                                    "Дождитесь завершения, не отправляя новых запросов."
                                ),
                            )
                        except Exception:
                            logger.debug("Failed to send interim skill script notice", exc_info=True)
                        first_notice_sent = True
                        last_notice_at = elapsed
                    elif first_notice_sent and elapsed - last_notice_at >= repeat_delay:
                        try:
                            await bot.send_message(
                                chat_id=chat_id_int,
                                text=(
                                    f"⏳ Скрипт «{script_name}» скила «{skill_id}» работает уже "
                                    f"{elapsed} с. Продолжаю ждать."
                                ),
                            )
                        except Exception:
                            logger.debug("Failed to send watchdog notice", exc_info=True)
                        last_notice_at = elapsed
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Skill script watchdog failed", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(_watchdog())

    def _audit(self, action: str, **fields: Any) -> None:
        if self.audit_file is None:
            return
        record = {"ts": int(time.time()), "action": action}
        for key, value in fields.items():
            if value is None:
                continue
            record[key] = value
        try:
            with self._audit_lock:
                self.audit_file.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(record, ensure_ascii=False)
                with open(self.audit_file, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            logger.debug("Failed to append skills audit entry", exc_info=True)

    def _truncate(self, value: str) -> str:
        if len(value) <= self.output_max_chars:
            return value
        omitted = len(value) - self.output_max_chars
        return value[: self.output_max_chars] + f"\n...[truncated {omitted} chars]"

    def _is_relative_to(self, path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
