from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .plugin import Plugin


logger = logging.getLogger(__name__)

TRUE_VALUES = {"1", "true", "yes", "on"}


class SkillsPlugin(Plugin):
    """
    Exposes local Codex-style skills as tools for the existing function-calling loop.
    """

    plugin_id = "skills"
    function_prefix = "skills"

    def __init__(self):
        self.skills_dir: Path | None = None
        self.state_file: Path | None = None
        self.workdir_root: Path | None = None
        self.audit_file: Path | None = None
        self.available_skills: Dict[str, Dict[str, Any]] = {}
        self.active_skills: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.allow_scripts = False
        self.script_admin_ids: set[int] = set()
        self.script_admin_all = False
        self.script_timeout = 120
        self.output_max_chars = 12000
        self.interim_after_seconds = 20
        self._state_lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._scan_cache: tuple[tuple, Dict[str, Dict[str, Any]]] | None = None

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        storage_path = Path(storage_root).resolve() if storage_root else Path.cwd().resolve()
        skills_dir = os.getenv("SKILLS_DIR")
        self.skills_dir = Path(skills_dir).expanduser().resolve() if skills_dir else storage_path / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = storage_path / "skills_state.json"
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
        self.output_max_chars = self._env_int("SKILLS_SCRIPT_OUTPUT_MAX_CHARS", default=12000, minimum=1000)
        self.interim_after_seconds = self._env_int(
            "SKILLS_SCRIPT_INTERIM_AFTER_SECONDS", default=20, minimum=5
        )
        self.script_admin_ids, self.script_admin_all = self._parse_admin_ids(
            os.getenv("SKILLS_SCRIPT_ADMIN_USER_IDS", "")
        )

        self.available_skills = self._scan_skills()
        self._log_available_skills()
        self._load_state()

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
        kwargs.pop("request_context", None)
        if function_name == "list_skills":
            return self._list_skills(refresh=self._bool_arg(kwargs.get("refresh"), default=True))
        if function_name == "get_skill":
            return self._get_skill(str(kwargs.get("skill_name") or ""))
        if function_name == "activate_skill":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            initial_context = call_kwargs.pop("initial_context", None)
            return self._activate_skill(
                skill_name,
                initial_context,
                **call_kwargs,
            )
        if function_name == "get_skill_status":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            return self._get_skill_status(skill_name, **call_kwargs)
        if function_name == "list_active_skills":
            return self._list_active_skills(**kwargs)
        if function_name == "update_skill_progress":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
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
            return self._deactivate_skill(skill_name, **call_kwargs)
        if function_name == "run_skill_script":
            call_kwargs = dict(kwargs)
            skill_name = str(call_kwargs.pop("skill_name", "") or "")
            script_name = str(call_kwargs.pop("script_name", "") or "")
            args_json = call_kwargs.pop("args_json", None)
            return await self._run_skill_script(
                skill_name,
                script_name,
                args_json,
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

    def _scan_skills(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_paths()
        signature = self._scan_signature()
        if self._scan_cache is not None and self._scan_cache[0] == signature:
            return self._scan_cache[1]

        skills: Dict[str, Dict[str, Any]] = {}
        for skill_path in sorted(self.skills_dir.iterdir(), key=lambda p: p.name):
            if not skill_path.is_dir():
                continue
            md_file = skill_path / "SKILL.md"
            if not md_file.exists():
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
                metadata, body = self._parse_skill_markdown(content)
                scripts = self._list_skill_scripts(skill_path)
                skill_id = skill_path.name
                skills[skill_id] = {
                    "id": skill_id,
                    "name": str(metadata.get("name") or skill_id),
                    "description": str(metadata.get("description") or ""),
                    "metadata": metadata,
                    "body": body,
                    "scripts": scripts,
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
        try:
            for skill_path in sorted(self.skills_dir.iterdir(), key=lambda p: p.name):
                if not skill_path.is_dir():
                    continue
                md_file = skill_path / "SKILL.md"
                if not md_file.exists():
                    continue
                try:
                    md_mtime = md_file.stat().st_mtime_ns
                except OSError:
                    continue
                scripts_dir = skill_path / "scripts"
                script_entries: List[tuple] = []
                if scripts_dir.exists() and scripts_dir.is_dir():
                    for path in sorted(scripts_dir.rglob("*")):
                        try:
                            if not path.is_file():
                                continue
                        except OSError:
                            continue
                        try:
                            script_entries.append((str(path.relative_to(scripts_dir)), path.stat().st_mtime_ns))
                        except OSError:
                            continue
                entries.append((skill_path.name, md_mtime, tuple(script_entries)))
        except OSError:
            return ()
        return tuple(entries)

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

    def _list_skill_scripts(self, skill_path: Path) -> List[str]:
        scripts_dir = skill_path / "scripts"
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            return []
        scripts = []
        for path in scripts_dir.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(scripts_dir)
            if "__pycache__" in relative_path.parts:
                continue
            if any(part.startswith(".") for part in relative_path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            scripts.append(relative_path.as_posix())
        return sorted(scripts)

    def _list_skills(self, *, refresh: bool = False) -> Dict[str, Any]:
        if refresh:
            self.available_skills = self._scan_skills()
        return {
            "success": True,
            "skills": [
                {
                    "id": skill_id,
                    "name": info["name"],
                    "description": info["description"],
                    "scripts": info["scripts"],
                    "allow_scripts": self._skill_scripts_allowed(info),
                }
                for skill_id, info in self.available_skills.items()
            ],
            "scripts_enabled": self.allow_scripts,
            "scripts_admin_restricted": not self.script_admin_all,
        }

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

    def _active_script_tool_calls(self, scope_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for skill_id in sorted(scope_state):
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
            },
        }

    def _activate_skill(self, skill_name: str, initial_context: Any = None, **kwargs) -> Dict[str, Any]:
        skill_id = self._resolve_skill_id(skill_name)
        if not skill_id:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        scope = self._scope_key(kwargs)
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
                "script_tool_calls": self._script_tool_calls(skill_id),
            },
            "scope": scope,
        }

    def _list_active_skills(self, **kwargs) -> Dict[str, Any]:
        scope = self._scope_key(kwargs)
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            entries = []
            for skill_id, state in scope_state.items():
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

    def _get_skill_status(self, skill_name: str = "", **kwargs) -> Dict[str, Any]:
        scope = self._scope_key(kwargs)
        with self._state_lock:
            scope_state = self.active_skills.get(scope, {})
            if not skill_name:
                return {
                    "success": True,
                    "scope": scope,
                    "active_skills": scope_state,
                    "active_script_tool_calls": self._active_script_tool_calls(scope_state),
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

    def _update_skill_progress(
        self,
        skill_name: str,
        step: int,
        context_update: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        scope = self._scope_key(kwargs)
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
        scope = self._scope_key(kwargs)
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

        scope = self._scope_key(kwargs)
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

    def _scope_key(self, kwargs: Dict[str, Any]) -> str:
        chat_id = kwargs.get("chat_id")
        if chat_id is not None:
            return f"chat:{chat_id}"
        user_id = kwargs.get("user_id")
        if user_id is not None:
            return f"user:{user_id}"
        return "global"

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

    def _parse_admin_ids(self, raw: str) -> tuple[set[int], bool]:
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
                logger.warning("Ignoring invalid SKILLS_SCRIPT_ADMIN_USER_IDS entry: %s", item)
        return ids, False

    def _is_script_admin(self, user_id: Any) -> bool:
        if self.script_admin_all:
            return True
        try:
            return int(user_id) in self.script_admin_ids
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
        delay = self.interim_after_seconds

        async def _send_interim():
            try:
                await asyncio.sleep(delay)
                await bot.send_message(
                    chat_id=chat_id_int,
                    text=(
                        f"⏳ Скрипт «{script_name}» скила «{skill_id}» всё ещё выполняется. "
                        "Дождитесь завершения, не отправляя новых запросов."
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Failed to send interim skill script notice", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(_send_interim())

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
