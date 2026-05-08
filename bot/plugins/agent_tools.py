from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, MessageHandler, filters

from ..skill_script_routing import _skill_script_routing_error
from .plugin import Plugin


CLOSED_STATUSES = {"completed", "cancelled"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
MAX_SUBAGENTS = 5
MAX_SUBAGENT_TOOL_ROUNDS = 10
MIN_SUBAGENT_TOOL_ROUNDS = 10
DEFAULT_SUBAGENT_TEMPERATURE = 0.2
SUBAGENT_BLOCKED_FUNCTIONS = {
    "agent_tools.ask_telegram_user",
    "agent_tools.cancel_pending_question",
    "agent_tools.deliver_to_user",
    "agent_tools.run_subagents",
    # Subagents share the parent's chat scope; activating or running skill scripts
    # mutates state and writes into directories the parent owns, so keep them out.
    "skills.activate_skill",
    "skills.deactivate_skill",
    "skills.run_skill_script",
    "terminal.terminal",
}


class _PendingAskReplyFilter(filters.MessageFilter):
    def __init__(self, plugin: "AgentToolsPlugin"):
        super().__init__(name="PendingAgentAskReply", data_filter=False)
        self.plugin = plugin

    def filter(self, message) -> bool:
        chat = getattr(message, "chat", None)
        user = getattr(message, "from_user", None)
        text = getattr(message, "text", None)
        if not chat or not text:
            return False
        return self.plugin.is_waiting_for_text(getattr(chat, "id", None), getattr(user, "id", None))


class AgentToolsPlugin(Plugin):
    """
    Lightweight agent-support tools for the existing function-calling loop.
    """

    plugin_id = "agent_tools"
    function_prefix = "agent_tools"

    def __init__(self):
        self.tasks: Dict[str, List[Dict[str, Any]]] = {}
        self.tasks_file = os.path.join(os.path.dirname(__file__), "agent_tasks.json")
        self.pending_file = os.path.join(os.path.dirname(__file__), "agent_pending_questions.json")
        self.pending_questions: Dict[str, Dict[str, Any]] = {}
        self.pending_by_chat: Dict[int, str] = {}
        self._orphaned_pending: List[Dict[str, Any]] = []
        self.load_tasks()
        self._load_orphaned_pending()

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            self.tasks_file = os.path.join(storage_root, "agent_tasks.json")
            self.pending_file = os.path.join(storage_root, "agent_pending_questions.json")
            self.load_tasks()
            self._load_orphaned_pending()

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        for question_id in list(self.pending_questions):
            pending = self.pending_questions.pop(question_id, None)
            if not pending:
                continue
            future = pending.get("future")
            if future and not future.done():
                future.cancel()
            if loop is not None:
                bot = pending.get("bot")
                chat_id = pending.get("chat_id")
                message_id = pending.get("message_id")
                if bot and chat_id is not None and message_id is not None:
                    loop.create_task(
                        self._clear_question_markup(bot, chat_id, message_id)
                    )
        self.pending_by_chat.clear()

    @staticmethod
    async def _clear_question_markup(bot, chat_id: int, message_id: int) -> None:
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=None,
            )
        except Exception:
            logging.debug(
                "Failed to clear ask_telegram_user inline markup for chat %s message %s",
                chat_id, message_id, exc_info=True,
            )

    def get_source_name(self) -> str:
        return "AgentTools"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "manage_plan_tasks",
                "description": (
                    "Manage a short planning task list for complex multi-step work in the current "
                    "Telegram chat. Use it to create a plan, inspect progress, and mark steps done."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "update", "list", "clear"],
                            "description": "Action to perform on the planning task list.",
                        },
                        "tasks": {
                            "type": "array",
                            "description": "Tasks for add/update actions.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Stable task id, for example T1."},
                                    "content": {"type": "string", "description": "Task text."},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed", "cancelled"],
                                    },
                                },
                            },
                        },
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "ask_telegram_user",
                "description": (
                    "Ask the Telegram user a concrete question and wait for a button or text answer. "
                    "Use when explicit confirmation, a choice, or missing information is required. "
                    "Answer variants must be passed in options, not embedded into the question text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to send to the user."},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6,
                            "description": "Required button labels for the answer choices.",
                        },
                        "allow_free_text": {
                            "type": "boolean",
                            "description": "Whether the user may answer with a normal Telegram text message.",
                        },
                        "multi_select": {
                            "type": "boolean",
                            "description": (
                                "If true, the user can pick several options. Each tap toggles a "
                                "selection; the user finalizes with the Confirm button. The answer "
                                "is returned as a comma-separated string. Free-text answers are "
                                "disabled in multi-select mode."
                            ),
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "How long to wait for the user's answer. Defaults to 1800.",
                        },
                    },
                    "required": ["question", "options"],
                },
            },
            {
                "name": "cancel_pending_question",
                "description": (
                    "Cancel an outstanding ask_telegram_user question for this Telegram chat. "
                    "Use when the question is no longer needed (e.g., the agent decided to "
                    "proceed differently). Clears the inline keyboard."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Optional short reason shown to the user instead of the buttons.",
                        }
                    },
                },
            },
            {
                "name": "deliver_to_user",
                "description": (
                    "Final delivery to the Telegram user. Send optional final text and zero or "
                    "more local files. Calling this tool ends the agent loop, delivers everything "
                    "to the user, and automatically deactivates any skills still active in this "
                    "chat scope. Use this as the only way to publish a final answer — do not "
                    "address the user with plain assistant text and do not call this twice."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Optional final text shown to the user.",
                        },
                        "artifacts": {
                            "type": "array",
                            "description": (
                                "Optional list of local files to send. Each item must be an "
                                "object with file_path (absolute path) and an optional caption."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Absolute path to an existing non-empty file.",
                                    },
                                    "caption": {
                                        "type": "string",
                                        "description": "Optional caption sent with the file.",
                                    },
                                },
                                "required": ["file_path"],
                            },
                        },
                    },
                },
            },
            {
                "name": "run_subagents",
                "description": (
                    "Run independent tool-capable subagents in parallel for bounded subtasks. "
                    "Subagents may call tools and skills, but they cannot ask Telegram questions, "
                    "publish final artifacts, or start nested subagents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shared_context": {
                            "type": "string",
                            "description": "Optional context shared with every subagent.",
                        },
                        "max_rounds": {
                            "type": "integer",
                            "description": (
                                f"Optional per-subagent tool-call rounds budget. "
                                f"Floor and default is {MIN_SUBAGENT_TOOL_ROUNDS}; pass a higher value "
                                "for harder tasks. Each round is one model call followed by parallel tool execution."
                            ),
                        },
                        "subagents": {
                            "type": "array",
                            "description": "Subagents to run in parallel.",
                            "minItems": 1,
                            "maxItems": MAX_SUBAGENTS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Stable short id, for example research_1.",
                                    },
                                    "role": {
                                        "type": "string",
                                        "description": "Role/persona for this subagent.",
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": "Concrete bounded task for this subagent.",
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "Optional extra context for this subagent only.",
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": (
                                            "Optional sampling temperature for this subagent only. "
                                            f"Defaults to {DEFAULT_SUBAGENT_TEMPERATURE}."
                                        ),
                                    },
                                    "max_rounds": {
                                        "type": "integer",
                                        "description": (
                                            f"Optional per-subagent rounds budget overriding the parent value. "
                                            f"Floor is {MIN_SUBAGENT_TOOL_ROUNDS}."
                                        ),
                                    },
                                },
                                "required": ["id", "role", "task"],
                            },
                        },
                    },
                    "required": ["subagents"],
                },
            },
        ]

    def get_commands(self) -> List[Dict]:
        return [
            {
                "callback_query_handler": self.handle_ask_callback,
                "callback_pattern": "^agentask:",
                "handler_kwargs": {},
                "add_to_menu": False,
            }
        ]

    def get_message_handlers(self) -> List[Dict]:
        return [
            {
                "handler": MessageHandler(_PendingAskReplyFilter(self), self.handle_text_answer),
            }
        ]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        kwargs.pop("request_context", None)
        if function_name == "manage_plan_tasks":
            return self._manage_plan_tasks(helper, **kwargs)
        if function_name == "ask_telegram_user":
            return await self._ask_telegram_user(helper, **kwargs)
        if function_name == "cancel_pending_question":
            return await self._cancel_pending_question(helper, **kwargs)
        if function_name == "run_subagents":
            return await self._run_subagents(helper, **kwargs)
        if function_name == "deliver_to_user":
            return await self._deliver_to_user(helper, **kwargs)
        return {"success": False, "error": f"Unknown agent tool: {function_name}"}

    def load_tasks(self) -> None:
        if not os.path.exists(self.tasks_file):
            self.tasks = {}
            return
        try:
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            broken_path = f"{self.tasks_file}.broken"
            logging.error(
                "Failed to load agent tasks from %s: %s. Quarantining to %s.",
                self.tasks_file, exc, broken_path,
            )
            try:
                os.replace(self.tasks_file, broken_path)
            except Exception:
                logging.exception("Could not move corrupted agent tasks to %s", broken_path)
            self.tasks = {}
            return
        self.tasks = data if isinstance(data, dict) else {}

    def save_tasks(self) -> None:
        try:
            tasks_dir = os.path.dirname(self.tasks_file) or "."
            os.makedirs(tasks_dir, exist_ok=True)
            tmp_path = f"{self.tasks_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.tasks_file)
        except Exception as exc:
            logging.exception("Failed to save agent tasks: %s", exc)

    def _load_orphaned_pending(self) -> None:
        if not self.pending_file or not os.path.exists(self.pending_file):
            self._orphaned_pending = []
            return
        try:
            with open(self.pending_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logging.exception("Failed to load pending questions snapshot from %s", self.pending_file)
            self._orphaned_pending = []
            return
        self._orphaned_pending = data if isinstance(data, list) else []

    def _save_pending_snapshot(self) -> None:
        if not self.pending_file:
            return
        snapshot: List[Dict[str, Any]] = []
        for question_id, pending in self.pending_questions.items():
            chat_id = pending.get("chat_id")
            message_id = pending.get("message_id")
            if chat_id is None or message_id is None:
                continue
            snapshot.append({
                "question_id": question_id,
                "chat_id": int(chat_id),
                "message_id": int(message_id),
            })
        try:
            pending_dir = os.path.dirname(self.pending_file) or "."
            os.makedirs(pending_dir, exist_ok=True)
            tmp_path = f"{self.pending_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.pending_file)
        except Exception:
            logging.exception("Failed to save pending questions snapshot")

    async def on_startup(self, application) -> None:
        if not self._orphaned_pending:
            return
        bot = getattr(application, "bot", None)
        if bot is None:
            return
        for record in self._orphaned_pending:
            chat_id = record.get("chat_id")
            message_id = record.get("message_id")
            if chat_id is None or message_id is None:
                continue
            await self._clear_question_markup(bot, int(chat_id), int(message_id))
            try:
                await bot.send_message(
                    chat_id=int(chat_id),
                    text="ℹ️ Бот был перезапущен — предыдущий вопрос больше не активен.",
                    reply_to_message_id=int(message_id),
                )
            except Exception:
                logging.debug(
                    "Failed to notify about orphaned pending question chat=%s message=%s",
                    chat_id, message_id, exc_info=True,
                )
        self._orphaned_pending = []
        try:
            if os.path.exists(self.pending_file):
                os.remove(self.pending_file)
        except OSError:
            logging.debug("Failed to remove pending questions snapshot", exc_info=True)

    def _scope_key(self, kwargs: Dict[str, Any]) -> str:
        chat_id = kwargs.get("chat_id")
        if chat_id is not None:
            return f"chat:{chat_id}"
        user_id = kwargs.get("user_id")
        if user_id is not None:
            return f"user:{user_id}"
        return "global"

    def _manage_plan_tasks(self, helper, **kwargs) -> Dict:
        action = str(kwargs.get("action") or "").strip()
        scope = self._scope_key(kwargs)
        tasks = self.tasks.setdefault(scope, [])

        if action == "add":
            items = kwargs.get("tasks") or []
            if not items:
                return {"success": False, "error": "No tasks provided"}
            for item in items:
                task_id = str(item.get("id") or "").strip()
                content = str(item.get("content") or "").strip()
                status = str(item.get("status") or "pending").strip()
                if not task_id or not content:
                    return {"success": False, "error": "Task requires id and content"}
                if status not in TASK_STATUSES:
                    return {"success": False, "error": f"Invalid task status: {status}"}
                existing = next((task for task in tasks if task.get("id") == task_id), None)
                if existing:
                    existing["content"] = content
                    existing["status"] = status
                    existing["updated_at"] = int(time.time())
                else:
                    tasks.append(
                        {
                            "id": task_id,
                            "content": content,
                            "status": status,
                            "created_at": int(time.time()),
                            "updated_at": int(time.time()),
                        }
                    )
            self.save_tasks()
            return self._tasks_response(action, tasks, changed=True)

        if action == "update":
            items = kwargs.get("tasks") or []
            if not items:
                return {"success": False, "error": "No tasks provided"}
            changed = False
            for item in items:
                task_id = str(item.get("id") or "").strip()
                existing = next((task for task in tasks if task.get("id") == task_id), None)
                if not existing:
                    continue
                item_changed = False
                if "content" in item:
                    content = str(item.get("content") or "").strip()
                    if content and existing.get("content") != content:
                        existing["content"] = content
                        item_changed = True
                if "status" in item:
                    status = str(item.get("status") or "").strip()
                    if status not in TASK_STATUSES:
                        return {"success": False, "error": f"Invalid task status: {status}"}
                    if existing.get("status") != status:
                        existing["status"] = status
                        item_changed = True
                if item_changed:
                    existing["updated_at"] = int(time.time())
                    changed = True
            if changed:
                self.save_tasks()
            return self._tasks_response(action, tasks, changed=changed)

        if action == "list":
            return self._tasks_response(action, tasks, changed=False)

        if action == "clear":
            active = [task for task in tasks if task.get("status") not in CLOSED_STATUSES]
            changed = len(active) != len(tasks)
            self.tasks[scope] = active
            if changed:
                self.save_tasks()
            return self._tasks_response(action, active, changed=changed)

        return {"success": False, "error": f"Unknown action: {action}"}

    def _tasks_response(self, action: str, tasks: List[Dict[str, Any]], *, changed: bool) -> Dict:
        snapshot = [
            {
                "id": str(task.get("id") or ""),
                "content": str(task.get("content") or ""),
                "status": str(task.get("status") or "pending"),
            }
            for task in tasks
        ]
        total = len(snapshot)
        closed = sum(1 for task in snapshot if task.get("status") in CLOSED_STATUSES)
        if snapshot:
            lines = [f"{task['id']} [{task['status']}]: {task['content']}" for task in snapshot]
            output = "\n".join(lines)
        else:
            output = "No plan tasks."
        return {
            "success": True,
            "output": output,
            "plan_tasks": {
                "action": action,
                "changed": changed,
                "tasks": snapshot,
                "progress": {
                    "total": total,
                    "closed": closed,
                    "open": total - closed,
                },
            },
        }

    async def _deliver_to_user(self, helper, **kwargs) -> Dict:
        text_raw = kwargs.get("text")
        final_text = "" if text_raw is None else str(text_raw).strip()
        artifact_items, error = self._normalize_delivery_artifacts(kwargs.get("artifacts"))
        if error:
            return {"success": False, "error": error}
        if not final_text and not artifact_items:
            return {
                "success": False,
                "error": "deliver_to_user requires at least one of text or artifacts",
            }

        cleanup_skills = self._cleanup_directives_for_active_skills(helper, kwargs)
        logging.info(
            "deliver_to_user text_chars=%s artifacts=%s cleanup_skills=%s chat_id=%s user_id=%s",
            len(final_text),
            len(artifact_items),
            len(cleanup_skills),
            kwargs.get("chat_id"),
            kwargs.get("user_id"),
        )
        direct_result: Dict[str, Any] = {
            "kind": "final",
            "format": "mixed",
            "defer": False,
            "text": final_text,
            "artifacts": artifact_items,
        }
        if cleanup_skills:
            direct_result["cleanup_skills"] = cleanup_skills
        return {
            "success": True,
            "text_chars": len(final_text),
            "artifacts_count": len(artifact_items),
            "direct_result": direct_result,
        }

    @staticmethod
    def _cleanup_directives_for_active_skills(helper, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        plugin_manager = getattr(helper, "plugin_manager", None)
        if plugin_manager is None:
            return []
        skills_plugin = None
        try:
            skills_plugin = plugin_manager.get_plugin("skills")
        except Exception:
            return []
        if skills_plugin is None:
            return []

        chat_id = kwargs.get("chat_id")
        user_id = kwargs.get("user_id")
        if chat_id is not None:
            scope = f"chat:{chat_id}"
        elif user_id is not None:
            scope = f"user:{user_id}"
        else:
            scope = "global"

        active_skills = getattr(skills_plugin, "active_skills", None) or {}
        scope_state = active_skills.get(scope) or {}
        plugin_id = getattr(skills_plugin, "plugin_id", "skills")
        return [
            {"plugin_id": plugin_id, "scope": scope, "skill_id": skill_id}
            for skill_id in scope_state
        ]

    @staticmethod
    def _normalize_delivery_artifacts(
        artifacts: Any,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        if artifacts is None:
            return [], None
        if isinstance(artifacts, str):
            text = artifacts.strip()
            if not text:
                return [], None
            try:
                artifacts = json.loads(text)
            except json.JSONDecodeError:
                artifacts = [{"file_path": text}]
        if not isinstance(artifacts, list):
            return [], "artifacts must be a list"

        items: List[Dict[str, Any]] = []
        for entry in artifacts:
            caption: str | None = None
            if isinstance(entry, str):
                file_path = entry
            elif isinstance(entry, dict):
                file_path = entry.get("file_path")
                raw_caption = entry.get("caption")
                if raw_caption is not None:
                    caption = str(raw_caption).strip() or None
            else:
                return [], "Each artifact must be an object with file_path or a path string"
            if not isinstance(file_path, str) or not file_path.strip():
                return [], "Artifact file_path is required"

            resolved = os.path.expanduser(file_path)
            if not os.path.exists(resolved):
                return [], f"Artifact file '{file_path}' does not exist"
            if not os.path.isfile(resolved):
                return [], f"Artifact path '{file_path}' is not a file"
            try:
                file_size = os.path.getsize(resolved)
            except OSError as exc:
                return [], f"Failed to stat artifact '{file_path}': {exc}"
            if file_size <= 0:
                return [], f"Artifact file '{file_path}' is empty"

            artifact: Dict[str, Any] = {
                "kind": "file",
                "format": "path",
                "value": resolved,
                "file_size": file_size,
            }
            if caption:
                artifact["caption"] = caption
            items.append(artifact)
        return items, None

    async def _run_subagents(self, helper, **kwargs) -> Dict:
        subagents = kwargs.get("subagents") or []
        if not isinstance(subagents, list) or not subagents:
            return {"success": False, "error": "subagents must be a non-empty list"}
        if len(subagents) > MAX_SUBAGENTS:
            return {"success": False, "error": f"At most {MAX_SUBAGENTS} subagents can run at once"}

        shared_context = str(kwargs.get("shared_context") or "").strip()
        parent_max_rounds = self._normalize_max_rounds(kwargs.get("max_rounds"))
        tasks = [
            self._run_one_subagent(helper, item, shared_context, kwargs, parent_max_rounds)
            for item in subagents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized_results = []
        for item, result in zip(subagents, results):
            subagent_id = str(item.get("id") or "subagent").strip()
            role = str(item.get("role") or "").strip()
            if isinstance(result, Exception):
                normalized_results.append({
                    "id": subagent_id,
                    "role": role,
                    "status": "error",
                    "error": str(result),
                })
            else:
                normalized_results.append(result)

        return {
            "success": True,
            "subagents": normalized_results,
        }

    async def _run_one_subagent(
        self,
        helper,
        item: Dict[str, Any],
        shared_context: str,
        kwargs: Dict[str, Any],
        parent_max_rounds: int,
    ) -> Dict:
        subagent_id = str(item.get("id") or "").strip()
        role = str(item.get("role") or "").strip()
        task = str(item.get("task") or "").strip()
        context = str(item.get("context") or "").strip()
        if not subagent_id or not role or not task:
            return {
                "id": subagent_id or "subagent",
                "role": role,
                "status": "error",
                "error": "Subagent requires id, role, and task",
            }

        user_id = kwargs.get("user_id")
        model_to_use = helper.get_current_model(user_id) if hasattr(helper, "get_current_model") else None
        model_to_use = model_to_use or getattr(helper, "model", None) or "llmgateway/high"

        max_rounds = self._normalize_max_rounds(item.get("max_rounds"), parent_max_rounds)
        temperature = self._normalize_temperature(item.get("temperature"))
        published: List[Dict[str, Any]] = []

        chat_id_log = kwargs.get("chat_id")
        user_id_log = kwargs.get("user_id")
        logging.info(
            "Subagent start id=%s role=%s model=%s max_rounds=%s temperature=%s chat_id=%s user_id=%s",
            subagent_id,
            role,
            model_to_use,
            max_rounds,
            temperature,
            chat_id_log,
            user_id_log,
        )
        started = time.monotonic()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bounded subagent. You may call available tools and skills to complete "
                    "your assigned subtask. Do not address the Telegram user, do not ask the user "
                    "questions, do not publish final artifacts, and do not start nested subagents. "
                    "When using skills, list/get/activate the relevant skill, run its scripts only "
                    "through skills.run_skill_script, and report outputs or artifact paths to the parent. "
                    "If a script belongs to an active skill, you MUST run it through skills.run_skill_script; "
                    "never reimplement or invoke that script through codeinterpreter.deep_analysis. "
                    "Return concise findings for the parent agent to synthesize."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Subagent id: {subagent_id}\n"
                    f"Role: {role}\n"
                    f"Task: {task}\n\n"
                    f"Shared context:\n{shared_context or '(none)'}\n\n"
                    f"Subagent-specific context:\n{context or '(none)'}"
                ),
            },
        ]
        status = "error"
        try:
            result_text = await self._run_subagent_completion_loop(
                helper, model_to_use, messages, kwargs, max_rounds, temperature,
                published=published,
            )
            status = "completed"
            return {
                "id": subagent_id,
                "role": role,
                "status": status,
                "result": result_text,
                "published": published,
            }
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            logging.info(
                "Subagent finish id=%s status=%s duration_ms=%s published=%s",
                subagent_id, status, duration_ms, len(published),
            )

    @staticmethod
    def _normalize_max_rounds(value: Any, default: int | None = None) -> int:
        baseline = default if default is not None else MIN_SUBAGENT_TOOL_ROUNDS
        if value is None:
            return baseline
        try:
            requested = int(value)
        except (TypeError, ValueError):
            return baseline
        return max(MIN_SUBAGENT_TOOL_ROUNDS, requested)

    @staticmethod
    def _normalize_temperature(value: Any) -> float:
        if value is None:
            return DEFAULT_SUBAGENT_TEMPERATURE
        try:
            requested = float(value)
        except (TypeError, ValueError):
            return DEFAULT_SUBAGENT_TEMPERATURE
        return max(0.0, min(requested, 2.0))

    async def _run_subagent_completion_loop(
        self,
        helper,
        model_to_use: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        max_rounds: int,
        temperature: float,
        *,
        published: List[Dict[str, Any]] | None = None,
    ) -> str:
        tools, allowed_function_names = self._subagent_tools(helper, model_to_use)

        for round_index in range(max_rounds + 1):
            is_final_round = round_index == max_rounds
            request_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
                "extra_headers": {"X-Title": "tgBot"},
            }
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "none" if is_final_round else "auto"

            response = await helper.client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            tool_calls = self._extract_tool_calls(choice)
            if tool_calls and not is_final_round:
                messages.append(self._assistant_tool_calls_message(choice, tool_calls))
                for call in tool_calls:
                    tool_response = await self._call_subagent_tool(
                        helper,
                        call,
                        allowed_function_names,
                        kwargs,
                        published=published,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._tool_result_content(helper, tool_response),
                    })
                continue

            text = self._choice_text(response)
            if text:
                return text
            if tool_calls:
                return (
                    f"Subagent reached the tool-call limit ({max_rounds}) "
                    "without producing a textual answer."
                )
            return ""

        return ""

    def _subagent_tools(self, helper, model_to_use: str) -> tuple[Any, set[str]]:
        plugin_manager = getattr(helper, "plugin_manager", None)
        if not plugin_manager:
            return None, set()

        tools = plugin_manager.get_functions_specs(helper, model_to_use, ["All"])
        if isinstance(tools, dict):
            logging.warning(
                "Subagents do not support Google-style tool specs (model=%s); running without tools",
                model_to_use,
            )
            return None, set()

        filtered_tools = []
        allowed_function_names = set()
        for tool in tools or []:
            function_spec = tool.get("function") or {}
            function_name = function_spec.get("name")
            if not function_name or function_name in SUBAGENT_BLOCKED_FUNCTIONS:
                continue
            filtered_tools.append(tool)
            allowed_function_names.add(function_name)
        filtered_tools.append(self._internal_publish_tool_spec())
        allowed_function_names.add("agent_tools.internal_publish")
        return filtered_tools, allowed_function_names

    @staticmethod
    def _internal_publish_tool_spec() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "agent_tools.internal_publish",
                "description": (
                    "Subagent-only: hand a finding or local artifact path back to the parent agent "
                    "without delivering anything to the Telegram user. The parent receives all "
                    "published items in the subagent's `published` list and can decide how to use them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["text", "file", "note"],
                            "description": "Item type: free-form text, an artifact file path, or a short note.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Text content or absolute file path, depending on kind.",
                        },
                        "caption": {
                            "type": "string",
                            "description": "Optional short caption that helps the parent pick the right item.",
                        },
                    },
                    "required": ["kind", "value"],
                },
            },
        }

    def _extract_tool_calls(self, choice) -> List[Dict[str, str]]:
        message = getattr(choice, "message", None)
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls = []
        for index, tool_call in enumerate(raw_tool_calls):
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", "") if function else ""
            if not name:
                continue
            tool_calls.append({
                "id": getattr(tool_call, "id", None) or f"sub_call_{index}",
                "name": name,
                "arguments": getattr(function, "arguments", "{}") or "{}",
            })
        return tool_calls

    def _assistant_tool_calls_message(self, choice, tool_calls: List[Dict[str, str]]) -> Dict[str, Any]:
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", None)
        if isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            normalized_content = "\n".join(part for part in parts if part) or None
        else:
            normalized_content = raw_content
        return {
            "role": "assistant",
            "content": normalized_content,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call.get("arguments") or "{}",
                    },
                }
                for call in tool_calls
            ],
        }

    async def _call_subagent_tool(
        self,
        helper,
        call: Dict[str, str],
        allowed_function_names: set[str],
        kwargs: Dict[str, Any],
        *,
        published: List[Dict[str, Any]] | None = None,
    ) -> str:
        tool_name = call["name"]
        if tool_name not in allowed_function_names:
            return json.dumps({"error": f"Tool {tool_name} is not available to subagents"}, ensure_ascii=False)

        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid arguments for {tool_name}"}, ensure_ascii=False)

        if tool_name == "agent_tools.internal_publish":
            return self._handle_internal_publish(args, published)

        if kwargs.get("chat_id") is not None:
            args["chat_id"] = kwargs.get("chat_id")
        args["user_id"] = (
            kwargs.get("user_id") if kwargs.get("user_id") is not None else kwargs.get("chat_id")
        )
        routing_error = _skill_script_routing_error(
            helper, kwargs.get("chat_id"), tool_name, args
        )
        if routing_error:
            return json.dumps(routing_error, ensure_ascii=False)
        return await helper.plugin_manager.call_function(
            tool_name,
            helper,
            json.dumps(args, ensure_ascii=False),
        )

    @staticmethod
    def _handle_internal_publish(
        args: Dict[str, Any],
        published: List[Dict[str, Any]] | None,
    ) -> str:
        if published is None:
            return json.dumps(
                {"error": "internal_publish is only available inside a subagent"},
                ensure_ascii=False,
            )
        kind = str(args.get("kind") or "").strip().lower()
        value = str(args.get("value") or "").strip()
        if kind not in {"text", "file", "note"}:
            return json.dumps({"error": "kind must be one of: text, file, note"}, ensure_ascii=False)
        if not value:
            return json.dumps({"error": "value is required"}, ensure_ascii=False)
        item: Dict[str, Any] = {"kind": kind, "value": value}
        caption = args.get("caption")
        if caption:
            item["caption"] = str(caption).strip()
        published.append(item)
        return json.dumps(
            {"success": True, "stored": True, "count": len(published)},
            ensure_ascii=False,
        )

    def _tool_result_content(self, helper, content: Any) -> str:
        formatter = getattr(helper, "_tool_result_content", None)
        if callable(formatter):
            return formatter(content)
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def _choice_text(self, response) -> str:
        try:
            content = response.choices[0].message.content
        except Exception:
            return ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content or "")

    async def _ask_telegram_user(self, helper, **kwargs) -> Dict:
        question = str(kwargs.get("question") or "").strip()
        if not question:
            return {"success": False, "error": "Question is required"}

        chat_id = kwargs.get("chat_id")
        if chat_id is None:
            return {"success": False, "error": "chat_id is required for Telegram interaction"}
        chat_id = int(chat_id)

        bot = getattr(helper, "bot", None) or getattr(getattr(self, "openai", None), "bot", None)
        if not bot:
            return {"success": False, "error": "Telegram bot is not configured"}

        self._clear_done_question_for_chat(chat_id)
        if chat_id in self.pending_by_chat:
            return {"success": False, "error": "A Telegram question is already waiting for this chat"}

        options = self._normalize_options(kwargs.get("options") or [])
        if not options:
            return {"success": False, "error": "options are required for ask_telegram_user"}
        multi_select = bool(kwargs.get("multi_select", False))
        allow_free_text = bool(kwargs.get("allow_free_text", True))
        if multi_select:
            allow_free_text = False
        timeout = self._normalize_timeout(kwargs.get("timeout_seconds"))
        user_id = self._normalize_optional_int(kwargs.get("user_id"))
        question_id = f"q{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending_questions[question_id] = {
            "future": future,
            "chat_id": chat_id,
            "user_id": user_id,
            "question": question,
            "options": options,
            "allow_free_text": allow_free_text,
            "multi_select": multi_select,
            "selected_indices": set(),
            "created_at": int(time.time()),
            "bot": bot,
            "message_id": None,
        }
        self.pending_by_chat[chat_id] = question_id

        try:
            sent_message = await bot.send_message(
                chat_id=chat_id,
                text=question,
                reply_markup=self._question_markup(
                    question_id, options, allow_free_text, multi_select=multi_select,
                ),
            )
            pending = self.pending_questions.get(question_id)
            if pending is not None:
                pending["message_id"] = getattr(sent_message, "message_id", None)
                self._save_pending_snapshot()
            answer = await asyncio.wait_for(future, timeout=timeout)
            return {
                "success": True,
                "answer": answer,
                "output": f"User answered: {answer}",
            }
        except asyncio.TimeoutError:
            pending = self.pending_questions.get(question_id) or {}
            message_id = pending.get("message_id")
            if message_id is not None:
                await self._clear_question_markup(bot, chat_id, message_id)
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="⏱ Время ответа истекло, вопрос закрыт.",
                        reply_to_message_id=message_id,
                    )
                except Exception:
                    logging.debug(
                        "Failed to send ask_telegram_user timeout notification",
                        exc_info=True,
                    )
            return {"success": False, "error": "Timed out waiting for Telegram user answer"}
        finally:
            self._drop_question(question_id)

    def _normalize_options(self, options: Any) -> List[str]:
        normalized: List[str] = []
        for option in options if isinstance(options, list) else []:
            text = str(option or "").strip()
            if text and text not in normalized:
                normalized.append(text)
            if len(normalized) >= 6:
                break
        return normalized

    @staticmethod
    def _normalize_timeout(value: Any) -> int:
        default = int(os.getenv("AGENT_ASK_USER_TIMEOUT_SECONDS", "1800"))
        try:
            timeout = int(value) if value is not None else default
        except (TypeError, ValueError):
            timeout = default
        return max(1, min(timeout, 86400))

    @staticmethod
    def _normalize_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _question_markup(
        self,
        question_id: str,
        options: List[str],
        allow_free_text: bool,
        *,
        multi_select: bool = False,
        selected_indices: set | None = None,
    ) -> InlineKeyboardMarkup | None:
        selected = selected_indices or set()
        rows = []
        for index, option in enumerate(options):
            label = option
            if multi_select:
                label = f"✅ {option}" if index in selected else f"☐ {option}"
            rows.append(
                [InlineKeyboardButton(label, callback_data=f"agentask:{question_id}:{index}")]
            )
        if multi_select and options:
            rows.append([InlineKeyboardButton("Confirm", callback_data=f"agentask:{question_id}:confirm")])
        elif allow_free_text and options:
            rows.append([InlineKeyboardButton("Reply with text", callback_data=f"agentask:{question_id}:text")])
        return InlineKeyboardMarkup(rows) if rows else None

    def is_waiting_for_text(self, chat_id: Any, user_id: Any = None) -> bool:
        if chat_id is None:
            return False
        self._clear_done_question_for_chat(int(chat_id))
        question_id = self.pending_by_chat.get(int(chat_id))
        if not question_id:
            return False
        pending = self.pending_questions.get(question_id) or {}
        return bool(pending.get("allow_free_text")) and self._answerer_allowed(pending, user_id)

    async def handle_text_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text:
            return
        chat_id = int(message.chat.id)
        question_id = self.pending_by_chat.get(chat_id)
        if not question_id:
            return
        pending = self.pending_questions.get(question_id)
        if not pending or not pending.get("allow_free_text"):
            return
        user = getattr(message, "from_user", None)
        if not self._answerer_allowed(pending, getattr(user, "id", None)):
            return
        answer = message.text.strip()
        if not answer:
            return
        if self._resolve_question(question_id, answer):
            await message.reply_text("Answer received.")

    async def handle_ask_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query:
            return
        data = str(query.data or "")
        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer()
            return
        _, question_id, answer_key = parts
        pending = self.pending_questions.get(question_id)
        if not pending:
            await query.answer("Question expired.", show_alert=True)
            return
        user = getattr(query, "from_user", None)
        if not self._answerer_allowed(pending, getattr(user, "id", None)):
            await query.answer("This question is waiting for another user.", show_alert=True)
            return
        if answer_key == "text":
            await query.answer("Send your answer as a message.")
            return

        options = pending.get("options") or []
        multi_select = bool(pending.get("multi_select"))

        if multi_select and answer_key == "confirm":
            selected = pending.get("selected_indices") or set()
            if not selected:
                await query.answer("Select at least one option.", show_alert=True)
                return
            chosen = [str(options[i]) for i in sorted(selected) if 0 <= i < len(options)]
            answer = ", ".join(chosen)
            if not self._resolve_question(question_id, answer):
                await query.answer("Question expired.", show_alert=True)
                return
            await query.answer("Answer received.")
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                logging.debug("Failed to clear ask_user markup", exc_info=True)
            return

        try:
            index = int(answer_key)
            if not 0 <= index < len(options):
                raise IndexError
        except (ValueError, IndexError):
            await query.answer("Invalid answer.", show_alert=True)
            return

        if multi_select:
            selected = pending.setdefault("selected_indices", set())
            if index in selected:
                selected.discard(index)
            else:
                selected.add(index)
            await query.answer()
            try:
                await query.edit_message_reply_markup(
                    reply_markup=self._question_markup(
                        question_id,
                        options,
                        bool(pending.get("allow_free_text")),
                        multi_select=True,
                        selected_indices=selected,
                    )
                )
            except Exception:
                logging.debug("Failed to update multi-select ask_user markup", exc_info=True)
            return

        answer = str(options[index])
        if not self._resolve_question(question_id, answer):
            await query.answer("Question expired.", show_alert=True)
            return
        await query.answer("Answer received.")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logging.debug("Failed to clear ask_user markup", exc_info=True)

    async def _cancel_pending_question(self, helper, **kwargs) -> Dict:
        chat_id = kwargs.get("chat_id")
        if chat_id is None:
            return {"success": False, "error": "chat_id is required"}
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return {"success": False, "error": "Invalid chat_id"}

        question_id = self.pending_by_chat.get(chat_id_int)
        if not question_id:
            return {"success": False, "error": "No pending question for this chat"}
        pending = self.pending_questions.get(question_id) or {}
        future = pending.get("future")
        if future and not future.done():
            future.cancel()
        bot = pending.get("bot")
        message_id = pending.get("message_id")
        reason = str(kwargs.get("reason") or "").strip()
        if bot and message_id is not None:
            await self._clear_question_markup(bot, chat_id_int, message_id)
            if reason:
                try:
                    await bot.send_message(
                        chat_id=chat_id_int,
                        text=f"❌ Вопрос отменён: {reason}",
                        reply_to_message_id=message_id,
                    )
                except Exception:
                    logging.debug("Failed to send cancel notice", exc_info=True)
        self._drop_question(question_id)
        return {"success": True, "cancelled": True}

    def _answerer_allowed(self, pending: Dict[str, Any], user_id: Any) -> bool:
        expected_user_id = pending.get("user_id")
        if expected_user_id is None:
            return True
        actual_user_id = self._normalize_optional_int(user_id)
        return actual_user_id == expected_user_id

    def _resolve_question(self, question_id: str, answer: str) -> bool:
        pending = self.pending_questions.get(question_id)
        if not pending:
            return False
        future = pending.get("future")
        if not future or future.done():
            return False
        future.set_result(answer)
        return True

    def _drop_question(self, question_id: str) -> None:
        pending = self.pending_questions.pop(question_id, None)
        if not pending:
            return
        chat_id = pending.get("chat_id")
        if chat_id is not None and self.pending_by_chat.get(int(chat_id)) == question_id:
            self.pending_by_chat.pop(int(chat_id), None)
        self._save_pending_snapshot()

    def _clear_done_question_for_chat(self, chat_id: int) -> None:
        question_id = self.pending_by_chat.get(chat_id)
        if not question_id:
            return
        pending = self.pending_questions.get(question_id)
        future = (pending or {}).get("future")
        if pending is None or future is None or future.done():
            self._drop_question(question_id)
