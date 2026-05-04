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

from .plugin import Plugin


CLOSED_STATUSES = {"completed", "cancelled"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


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
        self.pending_questions: Dict[str, Dict[str, Any]] = {}
        self.pending_by_chat: Dict[int, str] = {}
        self.load_tasks()

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            self.tasks_file = os.path.join(storage_root, "agent_tasks.json")
            self.load_tasks()

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
                    "Use when explicit confirmation, a choice, or missing information is required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to send to the user."},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional button labels for likely answers.",
                        },
                        "allow_free_text": {
                            "type": "boolean",
                            "description": "Whether the user may answer with a normal Telegram text message.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "How long to wait for the user's answer. Defaults to 1800.",
                        },
                    },
                    "required": ["question"],
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
        if function_name == "manage_plan_tasks":
            return self._manage_plan_tasks(helper, **kwargs)
        if function_name == "ask_telegram_user":
            return await self._ask_telegram_user(helper, **kwargs)
        return {"success": False, "error": f"Unknown agent tool: {function_name}"}

    def load_tasks(self) -> None:
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tasks = data if isinstance(data, dict) else {}
            else:
                self.tasks = {}
        except Exception as exc:
            logging.exception("Failed to load agent tasks: %s", exc)
            self.tasks = {}

    def save_tasks(self) -> None:
        try:
            with open(self.tasks_file, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.exception("Failed to save agent tasks: %s", exc)

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
        allow_free_text = bool(kwargs.get("allow_free_text", True))
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
            "created_at": int(time.time()),
        }
        self.pending_by_chat[chat_id] = question_id

        try:
            await bot.send_message(
                chat_id=chat_id,
                text=question,
                reply_markup=self._question_markup(question_id, options, allow_free_text),
            )
            answer = await asyncio.wait_for(future, timeout=timeout)
            return {
                "success": True,
                "answer": answer,
                "output": f"User answered: {answer}",
            }
        except asyncio.TimeoutError:
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
    ) -> InlineKeyboardMarkup | None:
        rows = [
            [InlineKeyboardButton(option, callback_data=f"agentask:{question_id}:{index}")]
            for index, option in enumerate(options)
        ]
        if allow_free_text and options:
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
        try:
            index = int(answer_key)
            answer = str(options[index])
        except (ValueError, IndexError):
            await query.answer("Invalid answer.", show_alert=True)
            return

        if not self._resolve_question(question_id, answer):
            await query.answer("Question expired.", show_alert=True)
            return
        await query.answer("Answer received.")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logging.debug("Failed to clear ask_user markup", exc_info=True)

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

    def _clear_done_question_for_chat(self, chat_id: int) -> None:
        question_id = self.pending_by_chat.get(chat_id)
        if not question_id:
            return
        pending = self.pending_questions.get(question_id)
        future = (pending or {}).get("future")
        if pending is None or future is None or future.done():
            self._drop_question(question_id)
