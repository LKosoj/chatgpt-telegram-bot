from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, MessageHandler, filters

from ..agent_delivery import send_agent_response, send_text_chunks
from ..model_constants import LLMGATEWAY_CHAT_MODELS
from ..request_context import RequestContext
from ..skill_script_routing import _skill_script_routing_error
from ..utils import compute_scope_key, get_thread_id, message_text
from .hooks import BeforeChatRequestPayload
from .background import BackgroundTask
from .plugin import Plugin


CLOSED_STATUSES = {"completed", "cancelled"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled", "blocked"}
BACKGROUND_CLOSED_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
GOAL_RUN_CLOSED_STATUSES = {
    "completed",
    "blocked",
    "failed",
    "cancelled",
    "interrupted",
    "budget_exhausted",
}
GOAL_RUN_DEFAULT_LIMITS = {
    "max_runtime_seconds": 1800,
}
GOAL_RUN_MAX_CONCURRENCY = 2
DELIVERY_ACTION_WORDS = (
    "attach",
    "deliver",
    "return",
    "send",
    "верн",
    "достав",
    "отправ",
    "переда",
    "прикреп",
)
DELIVERY_TARGET_WORDS = (
    "artifact",
    "document",
    "file",
    "pptx",
    "presentation",
    "user",
    "артефакт",
    "документ",
    "пользовател",
    "презентац",
    "файл",
)
MAX_SUBAGENTS = 5
MIN_SUBAGENT_TOOL_ROUNDS = 10
MAX_SUBAGENT_TOOL_ROUNDS = 50
DEFAULT_SUBAGENT_TEMPERATURE = 0.2
SUBAGENT_CONSECUTIVE_REPEAT_LIMIT = 3
SUBAGENT_TOTAL_REPEAT_LIMIT = 5
SUBAGENT_BLOCKED_FUNCTIONS = {
    "agent_tools.manage_plan_tasks",
    "agent_tools.update_working_checkpoint",
    "agent_tools.ask_telegram_user",
    "agent_tools.cancel_pending_question",
    "agent_tools.deliver_to_user",
    "agent_tools.run_subagents",
    "agent_tools.manage_goal_runs",
    "skills.find_installable_skills",
    "skills.install_skill",
    "skills.create_skill",
    "skills.run_skill_agent",
    "skills.update_skill_progress",
    "skills.deactivate_skill",
}
_CONTRACT_UNSET = object()

_SUBAGENT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "subagent_system.md"
_SUBAGENT_SYSTEM_PROMPT_CACHE: str | None = None

_REPLAN_TRIGGER_MARKER = "[re-plan trigger-v1] "
_REPLAN_ERROR_THRESHOLD = 2

_PLAN_RULE_MARKER = "[plan-rule-v1] "
_WORKING_CHECKPOINT_MARKER = "[working-checkpoint-v1] "
_PLAN_RULE_TEXT = (
    "Если для выполнения запроса понадобятся 3+ tool-вызова или последовательная координация "
    "шагов (несколько разных тулов, проверка промежуточных результатов, исправление ошибок) — "
    "ПЕРВЫМ ходом вызови agent_tools.manage_plan_tasks с goal/success_criteria/verification и "
    "зафиксируй план. Для тривиальных запросов (быстрый ответ из знаний, один очевидный тул, "
    "перевод/время/факт) план НЕ создавай."
)
_CHECKPOINT_MAX_TEXT_CHARS = 1600
_CHECKPOINT_MAX_LIST_ITEMS = 8
_MAP_REDUCE_STRATEGIES = {"synthesize", "compare", "rank", "deduplicate"}


def _load_subagent_system_prompt() -> str:
    global _SUBAGENT_SYSTEM_PROMPT_CACHE
    if _SUBAGENT_SYSTEM_PROMPT_CACHE is None:
        try:
            _SUBAGENT_SYSTEM_PROMPT_CACHE = _SUBAGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logging.error("Subagent system prompt missing at %s: %s", _SUBAGENT_PROMPT_PATH, exc)
            _SUBAGENT_SYSTEM_PROMPT_CACHE = (
                "You are a bounded subagent. Use available tools to complete the assigned task and "
                "return concise findings. Do not address the user directly."
            )
    return _SUBAGENT_SYSTEM_PROMPT_CACHE


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

    DELIVERY_DEDUP_WINDOW_SECONDS = 60
    DELIVERY_MAX_ARTIFACT_BYTES = 49 * 1024 * 1024
    TASKS_TTL_SECONDS = 2 * 24 * 3600

    def __init__(self):
        self.db = None
        self.db_handle = None
        self.pending_file = os.path.join(os.path.dirname(__file__), "agent_pending_questions.json")
        self.background_jobs_file = os.path.join(os.path.dirname(__file__), "agent_background_jobs.json")
        self.pending_questions: Dict[str, Dict[str, Any]] = {}
        self.pending_by_chat: Dict[int, str] = {}
        self.background_jobs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._background_job_tasks: Dict[str, asyncio.Task] = {}
        self._goal_run_tasks: Dict[str, asyncio.Task] = {}
        self._orphaned_pending: List[Dict[str, Any]] = []
        self._recent_deliveries: Dict[str, float] = {}
        self._background_subagent_tasks: set[asyncio.Task] = set()
        # Re-plan trigger state. Keyed by scope (compute_scope_key result).
        # _tool_error_streaks tracks consecutive failures for the SAME task_id.
        # _pending_replan is a one-shot inject flag consumed by on_before_chat_request.
        self._tool_error_streaks: Dict[str, Dict[str, Any]] = {}
        self._pending_replan: Dict[str, Dict[str, Any]] = {}
        self._replan_lock: Optional[asyncio.Lock] = None
        self._load_orphaned_pending()
        self._load_background_jobs()

    def initialize(self, openai=None, bot=None, storage_root: str | None = None,
                   db=None, plugin_config=None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        runtime_db = (
            getattr(openai, "db", None)
            or getattr(bot, "db", None)
            or getattr(db, "database", None)
        )
        self.db_handle = db  # async DbHandle facade from Stage 0 shim (may be None)
        self.db = None
        if storage_root:
            self.pending_file = os.path.join(storage_root, "agent_pending_questions.json")
            self.background_jobs_file = os.path.join(storage_root, "agent_background_jobs.json")
            self._load_orphaned_pending()
            self._load_background_jobs()
        self.db = runtime_db
        if self.db is not None:
            self._prune_stale_tasks()

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
        for task in list(self._background_subagent_tasks):
            task.cancel()
        self._background_subagent_tasks.clear()
        for task in list(self._background_job_tasks.values()):
            task.cancel()
        self._background_job_tasks.clear()
        for task in list(self._goal_run_tasks.values()):
            task.cancel()
        self._goal_run_tasks.clear()

    async def close_async(self) -> None:
        tasks = [task for task in self._goal_run_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._goal_run_tasks.clear()

    def get_background_tasks(self) -> List[BackgroundTask]:
        return [
            BackgroundTask(
                name="goal_runs",
                interval_seconds=1.0,
                coroutine_factory=self._goal_runs_tick,
            )
        ]

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

    async def on_before_chat_request(
        self,
        messages: List[Dict[str, Any]],
        payload: BeforeChatRequestPayload,
    ) -> Optional[List[Dict[str, Any]]]:
        """Inject a system-level reminder to call ``manage_plan_tasks`` for non-trivial work.

        Skipped when ``agent_tools`` is not in the active mode's allow-list, when the
        marker is already present (idempotency), or when the leading system prompt
        already mentions ``manage_plan_tasks`` (mode prompt covers the rule itself).

        Also delivers a one-shot re-plan trigger system message scheduled by
        :meth:`_schedule_replan`. The trigger is popped under a lock so concurrent
        chat requests cannot duplicate the inject.

        Returns ``None`` for "no change".
        """
        helper = getattr(self, "openai", None)
        if helper is None:
            return None
        chat_id = getattr(payload, "chat_id", None)
        if chat_id is None:
            return None
        user_id = getattr(payload, "user_id", None)

        # Pop pending re-plan trigger ONCE per scope; idempotent.
        scope = compute_scope_key(chat_id, user_id)
        pending = None
        try:
            async with self._get_replan_lock():
                pending = self._pending_replan.pop(scope, None)
        except Exception:
            logging.debug("agent_tools: replan-pop failed", exc_info=True)
            pending = None

        # Cheap idempotency / mode-prompt checks BEFORE the SQL-backed
        # ``resolve_allowed_plugins`` call: this mutator runs on the hot path
        # of every chat request, so short-circuit before touching the DB
        # whenever possible.
        plan_rule_present = False
        checkpoint_present = False
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.startswith(_PLAN_RULE_MARKER):
                plan_rule_present = True
            if isinstance(content, str) and content.startswith(_WORKING_CHECKPOINT_MARKER):
                checkpoint_present = True
            if plan_rule_present and checkpoint_present:
                break

        mode_prompt_covers_rule = False
        first_system = next(
            (m for m in messages
             if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        if first_system is not None:
            content = first_system.get("content")
            if isinstance(content, str) and "manage_plan_tasks" in content:
                mode_prompt_covers_rule = True

        inject_plan_rule = not (plan_rule_present or mode_prompt_covers_rule)
        allowed = None
        agent_tools_allowed = False
        if inject_plan_rule or (self.db is not None and not checkpoint_present):
            try:
                allowed = await helper.resolve_allowed_plugins(
                    chat_id, session_id=None, user_id=user_id,
                )
            except Exception:
                logging.debug(
                    "agent_tools plan-rule: resolve_allowed_plugins failed",
                    exc_info=True,
                )
                allowed = None
                inject_plan_rule = False
            if allowed is not None:
                plugin_id = self.get_plugin_id()
                agent_tools_allowed = allowed == ['All'] or plugin_id in (allowed or [])
                if not agent_tools_allowed:
                    inject_plan_rule = False

        checkpoint = None
        if agent_tools_allowed and self.db is not None and not checkpoint_present:
            try:
                checkpoint = await asyncio.to_thread(self._db_get_checkpoint, scope)
            except Exception:
                logging.debug("agent_tools: checkpoint read failed", exc_info=True)
                checkpoint = None

        if not inject_plan_rule and not pending and not checkpoint:
            return None

        new_messages = list(messages)
        insert_at = 0
        for msg in new_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                insert_at += 1
            else:
                break
        injected = 0
        if inject_plan_rule:
            new_messages.insert(insert_at, {
                "role": "system",
                "content": _PLAN_RULE_MARKER + _PLAN_RULE_TEXT,
            })
            injected += 1
        if checkpoint:
            new_messages.insert(insert_at + injected, {
                "role": "system",
                "content": _WORKING_CHECKPOINT_MARKER + self._format_working_checkpoint(checkpoint),
            })
            injected += 1
        if pending:
            reason = str(pending.get("reason") or "errors")
            task_id = str(pending.get("task_id") or "")
            body = self._replan_message_body(reason, task_id)
            new_messages.insert(insert_at + injected, {
                "role": "system",
                "content": body,
            })
        return new_messages

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "manage_plan_tasks",
                "description": (
                    "Manage the current Telegram chat's durable task plan and Definition of Done. "
                    "Call before starting work that needs more than two steps: record goal, success "
                    "criteria, verification checks, and the task list. Skip the durable plan for "
                    "one- or two-step requests. Keep task ids stable: update existing tasks instead "
                    "of inserting semantic duplicates. Use depends_on to model a task DAG; a task may "
                    "start only after its dependencies close. Without depends_on, preserve list order "
                    "and do not start later tasks while earlier tasks are still pending. Only one "
                    "task may be in_progress."
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
                            "description": "Tasks for add/update actions. Include depends_on when one task blocks another.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Stable task id, for example T1."},
                                    "content": {"type": "string", "description": "Task text."},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed", "cancelled", "blocked"],
                                    },
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Task ids that must be completed or cancelled before this task starts.",
                                    },
                                },
                            },
                        },
                        "definition_of_done": {
                            "type": "object",
                            "description": (
                                "Required when adding tasks to a durable plan: goal, success criteria, "
                                "and verification checks. Store it before or with the task plan."
                            ),
                            "properties": {
                                "goal": {"type": "string"},
                                "success_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "verification": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "constraints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["goal", "success_criteria", "verification"],
                        },
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "ask_telegram_user",
                "description": (
                    "Send the Telegram user a question with inline-button choices and wait for their reply. "
                    "Call when explicit confirmation, a choice between alternatives, or missing user "
                    "input is required before proceeding. Answer variants go in options, not in the "
                    "question text."
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
                "name": "update_working_checkpoint",
                "description": (
                    "Store, inspect, or clear a short working checkpoint for the current agent task. "
                    "Use it after meaningful progress, verified findings, or a change of next step. "
                    "This is operational scratch state, not long-term memory; do not store secrets, "
                    "credentials, or unverified user-private data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["update", "list", "clear"],
                            "description": "Update, list, or clear the working checkpoint.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short current-state summary based on verified progress.",
                        },
                        "current_task_id": {
                            "type": "string",
                            "description": "Optional current durable plan task id, for example T2.",
                        },
                        "next_step": {
                            "type": "string",
                            "description": "Concrete next action to take.",
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Short evidence pointers such as file:line, command results, or tool outputs.",
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Known blockers or risks that affect the next turn.",
                        },
                        "files_touched": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Local files changed or inspected for this task.",
                        },
                        "verification_status": {
                            "type": "string",
                            "description": "Concise status of verification already done or still needed.",
                        },
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "cancel_pending_question",
                "description": (
                    "Cancel the currently pending ask_telegram_user question in this Telegram chat and "
                    "clear its inline keyboard. Call when the question is no longer needed because the "
                    "answer was inferred from context or the plan changed."
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
                    "Publish the agent's final answer to the Telegram user as one message with optional "
                    "text and zero or more local files, ending the agent loop. Call exactly once when "
                    "the task is finished or cannot continue; this is the only sanctioned way to address "
                    "the user — do not produce a plain assistant reply alongside it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Optional final text shown to the user.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["completed", "blocked"],
                            "description": (
                                "Machine-readable final state. Use completed when the task is done; "
                                "use blocked when work cannot continue or finish."
                            ),
                        },
                        "verification_summary": {
                            "type": "string",
                            "description": (
                                "Concise summary of checks performed before delivery, for example "
                                "validated generated files or reviewed tool outputs."
                            ),
                        },
                        "blocked_reason": {
                            "type": "string",
                            "description": "Required when status is blocked; explain the concrete blocker.",
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
                    "Fan out independent, bounded subtasks to tool-capable subagents that run in parallel "
                    "and return structured findings to the parent. Call when the work has two or more "
                    "independent investigation branches and you want them executed concurrently with their "
                    "own tool budgets. Subagents cannot ask Telegram questions, publish final artifacts, "
                    "or spawn nested subagents; give each one an explicit output contract (what to inspect, "
                    "what evidence to return, what files it may touch)."
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
                        "background": {
                            "type": "boolean",
                            "description": (
                                "When true, schedule subagents and return immediately. Use only for non-user-facing "
                                "background reflection or cleanup work."
                            ),
                            "default": False,
                        },
                        "map_reduce": {
                            "type": "object",
                            "description": (
                                "Optional map-reduce contract. Use when subagents are parallel map workers "
                                "and the parent will synthesize, compare, rank, or deduplicate their findings."
                            ),
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "description": "Set false to explicitly disable the contract.",
                                },
                                "reduce_goal": {
                                    "type": "string",
                                    "description": "Question or objective the parent reducer must answer from worker findings.",
                                },
                                "merge_strategy": {
                                    "type": "string",
                                    "enum": ["synthesize", "compare", "rank", "deduplicate"],
                                },
                                "worker_output_contract": {
                                    "type": "object",
                                    "properties": {
                                        "summary": {"type": "string"},
                                        "required_sections": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "evidence_required": {"type": "boolean"},
                                        "risk_required": {"type": "boolean"},
                                    },
                                },
                            },
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
                                    "map_key": {
                                        "type": "string",
                                        "description": "Optional stable key for this worker's map partition.",
                                    },
                                    "expected_output": {
                                        "type": "string",
                                        "description": "Optional worker-specific output format or required sections.",
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
                                    "model": {
                                        "type": "string",
                                        "enum": list(LLMGATEWAY_CHAT_MODELS),
                                        "description": (
                                            "Optional model override for this subagent only. "
                                            "Pick a lighter model for parsing/routing/lookup work and a heavier model "
                                            "for reasoning. Defaults to the parent's current model."
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
            {
                "name": "manage_goal_runs",
                "description": (
                    "Start, list, inspect, or cancel controlled background goal runs for this Telegram chat. "
                    "Use for long-running objectives that should continue outside the current request. "
                    "Runs are DB-backed, scoped to the chat/user, and have explicit runtime/token limits."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "list", "status", "cancel", "clear"],
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Goal prompt for action=start.",
                        },
                        "definition_of_done": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string"},
                                "success_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "verification": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "constraints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "limits": {
                            "type": "object",
                            "properties": {
                                "max_runtime_seconds": {"type": "integer"},
                                "token_budget": {"type": "integer"},
                            },
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Run id for status/cancel.",
                        },
                    },
                    "required": ["action"],
                },
            },
        ]

    def get_commands(self) -> List[Dict]:
        return [
            {
                "command": "background",
                "description": "Run an agent task in the background and deliver the result to this chat.",
                "handler": self.handle_background_command,
                "handler_kwargs": {},
                "add_to_menu": True,
            },
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

    async def handle_background_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        if not message:
            return
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else chat_id
        args_text = message_text(message).strip()
        if not args_text or args_text == "list":
            await message.reply_text(self._format_background_jobs(chat_id, user_id))
            return

        action, _, rest = args_text.partition(" ")
        action = action.lower().strip()
        if action == "status":
            await message.reply_text(
                self._format_background_job_status(chat_id, user_id, rest.strip()),
            )
            return
        if action == "cancel":
            await message.reply_text(
                await self._cancel_background_job(chat_id, user_id, rest.strip()),
                parse_mode="Markdown",
            )
            return
        if action == "clear":
            await message.reply_text(self._clear_background_jobs(chat_id, user_id), parse_mode="Markdown")
            return

        job = self._create_background_job(
            chat_id=chat_id,
            user_id=user_id,
            prompt=args_text,
            reply_to_message_id=message.message_id,
            message_thread_id=get_thread_id(update),
        )
        task = context.application.create_task(
            self._run_background_job(context.bot, job["scope"], job["id"]),
            update=update,
        )
        self._background_job_tasks[job["id"]] = task
        task.add_done_callback(lambda _task, job_id=job["id"]: self._background_job_tasks.pop(job_id, None))
        await message.reply_text(
            (
                f"Background job `{job['id']}` started.\n"
                f"Use `/background status {job['id']}` or `/background cancel {job['id']}`."
            ),
            parse_mode="Markdown",
        )

    def _background_scope(self, chat_id: int | None, user_id: int | None) -> str:
        return compute_scope_key(chat_id=chat_id, user_id=user_id)

    def _load_background_jobs(self) -> None:
        try:
            if not os.path.exists(self.background_jobs_file):
                self.background_jobs = {}
                return
            with open(self.background_jobs_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.background_jobs = data if isinstance(data, dict) else {}
            changed = False
            for jobs in self.background_jobs.values():
                if not isinstance(jobs, dict):
                    continue
                for job in jobs.values():
                    if isinstance(job, dict) and job.get("status") in {"queued", "running"}:
                        job["status"] = "interrupted"
                        job["finished_at"] = int(time.time())
                        changed = True
            if changed:
                self._save_background_jobs()
        except Exception:
            logging.exception("Failed to load agent background jobs")
            self.background_jobs = {}

    def _save_background_jobs(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.background_jobs_file), exist_ok=True)
            with open(self.background_jobs_file, "w", encoding="utf-8") as fh:
                json.dump(self.background_jobs, fh, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("Failed to save agent background jobs")

    def _create_background_job(
        self,
        *,
        chat_id: int,
        user_id: int,
        prompt: str,
        reply_to_message_id: int | None,
        message_thread_id: int | None,
    ) -> Dict[str, Any]:
        scope = self._background_scope(chat_id, user_id)
        job_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
        job = {
            "id": job_id,
            "scope": scope,
            "chat_id": chat_id,
            "user_id": user_id,
            "prompt": prompt,
            "status": "queued",
            "created_at": int(time.time()),
            "reply_to_message_id": reply_to_message_id,
            "message_thread_id": message_thread_id,
        }
        self.background_jobs.setdefault(scope, {})[job_id] = job
        self._save_background_jobs()
        return job

    def _get_background_job(self, scope: str, job_id: str) -> Dict[str, Any] | None:
        if not job_id:
            return None
        return (self.background_jobs.get(scope) or {}).get(job_id)

    def _format_background_jobs(self, chat_id: int, user_id: int) -> str:
        scope = self._background_scope(chat_id, user_id)
        jobs = list((self.background_jobs.get(scope) or {}).values())
        if not jobs:
            return (
                "No background jobs yet.\n\n"
                "Usage:\n"
                "`/background summarize the latest uploaded document`\n"
                "`/background status <job_id>`\n"
                "`/background cancel <job_id>`"
            )
        jobs.sort(key=lambda item: item.get("created_at", 0), reverse=True)
        lines = ["Background jobs:"]
        for job in jobs[:10]:
            prompt = str(job.get("prompt") or "").replace("\n", " ")
            if len(prompt) > 80:
                prompt = prompt[:77] + "..."
            lines.append(f"- `{job.get('id')}` {job.get('status')}: {prompt}")
        lines.append("\nUse `/background clear` to remove closed jobs from this list.")
        return "\n".join(lines)

    def _format_background_job_status(self, chat_id: int, user_id: int, job_id: str) -> str:
        scope = self._background_scope(chat_id, user_id)
        job = self._get_background_job(scope, job_id)
        if not job:
            return "Background job not found."
        lines = [
            f"Background job `{job['id']}`",
            f"Status: {job.get('status')}",
            f"Prompt: {job.get('prompt')}",
        ]
        if job.get("result_preview"):
            lines.append(f"Result preview: {job['result_preview']}")
        if job.get("error"):
            lines.append(f"Error: {job['error']}")
        return "\n".join(lines)

    async def _cancel_background_job(self, chat_id: int, user_id: int, job_id: str) -> str:
        scope = self._background_scope(chat_id, user_id)
        job = self._get_background_job(scope, job_id)
        if not job:
            return "Background job not found."
        if job.get("status") in BACKGROUND_CLOSED_STATUSES:
            return f"Background job `{job_id}` is already {job.get('status')}."
        job["status"] = "cancelled"
        job["finished_at"] = int(time.time())
        task = self._background_job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        self._save_background_jobs()
        return f"Background job `{job_id}` cancelled."

    def _clear_background_jobs(self, chat_id: int, user_id: int) -> str:
        scope = self._background_scope(chat_id, user_id)
        jobs = self.background_jobs.get(scope) or {}
        before = len(jobs)
        self.background_jobs[scope] = {
            job_id: job
            for job_id, job in jobs.items()
            if job.get("status") not in BACKGROUND_CLOSED_STATUSES
        }
        removed = before - len(self.background_jobs[scope])
        self._save_background_jobs()
        return f"Removed {removed} closed background job(s)."

    async def _run_background_job(self, bot, scope: str, job_id: str) -> None:
        job = self._get_background_job(scope, job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = int(time.time())
        self._save_background_jobs()
        try:
            helper = getattr(self, "openai", None)
            if not helper or not hasattr(helper, "get_chat_response"):
                raise RuntimeError("OpenAI helper is not available for background jobs")
            request_context = RequestContext(
                chat_id=int(job["chat_id"]),
                user_id=int(job["user_id"]),
                message_id=job.get("reply_to_message_id"),
                request_id=f"background_{job_id}",
            )
            response, total_tokens = await helper.get_chat_response(
                chat_id=int(job["chat_id"]),
                query=str(job["prompt"]),
                request_id=f"background_{job_id}",
                user_id=int(job["user_id"]),
                request_context=request_context,
            )
            if job.get("status") == "cancelled":
                return
            job["status"] = "completed"
            job["finished_at"] = int(time.time())
            job["tokens"] = total_tokens
            job["result_preview"] = self._background_result_preview(response)
            self._save_background_jobs()
            await send_agent_response(
                bot,
                chat_id=int(job["chat_id"]),
                response=response,
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
                title=f"Background job `{job_id}` completed.",
            )
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["finished_at"] = int(time.time())
            self._save_background_jobs()
            return
        except Exception as exc:
            logging.exception("Background job %s failed", job_id)
            job["status"] = "failed"
            job["finished_at"] = int(time.time())
            job["error"] = str(exc)
            self._save_background_jobs()
            await send_text_chunks(
                bot,
                chat_id=int(job["chat_id"]),
                text=f"Background job `{job_id}` failed: {exc}",
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
            )

    @staticmethod
    def _background_result_preview(response: Any) -> str:
        if isinstance(response, dict):
            payload = response.get("direct_result")
            if isinstance(payload, dict):
                text = payload.get("text") or payload.get("value") or payload.get("caption") or payload.get("kind")
                return str(text or "")[:300]
        return str(response or "")[:300]

    @staticmethod
    def _goal_status_from_response(response: Any) -> str:
        if isinstance(response, dict):
            payload = response.get("direct_result")
            if isinstance(payload, dict):
                status = str(payload.get("status") or "").strip().lower()
                if status in {"completed", "blocked"}:
                    return status
        return "completed"

    @staticmethod
    def _normalize_goal_limits(value: Any) -> Dict[str, int]:
        raw = value if isinstance(value, dict) else {}

        def bounded_int(name: str, default: int, lower: int, upper: int) -> int:
            try:
                requested = int(raw.get(name, default))
            except (TypeError, ValueError):
                requested = default
            return max(lower, min(requested, upper))

        limits = {
            "max_runtime_seconds": bounded_int(
                "max_runtime_seconds",
                GOAL_RUN_DEFAULT_LIMITS["max_runtime_seconds"],
                60,
                14400,
            ),
        }
        if raw.get("token_budget") is not None:
            try:
                limits["token_budget"] = max(1, int(raw.get("token_budget")))
            except (TypeError, ValueError):
                pass
        return limits

    @staticmethod
    def _goal_run_prompt(prompt: str, definition_of_done: Dict[str, Any] | None, limits: Dict[str, int]) -> str:
        dod_text = (
            json.dumps(definition_of_done, ensure_ascii=False, sort_keys=True)
            if definition_of_done else "{}"
        )
        return (
            "Controlled background goal run.\n"
            f"Objective:\n{prompt}\n\n"
            f"Definition of done:\n{dod_text}\n\n"
            "Limits to respect:\n"
            f"- max_runtime_seconds: {limits.get('max_runtime_seconds')}\n"
            f"- token_budget: {limits.get('token_budget', 'not set')}\n\n"
            "Use tools only when needed. Finish through agent_tools.deliver_to_user with "
            "status=completed or status=blocked and include verification_summary or blocked_reason."
        )

    async def _manage_goal_runs(self, helper, *, request_context=None, **kwargs) -> Dict[str, Any]:
        if self.db_handle is None:
            return {"success": False, "error": "agent_tools goal runs require database"}
        action = str(kwargs.get("action") or "").strip().lower()
        chat_id = kwargs.get("chat_id")
        user_id = kwargs.get("user_id")
        if request_context is not None:
            chat_id = getattr(request_context, "plugin_chat_id", None) or chat_id
            user_id = getattr(request_context, "user_id", None) or user_id
        scope = self._background_scope(chat_id, user_id)
        if action == "start":
            prompt = str(kwargs.get("prompt") or "").strip()
            if not prompt:
                return {"success": False, "error": "manage_goal_runs start requires prompt"}
            definition = self._normalize_existing_contract(kwargs.get("definition_of_done"))
            if definition is None:
                return {
                    "success": False,
                    "error": "manage_goal_runs start requires definition_of_done with goal, success_criteria, and verification",
                }
            validation_error = self._contract_validation_error(definition)
            if validation_error:
                return {"success": False, "error": validation_error}
            limits = self._normalize_goal_limits(kwargs.get("limits"))
            run_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
            now = int(time.time())
            await self.db_handle.execute(
                """
                INSERT INTO agent_goal_runs
                (run_id, scope, chat_id, user_id, reply_to_message_id, message_thread_id,
                 prompt, definition_of_done, status, limits, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    run_id,
                    scope,
                    int(chat_id),
                    int(user_id if user_id is not None else chat_id),
                    kwargs.get("message_id"),
                    getattr(request_context, "message_thread_id", None) if request_context is not None else None,
                    prompt,
                    json.dumps(definition, ensure_ascii=False),
                    json.dumps(limits, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            await self._append_goal_event(run_id, "queued", "Goal run queued", {"limits": limits})
            return {
                "success": True,
                "goal_run": {
                    "run_id": run_id,
                    "status": "queued",
                    "limits": limits,
                    "definition_of_done": definition,
                },
                "output": f"Goal run `{run_id}` queued.",
            }
        if action == "list":
            rows = await self._list_goal_runs(scope)
            return {"success": True, "goal_runs": rows, "output": self._format_goal_run_list(rows)}
        if action == "status":
            run_id = str(kwargs.get("run_id") or "").strip()
            row = await self._get_goal_run(scope, run_id)
            if not row:
                return {"success": False, "error": "Goal run not found"}
            return {"success": True, "goal_run": row, "output": self._format_goal_run_status(row)}
        if action == "cancel":
            run_id = str(kwargs.get("run_id") or "").strip()
            row = await self._get_goal_run(scope, run_id)
            if not row:
                return {"success": False, "error": "Goal run not found"}
            if row.get("status") in GOAL_RUN_CLOSED_STATUSES:
                return {"success": True, "goal_run": row, "output": f"Goal run `{run_id}` is already {row.get('status')}."}
            now = int(time.time())
            if row.get("status") == "queued":
                cancelled = await self.db_handle.fetch_one(
                    """
                    UPDATE agent_goal_runs
                    SET cancel_requested = 1, status = 'cancelled', finished_at = ?, updated_at = ?
                    WHERE scope = ? AND run_id = ? AND status = 'queued'
                    RETURNING *
                    """,
                    (now, now, scope, run_id),
                )
                if cancelled:
                    await self._append_goal_event(run_id, "cancelled", "Queued goal run cancelled", {})
                    return {"success": True, "run_id": run_id, "output": f"Goal run `{run_id}` cancelled."}
                row = await self._get_goal_run(scope, run_id)
                if not row:
                    return {"success": False, "error": "Goal run not found"}
                if row.get("status") in GOAL_RUN_CLOSED_STATUSES:
                    return {"success": True, "goal_run": row, "output": f"Goal run `{run_id}` is already {row.get('status')}."}
            if row.get("status") == "cancelling":
                return {"success": True, "run_id": run_id, "output": f"Goal run `{run_id}` cancellation already requested."}
            cancelling = await self.db_handle.fetch_one(
                """
                UPDATE agent_goal_runs
                SET cancel_requested = 1, status = 'cancelling', updated_at = ?
                WHERE scope = ? AND run_id = ? AND status = 'running'
                RETURNING *
                """,
                (now, scope, run_id),
            )
            if not cancelling:
                row = await self._get_goal_run(scope, run_id)
                if row and row.get("status") in GOAL_RUN_CLOSED_STATUSES:
                    return {"success": True, "goal_run": row, "output": f"Goal run `{run_id}` is already {row.get('status')}."}
                return {"success": False, "error": "Goal run could not be cancelled from its current state"}
            task = self._goal_run_tasks.get(run_id)
            if task and not task.done():
                task.cancel()
            await self._append_goal_event(run_id, "cancel_requested", "Goal run cancellation requested", {})
            return {"success": True, "run_id": run_id, "output": f"Goal run `{run_id}` cancellation requested."}
        if action == "clear":
            rows = await self.db_handle.fetch_all(
                f"SELECT run_id FROM agent_goal_runs WHERE scope = ? AND status IN ({','.join('?' for _ in GOAL_RUN_CLOSED_STATUSES)})",
                (scope, *sorted(GOAL_RUN_CLOSED_STATUSES)),
            )
            run_ids = [str(row["run_id"]) for row in rows]
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                async with self.db_handle.transaction() as tx:
                    await tx.execute(f"DELETE FROM agent_goal_run_events WHERE run_id IN ({placeholders})", tuple(run_ids))
                    await tx.execute(f"DELETE FROM agent_goal_runs WHERE run_id IN ({placeholders})", tuple(run_ids))
            return {"success": True, "cleared": len(run_ids), "output": f"Removed {len(run_ids)} closed goal run(s)."}
        return {"success": False, "error": f"Unknown goal run action: {action}"}

    async def _list_goal_runs(self, scope: str) -> List[Dict[str, Any]]:
        rows = await self.db_handle.fetch_all(
            """
            SELECT run_id, status, prompt, turns_used, tool_calls_used, tokens_used,
                   result_preview, error, created_at, updated_at, finished_at
            FROM agent_goal_runs
            WHERE scope = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (scope,),
        )
        return rows

    async def _get_goal_run(self, scope: str, run_id: str) -> Dict[str, Any] | None:
        if not run_id:
            return None
        return await self.db_handle.fetch_one(
            """
            SELECT *
            FROM agent_goal_runs
            WHERE scope = ? AND run_id = ?
            """,
            (scope, run_id),
        )

    async def _append_goal_event(self, run_id: str, event_type: str, message: str, payload: Any) -> None:
        if self.db_handle is None:
            return
        await self.db_handle.execute(
            """
            INSERT INTO agent_goal_run_events (run_id, event_type, message, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event_type,
                message,
                json.dumps(payload or {}, ensure_ascii=False),
                int(time.time()),
            ),
        )

    @staticmethod
    def _format_goal_run_list(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "No goal runs yet."
        lines = ["Goal runs:"]
        for row in rows:
            prompt = str(row.get("prompt") or "").replace("\n", " ")
            if len(prompt) > 80:
                prompt = prompt[:77] + "..."
            lines.append(f"- `{row.get('run_id')}` {row.get('status')}: {prompt}")
        return "\n".join(lines)

    @staticmethod
    def _format_goal_run_status(row: Dict[str, Any]) -> str:
        lines = [
            f"Goal run `{row.get('run_id')}`",
            f"Status: {row.get('status')}",
            f"Prompt: {row.get('prompt')}",
        ]
        if row.get("tokens_used") is not None:
            lines.append(f"Tokens: {row.get('tokens_used')}")
        if row.get("result_preview"):
            lines.append(f"Result preview: {row.get('result_preview')}")
        if row.get("error"):
            lines.append(f"Error: {row.get('error')}")
        return "\n".join(lines)

    async def _goal_runs_tick(self, *, application=None) -> None:
        if self.db_handle is None:
            return
        active = {run_id: task for run_id, task in self._goal_run_tasks.items() if not task.done()}
        self._goal_run_tasks = active
        await self._interrupt_orphaned_goal_runs(set(active))
        if len(active) >= GOAL_RUN_MAX_CONCURRENCY:
            return
        rows = await self.db_handle.fetch_all(
            """
            SELECT run_id
            FROM agent_goal_runs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (GOAL_RUN_MAX_CONCURRENCY - len(active),),
        )
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if not run_id or run_id in self._goal_run_tasks:
                continue
            task = asyncio.create_task(self._run_goal_run(application, run_id))
            self._goal_run_tasks[run_id] = task
            task.add_done_callback(lambda _task, rid=run_id: self._goal_run_tasks.pop(rid, None))

    async def _interrupt_orphaned_goal_runs(self, active_run_ids: set[str]) -> None:
        rows = await self.db_handle.fetch_all(
            "SELECT run_id FROM agent_goal_runs WHERE status IN ('running', 'cancelling')",
            (),
        )
        now = int(time.time())
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if not run_id or run_id in active_run_ids:
                continue
            interrupted = await self.db_handle.fetch_one(
                """
                UPDATE agent_goal_runs
                SET status = 'interrupted',
                    error = COALESCE(error, 'Goal run interrupted before completion'),
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?
                WHERE run_id = ? AND status IN ('running', 'cancelling')
                RETURNING run_id
                """,
                (now, now, run_id),
            )
            if interrupted:
                await self._append_goal_event(
                    run_id,
                    "interrupted",
                    "Goal run interrupted before completion",
                    {},
                )

    async def _run_goal_run(self, application, run_id: str) -> None:
        now = int(time.time())
        row = await self.db_handle.fetch_one(
            """
            UPDATE agent_goal_runs
            SET status = 'running', started_at = ?, updated_at = ?
            WHERE run_id = ? AND status = 'queued' AND cancel_requested = 0
            RETURNING *
            """,
            (now, now, run_id),
        )
        if not row:
            return
        await self._append_goal_event(run_id, "started", "Goal run started", {})
        try:
            helper = getattr(self, "openai", None)
            if not helper or not hasattr(helper, "get_chat_response"):
                raise RuntimeError("OpenAI helper is not available for goal runs")
            limits = json.loads(row.get("limits") or "{}")
            definition = json.loads(row.get("definition_of_done") or "{}")
            prompt = self._goal_run_prompt(str(row.get("prompt") or ""), definition, limits)
            bot = getattr(application, "bot", None) if application is not None else None
            request_context = RequestContext(
                chat_id=int(row["chat_id"]),
                user_id=int(row["user_id"]),
                message_id=row.get("reply_to_message_id"),
                request_id=f"goal_{run_id}",
            )
            response, total_tokens = await asyncio.wait_for(
                helper.get_chat_response(
                    chat_id=int(row["chat_id"]),
                    query=prompt,
                    request_id=f"goal_{run_id}",
                    user_id=int(row["user_id"]),
                    request_context=request_context,
                ),
                timeout=int(limits.get("max_runtime_seconds") or GOAL_RUN_DEFAULT_LIMITS["max_runtime_seconds"]),
            )
            current = await self.db_handle.fetch_one("SELECT cancel_requested FROM agent_goal_runs WHERE run_id = ?", (run_id,))
            if current and int(current.get("cancel_requested") or 0):
                await self._finish_goal_run(run_id, "cancelled", tokens_used=total_tokens)
                return
            status = self._goal_status_from_response(response)
            if limits.get("token_budget") and total_tokens and int(total_tokens) > int(limits["token_budget"]):
                status = "budget_exhausted"
            preview = self._background_result_preview(response)
            await self._finish_goal_run(
                run_id,
                status,
                tokens_used=total_tokens,
                result_preview=preview,
                final_result=response,
            )
            if bot is not None:
                await send_agent_response(
                    bot,
                    chat_id=int(row["chat_id"]),
                    response=response,
                    reply_to_message_id=row.get("reply_to_message_id"),
                    message_thread_id=row.get("message_thread_id"),
                    title=f"Goal run `{run_id}` {status}.",
                )
        except asyncio.TimeoutError:
            await self._finish_goal_run(run_id, "budget_exhausted", error="Goal run exceeded max_runtime_seconds")
        except asyncio.CancelledError:
            await self._finish_goal_run(run_id, "cancelled")
            raise
        except Exception as exc:
            logging.exception("Goal run %s failed", run_id)
            await self._finish_goal_run(run_id, "failed", error=str(exc))

    async def _finish_goal_run(
        self,
        run_id: str,
        status: str,
        *,
        tokens_used: int | None = None,
        result_preview: str | None = None,
        final_result: Any = None,
        error: str | None = None,
    ) -> None:
        now = int(time.time())
        await self.db_handle.execute(
            """
            UPDATE agent_goal_runs
            SET status = ?, tokens_used = COALESCE(?, tokens_used),
                result_preview = COALESCE(?, result_preview),
                final_result = COALESCE(?, final_result),
                error = COALESCE(?, error),
                finished_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                status,
                tokens_used,
                result_preview,
                json.dumps(final_result, ensure_ascii=False, default=str) if final_result is not None else None,
                error,
                now,
                now,
                run_id,
            ),
        )
        await self._append_goal_event(run_id, status, f"Goal run {status}", {"error": error} if error else {})

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        request_context = kwargs.pop("request_context", None)
        if function_name == "manage_plan_tasks":
            # _manage_plan_tasks ходит в SQLite через sync get_connection
            # (_db_get_plan / _db_save_plan / _db_clear_plan). На event loop это
            # давало блокирующий write под WAL-busy. to_thread снимает блокировку.
            return await asyncio.to_thread(self._manage_plan_tasks, helper, **kwargs)
        if function_name == "update_working_checkpoint":
            return await asyncio.to_thread(self._manage_working_checkpoint, helper, **kwargs)
        if function_name == "ask_telegram_user":
            return await self._ask_telegram_user(helper, **kwargs)
        if function_name == "cancel_pending_question":
            return await self._cancel_pending_question(helper, **kwargs)
        if function_name == "run_subagents":
            return await self._run_subagents(helper, request_context=request_context, **kwargs)
        if function_name == "deliver_to_user":
            return await self._deliver_to_user(helper, request_context=request_context, **kwargs)
        if function_name == "manage_goal_runs":
            return await self._manage_goal_runs(helper, request_context=request_context, **kwargs)
        return {"success": False, "error": f"Unknown agent tool: {function_name}"}

    def register_schema(self) -> List[str]:
        return [
            '''
                CREATE TABLE IF NOT EXISTS agent_plan_contracts (
                    scope TEXT PRIMARY KEY,
                    contract TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            '''
                CREATE TABLE IF NOT EXISTS agent_plan_tasks (
                    scope TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    depends_on TEXT NOT NULL DEFAULT '[]',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (scope, task_id)
                )
            ''',
            '''
                CREATE INDEX IF NOT EXISTS idx_agent_plan_tasks_scope_position
                ON agent_plan_tasks(scope, position)
            ''',
            '''
                CREATE TABLE IF NOT EXISTS agent_working_checkpoints (
                    scope TEXT PRIMARY KEY,
                    checkpoint TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            ''',
            '''
                CREATE TABLE IF NOT EXISTS agent_goal_runs (
                    run_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    reply_to_message_id INTEGER,
                    message_thread_id INTEGER,
                    prompt TEXT NOT NULL,
                    definition_of_done TEXT,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    limits TEXT NOT NULL,
                    turns_used INTEGER NOT NULL DEFAULT 0,
                    tool_calls_used INTEGER NOT NULL DEFAULT 0,
                    tokens_used INTEGER,
                    result_preview TEXT,
                    final_result TEXT,
                    error TEXT,
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    updated_at INTEGER NOT NULL,
                    finished_at INTEGER
                )
            ''',
            '''
                CREATE INDEX IF NOT EXISTS idx_agent_goal_runs_scope_created
                ON agent_goal_runs(scope, created_at)
            ''',
            '''
                CREATE INDEX IF NOT EXISTS idx_agent_goal_runs_status_updated
                ON agent_goal_runs(status, updated_at)
            ''',
            '''
                CREATE TABLE IF NOT EXISTS agent_goal_run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload TEXT,
                    created_at INTEGER NOT NULL
                )
            ''',
            '''
                CREATE INDEX IF NOT EXISTS idx_agent_goal_run_events_run_id_id
                ON agent_goal_run_events(run_id, id)
            ''',
        ]

    def _db_save_plan(
        self,
        scope: str,
        tasks: List[Dict[str, Any]],
        contract: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the full agent plan for a scope."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM agent_plan_tasks WHERE scope = ?', (scope,))
                for position, task in enumerate(tasks):
                    cursor.execute('''
                        INSERT INTO agent_plan_tasks
                        (scope, task_id, position, content, status, depends_on, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        scope,
                        str(task.get('id') or ''),
                        position,
                        str(task.get('content') or ''),
                        str(task.get('status') or 'pending'),
                        json.dumps(task.get('depends_on') or [], ensure_ascii=False),
                        int(task.get('created_at') or 0),
                        int(task.get('updated_at') or 0),
                    ))
                if contract is not None:
                    cursor.execute('''
                        INSERT INTO agent_plan_contracts (scope, contract)
                        VALUES (?, ?)
                        ON CONFLICT(scope) DO UPDATE SET
                        contract = excluded.contract,
                        updated_at = CURRENT_TIMESTAMP
                    ''', (scope, json.dumps(contract, ensure_ascii=False)))
        except Exception as e:
            logging.error(f'Error saving agent plan: {e}', exc_info=True)
            raise

    def _db_get_plan(self, scope: str) -> Dict[str, Any]:
        """Return persisted agent plan tasks and optional DoD contract."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT task_id, content, status, depends_on, created_at, updated_at
                    FROM agent_plan_tasks
                    WHERE scope = ?
                    ORDER BY position ASC
                ''', (scope,))
                tasks = []
                for row in cursor.fetchall():
                    try:
                        depends_on = json.loads(row['depends_on'] or '[]')
                    except json.JSONDecodeError:
                        depends_on = []
                    tasks.append({
                        'id': row['task_id'],
                        'content': row['content'],
                        'status': row['status'],
                        'depends_on': depends_on if isinstance(depends_on, list) else [],
                        'created_at': int(row['created_at']),
                        'updated_at': int(row['updated_at']),
                    })

                cursor.execute(
                    'SELECT contract FROM agent_plan_contracts WHERE scope = ?',
                    (scope,),
                )
                result = cursor.fetchone()
                contract = None
                if result:
                    try:
                        loaded = json.loads(result['contract'])
                    except json.JSONDecodeError:
                        loaded = None
                    contract = loaded if isinstance(loaded, dict) else None
                return {'tasks': tasks, 'contract': contract}
        except Exception as e:
            logging.error(f'Error getting agent plan: {e}', exc_info=True)
            raise

    def _db_clear_plan(self, scope: str, *, clear_contract: bool = False) -> None:
        """Remove persisted agent plan tasks and optionally the DoD contract."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM agent_plan_tasks WHERE scope = ?', (scope,))
                if clear_contract:
                    cursor.execute('DELETE FROM agent_plan_contracts WHERE scope = ?', (scope,))
        except Exception as e:
            logging.error(f'Error clearing agent plan: {e}', exc_info=True)
            raise

    def _db_prune_plans(self, cutoff_timestamp: int) -> int:
        """Delete stale agent plan tasks and empty contracts older than cutoff."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'DELETE FROM agent_plan_tasks WHERE updated_at < ?',
                    (int(cutoff_timestamp),),
                )
                deleted = cursor.rowcount
                cursor.execute('''
                    DELETE FROM agent_plan_contracts
                    WHERE scope NOT IN (SELECT DISTINCT scope FROM agent_plan_tasks)
                    AND strftime('%s', updated_at) < ?
                ''', (int(cutoff_timestamp),))
                return deleted
        except Exception as e:
            logging.error(f'Error pruning agent plans: {e}', exc_info=True)
            raise

    def _db_get_checkpoint(self, scope: str) -> Dict[str, Any] | None:
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT checkpoint FROM agent_working_checkpoints WHERE scope = ?",
                    (scope,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            raw = row["checkpoint"] if hasattr(row, "keys") else row[0]
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else None
        except Exception as e:
            logging.error(f"Error getting agent checkpoint: {e}", exc_info=True)
            raise

    def _db_save_checkpoint(self, scope: str, checkpoint: Dict[str, Any]) -> None:
        now = int(time.time())
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_working_checkpoints (scope, checkpoint, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                    checkpoint = excluded.checkpoint,
                    updated_at = excluded.updated_at
                    """,
                    (scope, json.dumps(checkpoint, ensure_ascii=False), now, now),
                )
        except Exception as e:
            logging.error(f"Error saving agent checkpoint: {e}", exc_info=True)
            raise

    def _db_clear_checkpoint(self, scope: str) -> bool:
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM agent_working_checkpoints WHERE scope = ?",
                    (scope,),
                )
                return bool(cursor.rowcount)
        except Exception as e:
            logging.error(f"Error clearing agent checkpoint: {e}", exc_info=True)
            raise

    def _db_prune_checkpoints(self, cutoff_timestamp: int) -> int:
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM agent_working_checkpoints WHERE updated_at < ?",
                    (int(cutoff_timestamp),),
                )
                return int(cursor.rowcount or 0)
        except Exception as e:
            logging.error(f"Error pruning agent checkpoints: {e}", exc_info=True)
            raise

    async def _db_clear_terminal_tasks_async(self, scope: str) -> bool:
        """Remove all tasks under scope IFF every task is in a closed status. Returns True if cleared."""
        rows = await self.db_handle.fetch_all(
            "SELECT status FROM agent_plan_tasks WHERE scope = ?", (scope,)
        )
        contract_rows = await self.db_handle.fetch_all(
            "SELECT 1 AS one FROM agent_plan_contracts WHERE scope = ? LIMIT 1", (scope,)
        )
        if not rows and not contract_rows:
            return False
        if any(r["status"] not in CLOSED_STATUSES for r in rows):
            return False
        async with self.db_handle.transaction() as tx:
            await tx.execute("DELETE FROM agent_plan_tasks WHERE scope = ?", (scope,))
            await tx.execute("DELETE FROM agent_plan_contracts WHERE scope = ?", (scope,))
            await tx.execute("DELETE FROM agent_working_checkpoints WHERE scope = ?", (scope,))
        return True

    async def _db_clear_all_tasks_async(self, scope: str) -> bool:
        rows = await self.db_handle.fetch_all(
            """
            SELECT 1 AS one FROM agent_plan_tasks WHERE scope = ?
            UNION ALL
            SELECT 1 AS one FROM agent_plan_contracts WHERE scope = ?
            UNION ALL
            SELECT 1 AS one FROM agent_working_checkpoints WHERE scope = ?
            LIMIT 1
            """,
            (scope, scope, scope),
        )
        if not rows:
            return False
        async with self.db_handle.transaction() as tx:
            await tx.execute("DELETE FROM agent_plan_tasks WHERE scope = ?", (scope,))
            await tx.execute("DELETE FROM agent_plan_contracts WHERE scope = ?", (scope,))
            await tx.execute("DELETE FROM agent_working_checkpoints WHERE scope = ?", (scope,))
        return True

    async def on_session_reset(self, payload) -> None:
        if self.db_handle is None:
            return
        scope = compute_scope_key(payload.chat_id, payload.user_id)
        try:
            if payload.terminal_only:
                cleared = await self._db_clear_terminal_tasks_async(scope)
                if cleared:
                    await self.db_handle.execute(
                        "DELETE FROM agent_working_checkpoints WHERE scope = ?",
                        (scope,),
                    )
            else:
                cleared = await self._db_clear_all_tasks_async(scope)
            if cleared:
                logging.info(
                    "Cleared agent plan scope=%s reason=%s terminal_only=%s",
                    scope, payload.reason, payload.terminal_only,
                )
        except Exception:
            logging.exception("agent_tools.on_session_reset failed scope=%s", scope)
        # Re-plan trigger state is in-process only and lives until the next
        # chat request. Only wipe it on a full session reset (terminal_only=False);
        # a terminal-only reset clears finished tasks but the session continues,
        # so a pending re-plan inject scheduled earlier in the same session must
        # still be delivered on the next chat request.
        if not payload.terminal_only:
            self._tool_error_streaks.pop(scope, None)
            self._pending_replan.pop(scope, None)

    def _prune_stale_tasks(self) -> bool:
        cutoff = int(time.time()) - self.TASKS_TTL_SECONDS
        if self.db is not None:
            try:
                deleted_plans = self._db_prune_plans(cutoff)
                deleted_checkpoints = self._db_prune_checkpoints(cutoff)
                return bool(deleted_plans or deleted_checkpoints)
            except Exception:
                logging.exception("Failed to prune stale agent plans from database")
                return False

        return False

    def _get_scope_plan(self, scope: str) -> Dict[str, Any]:
        if self.db is not None:
            plan = self._db_get_plan(scope)
            return {
                "tasks": [
                    self._normalize_existing_task(task)
                    for task in (plan.get("tasks") or [])
                ],
                "contract": self._normalize_existing_contract(plan.get("contract")),
            }
        return {"tasks": [], "contract": None}

    def _save_scope_plan(
        self,
        scope: str,
        tasks: List[Dict[str, Any]],
        *,
        contract: Any = _CONTRACT_UNSET,
    ) -> None:
        normalized_tasks = [self._normalize_existing_task(task) for task in tasks]
        if self.db is not None:
            current_contract = self._db_get_plan(scope).get("contract")
            contract_to_save = current_contract if contract is _CONTRACT_UNSET else contract
            self._db_save_plan(scope, normalized_tasks, contract_to_save)
            return
        raise RuntimeError("agent_tools plan storage requires database")

    def _clear_scope_tasks(self, scope: str) -> None:
        if self.db is not None:
            self._db_clear_plan(scope, clear_contract=False)
            return

    # ---- re-plan trigger helpers --------------------------------------------------

    def _get_replan_lock(self) -> asyncio.Lock:
        # Lazy-init: __init__ may run before any event loop exists.
        if self._replan_lock is None:
            self._replan_lock = asyncio.Lock()
        return self._replan_lock

    def _current_in_progress_task_id(self, scope: str) -> Optional[str]:
        if self.db is None:
            return None
        try:
            plan = self._get_scope_plan(scope)
        except Exception:
            logging.debug("agent_tools: _current_in_progress_task_id read failed", exc_info=True)
            return None
        for task in plan.get("tasks") or []:
            if task.get("status") == "in_progress":
                task_id = str(task.get("id") or "").strip()
                return task_id or None
        return None

    def _schedule_replan(self, scope: str, reason: str, task_id: str) -> None:
        if not scope or not task_id:
            return
        existing = self._pending_replan.get(scope)
        new_entry = {"reason": reason, "task_id": task_id}
        if existing == new_entry:
            return
        self._pending_replan[scope] = new_entry

    async def _record_tool_outcome(
        self,
        scope: str,
        task_id: Optional[str],
        success: bool,
        tool_name: str,
    ) -> None:
        """Update consecutive-error streak and schedule re-plan if threshold reached.

        Args:
            scope: canonical scope key from ``compute_scope_key``.
            task_id: id of the in_progress task at the start of the batch, may be None.
            success: whether the tool call succeeded.
            tool_name: tool function name (unused for now; reserved for filtering).
        """
        if not scope:
            return
        if success:
            self._tool_error_streaks.pop(scope, None)
            return
        if not task_id:
            # No in_progress task — no task-scoped streak to track.
            return
        entry = self._tool_error_streaks.get(scope)
        if not entry or entry.get("task_id") != task_id:
            self._tool_error_streaks[scope] = {"task_id": task_id, "count": 1}
            return
        entry["count"] = int(entry.get("count") or 0) + 1
        if entry["count"] >= _REPLAN_ERROR_THRESHOLD:
            self._schedule_replan(scope, "errors", task_id)
            # Reset streak so the next streak must accumulate from zero
            # (prevents trigger spam on subsequent failures).
            self._tool_error_streaks.pop(scope, None)

    @staticmethod
    def _replan_message_body(reason: str, task_id: str) -> str:
        if reason == "blocked":
            return (
                f"{_REPLAN_TRIGGER_MARKER}task {task_id} marked blocked. "
                "Опиши блок и предложи альтернативную ветку через manage_plan_tasks(action=update)."
            )
        # default to "errors"
        return (
            f"{_REPLAN_TRIGGER_MARKER}consecutive errors on task {task_id}. "
            "Пересмотри план через manage_plan_tasks(action=update). "
            "Опиши, что не сработало, и предложи альтернативную ветку плана."
        )

    @staticmethod
    def _normalize_existing_task(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(task.get("id") or task.get("task_id") or "").strip(),
            "content": str(task.get("content") or "").strip(),
            "status": str(task.get("status") or "pending").strip(),
            "depends_on": AgentToolsPlugin._normalize_depends_on(task.get("depends_on")),
            "created_at": int(task.get("created_at") or int(time.time())),
            "updated_at": int(task.get("updated_at") or task.get("created_at") or int(time.time())),
        }

    @staticmethod
    def _normalize_existing_contract(contract: Any) -> Dict[str, Any] | None:
        if not isinstance(contract, dict):
            return None
        normalized = {
            "goal": str(contract.get("goal") or "").strip(),
            "success_criteria": AgentToolsPlugin._normalize_string_list(
                contract.get("success_criteria") or contract.get("criteria")
            ),
            "verification": AgentToolsPlugin._normalize_string_list(contract.get("verification")),
            "constraints": AgentToolsPlugin._normalize_string_list(contract.get("constraints")),
        }
        if not normalized["goal"] and not any(
            normalized[key] for key in ("success_criteria", "verification", "constraints")
        ):
            return None
        return normalized

    @staticmethod
    def _normalize_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        normalized: List[str] = []
        for item in candidates:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_depends_on(value: Any) -> List[str]:
        return AgentToolsPlugin._normalize_string_list(value)

    def _contract_from_kwargs(self, kwargs: Dict[str, Any]) -> tuple[Any, str | None]:
        if "definition_of_done" not in kwargs:
            return _CONTRACT_UNSET, None
        contract = self._normalize_existing_contract(kwargs.get("definition_of_done"))
        if contract is None:
            return _CONTRACT_UNSET, "definition_of_done must include goal, success_criteria, verification, or constraints"
        validation_error = self._contract_validation_error(contract)
        if validation_error:
            return _CONTRACT_UNSET, validation_error
        return contract, None

    @staticmethod
    def _contract_validation_error(contract: Dict[str, Any] | None) -> str | None:
        if not contract:
            return "definition_of_done is required for durable agent plans"
        if not str(contract.get("goal") or "").strip():
            return "definition_of_done.goal is required"
        if not contract.get("success_criteria"):
            return "definition_of_done.success_criteria must include at least one item"
        if not contract.get("verification"):
            return "definition_of_done.verification must include at least one item"
        return None

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
                    text=self.t("agent_tools_orphaned_question_inactive"),
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

    def get_plan_tasks(self, chat_id=None, user_id=None) -> List[Dict[str, Any]]:
        """
        Return a snapshot of the current plan tasks for the given scope, ordered as stored.
        Used by the live status message to render plan progress to the user.
        """
        scope = compute_scope_key(chat_id, user_id)
        tasks = self._get_scope_plan(scope).get("tasks") or []
        return [
            {
                "id": str(task.get("id") or ""),
                "content": str(task.get("content") or ""),
                "status": str(task.get("status") or "pending"),
                "depends_on": self._normalize_depends_on(task.get("depends_on")),
            }
            for task in tasks
        ]

    def clear_plan_tasks(self, chat_id=None, user_id=None) -> bool:
        """Remove all plan tasks for the given scope."""
        scope = compute_scope_key(chat_id, user_id)
        tasks = self._get_scope_plan(scope).get("tasks") or []
        if not tasks:
            return False
        self._clear_scope_tasks(scope)
        return True

    def clear_terminal_plan_tasks(self, chat_id=None, user_id=None) -> bool:
        """Remove the plan only when every task in the scope is already closed."""
        scope = compute_scope_key(chat_id, user_id)
        tasks = self._get_scope_plan(scope).get("tasks") or []
        if not tasks:
            return False
        if any(task.get("status") not in CLOSED_STATUSES for task in tasks):
            return False
        self._clear_scope_tasks(scope)
        return True

    @staticmethod
    def _copy_plan_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(task) for task in tasks]

    @staticmethod
    def _is_delivery_task(content: str) -> bool:
        lowered = content.lower()
        has_action = any(word in lowered for word in DELIVERY_ACTION_WORDS)
        has_target = any(word in lowered for word in DELIVERY_TARGET_WORDS)
        return has_action and has_target

    def _validate_plan_tasks(self, tasks: List[Dict[str, Any]]) -> str | None:
        task_ids = [str(task.get("id") or "") for task in tasks]
        duplicate_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicate_ids:
            return f"Duplicate plan task id(s): {', '.join(duplicate_ids)}"

        task_by_id = {
            str(task.get("id") or ""): task
            for task in tasks
            if str(task.get("id") or "")
        }
        for task in tasks:
            task_id = str(task.get("id") or "")
            dependencies = self._normalize_depends_on(task.get("depends_on"))
            task["depends_on"] = dependencies
            if task_id in dependencies:
                return f"Task {task_id} cannot depend on itself"
            missing = [dep for dep in dependencies if dep not in task_by_id]
            if missing:
                return f"Task {task_id} depends on unknown task(s): {', '.join(missing)}"

        cycle = self._find_dependency_cycle(task_by_id)
        if cycle:
            return f"Plan task dependency cycle detected: {' -> '.join(cycle)}"

        in_progress = [
            (index, task)
            for index, task in enumerate(tasks)
            if task.get("status") == "in_progress"
        ]
        if len(in_progress) > 1:
            ids = ", ".join(str(task.get("id") or "") for _, task in in_progress)
            return f"Only one plan task may be in_progress at a time: {ids}"

        if in_progress:
            active_index, active_task = in_progress[0]
            dependencies = self._normalize_depends_on(active_task.get("depends_on"))
            if dependencies:
                open_dependencies = [
                    dep for dep in dependencies
                    if (task_by_id.get(dep) or {}).get("status") not in CLOSED_STATUSES
                ]
                if open_dependencies:
                    return (
                        f"Cannot set {active_task.get('id')} in_progress while "
                        f"dependencies are still open: {', '.join(open_dependencies)}"
                    )
            else:
                earlier_open = [
                    str(task.get("id") or "")
                    for task in tasks[:active_index]
                    if task.get("status") not in CLOSED_STATUSES
                ]
                if earlier_open:
                    return (
                        f"Cannot set {active_task.get('id')} in_progress while earlier "
                        f"tasks are still open: {', '.join(earlier_open)}"
                    )

        delivery_tasks = [
            task
            for task in tasks
            if task.get("status") not in CLOSED_STATUSES
            and self._is_delivery_task(str(task.get("content") or ""))
        ]
        if len(delivery_tasks) > 1:
            first_id = delivery_tasks[0].get("id")
            duplicate_ids = ", ".join(str(task.get("id") or "") for task in delivery_tasks[1:])
            return (
                f"Plan already has open delivery task {first_id}; update it instead "
                f"of adding duplicate delivery task(s): {duplicate_ids}"
            )

        return None

    @staticmethod
    def _find_dependency_cycle(task_by_id: Dict[str, Dict[str, Any]]) -> List[str] | None:
        visiting: set[str] = set()
        visited: set[str] = set()
        path: List[str] = []

        def visit(task_id: str) -> List[str] | None:
            if task_id in visited:
                return None
            if task_id in visiting:
                if task_id in path:
                    return path[path.index(task_id):] + [task_id]
                return [task_id, task_id]
            visiting.add(task_id)
            path.append(task_id)
            for dependency in task_by_id.get(task_id, {}).get("depends_on") or []:
                cycle = visit(str(dependency))
                if cycle:
                    return cycle
            path.pop()
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task_id in task_by_id:
            cycle = visit(task_id)
            if cycle:
                return cycle
        return None

    @staticmethod
    def _trim_text(value: Any, limit: int = _CHECKPOINT_MAX_TEXT_CHARS) -> str:
        text = str(value or "").strip()
        if len(text) > limit:
            return text[:limit].rstrip() + "..."
        return text

    @staticmethod
    def _trim_list(value: Any) -> List[str]:
        return [
            AgentToolsPlugin._trim_text(item, 320)
            for item in AgentToolsPlugin._normalize_string_list(value)[:_CHECKPOINT_MAX_LIST_ITEMS]
        ]

    @staticmethod
    def _normalize_checkpoint(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = {
            "summary": AgentToolsPlugin._trim_text(kwargs.get("summary")),
            "current_task_id": AgentToolsPlugin._trim_text(kwargs.get("current_task_id"), 80),
            "next_step": AgentToolsPlugin._trim_text(kwargs.get("next_step"), 800),
            "evidence": AgentToolsPlugin._trim_list(kwargs.get("evidence")),
            "blockers": AgentToolsPlugin._trim_list(kwargs.get("blockers")),
            "files_touched": AgentToolsPlugin._trim_list(kwargs.get("files_touched")),
            "verification_status": AgentToolsPlugin._trim_text(kwargs.get("verification_status"), 800),
        }
        normalized = {key: value for key, value in checkpoint.items() if value not in ("", [], None)}
        if normalized:
            normalized["updated_at"] = int(time.time())
        return normalized

    @staticmethod
    def _format_working_checkpoint(checkpoint: Dict[str, Any]) -> str:
        lines = ["Working checkpoint for this agent task:"]
        for key, label in (
            ("summary", "Summary"),
            ("current_task_id", "Current task"),
            ("next_step", "Next step"),
            ("verification_status", "Verification"),
        ):
            value = checkpoint.get(key)
            if value:
                lines.append(f"- {label}: {value}")
        for key, label in (
            ("evidence", "Evidence"),
            ("blockers", "Blockers"),
            ("files_touched", "Files touched"),
        ):
            values = checkpoint.get(key)
            if values:
                rendered = "; ".join(str(item) for item in values)
                lines.append(f"- {label}: {rendered}")
        lines.append("Treat this as operational scratch state; current user input and tool results take precedence.")
        return "\n".join(lines)

    def _checkpoint_response(
        self,
        action: str,
        checkpoint: Dict[str, Any] | None,
        *,
        changed: bool,
    ) -> Dict[str, Any]:
        if checkpoint:
            output = self._format_working_checkpoint(checkpoint)
        else:
            output = "No working checkpoint."
        return {
            "success": True,
            "output": output,
            "working_checkpoint": {
                "action": action,
                "changed": changed,
                "checkpoint": checkpoint,
            },
        }

    def _manage_working_checkpoint(self, helper, **kwargs) -> Dict[str, Any]:
        if self.db is None:
            return {"success": False, "error": "agent_tools checkpoint storage requires database"}
        action = str(kwargs.get("action") or "").strip().lower()
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        if action == "list":
            return self._checkpoint_response(action, self._db_get_checkpoint(scope), changed=False)
        if action == "clear":
            changed = self._db_clear_checkpoint(scope)
            return self._checkpoint_response(action, None, changed=changed)
        if action == "update":
            checkpoint = self._normalize_checkpoint(kwargs)
            if not checkpoint:
                return {"success": False, "error": "checkpoint update requires at least one content field"}
            self._db_save_checkpoint(scope, checkpoint)
            return self._checkpoint_response(action, checkpoint, changed=True)
        return {"success": False, "error": f"Unknown checkpoint action: {action}"}

    def _manage_plan_tasks(self, helper, **kwargs) -> Dict:
        if self.db is None:
            return {"success": False, "error": "agent_tools plan storage requires database"}
        action = str(kwargs.get("action") or "").strip()
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        plan = self._get_scope_plan(scope)
        tasks = plan.get("tasks") or []
        current_contract = plan.get("contract")
        contract_update, contract_error = self._contract_from_kwargs(kwargs)
        if contract_error:
            if action == "add":
                return self._plan_contract_repair_response(contract_error, kwargs.get("tasks") or [])
            return {"success": False, "error": contract_error}
        effective_contract = current_contract if contract_update is _CONTRACT_UNSET else contract_update
        contract_changed = contract_update is not _CONTRACT_UNSET and contract_update != current_contract

        if action == "add":
            items = kwargs.get("tasks") or []
            if not items and contract_update is _CONTRACT_UNSET:
                return {"success": False, "error": "No tasks provided"}
            candidate_tasks = self._copy_plan_tasks(tasks)
            blocked_transitions: List[str] = []
            for item in items:
                task_id = str(item.get("id") or "").strip()
                content = str(item.get("content") or "").strip()
                status = str(item.get("status") or "pending").strip()
                depends_on = self._normalize_depends_on(item.get("depends_on"))
                if not task_id or not content:
                    return {"success": False, "error": "Task requires id and content"}
                if status not in TASK_STATUSES:
                    return {"success": False, "error": f"Invalid task status: {status}"}
                existing = next((task for task in candidate_tasks if task.get("id") == task_id), None)
                previous_status = existing.get("status") if existing else None
                if existing:
                    existing["content"] = content
                    existing["status"] = status
                    existing["depends_on"] = depends_on
                    existing["updated_at"] = int(time.time())
                else:
                    now = int(time.time())
                    candidate_tasks.append(
                        {
                            "id": task_id,
                            "content": content,
                            "status": status,
                            "depends_on": depends_on,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                if status == "blocked" and previous_status != "blocked":
                    blocked_transitions.append(task_id)
            validation_error = self._validate_plan_tasks(candidate_tasks)
            if validation_error:
                return {"success": False, "error": validation_error}
            if candidate_tasks:
                contract_validation_error = self._contract_validation_error(effective_contract)
                if contract_validation_error:
                    return self._plan_contract_repair_response(contract_validation_error, items)
            changed = bool(items) or contract_changed
            if changed:
                self._save_scope_plan(scope, candidate_tasks, contract=effective_contract)
                for task_id in blocked_transitions:
                    self._schedule_replan(scope, "blocked", task_id)
            return self._tasks_response(
                action,
                candidate_tasks,
                changed=changed,
                contract=effective_contract,
            )

        if action == "update":
            items = kwargs.get("tasks") or []
            if not items and contract_update is _CONTRACT_UNSET:
                return {"success": False, "error": "No tasks provided"}
            candidate_tasks = self._copy_plan_tasks(tasks)
            changed = contract_changed
            blocked_transitions: List[str] = []
            for item in items:
                task_id = str(item.get("id") or "").strip()
                existing = next((task for task in candidate_tasks if task.get("id") == task_id), None)
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
                        if status == "blocked":
                            blocked_transitions.append(task_id)
                if "depends_on" in item:
                    depends_on = self._normalize_depends_on(item.get("depends_on"))
                    if self._normalize_depends_on(existing.get("depends_on")) != depends_on:
                        existing["depends_on"] = depends_on
                        item_changed = True
                if item_changed:
                    existing["updated_at"] = int(time.time())
                    changed = True
            if changed:
                validation_error = self._validate_plan_tasks(candidate_tasks)
                if validation_error:
                    return {"success": False, "error": validation_error}
                self._save_scope_plan(scope, candidate_tasks, contract=effective_contract)
                for task_id in blocked_transitions:
                    self._schedule_replan(scope, "blocked", task_id)
            return self._tasks_response(
                action,
                candidate_tasks,
                changed=changed,
                contract=effective_contract,
            )

        if action == "list":
            return self._tasks_response(action, tasks, changed=False, contract=current_contract)

        if action == "clear":
            active = [task for task in tasks if task.get("status") not in CLOSED_STATUSES]
            changed = len(active) != len(tasks)
            if contract_changed:
                changed = True
            if changed:
                self._save_scope_plan(scope, active, contract=effective_contract)
            self._tool_error_streaks.pop(scope, None)
            self._pending_replan.pop(scope, None)
            return self._tasks_response(
                action,
                active,
                changed=changed,
                contract=effective_contract,
            )

        return {"success": False, "error": f"Unknown action: {action}"}

    @staticmethod
    def _plan_contract_repair_response(error: str, tasks: Any) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "retryable": True,
            "retry_instruction": (
                "Retry agent_tools.manage_plan_tasks with action='add', the same tasks, "
                "and definition_of_done containing goal, success_criteria, and verification."
            ),
            "required_definition_of_done": {
                "goal": "Concrete end state for this user request.",
                "success_criteria": ["Observable condition that means the work is done."],
                "verification": ["Specific check to run before final delivery."],
                "constraints": ["Optional constraints from the user or active skill."],
            },
            "tasks_to_retry": tasks if isinstance(tasks, list) else [],
        }

    def _tasks_response(
        self,
        action: str,
        tasks: List[Dict[str, Any]],
        *,
        changed: bool,
        contract: Dict[str, Any] | None = None,
    ) -> Dict:
        snapshot = [
            {
                "id": str(task.get("id") or ""),
                "content": str(task.get("content") or ""),
                "status": str(task.get("status") or "pending"),
                "depends_on": self._normalize_depends_on(task.get("depends_on")),
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
                "definition_of_done": contract,
                "progress": {
                    "total": total,
                    "closed": closed,
                    "open": total - closed,
                },
            },
        }

    def _delivery_plan_error(self, scope: str, status: str, verification_summary: str) -> str | None:
        if status != "completed":
            return None
        plan = self._get_scope_plan(scope)
        tasks = plan.get("tasks") or []
        open_tasks = [task for task in tasks if task.get("status") not in CLOSED_STATUSES]
        if open_tasks:
            ids = ", ".join(str(task.get("id") or "") for task in open_tasks)
            return f"Cannot deliver completed status while plan tasks are still open: {ids}"
        contract = plan.get("contract")
        if isinstance(contract, dict) and contract.get("verification") and not verification_summary:
            return "deliver_to_user completed status requires verification_summary for the active plan"
        return None

    async def _deliver_to_user(self, helper, *, request_context=None, **kwargs) -> Dict:
        text_raw = kwargs.get("text")
        final_text = "" if text_raw is None else str(text_raw).strip()
        status = str(kwargs.get("status") or "completed").strip().lower()
        if status not in {"completed", "blocked"}:
            return {"success": False, "error": "deliver_to_user status must be completed or blocked"}
        verification_summary = str(kwargs.get("verification_summary") or "").strip()
        blocked_reason = str(kwargs.get("blocked_reason") or "").strip()
        if status == "blocked" and not blocked_reason:
            return {"success": False, "error": "deliver_to_user blocked status requires blocked_reason"}
        if status == "blocked" and not final_text and blocked_reason:
            final_text = blocked_reason
        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        plan_error = self._delivery_plan_error(scope, status, verification_summary)
        if plan_error:
            return {"success": False, "error": plan_error}
        allowed_roots = self._allowed_artifact_roots(helper)
        artifact_items, error = self._normalize_delivery_artifacts(
            kwargs.get("artifacts"), allowed_roots=allowed_roots,
        )
        if error:
            return {"success": False, "error": error}
        if not final_text and not artifact_items:
            return {
                "success": False,
                "error": "deliver_to_user requires at least one of text or artifacts",
            }

        now = time.monotonic()
        dedup_key = self._delivery_dedup_key(kwargs, request_context)
        last_delivery_at = self._recent_deliveries.get(dedup_key) if dedup_key else None
        if last_delivery_at is not None and (now - last_delivery_at) < self.DELIVERY_DEDUP_WINDOW_SECONDS:
            logging.warning(
                "deliver_to_user duplicate suppressed scope=%s dedup_key=%s elapsed=%.1fs",
                scope, dedup_key, now - last_delivery_at,
            )
            return {
                "success": True,
                "skipped": True,
                "reason": "Already delivered to user in this turn; do not call deliver_to_user again.",
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
            "status": status,
            "text": final_text,
            "artifacts": artifact_items,
        }
        if verification_summary:
            direct_result["verification_summary"] = verification_summary
        if blocked_reason:
            direct_result["blocked_reason"] = blocked_reason
        if cleanup_skills:
            direct_result["cleanup_skills"] = cleanup_skills
        if dedup_key:
            self._recent_deliveries[dedup_key] = now
        return {
            "success": True,
            "text_chars": len(final_text),
            "artifacts_count": len(artifact_items),
            "direct_result": direct_result,
        }

    @staticmethod
    def _delivery_dedup_key(kwargs: Dict[str, Any], request_context=None) -> str | None:
        request_id = getattr(request_context, "request_id", None) if request_context is not None else None
        if request_id:
            return f"request:{request_id}"
        message_id = kwargs.get("message_id")
        chat_id = kwargs.get("chat_id")
        if chat_id is not None and message_id is not None:
            return f"message:{chat_id}:{message_id}"
        return None

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

        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))

        active_skills = getattr(skills_plugin, "active_skills", None) or {}
        scope_state = active_skills.get(scope) or {}
        plugin_id = getattr(skills_plugin, "plugin_id", "skills")
        return [
            {"plugin_id": plugin_id, "scope": scope, "skill_id": skill_id}
            for skill_id in scope_state
        ]

    @staticmethod
    def _allowed_artifact_roots(helper) -> List[str]:
        roots = [Path("/tmp")]
        plugin_manager = getattr(helper, "plugin_manager", None)
        storage_root = getattr(plugin_manager, "storage_root", None) if plugin_manager else None
        if storage_root:
            roots.append(Path(storage_root))
        if plugin_manager is not None:
            try:
                skills_plugin = plugin_manager.get_plugin("skills")
            except Exception:
                skills_plugin = None
            if skills_plugin is not None:
                for attr in ("skills_dir", "workdir_root"):
                    candidate = getattr(skills_plugin, attr, None)
                    if candidate:
                        roots.append(Path(candidate))
        resolved: List[str] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved_root = str(Path(root).resolve())
            except OSError:
                continue
            if resolved_root and resolved_root not in seen:
                resolved.append(resolved_root)
                seen.add(resolved_root)
        return resolved

    @classmethod
    def _normalize_delivery_artifacts(
        cls,
        artifacts: Any,
        *,
        allowed_roots: List[str] | None = None,
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

            resolved = os.path.realpath(os.path.expanduser(file_path))
            if not os.path.exists(resolved):
                return [], f"Artifact file '{file_path}' does not exist"
            if not os.path.isfile(resolved):
                return [], f"Artifact path '{file_path}' is not a file"
            if allowed_roots:
                in_allowed = any(
                    resolved == root or resolved.startswith(root + os.sep)
                    for root in allowed_roots
                )
                if not in_allowed:
                    return [], (
                        f"Artifact path '{file_path}' is outside allowed roots "
                        f"({', '.join(allowed_roots)})"
                    )
            try:
                file_size = os.path.getsize(resolved)
            except OSError as exc:
                return [], f"Failed to stat artifact '{file_path}': {exc}"
            if file_size <= 0:
                return [], f"Artifact file '{file_path}' is empty"
            if file_size > cls.DELIVERY_MAX_ARTIFACT_BYTES:
                limit_mb = cls.DELIVERY_MAX_ARTIFACT_BYTES // (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                return [], (
                    f"Artifact '{file_path}' is {actual_mb:.1f} MB which exceeds the "
                    f"Telegram delivery limit of {limit_mb} MB"
                )

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

    async def _run_subagents(self, helper, *, request_context=None, **kwargs) -> Dict:
        subagents = kwargs.get("subagents") or []
        if not isinstance(subagents, list) or not subagents:
            return {"success": False, "error": "subagents must be a non-empty list"}
        if len(subagents) > MAX_SUBAGENTS:
            return {"success": False, "error": f"At most {MAX_SUBAGENTS} subagents can run at once"}

        shared_context = str(kwargs.get("shared_context") or "").strip()
        map_reduce_contract, contract_error = self._normalize_map_reduce_contract(kwargs.get("map_reduce"))
        if contract_error:
            return {"success": False, "error": contract_error}
        inherited_skill_context = self._active_skill_context_for_subagents(helper, kwargs)
        if inherited_skill_context:
            shared_context = "\n\n".join(
                part for part in (shared_context, inherited_skill_context) if part
            )
        parent_max_rounds = self._normalize_max_rounds(kwargs.get("max_rounds"))
        parent_allowed_plugins = await self._resolve_parent_allowed_plugins(helper, request_context, kwargs)
        tasks = [
            self._run_one_subagent(
                helper, item, shared_context, kwargs, parent_max_rounds,
                request_context=request_context,
                parent_allowed_plugins=parent_allowed_plugins,
                map_reduce_contract=map_reduce_contract,
            )
            for item in subagents
        ]
        if self._bool_arg(kwargs.get("background"), default=False):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return {"success": False, "error": "No running event loop for background subagents"}
            scheduled = []
            for item, task_coro in zip(subagents, tasks):
                task = loop.create_task(task_coro)
                self._background_subagent_tasks.add(task)
                task.add_done_callback(self._background_subagent_done)
                scheduled.append(str(item.get("id") or "subagent").strip() or "subagent")
            return {
                "success": True,
                "background": True,
                "scheduled": len(scheduled),
                "subagents": scheduled,
                **(
                    {"map_reduce": self._map_reduce_summary(map_reduce_contract, [])}
                    if map_reduce_contract else {}
                ),
            }
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

        response = {
            "success": True,
            "subagents": normalized_results,
        }
        if map_reduce_contract:
            response["map_reduce"] = self._map_reduce_summary(map_reduce_contract, normalized_results)
        return response

    @staticmethod
    def _normalize_map_reduce_contract(value: Any) -> tuple[Dict[str, Any] | None, str | None]:
        if value is None:
            return None, None
        if not isinstance(value, dict):
            return None, "map_reduce must be an object"
        if value.get("enabled") is False:
            return None, None
        reduce_goal = str(value.get("reduce_goal") or "").strip()
        if not reduce_goal:
            return None, "map_reduce.reduce_goal is required when map_reduce is provided"
        merge_strategy = str(value.get("merge_strategy") or "synthesize").strip() or "synthesize"
        if merge_strategy not in _MAP_REDUCE_STRATEGIES:
            return None, f"map_reduce.merge_strategy must be one of: {', '.join(sorted(_MAP_REDUCE_STRATEGIES))}"
        worker_contract_raw = value.get("worker_output_contract")
        worker_contract: Dict[str, Any] = {}
        if isinstance(worker_contract_raw, dict):
            summary = str(worker_contract_raw.get("summary") or "").strip()
            if summary:
                worker_contract["summary"] = summary
            sections = AgentToolsPlugin._normalize_string_list(
                worker_contract_raw.get("required_sections")
            )
            if sections:
                worker_contract["required_sections"] = sections
            for key in ("evidence_required", "risk_required"):
                if key in worker_contract_raw:
                    worker_contract[key] = AgentToolsPlugin._bool_arg(worker_contract_raw.get(key))
        return {
            "enabled": True,
            "reduce_goal": reduce_goal,
            "merge_strategy": merge_strategy,
            "worker_output_contract": worker_contract,
        }, None

    @staticmethod
    def _map_reduce_summary(
        contract: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        completed = sum(1 for item in results if item.get("status") == "completed")
        errored = sum(1 for item in results if item.get("status") == "error")
        return {
            "enabled": True,
            "reduce_goal": contract.get("reduce_goal"),
            "merge_strategy": contract.get("merge_strategy", "synthesize"),
            "worker_count": len(results),
            "completed": completed,
            "errored": errored,
        }

    @staticmethod
    def _active_skill_context_for_subagents(helper, kwargs: Dict[str, Any]) -> str:
        plugin_manager = getattr(helper, "plugin_manager", None) if helper is not None else None
        if plugin_manager is None:
            return ""
        try:
            skills_plugin = plugin_manager.get_plugin("skills")
        except Exception:
            return ""
        if skills_plugin is None:
            return ""

        scope = compute_scope_key(kwargs.get("chat_id"), kwargs.get("user_id"))
        active_skills = getattr(skills_plugin, "active_skills", None) or {}
        scope_state = active_skills.get(scope) or {}
        if not scope_state:
            return ""

        disabled_skills = set()
        disabled_lookup = getattr(skills_plugin, "_disabled_skills_for_user", None)
        if callable(disabled_lookup):
            try:
                disabled_skills = set(disabled_lookup(helper, kwargs.get("user_id")) or set())
            except Exception:
                logging.debug("Failed to resolve disabled skills for subagent context", exc_info=True)

        available_skills = getattr(skills_plugin, "available_skills", None) or {}
        lines = [
            "Active skills inherited from the parent agent:",
            (
                "If a subtask belongs to one of these skills, use that skill first. "
                "Call skills.get_skill / skills.get_skill_status for full instructions before scripts or execution."
            ),
        ]
        for skill_id, state in sorted(scope_state.items()):
            if skill_id in disabled_skills:
                continue
            info = available_skills.get(skill_id, {})
            name = info.get("name") or skill_id
            description = str(info.get("description") or "").strip().replace("\n", " ")
            if len(description) > 240:
                description = description[:240].rstrip() + "..."
            line = f"- {skill_id} ({name})"
            if description:
                line += f": {description}"
            current_step = state.get("current_step")
            if current_step is not None:
                line += f"; current_step={current_step}"
            context = state.get("context")
            if context:
                try:
                    context_text = json.dumps(context, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    context_text = str(context)
                if len(context_text) > 500:
                    context_text = context_text[:500].rstrip() + "..."
                line += f"; active_context={context_text}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 2 else ""

    def _background_subagent_done(self, task: asyncio.Task) -> None:
        self._background_subagent_tasks.discard(task)
        try:
            result = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception("Background subagent failed")
            return
        logging.info(
            "Background subagent finished id=%s status=%s",
            result.get("id"),
            result.get("status"),
        )

    @staticmethod
    async def _resolve_parent_allowed_plugins(helper, request_context, kwargs: Dict[str, Any]) -> List[str]:
        resolver = getattr(helper, "resolve_allowed_plugins", None)
        if not callable(resolver):
            return ["All"]
        chat_id = (
            getattr(request_context, "chat_id", None) if request_context is not None else None
        ) or kwargs.get("chat_id")
        session_id = getattr(request_context, "session_id", None) if request_context is not None else None
        user_id = (
            getattr(request_context, "user_id", None) if request_context is not None else None
        )
        if user_id is None:
            user_id = kwargs.get("user_id")
        try:
            allowed = await resolver(chat_id, session_id, user_id) if chat_id is not None else ["All"]
        except Exception:
            logging.exception("Failed to resolve parent allowed_plugins for subagents")
            return ["All"]
        return list(allowed) if allowed else ["All"]

    async def _run_one_subagent(
        self,
        helper,
        item: Dict[str, Any],
        shared_context: str,
        kwargs: Dict[str, Any],
        parent_max_rounds: int,
        *,
        request_context=None,
        parent_allowed_plugins: List[str] | None = None,
        map_reduce_contract: Dict[str, Any] | None = None,
    ) -> Dict:
        subagent_id = str(item.get("id") or "").strip()
        role = str(item.get("role") or "").strip()
        task = str(item.get("task") or "").strip()
        context = str(item.get("context") or "").strip()
        map_key = str(item.get("map_key") or subagent_id or "subagent").strip()
        expected_output = str(item.get("expected_output") or "").strip()
        if not subagent_id or not role or not task:
            return {
                "id": subagent_id or "subagent",
                "role": role,
                "status": "error",
                "error": "Subagent requires id, role, and task",
            }

        user_id = kwargs.get("user_id")
        override_model = str(item.get("model") or "").strip() or None
        if override_model and override_model not in LLMGATEWAY_CHAT_MODELS:
            logging.warning(
                "Subagent %s requested unsupported model %r; falling back to parent model",
                subagent_id,
                override_model,
            )
            override_model = None
        if override_model:
            model_to_use = override_model
        else:
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

        task_prompt = (
            f"Subagent id: {subagent_id}\n"
            f"Role: {role}\n"
            f"Task: {task}\n\n"
            f"Shared context:\n{shared_context or '(none)'}\n\n"
            f"Subagent-specific context:\n{context or '(none)'}"
        )
        if map_reduce_contract:
            worker_contract = map_reduce_contract.get("worker_output_contract") or {}
            contract_lines = [
                "",
                "Map-reduce contract:",
                "- You are a map worker, not the reducer.",
                f"- Map key: {map_key}",
                f"- Reduce goal: {map_reduce_contract.get('reduce_goal')}",
                f"- Merge strategy: {map_reduce_contract.get('merge_strategy', 'synthesize')}",
                "- Return findings for the parent reducer. Do not synthesize other workers.",
            ]
            if expected_output:
                contract_lines.append(f"- Worker-specific expected output: {expected_output}")
            if worker_contract:
                contract_lines.append(
                    "- Worker output contract: "
                    + json.dumps(worker_contract, ensure_ascii=False, sort_keys=True)
                )
            task_prompt += "\n".join(contract_lines)

        messages = [
            {
                "role": "system",
                "content": _load_subagent_system_prompt(),
            },
            {
                "role": "user",
                "content": task_prompt,
            },
        ]
        status = "error"
        try:
            result_text = await self._run_subagent_completion_loop(
                helper, model_to_use, messages, kwargs, max_rounds, temperature,
                published=published,
                request_context=request_context,
                parent_allowed_plugins=parent_allowed_plugins,
            )
            status = "completed"
            result = {
                "id": subagent_id,
                "role": role,
                "status": status,
                "result": result_text,
                "published": published,
            }
            if map_reduce_contract or item.get("map_key"):
                result["map_key"] = map_key
            return result
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
        return max(MIN_SUBAGENT_TOOL_ROUNDS, min(requested, MAX_SUBAGENT_TOOL_ROUNDS))

    @staticmethod
    def _normalize_temperature(value: Any) -> float:
        if value is None:
            return DEFAULT_SUBAGENT_TEMPERATURE
        try:
            requested = float(value)
        except (TypeError, ValueError):
            return DEFAULT_SUBAGENT_TEMPERATURE
        return max(0.0, min(requested, 2.0))

    @staticmethod
    def _bool_arg(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

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
        request_context=None,
        parent_allowed_plugins: List[str] | None = None,
    ) -> str:
        tools, allowed_function_names = self._subagent_tools(
            helper, model_to_use, parent_allowed_plugins=parent_allowed_plugins,
        )
        repeated_fingerprints: set[str] = set()
        fingerprint_counts: Dict[str, int] = {}
        last_fingerprint = ""
        consecutive_count = 0

        for round_index in range(max_rounds + 1):
            is_final_round = round_index == max_rounds
            request_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": helper.get_output_max_tokens(),
                "stream": False,
            }
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "none" if is_final_round else "auto"

            response = await helper.chat_completion(**request_kwargs)
            choice = response.choices[0]
            tool_calls = self._extract_tool_calls(choice)
            if tool_calls and not is_final_round:
                messages.append(self._assistant_tool_calls_message(choice, tool_calls))
                tool_responses: List[str | None] = [None] * len(tool_calls)
                pending_calls: List[tuple[int, Dict[str, str]]] = []
                for index, call in enumerate(tool_calls):
                    fingerprint = self._tool_call_fingerprint(call)
                    fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
                    if fingerprint == last_fingerprint:
                        consecutive_count += 1
                    else:
                        last_fingerprint = fingerprint
                        consecutive_count = 1
                    repeated = (
                        consecutive_count >= SUBAGENT_CONSECUTIVE_REPEAT_LIMIT
                        or fingerprint_counts[fingerprint] >= SUBAGENT_TOTAL_REPEAT_LIMIT
                    )
                    if repeated:
                        error = self._repeated_tool_call_error(
                            call,
                            consecutive_count=consecutive_count,
                            total_count=fingerprint_counts[fingerprint],
                        )
                        if fingerprint in repeated_fingerprints:
                            raise RuntimeError(error)
                        repeated_fingerprints.add(fingerprint)
                        tool_responses[index] = json.dumps(
                            {
                                "success": False,
                                "stuck_loop": True,
                                "error": error,
                                "instruction": "Choose a different tool call, change the arguments, or stop and report the blocker.",
                            },
                            ensure_ascii=False,
                        )
                        continue
                    pending_calls.append((index, call))
                if pending_calls:
                    pending_responses = await asyncio.gather(*[
                        self._call_subagent_tool(
                            helper,
                            call,
                            allowed_function_names,
                            kwargs,
                            published=published,
                            request_context=request_context,
                        )
                        for _index, call in pending_calls
                    ])
                    for (index, _call), response in zip(pending_calls, pending_responses):
                        tool_responses[index] = response
                for call, tool_response in zip(tool_calls, tool_responses):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._tool_result_content(helper, tool_response or ""),
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

    @staticmethod
    def _tool_call_fingerprint(call: Dict[str, str]) -> str:
        raw_arguments = call.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_arguments)
            normalized_arguments = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except json.JSONDecodeError:
            normalized_arguments = raw_arguments.strip()
        return f"{call.get('name') or ''}:{normalized_arguments}"

    @staticmethod
    def _repeated_tool_call_error(
        call: Dict[str, str],
        *,
        consecutive_count: int,
        total_count: int,
    ) -> str:
        return (
            f"Repeated identical tool call detected for {call.get('name')}: "
            f"{consecutive_count} consecutive repeats, {total_count} total repeats. "
            "Choose different arguments or stop and report the blocker."
        )

    def _subagent_tools(
        self,
        helper,
        model_to_use: str,
        *,
        parent_allowed_plugins: List[str] | None = None,
    ) -> tuple[Any, set[str]]:
        plugin_manager = getattr(helper, "plugin_manager", None)
        if not plugin_manager:
            return None, set()

        get_subagent_specs = getattr(plugin_manager, "get_subagent_function_specs", None)
        if not callable(get_subagent_specs):
            logging.warning(
                "plugin_manager has no get_subagent_function_specs; subagents disabled"
            )
            return None, set()

        filtered_tools, allowed_function_names = get_subagent_specs(
            helper,
            model_to_use,
            parent_allowed_plugins=parent_allowed_plugins,
            blocked_function_names=SUBAGENT_BLOCKED_FUNCTIONS,
        )
        if not filtered_tools and not allowed_function_names:
            logging.warning(
                "Subagents have no callable tools (model=%s); running without tools",
                model_to_use,
            )
            return None, set()

        filtered_tools = list(filtered_tools)
        filtered_tools.append(self._internal_publish_tool_spec())
        allowed_function_names = set(allowed_function_names)
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
        request_context=None,
    ) -> str:
        tool_name = call["name"]
        if tool_name == "agent_tools.internal_publish":
            try:
                args = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                return json.dumps({"error": f"Invalid arguments for {tool_name}"}, ensure_ascii=False)
            return self._handle_internal_publish(args, published)
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
            request_context=request_context,
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
            return {"success": False, "error": self.t("agent_tools_question_required")}

        chat_id = kwargs.get("chat_id")
        if chat_id is None:
            return {"success": False, "error": self.t("agent_tools_chat_id_required_for_telegram")}
        chat_id = int(chat_id)

        bot = getattr(helper, "bot", None) or getattr(getattr(self, "openai", None), "bot", None)
        if not bot:
            return {"success": False, "error": self.t("agent_tools_bot_not_configured")}

        self._clear_done_question_for_chat(chat_id)
        if chat_id in self.pending_by_chat:
            return {"success": False, "error": self.t("agent_tools_question_already_waiting")}

        options = self._normalize_options(kwargs.get("options") or [])
        if not options:
            return {"success": False, "error": self.t("agent_tools_options_required")}
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
                        text=self.t("agent_tools_question_timeout_notice"),
                        reply_to_message_id=message_id,
                    )
                except Exception:
                    logging.debug(
                        "Failed to send ask_telegram_user timeout notification",
                        exc_info=True,
                    )
            return {"success": False, "error": self.t("agent_tools_question_timeout_error")}
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
        try:
            default = int(os.getenv("AGENT_ASK_USER_TIMEOUT_SECONDS", "1800"))
        except (TypeError, ValueError):
            default = 1800
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
            await message.reply_text(self.t("agent_tools_answer_received"))

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
            await query.answer(self.t("agent_tools_question_expired"), show_alert=True)
            return
        user = getattr(query, "from_user", None)
        if not self._answerer_allowed(pending, getattr(user, "id", None)):
            await query.answer(self.t("agent_tools_wrong_user"), show_alert=True)
            return
        if answer_key == "text":
            await query.answer(self.t("agent_tools_send_answer_as_message"))
            return

        options = pending.get("options") or []
        multi_select = bool(pending.get("multi_select"))

        if multi_select and answer_key == "confirm":
            selected = pending.get("selected_indices") or set()
            if not selected:
                await query.answer(self.t("agent_tools_select_at_least_one"), show_alert=True)
                return
            chosen = [str(options[i]) for i in sorted(selected) if 0 <= i < len(options)]
            answer = ", ".join(chosen)
            if not self._resolve_question(question_id, answer):
                await query.answer(self.t("agent_tools_question_expired"), show_alert=True)
                return
            await query.answer(self.t("agent_tools_answer_received"))
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
            await query.answer(self.t("agent_tools_invalid_answer"), show_alert=True)
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
            await query.answer(self.t("agent_tools_question_expired"), show_alert=True)
            return
        await query.answer(self.t("agent_tools_answer_received"))
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logging.debug("Failed to clear ask_user markup", exc_info=True)

    async def _cancel_pending_question(self, helper, **kwargs) -> Dict:
        chat_id = kwargs.get("chat_id")
        if chat_id is None:
            return {"success": False, "error": self.t("agent_tools_chat_id_required")}
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return {"success": False, "error": self.t("agent_tools_invalid_chat_id")}

        question_id = self.pending_by_chat.get(chat_id_int)
        if not question_id:
            return {"success": False, "error": self.t("agent_tools_no_pending_question")}
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
                        text=self.t("agent_tools_question_cancelled", reason=reason),
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
