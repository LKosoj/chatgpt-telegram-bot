from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, cast

from telegram import Update
from telegram.ext import ContextTypes

from ..agent_delivery import send_agent_response, send_text_chunks
from ..request_context import RequestContext
from ..utils import compute_scope_key, get_thread_id, message_text
from .hooks import AssistantResponsePayload
from .plugin import Plugin


logger = logging.getLogger(__name__)


WEEKDAYS = {
    "mon": 0, "monday": 0, "понедельник": 0, "пн": 0,
    "tue": 1, "tuesday": 1, "вторник": 1, "вт": 1,
    "wed": 2, "wednesday": 2, "среда": 2, "ср": 2,
    "thu": 3, "thursday": 3, "четверг": 3, "чт": 3,
    "fri": 4, "friday": 4, "пятница": 4, "пт": 4,
    "sat": 5, "saturday": 5, "суббота": 5, "сб": 5,
    "sun": 6, "sunday": 6, "воскресенье": 6, "вс": 6,
}

# Lease window for a claimed-but-not-finished cron job. A run is a full agent
# turn (helper.get_chat_response, possibly several tool-call rounds), longer
# than a single hindsight-extraction lease (900s), so this is sized with extra
# headroom rather than copied from that unrelated worker.
AGENT_CRON_JOB_LEASE_SECONDS = 1800


class AgentCronPlugin(Plugin):
    plugin_id = "agent_cron"
    function_prefix = "agent_cron"

    def __init__(self):
        self.jobs_file = os.path.join(os.path.dirname(__file__), "agent_cron_jobs.json")
        self.db_handle: Any = None
        self._checker_task: asyncio.Task | None = None
        self._running_tasks: Dict[str, asyncio.Task] = {}

    def get_source_name(self) -> str:
        return "Agent Cron"

    def initialize(self, openai=None, bot=None, storage_root: str | None = None,
                    db=None, plugin_config=None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        self.db_handle = db
        if storage_root:
            self.jobs_file = os.path.join(storage_root, "agent_cron_jobs.json")
        if self.db_handle is not None:
            self.db_handle.run_sync_blocking(self._import_json_jobs_sync)

    async def on_startup(self, application) -> None:
        if self._checker_task is None or self._checker_task.done():
            self._checker_task = application.create_task(self._checker_loop(application.bot))

    def close(self) -> None:
        if self._checker_task and not self._checker_task.done():
            self._checker_task.cancel()
        for task in list(self._running_tasks.values()):
            task.cancel()
        self._running_tasks.clear()

    def register_schema(self) -> List[str]:
        return [
            '''
            CREATE TABLE IF NOT EXISTS agent_cron_jobs (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                schedule TEXT NOT NULL,
                prompt TEXT NOT NULL,
                schedule_type TEXT NOT NULL,
                next_run_at TEXT,
                interval_seconds INTEGER,
                hour INTEGER,
                minute INTEGER,
                weekday INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                paused INTEGER NOT NULL DEFAULT 0,
                reply_to_message_id INTEGER,
                message_thread_id INTEGER,
                created_at TEXT NOT NULL,
                last_started_at TEXT,
                last_finished_at TEXT,
                last_error TEXT,
                last_tokens INTEGER,
                locked_at TEXT,
                locked_by TEXT
            )
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_agent_cron_jobs_due
                ON agent_cron_jobs(paused, next_run_at)
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_agent_cron_jobs_scope
                ON agent_cron_jobs(scope)
            ''',
        ]

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "create_cron_job",
            "description": (
                "Schedule an agent task for this Telegram chat. Use only when the user explicitly "
                "asks for recurring or delayed agent work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule": {"type": "string", "description": "Natural schedule text, e.g. in 10 minutes, daily at 09:00."},
                    "prompt": {"type": "string", "description": "Agent task prompt to run at the scheduled time."},
                },
                "required": ["schedule", "prompt"],
            },
        }]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        if function_name != "create_cron_job":
            return {"error": f"Unknown agent cron function: {function_name}"}
        schedule = str(kwargs.get("schedule") or "").strip()
        prompt = str(kwargs.get("prompt") or "").strip()
        parsed = self._parse_schedule(schedule)
        if not parsed:
            return {"error": self._usage()}
        chat_id = int(cast(int, kwargs.get("chat_id") or kwargs.get("user_id")))
        user_id = int(kwargs.get("user_id") or chat_id)
        job = await self._create_job(chat_id=chat_id, user_id=user_id, schedule=schedule, prompt=prompt, parsed=parsed)
        return {"direct_result": {"kind": "text", "format": "markdown", "value": self._format_created(job)}}

    def get_commands(self) -> List[Dict]:
        return [{
            "command": "cron",
            "description": "Schedule, list, pause, resume, run, or remove agent tasks.",
            "handler": self.handle_cron_command,
            "handler_kwargs": {},
            "add_to_menu": True,
        }]

    async def handle_cron_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        if not message:
            return
        chat_id = message.chat_id
        user_id = update.effective_user.id if update.effective_user else chat_id
        args_text = message_text(message).strip()
        if not args_text or args_text == "list":
            await message.reply_text(await self._format_jobs(chat_id, user_id))
            return

        action, _, rest = args_text.partition(" ")
        action = action.lower().strip()
        if action == "add":
            schedule, sep, prompt = rest.partition("|")
            if not sep or not prompt.strip():
                await message.reply_text(self._usage(), parse_mode="Markdown")
                return
            parsed = self._parse_schedule(schedule.strip())
            if not parsed:
                await message.reply_text(self._usage(), parse_mode="Markdown")
                return
            job = await self._create_job(
                chat_id=chat_id,
                user_id=user_id,
                schedule=schedule.strip(),
                prompt=prompt.strip(),
                parsed=parsed,
                reply_to_message_id=message.message_id,
                message_thread_id=get_thread_id(update),
            )
            await message.reply_text(self._format_created(job), parse_mode="Markdown")
            return

        if action in {"pause", "resume", "remove", "run"}:
            await self._handle_job_action(action, rest.strip(), chat_id, user_id, context.bot, message)
            return

        await message.reply_text(self._usage(), parse_mode="Markdown")

    async def _handle_job_action(self, action: str, job_id: str, chat_id: int, user_id: int, bot, message) -> None:
        scope = compute_scope_key(chat_id=chat_id, user_id=user_id)
        job = await self.db_handle.fetch_one(
            "SELECT * FROM agent_cron_jobs WHERE id = ? AND scope = ?", (job_id, scope)
        )
        if not job:
            await message.reply_text("Cron job not found.")
            return
        if action == "pause":
            await self.db_handle.execute(
                "UPDATE agent_cron_jobs SET paused = 1 WHERE id = ? AND scope = ?", (job_id, scope)
            )
            await message.reply_text(f"Cron job `{job_id}` paused.", parse_mode="Markdown")
            return
        if action == "resume":
            if self._parse_iso(job.get("next_run_at")) is None:
                parsed = self._parse_schedule(job.get("schedule", ""))
                if parsed:
                    await self.db_handle.execute(
                        '''UPDATE agent_cron_jobs
                           SET paused = 0, schedule_type = ?, next_run_at = ?, interval_seconds = ?,
                               hour = ?, minute = ?, weekday = ?
                           WHERE id = ? AND scope = ?''',
                        (
                            parsed.get("schedule_type"), parsed.get("next_run_at"), parsed.get("interval_seconds"),
                            parsed.get("hour"), parsed.get("minute"), parsed.get("weekday"), job_id, scope,
                        ),
                    )
                    await message.reply_text(f"Cron job `{job_id}` resumed.", parse_mode="Markdown")
                    return
            await self.db_handle.execute(
                "UPDATE agent_cron_jobs SET paused = 0 WHERE id = ? AND scope = ?", (job_id, scope)
            )
            await message.reply_text(f"Cron job `{job_id}` resumed.", parse_mode="Markdown")
            return
        if action == "remove":
            await self.db_handle.execute(
                "DELETE FROM agent_cron_jobs WHERE id = ? AND scope = ?", (job_id, scope)
            )
            await message.reply_text(f"Cron job `{job_id}` removed.", parse_mode="Markdown")
            return
        task = asyncio.create_task(self._run_job(bot, scope, job_id, manual=True))
        self._running_tasks[job_id] = task
        task.add_done_callback(lambda _task: self._running_tasks.pop(job_id, None))
        await message.reply_text(f"Cron job `{job_id}` queued for manual run.", parse_mode="Markdown")

    async def _checker_loop(self, bot) -> None:
        while True:
            try:
                await self._check_due_jobs(bot)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Agent cron checker failed")
            await asyncio.sleep(60)

    async def _check_due_jobs(self, bot) -> None:
        if self.db_handle is None:
            return
        now_iso, lease_cutoff_iso, worker_id = self._lease_params()
        # Exclude ids already tracked in _running_tasks (e.g. a manual /cron run
        # queued moments ago, before it has performed its own DB claim) so the
        # automatic path never claims a row it isn't going to execute — see W1.
        claimed = await self.db_handle.run_sync(
            self._claim_due_jobs_sync, now_iso, lease_cutoff_iso, worker_id,
            frozenset(self._running_tasks.keys()),
        )
        for job in claimed:
            job_id = job["id"]
            if job_id in self._running_tasks:
                continue
            task = asyncio.create_task(self._run_job(bot, job["scope"], job_id))
            self._running_tasks[job_id] = task

            def _forget_cron_job(_task: object, jid: Any = job_id) -> None:
                self._running_tasks.pop(jid, None)

            task.add_done_callback(_forget_cron_job)

    @staticmethod
    def _lease_params() -> tuple[str, str, str]:
        now = datetime.now()
        now_iso = now.isoformat(timespec="seconds")
        lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
        worker_id = str(os.getpid())
        return now_iso, lease_cutoff_iso, worker_id

    def _claim_due_jobs_sync(
        self, db, now_iso: str, lease_cutoff_iso: str, worker_id: str,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                '''
                SELECT id FROM agent_cron_jobs
                WHERE paused = 0
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                  AND (status != 'running' OR locked_at IS NULL OR locked_at <= ?)
                ORDER BY next_run_at ASC
                ''',
                (now_iso, lease_cutoff_iso),
            )
            job_ids = [row[0] for row in cursor.fetchall() if row[0] not in exclude_ids]
            if not job_ids:
                return []
            placeholders = ",".join("?" for _ in job_ids)
            cursor.execute(
                f'''UPDATE agent_cron_jobs
                    SET status='running', locked_at=?, locked_by=?, last_started_at=?
                    WHERE id IN ({placeholders})''',
                (now_iso, worker_id, now_iso, *job_ids),
            )
            cursor.execute(f"SELECT * FROM agent_cron_jobs WHERE id IN ({placeholders})", job_ids)
            rows = [dict(r) for r in cursor.fetchall()]
        order = {jid: i for i, jid in enumerate(job_ids)}
        rows.sort(key=lambda r: order.get(r["id"], 0))
        return rows

    def _claim_job_by_id_sync(
        self, db, job_id: str, scope: str, now_iso: str, lease_cutoff_iso: str, worker_id: str
    ) -> Dict[str, Any] | None:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                '''UPDATE agent_cron_jobs
                   SET status='running', locked_at=?, locked_by=?, last_started_at=?
                   WHERE id = ? AND scope = ?
                     AND (status != 'running' OR locked_at IS NULL OR locked_at <= ?)''',
                (now_iso, worker_id, now_iso, job_id, scope, lease_cutoff_iso),
            )
            if cursor.rowcount == 0:
                return None
            cursor.execute("SELECT * FROM agent_cron_jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    async def _run_job(self, bot, scope: str, job_id: str, *, manual: bool = False) -> None:
        now_iso, lease_cutoff_iso, worker_id = self._lease_params()
        if manual:
            job = await self.db_handle.run_sync(
                self._claim_job_by_id_sync, job_id, scope, now_iso, lease_cutoff_iso, worker_id
            )
        else:
            job = await self.db_handle.fetch_one(
                "SELECT * FROM agent_cron_jobs WHERE id = ? AND scope = ?", (job_id, scope)
            )
        if not job:
            return
        try:
            helper = getattr(self, "openai", None)
            if not helper or not hasattr(helper, "get_chat_response"):
                raise RuntimeError("OpenAI helper is not available for agent cron")
            request_context = RequestContext(
                chat_id=int(job["chat_id"]),
                user_id=int(job["user_id"]),
                request_id=f"agent_cron_{job_id}",
                autonomous=True,
            )
            response, total_tokens = await helper.get_chat_response(
                chat_id=int(job["chat_id"]),
                query=str(job["prompt"]),
                request_id=f"agent_cron_{job_id}",
                user_id=int(job["user_id"]),
                request_context=request_context,
            )
            live = await self.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id = ?", (job_id,))
            if live is None:
                return
            job = live
            job["status"] = "active"
            job["last_finished_at"] = datetime.now().isoformat(timespec="seconds")
            job["last_error"] = ""
            job["last_tokens"] = total_tokens
            if not manual:
                self._advance_job(job)
            await self._finish_job(job)
            await self._maybe_dispatch_autonomous_response_hook(helper, job, job_id, response, total_tokens)
            await send_agent_response(
                bot,
                chat_id=int(job["chat_id"]),
                response=response,
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
                title=f"Cron job `{job_id}` completed.",
                config=getattr(helper, "config", None),
            )
        except Exception as exc:
            logger.exception("Agent cron job %s failed", job_id)
            live = await self.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id = ?", (job_id,))
            if live is None:
                return
            job = live
            job["status"] = "failed"
            job["last_error"] = str(exc)
            if not manual:
                self._advance_job(job)
            await self._finish_job(job)
            await send_text_chunks(
                bot,
                chat_id=int(job["chat_id"]),
                text=f"Cron job `{job_id}` failed: {exc}",
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
                config=getattr(locals().get("helper"), "config", None),
            )

    async def _finish_job(self, job: Dict[str, Any]) -> None:
        await self.db_handle.execute(
            '''UPDATE agent_cron_jobs
               SET status = ?, last_finished_at = ?, last_error = ?, last_tokens = ?,
                   paused = ?, next_run_at = ?, locked_at = NULL, locked_by = NULL
               WHERE id = ?''',
            (
                job.get("status"), job.get("last_finished_at"), job.get("last_error"),
                job.get("last_tokens"), int(bool(job.get("paused"))), job.get("next_run_at"),
                job["id"],
            ),
        )

    @staticmethod
    def _autonomous_capture_enabled() -> bool:
        return os.environ.get("HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED", "false").strip().lower() == "true"

    async def _maybe_dispatch_autonomous_response_hook(
        self, helper, job: Dict[str, Any], job_id: str, response: Any, total_tokens: int,
    ) -> None:
        """Fire the standard on_assistant_response observer hook for this cron turn,
        marked autonomous=True. Cron calls helper.get_chat_response() directly and
        never goes through telegram_bot.py, which is the only place this hook is
        normally dispatched from — without this, memory-capture plugins never see
        cron turns at all. Off by default (HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED):
        turning it on is a behavior change an operator must opt into.
        A dispatch failure must not turn a successful cron run into a failed one,
        so it is caught and logged rather than left to propagate to the caller.
        """
        if not self._autonomous_capture_enabled():
            return
        if not isinstance(response, str) or not response:
            return
        plugin_manager = getattr(helper, "plugin_manager", None)
        if plugin_manager is None:
            return
        try:
            user_id = int(job["user_id"])
            await plugin_manager.dispatch_observe(
                "on_assistant_response",
                AssistantResponsePayload(
                    chat_id=int(job["chat_id"]),
                    user_id=user_id,
                    request_id=f"agent_cron_{job_id}",
                    text=response,
                    tokens=int(total_tokens or 0),
                    model=str(getattr(helper, "config", {}).get("model", "")),
                    ts=time.time(),
                    autonomous=True,
                ),
                user_id=user_id,
            )
        except Exception:
            logger.exception("Agent cron autonomous response hook dispatch failed for job_id=%s", job_id)

    def _import_json_jobs_sync(self, db) -> None:
        with db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM agent_cron_jobs").fetchone()[0]
        if count > 0:
            # Table already has data: either import already ran, or the
            # process crashed after the INSERT commit below but before the
            # rename. Finish the rename so the legacy file doesn't linger
            # forever, but never re-import (idempotent).
            if os.path.exists(self.jobs_file):
                os.replace(self.jobs_file, self.jobs_file + ".migrated")
            return
        if not os.path.exists(self.jobs_file):
            return
        try:
            with open(self.jobs_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            logger.exception("Failed to read agent_cron_jobs.json for import")
            return
        if not isinstance(data, dict):
            return
        rows = []
        for scope, jobs in data.items():
            if not isinstance(jobs, dict):
                continue
            for job_id, job in jobs.items():
                if not isinstance(job, dict):
                    continue
                status = job.get("status")
                rows.append((
                    job_id, scope, job.get("chat_id"), job.get("user_id"),
                    job.get("schedule", ""), job.get("prompt", ""),
                    job.get("schedule_type", "once"), job.get("next_run_at"),
                    job.get("interval_seconds"), job.get("hour"), job.get("minute"),
                    job.get("weekday"), "active" if status == "running" else (status or "active"),
                    int(bool(job.get("paused"))), job.get("reply_to_message_id"),
                    job.get("message_thread_id"), job.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                    job.get("last_started_at"), job.get("last_finished_at"),
                    job.get("last_error"), job.get("last_tokens"), None, None,
                ))
        with db.get_connection() as conn:
            conn.executemany(
                '''INSERT OR IGNORE INTO agent_cron_jobs (
                    id, scope, chat_id, user_id, schedule, prompt, schedule_type, next_run_at,
                    interval_seconds, hour, minute, weekday, status, paused, reply_to_message_id,
                    message_thread_id, created_at, last_started_at, last_finished_at, last_error,
                    last_tokens, locked_at, locked_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                rows,
            )
        os.replace(self.jobs_file, self.jobs_file + ".migrated")

    async def _create_job(
        self,
        *,
        chat_id: int,
        user_id: int,
        schedule: str,
        prompt: str,
        parsed: Dict[str, Any],
        reply_to_message_id: int | None = None,
        message_thread_id: int | None = None,
    ) -> Dict[str, Any]:
        scope = compute_scope_key(chat_id=chat_id, user_id=user_id)
        job_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
        job = {
            "id": job_id,
            "scope": scope,
            "chat_id": chat_id,
            "user_id": user_id,
            "schedule": schedule,
            "prompt": prompt,
            "status": "active",
            "paused": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reply_to_message_id": reply_to_message_id,
            "message_thread_id": message_thread_id,
            **parsed,
        }
        await self.db_handle.execute(
            '''INSERT INTO agent_cron_jobs (
                id, scope, chat_id, user_id, schedule, prompt, schedule_type, next_run_at,
                interval_seconds, hour, minute, weekday, status, paused, reply_to_message_id,
                message_thread_id, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (
                job["id"], job["scope"], job["chat_id"], job["user_id"], job["schedule"], job["prompt"],
                job.get("schedule_type"), job.get("next_run_at"), job.get("interval_seconds"),
                job.get("hour"), job.get("minute"), job.get("weekday"), job["status"],
                int(job["paused"]), job.get("reply_to_message_id"), job.get("message_thread_id"),
                job["created_at"],
            ),
        )
        return job

    def _advance_job(self, job: Dict[str, Any]) -> None:
        now = datetime.now()
        kind = job.get("schedule_type")
        if kind == "once":
            job["paused"] = True
            job["next_run_at"] = None
            return
        if kind == "interval":
            seconds = int(job.get("interval_seconds") or 0)
            if seconds <= 0:
                logger.error("Agent cron job %s has invalid interval_seconds=%r; pausing", job.get("id"), seconds)
                job["paused"] = True
                job["next_run_at"] = None
                return
            next_run = self._parse_iso(job.get("next_run_at")) or now
            while next_run <= now:
                next_run += timedelta(seconds=seconds)
            job["next_run_at"] = next_run.isoformat(timespec="seconds")
            return
        if kind == "daily":
            hour, minute = int(job["hour"]), int(job["minute"])
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            job["next_run_at"] = next_run.isoformat(timespec="seconds")
            return
        if kind == "weekly":
            hour, minute, weekday = int(job["hour"]), int(job["minute"]), int(job["weekday"])
            days_ahead = (weekday - now.weekday()) % 7
            next_run = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=7)
            job["next_run_at"] = next_run.isoformat(timespec="seconds")

    async def _format_jobs(self, chat_id: int, user_id: int) -> str:
        scope = compute_scope_key(chat_id=chat_id, user_id=user_id)
        jobs = await self.db_handle.fetch_all(
            "SELECT * FROM agent_cron_jobs WHERE scope = ? ORDER BY created_at DESC LIMIT 20", (scope,)
        )
        if not jobs:
            return self._usage()
        lines = ["Agent cron jobs:"]
        for job in jobs:
            prompt = str(job.get("prompt") or "").replace("\n", " ")
            if len(prompt) > 70:
                prompt = prompt[:67] + "..."
            state = "paused" if job.get("paused") else job.get("status", "active")
            lines.append(f"- `{job['id']}` {state}; next: {job.get('next_run_at') or '-'}; {prompt}")
        return "\n".join(lines)

    @staticmethod
    def _format_created(job: Dict[str, Any]) -> str:
        return (
            f"Cron job `{job['id']}` scheduled.\n"
            f"Next run: `{job.get('next_run_at')}`\n"
            f"Use `/cron pause {job['id']}`, `/cron resume {job['id']}`, "
            f"`/cron run {job['id']}`, or `/cron remove {job['id']}`."
        )

    @staticmethod
    def _usage() -> str:
        return (
            "Usage:\n"
            "`/cron add in 10 minutes | summarize this chat`\n"
            "`/cron add daily at 09:00 | send me a morning brief`\n"
            "`/cron add every 2 hours | check the project status`\n"
            "`/cron list`\n"
            "`/cron pause <job_id>` / `/cron resume <job_id>` / `/cron run <job_id>` / `/cron remove <job_id>`"
        )

    def _parse_schedule(self, schedule: str) -> Dict[str, Any] | None:
        text = str(schedule or "").strip().lower()
        now = datetime.now()
        exact = self._parse_exact_datetime(text)
        if exact:
            return {"schedule_type": "once", "next_run_at": exact.isoformat(timespec="seconds")}

        match = re.search(r"(?:in|через)\s+(\d+)\s+([a-zа-я]+)", text)
        if match:
            seconds = self._unit_seconds(match.group(2))
            if seconds:
                next_run = now + timedelta(seconds=int(match.group(1)) * seconds)
                return {"schedule_type": "once", "next_run_at": next_run.isoformat(timespec="seconds")}

        match = re.search(r"(?:every|каждые|каждый|каждую)\s+(\d+)\s+([a-zа-я]+)", text)
        if match:
            seconds = self._unit_seconds(match.group(2))
            if seconds:
                interval_seconds = int(match.group(1)) * seconds
                if interval_seconds <= 0:
                    return None
                next_run = now + timedelta(seconds=interval_seconds)
                return {
                    "schedule_type": "interval",
                    "interval_seconds": interval_seconds,
                    "next_run_at": next_run.isoformat(timespec="seconds"),
                }

        match = re.search(r"(?:daily|every day|ежедневно|каждый день)(?:\s+(?:at|в))?\s+(\d{1,2})(?::(\d{2}))?", text)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2) or 0)
            return self._daily_schedule(hour, minute, now)

        match = re.search(r"(?:tomorrow|завтра)(?:\s+(?:at|в))?\s+(\d{1,2})(?::(\d{2}))?", text)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2) or 0)
            next_run = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            return {"schedule_type": "once", "next_run_at": next_run.isoformat(timespec="seconds")}

        match = re.search(r"(?:weekly|еженедельно)\s+([a-zа-я]+)(?:\s+(?:at|в))?\s+(\d{1,2})(?::(\d{2}))?", text)
        if match:
            weekday = WEEKDAYS.get(match.group(1))
            if weekday is not None:
                return self._weekly_schedule(weekday, int(match.group(2)), int(match.group(3) or 0), now)
        return None

    @staticmethod
    def _unit_seconds(unit: str) -> int | None:
        unit = unit.lower()
        if unit.startswith(("min", "мин")):
            return 60
        if unit.startswith(("hour", "час")):
            return 3600
        if unit.startswith(("day", "дн", "ден")):
            return 86400
        return None

    @staticmethod
    def _parse_exact_datetime(text: str) -> datetime | None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_iso(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @staticmethod
    def _daily_schedule(hour: int, minute: int, now: datetime) -> Dict[str, Any] | None:
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return {
            "schedule_type": "daily",
            "hour": hour,
            "minute": minute,
            "next_run_at": next_run.isoformat(timespec="seconds"),
        }

    @staticmethod
    def _weekly_schedule(weekday: int, hour: int, minute: int, now: datetime) -> Dict[str, Any] | None:
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        days_ahead = (weekday - now.weekday()) % 7
        next_run = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=7)
        return {
            "schedule_type": "weekly",
            "weekday": weekday,
            "hour": hour,
            "minute": minute,
            "next_run_at": next_run.isoformat(timespec="seconds"),
        }
