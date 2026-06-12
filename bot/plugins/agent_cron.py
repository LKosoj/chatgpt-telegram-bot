from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from telegram import Update
from telegram.ext import ContextTypes

from ..agent_delivery import send_agent_response, send_text_chunks
from ..request_context import RequestContext
from ..utils import compute_scope_key, get_thread_id, message_text
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


class AgentCronPlugin(Plugin):
    plugin_id = "agent_cron"
    function_prefix = "agent_cron"

    def __init__(self):
        self.jobs_file = os.path.join(os.path.dirname(__file__), "agent_cron_jobs.json")
        self.jobs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._checker_task: asyncio.Task | None = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._load_jobs()

    def get_source_name(self) -> str:
        return "Agent Cron"

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            self.jobs_file = os.path.join(storage_root, "agent_cron_jobs.json")
            self._load_jobs()

    async def on_startup(self, application) -> None:
        if self._checker_task is None or self._checker_task.done():
            self._checker_task = application.create_task(self._checker_loop(application.bot))

    def close(self) -> None:
        if self._checker_task and not self._checker_task.done():
            self._checker_task.cancel()
        for task in list(self._running_tasks.values()):
            task.cancel()
        self._running_tasks.clear()

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
        chat_id = int(kwargs.get("chat_id") or kwargs.get("user_id"))
        user_id = int(kwargs.get("user_id") or chat_id)
        job = self._create_job(chat_id=chat_id, user_id=user_id, schedule=schedule, prompt=prompt, parsed=parsed)
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
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else chat_id
        args_text = message_text(message).strip()
        if not args_text or args_text == "list":
            await message.reply_text(self._format_jobs(chat_id, user_id))
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
            job = self._create_job(
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
        job = (self.jobs.get(scope) or {}).get(job_id)
        if not job:
            await message.reply_text("Cron job not found.")
            return
        if action == "pause":
            job["paused"] = True
            self._save_jobs()
            await message.reply_text(f"Cron job `{job_id}` paused.", parse_mode="Markdown")
            return
        if action == "resume":
            job["paused"] = False
            if self._parse_iso(job.get("next_run_at")) is None:
                parsed = self._parse_schedule(job.get("schedule", ""))
                if parsed:
                    job.update(parsed)
            self._save_jobs()
            await message.reply_text(f"Cron job `{job_id}` resumed.", parse_mode="Markdown")
            return
        if action == "remove":
            del self.jobs[scope][job_id]
            self._save_jobs()
            await message.reply_text(f"Cron job `{job_id}` removed.", parse_mode="Markdown")
            return
        task = asyncio.create_task(self._run_job(bot, scope, job_id, manual=True))
        self._running_tasks[job_id] = task
        task.add_done_callback(lambda _task, jid=job_id: self._running_tasks.pop(jid, None))
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
        self._load_jobs()
        now = datetime.now()
        for scope, jobs in list(self.jobs.items()):
            for job_id, job in list(jobs.items()):
                if job.get("paused") or job.get("status") == "running":
                    continue
                next_run = self._parse_iso(job.get("next_run_at"))
                if next_run and next_run <= now and job_id not in self._running_tasks:
                    task = asyncio.create_task(self._run_job(bot, scope, job_id))
                    self._running_tasks[job_id] = task
                    task.add_done_callback(lambda _task, jid=job_id: self._running_tasks.pop(jid, None))

    async def _run_job(self, bot, scope: str, job_id: str, *, manual: bool = False) -> None:
        job = (self.jobs.get(scope) or {}).get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["last_started_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_jobs()
        try:
            helper = getattr(self, "openai", None)
            if not helper or not hasattr(helper, "get_chat_response"):
                raise RuntimeError("OpenAI helper is not available for agent cron")
            request_context = RequestContext(
                chat_id=int(job["chat_id"]),
                user_id=int(job["user_id"]),
                request_id=f"agent_cron_{job_id}",
            )
            response, total_tokens = await helper.get_chat_response(
                chat_id=int(job["chat_id"]),
                query=str(job["prompt"]),
                request_id=f"agent_cron_{job_id}",
                user_id=int(job["user_id"]),
                request_context=request_context,
            )
            live = (self.jobs.get(scope) or {}).get(job_id)
            if live is None:
                return
            job = live
            job["status"] = "active"
            job["last_finished_at"] = datetime.now().isoformat(timespec="seconds")
            job["last_error"] = ""
            job["last_tokens"] = total_tokens
            if not manual:
                self._advance_job(job)
            self._save_jobs()
            await send_agent_response(
                bot,
                chat_id=int(job["chat_id"]),
                response=response,
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
                title=f"Cron job `{job_id}` completed.",
            )
        except Exception as exc:
            logger.exception("Agent cron job %s failed", job_id)
            live = (self.jobs.get(scope) or {}).get(job_id)
            if live is None:
                return
            job = live
            job["status"] = "failed"
            job["last_error"] = str(exc)
            if not manual:
                self._advance_job(job)
            self._save_jobs()
            await send_text_chunks(
                bot,
                chat_id=int(job["chat_id"]),
                text=f"Cron job `{job_id}` failed: {exc}",
                reply_to_message_id=job.get("reply_to_message_id"),
                message_thread_id=job.get("message_thread_id"),
            )

    def _load_jobs(self) -> None:
        try:
            if not os.path.exists(self.jobs_file):
                self.jobs = {}
                return
            with open(self.jobs_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.jobs = data if isinstance(data, dict) else {}
            changed = False
            for jobs in self.jobs.values():
                if not isinstance(jobs, dict):
                    continue
                for job_id, job in jobs.items():
                    if isinstance(job, dict) and job.get("status") == "running":
                        if job_id not in self._running_tasks:
                            job["status"] = "active"
                            changed = True
            if changed:
                self._save_jobs()
        except Exception:
            logger.exception("Failed to load agent cron jobs")
            self.jobs = {}

    def _save_jobs(self) -> None:
        try:
            jobs_dir = os.path.dirname(self.jobs_file)
            if jobs_dir:
                os.makedirs(jobs_dir, exist_ok=True)
            tmp_path = self.jobs_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self.jobs, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.jobs_file)
        except Exception:
            logger.exception("Failed to save agent cron jobs")

    def _create_job(
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
        self.jobs.setdefault(scope, {})[job_id] = job
        self._save_jobs()
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

    def _format_jobs(self, chat_id: int, user_id: int) -> str:
        scope = compute_scope_key(chat_id=chat_id, user_id=user_id)
        jobs = list((self.jobs.get(scope) or {}).values())
        if not jobs:
            return self._usage()
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        lines = ["Agent cron jobs:"]
        for job in jobs[:20]:
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
