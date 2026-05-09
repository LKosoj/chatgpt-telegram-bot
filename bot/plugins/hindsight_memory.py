from __future__ import annotations

import io
import json
from typing import Any, Dict, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from .plugin import Plugin
from ..hindsight_client import format_recall_results
from ..utils import message_text


class HindsightMemoryPlugin(Plugin):
    plugin_id = "hindsight_memory"
    function_prefix = "hindsight_memory"

    def get_source_name(self) -> str:
        return "Hindsight Memory"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "recall",
                "description": "Search the current Telegram user's Hindsight long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Memory search query."},
                        "budget": {
                            "type": "string",
                            "enum": ["low", "mid", "high"],
                            "description": "Recall budget. Use mid by default.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum recall payload size in tokens.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "list_memories",
                "description": "List recently saved memories for the current Telegram user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum memories to return."},
                        "offset": {"type": "integer", "description": "Pagination offset."},
                        "query": {"type": "string", "description": "Optional text filter."},
                        "memory_type": {
                            "type": "string",
                            "description": "Optional memory type filter, such as world or experience.",
                        },
                    },
                },
            },
            {
                "name": "stats",
                "description": "Get Hindsight memory statistics for the current Telegram user.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def get_commands(self) -> List[Dict]:
        return [
            {
                "command": "memory",
                "description": "Show, search, export, or clear Hindsight memory.",
                "handler": self.handle_memory_command,
                "handler_kwargs": {},
                "add_to_menu": True,
            },
            {
                "callback_query_handler": self.handle_memory_callback,
                "callback_pattern": "^memory:",
                "handler_kwargs": {},
                "add_to_menu": False,
            },
        ]

    async def execute(self, function_name: str, helper: Any, **kwargs) -> Dict:
        if not getattr(helper, "is_hindsight_enabled", lambda: False)():
            return {"error": "Hindsight memory is not configured."}

        user_id = kwargs.get("user_id")
        if not user_id:
            return {"error": "Telegram user_id is required for Hindsight memory."}

        bank_id = helper.get_hindsight_bank_id(user_id)
        client = helper.hindsight_client

        if function_name == "recall":
            data = await client.recall(
                bank_id,
                str(kwargs["query"]),
                budget=str(kwargs.get("budget") or helper.config.get("hindsight_recall_budget", "mid")),
                max_tokens=int(kwargs.get("max_tokens") or helper.config.get("hindsight_recall_max_tokens", 4096)),
                memory_types=helper._hindsight_memory_types(),
            )
            return {
                "bank_id": bank_id,
                "summary": format_recall_results(data),
                "results": data.get("results", []),
            }

        if function_name == "list_memories":
            return await client.list_memories(
                bank_id,
                limit=int(kwargs.get("limit") or 20),
                offset=int(kwargs.get("offset") or 0),
                query=kwargs.get("query"),
                memory_type=kwargs.get("memory_type"),
            )

        if function_name == "stats":
            return await client.stats(bank_id)

        return {"error": f"Unknown Hindsight memory function: {function_name}"}

    async def handle_memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        user_id = update.effective_user.id if update.effective_user else None
        if not message or not user_id:
            return
        helper = getattr(self, "openai", None)
        if not self._is_hindsight_available(helper):
            await message.reply_text("Hindsight memory is not configured.")
            return

        args_text = message_text(message).strip()
        action, _, rest = args_text.partition(" ")
        action = action.lower()
        if action == "search" and rest.strip():
            await self._send_memory_search(message, helper, user_id, rest.strip())
            return
        if action == "export":
            await self._send_memory_export(message, helper, user_id)
            return
        if action == "clear":
            await message.reply_text(
                "Clear all Hindsight memories for this Telegram user?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Clear memory", callback_data="memory:clear_confirm")],
                    [InlineKeyboardButton("Cancel", callback_data="memory:status")],
                ]),
            )
            return
        await message.reply_text(
            await self._memory_status_text(helper, user_id),
            reply_markup=self._memory_menu(),
            parse_mode="Markdown",
        )

    async def handle_memory_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        user_id = update.effective_user.id if update.effective_user else None
        helper = getattr(self, "openai", None)
        if not query or not user_id:
            return
        await query.answer()
        if not self._is_hindsight_available(helper):
            await query.edit_message_text("Hindsight memory is not configured.")
            return
        action = str(query.data or "memory:status").split(":", 1)[1]
        if action == "close":
            await query.message.delete()
            return
        if action == "export":
            await self._send_memory_export(query.message, helper, user_id)
            return
        if action == "clear":
            await query.edit_message_text(
                "Clear all Hindsight memories for this Telegram user?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Clear memory", callback_data="memory:clear_confirm")],
                    [InlineKeyboardButton("Cancel", callback_data="memory:status")],
                ]),
            )
            return
        if action == "clear_confirm":
            await self._clear_memory(helper, user_id)
            await query.edit_message_text(
                await self._memory_status_text(helper, user_id),
                reply_markup=self._memory_menu(),
                parse_mode="Markdown",
            )
            return
        await query.edit_message_text(
            await self._memory_status_text(helper, user_id),
            reply_markup=self._memory_menu(),
            parse_mode="Markdown",
        )

    @staticmethod
    def _is_hindsight_available(helper) -> bool:
        return bool(helper and getattr(helper, "is_hindsight_enabled", lambda: False)())

    @staticmethod
    def _memory_menu() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Refresh", callback_data="memory:status"),
                InlineKeyboardButton("Export", callback_data="memory:export"),
            ],
            [InlineKeyboardButton("Clear", callback_data="memory:clear")],
            [InlineKeyboardButton("Close", callback_data="memory:close")],
        ])

    async def _memory_status_text(self, helper, user_id: int) -> str:
        bank_id = helper.get_hindsight_bank_id(user_id)
        try:
            stats = await helper.hindsight_client.stats(bank_id)
            count = self._stats_memory_count(stats)
        except Exception as exc:
            return f"Hindsight memory is enabled for bank `{bank_id}`.\nStats failed: {exc}"
        count_text = str(count) if count is not None else "unknown"
        return (
            f"Hindsight memory is enabled.\n"
            f"Bank: `{bank_id}`\n"
            f"Memories: `{count_text}`\n\n"
            "Commands:\n"
            "`/memory search <query>`\n"
            "`/memory export`\n"
            "`/memory clear`"
        )

    async def _send_memory_search(self, message, helper, user_id: int, query: str) -> None:
        bank_id = helper.get_hindsight_bank_id(user_id)
        try:
            data = await helper.hindsight_client.recall(
                bank_id,
                query,
                budget=helper.config.get("hindsight_recall_budget", "mid"),
                max_tokens=int(helper.config.get("hindsight_recall_max_tokens", 4096)),
                memory_types=helper._hindsight_memory_types(),
            )
            summary = format_recall_results(data) or "No matching memories found."
        except Exception as exc:
            summary = f"Memory search failed: {exc}"
        await message.reply_text(summary, parse_mode=None)

    async def _send_memory_export(self, message, helper, user_id: int) -> None:
        bank_id = helper.get_hindsight_bank_id(user_id)
        try:
            data = await helper.hindsight_client.list_memories(bank_id, limit=1000, offset=0)
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            file_obj = io.BytesIO(payload)
            file_obj.name = f"hindsight_{bank_id}.json"
            await message.reply_document(
                document=file_obj,
                filename=file_obj.name,
                caption=f"Hindsight memory export for {bank_id}",
            )
        except Exception as exc:
            await message.reply_text(f"Memory export failed: {exc}")

    async def _clear_memory(self, helper, user_id: int) -> None:
        bank_id = helper.get_hindsight_bank_id(user_id)
        await helper.hindsight_client.clear_bank(bank_id)

    @staticmethod
    def _stats_memory_count(stats: Dict[str, Any]) -> int | None:
        candidates = [
            stats.get("memory_count"),
            stats.get("memories_count"),
            stats.get("total_memories"),
            stats.get("count"),
        ]
        memories = stats.get("memories")
        if isinstance(memories, dict):
            candidates.extend([
                memories.get("count"),
                memories.get("total"),
            ])
        for value in candidates:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None
