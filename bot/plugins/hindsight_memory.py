from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from .plugin import Plugin
from ..hindsight_client import HindsightClient, format_recall_results
from ..utils import message_text

logger = logging.getLogger(__name__)


HINDSIGHT_MEMORY_MARKER = "[HINDSIGHT_MEMORY_CONTEXT]"

HINDSIGHT_CONTEXT_PROMPT = f"""{HINDSIGHT_MEMORY_MARKER}
Long-term memory recalled for this Telegram user:
{{memory}}

Use this only as background context when it is relevant. If the current user message
contradicts this memory, prefer the current message. Do not mention Hindsight or memory
retrieval unless the user asks about it."""


class HindsightMemoryPlugin(Plugin):
    plugin_id = "hindsight_memory"
    function_prefix = "hindsight_memory"

    def get_config_prefix(self) -> str | None:
        return "hindsight_"

    def initialize(self, openai=None, bot=None, storage_root: str | None = None,
                   db=None, plugin_config=None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        # Plugin-owned config slice (prefix "hindsight_"). Defaults set HERE.
        self.config: Dict[str, Any] = dict(plugin_config or {})
        self.config.setdefault('hindsight_base_url', '')
        self.config.setdefault('hindsight_api_token', '')
        self.config['hindsight_enabled'] = bool(
            self.config.get('hindsight_base_url')
            and self.config.get('hindsight_api_token')
        )
        self.config.setdefault('hindsight_auto_recall', True)
        self.config.setdefault('hindsight_auto_save', True)
        self.config.setdefault('hindsight_namespace', 'default')
        self.config.setdefault('hindsight_bank_prefix', 'telegram-')
        self.config.setdefault('hindsight_recall_budget', 'mid')
        self.config.setdefault('hindsight_recall_max_tokens', 4096)
        self.config.setdefault('hindsight_memory_types', 'world,experience')
        self.config.setdefault('hindsight_async_store', True)
        self.config.setdefault('hindsight_timeout', 30.0)
        self.config.setdefault('hindsight_max_auto_save_items', 5)
        # Stage 4A: mirror defaults into openai.config so remaining helper-side
        # readers (_prepare_*/finalize_*) keep working until removed in 4B/4C.
        # setdefault (not assignment) — never overwrite an already-set value.
        if openai is not None and getattr(openai, "config", None) is not None:
            for key in (
                'hindsight_base_url', 'hindsight_api_token',
                'hindsight_enabled', 'hindsight_auto_recall', 'hindsight_auto_save',
                'hindsight_namespace', 'hindsight_bank_prefix',
                'hindsight_recall_budget', 'hindsight_recall_max_tokens',
                'hindsight_memory_types', 'hindsight_async_store',
                'hindsight_timeout', 'hindsight_max_auto_save_items',
            ):
                openai.config.setdefault(key, self.config[key])

        self.client: HindsightClient | None = None
        if self.config['hindsight_enabled']:
            self.client = HindsightClient(
                self.config.get('hindsight_base_url', ''),
                self.config.get('hindsight_api_token', ''),
                namespace=self.config.get('hindsight_namespace', 'default'),
                timeout=float(self.config.get('hindsight_timeout', 30.0)),
            )

    @property
    def is_active(self) -> bool:
        cfg = getattr(self, "config", None) or {}
        return bool(
            cfg.get('hindsight_enabled')
            and cfg.get('hindsight_api_token')
            and self.client is not None
            and self.client.enabled
        )

    @property
    def auto_recall_enabled(self) -> bool:
        return bool((getattr(self, "config", None) or {}).get('hindsight_auto_recall', True))

    @property
    def auto_save_enabled(self) -> bool:
        return bool((getattr(self, "config", None) or {}).get('hindsight_auto_save', True))

    def bank_id_for(self, user_id: int | str) -> str:
        prefix = (getattr(self, "config", None) or {}).get('hindsight_bank_prefix', 'telegram-')
        return f"{prefix}{user_id}"

    @property
    def memory_types(self) -> list[str]:
        value = (getattr(self, "config", None) or {}).get('hindsight_memory_types', 'world,experience')
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return ['world', 'experience']

    @staticmethod
    def is_hindsight_memory_message(message: Dict[str, Any]) -> bool:
        if not isinstance(message, dict):
            return False
        if message.get("role") != "system":
            return False
        content = message.get("content")
        return isinstance(content, str) and content.startswith(HINDSIGHT_MEMORY_MARKER)

    def close(self) -> None:
        """No-op: client teardown handled by OpenAIHelper.close() via property
        delegation in Stage 4A; moves to plugin lifecycle in Stage 4C."""
        return None

    async def on_before_chat_request(
        self,
        messages: List[Dict[str, Any]],
        payload: Any,
    ) -> List[Dict[str, Any]] | None:
        """Inject Hindsight long-term memory as a system message before chat request.

        Returns ``None`` for "no change" (inactive/disabled/cached/empty/failed).
        """
        if not self.is_active or not self.auto_recall_enabled:
            return None
        user_id = getattr(payload, "user_id", None)
        if user_id is None:
            return None
        if any(self.is_hindsight_memory_message(msg) for msg in messages):
            return None
        last_user = next(
            (msg for msg in reversed(messages)
             if isinstance(msg, dict) and msg.get("role") == "user"),
            None,
        )
        if not last_user:
            return None
        query = last_user.get("content")
        if not isinstance(query, str) or not query.strip():
            return None

        try:
            data = await self.client.recall(
                self.bank_id_for(user_id),
                query,
                budget=self.config.get('hindsight_recall_budget', 'mid'),
                max_tokens=int(self.config.get('hindsight_recall_max_tokens', 4096)),
                memory_types=self.memory_types,
                trace=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Hindsight recall failed for user_id=%s: %s", user_id, exc,
            )
            return None

        memory = format_recall_results(data) if data else None
        if not memory:
            return None

        memory_message = {
            "role": "system",
            "content": HINDSIGHT_CONTEXT_PROMPT.format(memory=memory),
        }
        new_messages = list(messages)
        insert_at = 0
        for msg in new_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                insert_at += 1
            else:
                break
        new_messages.insert(insert_at, memory_message)
        logger.info("Hindsight recalled memory for bank %s", self.bank_id_for(user_id))
        return new_messages

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
        if not self.is_active:
            return {"error": "Hindsight memory is not configured."}

        user_id = kwargs.get("user_id")
        if not user_id:
            return {"error": "Telegram user_id is required for Hindsight memory."}

        bank_id = self.bank_id_for(user_id)
        client = self.client

        if function_name == "recall":
            data = await client.recall(
                bank_id,
                str(kwargs["query"]),
                budget=str(kwargs.get("budget") or self.config.get('hindsight_recall_budget', 'mid')),
                max_tokens=int(kwargs.get("max_tokens") or self.config.get('hindsight_recall_max_tokens', 4096)),
                memory_types=self.memory_types,
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
        if not self.is_active:
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
        if not self.is_active:
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
        bank_id = self.bank_id_for(user_id)
        count_text = await self._memory_count_text(bank_id)
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
        bank_id = self.bank_id_for(user_id)
        try:
            data = await self.client.recall(
                bank_id,
                query,
                budget=self.config.get('hindsight_recall_budget', 'mid'),
                max_tokens=int(self.config.get('hindsight_recall_max_tokens', 4096)),
                memory_types=self.memory_types,
            )
            summary = format_recall_results(data) or "No matching memories found."
        except Exception as exc:
            summary = f"Memory search failed: {exc}"
        await message.reply_text(summary, parse_mode=None)

    async def _send_memory_export(self, message, helper, user_id: int) -> None:
        bank_id = self.bank_id_for(user_id)
        try:
            data = await self.client.list_memories(bank_id, limit=1000, offset=0)
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
        bank_id = self.bank_id_for(user_id)
        await self.client.clear_bank(bank_id)

    async def _memory_count_text(self, bank_id: str) -> str:
        try:
            stats = await self.client.stats(bank_id)
            count = self._stats_memory_count(stats)
            if count is not None:
                return str(count)
        except Exception:
            pass

        try:
            data = await self.client.list_memories(bank_id, limit=1000, offset=0)
            count = self._stats_memory_count(data)
            if count is not None:
                return str(count)
        except Exception:
            return "unknown"
        return "unknown"

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
        for nested_key in ("memory", "stats", "summary", "pagination", "meta"):
            nested = stats.get(nested_key)
            if isinstance(nested, dict):
                candidates.extend([
                    nested.get("memory_count"),
                    nested.get("memories_count"),
                    nested.get("total_memories"),
                    nested.get("count"),
                    nested.get("total"),
                ])
        for value in candidates:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        for list_key in ("items", "memories", "results", "data"):
            value = stats.get(list_key)
            if isinstance(value, list):
                return len(value)
        return None
