from __future__ import annotations

import datetime
import io
import json
import logging
import uuid
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

HINDSIGHT_EXTRACTOR_PROMPT = """Extract durable memories from the Telegram conversation.
Return only JSON in this exact shape:
{"items":[{"content":"...","context":"...","tags":["..."]}]}

Save only facts that are clearly durable and likely useful in future conversations with the same user:
- explicit "remember this" facts;
- stable user preferences, identity details, long-term goals, ongoing projects, durable constraints, and decisions;
- important project facts or agreements that should survive across sessions.

Do not infer preferences from weak signals. For example, do not save "user prefers Russian"
only because the conversation is in Russian; save it only if the user explicitly says it or clearly corrects the assistant.

Do not save one-off tasks or transient requests: image generation/editing requests, uploaded-image descriptions,
audio/transcription requests, web searches, debugging logs, single SQL questions, generic chit-chat, or temporary commands.
Do not save passwords, API keys, tokens, secrets, credentials, private auth data, or facts contradicted inside the exchange.

Examples to reject:
- User asked to draw or edit a cat with a hat.
- User asked what is shown in an uploaded image.
- User asked one isolated technical question and got an answer.

When in doubt, save nothing.
If there is nothing worth saving, return {"items":[]}."""


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

    def register_schema(self) -> List[str]:
        return [
            '''
            CREATE TABLE IF NOT EXISTS hindsight_finalize_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER DEFAULT NULL,
                last_error TEXT DEFAULT NULL,
                locked_at TIMESTAMP DEFAULT NULL,
                next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_hindsight_finalize_jobs_status_next_attempt
            ON hindsight_finalize_jobs(status, next_attempt_at, created_at)
            ''',
        ]

    async def finalize_session_memory(
        self,
        user_id: int,
        session_id: str | None,
        messages: list[dict[str, Any]],
        *,
        raise_on_error: bool = False,
        async_store: bool | None = None,
    ) -> int:
        if not session_id or not self.is_active or not self.auto_save_enabled:
            return 0
        try:
            transcript = self._session_transcript_for_hindsight(messages)
            if not transcript:
                return 0
            items = await self._extract_hindsight_memory_items(transcript)
            if not items:
                return 0
            await self._retain_hindsight_items(
                user_id=user_id,
                chat_id=user_id,
                session_id=session_id,
                items=items,
                mode="session_close",
                document_id=f"telegram-{user_id}-{session_id}-final",
                async_store=async_store,
            )
            return len(items)
        except Exception as e:
            logger.warning(
                "Hindsight session finalize failed for user_id=%s session_id=%s: %s",
                user_id, session_id, e,
            )
            if raise_on_error:
                raise
            return 0

    def _session_transcript_for_hindsight(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if isinstance(content, list):
                text = json.dumps(content, ensure_ascii=False)
            else:
                text = str(content or "")
            text = text.strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n\n".join(lines)

    async def _retain_hindsight_items(
        self,
        chat_id: int,
        user_id: int,
        items: list[dict[str, Any]],
        session_id: str | None,
        mode: str,
        document_id: str | None = None,
        async_store: bool | None = None,
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bank_id = self.bank_id_for(user_id)
        normalized = []
        for item in items:
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
            for tag in ("telegram", "auto_memory", f"user:{user_id}"):
                if tag not in tags:
                    tags.append(tag)

            normalized.append({
                "content": item["content"],
                "context": item.get("context") or "Auto-extracted from a Telegram bot conversation.",
                "document_id": document_id or f"telegram-{user_id}-{session_id or chat_id}-{uuid.uuid4().hex}",
                "timestamp": now,
                "tags": tags,
                "metadata": {
                    "source": "telegram_bot",
                    "chat_id": str(chat_id),
                    "user_id": str(user_id),
                    "session_id": str(session_id or ""),
                    "mode": mode,
                },
            })

        await self.client.retain_memories(
            bank_id,
            normalized,
            async_store=bool(self.config.get('hindsight_async_store', True)) if async_store is None else async_store,
        )
        logger.info("Saved %s Hindsight memory item(s) to bank %s", len(normalized), bank_id)

    async def _extract_hindsight_memory_items(self, transcript: str) -> list[dict[str, Any]]:
        from ..openai_helper import _first_choice_or_raise, LLMGATEWAY_LIGHT_MODEL

        messages = [
            {"role": "system", "content": HINDSIGHT_EXTRACTOR_PROMPT},
            {
                "role": "user",
                "content": (
                    "<session_transcript>\n"
                    f"{transcript}\n"
                    "</session_transcript>"
                ),
            },
        ]
        response = await self.openai.client.chat.completions.create(
            model=self.openai.config.get('light_model', LLMGATEWAY_LIGHT_MODEL),
            messages=messages,
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"},
            stream=False,
            extra_headers={"X-Title": "tgBot"},
        )
        content = _first_choice_or_raise(response).message.content or ""
        items = self._parse_hindsight_memory_items(content)
        if not items:
            logger.info("Hindsight extractor returned no memory items. content_preview=%r", content[:300])
        return items

    def _parse_hindsight_memory_items(self, content: str) -> list[dict[str, Any]]:
        text = (content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []

        raw_items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(raw_items, list):
            return []

        max_items = int(self.config.get('hindsight_max_auto_save_items', 5))
        items = []
        for item in raw_items[:max_items]:
            if not isinstance(item, dict):
                continue
            content_text = str(item.get("content") or "").strip()
            if not content_text or self._looks_sensitive_memory(content_text):
                continue
            parsed = {
                "content": content_text,
                "context": str(item.get("context") or "").strip(),
            }
            tags = item.get("tags")
            if isinstance(tags, list):
                parsed["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
            items.append(parsed)
        return items

    def _looks_sensitive_memory(self, content: str) -> bool:
        lowered = content.lower()
        sensitive_markers = (
            "api key",
            "api_key",
            "bearer ",
            "password",
            "secret",
            "token",
            "credential",
            "пароль",
            "секрет",
            "токен",
            "ключ api",
            "sk-",
            "ai-serv-",
        )
        return any(marker in lowered for marker in sensitive_markers)

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

    async def contribute_prompt_fragment(self, slot: str, payload: Any) -> Any | None:
        if not self.is_active:
            return None
        if slot == "stats_block":
            user_id = getattr(payload, "user_id", None)
            if user_id is None:
                return None
            bank_id = self.bank_id_for(user_id)
            try:
                count_text = await self._memory_count_text(bank_id)
                return f"\nHindsight memory: enabled; bank `{bank_id}`; memories `{count_text}`.\n"
            except Exception as exc:
                return f"\nHindsight memory: enabled; bank `{bank_id}`; stats failed: {exc}\n"
        if slot == "settings_menu_buttons":
            return [[InlineKeyboardButton("Hindsight memory", callback_data="memory:status")]]
        return None

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
