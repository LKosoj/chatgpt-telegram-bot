from __future__ import annotations

import asyncio
import logging
import os
import io
import tempfile
import time
import json
import sys
import yaml
import re
from typing import Dict
from functools import lru_cache 
import httpx

from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand, ForceReply
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext, TypeHandler

from pydub import AudioSegment
from PIL import Image

from .utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files, send_long_response_as_file, BusyStatusMessage, direct_result_inline_fallback_text, \
    should_send_text_as_file
from .openai_helper import OpenAIHelper, O_MODELS, ANTHROPIC, GOOGLE, MISTRALAI, DEEPSEEK, PERPLEXITY
from .plugins.hooks import AssistantResponsePayload, HookEvent, SessionBeforeDeletePayload, SessionResetPayload, SettingsMenuPayload, StatsBlockPayload, UserMessagePayload
from .i18n import DEFAULT_LANGUAGE, is_auto_language, language_name, localized_text, normalize_language, set_current_language, supported_languages
from .plugins.haiper_image_to_video import WAITING_PROMPT
from .usage_tracker import UsageTracker
from .database import Database
from .conversation_key import get_conversation_key
from .request_context import RequestContext
from .user_settings import (
    USER_DISABLED_PLUGINS_SETTING,
    USER_DISABLED_SKILLS_SETTING,
    USER_LANGUAGE_SETTING,
    USER_TTS_MODEL_SETTING,
    USER_TTS_VOICE_SETTING,
    get_user_settings,
    normalize_string_list,
    set_disabled_value,
)

#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WAITING_PROMPT = 1
DEFAULT_TELEGRAM_BASE_URL = 'http://localhost:8081/bot'
LANGUAGE_MENU_PAGE_SIZE = 10
SETTINGS_MENU_PAGE_SIZE = 10


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper, db: Database):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        # Добавляем словарь для буферизации сообщений
        self.message_buffer = {}
        # Добавляем время ожидания для буфера (в секундах)
        self.buffer_timeout = 1.0

        self.config = config
        self.db = db
        self.openai = openai
        self.openai.bot = None
        self._user_language_cache = {}
        # Tests construct the bot without going through __main__, so re-wire
        # PluginManager here too. Both setters are idempotent.
        self.openai.plugin_manager.set_db(self.db)
        self.openai.plugin_manager.set_openai(self.openai)
        
        # Кешируем chat_modes.yml
        self._chat_modes_cache = None
        self._chat_modes_cache_time = 0
        bot_language = self.config['bot_language']
        self.commands = [
            BotCommand(command='help', description=localized_text('help_description', bot_language)),
            BotCommand(command='reset', description=localized_text('reset_description', bot_language)),
            BotCommand(command='stats', description=localized_text('stats_description', bot_language)),
            BotCommand(command='resend', description=localized_text('resend_description', bot_language)),
            BotCommand(command='plugins', description=localized_text('plugins_description', bot_language)),
            BotCommand(command='settings', description=localized_text('settings_description', bot_language)),
        ]
        # If imaging is enabled, add the "image" command to the list
        if self.config.get('enable_image_generation', False):
            self.commands.append(
                BotCommand(command='image', description=localized_text('image_description', bot_language)))

        if self.config.get('enable_tts_generation', False):
            self.commands.append(BotCommand(command='tts', description=localized_text('tts_description', bot_language)))

        self.group_commands = [BotCommand(
            command='chat', description=localized_text('chat_description', bot_language)
        )] + self.commands
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self._inline_cache_cleanup_time = 0  # Время последней очистки кеша
        self.buffer_lock = asyncio.Lock()  # Добавьте блокировку для потокобезопасности
        self._conversation_locks = {}
        self._conversation_locks_guard = asyncio.Lock()
        self.application = None
        # Убираем повторную инициализацию Database
        self.plugin_command_index = {}
        self.plugin_menu_entries = []
        self.plugin_menu_page_size = int(os.getenv("PLUGIN_MENU_PAGE_SIZE", "8"))
        self._background_tasks = []
        self._cleanup_called = False
        self._plugin_message_handlers_registered = False
        self._message_tail_handlers_registered = False

    def _is_auto_language_enabled(self) -> bool:
        return is_auto_language(self.config.get('bot_language', DEFAULT_LANGUAGE))

    def _configured_language(self) -> str:
        if self._is_auto_language_enabled():
            return DEFAULT_LANGUAGE
        return normalize_language(self.config.get('bot_language', DEFAULT_LANGUAGE))

    def _detect_user_language(self, update: Update) -> str:
        user = getattr(update, 'effective_user', None)
        return normalize_language(getattr(user, 'language_code', None))

    def _get_user_language(self, update: Update) -> str:
        user = getattr(update, 'effective_user', None)
        user_id = getattr(user, 'id', None)
        if user_id is None:
            return self._configured_language()

        cached_language = self._user_language_cache.get(user_id)
        if cached_language:
            return cached_language

        settings = self.db.get_user_settings(user_id) or {}
        if not isinstance(settings, dict):
            settings = {}

        language = settings.get(USER_LANGUAGE_SETTING)
        if language:
            language = normalize_language(language)
        elif self._is_auto_language_enabled():
            language = self._detect_user_language(update)
            settings[USER_LANGUAGE_SETTING] = language
            self.db.save_user_settings(user_id, settings)
        else:
            language = self._configured_language()

        self._user_language_cache[user_id] = language
        return language

    async def prepare_user_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        set_current_language(self._get_user_language(update))

    def get_chat_modes(self):
        """
        Получает chat_modes с кешированием
        """
        current_time = time.time()
        
        # Если кеш устарел (старше 5 минут) или не существует, перезагружаем
        if (self._chat_modes_cache is None or 
            current_time - self._chat_modes_cache_time > 300):
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            chat_modes_path = os.path.join(current_dir, 'chat_modes.yml')
            
            try:
                with open(chat_modes_path, 'r', encoding='utf-8') as file:
                    self._chat_modes_cache = yaml.safe_load(file)
                    self._chat_modes_cache_time = current_time
            except Exception as e:
                logger.error(f"Ошибка загрузки chat_modes.yml: {e}")
                if self._chat_modes_cache is None:
                    self._chat_modes_cache = {}
        
        return self._chat_modes_cache

    def _image_file_id_from_message(self, message):
        if not message:
            return None
        if getattr(message, 'photo', None):
            return message.photo[-1].file_id
        document = getattr(message, 'document', None)
        if document and document.mime_type and document.mime_type.startswith('image/'):
            return document.file_id
        return None

    def _document_file_from_message(self, message):
        if not message:
            return None
        document = getattr(message, 'document', None)
        if not document:
            return None
        mime_type = getattr(document, 'mime_type', None) or ''
        if mime_type.startswith('image/'):
            return None
        return document

    @staticmethod
    def _safe_reply_file_name(document) -> str:
        fallback = f"telegram-file-{getattr(document, 'file_unique_id', '') or getattr(document, 'file_id', 'file')}"
        raw_name = os.path.basename(getattr(document, 'file_name', None) or fallback).strip()
        raw_name = raw_name or fallback
        safe_name = re.sub(r"[^\w.\- ]+", "_", raw_name, flags=re.UNICODE).strip(" ._")
        return safe_name or fallback

    async def _download_replied_file_for_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> dict | None:
        message = update.effective_message
        replied_message = getattr(message, 'reply_to_message', None)
        document = self._document_file_from_message(replied_message)
        if not document:
            return None

        file_name = self._safe_reply_file_name(document)
        temp_dir = tempfile.mkdtemp(prefix="telegram_reply_file_")
        local_path = os.path.join(temp_dir, file_name)
        try:
            telegram_file = await context.bot.get_file(document.file_id)
            await telegram_file.download_to_drive(local_path)
        except Exception:
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                os.rmdir(temp_dir)
            except OSError:
                logger.debug("Could not clean up failed replied file download at %s", local_path, exc_info=True)
            raise
        logger.info(
            "Downloaded replied Telegram file for model use: file_id=%s path=%s",
            document.file_id,
            local_path,
        )
        return {
            "local_path": local_path,
            "file_name": getattr(document, 'file_name', None) or file_name,
            "mime_type": getattr(document, 'mime_type', None) or "unknown",
            "file_size": getattr(document, 'file_size', None),
        }

    @staticmethod
    def _prompt_with_replied_file_context(prompt: str, file_context: dict) -> str:
        size = file_context.get("file_size")
        size_text = str(size) if size is not None else "unknown"
        return (
            f"{prompt}\n\n"
            "Telegram reply context:\n"
            "The user replied to a file. It has been downloaded locally for this request.\n"
            f"- local_path: {file_context['local_path']}\n"
            f"- file_name: {file_context['file_name']}\n"
            f"- mime_type: {file_context['mime_type']}\n"
            f"- size_bytes: {size_text}\n\n"
            "Use local_path as the source file. If the user asks to edit, convert, or analyze it, "
            "work from this file and return the resulting file to the user. Do not ask the user to "
            "resend the file unless reading local_path fails."
        )

    @staticmethod
    def _cleanup_replied_file_context(file_context: dict | None) -> None:
        if not file_context:
            return
        local_path = file_context.get("local_path")
        if not local_path:
            return
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
            temp_dir = os.path.dirname(local_path)
            if temp_dir:
                os.rmdir(temp_dir)
        except OSError:
            logger.debug("Could not clean up replied file context at %s", local_path, exc_info=True)

    def _reply_context_kind(self, update: Update) -> str | None:
        message = update.effective_message
        replied_message = getattr(message, 'reply_to_message', None)
        if not replied_message:
            return None
        if self._image_file_id_from_message(replied_message):
            return "image"
        if getattr(replied_message, 'text', None) or getattr(replied_message, 'caption', None):
            return "text"
        return "other"

    async def _classify_reply_intent(self, update: Update, prompt: str | None) -> str | None:
        if not prompt:
            return None
        replied_message_kind = self._reply_context_kind(update)
        if not replied_message_kind:
            return None
        try:
            intent = await self.openai.classify_reply_intent(prompt, replied_message_kind)
            logger.info("Classified reply intent as %s for %s context", intent, replied_message_kind)
            if intent in {"image_edit", "image_describe", "text_reply"}:
                return intent
        except Exception:
            logger.warning("Failed to classify reply intent with light model", exc_info=True)
        return None

    def _remember_sent_image_messages(self, update: Update, sent_messages) -> None:
        if not sent_messages:
            return
        if not isinstance(sent_messages, (list, tuple)):
            sent_messages = [sent_messages]

        user = update.effective_user
        chat = update.effective_chat
        if not user or not chat:
            return

        for sent_message in sent_messages:
            file_id = self._image_file_id_from_message(sent_message)
            if not file_id:
                continue
            try:
                self.db.save_image(user.id, chat.id, file_id)
                self.openai.set_last_image_file_id(chat.id, file_id)
                if user.id != chat.id:
                    self.openai.set_last_image_file_id(user.id, file_id)
                logger.info("Stored sent image file_id for user %s chat %s", user.id, chat.id)
            except Exception:
                logger.warning("Failed to store sent image file_id for user %s chat %s", user.id, chat.id, exc_info=True)

    def _build_plan_status_provider(self, chat_id, user_id):
        """
        Return (plan_provider, interval) for BusyStatusMessage. The provider yields
        the agent's current plan_tasks for this scope so the live status message
        can render step progress; falls back to (None, default_interval) when the
        agent_tools plugin is unavailable.
        """
        agent_tools_plugin = None
        plugin_manager = getattr(self.openai, "plugin_manager", None)
        get_plugin = getattr(plugin_manager, "get_plugin", None) if plugin_manager else None
        if callable(get_plugin):
            try:
                agent_tools_plugin = get_plugin("agent_tools")
            except Exception:
                logger.debug("Failed to get agent_tools plugin", exc_info=True)
                agent_tools_plugin = None
        get_plan_tasks = getattr(agent_tools_plugin, "get_plan_tasks", None) if agent_tools_plugin else None
        if not callable(get_plan_tasks):
            return None, 30.0

        def _provider():
            try:
                return get_plan_tasks(chat_id=chat_id, user_id=user_id)
            except Exception:
                return []

        return _provider, 5.0

    async def _handle_direct_result(self, update: Update, response):
        sent_messages = []
        delivery_error: Exception | None = None
        try:
            sent_messages = await handle_direct_result(self.config, update, response)
            self._remember_sent_image_messages(update, sent_messages)
        except Exception as exc:
            delivery_error = exc
            logger.exception("Failed to deliver direct_result to chat")
            try:
                effective_message = update.effective_message
                if effective_message is not None:
                    await effective_message.reply_text(
                        localized_text(
                            "direct_result_delivery_error",
                            self.config['bot_language']
                        ).format(error=exc)
                    )
            except Exception:
                logger.exception("Failed to notify user about delivery error")
        finally:
            try:
                await self._run_post_delivery_cleanup(response)
            except Exception:
                logger.exception("Post-delivery cleanup raised unexpectedly")
        if delivery_error is not None:
            return
        if not sent_messages:
            logger.warning("handle_direct_result returned no messages; nothing was delivered to user")
            return
        chat = getattr(update, "effective_chat", None)
        user = getattr(update, "effective_user", None)
        await self.openai.plugin_manager.dispatch_observe(
            "on_session_reset",
            SessionResetPayload(
                chat_id=getattr(chat, "id", None),
                user_id=getattr(user, "id", None),
                reason="final_delivery",
                terminal_only=False,
            ),
            user_id=getattr(user, "id", None),
        )

    @staticmethod
    def _direct_result_observer_text(response) -> str:
        try:
            payload = response if isinstance(response, dict) else json.loads(response)
        except Exception:
            return ""
        direct_result = payload.get("direct_result") if isinstance(payload, dict) else None
        if not isinstance(direct_result, dict):
            return ""
        for key in ("text", "caption", "add_value"):
            value = direct_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        kind = str(direct_result.get("kind") or "").strip()
        value = direct_result.get("value")
        if kind in {"text", "final"} and isinstance(value, str) and value.strip():
            return value.strip()
        artifacts = direct_result.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            return f"{kind or 'direct_result'} ({len(artifacts)} artifacts)"
        return kind

    async def _dispatch_assistant_response_observer(
        self,
        *,
        chat_id: int,
        user_id: int,
        request_id: str,
        text: str | None,
        tokens: int,
        model: str,
    ) -> None:
        if not text:
            return
        await self.openai.plugin_manager.dispatch_observe(
            "on_assistant_response",
            AssistantResponsePayload(
                chat_id=chat_id,
                user_id=user_id,
                request_id=request_id,
                text=text,
                tokens=tokens,
                model=model,
                ts=time.time(),
            ),
            user_id=user_id,
        )

    async def _run_post_delivery_cleanup(self, response):
        try:
            payload = response if isinstance(response, dict) else json.loads(response)
        except Exception:
            return
        direct_result = payload.get("direct_result") if isinstance(payload, dict) else None
        if not isinstance(direct_result, dict):
            return
        directives = []
        single = direct_result.get("cleanup_skill")
        if isinstance(single, dict):
            directives.append(single)
        many = direct_result.get("cleanup_skills")
        if isinstance(many, list):
            directives.extend(item for item in many if isinstance(item, dict))
        if not directives:
            return
        plugin_manager = getattr(self.openai, "plugin_manager", None)
        get_plugin = getattr(plugin_manager, "get_plugin", None) if plugin_manager else None
        if not callable(get_plugin):
            return
        for directive in directives:
            plugin_id = directive.get("plugin_id")
            if not plugin_id:
                continue
            plugin = get_plugin(plugin_id)
            cleanup = getattr(plugin, "cleanup_after_delivery", None) if plugin else None
            if not callable(cleanup):
                continue
            try:
                result = cleanup(directive)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Post-delivery cleanup failed for %s", directive)

    async def _dispatch_session_before_delete(self, user_id: int, session_id: str | None) -> int:
        """Fire ``on_session_before_delete`` so plugin subscribers can react.

        Returns 1 if a payload was dispatched, 0 if skipped (no session_id /
        empty context).
        """
        if not session_id:
            return 0
        pm = getattr(self.openai, "plugin_manager", None)
        if pm is None:
            return 0
        try:
            context, _, _, _, _ = self.db.get_conversation_context(user_id, session_id)
            messages = context.get('messages', []) if isinstance(context, dict) else []
            messages = [dict(message) for message in messages if isinstance(message, dict)]
        except Exception:
            logger.exception(
                "Failed to snapshot session before delete dispatch for user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            raise

        payload = SessionBeforeDeletePayload(
            user_id=user_id,
            session_id=session_id,
            messages=tuple(messages),
        )
        try:
            await pm.dispatch_blocking(
                HookEvent.ON_SESSION_BEFORE_DELETE,
                payload,
                user_id=user_id,
            )
        except Exception:
            logger.exception(
                "on_session_before_delete dispatch failed for user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            return 0
        return 1

    async def _dispatch_and_delete_oldest_sessions_for_limit(self, user_id: int, max_sessions: int) -> None:
        session_ids = self.db.get_oldest_session_ids_for_limit(user_id, max_sessions=max_sessions)
        if not session_ids:
            return
        for session_id in session_ids:
            await self._dispatch_session_before_delete(user_id, session_id)
        if not self.db.delete_sessions_by_ids(user_id, session_ids):
            raise RuntimeError(f"Failed to delete old sessions for user {user_id}: {session_ids}")

    def _image_edit_source_file_id(
        self,
        update: Update,
        user_id: int | None = None,
        chat_id: int | None = None,
        prompt: str | None = None,
    ) -> str | None:
        message = update.effective_message
        current_image = self._image_file_id_from_message(message)
        if current_image:
            return current_image
        replied_image = self._image_file_id_from_message(getattr(message, 'reply_to_message', None))
        if replied_image:
            return replied_image
        return None

    def _image_description_source_file_id(self, update: Update, user_id: int, chat_id: int, prompt: str | None = None) -> str | None:
        message = update.effective_message
        current_image = self._image_file_id_from_message(message)
        if current_image:
            return current_image
        replied_image = self._image_file_id_from_message(getattr(message, 'reply_to_message', None))
        if replied_image:
            return replied_image
        return None

    async def _telegram_image_as_png(self, file_id: str) -> io.BytesIO:
        image_bytes = await self.openai.download_file_as_bytes(file_id)
        temp_file_png = io.BytesIO()
        Image.open(io.BytesIO(image_bytes)).save(temp_file_png, format='PNG')
        temp_file_png.seek(0)
        return temp_file_png

    async def _edit_image_from_context(self, update: Update, prompt: str, file_id: str) -> None:
        image_value, image_format = await self.openai.edit_telegram_image(prompt, file_id)
        await self._handle_direct_result(update, {
            "direct_result": {
                "kind": "photo",
                "format": image_format,
                "value": image_value,
                "add_value": localized_text("image_edit_success", self.config['bot_language']),
            }
        })

    async def _describe_image_from_context(self, update: Update, prompt: str, file_id: str) -> None:
        bot_language = self.config['bot_language']
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        try:
            image_file = await self._telegram_image_as_png(file_id)
            interpretation, total_tokens = await self.openai.interpret_image(chat_id, image_file, prompt=prompt)
            try:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=interpretation,
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            except BadRequest:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=interpretation
                )

            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.effective_user.name)
            vision_token_price = self.config['vision_token_price']
            self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)
            allowed_user_ids = self.config['allowed_user_ids'].split(',')
            if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                self.usage["guests"].add_vision_tokens(total_tokens, vision_token_price)
        except Exception as e:
            logger.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('vision_fail', bot_language)}: {str(e)}"
            )

    def cleanup_inline_cache(self):
        """
        Очищает устаревшие записи из кеша inline запросов
        """
        current_time = time.time()
        
        # Очищаем кеш каждые 10 минут
        if current_time - self._inline_cache_cleanup_time > 600:
            # Оставляем только последние 100 записей
            if len(self.inline_queries_cache) > 100:
                items = list(self.inline_queries_cache.items())
                self.inline_queries_cache = dict(items[-100:])
            
            self._inline_cache_cleanup_time = current_time

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        bot_language = self.config['bot_language']
        user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
        plugin_help_text = self._plugin_help_text(user_id)
        #tool_list = "\n".join([f"- {tool['name']}" for tool in TOOLS])
        help_text = (
            localized_text('help_text', bot_language)[0]
            + '\n\n'
            + '\n'.join(commands_description)
            + '\n\n'
            + localized_text('help_text', bot_language)[1]
            + '\n\n'
            + localized_text('help_text', bot_language)[2]
            + (('\n\n' + plugin_help_text) if plugin_help_text else '')
            + '\n\n'
            + localized_text('help_extra', bot_language)
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    def _plugin_help_text(self, user_id: int | None) -> str:
        plugin_manager = getattr(self.openai, 'plugin_manager', None)
        get_plugin_help_texts = getattr(plugin_manager, 'get_plugin_help_texts', None)
        if not callable(get_plugin_help_texts):
            return ''
        help_sections = []
        for item in get_plugin_help_texts():
            plugin_name = item.get('plugin_name')
            if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
                continue
            text = str(item.get('text') or '').strip()
            if text:
                help_sections.append(text)
        return '\n\n'.join(help_sections)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Возвращает статистику использования токенов за текущий день и месяц,
        а также информацию о сессиях пользователя.
        """
        if not await is_allowed(self.config, update, context):
            logger.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logger.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     'requested their usage statistics')

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)
        bot_language = self.config['bot_language']

        # Получаем информацию о сессиях пользователя
        sessions = self.db.list_user_sessions(user_id)
        active_session = next((s for s in sessions if s['is_active']), None)
        
        # Формируем текст о текущей сессии
        text_current_session = ""
        if active_session:
            text_current_session = (
                localized_text('stats_active_session', bot_language) + "\n"
                + localized_text('stats_session_name', bot_language).format(
                    session_name=active_session['session_name']
                ) + "\n"
                + localized_text('stats_session_messages', bot_language).format(
                    message_count=active_session['message_count']
                ) + "\n"
            )
            
            # Добавляем информацию о модели сессии
            if active_session.get('model',''):
                text_current_session += localized_text('stats_session_model', bot_language).format(
                    model=active_session.get('model', '')
                ) + "\n"
            
            # Добавляем информацию о режиме
            if active_session['context'].get('messages'):
                last_system_message = next(
                    (msg for msg in active_session['context']['messages'] if msg.get('role') == 'system'),
                    None
                )
                if last_system_message:
                    # Используем кешированные режимы
                    chat_modes = self.get_chat_modes()
                    
                    for mode_key, mode_data in chat_modes.items():
                        if mode_data.get('prompt_start', '').strip() == last_system_message.get('content', '').strip():
                            text_current_session += localized_text('stats_session_mode', bot_language).format(
                                mode=mode_data.get('name', mode_key)
                            ) + "\n"
                            break
            
            text_current_session += localized_text('stats_session_temperature', bot_language).format(
                temperature=active_session['temperature']
            ) + "\n"
            text_current_session += localized_text('stats_session_max_tokens', bot_language).format(
                max_tokens_percent=active_session['max_tokens_percent']
            ) + "\n"
            text_current_session += "----------------------------\n"

        # Формируем текст о всех сессиях
        text_all_sessions = (
            localized_text('stats_sessions_total', bot_language).format(
                current=len(sessions),
                maximum=self.config.get('max_sessions', 5)
            ) + "\n"
            "----------------------------\n"
        )

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
         transcribe_seconds_month) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = await self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)

        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ""
        if self.config.get('enable_image_generation', False) and images_today:
            text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"

        text_today_vision = ""
        if self.config.get('enable_vision', False) and vision_today:
            text_today_vision = f"{vision_today} {localized_text('stats_vision', bot_language)}\n"

        text_today_tts = ""
        if self.config.get('enable_tts_generation', False) and characters_today:
            text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"

        text_today = f"*{localized_text('usage_today', bot_language)}:*\n"
        if tokens_today:
            text_today += f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
        text_today += f"{text_today_images}"  # Include the image statistics for today if applicable
        text_today += f"{text_today_vision}"
        text_today += f"{text_today_tts}"
        if transcribe_minutes_today:
            text_today += f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
        if transcribe_seconds_today or transcribe_minutes_today:
            text_today += f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
        text_today += f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
        text_today += "----------------------------\n"

        text_month_images = ""
        if self.config.get('enable_image_generation', False):
            text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"

        text_month_vision = ""
        if self.config.get('enable_vision', False) and vision_month:
            text_month_vision = f"{vision_month} {localized_text('stats_vision', bot_language)}\n"

        text_month_tts = ""
        if self.config.get('enable_tts_generation', False) and characters_month:
            text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = f"*{localized_text('usage_month', bot_language)}:*\n"
        if tokens_month:
            text_month += f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
        text_month += f"{text_month_images}"  # Include the image statistics for the month if applicable
        text_month += f"{text_month_vision}"
        text_month += f"{text_month_tts}"
        if transcribe_minutes_month:
            text_month += f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
        if transcribe_seconds_month or transcribe_minutes_month:
            text_month += f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
        text_month += f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"

        # text_budget filled with conditional content
        text_budget = "\n\n"
        budget_period = self.config['budget_period']
        budget_period_label = 'all-time' if budget_period == 'total' else budget_period
        if remaining_budget < float('inf'):
            text_budget += (
                f"{localized_text('stats_budget', bot_language)}"
                f"{localized_text(budget_period_label, bot_language)}: "
                f"${remaining_budget:.2f}.\n"
            )
        # If use vsegpt, return money rest
        if is_admin(self.config, user_id) and 'vsegpt' in self.config['openai_base']:
             text_budget += (
                 " VSEGPT "
                 f"{await self.get_credits()}"
             )

        stats_fragments = await self.openai.plugin_manager.collect_fragments(
            "stats_block",
            StatsBlockPayload(user_id=user_id, chat_id=update.effective_chat.id, bot_language=bot_language),
            user_id=user_id,
        )
        usage_text = (
            text_current_session + text_all_sessions + text_today + text_month + text_budget
            + "".join(stats_fragments)
        )
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def get_credits(self):
        api_key = self.config['api_key']
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url="https://api.vsegpt.ru/v1/balance",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
            )
        if response.status_code == 200:
            response_big = json.loads(response.text)
            if response_big.get("status") == "ok":
                return float(response_big.get("data").get("credits"))
            else:
                return response_big.get("reason") # reason of error
        else:
            return f"{response.status_code}: {response.text}"

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logger.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            ' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logger.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            ' does not have anything to resend')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language'])
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logger.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await is_allowed(self.config, update, context):
            await self.send_disallowed_message(update, context)
            return

        set_current_language(self._get_user_language(update))
        bot_language = self._get_user_language(update)
        user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=self._settings_text(bot_language, user_id),
            reply_markup=await self._build_settings_menu(bot_language, user_id),
        )

    async def handle_settings_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query:
            return

        set_current_language(self._get_user_language(update))
        bot_language = self._get_user_language(update)
        user = getattr(update, 'effective_user', None)
        user_id = getattr(user, 'id', None)

        if not await is_allowed(self.config, update, context):
            await query.answer()
            await query.edit_message_text(
                text=localized_text('access_denied_command', self.config['bot_language'])
            )
            return

        parts = str(query.data or '').split(':')
        action = parts[1] if len(parts) > 1 else 'root'

        if action == 'close':
            await query.answer()
            await query.message.delete()
            return

        if action == 'root':
            await query.answer()
            await query.edit_message_text(
                text=self._settings_text(bot_language, user_id),
                reply_markup=await self._build_settings_menu(bot_language, user_id),
            )
            return

        if action == 'language':
            await query.answer()
            page = self._settings_page(parts)
            await query.edit_message_text(
                text=localized_text('settings_language_choose', self.config['bot_language']),
                reply_markup=self._build_language_settings_menu(page, bot_language),
            )
            return

        if action == 'lang_page':
            await query.answer()
            page = self._settings_page(parts)
            await query.edit_message_text(
                text=localized_text('settings_language_choose', self.config['bot_language']),
                reply_markup=self._build_language_settings_menu(page, bot_language),
            )
            return

        if action == 'lang_set' and len(parts) >= 3:
            if user_id is None:
                await query.answer()
                return
            new_language = normalize_language(parts[2])
            settings = self.db.get_user_settings(user_id) or {}
            if not isinstance(settings, dict):
                settings = {}
            settings[USER_LANGUAGE_SETTING] = new_language
            self.db.save_user_settings(user_id, settings)
            self._user_language_cache[user_id] = new_language
            set_current_language(new_language)
            await query.answer(localized_text('settings_language_saved', self.config['bot_language']))
            await query.edit_message_text(
                text=self._settings_text(new_language, user_id),
                reply_markup=await self._build_settings_menu(new_language, user_id),
            )
            return

        if action in {'tts_model', 'tts_model_page'}:
            await query.answer()
            page = self._settings_page(parts)
            await self._show_tts_model_settings(query, page, user_id)
            return

        if action == 'tts_model_set' and len(parts) >= 4:
            await self._set_tts_model_setting(query, parts, user_id, bot_language)
            return

        if action in {'tts_voice', 'tts_voice_page'}:
            await query.answer()
            page = self._settings_page(parts)
            await self._show_tts_voice_settings(query, page, user_id)
            return

        if action == 'tts_voice_set' and len(parts) >= 4:
            await self._set_tts_voice_setting(query, parts, user_id, bot_language)
            return

        if action in {'plugins', 'plugin_page'}:
            await query.answer()
            page = self._settings_page(parts)
            await query.edit_message_text(
                text=localized_text('settings_plugins_choose', self.config['bot_language']),
                reply_markup=self._build_plugin_settings_menu(page, user_id),
            )
            return

        if action == 'plugin_toggle' and len(parts) >= 4:
            await self._toggle_plugin_setting(query, parts, user_id)
            return

        if action in {'skills', 'skill_page'}:
            await query.answer()
            page = self._settings_page(parts)
            await query.edit_message_text(
                text=localized_text('settings_skills_choose', self.config['bot_language']),
                reply_markup=self._build_skill_settings_menu(page, user_id),
            )
            return

        if action == 'skill_toggle' and len(parts) >= 4:
            await self._toggle_skill_setting(query, parts, user_id)
            return

        await query.answer()

    @staticmethod
    def _settings_page(parts: list[str]) -> int:
        try:
            return max(0, int(parts[2]))
        except (IndexError, TypeError, ValueError):
            return 0

    @staticmethod
    def _settings_index(parts: list[str]) -> int | None:
        try:
            return max(0, int(parts[3]))
        except (IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _settings_label(value: str, max_chars: int = 34) -> str:
        value = str(value)
        if len(value) <= max_chars:
            return value
        return value[:max_chars - 3] + "..."

    def _settings_for_user(self, user_id: int | None) -> dict:
        if not getattr(self, 'db', None):
            return {}
        return get_user_settings(self.db, user_id)

    def _settings_text(self, bot_language: str, user_id: int | None = None) -> str:
        settings = self._settings_for_user(user_id)
        tts_model = str(settings.get(USER_TTS_MODEL_SETTING) or self.openai.config.get('tts_model', ''))
        tts_voice = str(settings.get(USER_TTS_VOICE_SETTING) or self.openai.config.get('tts_voice', ''))
        disabled_plugins = normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING))
        disabled_skills = normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING))
        return (
            localized_text('settings_title', self.config['bot_language'])
            + "\n\n"
            + localized_text('settings_language_current', self.config['bot_language']).format(
                language=language_name(bot_language)
            )
            + "\n"
            + localized_text('settings_tts_model_current', self.config['bot_language']).format(model=tts_model)
            + "\n"
            + localized_text('settings_tts_voice_current', self.config['bot_language']).format(voice=tts_voice)
            + "\n"
            + localized_text('settings_disabled_plugins_current', self.config['bot_language']).format(
                count=len(disabled_plugins)
            )
            + "\n"
            + localized_text('settings_disabled_skills_current', self.config['bot_language']).format(
                count=len(disabled_skills)
            )
        )

    async def _build_settings_menu(self, bot_language: str, user_id: int | None = None) -> InlineKeyboardMarkup:
        settings = self._settings_for_user(user_id)
        tts_model = str(settings.get(USER_TTS_MODEL_SETTING) or self.openai.config.get('tts_model', ''))
        tts_voice = str(settings.get(USER_TTS_VOICE_SETTING) or self.openai.config.get('tts_voice', ''))
        keyboard = [
            [
                InlineKeyboardButton(
                    localized_text('settings_language_button', self.config['bot_language']).format(
                        language=language_name(bot_language)
                    ),
                    callback_data='settings:language:0',
                )
            ],
            [
                InlineKeyboardButton(
                    localized_text('settings_tts_model_button', self.config['bot_language']).format(
                        model=self._settings_label(tts_model)
                    ),
                    callback_data='settings:tts_model:0',
                )
            ],
            [
                InlineKeyboardButton(
                    localized_text('settings_tts_voice_button', self.config['bot_language']).format(
                        voice=self._settings_label(tts_voice)
                    ),
                    callback_data='settings:tts_voice:0',
                )
            ],
            [
                InlineKeyboardButton(
                    localized_text('settings_plugins_button', self.config['bot_language']),
                    callback_data='settings:plugins:0',
                )
            ],
            [
                InlineKeyboardButton(
                    localized_text('settings_skills_button', self.config['bot_language']),
                    callback_data='settings:skills:0',
                )
            ],
        ]
        extra_button_groups = await self.openai.plugin_manager.collect_objects(
            "settings_menu_buttons",
            SettingsMenuPayload(user_id=user_id, bot_language=bot_language),
            user_id=user_id,
        )
        for plugin_rows in extra_button_groups:
            if isinstance(plugin_rows, list):
                for row in plugin_rows:
                    if isinstance(row, list):
                        keyboard.append(row)
        keyboard.append([
            InlineKeyboardButton(
                localized_text('settings_close', self.config['bot_language']),
                callback_data='settings:close',
            )
        ])
        return InlineKeyboardMarkup(keyboard)

    def _build_language_settings_menu(self, page: int, current_language: str) -> InlineKeyboardMarkup:
        languages = list(supported_languages())
        page_size = LANGUAGE_MENU_PAGE_SIZE
        total_pages = max(1, (len(languages) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        end = start + page_size
        page_languages = languages[start:end]

        keyboard = []
        for index in range(0, len(page_languages), 2):
            row = []
            for language in page_languages[index:index + 2]:
                prefix = "✓ " if language == current_language else ""
                row.append(InlineKeyboardButton(
                    f"{prefix}{language_name(language)}",
                    callback_data=f"settings:lang_set:{language}",
                ))
            keyboard.append(row)

        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("⬅️", callback_data=f"settings:lang_page:{page - 1}"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("➡️", callback_data=f"settings:lang_page:{page + 1}"))
        if nav_row:
            nav_row.append(InlineKeyboardButton(
                localized_text('settings_page', self.config['bot_language']).format(
                    current=page + 1,
                    total=total_pages,
                ),
                callback_data=f"settings:lang_page:{page}",
            ))
            keyboard.append(nav_row)

        keyboard.append([
            InlineKeyboardButton(
                localized_text('settings_back', self.config['bot_language']),
                callback_data='settings:root',
            )
        ])
        return InlineKeyboardMarkup(keyboard)

    def _build_option_settings_menu(
        self,
        items: list[str],
        *,
        page: int,
        current_value: str | None,
        page_action: str,
        set_action: str,
    ) -> InlineKeyboardMarkup:
        page_size = SETTINGS_MENU_PAGE_SIZE
        total_pages = max(1, (len(items) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        page_items = items[start:start + page_size]

        keyboard = []
        for index in range(0, len(page_items), 2):
            row = []
            for offset, item in enumerate(page_items[index:index + 2]):
                global_index = start + index + offset
                prefix = "✓ " if item == current_value else ""
                row.append(InlineKeyboardButton(
                    f"{prefix}{self._settings_label(item)}",
                    callback_data=f"settings:{set_action}:{page}:{global_index}",
                ))
            keyboard.append(row)

        self._append_settings_nav_rows(keyboard, page, total_pages, page_action)
        return InlineKeyboardMarkup(keyboard)

    def _build_toggle_settings_menu(
        self,
        items: list[str],
        *,
        page: int,
        disabled_values: list[str],
        page_action: str,
        toggle_action: str,
    ) -> InlineKeyboardMarkup:
        page_size = SETTINGS_MENU_PAGE_SIZE
        total_pages = max(1, (len(items) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        page_items = items[start:start + page_size]
        disabled = set(disabled_values)

        keyboard = []
        for index in range(0, len(page_items), 2):
            row = []
            for offset, item in enumerate(page_items[index:index + 2]):
                global_index = start + index + offset
                status_key = 'settings_toggle_disabled' if item in disabled else 'settings_toggle_enabled'
                row.append(InlineKeyboardButton(
                    localized_text(status_key, self.config['bot_language']).format(
                        item=self._settings_label(item)
                    ),
                    callback_data=f"settings:{toggle_action}:{page}:{global_index}",
                ))
            keyboard.append(row)

        self._append_settings_nav_rows(keyboard, page, total_pages, page_action)
        return InlineKeyboardMarkup(keyboard)

    def _append_settings_nav_rows(self, keyboard: list, page: int, total_pages: int, page_action: str) -> None:
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("⬅️", callback_data=f"settings:{page_action}:{page - 1}"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("➡️", callback_data=f"settings:{page_action}:{page + 1}"))
        if nav_row:
            nav_row.append(InlineKeyboardButton(
                localized_text('settings_page', self.config['bot_language']).format(
                    current=page + 1,
                    total=total_pages,
                ),
                callback_data=f"settings:{page_action}:{page}",
            ))
            keyboard.append(nav_row)

        keyboard.append([
            InlineKeyboardButton(
                localized_text('settings_back', self.config['bot_language']),
                callback_data='settings:root',
            )
        ])

    async def _show_tts_model_settings(self, query, page: int, user_id: int | None) -> None:
        models = await self.openai.get_available_tts_models()
        if not models:
            await query.edit_message_text(
                text=localized_text('settings_api_list_failed', self.config['bot_language']),
                reply_markup=self._build_back_settings_menu(),
            )
            return
        current_model = self.openai.get_user_tts_model(user_id)
        await query.edit_message_text(
            text=localized_text('settings_tts_model_choose', self.config['bot_language']),
            reply_markup=self._build_option_settings_menu(
                models,
                page=page,
                current_value=current_model,
                page_action='tts_model_page',
                set_action='tts_model_set',
            ),
        )

    async def _set_tts_model_setting(self, query, parts: list[str], user_id: int | None, bot_language: str) -> None:
        if user_id is None:
            await query.answer()
            return
        page = self._settings_page(parts)
        index = self._settings_index(parts)
        models = await self.openai.get_available_tts_models()
        if index is None or index >= len(models):
            await query.answer()
            await self._show_tts_model_settings(query, page, user_id)
            return
        settings = self._settings_for_user(user_id)
        settings[USER_TTS_MODEL_SETTING] = models[index]
        self.db.save_user_settings(user_id, settings)
        await query.answer(localized_text('settings_tts_model_saved', self.config['bot_language']))
        await query.edit_message_text(
            text=self._settings_text(bot_language, user_id),
            reply_markup=await self._build_settings_menu(bot_language, user_id),
        )

    async def _show_tts_voice_settings(self, query, page: int, user_id: int | None) -> None:
        model = self.openai.get_user_tts_model(user_id)
        voices = await self.openai.get_available_tts_voices(model)
        if not voices:
            await query.edit_message_text(
                text=localized_text('settings_api_list_failed', self.config['bot_language']),
                reply_markup=self._build_back_settings_menu(),
            )
            return
        current_voice = self.openai.get_user_tts_voice(user_id)
        await query.edit_message_text(
            text=localized_text('settings_tts_voice_choose', self.config['bot_language']),
            reply_markup=self._build_option_settings_menu(
                voices,
                page=page,
                current_value=current_voice,
                page_action='tts_voice_page',
                set_action='tts_voice_set',
            ),
        )

    async def _set_tts_voice_setting(self, query, parts: list[str], user_id: int | None, bot_language: str) -> None:
        if user_id is None:
            await query.answer()
            return
        page = self._settings_page(parts)
        index = self._settings_index(parts)
        voices = await self.openai.get_available_tts_voices(self.openai.get_user_tts_model(user_id))
        if index is None or index >= len(voices):
            await query.answer()
            await self._show_tts_voice_settings(query, page, user_id)
            return
        settings = self._settings_for_user(user_id)
        settings[USER_TTS_VOICE_SETTING] = voices[index]
        self.db.save_user_settings(user_id, settings)
        await query.answer(localized_text('settings_tts_voice_saved', self.config['bot_language']))
        await query.edit_message_text(
            text=self._settings_text(bot_language, user_id),
            reply_markup=await self._build_settings_menu(bot_language, user_id),
        )

    def _available_plugin_names(self) -> list[str]:
        plugin_manager = getattr(self.openai, 'plugin_manager', None)
        plugins = getattr(plugin_manager, 'plugins', {}) or {}
        return sorted(str(name) for name in plugins.keys())

    def _available_skill_names(self) -> list[str]:
        plugin_manager = getattr(self.openai, 'plugin_manager', None)
        has_plugin = getattr(plugin_manager, 'has_plugin', None)
        get_plugin = getattr(plugin_manager, 'get_plugin', None)
        skills_plugin = (
            get_plugin('skills')
            if callable(has_plugin) and callable(get_plugin) and has_plugin('skills')
            else None
        )
        available_skills = getattr(skills_plugin, 'available_skills', {}) or {}
        return sorted(str(name) for name in available_skills.keys())

    def _build_plugin_settings_menu(self, page: int, user_id: int | None) -> InlineKeyboardMarkup:
        settings = self._settings_for_user(user_id)
        plugins = self._available_plugin_names()
        if not plugins:
            return self._build_back_settings_menu()
        return self._build_toggle_settings_menu(
            plugins,
            page=page,
            disabled_values=normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING)),
            page_action='plugin_page',
            toggle_action='plugin_toggle',
        )

    def _build_skill_settings_menu(self, page: int, user_id: int | None) -> InlineKeyboardMarkup:
        settings = self._settings_for_user(user_id)
        skills = self._available_skill_names()
        if not skills:
            return self._build_back_settings_menu()
        return self._build_toggle_settings_menu(
            skills,
            page=page,
            disabled_values=normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING)),
            page_action='skill_page',
            toggle_action='skill_toggle',
        )

    async def _toggle_plugin_setting(self, query, parts: list[str], user_id: int | None) -> None:
        if user_id is None:
            await query.answer()
            return
        page = self._settings_page(parts)
        index = self._settings_index(parts)
        plugins = self._available_plugin_names()
        if index is None or index >= len(plugins):
            await query.answer()
            return
        settings = self._settings_for_user(user_id)
        disabled = plugins[index] not in set(normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING)))
        set_disabled_value(settings, USER_DISABLED_PLUGINS_SETTING, plugins[index], disabled)
        self.db.save_user_settings(user_id, settings)
        await query.answer(localized_text(
            'settings_plugin_disabled' if disabled else 'settings_plugin_enabled',
            self.config['bot_language'],
        ).format(plugin=plugins[index]))
        await query.edit_message_text(
            text=localized_text('settings_plugins_choose', self.config['bot_language']),
            reply_markup=self._build_plugin_settings_menu(page, user_id),
        )

    async def _toggle_skill_setting(self, query, parts: list[str], user_id: int | None) -> None:
        if user_id is None:
            await query.answer()
            return
        page = self._settings_page(parts)
        index = self._settings_index(parts)
        skills = self._available_skill_names()
        if index is None or index >= len(skills):
            await query.answer()
            return
        settings = self._settings_for_user(user_id)
        disabled = skills[index] not in set(normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING)))
        set_disabled_value(settings, USER_DISABLED_SKILLS_SETTING, skills[index], disabled)
        self.db.save_user_settings(user_id, settings)
        await query.answer(localized_text(
            'settings_skill_disabled' if disabled else 'settings_skill_enabled',
            self.config['bot_language'],
        ).format(skill=skills[index]))
        await query.edit_message_text(
            text=localized_text('settings_skills_choose', self.config['bot_language']),
            reply_markup=self._build_skill_settings_menu(page, user_id),
        )

    def _build_back_settings_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                localized_text('settings_back', self.config['bot_language']),
                callback_data='settings:root',
            )
        ]])

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error: bool = False):
        """
        Сброс контекста разговора и управление сессиями
        """
        # Проверяем, что effective_chat существует (не для inline queries)
        if not update.effective_chat:
            logger.warning("reset called without effective_chat (likely inline query)")
            return
            
        is_callback = bool(update.callback_query)
        
        # Получаем информацию о пользователе из callback_query или message
        if is_callback:
            if not await is_allowed(self.config, update, context):
                await update.callback_query.edit_message_text(
                    text=localized_text('access_denied_command', self.config['bot_language'])
                )
                return
        elif update.message:
            # Для обычных сообщений используем стандартную проверку
            if not await is_allowed(self.config, update, context):
                await self.send_disallowed_message(update, context)
                return
        else:
            logger.error("Neither callback_query nor message found in update")
            return
        
        if error:
            # Сброс из-за ошибки
            message_text = localized_text('reset_error', self.config['bot_language'])
            if is_callback:
                await update.callback_query.edit_message_text(text=message_text)
            else:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=message_text
                )
            return

        try:
            conversation_key = get_conversation_key(update)
            # Получаем список сессий текущего conversation key
            sessions = self.db.list_user_sessions(conversation_key)
            
            # Используем кешированные режимы для определения имен режимов
            chat_modes = self.get_chat_modes()
            
            # Создаем клавиатуру с кнопками управления сессиями
            keyboard = []
            active_session = next((s for s in sessions if s['is_active']), None)
            if active_session:
                preview_button = InlineKeyboardButton(
                    localized_text('session_preview_active', self.config['bot_language']),
                    callback_data=f"session:preview:{active_session['session_id']}"
                )
                keyboard.append([preview_button])
   
            # Добавляем кнопку создания новой сессии
            keyboard.append([InlineKeyboardButton(
                text=localized_text('session_new', self.config['bot_language']),
                callback_data="session:new"
            )])
            
            # Если есть существующие сессии, добавляем их в список
            if sessions:
                for session in sessions:
                    # Добавляем маркер активной сессии и информацию о режиме
                    session_name = session['session_name']
                    
                    # Определяем текущий режим сессии
                    current_mode = None
                    if session['context'].get('messages'):
                        last_system_message = next(
                            (msg for msg in session['context']['messages'] if msg.get('role') == 'system'),
                            None
                        )
                        if last_system_message:
                            # Ищем режим по системному сообщению
                            for mode_key, mode_data in chat_modes.items():
                                if mode_data.get('prompt_start', '').strip() == last_system_message.get('content', '').strip():
                                    current_mode = mode_data.get('name', mode_key)
                                    break
                    
                    # Формируем имя сессии с информацией о режиме и модели
                    if session['is_active']:
                        session_name = f"✓ {session_name}"
                    else:
                        session_name = f"{session_name}"
                    if current_mode:
                        session_name = f"{session_name}\n💫 {current_mode}"
                    
                    # Добавляем кнопку для каждой сессии
                    keyboard.append([
                        InlineKeyboardButton(
                            text=session_name,
                            callback_data=f"session:switch:{session['session_id']}"
                        ),
                        InlineKeyboardButton(
                            text="🗑️",
                            callback_data=f"session:delete:{session['session_id']}"
                        )
                    ])
            
            # Добавляем разделитель
            keyboard.append([InlineKeyboardButton(
                text=localized_text('session_change_mode', self.config['bot_language']),
                callback_data="session:change_mode"
            )])

            # Добавляем кнопку экспорта сессий
            keyboard.append([InlineKeyboardButton(
                text=localized_text('session_export', self.config['bot_language']),
                callback_data="session:export"
            )])

            # Добавляем кнопку закрытия меню
            keyboard.append([InlineKeyboardButton(
                text=localized_text('session_close_menu', self.config['bot_language']),
                callback_data="session:close"
            )])

            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Формируем текст сообщения
            message_text = localized_text('session_management_title', self.config['bot_language']) + "\n\n"
            if sessions:
                active_session = next((s for s in sessions if s['is_active']), None)
                if active_session:
                    message_text += localized_text('session_active_label', self.config['bot_language']).format(
                        session_name=active_session['session_name']
                    ) + "\n"
                    message_text += localized_text('session_messages_label', self.config['bot_language']).format(
                        message_count=active_session['message_count']
                    ) + "\n"
                    
                    # Добавляем информацию о модели сессии
                    current_model = self.openai.get_current_model(conversation_key)
                    message_text += localized_text('session_model_label', self.config['bot_language']).format(
                        model=current_model
                    ) + "\n"
                    
                    # Добавляем информацию о режиме активной сессии
                    if active_session['context'].get('messages'):
                        last_system_message = next(
                            (msg for msg in active_session['context']['messages'] if msg.get('role') == 'system'),
                            None
                        )
                        if last_system_message:
                            for mode_key, mode_data in chat_modes.items():
                                if mode_data.get('prompt_start', '').strip() == last_system_message.get('content', '').strip():
                                    message_text += localized_text('session_mode_label', self.config['bot_language']).format(
                                        mode=mode_data.get('name', mode_key)
                                    ) + "\n"
                                    break
                                    
                message_text += "\n" + localized_text('stats_sessions_total', self.config['bot_language']).format(
                    current=len(sessions),
                    maximum=self.config.get('max_sessions', 5)
                )
            else:
                message_text += localized_text('session_no_active', self.config['bot_language'])

            # Отправляем или редактируем сообщение в зависимости от типа обновления
            try:
                if is_callback:
                    await update.callback_query.edit_message_text(
                        text=message_text,
                        reply_markup=reply_markup
                    )
                else:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=message_text,
                        reply_markup=reply_markup
                    )
            except BadRequest as e:
                if "Message is not modified" not in str(e):
                    raise
            
        except Exception as e:
            logger.error(f"Error in reset: {str(e)}", exc_info=True)
            error_text = localized_text('session_management_error', self.config['bot_language'])
            if is_callback:
                try:
                    await update.callback_query.edit_message_text(text=error_text)
                except BadRequest as e:
                    if "Message is not modified" not in str(e):
                        raise
            else:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=error_text
                )

    async def handle_prompt_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обрабатывает выбор промпта пользователем
        """
        query = update.callback_query
        await query.answer()

        if not await is_allowed(self.config, update, context):
            await query.edit_message_text(
                text=localized_text('access_denied_command', self.config['bot_language'])
            )
            return
        
        conversation_key = get_conversation_key(update)
        
        # Безопасное разделение данных с расширенной обработкой
        data_parts = query.data.split(':')
        action = data_parts[0] if data_parts else ''
        value = data_parts[1] if len(data_parts) > 1 else ''
                
        # Используем кешированные промпты
        chat_modes = self.get_chat_modes()

        try:
            if action == "promptgroup":
                # Показываем режимы выбранной группы
                keyboard = []
                for mode_key, mode_data in chat_modes.items():
                    if mode_data.get('group', localized_text('session_group_other', self.config['bot_language'])) == value:
                        keyboard.append([InlineKeyboardButton(
                            text=mode_data.get('name', mode_key),
                            callback_data=f"prompt:{mode_key}"
                        )])
                
                # Добавляем кнопку "Назад"
                keyboard.append([InlineKeyboardButton(
                    text=localized_text('prompt_back_to_groups', self.config['bot_language']),
                    callback_data="promptback:main"
                )])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    text=localized_text('prompt_choose_from_group', self.config['bot_language']).format(group=value),
                    reply_markup=reply_markup
                )
                
            elif action == "promptback":
                # Возвращаемся к списку групп
                mode_groups = {}
                for mode_key, mode_data in chat_modes.items():
                    group = mode_data.get('group', localized_text('session_group_other', self.config['bot_language']))
                    if group not in mode_groups:
                        mode_groups[group] = []
                    mode_groups[group].append((mode_key, mode_data))

                # Создаем клавиатуру с группами
                keyboard = []
                for group_name in sorted(mode_groups.keys()):
                    keyboard.append([InlineKeyboardButton(
                        text=group_name,
                        callback_data=f"promptgroup:{group_name}"
                    )])
                
                # Добавляем кнопку возврата к сессиям
                keyboard.append([InlineKeyboardButton(
                    text=localized_text('session_back_to_sessions', self.config['bot_language']),
                    callback_data="session:back"
                )])

                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    text=localized_text('prompt_choose_group', self.config['bot_language']),
                    reply_markup=reply_markup
                )
                
            elif action == "prompt":
                mode = value
                if mode in chat_modes:
                    # Получаем текущую активную сессию
                    sessions = self.db.list_user_sessions(conversation_key, is_active=1)
                    active_session = next((s for s in sessions if s['is_active']), None)
                    
                    if not active_session:
                        max_sessions = self.config.get('max_sessions', 5)
                        await self._dispatch_and_delete_oldest_sessions_for_limit(conversation_key, max_sessions)
                        # Если нет активной сессии, создаем новую
                        session_id = self.db.create_session(
                            user_id=conversation_key,
                            max_sessions=max_sessions,
                            openai_helper=self.openai,
                            prune_old_sessions=False,
                        )
                    else:
                        session_id = active_session['session_id']
                    
                    mode_data = chat_modes[mode]
                    # Получаем текущий контекст сессии
                    current_context = self.openai.conversations.get(conversation_key, [])
                    
                    # Добавляем системное сообщение в начало контекста
                    reset_content = mode_data.get('prompt_start', '')
                    system_message = {"role": "system", "content": reset_content, "mode_key": mode}
                    
                    # Если текущий контекст уже содержит системное сообщение, заменяем его
                    if current_context and current_context[0].get('role') == 'system':
                        current_context[0] = system_message
                    else:
                        current_context.insert(0, system_message)
                    
                    # Обновляем контекст в OpenAI и базе данных
                    self.openai.conversations[conversation_key] = current_context
                    self.openai.loaded_conversation_sessions[conversation_key] = session_id
                    
                    # Сохраняем настройки режима в базу данных
                    save_context = getattr(self.openai, "_save_conversation_context", None)
                    if callable(save_context):
                        await save_context(
                            conversation_key,
                            {'messages': current_context},
                            mode_data.get('parse_mode', 'HTML'),
                            mode_data.get('temperature', self.openai.config['temperature']),
                            mode_data.get('max_tokens_percent', 80),
                            session_id,
                        )
                    else:
                        self.db.save_conversation_context(
                            conversation_key,
                            {'messages': current_context},
                            mode_data.get('parse_mode', 'HTML'),
                            mode_data.get('temperature', self.openai.config['temperature']),
                            mode_data.get('max_tokens_percent', 80),
                            session_id
                        )
                    
                    # Возвращаемся в главное меню сессий
                    await self.reset(update, context)
                else:
                    await query.edit_message_text(
                        text=localized_text('prompt_select_error', self.config['bot_language'])
                    )
            else:
                # Обработка неизвестных callback-данных
                logger.warning(f"Неизвестный callback: {query.data}")
                await query.edit_message_text(
                    text=localized_text('generic_error_try_again', self.config['bot_language'])
                )
        except Exception as e:
            logger.error(f"Ошибка в handle_prompt_selection: {e}", exc_info=True)
            await query.edit_message_text(
                text=localized_text('error_with_details', self.config['bot_language']).format(error=str(e))
            )

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Перезапускает бота. Доступно только администраторам.
        """
        if not is_admin(self.config, update.message.from_user.id):
            logger.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                          'tried to restart the bot but is not admin')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('restart_admin_only', self.config['bot_language'])
            )
            return

        logger.info(f'Restarting bot by admin {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})...')
        
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('restart_in_progress', self.config['bot_language'])
        )

        # Очищаем все задачи и закрываем соединения
        #await self.cleanup()
        
        # Пробуем перезапустить systemd сервис
        try:
            import subprocess
            service_name = os.environ.get('SYSTEMD_SERVICE_NAME', 'tg_bot')
            
            # Проверяем статус сервиса
            process = await asyncio.create_subprocess_exec(
                'systemctl', 'is-active', service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout.decode().strip() == 'active':
                # Если сервис активен, перезапускаем его
                restart_process = await asyncio.create_subprocess_exec(
                    'sudo', 'systemctl', 'restart', service_name
                )
                await restart_process.wait()
                await asyncio.sleep(1)  # Даем время на перезапуск сервиса
                sys.exit(0)
                
        except Exception as e:
            logger.warning(f"Failed to restart systemd service: {e}")
        
        # Если не получилось перезапустить сервис, перезапускаем процесс Python
        os.execl(sys.executable, sys.executable, *sys.argv)

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        if not self.config['enable_image_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        image_query = message_text(update.message)
        if image_query == '':
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('image_no_prompt', self.config['bot_language'])
            )
            return

        logger.info(f'New image generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                if self.config['image_receive_mode'] == 'photo':
                    sent_message = await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        photo=image_url
                    )
                elif self.config['image_receive_mode'] == 'document':
                    sent_message = await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        document=image_url
                    )
                else:
                    raise Exception(
                        f"env variable IMAGE_RECEIVE_MODE has invalid value {self.config['image_receive_mode']}")
                self._remember_sent_image_messages(update, sent_message)
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logger.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config['enable_tts_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        tts_query = message_text(update.message)
        if tts_query == '':
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('tts_no_prompt', self.config['bot_language'])
            )
            return

        logger.info(f'New speech generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                user_id = update.message.from_user.id
                tts_model = self.openai.get_user_tts_model(user_id)
                speech_file, text_length = await self.openai.generate_speech(text=tts_query, user_id=user_id)

                audio_format = self.openai.config.get('tts_response_format', 'wav')
                reply_args = {
                    'reply_to_message_id': get_reply_to_message_id(self.config, update),
                }
                if audio_format == 'opus':
                    await update.effective_message.reply_voice(
                        **reply_args,
                        voice=speech_file
                    )
                else:
                    await update.effective_message.reply_audio(
                        **reply_args,
                        audio=speech_file,
                        filename=f"speech.{audio_format}"
                    )
                speech_file.close()
                # add image request to users usage tracker
                self.usage[user_id].add_tts_request(text_length, tts_model, self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_tts_request(text_length, tts_model,
                                                         self.config['tts_prices'])

            except Exception as e:
                logger.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_VOICE)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if is_group_chat(update) and self.config['ignore_group_transcriptions']:
            logger.info('Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        filename = update.message.effective_attachment.file_unique_id
        file_unique_id = filename
        filename_mp3 = None
        max_retries = 3
        retry_delay = 2

        async def _execute():
            nonlocal filename, filename_mp3
            filename = f'{file_unique_id}'
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['bot_language']
            
            # Создаем временную директорию для файлов
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            try:
                os.makedirs(temp_dir, exist_ok=True)
            except PermissionError:
                # Используем системную временную директорию
                import tempfile
                temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            file_path_mp3 = os.path.join(temp_dir, filename_mp3)
            
            for attempt in range(max_retries):
                try:
                    try:
                        media_file = await self.application.bot.get_file(update.message.effective_attachment.file_id)
                        await media_file.download_to_drive(file_path)
                        break
                    except TimedOut:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed with timeout, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            continue
                        raise
                    except Exception as e:
                        logger.exception(e)
                        raise
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.exception(e)
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update),
                            text=(
                                f"{localized_text('media_download_fail', bot_language)[0]}: "
                                f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                            ),
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                        return

            try:
                audio_track = AudioSegment.from_file(file_path)
                audio_track.export(file_path_mp3, format="mp3")
                logger.info(f'New transcribe request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logger.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text('media_type_fail', bot_language)
                )
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(file_path_mp3):
                    os.remove(file_path_mp3)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)
            request_context = RequestContext(
                chat_id=chat_id,
                user_id=user_id,
                message_id=update.message.message_id,
                request_id=f"{chat_id}_{update.message.message_id}",
            )

            try:
                transcript = await self.openai.transcribe(file_path_mp3)

                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(transcript.lower().startswith(prefix.lower()) if prefix else False
                                                for prefix in self.config['voice_reply_prompts'])

                if self.config['voice_reply_transcript'] and not response_to_transcription:

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=None
                        )
                else:
                    # Get the response of the transcript
                    if len(transcript) > 60000:
                        kwargs = {
                            'big_context': True
                        }
                    else:
                        kwargs = {}
                    await self.openai.plugin_manager.dispatch_observe(
                        "on_session_reset",
                        SessionResetPayload(
                            chat_id=chat_id,
                            user_id=user_id,
                            reason="request_start",
                            terminal_only=False,
                        ),
                        user_id=user_id,
                    )
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=chat_id,
                        query=transcript,
                        user_id=user_id,
                        request_context=request_context,
                        **kwargs,
                    )

                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])

                    if is_direct_result(response):
                        return await self._handle_direct_result(update, response)
                    
                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    logger.info(f"Transcript output: {transcript_output}")
                    chunks = split_into_chunks(transcript_output)
                    # Если ответ больше 3х частей, то формируем файл с ответом и отправлем его
                    if should_send_text_as_file(transcript_output, chunks):
                        # Получаем имя текущей сессии
                        sessions = self.db.list_user_sessions(user_id, is_active=1)
                        active_session = next((s for s in sessions if s['is_active']), None)
                        session_name = active_session['session_name'] if active_session else 'transcription'
                        
                        await send_long_response_as_file(self.config, update, transcript_output, session_name)
                    else:
                        for index, transcript_chunk in enumerate(chunks):
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                                text=transcript_chunk,
                                parse_mode=None
                            )

            except Exception as e:
                logger.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                # Очищаем временные файлы
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(file_path_mp3):
                    os.remove(file_path_mp3)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Interpret image using vision model.
        """
        if not (self.config['enable_vision'] or self.config['enable_image_generation']) \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = update.message.caption
        
        logger.info(f"Vision handler called for chat_id: {chat_id}, user_id: {user_id}")

        # Cleanup old images first
        self.db.cleanup_old_images()

        image = None
        # Store the image in database
        if len(update.message.photo) > 0:
            image = update.message.photo[-1]
            file_id = image.file_id
            logger.info(f"Storing photo file_id: {file_id}")
            self.db.save_image(user_id, chat_id, file_id)
        elif update.message.document and update.message.document.mime_type.startswith('image/'):
            image = update.message.document
            file_id = image.file_id
            logger.info(f"Storing document file_id: {file_id}")
            self.db.save_image(user_id, chat_id, file_id)

        if not self.config['enable_vision']:
            return

        # Only proceed with vision if there's a caption or it's a valid vision request
        if prompt or (is_group_chat(update) and not self.config['ignore_group_vision']):
            if is_group_chat(update):
                if self.config['ignore_group_vision']:
                    logger.info('Vision coming from group chat, ignoring...')
                    return
                else:
                    trigger_keyword = self.config['group_trigger_keyword']
                    if (prompt is None and trigger_keyword != '') or \
                            (prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())):
                        logger.info('Vision coming from group chat with wrong keyword, ignoring...')
                        return

            if image is None:
                logger.warning("Vision handler received no supported image attachment")
                return

            async def _execute():
                bot_language = self.config['bot_language']
                try:
                    media_file = await self.application.bot.get_file(image.file_id)
                    temp_file = io.BytesIO(await media_file.download_as_bytearray())
                except Exception as e:
                    logger.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text=(
                            f"{localized_text('media_download_fail', bot_language)[0]}: "
                            f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                        ),
                        parse_mode=constants.ParseMode.MARKDOWN
                    )
                    return

                # convert jpg from telegram to png as understood by openai

                temp_file_png = io.BytesIO()

                try:
                    original_image = Image.open(temp_file)

                    original_image.save(temp_file_png, format='PNG')
                    logger.info(f'New vision request received from user {update.message.from_user.name} '
                                f'(id: {update.message.from_user.id})')

                except Exception as e:
                    logger.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text=localized_text('media_type_fail', bot_language)
                    )

                user_id = update.message.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

                if self.config['stream']:

                    stream_response = self.openai.interpret_image_stream(chat_id=chat_id, fileobj=temp_file_png,
                                                                        prompt=prompt)
                    i = 0
                    prev = ''
                    sent_message = None
                    backoff = 0
                    # Индекс последнего «закрытого» чанка, опубликованного как
                    # отдельное сообщение. Растущий tail-чанк публикуется отдельно.
                    last_published_chunk = 0

                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            return await self._handle_direct_result(update, content)

                        if len(content.strip()) == 0:
                            continue

                        stream_chunks = split_into_chunks(content)
                        if len(stream_chunks) > 1:
                            content = stream_chunks[-1]
                            if last_published_chunk != len(stream_chunks) - 1:
                                last_published_chunk += 1
                                previous_chunk = stream_chunks[-2]
                                if sent_message is not None:
                                    try:
                                        await edit_message_with_retry(
                                            context, chat_id, str(sent_message.message_id),
                                            previous_chunk,
                                        )
                                    except Exception:
                                        logger.debug(
                                            "vision stream: edit previous chunk failed",
                                            exc_info=True,
                                        )
                                else:
                                    # Первое сообщение ещё не отправлено: публикуем
                                    # завершённый предыдущий чанк как новое сообщение.
                                    try:
                                        sent_message = await update.effective_message.reply_text(
                                            message_thread_id=get_thread_id(update),
                                            text=previous_chunk or "...",
                                        )
                                    except Exception:
                                        logger.debug(
                                            "vision stream: initial reply for previous chunk failed",
                                            exc_info=True,
                                        )
                                try:
                                    sent_message = await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content if len(content) > 0 else "..."
                                    )
                                except Exception:
                                    logger.debug(
                                        "vision stream: reply for new chunk failed",
                                        exc_info=True,
                                    )
                                continue

                        cutoff = get_stream_cutoff_values(update, content)
                        cutoff += backoff

                        if i == 0:
                            try:
                                if sent_message is not None:
                                    await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                    message_id=sent_message.message_id)
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=content,
                                )
                            except Exception:
                                logger.debug("vision stream: initial reply failed", exc_info=True)
                                continue

                        elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                            prev = content

                            try:
                                use_markdown = tokens != 'not_finished'
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              text=content, markdown=use_markdown)

                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue

                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue

                            except Exception:
                                backoff += 5
                                continue

                            await asyncio.sleep(0.01)

                        i += 1
                        if tokens != 'not_finished':
                            total_tokens = int(tokens)

                else:

                    try:
                        interpretation, total_tokens = await self.openai.interpret_image(chat_id, temp_file_png,
                                                                                        prompt=prompt)

                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=interpretation,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except BadRequest:
                            try:
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=interpretation
                                )
                            except Exception as e:
                                logger.exception(e)
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=f"{localized_text('vision_fail', bot_language)}: {str(e)}"
                                )
                    except Exception as e:
                        logger.exception(e)
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update),
                            text=f"{localized_text('vision_fail', bot_language)}: {str(e)}"
                        )
                vision_token_price = self.config['vision_token_price']
                self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_vision_tokens(total_tokens, vision_token_price)

            await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)
        else:
            # If no caption, just acknowledge receipt of image
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                text=localized_text('image_received', self.config['bot_language'])
                            )

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        message_id = update.message.message_id

        logger.info(f"Prompt handler called for chat_id: {chat_id}, user_id: {user_id}")

        # Get last active image from database if exists
        user_images = self.db.get_user_images(user_id, chat_id, limit=1)
        if user_images:
            last_image = user_images[0]
            if last_image['status'] == 'active':
                self.openai.set_last_image_file_id(chat_id, last_image['file_id'])
                if user_id != chat_id:
                    self.openai.set_last_image_file_id(user_id, last_image['file_id'])
                logger.info(f"Found active image {last_image['file_id']} for user {user_id}")

        async with self.buffer_lock:
            # Инициализируем буфер для чата, если его нет
            if chat_id not in self.message_buffer:
                self.message_buffer[chat_id] = {
                    'messages': [],  # Очередь сообщений
                    'processing': False,  # Флаг обработки
                    'timer': None  # Таймер для обработки буфера
                }

            buffer_data = self.message_buffer[chat_id]
            
            # Добавляем сообщение в буфер
            buffer_data['messages'].append({
                'text': prompt,
                'update': update,
                'context': context,
                'message_id': message_id,
                'message_timestamp': update.message.date.timestamp()
            })
            
            # Запускаем таймер обработки буфера, если он не запущен
            if buffer_data['timer'] is None or buffer_data['timer'].done():
                buffer_data['timer'] = asyncio.create_task(
                    self._delayed_process_buffer(chat_id)
                )

    async def _delayed_process_buffer(self, chat_id: int):
        """
        Задержка перед обработкой буфера для сбора всех сообщений
        """
        await asyncio.sleep(self.buffer_timeout)
        await self.process_buffer(chat_id)

    async def process_buffer(self, chat_id: int):
        """
        Process all messages in the buffer sequentially, combining messages from the same user
        within the buffer timeout period
        """
        started_processing = False
        try:
            async with self.buffer_lock:
                if chat_id not in self.message_buffer:
                    return
                
                buffer_data = self.message_buffer[chat_id]
                if buffer_data['processing']:
                    return
                    
                buffer_data['processing'] = True
                started_processing = True
                messages = buffer_data['messages']
                buffer_data['messages'] = []

            if not messages:
                return

            # Group messages by user_id and sort by timestamp
            user_messages = {}
            for msg in messages:
                user_id = msg['update'].message.from_user.id
                if user_id not in user_messages:
                    user_messages[user_id] = []
                user_messages[user_id].append(msg)

            for user_msg_list in user_messages.values():
                if not user_msg_list:
                    continue

                # Sort messages by timestamp
                user_msg_list.sort(key=lambda x: x['message_timestamp'])
                combined_messages = []
                current_group = [user_msg_list[0]]
                
                for msg in user_msg_list[1:]:
                    time_diff = msg['message_timestamp'] - current_group[-1]['message_timestamp']
                    if time_diff <= self.buffer_timeout:
                        current_group.append(msg)
                    else:
                        combined_messages.append(current_group)
                        current_group = [msg]
                
                if current_group:
                    combined_messages.append(current_group)

                # Process each group
                for msg_group in combined_messages:
                    combined_text = " ".join(msg['text'] for msg in msg_group)
                    first_msg = msg_group[0]
                    
                    try:
                        await self.process_message(combined_text, first_msg['update'], first_msg['context'])
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue

                    await asyncio.sleep(0.1)  # Prevent flooding

        except Exception as e:
            logger.error(f"Error in process_buffer: {e}")
        finally:
            # Reset processing flag
            if started_processing:
                async with self.buffer_lock:
                    if chat_id in self.message_buffer:
                        self.message_buffer[chat_id]['processing'] = False
                    
    async def process_message(self, prompt: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
        conversation_key = get_conversation_key(update)
        conversation_lock = await self._get_conversation_lock(conversation_key)
        async with conversation_lock:
            return await self._process_message_locked(prompt, update, context)

    async def _get_conversation_lock(self, conversation_key: int) -> asyncio.Lock:
        if not hasattr(self, '_conversation_locks'):
            self._conversation_locks = {}
        if not hasattr(self, '_conversation_locks_guard'):
            self._conversation_locks_guard = asyncio.Lock()

        async with self._conversation_locks_guard:
            lock = self._conversation_locks.get(conversation_key)
            if lock is None:
                lock = asyncio.Lock()
                self._conversation_locks[conversation_key] = lock
            return lock

    async def _process_message_locked(self, prompt: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обрабатывает полное сообщение
        """
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        message_id = update.message.message_id
        self.last_message[chat_id] = prompt
        request_id = f"{chat_id}_{message_id}"
        request_context = RequestContext(
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            request_id=request_id,
        )
            
        logger.info(
            f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')

        if is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']

            if prompt.lower().startswith(trigger_keyword.lower()) or update.message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword):].strip()
                if update.message.reply_to_message and \
                        update.message.reply_to_message.text and \
                        update.message.reply_to_message.from_user.id != context.bot.id:
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logger.info('Message is a reply to the bot, allowing...')
                else:
                    logger.warning('Message does not start with trigger keyword, ignoring...')
                    return

        reply_intent = await self._classify_reply_intent(update, prompt)

        if self.config['enable_image_generation'] and reply_intent == "image_edit":
            source_file_id = self._image_edit_source_file_id(update)
            if source_file_id:
                async def _edit():
                    await self._edit_image_from_context(update, prompt, source_file_id)

                await wrap_with_indicator(update, context, _edit, constants.ChatAction.UPLOAD_PHOTO)
                return
            logger.info("Image edit route matched but no source image file_id was found")

        if self.config['enable_vision'] and reply_intent == "image_describe":
            source_file_id = self._image_description_source_file_id(update, user_id, chat_id, prompt)
            if source_file_id:
                async def _describe():
                    await self._describe_image_from_context(update, prompt, source_file_id)

                await wrap_with_indicator(update, context, _describe, constants.ChatAction.TYPING)
                return

        if await self._try_handle_plugin_prompt(prompt, update, context, user_id):
            return

        replied_file_context = None
        assistant_response_text = None
        await self.openai.plugin_manager.dispatch_observe(
            "on_user_message",
            UserMessagePayload(
                chat_id=chat_id,
                user_id=user_id,
                request_id=request_id,
                text=prompt,
                has_image=False,
                has_voice=False,
                is_command=bool(update.message.text and update.message.text.startswith("/")),
                ts=time.time(),
            ),
            user_id=user_id,
        )
        try:
            total_tokens = 0
            replied_file_context = await self._download_replied_file_for_model(update, context)
            if replied_file_context:
                prompt = self._prompt_with_replied_file_context(prompt, replied_file_context)

            model_to_use = self.openai.get_current_model(user_id)
            await self.openai.plugin_manager.dispatch_observe(
                "on_session_reset",
                SessionResetPayload(
                    chat_id=chat_id,
                    user_id=user_id,
                    reason="request_start",
                    terminal_only=False,
                ),
                user_id=user_id,
            )
                
            if self.config['stream'] and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY):

                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )

                stream_response = self.openai.get_chat_response_stream(
                    chat_id=chat_id,
                    query=prompt,
                    request_id=request_id,
                    user_id=user_id,
                    request_context=request_context,
                )
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                # Индекс последнего «закрытого» чанка, опубликованного как
                # отдельное сообщение. Растущий tail-чанк публикуется отдельно.
                last_published_chunk = 0
                last_stream_content = ''

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        assistant_response_text = self._direct_result_observer_text(content)
                        await self._handle_direct_result(update, content)
                        await self._dispatch_assistant_response_observer(
                            chat_id=chat_id,
                            user_id=user_id,
                            request_id=request_id,
                            text=assistant_response_text,
                            tokens=total_tokens,
                            model=model_to_use,
                        )
                        return

                    if len(content.strip()) == 0:
                        continue
                    last_stream_content = content

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if last_published_chunk != len(stream_chunks) - 1:
                            last_published_chunk += 1
                            previous_chunk = stream_chunks[-2]
                            if sent_message is not None:
                                try:
                                    await edit_message_with_retry(
                                        context, chat_id, str(sent_message.message_id),
                                        previous_chunk,
                                    )
                                except Exception:
                                    logger.debug(
                                        "chat stream: edit previous chunk failed",
                                        exc_info=True,
                                    )
                            else:
                                # Первое сообщение ещё не отправлено: публикуем
                                # завершённый предыдущий чанк как новое сообщение.
                                try:
                                    sent_message = await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=previous_chunk or "...",
                                    )
                                except Exception:
                                    logger.debug(
                                        "chat stream: initial reply for previous chunk failed",
                                        exc_info=True,
                                    )
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except Exception:
                                logger.debug(
                                    "chat stream: reply for new chunk failed",
                                    exc_info=True,
                                )
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                message_id=sent_message.message_id)
                            _, parse_mode, _, _, _ = self.db.get_conversation_context(chat_id) or (
                                None, None, None, None, None
                            )
                            parse_mode = parse_mode or constants.ParseMode.HTML
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                                parse_mode=parse_mode
                            )
                        except Exception:
                            logger.error("Failed to send initial streaming message", exc_info=True)
                            try:
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=localized_text('chat_fail', self.config['bot_language'])
                                )
                            except Exception:
                                logger.error("Failed to send streaming error message", exc_info=True)
                            break

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          text=content, markdown=use_markdown)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)
                assistant_response_text = last_stream_content

            else:
                async def _reply():
                    nonlocal total_tokens, assistant_response_text
                    plan_provider, plan_interval = self._build_plan_status_provider(chat_id, user_id)
                    busy_status = BusyStatusMessage(
                        update,
                        context,
                        localized_text("busy_status_preparing", self.config['bot_language']),
                        config=self.config,
                        plan_provider=plan_provider,
                        interval=plan_interval,
                    )
                    await busy_status.start()
                    try:
                        response, total_tokens = await self.openai.get_chat_response(
                            chat_id=chat_id,
                            query=prompt,
                            request_id=request_id,
                            user_id=user_id,
                            request_context=request_context,
                        )
                        await busy_status.stop()

                        if is_direct_result(response):
                            assistant_response_text = self._direct_result_observer_text(response)
                            return await self._handle_direct_result(update, response)

                        assistant_response_text = response
                        # Split into chunks of 4096 characters (Telegram's message limit)
                        chunks = split_into_chunks(response)

                        # Если ответ больше 3х частей, то формируем файл с ответом и отправлем его
                        if should_send_text_as_file(response, chunks):
                            # Получаем имя текущей сессии
                            sessions = self.db.list_user_sessions(user_id, is_active=1)
                            active_session = next((s for s in sessions if s['is_active']), None)
                            session_name = active_session['session_name'] if active_session else 'transcription'

                            await send_long_response_as_file(self.config, update, response, session_name)
                        else:
                            for index, chunk in enumerate(chunks):
                                try:
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        reply_to_message_id=get_reply_to_message_id(self.config,
                                                                                    update) if index == 0 else None,
                                        text=chunk,
                                        parse_mode=constants.ParseMode.MARKDOWN
                                    )
                                except Exception:
                                    try:
                                        await update.effective_message.reply_text(
                                            message_thread_id=get_thread_id(update),
                                            reply_to_message_id=get_reply_to_message_id(self.config,
                                                                                        update) if index == 0 else None,
                                            text=chunk
                                        )
                                    except Exception as exception:
                                        raise exception
                    finally:
                        await busy_status.stop()

                await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)
            await self._dispatch_assistant_response_observer(
                chat_id=chat_id,
                user_id=user_id,
                request_id=request_id,
                text=assistant_response_text,
                tokens=total_tokens,
                model=model_to_use,
            )

            result = add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)
            #if not result:
            #    await self.reset(update, context, True)

        except Exception as e:
            logger.exception(e)
            from .utils import escape_markdown
            error_message = escape_markdown(str(e))
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {error_message}",
                parse_mode=constants.ParseMode.MARKDOWN
            )
        finally:
            self._cleanup_replied_file_context(replied_file_context)

    async def _try_handle_plugin_prompt(
        self,
        prompt: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
    ) -> bool:
        plugin_manager = getattr(self.openai, 'plugin_manager', None)
        get_prompt_handlers = getattr(plugin_manager, 'get_prompt_handlers', None)
        if not plugin_manager or not callable(get_prompt_handlers):
            return False

        for handler_config in get_prompt_handlers():
            plugin_name = handler_config.get("plugin_name")
            if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
                continue
            handler = handler_config.get("handler")
            if not callable(handler):
                continue

            async def _run_handler(h=handler):
                result = h(
                    prompt=prompt,
                    update=update,
                    context=context,
                    helper=self.openai,
                    bot=self,
                )
                if asyncio.iscoroutine(result):
                    result = await result
                return await self._handle_plugin_prompt_result(result, update)

            chat_action = handler_config.get("chat_action")
            if chat_action:
                handled = await wrap_with_indicator(update, context, _run_handler, chat_action)
            else:
                handled = await _run_handler()
            if handled:
                return True
        return False

    async def _handle_plugin_prompt_result(self, result, update: Update) -> bool:
        if result is False or result is None:
            return False
        if result is True:
            return True
        if is_direct_result(result):
            await self._handle_direct_result(update, result)
            return True
        if isinstance(result, dict) and "error" in result:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=localized_text('error_with_details', self.config['bot_language']).format(
                    error=result['error']
                ),
            )
            return True
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            reply_to_message_id=get_reply_to_message_id(self.config, update),
            text=str(result),
        )
        return True

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        # Очищаем кеш от устаревших записей
        self.cleanup_inline_cache()
        
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
            return

        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f'{callback_data_suffix}{result_id}'

        await self.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=""):
        """
        Send inline query result
        """
        try:
            reply_markup = None
            bot_language = self.config['bot_language']
            if callback_data:
                reply_markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                         callback_data=callback_data)
                ]])

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumbnail_url='https://github.com/LKosoj/chatgpt-telegram-bot/blob/main/IMG_3980.jpg?raw=true',
                reply_markup=reply_markup
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logger.error(f'An error occurred while generating the result card for inline query {e}')

    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        """
        Handle the callback query from the inline query result
        """
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        query = ""
        bot_language = self.config['bot_language']
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        try:
            if callback_data.startswith(callback_data_suffix):
                unique_id = callback_data.split(':')[1]
                total_tokens = 0

                # Retrieve the prompt from the cache
                query = self.inline_queries_cache.get(unique_id)
                if query:
                    self.inline_queries_cache.pop(unique_id)
                else:
                    error_message = (
                        f'{localized_text("error", bot_language)}. '
                        f'{localized_text("try_again", bot_language)}'
                    )
                    await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                  text=f'{query}\n\n_{answer_tr}:_\n{error_message}',
                                                  is_inline=True)
                    return

                model_to_use = self.openai.get_current_model(user_id)
                request_context = RequestContext(chat_id=user_id, user_id=user_id)
                await self.openai.plugin_manager.dispatch_observe(
                    "on_session_reset",
                    SessionResetPayload(
                        chat_id=user_id,
                        user_id=user_id,
                        reason="request_start",
                        terminal_only=False,
                    ),
                    user_id=user_id,
                )
                    
                unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
                if self.config['stream'] and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY):
                    stream_response = self.openai.get_chat_response_stream(
                        chat_id=user_id,
                        query=query,
                        user_id=user_id,
                        request_context=request_context,
                    )
                    i = 0
                    prev = ''
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            fallback_text = direct_result_inline_fallback_text(content, unavailable_message)
                            cleanup_intermediate_files(content)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{fallback_text}',
                                                          is_inline=True)
                            return

                        if len(content.strip()) == 0:
                            continue

                        cutoff = get_stream_cutoff_values(update, content)
                        cutoff += backoff

                        if i == 0:
                            try:
                                await edit_message_with_retry(context, chat_id=None,
                                                                  message_id=inline_message_id,
                                                                  text=f'{query}\n\n{answer_tr}:\n{content}',
                                                                  is_inline=True)
                            except Exception:
                                logger.debug("inline stream: initial edit failed", exc_info=True)
                                continue

                        elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                            prev = content
                            try:
                                use_markdown = tokens != 'not_finished'
                                divider = '_' if use_markdown else ''
                                text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                                await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                              text=text, markdown=use_markdown, is_inline=True)

                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue
                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue
                            except Exception:
                                backoff += 5
                                continue

                            await asyncio.sleep(0.01)

                        i += 1
                        if tokens != 'not_finished':
                            total_tokens = int(tokens)

                else:
                    async def _send_inline_query_response():
                        nonlocal total_tokens
                        # Edit the current message to indicate that the answer is being processed
                        await context.bot.edit_message_text(inline_message_id=inline_message_id,
                                                            text=f'{query}\n\n_{answer_tr}:_\n{loading_tr}',
                                                            parse_mode=constants.ParseMode.MARKDOWN)

                        logger.info(f'Generating response for inline query by {name}')
                        response, total_tokens = await self.openai.get_chat_response(
                            chat_id=user_id,
                            query=query,
                            user_id=user_id,
                            request_context=request_context,
                        )

                        if is_direct_result(response):
                            fallback_text = direct_result_inline_fallback_text(response, unavailable_message)
                            cleanup_intermediate_files(response)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{fallback_text}',
                                                          is_inline=True)
                            return

                        text_content = f'{query}\n\n_{answer_tr}:_\n{response}'

                        # Edit the original message with the generated content
                        await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                      text=text_content, is_inline=True)

                    await wrap_with_indicator(update, context, _send_inline_query_response,
                                              constants.ChatAction.TYPING, is_inline=True)

                result = add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)
                if not result:
                    await self.reset(update, context, True)

        except Exception as e:
            logger.error(f'Failed to respond to an inline query via button callback: {e}')
            logger.exception(e)
            localized_answer = localized_text('chat_fail', self.config['bot_language'])
            await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                          text=f"{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}",
                                          is_inline=True)

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                              is_inline=False) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        if is_inline and update.inline_query:
            user = update.inline_query.from_user
        elif update.callback_query:
            user = update.callback_query.from_user
        elif update.message:
            user = update.message.from_user
        else:
            user = update.effective_user
        name = user.name if user else "unknown"
        user_id = user.id if user else None

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logger.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logger.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        disallowed_message = localized_text('disallowed', self.config['bot_language'])
        if not is_inline:
            #chat_id = update.effective_chat.id
            #chat_context, parse_mode, temperature = self.db.get_conversation_context(chat_id) or {}
            message = update.effective_message or (update.callback_query.message if update.callback_query else None)
            if message:
                await message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=disallowed_message,
                    disable_web_page_preview=True
                )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        budget_limit_message = localized_text('budget_limit', self.config['bot_language'])
        if not is_inline:
            #chat_id = update.effective_chat.id
            #chat_context, parse_mode, temperature = self.db.get_conversation_context(chat_id) or {}
            message = update.effective_message or (update.callback_query.message if update.callback_query else None)
            if message:
                await message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=budget_limit_message,
                )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=budget_limit_message)

    def _register_plugin_message_handlers(
        self,
        application: Application,
        source: str,
    ):
        if getattr(self, "_plugin_message_handlers_registered", False):
            logger.info(
                "Plugin message handlers already registered; skipping %s",
                source,
            )
            return

        message_handlers = self.openai.plugin_manager.get_message_handlers()
        for handler_config in message_handlers:
            if 'handler' in handler_config and 'filters' not in handler_config:
                handler = handler_config['handler']
                try:
                    application.add_handler(handler)
                    logger.info(
                        "Successfully added handler in %s: %s",
                        source,
                        type(handler).__name__,
                    )
                except TypeError as e:
                    logger.error(
                        "Invalid handler type %s in %s: %s",
                        type(handler).__name__,
                        source,
                        e,
                    )
                    continue
            elif 'filters' in handler_config:
                filter_obj = handler_config['filters']
                if isinstance(filter_obj, str):
                    key = filter_obj.replace("filters.", "").strip()
                    filter_obj = getattr(filters, key, None)
                if filter_obj is None:
                    logger.error(
                        "Invalid filter in plugin handler config: %s",
                        handler_config.get('filters'),
                    )
                    continue

                def plugin_message_handler(update, context, h=handler_config):
                    handler_kwargs = h.get('handler_kwargs') or {}
                    return self.handle_plugin_command(
                        update,
                        context,
                        {
                            "handler": h['handler'],
                            "plugin_name": h.get('plugin_name'),
                            "handler_kwargs": handler_kwargs,
                            **handler_kwargs,
                        },
                    )

                handler = MessageHandler(
                    filter_obj,
                    plugin_message_handler,
                )
                application.add_handler(handler)

        self._plugin_message_handlers_registered = True

    def _register_message_tail_handlers(self, application: Application):
        if getattr(self, "_message_tail_handlers_registered", False):
            return

        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))

        application.add_handler(
            MessageHandler(
                filters.REPLY & filters.TEXT,
                self.handle_plugin_menu_args_reply
            )
        )

        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
            self.prompt
        ))

        application.add_error_handler(error_handler)
        self._message_tail_handlers_registered = True

    async def post_init(self, application: Application):
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.commands)
        await application.bot.set_my_commands(
            self.group_commands,
            scope=BotCommandScopeAllGroupChats()
        )
        if not self._background_tasks:
            self._background_tasks = [
                asyncio.create_task(self.buffer_data_checker(), name="buffer_data_checker"),
            ]
            await self.openai.plugin_manager.start_background_tasks(application)

        # Регистрируем команды от плагинов
        build = self.openai.plugin_manager.build_bot_commands()
        plugin_commands = build["plugin_commands"]
        # Команды плагинов теперь доступны через /plugins меню
        self.plugin_menu_entries = [
            cmd for cmd in plugin_commands
            if cmd.get("add_to_menu") and cmd.get("command") and cmd.get("description")
        ]
        self.plugin_command_index = {
            str(i): cmd for i, cmd in enumerate(self.plugin_menu_entries)
        }
        for cmd in plugin_commands:
            # Регистрируем обработчик callback_query если он есть
            if 'callback_query_handler' in cmd and 'callback_pattern' in cmd:
                handler = CallbackQueryHandler(
                    lambda update, context, cmd=cmd: self.handle_plugin_callback_query(update, context, cmd),
                    pattern=cmd['callback_pattern']
                )
                application.add_handler(handler)
                continue
                
            # Регистрируем обычную команду
            command_name = cmd.get('command')
            if not command_name:
                continue
            handler = CommandHandler(
                command_name,
                lambda update, context, cmd=cmd: self.handle_plugin_command(update, context, cmd),
                filters=filters.COMMAND
            )
            application.add_handler(handler)

        self._register_plugin_message_handlers(application, "post_init")
        self._register_message_tail_handlers(application)

        # Обновляем команды бота
        await application.bot.set_my_commands(self.commands)
        await application.bot.set_my_commands(
            self.group_commands,
            scope=BotCommandScopeAllGroupChats()
        )

        # Регистрируем стандартные обработчики callback_query
        application.add_handler(CallbackQueryHandler(self.handle_prompt_selection, pattern="^prompt|promptgroup|promptback"))
        application.add_handler(CallbackQueryHandler(self.handle_session_callback, pattern="^session"))
        application.add_handler(CallbackQueryHandler(self.handle_settings_callback, pattern="^settings"))
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query, pattern="^gpt:"))
        application.add_handler(CallbackQueryHandler(self.handle_plugin_menu_callback, pattern="^pluginmenu:"))
        application.add_handler(CommandHandler("plugins", self.handle_plugins_menu, filters=filters.COMMAND))

        for plugin_instance in getattr(self.openai.plugin_manager, "plugin_instances", {}).values():
            startup_hook = getattr(plugin_instance, "on_startup", None)
            if not startup_hook:
                continue
            try:
                await startup_hook(application)
            except Exception:
                logging.exception(
                    "on_startup hook failed for plugin %s",
                    plugin_instance.get_plugin_id(),
                )

    async def handle_plugin_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, cmd: Dict):
        query = update.callback_query
        if not await is_allowed(self.config, update, context):
            if query:
                await query.edit_message_text(
                    text=localized_text('access_denied_command', self.config['bot_language'])
                )
            return
        user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
        plugin_name = cmd.get('plugin_name')
        if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
            if query:
                await query.edit_message_text(
                    text=localized_text('settings_plugin_disabled', self.config['bot_language']).format(
                        plugin=plugin_name
                    )
                )
            return
        return await cmd['callback_query_handler'](update, context)

    async def handle_plugin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, cmd: Dict):
        """Обработчик команд плагинов"""
        try:
            update_for_handler = self._wrap_update_with_message(update)
            message = update_for_handler.effective_message or (update_for_handler.callback_query.message if update_for_handler.callback_query else None)
            # Проверяем права доступа
            if not await is_allowed(self.config, update_for_handler, context):
                await self.send_disallowed_message(update, context)
                return

            user_id = getattr(getattr(update_for_handler, 'effective_user', None), 'id', None)
            plugin_name = cmd.get('plugin_name')
            if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
                if message:
                    await message.reply_text(
                        localized_text('settings_plugin_disabled', self.config['bot_language']).format(
                            plugin=plugin_name
                        )
                    )
                return

            # Получаем существующий инстанс плагина
            plugin_instance = self.openai.plugin_manager.get_plugin(plugin_name)
            if not plugin_instance:
                raise ValueError(f"Plugin {plugin_name} not found")

            handler = getattr(plugin_instance, cmd.get('handler').__name__)
            if not handler:
                raise ValueError("Handler not specified in command")

            # Анализируем сигнатуру обработчика
            import inspect
            handler_params = inspect.signature(handler).parameters
            is_telegram_handler = 'update' in handler_params and 'context' in handler_params

            if is_telegram_handler:
                # Для обработчиков команд Telegram
                result = await handler(update_for_handler, context)
                if result:  # Если обработчик что-то вернул
                    if is_direct_result(result):
                        await self._handle_direct_result(update_for_handler, result)
                    elif isinstance(result, dict) and "error" in result:
                        if message:
                            await message.reply_text(
                                localized_text('error_with_details', self.config['bot_language']).format(
                                    error=result['error']
                                )
                            )
                    elif isinstance(result, dict) and "text" in result and "parse_mode" in result:
                        if message:
                            await message.reply_text(
                                text=result["text"],
                                parse_mode=result["parse_mode"]
                            )
                    else:
                        if message:
                            await message.reply_text(str(result))
                return

            # Для обработчиков функций плагина
            args = context.args
            kwargs = cmd['handler_kwargs'].copy()
            
            # Если команда требует аргументы, но они не предоставлены
            if cmd.get('args') and not args:
                if message:
                    await message.reply_text(
                        localized_text('plugins_menu_usage', self.config['bot_language']).format(
                            command=cmd.get('command') or cmd.get('name') or '',
                            args=cmd.get('args', '')
                        )
                        + "\n"
                        + localized_text('plugins_menu_description_label', self.config['bot_language']).format(
                            description=cmd.get('description', '')
                        )
                    )
                return

            # Добавляем chat_id и аргументы в kwargs
            kwargs['chat_id'] = str(update_for_handler.effective_chat.id)
            kwargs['update'] = update_for_handler
            kwargs['function_name'] = cmd['handler_kwargs'].get('function_name')  # Берем из handler_kwargs
            if cmd.get('args'):
                kwargs['query'] = ' '.join(args)
                if '<document_id>' in cmd.get('args'):
                    kwargs['document_id'] = args[0]

            # Вызываем обработчик команды
            result = await handler(kwargs['function_name'], self.openai, **{k:v for k,v in kwargs.items() if k != 'function_name'})
            
            # Обрабатываем результат
            if is_direct_result(result):
                await self._handle_direct_result(update_for_handler, result)
            elif isinstance(result, dict) and 'error' in result:
                if message:
                    await message.reply_text(
                        localized_text('error_with_details', self.config['bot_language']).format(
                            error=result['error']
                        )
                    )
            elif isinstance(result, dict) and "text" in result and "parse_mode" in result:
                if message:
                    await message.reply_text(
                        text=result["text"],
                        parse_mode=result["parse_mode"]
                    )
            elif result:
                if message:
                    await message.reply_text(str(result))

        except Exception as e:
            logger.error(f"Ошибка при обработке команды плагина: {e}")
            message = update.effective_message or (update.callback_query.message if update.callback_query else None)
            if message:
                await message.reply_text(
                    localized_text('plugin_command_error', self.config['bot_language']).format(
                        error=str(e)
                    )
                )

    def _wrap_update_with_message(self, update: Update) -> Update:
        if update.message or not update.callback_query or not update.callback_query.message:
            return update

        class _UpdateProxy:
            __slots__ = ("_update", "message")
            def __init__(self, original, message):
                self._update = original
                self.message = message
            def __getattr__(self, name):
                return getattr(self._update, name)
            @property
            def effective_message(self):
                return self._update.effective_message or self.message

        return _UpdateProxy(update, update.callback_query.message)

    async def handle_plugins_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает меню плагинов с командами."""
        if not await is_allowed(self.config, update, context):
            await self.send_disallowed_message(update, context)
            return

        bot_language = self.config['bot_language']
        user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
        disabled_plugins = self.openai.plugin_manager.disabled_plugins_for_user(user_id)
        plugin_commands = self.openai.plugin_manager.build_bot_commands()["plugin_commands"]
        menu_entries = [
            cmd for cmd in plugin_commands
            if cmd.get("add_to_menu") and cmd.get("command") and cmd.get("description")
            and cmd.get("plugin_name") not in disabled_plugins
        ]
        self.plugin_menu_entries = menu_entries
        if not self.plugin_menu_entries:
            await update.message.reply_text(localized_text('plugins_menu_no_plugins', bot_language))
            return

        reply_markup = self._build_plugins_menu(page=0, plugin=None)
        await update.message.reply_text(
            localized_text('plugins_menu_plugins_title', bot_language),
            reply_markup=reply_markup
        )

    async def handle_plugin_menu_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик выбора команды из меню плагинов."""
        query = update.callback_query
        await query.answer()
        if not await is_allowed(self.config, update, context):
            await query.edit_message_text(
                text=localized_text('access_denied_command', self.config['bot_language'])
            )
            return

        data = query.data.split(":")
        if len(data) < 2:
            return
        bot_language = self.config['bot_language']
        action = data[1]
        if action == "close":
            context.user_data.pop("plugin_menu_pending", None)
            await query.message.delete()
            return

        if action == "page" and len(data) == 4:
            scope = data[2]
            try:
                page = int(data[3])
            except ValueError:
                return
            if scope == "root":
                reply_markup = self._build_plugins_menu(page=page, plugin=None)
                await query.edit_message_text(
                    localized_text('plugins_menu_plugins_title', bot_language),
                    reply_markup=reply_markup
                )
            else:
                reply_markup = self._build_plugins_menu(page=page, plugin=scope)
                await query.edit_message_text(
                    localized_text('plugins_menu_plugin_title', bot_language).format(
                        plugin=self._format_plugin_title(scope)
                    ),
                    reply_markup=reply_markup
                )
            return

        if action == "plugin" and len(data) == 3:
            plugin_name = data[2]
            reply_markup = self._build_plugins_menu(page=0, plugin=plugin_name)
            await query.edit_message_text(
                localized_text('plugins_menu_plugin_title', bot_language).format(
                    plugin=self._format_plugin_title(plugin_name)
                ),
                reply_markup=reply_markup
            )
            return

        if action == "input" and len(data) == 4:
            plugin_name = data[2]
            cmd_id = data[3]
            cmd = self._get_plugin_command(plugin_name, cmd_id)
            if not cmd:
                await query.edit_message_text(
                    localized_text('plugins_menu_command_unavailable', bot_language)
                )
                return
            user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
            if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
                await query.edit_message_text(
                    localized_text('settings_plugin_disabled', bot_language).format(plugin=plugin_name)
                )
                return
            prompt_message = await query.message.reply_text(
                localized_text('plugins_menu_enter_params_prompt', bot_language).format(
                    command=cmd.get('command') or cmd.get('name') or '',
                    args=cmd.get('args', '')
                ),
                reply_markup=ForceReply(selective=True)
            )
            context.user_data["plugin_menu_pending"] = {
                "plugin": plugin_name,
                "cmd_id": cmd_id,
                "prompt_message_id": prompt_message.message_id,
            }
            return

        if action != "cmd" or len(data) != 4:
            return
        plugin_name = data[2]
        cmd_id = data[3]
        cmd = self._get_plugin_command(plugin_name, cmd_id)
        if not cmd:
            await query.edit_message_text(
                localized_text('plugins_menu_command_unavailable', bot_language)
            )
            return
        user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
        if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
            await query.edit_message_text(
                localized_text('settings_plugin_disabled', bot_language).format(plugin=plugin_name)
            )
            return

        if cmd.get("args"):
            back_page = self._get_page_for_command_id(cmd_id)
            keyboard = [
                [
                    InlineKeyboardButton(
                        localized_text('plugins_menu_enter_params', bot_language),
                        callback_data=f"pluginmenu:input:{plugin_name}:{cmd_id}"
                    )
                ],
                [
                    InlineKeyboardButton(
                        localized_text('plugins_menu_back', bot_language),
                        callback_data=f"pluginmenu:page:{plugin_name}:{back_page}"
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"❌ {localized_text('settings_close', bot_language)}",
                        callback_data="pluginmenu:close"
                    )
                ],
            ]
            await query.edit_message_text(
                localized_text('plugins_menu_usage', bot_language).format(
                    command=cmd.get('command') or cmd.get('name') or '',
                    args=cmd.get('args', '')
                )
                + "\n"
                + localized_text('plugins_menu_description_label', bot_language).format(
                    description=cmd.get('description', '')
                ),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        try:
            await self.handle_plugin_command(update, context, cmd)
        except Exception as e:
            logger.error(f"Ошибка при выполнении команды из меню: {e}")
            await query.edit_message_text(
                localized_text('plugins_menu_error', bot_language).format(error=str(e))
            )

    async def handle_plugin_menu_args_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ответов на запрос параметров команд плагинов."""
        if not update.message:
            return

        pending = context.user_data.get("plugin_menu_pending")
        is_plugin_menu_reply = (
            pending
            and update.message.reply_to_message
            and update.message.reply_to_message.message_id == pending.get("prompt_message_id")
        )
        if not is_plugin_menu_reply:
            if not filters.COMMAND.check_update(update):
                await self.prompt(update, context)
            return

        plugin_name = pending.get("plugin")
        cmd = self._get_plugin_command(plugin_name, pending.get("cmd_id")) if plugin_name else None
        if not cmd:
            await update.effective_message.reply_text(
                localized_text('plugins_menu_command_unavailable', self.config['bot_language'])
            )
            context.user_data.pop("plugin_menu_pending", None)
            return

        context.args = update.message.text.split() if update.message.text else []
        context.user_data.pop("plugin_menu_pending", None)
        await self.handle_plugin_command(update, context, cmd)

    def _build_plugins_menu(self, page: int = 0, plugin: str | None = None) -> InlineKeyboardMarkup:
        if plugin is None:
            items = list(self._get_plugins_list())
        else:
            items = list(self._get_plugin_commands(plugin))
        page_size = max(1, self.plugin_menu_page_size)
        total_pages = (len(items) + page_size - 1) // page_size
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        end = start + page_size

        keyboard = []
        if plugin is None:
            for plugin_name in items[start:end]:
                title = self._format_plugin_title(plugin_name)
                keyboard.append([
                    InlineKeyboardButton(
                        title,
                        callback_data=f"pluginmenu:plugin:{plugin_name}"
                    )
                ])
        else:
            for idx, cmd in items[start:end]:
                cmd_name = cmd.get('command') or cmd.get('name') or ''
                title = f"/{cmd_name} — {cmd.get('description', '')}"
                keyboard.append([
                    InlineKeyboardButton(
                        title,
                        callback_data=f"pluginmenu:cmd:{plugin}:{idx}"
                    )
                ])

        nav_row = []
        if page > 0:
            scope = "root" if plugin is None else plugin
            nav_row.append(InlineKeyboardButton("⬅️", callback_data=f"pluginmenu:page:{scope}:{page-1}"))
        if page < total_pages - 1:
            scope = "root" if plugin is None else plugin
            nav_row.append(InlineKeyboardButton("➡️", callback_data=f"pluginmenu:page:{scope}:{page+1}"))
        if plugin is not None:
            nav_row.append(InlineKeyboardButton(
                localized_text('plugins_menu_back', self.config['bot_language']),
                callback_data="pluginmenu:page:root:0"
            ))
        if nav_row:
            keyboard.append(nav_row)
        keyboard.append([
            InlineKeyboardButton(
                f"❌ {localized_text('settings_close', self.config['bot_language'])}",
                callback_data="pluginmenu:close"
            )
        ])
        return InlineKeyboardMarkup(keyboard)

    def _get_plugins_list(self) -> list[str]:
        return sorted({cmd.get("plugin_name") for cmd in self.plugin_menu_entries if cmd.get("plugin_name")})

    def _get_plugin_commands(self, plugin_name: str) -> list[tuple[str, dict]]:
        commands = [
            cmd for cmd in self.plugin_menu_entries
            if cmd.get("plugin_name") == plugin_name
        ]
        return [(str(i), cmd) for i, cmd in enumerate(commands)]

    def _get_plugin_command(self, plugin_name: str, cmd_id: str) -> dict | None:
        try:
            idx = int(cmd_id)
        except ValueError:
            return None
        commands = [
            cmd for cmd in self.plugin_menu_entries
            if cmd.get("plugin_name") == plugin_name
        ]
        if 0 <= idx < len(commands):
            return commands[idx]
        return None

    def _format_plugin_title(self, plugin_name: str) -> str:
        return plugin_name.replace("_", " ").strip()

    def _get_page_for_command_id(self, cmd_id: str) -> int:
        try:
            idx = int(cmd_id)
        except ValueError:
            return 0
        page_size = max(1, self.plugin_menu_page_size)
        return idx // page_size

    async def cleanup(self):
        """
        Comprehensive cleanup:
        - Cancel all timers
        - Clear message buffers
        - Stop background tasks
        - Close any open connections/resources
        """
        try:
            if self._cleanup_called:
                return
            self._cleanup_called = True
            # Stop plugin-declared background tasks
            try:
                if hasattr(self, 'openai') and hasattr(self.openai, 'plugin_manager'):
                    await self.openai.plugin_manager.stop_background_tasks(timeout=10.0)
            except Exception as e:
                logger.warning(f"Error stopping plugin background tasks: {e}")
            # Stop all background tasks first
            tasks = []
            if hasattr(self, '_background_tasks'):
                tasks.extend(self._background_tasks)
            
            # Get all buffer timers
            async with self.buffer_lock:
                for buffer_data in self.message_buffer.values():
                    if buffer_data.get('timer'):
                        tasks.append(buffer_data['timer'])
                
                # Clear message buffers
                self.message_buffer.clear()

            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except Exception as e:
                        logger.error(f"Error cancelling task: {e}")

            # Cancel all running tasks except current
            current_task = asyncio.current_task()
            running_tasks = [t for t in asyncio.all_tasks() 
                           if t is not current_task and not t.done()]
            
            if running_tasks:
                logger.info(f"Cancelling {len(running_tasks)} remaining tasks...")
                for task in running_tasks:
                    task.cancel()
                
                await asyncio.gather(*running_tasks, return_exceptions=True)

            # Close any open resources
            if hasattr(self, 'openai'):
                await self.openai.close()
            if hasattr(self.openai, 'plugin_manager'):
                try:
                    await self.openai.plugin_manager.close_all_async()
                except Exception as e:
                    logger.warning(f"Error in plugin close_all_async: {e}")
                self.openai.plugin_manager.close_all()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

    async def _post_shutdown(self, application: Application):
        await self.cleanup()

    async def buffer_data_checker(self):
        """
        Periodically checks and processes buffered messages
        """
        while True:
            try:
                async with self.buffer_lock:
                    for chat_id, buffer_data in list(self.message_buffer.items()):  # Create copy for iteration                
                        if not buffer_data['processing'] and buffer_data['messages']:
                            if buffer_data.get('timer'):
                                buffer_data['timer'].cancel()
                            buffer_data['timer'] = asyncio.create_task(
                                self.process_buffer(chat_id)
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in buffer data checker: {e}")
            
            await asyncio.sleep(1)


    async def handle_session_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик callback-запросов для управления сессиями.
        """
        query = update.callback_query
        await query.answer()

        if not await is_allowed(self.config, update, context):
            await query.edit_message_text(
                text=localized_text('access_denied_command', self.config['bot_language'])
            )
            return

        conversation_key = get_conversation_key(update)
        conversation_lock = await self._get_conversation_lock(conversation_key)
        async with conversation_lock:
            return await self._handle_session_callback_locked(
                update,
                context,
                query,
                conversation_key,
            )

    async def _handle_session_callback_locked(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query,
        conversation_key,
    ):
        data = query.data.split(':')
        action = data[1]

        # Существующие действия...
        if action == 'preview':
            session_id = data[2]
            
            # Получаем детали сессии
            session = self.db.get_session_details(conversation_key, session_id)
            if not session:
                await query.edit_message_text(
                    localized_text('session_not_found', self.config['bot_language'])
                )
                return

            # Получаем все сообщения сессии
            context_messages = self.db.get_conversation_context(
                conversation_key,
                session_id=session_id, 
                openai_helper=self.openai
            )

            # Получаем список сообщений из контекста
            context_messages, _, _, _, _ = context_messages
            context_messages = context_messages.get('messages', [])

            # Формируем preview
            preview_text = localized_text('session_preview_title', self.config['bot_language']).format(
                session_name=session['session_name']
            ) + "\n\n"
            for msg in context_messages:
                role = "🤖" if msg['role'] == 'assistant' or msg['role'] == 'system' else "👤"
                if len(msg['content']) > 200:
                    preview_text += f"{role} {msg['content'][:200]}...\n"
                else:
                    preview_text += f"{role} {msg['content']}\n"

            preview_text += "\n" + localized_text(
                'session_preview_total_messages', self.config['bot_language']
            ).format(count=session['message_count'])
            preview_text += "\n" + localized_text(
                'session_preview_created_at', self.config['bot_language']
            ).format(created_at=session['created_at'])

            # Добавляем inline-кнопки для возврата
            keyboard = [
                [InlineKeyboardButton(
                    localized_text('session_back_to_sessions_label', self.config['bot_language']),
                    callback_data="session:back"
                )]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                preview_text, 
                reply_markup=reply_markup
            )
            return

        try:
            if action == "close":
                # Просто удаляем сообщение
                await query.message.delete()
                return

            # Остальной существующий код остается без изменений
            if action == "new":
                max_sessions = self.config.get('max_sessions', 5)
                await self._dispatch_and_delete_oldest_sessions_for_limit(conversation_key, max_sessions)
                # Создаем новую сессию
                session_id = self.db.create_session(
                    user_id=conversation_key,
                    max_sessions=max_sessions,
                    openai_helper=self.openai,
                    prune_old_sessions=False,
                )
                
                if not session_id:
                    await query.edit_message_text(
                        text=localized_text('session_create_failed', self.config['bot_language'])
                    )
                    return
                
                # Сбрасываем историю чата для новой сессии
                await self.openai.reset_chat_history(
                    chat_id=conversation_key,
                    content='',
                    session_id=session_id
                )
                await self.reset(update, context)  # Обновляем список сессий
                    
            elif action == "switch":
                # Переключаемся на выбранную сессию
                session_id = data[2]
                if not self.db.switch_active_session(conversation_key, session_id):
                    await query.edit_message_text(
                        localized_text('session_not_found', self.config['bot_language'])
                    )
                    return
                # Загружаем контекст выбранной сессии
                current_context, parse_mode, temperature, max_tokens_percent, _ = self.db.get_conversation_context(conversation_key, session_id)
                if current_context and 'messages' in current_context:
                    self.openai.conversations[conversation_key] = current_context['messages']
                    self.openai.loaded_conversation_sessions[conversation_key] = session_id
                await self.reset(update, context)  # Обновляем список сессий
                
            elif action == "delete":
                # Удаляем сессию
                session_id = data[2]
                await self._dispatch_session_before_delete(conversation_key, session_id)
                self.db.delete_session(conversation_key, session_id, openai_helper=self.openai)
                # Получаем контекст активной сессии
                session_id = self.db.get_active_session_id(conversation_key)
                current_context, _, _, _, _ = self.db.get_conversation_context(conversation_key, session_id, openai_helper=self.openai)
                if current_context and 'messages' in current_context:
                    self.openai.conversations[conversation_key] = current_context['messages']
                    self.openai.loaded_conversation_sessions[conversation_key] = session_id
                await self.reset(update, context)  # Обновляем список сессий
                
            elif action == "change_mode":
                # Показываем меню выбора режима для текущей сессии
                keyboard = []
                
                # Используем кешированные режимы
                chat_modes = self.get_chat_modes()
                
                # Группируем режимы по group
                mode_groups = {}
                for mode_key, mode_data in chat_modes.items():
                    group = mode_data.get('group', localized_text('session_group_other', self.config['bot_language']))
                    if group not in mode_groups:
                        mode_groups[group] = []
                    mode_groups[group].append((mode_key, mode_data))
                
                # Добавляем группы режимов
                for group_name in sorted(mode_groups.keys()):
                    keyboard.append([InlineKeyboardButton(
                        text=group_name,
                        callback_data=f"promptgroup:{group_name}"
                    )])
                
                # Добавляем кнопку "Назад"
                keyboard.append([InlineKeyboardButton(
                    text=localized_text('session_back_to_sessions', self.config['bot_language']),
                    callback_data="session:back"
                )])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    text=localized_text('session_choose_mode_group', self.config['bot_language']),
                    reply_markup=reply_markup
                )
                
            elif action == "change_model":
                logger.info("Model switching callback ignored because model selection is disabled.")
                await self.reset(update, context)
                
            elif action == "back":
                # Возвращаемся к списку сессий
                await self.reset(update, context)
                
            elif action == "export":
                # Экспортируем сессии в YAML
                try:
                    filepath = self.db.export_sessions_to_yaml(conversation_key)
                    
                    if filepath:
                        # Отправляем файл пользователю
                        with open(filepath, 'rb') as file:
                            await query.message.reply_document(
                                document=file, 
                                filename=os.path.basename(filepath),
                                caption=localized_text('session_export_done', self.config['bot_language'])
                            )
                        
                        # Удаляем файл после отправки
                        os.remove(filepath)
                    else:
                        await query.edit_message_text(
                            localized_text('session_export_failed', self.config['bot_language'])
                        )
                except Exception as e:
                    logger.error(f"Ошибка экспорта сессий: {e}")
                    await query.edit_message_text(
                        localized_text('session_export_error', self.config['bot_language'])
                    )
                
                # Возвращаемся к списку сессий
                await self.reset(update, context)
        except Exception as e:
            logger.error(f'Error in handle_session_callback: {e}', exc_info=True)
            await query.edit_message_text(
                text=localized_text('error_with_details', self.config['bot_language']).format(
                    error=str(e)
                )
            )

    def run(self):
        """
        Runs the bot indefinitely.
        """
        try:
            # Проверяем, есть ли уже запущенный event loop
            try:
                loop = asyncio.get_running_loop()
                logger.warning("Event loop is already running, using existing loop")
            except RuntimeError:
                # Нет активного loop, создаем новый
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            builder = ApplicationBuilder() \
                .token(self.config['token']) \
                .post_init(self.post_init) \
                .post_shutdown(self._post_shutdown) \
                .concurrent_updates(True)

            telegram_local_mode = self.config.get('telegram_local_mode', True)
            telegram_base_url = self.config.get(
                'telegram_base_url',
                DEFAULT_TELEGRAM_BASE_URL
            )
            builder = builder.local_mode(telegram_local_mode)
            if telegram_local_mode and telegram_base_url:
                builder = builder.base_url(telegram_base_url)

            application = builder.build()

            self.application = application
            self.openai.bot = application.bot
            self._background_tasks = []
            self._plugin_message_handlers_registered = False
            self._message_tail_handlers_registered = False

            application.add_handler(TypeHandler(Update, self.prepare_user_language), group=-100)
            application.add_handler(CommandHandler('restart', self.restart))
            application.add_handler(CommandHandler('reset', self.reset))
            application.add_handler(CommandHandler('help', self.help))
            application.add_handler(CommandHandler('settings', self.settings))
            application.add_handler(CommandHandler('image', self.image))
            application.add_handler(CommandHandler('tts', self.tts))
            application.add_handler(CommandHandler('start', self.help))
            application.add_handler(CommandHandler('stats', self.stats))
            application.add_handler(CommandHandler('resend', self.resend))
            application.add_handler(CommandHandler(
                'chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)
            )

            application.add_handler(MessageHandler(
                filters.PHOTO | filters.Document.IMAGE,
                self.vision))
            application.add_handler(MessageHandler(
                filters.AUDIO | filters.VOICE | filters.Document.AUDIO |
                filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
                self.transcribe))

            application.run_polling()
        finally:
            # Завершаем выполнение задач в текущем событийном цикле
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(self.cleanup())
