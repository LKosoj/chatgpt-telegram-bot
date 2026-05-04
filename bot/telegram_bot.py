from __future__ import annotations

import asyncio
import logging
import os
import io
import requests
import json
import sys
import yaml
import re
from typing import Dict
from functools import lru_cache 

from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand, ForceReply
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
from PIL import Image

from .utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files, send_long_response_as_file
from .openai_helper import OpenAIHelper, O_MODELS, ANTHROPIC, GOOGLE, MISTRALAI, DEEPSEEK, PERPLEXITY
from .i18n import localized_text
from .plugins.haiper_image_to_video import WAITING_PROMPT
from .usage_tracker import UsageTracker
from .database import Database
from .conversation_key import get_conversation_key

#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WAITING_PROMPT = 1


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
        # Устанавливаем openai в plugin_manager
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
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
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

    def get_chat_modes(self):
        """
        Получает chat_modes с кешированием
        """
        import time
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
            logger.warning("Failed to classify reply intent with light model; using legacy routing", exc_info=True)
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

    async def _handle_direct_result(self, update: Update, response):
        sent_messages = await handle_direct_result(self.config, update, response)
        self._remember_sent_image_messages(update, sent_messages)

    def _schedule_hindsight_session_finalize(self, user_id: int, session_id: str | None) -> None:
        if not session_id:
            return
        if not self.openai.is_hindsight_enabled() or not self.openai.config.get('hindsight_auto_save', True):
            return

        try:
            context, _, _, _, _ = self.db.get_conversation_context(user_id, session_id)
            messages = context.get('messages', []) if isinstance(context, dict) else []
            messages = [dict(message) for message in messages if isinstance(message, dict)]
        except Exception:
            logger.warning(
                "Failed to snapshot session before background Hindsight finalize for user_id=%s session_id=%s",
                user_id,
                session_id,
                exc_info=True,
            )
            return

        async def _finalize():
            try:
                saved_count = await self.openai.finalize_hindsight_session_memory(user_id, session_id, messages=messages)
                if saved_count:
                    logger.info(
                        "Background Hindsight session finalize saved %s item(s) for user_id=%s session_id=%s",
                        saved_count,
                        user_id,
                        session_id,
                    )
            except Exception:
                logger.warning(
                    "Background Hindsight session finalize failed for user_id=%s session_id=%s",
                    user_id,
                    session_id,
                    exc_info=True,
                )

        asyncio.create_task(_finalize(), name=f"hindsight_finalize_{user_id}_{session_id}")

    def _is_image_edit_request(self, text: str | None) -> bool:
        prompt = (text or '').strip().lower()
        if not prompt:
            return False
        edit_markers = (
            'отредакт', 'редакт', 'измени', 'изменить', 'поменяй', 'поменять',
            'добавь', 'добавить', 'убери', 'убрать', 'замени', 'заменить',
            'дорисуй', 'дорисовать', 'перерисуй', 'перерисовать',
            'нарисуй', 'нарисовать',
            'надень', 'одень', 'сделай ему', 'сделай ей', 'сделай им',
            'edit', 'modify', 'change', 'draw ', 'add ', 'remove ', 'replace', 'put ',
        )
        return any(marker in prompt for marker in edit_markers)

    def _can_use_last_image_for_edit(self, text: str | None) -> bool:
        prompt = (text or '').strip().lower()
        strong_edit_markers = (
            'отредакт', 'редакт', 'измени', 'изменить', 'поменяй', 'поменять',
            'перерисуй', 'перерисовать', 'нарисуй', 'нарисовать',
            'edit', 'modify', 'change', 'draw ',
        )
        image_reference_markers = (
            'это изображ', 'эту карт', 'эта карт', 'этот рисун', 'этого кот',
            'этому', 'этой', 'этому кот', 'ему ', 'ей ', 'на нем', 'на ней',
            'this image', 'that image', 'this picture', 'it ', 'him ', 'her ',
        )
        return any(marker in prompt for marker in strong_edit_markers + image_reference_markers)

    def _is_image_description_request(self, text: str | None) -> bool:
        prompt = (text or '').strip().lower()
        if not prompt:
            return False
        description_markers = (
            'опиши', 'описать', 'что на', 'что изображено', 'что тут',
            'что здесь', 'что это', 'расскажи про изображение',
            'разбери изображение', 'проанализируй изображение',
            'describe', 'what is in', 'what is on', 'what\'s in',
            'analyze this image', 'analyse this image',
        )
        return any(marker in prompt for marker in description_markers)

    def _can_use_last_image_for_description(self, text: str | None) -> bool:
        prompt = (text or '').strip().lower()
        last_image_markers = (
            'опиши', 'описать', 'что на', 'что изображено',
            'расскажи про изображение', 'разбери изображение',
            'проанализируй изображение', 'describe', 'what is in',
            'what is on', 'what\'s in', 'analyze this image', 'analyse this image',
        )
        return any(marker in prompt for marker in last_image_markers)

    def _last_saved_image_file_id(self, user_id: int, chat_id: int) -> str | None:
        images = self.db.get_user_images(user_id, chat_id, limit=5)
        if not images:
            return None
        active = next((image for image in images if image.get('status') == 'active'), None)
        return (active or images[0]).get('file_id')

    def _image_edit_source_file_id(self, update: Update, user_id: int, chat_id: int, prompt: str | None = None) -> str | None:
        message = update.effective_message
        current_image = self._image_file_id_from_message(message)
        if current_image:
            return current_image
        replied_image = self._image_file_id_from_message(getattr(message, 'reply_to_message', None))
        if replied_image:
            return replied_image
        if not self._can_use_last_image_for_edit(prompt):
            return None
        return self._last_saved_image_file_id(user_id, chat_id)

    def _image_description_source_file_id(self, update: Update, user_id: int, chat_id: int, prompt: str | None = None) -> str | None:
        message = update.effective_message
        current_image = self._image_file_id_from_message(message)
        if current_image:
            return current_image
        replied_image = self._image_file_id_from_message(getattr(message, 'reply_to_message', None))
        if replied_image:
            return replied_image
        if not self._can_use_last_image_for_description(prompt):
            return None
        return self._last_saved_image_file_id(user_id, chat_id)

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
                "add_value": "Изображение отредактировано через LLMGateway.",
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
        import time
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
        #tool_list = "\n".join([f"- {tool['name']}" for tool in TOOLS])
        help_text = (
            localized_text('help_text', bot_language)[0]
            + '\n\n'
            + '\n'.join(commands_description)
            + '\n\n'
            + localized_text('help_text', bot_language)[1]
            + '\n\n'
            + localized_text('help_text', bot_language)[2]
            + '\n\n'
            + localized_text('help_extra', bot_language)
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

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
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
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
        if remaining_budget < float('inf'):
            text_budget += (
                f"{localized_text('stats_budget', bot_language)}"
                f"{localized_text(budget_period, bot_language)}: "
                f"${remaining_budget:.2f}.\n"
            )
        # If use vsegpt, return money rest
        if is_admin(self.config, user_id) and 'vsegpt' in self.config['openai_base']:
             text_budget += (
                 " VSEGPT "
                 f"{self.get_credits()}"
             )

        usage_text = text_current_session + text_all_sessions + text_today + text_month + text_budget
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)
    
    def get_credits(self):
        api_key = self.config['api_key']
        response = requests.get(
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
            return ValueError(str(response.status_code) + ": " + response.text)        

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

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error: bool = False):
        """
        Сброс контекста разговора и управление сессиями
        """
        # Проверяем, что effective_chat существует (не для inline queries)
        if not update.effective_chat:
            logger.warning("reset called without effective_chat (likely inline query)")
            return
            
        chat_id = update.effective_chat.id
        is_callback = bool(update.callback_query)
        user_id = None
        
        # Получаем информацию о пользователе из callback_query или message
        if is_callback:
            user_id = update.callback_query.from_user.id
            if not await is_allowed(self.config, update, context):
                await update.callback_query.edit_message_text(
                    text=localized_text('access_denied_command', self.config['bot_language'])
                )
                return
        elif update.message:
            user_id = update.message.from_user.id
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
            # Получаем список сессий пользователя
            sessions = self.db.list_user_sessions(user_id)
            
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
                    current_model = self.openai.get_current_model(user_id)
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
        
        chat_id = query.message.chat_id
        user_id = query.from_user.id
        
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
                    sessions = self.db.list_user_sessions(user_id, is_active=1)
                    active_session = next((s for s in sessions if s['is_active']), None)
                    
                    if not active_session:
                        # Если нет активной сессии, создаем новую
                        session_id = self.db.create_session(
                            user_id=user_id,
                            max_sessions=self.config.get('max_sessions', 5),
                            openai_helper=self.openai
                        )
                    else:
                        session_id = active_session['session_id']
                    
                    mode_data = chat_modes[mode]
                    # Получаем текущий контекст сессии
                    current_context = self.openai.conversations.get(chat_id, [])
                    
                    # Добавляем системное сообщение в начало контекста
                    reset_content = mode_data.get('prompt_start', '')
                    system_message = {"role": "system", "content": reset_content}
                    
                    # Если текущий контекст уже содержит системное сообщение, заменяем его
                    if current_context and current_context[0].get('role') == 'system':
                        current_context[0] = system_message
                    else:
                        current_context.insert(0, system_message)
                    
                    # Обновляем контекст в OpenAI и базе данных
                    self.openai.conversations[chat_id] = current_context
                    
                    # Сохраняем настройки режима в базу данных
                    self.db.save_conversation_context(
                        chat_id, 
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
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

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
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_tts_request(text_length, self.config['tts_model'],
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
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=transcript, **kwargs)

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
                    if len(chunks) > 3 or (len(chunks) > 1 and '```' in response):
                        # Получаем имя текущей сессии
                        sessions = self.db.list_user_sessions(user_id, is_active=1)
                        active_session = next((s for s in sessions if s['is_active']), None)
                        session_name = active_session['session_name'] if active_session else 'transcription'
                        
                        await send_long_response_as_file(self.config, update, response, session_name)
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

        # Store the image in database
        if len(update.message.photo) > 0:
            file_id = update.message.photo[-1].file_id
            logger.info(f"Storing photo file_id: {file_id}")
            self.db.save_image(user_id, chat_id, file_id)
        elif update.message.document and update.message.document.mime_type.startswith('image/'):
            file_id = update.message.document.file_id
            logger.info(f"Storing document file_id: {file_id}")
            self.db.save_image(user_id, chat_id, file_id)

        if prompt and self.config['enable_image_generation'] and self._is_image_edit_request(prompt):
            source_file_id = self._image_edit_source_file_id(update, user_id, chat_id, prompt)
            if source_file_id:
                async def _edit():
                    await self._edit_image_from_context(update, prompt, source_file_id)

                await wrap_with_indicator(update, context, _edit, constants.ChatAction.UPLOAD_PHOTO)
                return

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

            image = update.message.effective_attachment[-1]

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
                    stream_chunk = 0

                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            return await self._handle_direct_result(update, content)

                        if len(content.strip()) == 0:
                            continue

                        stream_chunks = split_into_chunks(content)
                        if len(stream_chunks) > 1:
                            content = stream_chunks[-1]
                            if stream_chunk != len(stream_chunks) - 1:
                                stream_chunk += 1
                                try:
                                    await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                                stream_chunks[-2])
                                except:
                                    pass
                                try:
                                    sent_message = await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content if len(content) > 0 else "..."
                                    )
                                except:
                                    pass
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
                            except:
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
        try:
            async with self.buffer_lock:
                if chat_id not in self.message_buffer:
                    return
                
                buffer_data = self.message_buffer[chat_id]
                if buffer_data['processing']:
                    return
                    
                buffer_data['processing'] = True
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
        use_legacy_image_routing = reply_intent in (None, "unknown")

        if self.config['enable_image_generation'] and (
            reply_intent == "image_edit"
            or (use_legacy_image_routing and self._is_image_edit_request(prompt))
        ):
            source_file_id = self._image_edit_source_file_id(update, user_id, chat_id, prompt)
            if source_file_id:
                async def _edit():
                    await self._edit_image_from_context(update, prompt, source_file_id)

                await wrap_with_indicator(update, context, _edit, constants.ChatAction.UPLOAD_PHOTO)
                return
            logger.info("Image edit route matched but no source image file_id was found")

        if self.config['enable_vision'] and (
            reply_intent == "image_describe"
            or (use_legacy_image_routing and self._is_image_description_request(prompt))
        ):
            source_file_id = self._image_description_source_file_id(update, user_id, chat_id, prompt)
            if source_file_id:
                async def _describe():
                    await self._describe_image_from_context(update, prompt, source_file_id)

                await wrap_with_indicator(update, context, _describe, constants.ChatAction.TYPING)
                return

        try:
            total_tokens = 0

            model_to_use = self.openai.get_current_model(user_id)
                
            if self.config['stream'] and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY):

                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )

                # Store message_id in openai object for this request
                self.openai.message_ids = getattr(self.openai, 'message_ids', {})
                self.openai.message_ids[request_id] = message_id

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt, request_id=request_id, user_id=user_id)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await self._handle_direct_result(update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              stream_chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
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

            else:
                async def _reply():
                    nonlocal total_tokens
                    # Store message_id in openai object for this request
                    self.openai.message_ids = getattr(self.openai, 'message_ids', {})
                    self.openai.message_ids[request_id] = message_id

                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt, request_id=request_id, user_id=user_id)

                    if is_direct_result(response):
                        analytics_plugin = self.openai.plugin_manager.get_plugin('conversation_analytics')
                        if analytics_plugin:
                            message_data = {
                                'text': prompt,
                                'tokens': total_tokens,
                                'user_id': user_id
                            }
                            analytics_plugin.update_stats(str(chat_id), message_data)
                        return await self._handle_direct_result(update, response)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    # Если ответ больше 3х частей, то формируем файл с ответом и отправлем его
                    if len(chunks) > 3 or (len(chunks) > 1 and '```' in response):
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

                await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)
            # Cleanup the stored message_id after processing is complete
            if hasattr(self.openai, 'message_ids'):
                request_id = f"{chat_id}_{message_id}"
                self.openai.message_ids.pop(request_id, None)

            analytics_plugin = self.openai.plugin_manager.get_plugin('conversation_analytics')
            if analytics_plugin:
                message_data = {
                    'text': prompt,
                    'tokens': total_tokens,
                    'user_id': user_id
                }
                analytics_plugin.update_stats(str(chat_id), message_data)

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
                    
                unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
                if self.config['stream'] and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY):
                    stream_response = self.openai.get_chat_response_stream(chat_id=user_id, query=query)
                    i = 0
                    prev = ''
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            cleanup_intermediate_files(content)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
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
                            except:
                                continue

                        elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                            prev = content
                            try:
                                use_markdown = tokens != 'not_finished'
                                divider = '_' if use_markdown else ''
                                text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                                text = text[:4096]

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
                        response, total_tokens = await self.openai.get_chat_response(chat_id=user_id, query=query)

                        if is_direct_result(response):
                            cleanup_intermediate_files(response)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                          is_inline=True)
                            return

                        text_content = f'{query}\n\n_{answer_tr}:_\n{response}'

                        # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                        text_content = text_content[:4096]

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
        if not is_inline:
            #chat_id = update.effective_chat.id
            #chat_context, parse_mode, temperature = self.db.get_conversation_context(chat_id) or {}
            message = update.effective_message or (update.callback_query.message if update.callback_query else None)
            if message:
                await message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=self.disallowed_message,
                    disable_web_page_preview=True
                )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            #chat_id = update.effective_chat.id
            #chat_context, parse_mode, temperature = self.db.get_conversation_context(chat_id) or {}
            message = update.effective_message or (update.callback_query.message if update.callback_query else None)
            if message:
                await message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=self.budget_limit_message,
                )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

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
                asyncio.create_task(self.start_reminder_checker(self.openai.plugin_manager), name="reminder_checker"),
            ]

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
                    lambda update, context, cmd=cmd: cmd['callback_query_handler'](update, context),
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

        # Регистрируем обработчики сообщений от плагинов
        for handler_config in self.openai.plugin_manager.get_message_handlers():
            if 'handler' in handler_config and 'filters' not in handler_config:
                # Если handler уже является готовым обработчиком (например, ConversationHandler)
                handler = handler_config['handler']
                try:
                    application.add_handler(handler)
                    logger.info(f"Successfully added handler in post_init: {type(handler).__name__}")
                except TypeError as e:
                    logger.error(f"Invalid handler type {type(handler).__name__} in post_init: {e}")
                    continue
            elif 'filters' in handler_config:
                # Если указаны фильтры, создаем MessageHandler
                filter_obj = handler_config['filters']
                if isinstance(filter_obj, str):
                    key = filter_obj.replace("filters.", "").strip()
                    filter_obj = getattr(filters, key, None)
                if filter_obj is None:
                    logger.error(f"Invalid filter in plugin handler config: {handler_config.get('filters')}")
                    continue
                handler = MessageHandler(
                    filter_obj,
                    lambda update, context, h=handler_config: self.handle_plugin_command(
                        update, context, {"handler": h['handler'], **h['handler_kwargs']}
                    )
                )
                application.add_handler(handler)
        
        # Обновляем команды бота
        await application.bot.set_my_commands(self.commands)
        await application.bot.set_my_commands(
            self.group_commands,
            scope=BotCommandScopeAllGroupChats()
        )

        # Регистрируем стандартные обработчики callback_query
        application.add_handler(CallbackQueryHandler(self.handle_prompt_selection, pattern="^prompt|promptgroup|promptback"))
        application.add_handler(CallbackQueryHandler(self.handle_session_callback, pattern="^session"))
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query, pattern="^gpt:"))
        application.add_handler(CallbackQueryHandler(self.handle_plugin_menu_callback, pattern="^pluginmenu:"))
        application.add_handler(CommandHandler("plugins", self.handle_plugins_menu, filters=filters.COMMAND))

    async def handle_plugin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, cmd: Dict):
        """Обработчик команд плагинов"""
        try:
            update_for_handler = self._wrap_update_with_message(update)
            message = update_for_handler.effective_message or (update_for_handler.callback_query.message if update_for_handler.callback_query else None)
            # Проверяем права доступа
            if not await is_allowed(self.config, update_for_handler, context):
                await self.send_disallowed_message(update, context)
                return

            # Получаем существующий инстанс плагина
            plugin_instance = self.openai.plugin_manager.get_plugin(cmd.get('plugin_name'))
            if not plugin_instance:
                raise ValueError(f"Plugin {cmd.get('plugin_name')} not found")

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
                    if isinstance(result, dict) and "text" in result and "parse_mode" in result:
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
        plugin_commands = self.openai.plugin_manager.build_bot_commands()["plugin_commands"]
        menu_entries = [
            cmd for cmd in plugin_commands
            if cmd.get("add_to_menu") and cmd.get("command") and cmd.get("description")
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

        data = query.data.split(":")
        if len(data) < 2:
            return
        bot_language = self.config['bot_language']
        action = data[1]
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

    async def start_reminder_checker(self, plugin_manager):
        reminders_plugin = plugin_manager.get_plugin('reminders')
        if reminders_plugin:
            # Continuously check reminders
            while True:
                try:
                    await reminders_plugin.check_reminders(self.application.bot)
                except Exception as e:
                    logger.error(f"Error in reminder checker: {e}")
                
                # Sleep for a minute between checks to avoid excessive processing
                await asyncio.sleep(60)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик для загруженных документов
        """
        if not await is_allowed(self.config, update, context):
            logger.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                          'is not allowed to upload documents')
            await self.send_disallowed_message(update, context)
            return

        try:
            document = update.message.document
            logger.info(f"Получен документ: {document.file_name}, mime_type: {document.mime_type}")
            
            # Список поддерживаемых MIME-типов
            supported_mimes = [
                'text/plain',                   # .txt
                'application/msword',           # .doc
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
                'application/pdf',              # .pdf
                'application/rtf',              # .rtf
                'application/vnd.oasis.opendocument.text',  # .odt
                'text/markdown'                 # .md
            ]
            
            # Список поддерживаемых расширений (как запасной вариант)
            supported_extensions = ['.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt', '.md']
            file_extension = os.path.splitext(document.file_name)[1].lower()
            
            logger.info(f"Проверка типа файла: mime_type={document.mime_type}, extension={file_extension}")
            
            # Проверяем сначала MIME-тип, потом расширение
            if document.mime_type not in supported_mimes and file_extension not in supported_extensions:
                await update.message.reply_text(
                    localized_text('document_unsupported_format', self.config['bot_language']).format(
                        formats=", ".join(supported_extensions)
                    )
                )
                logger.warning(f"Файл отклонен: неподдерживаемый формат {document.mime_type} / {file_extension}")
                return

            logger.info("Начинаем скачивание файла...")
            # Скачиваем файл
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            logger.info(f"Файл успешно скачан, размер: {len(file_content)} байт")
            
            # Вызываем плагин для обработки документа
            plugin = self.openai.plugin_manager.get_plugin('text_document_qa')
            if not plugin:
                logger.error("Плагин text_document_qa не найден")
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=localized_text('document_processing_unavailable', self.config['bot_language'])
                )
                return

            logger.info("Передаем файл в плагин для обработки...")
            # Execute the plugin function directly
            result = await plugin.execute(
                'upload_document',
                self.openai,
                file_content=file_content,
                file_name=document.file_name,
                chat_id=str(update.effective_chat.id),
                update=update
            )

            # Обрабатываем результат
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Ошибка от плагина: {result['error']}")
                await update.message.reply_text(
                    localized_text('error_with_details', self.config['bot_language']).format(
                        error=result['error']
                    )
                )
            else:
                try:
                    logger.info("Файл успешно обработан, отправляем результат")
                    await self._handle_direct_result(update, result)
                except Exception as e:
                    logger.error(f"Error handling direct result: {e}")
                    await update.message.reply_text(str(result))

        except Exception as e:
            error_text = localized_text('document_processing_error', self.config['bot_language']).format(
                error=str(e)
            )
            logger.error(error_text)
            await update.message.reply_text(error_text)

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

        chat_id = query.message.chat_id
        user_id = query.from_user.id
        data = query.data.split(':')
        action = data[1]

        # Существующие действия...
        if action == 'preview':
            session_id = data[2]
            
            # Получаем детали сессии
            session = self.db.get_session_details(user_id, session_id)
            if not session:
                await query.edit_message_text(
                    localized_text('session_not_found', self.config['bot_language'])
                )
                return

            # Получаем все сообщения сессии
            context_messages = self.db.get_conversation_context(
                user_id, 
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
                # Создаем новую сессию
                session_id = self.db.create_session(
                    user_id=user_id,
                    max_sessions=self.config.get('MAX_SESSIONS', 5),
                    openai_helper=self.openai
                )
                
                if not session_id:
                    await query.edit_message_text(
                        text=localized_text('session_create_failed', self.config['bot_language'])
                    )
                    return
                
                # Сбрасываем историю чата для новой сессии
                self.openai.reset_chat_history(
                    chat_id=user_id,
                    content='',
                    session_id=session_id
                )
                await self.reset(update, context)  # Обновляем список сессий
                    
            elif action == "switch":
                # Переключаемся на выбранную сессию
                session_id = data[2]
                self.db.switch_active_session(user_id, session_id)
                # Загружаем контекст выбранной сессии
                current_context, parse_mode, temperature, max_tokens_percent, _ = self.db.get_conversation_context(user_id, session_id)
                if current_context and 'messages' in current_context:
                    self.openai.conversations[user_id] = current_context['messages']
                await self.reset(update, context)  # Обновляем список сессий
                
            elif action == "delete":
                # Удаляем сессию
                session_id = data[2]
                self._schedule_hindsight_session_finalize(user_id, session_id)
                self.db.delete_session(user_id, session_id, openai_helper=self.openai)
                # Получаем контекст активной сессии
                session_id = self.db.get_active_session_id(user_id)
                current_context, _, _, _, _ = self.db.get_conversation_context(user_id, session_id, openai_helper=self.openai)
                if current_context and 'messages' in current_context:
                    self.openai.conversations[user_id] = current_context['messages']
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
                    filepath = self.db.export_sessions_to_yaml(user_id)
                    
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

            application = ApplicationBuilder() \
                .token(self.config['token']) \
                .post_init(self.post_init) \
                .post_shutdown(self._post_shutdown) \
                .concurrent_updates(True) \
                .local_mode(True) \
                .base_url('http://localhost:8081/bot') \
                .build()

            self.application = application
            self.openai.bot = application.bot
            self._background_tasks = []

            application.add_handler(CommandHandler('restart', self.restart))
            application.add_handler(CommandHandler('reset', self.reset))
            application.add_handler(CommandHandler('help', self.help))
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

            # Регистрируем обработчики сообщений от плагинов
            for handler_config in self.openai.plugin_manager.get_message_handlers():
                if 'handler' in handler_config and 'filters' not in handler_config:
                    # Если handler уже является готовым обработчиком (например, ConversationHandler)
                    handler = handler_config['handler']
                    # Проверяем, что это действительно валидный обработчик
                    try:
                        application.add_handler(handler)
                        logger.info(f"Successfully added handler: {type(handler).__name__}")
                    except TypeError as e:
                        logger.error(f"Invalid handler type {type(handler).__name__}: {e}")
                        continue
                elif 'filters' in handler_config:
                    # Если указаны фильтры, создаем MessageHandler
                    filter_obj = handler_config['filters']
                    if isinstance(filter_obj, str):
                        key = filter_obj.replace("filters.", "").strip()
                        filter_obj = getattr(filters, key, None)
                    if filter_obj is None:
                        logger.error(f"Invalid filter in plugin handler config: {handler_config.get('filters')}")
                        continue
                    handler = MessageHandler(
                        filter_obj,
                        lambda update, context, h=handler_config: self.handle_plugin_command(
                            update, context, {"handler": h['handler'], **h['handler_kwargs']}
                        )
                    )
                    application.add_handler(handler)

            application.add_handler(MessageHandler(
                filters.Document.TXT |
                filters.Document.DOC |
                filters.Document.DOCX |
                filters.Document.MimeType('application/pdf') |
                filters.Document.MimeType('application/rtf') |
                filters.Document.MimeType('text/markdown'),
                self.handle_document))

            application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
                constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
            ]))

            application.add_handler(
                MessageHandler(
                    filters.REPLY & filters.TEXT,
                    self.handle_plugin_menu_args_reply
                )
            )

            # Регистрируем глобальный обработчик текстовых сообщений после всех остальных обработчиков
            application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
                self.prompt
            ))

            application.add_error_handler(error_handler)

            application.run_polling()
        finally:
            # Завершаем выполнение задач в текущем событийном цикле
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(self.cleanup())
