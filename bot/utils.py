from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import io
import base64
import re
import time
from PIL import Image
import uuid

import telegram
import telegramify_markdown
from telegram import Message, MessageEntity, Update, ChatMember, constants
from telegram.ext import CallbackContext, ContextTypes
from .i18n import localized_text

from .usage_tracker import UsageTracker
from .html_utils import HTMLVisualizer
from .tool_result import direct_result_payload
from .telegram_rich import (
    MAX_RICH_MARKDOWN_BYTES,
    rich_markdown_fits,
    rich_messages_enabled,
    rich_messages_required,
    send_rich_markdown,
)

_GROUP_MEMBERSHIP_CACHE: dict[tuple[int, str], tuple[float, bool]] = {}
_GROUP_MEMBERSHIP_CACHE_TTL_SECONDS = 60.0


def log_value_shape(value, *, key=None, max_depth: int = 3):
    return value


def log_json_shape(value, *, max_depth: int = 3) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def log_exception_shape(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def message_text(message: Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    message_txt = message.text
    if message_txt is None:
        return ''

    for _, text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(),
                          key=(lambda item: item[0].offset)):
        message_txt = message_txt.replace(text, '').strip()

    return message_txt if len(message_txt) > 0 else ''

async def is_user_in_group(update: Update, context: CallbackContext, user_id: int) -> bool:
    """
    Checks if user_id is a member of the group
    """
    try:
        chat = update.effective_chat
        if chat is None and update.callback_query and update.callback_query.message:
            chat = getattr(update.callback_query.message, "chat", None)
        if chat is None and update.message:
            chat = getattr(update.message, "chat", None)
        if chat is None:
            return False
        now = time.monotonic()
        cache_key = (chat.id, str(user_id))
        cached = _GROUP_MEMBERSHIP_CACHE.get(cache_key)
        if cached is not None:
            expires_at, allowed = cached
            if expires_at > now:
                return allowed
        chat_member = await context.bot.get_chat_member(chat.id, user_id)
        allowed = chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
        _GROUP_MEMBERSHIP_CACHE[cache_key] = (
            now + _GROUP_MEMBERSHIP_CACHE_TTL_SECONDS,
            allowed,
        )
        return allowed
    except telegram.error.BadRequest as e:
        if str(e) == "User not found":
            if 'cache_key' in locals():
                _GROUP_MEMBERSHIP_CACHE[cache_key] = (
                    time.monotonic() + _GROUP_MEMBERSHIP_CACHE_TTL_SECONDS,
                    False,
                )
            return False
        else:
            raise e
    except Exception as e:
        raise e

def get_thread_id(update: Update) -> int | None:
    """
    Gets the message thread id for the update, if any
    """
    if update.effective_message and update.effective_message.is_topic_message:
        return update.effective_message.message_thread_id
    return None


def _read_file_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()

_PLAN_STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "cancelled": "⛔",
}


class BusyStatusMessage:
    """
    Maintains a temporary progress message for long-running chat responses.
    If a plan_provider is supplied, the message also lists current plan steps
    with their statuses so the user can see real-time agent progress.
    """

    def __init__(
        self,
        update: Update,
        context: CallbackContext,
        description: str,
        *,
        config: dict | None = None,
        interval: float = 30.0,
        plan_provider=None,
    ):
        self.update = update
        self.context = context
        self.description = description
        self.config = config
        self.interval = interval
        self.plan_provider = plan_provider
        self.message = None
        self._started_at = time.monotonic()
        self._task = None
        self._stopped = False
        self._last_text: str | None = None

    async def start(self):
        if self._task is not None:
            return self

        application = getattr(self.context, "application", None)
        create_task = getattr(application, "create_task", None)
        if create_task is not None:
            self._task = create_task(self._run(), update=self.update)
        else:
            self._task = asyncio.create_task(self._run())
        return self

    async def stop(self):
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self.message is not None:
            try:
                await self.message.delete()
            except Exception as e:
                logging.warning("Failed to delete busy status message error=%s", log_exception_shape(e))
            finally:
                self.message = None

    async def _run(self):
        try:
            while not self._stopped:
                if self.message is None:
                    if not await self._send():
                        return
                else:
                    await self._edit()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass

    async def _send(self) -> bool:
        if not self.update.effective_message:
            return False
        try:
            self.message = await self.update.effective_message.reply_text(
                message_thread_id=get_thread_id(self.update),
                reply_to_message_id=(
                    get_reply_to_message_id(self.config, self.update)
                    if self.config is not None else None
                ),
                text=self._text(),
            )
            return True
        except Exception as e:
            logging.warning("Failed to send busy status message error=%s", log_exception_shape(e))
            return False

    async def _edit(self):
        text = self._text()
        if text == self._last_text:
            return
        try:
            await self.message.edit_text(text=text)
            self._last_text = text
        except telegram.error.BadRequest as e:
            if str(e).startswith("Message is not modified"):
                self._last_text = text
                return
            logging.warning("Failed to edit busy status message error=%s", log_exception_shape(e))
        except Exception as e:
            logging.warning("Failed to edit busy status message error=%s", log_exception_shape(e))

    def _text(self) -> str:
        elapsed_seconds = max(0, int(time.monotonic() - self._started_at))
        minutes, seconds = divmod(elapsed_seconds, 60)
        bot_language = (self.config or {}).get("bot_language", "en")
        elapsed_label = localized_text("busy_status_elapsed", bot_language)
        header = f"{self.description}\n{elapsed_label}: {minutes:02d}:{seconds:02d}"
        plan_lines = self._plan_lines()
        if not plan_lines:
            return header
        return header + f"\n\n📋 {localized_text('busy_status_plan', bot_language)}:\n" + "\n".join(plan_lines)

    def _plan_lines(self) -> list[str]:
        provider = self.plan_provider
        if not callable(provider):
            return []
        try:
            tasks = provider() or []
        except Exception as exc:
            logging.debug("plan_provider raised error=%s", log_exception_shape(exc))
            return []
        lines: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status") or "pending")
            icon = _PLAN_STATUS_ICONS.get(status, "•")
            content = str(task.get("content") or "").strip()
            if not content:
                continue
            task_id = str(task.get("id") or "").strip()
            prefix = f"{icon} {task_id}. " if task_id else f"{icon} "
            lines.append(prefix + content)
        return lines

def get_stream_cutoff_values(update: Update, content: str) -> int:
    """
    Gets the stream cutoff values for the message length
    """
    if is_group_chat(update):
        # group chats have stricter flood limits
        return 180 if len(content) > 1000 else 120 if len(content) > 200 \
            else 90 if len(content) > 50 else 50
    return 90 if len(content) > 1000 else 45 if len(content) > 200 \
        else 25 if len(content) > 50 else 15

def is_group_chat(update: Update) -> bool:
    """
    Checks if the message was sent from a group chat
    """
    if not update.effective_chat:
        return False
    return update.effective_chat.type in [
        constants.ChatType.GROUP,
        constants.ChatType.SUPERGROUP
    ]

def _utf16_len(s: str) -> int:
    """Returns the number of UTF-16 code units in s (Telegram's character counting unit)."""
    return len(s.encode('utf-16-le')) // 2


def split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a given size while preserving Markdown formatting.

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (in UTF-16 code units)

    Returns:
        List of chunks with preserved Markdown formatting
    """
    if _utf16_len(text) <= chunk_size:
        return [text]

    # Предварительная обработка очень длинных строк
    max_line_utf16 = 3800  # Немного меньше чем chunk_size для обеспечения безопасности
    processed_lines = []

    for line in text.split('\n'):
        if _utf16_len(line) > max_line_utf16:
            # Разбиваем длинную строку на части накоплением по UTF-16
            part = ""
            part_len = 0
            for ch in line:
                ch_len = _utf16_len(ch)
                if part_len + ch_len > max_line_utf16:
                    processed_lines.append(part)
                    part = ch
                    part_len = ch_len
                else:
                    part += ch
                    part_len += ch_len
            if part:
                processed_lines.append(part)
        else:
            processed_lines.append(line)

    chunks = []
    current_chunk = ""
    current_len = 0
    markdown_stack = []  # Стек для отслеживания открытых Markdown-элементов

    def close_markdown_markers(chunk: str) -> str:
        for md in reversed(markdown_stack):
            if md == '```':
                if chunk and not chunk.endswith('\n'):
                    chunk += '\n'
                chunk += md
            elif not chunk.endswith(md):
                chunk += md
        return chunk

    def opening_markdown_prefix() -> str:
        prefix = ""
        for md in markdown_stack:
            if md == '```':
                if prefix and not prefix.endswith('\n'):
                    prefix += '\n'
                prefix += md
            else:
                prefix += md
        return prefix

    def update_markdown_stack(line: str) -> None:
        stripped = line.lstrip()
        if stripped.startswith('```'):
            if '```' in markdown_stack:
                for index in range(len(markdown_stack) - 1, -1, -1):
                    if markdown_stack[index] == '```':
                        markdown_stack.pop(index)
                        break
            else:
                markdown_stack.append('```')
            return

        if '```' in markdown_stack:
            return

        index = 0
        while index < len(line):
            char = line[index]
            if char not in ['*', '_', '`']:
                index += 1
                continue

            if char == '`':
                run_end = index
                while run_end < len(line) and line[run_end] == '`':
                    run_end += 1
                run_length = run_end - index
                if run_length >= 3:
                    index = run_end
                    continue
                marker = char * run_length
                index = run_end
            elif index + 1 < len(line) and line[index + 1] == char:
                marker = char * 2
                index += 2
            else:
                marker = char
                index += 1

            if markdown_stack and markdown_stack[-1] == marker:
                markdown_stack.pop()
            else:
                markdown_stack.append(marker)

    # Используем предварительно обработанные строки
    for line in processed_lines:
        line_len = _utf16_len(line)
        separator_len = 1 if current_chunk else 0  # '\n'
        closing_overhead = (
            _utf16_len(close_markdown_markers(current_chunk)) - _utf16_len(current_chunk)
        )

        # Если текущая строка с переносом превысит размер чанка
        if current_len + separator_len + line_len + closing_overhead > chunk_size:
            # Закрываем все открытые Markdown-элементы
            current_chunk = close_markdown_markers(current_chunk)

            chunks.append(current_chunk.strip())

            # Открываем Markdown-элементы для нового чанка
            current_chunk = opening_markdown_prefix()
            current_len = _utf16_len(current_chunk)

        # Добавляем строку к текущему чанку
        if current_chunk:
            current_chunk += '\n'
            current_len += 1
        current_chunk += line
        current_len += line_len

        # Отслеживаем Markdown-элементы в строке
        update_markdown_stack(line)

    # Добавляем последний чанк
    if current_chunk:
        # Закрываем все открытые Markdown-элементы
        current_chunk = close_markdown_markers(current_chunk)
        chunks.append(current_chunk.strip())

    return chunks


def render_markdown_message_entities(markdown_text: str, max_utf16_len: int = 4096) -> list[tuple[str, list[MessageEntity]]]:
    """
    Converts Markdown to Telegram text/entities chunks without parse_mode.
    """
    text, raw_entities = telegramify_markdown.convert(markdown_text)
    chunks = telegramify_markdown.split_entities(text, raw_entities, max_utf16_len)
    return [
        (chunk_text, [MessageEntity(**entity.to_dict()) for entity in chunk_entities])
        for chunk_text, chunk_entities in chunks
    ]


def looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines()]
    for index in range(len(lines) - 1):
        header = lines[index]
        separator = lines[index + 1]
        if "|" not in header or "|" not in separator:
            continue
        cells = [cell.strip() for cell in separator.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            return True
    return False


def should_send_text_as_file(text: str, chunks: list[str] | None = None, *, force_html_file: bool = False) -> bool:
    chunks = chunks if chunks is not None else split_into_chunks(text)
    return (
        len(chunks) > 3
        or (len(chunks) > 1 and '```' in str(text or ""))
        or (force_html_file and len(chunks) > 1)
        or (looks_like_markdown_table(text) and (len(chunks) > 1 or len(str(text or "")) > 1500))
    )


def _message_bot(message):
    get_bot = getattr(message, "get_bot", None)
    if callable(get_bot):
        try:
            return get_bot()
        except Exception as exc:
            logging.debug("Could not resolve message bot for rich delivery error=%s", log_exception_shape(exc))
    return getattr(message, "bot", None)


async def try_send_rich_markdown_response(
    config: dict | None,
    *,
    bot=None,
    message=None,
    chat_id: int | str | None = None,
    text: str,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    fallback_label: str = "telegram response",
):
    if not rich_messages_enabled(config):
        return []

    text = str(text or "")
    required = rich_messages_required(config)
    if not rich_markdown_fits(text):
        error = ValueError(
            "Telegram rich markdown exceeds "
            f"{MAX_RICH_MARKDOWN_BYTES} bytes"
        )
        if required:
            raise error
        logging.warning(
            "Telegram rich delivery skipped; falling back to legacy delivery "
            "label=%s text_bytes=%s limit=%s",
            fallback_label,
            len(text.encode("utf-8")),
            MAX_RICH_MARKDOWN_BYTES,
        )
        return []

    if bot is None and message is not None:
        bot = _message_bot(message)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)
    if bot is None or chat_id is None:
        error = RuntimeError("Telegram rich delivery requires bot and chat_id")
        if required:
            raise error
        logging.warning(
            "Telegram rich delivery unavailable; falling back to legacy delivery "
            "label=%s bot_available=%s chat_id_available=%s",
            fallback_label,
            bot is not None,
            chat_id is not None,
        )
        return []

    try:
        sent_message = await send_rich_markdown(
            bot,
            chat_id=chat_id,
            markdown=text,
            message_thread_id=message_thread_id,
            reply_to_message_id=reply_to_message_id,
        )
        return [sent_message]
    except Exception as exc:
        if required:
            raise
        logging.warning(
            "Telegram rich delivery failed; falling back to legacy delivery "
            "label=%s error=%s text_chars=%s",
            fallback_label,
            log_exception_shape(exc),
            len(text),
        )
        return []


async def wrap_with_indicator(update: Update, context: CallbackContext, coroutine,
                            chat_action: constants.ChatAction = "", is_inline=False):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    task = context.application.create_task(coroutine(), update=update)
    try:
        # Keep long-running model/tool requests alive while still bounding stuck tasks.
        async with asyncio.timeout(4000):
            while not task.done():
                if not is_inline:
                    try:
                        await update.effective_chat.send_action(
                            chat_action, 
                            message_thread_id=get_thread_id(update)
                        )
                    except Exception as e:
                        logging.warning("Error sending chat action error=%s", log_exception_shape(e))
                try:
                    await asyncio.wait_for(asyncio.shield(task), 4.5)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    logging.error("Error in wrap_with_indicator error=%s", log_exception_shape(e))
                    break
            
            return await task
    except asyncio.TimeoutError:
        task.cancel()
        raise telegram.error.TimedOut("Operation timed out")
    except Exception as e:
        task.cancel()
        raise e
    
async def edit_message_with_retry(context: ContextTypes.DEFAULT_TYPE, chat_id: int | None,
                                  message_id: str, text: str, markdown: bool = True, is_inline: bool = False):
    """
    Edit a message with retry logic in case of failure (e.g. broken markdown)
    :param context: The context to use
    :param chat_id: The chat id to edit the message in
    :param message_id: The message id to edit
    :param text: The text to edit the message with
    :param markdown: Whether to render Markdown formatting
    :param is_inline: Whether the message to edit is an inline message
    :return: None
    """
    try:
        entities = None
        text_to_send = text
        if markdown:
            safe_text = text
            if _utf16_len(text) > 4096:
                logging.warning(
                    'edit_message_with_retry: text exceeds 4096 UTF-16 units (%d), truncating',
                    _utf16_len(text),
                )
                # Truncate by accumulating characters until we reach the limit
                buf = ""
                buf_len = 0
                for ch in text:
                    ch_len = _utf16_len(ch)
                    if buf_len + ch_len > 4096:
                        break
                    buf += ch
                    buf_len += ch_len
                safe_text = buf
            parts = render_markdown_message_entities(safe_text)
            if parts:
                text_to_send, entities = parts[0]

        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id) if not is_inline else None,
            inline_message_id=message_id if is_inline else None,
            text=text_to_send,
            parse_mode=None,
            entities=entities,
        )
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            return
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id) if not is_inline else None,
                inline_message_id=message_id if is_inline else None,
                text=text,
            )
        except Exception as e:
            logging.warning("Failed to edit message error=%s", log_exception_shape(e))
            raise e

    except Exception as e:
        logging.warning("Failed to edit message error=%s", log_exception_shape(e))
        raise e

async def error_handler(_: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles errors in the telegram-python-bot library.
    """
    error = context.error
    error_shape = (
        log_exception_shape(error)
        if isinstance(error, BaseException)
        else log_value_shape(error, key="value")
    )
    logging.error("Exception while handling an update error=%s", error_shape)

async def is_allowed(config, update: Update, context: CallbackContext, is_inline=False) -> bool:
    """
    Checks if the user is allowed to use the bot.
    """
    if config['allowed_user_ids'] == '*':
        return True

    if is_inline and update.inline_query:
        user = update.inline_query.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    elif update.message:
        user = update.message.from_user
    else:
        user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return False
    if is_admin(config, user_id):
        return True
    name = user.name if user else "unknown"
    allowed_user_ids = [x.strip() for x in config['allowed_user_ids'].split(',') if x.strip()]
    # Check if user is allowed
    if str(user_id) in allowed_user_ids:
        return True
    # Check if it's a group a chat with at least one authorized member
    if not is_inline and is_group_chat(update):
        admin_user_ids = [x.strip() for x in config['admin_user_ids'].split(',') if x.strip()]
        for user in itertools.chain(allowed_user_ids, admin_user_ids):
            if await is_user_in_group(update, context, user):
                logging.info(f'{user} is a member. Allowing group chat message...')
                return True
        logging.info(f'Group chat messages from user {name} '
                     f'(id: {user_id}) are not allowed')
    return False

def is_admin(config, user_id: int, log_no_admin=False) -> bool:
    """
    Checks if the user is the admin of the bot.
    The first user in the user list is the admin.
    """
    if config['admin_user_ids'] == '-':
        if log_no_admin:
            logging.info('No admin user defined.')
        return False

    admin_user_ids = [x.strip() for x in config['admin_user_ids'].split(',') if x.strip()]

    # Check if user is in the admin user list
    if str(user_id) in admin_user_ids:
        return True

    return False

def get_user_budget(config, user_id) -> float | None:
    """
    Get the user's budget based on their user ID and the bot configuration.
    :param config: The bot configuration object
    :param user_id: User id
    :return: The user's budget as a float, or None if the user is not found in the allowed user list
    """

    # no budget restrictions for admins and '*'-budget lists
    if is_admin(config, user_id) or config['user_budgets'] == '*':
        return float('inf')

    user_budgets = [x.strip() for x in config['user_budgets'].split(',') if x.strip()]
    if config['allowed_user_ids'] == '*':
        # same budget for all users, use value in first position of budget list
        if len(user_budgets) > 1:
            logging.warning('multiple values for budgets set with unrestricted user list '
                            'only the first value is used as budget for everyone.')
        return float(user_budgets[0]) if user_budgets else 0.0

    allowed_user_ids = [x.strip() for x in config['allowed_user_ids'].split(',') if x.strip()]
    if str(user_id) in allowed_user_ids:
        user_index = allowed_user_ids.index(str(user_id))
        if len(user_budgets) <= user_index:
            logging.warning(f'No budget set for user id: {user_id}. Budget list shorter than user list.')
            return 0.0
        return float(user_budgets[user_index])
    return None

def get_remaining_budget(config, usage, update: Update, is_inline=False) -> float:
    """
    Calculate the remaining budget for a user based on their current usage.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: The remaining budget for the user as a float
    """
    # Mapping of budget period to cost period
    budget_cost_map = {
        "monthly": "cost_month",
        "weekly": "cost_week",
        "daily": "cost_today",
        "all-time": "cost_all_time",
        "total": "cost_all_time",
    }

    if is_inline and update.inline_query:
        user = update.inline_query.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    elif update.message:
        user = update.message.from_user
    else:
        user = update.effective_user
    user_id = user.id if user else None
    name = user.name if user else "unknown"
    if user_id is None:
        return 0.0
    if user_id not in usage:
        usage[user_id] = make_usage_tracker(config, user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map.get(budget_period, "cost_month")]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = make_usage_tracker(config, 'guests', 'all guest users in group chats')
    cost = usage['guests'].get_current_cost()[budget_cost_map.get(budget_period, "cost_month")]
    return config['guest_budget'] - cost

def is_within_budget(config, usage, update: Update, is_inline=False) -> bool:
    """
    Checks if the user reached their usage limit.
    Initializes UsageTracker for user and guest when needed.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: Boolean indicating if the user has a positive budget
    """
    # Инициализация UsageTracker и расчет остатка бюджета выполняются внутри get_remaining_budget
    remaining_budget = get_remaining_budget(config, usage, update, is_inline=is_inline)
    return remaining_budget > 0

def make_usage_tracker(config, user_id, user_name, logs_dir="usage_logs"):
    """
    Construct a UsageTracker pre-loaded with the prices from `config`, so that
    subsequent add_* calls do not need to re-pass them.
    """
    return UsageTracker(
        user_id, user_name, logs_dir,
        token_price=config.get('token_price'),
        image_prices=config.get('image_prices'),
        vision_token_price=config.get('vision_token_price'),
        tts_prices=config.get('tts_prices'),
        transcription_price=config.get('transcription_price'),
        history_days=config.get('usage_retention_days'),
    )

def _charge_user_and_guest(usage, config, user_id, charge_fn):
    if user_id not in usage:
        logging.warning(f'No UsageTracker for user_id={user_id}; skipping charge.')
        return False
    try:
        charge_fn(usage[user_id])
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            charge_fn(usage['guests'])
        return True
    except Exception as e:
        logging.warning("Failed to record usage error=%s", log_exception_shape(e))
        return False

def _positive_int_usage(value, label):
    if isinstance(value, bool):
        logging.warning(f'Invalid {label}; not adding request to usage tracker.')
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        logging.warning(f'Invalid {label}; not adding request to usage tracker.')
        return None
    if value <= 0:
        logging.warning(f'No {label} used. Not adding request to usage tracker.')
        return None
    return value

def record_chat_tokens(usage, config, user_id, used_tokens):
    used_tokens = _positive_int_usage(used_tokens, 'chat tokens')
    if used_tokens is None:
        return False
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_chat_tokens(used_tokens),
    )

def record_image_request(usage, config, user_id, image_size):
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_image_request(image_size),
    )

def record_vision_tokens(usage, config, user_id, used_tokens):
    used_tokens = _positive_int_usage(used_tokens, 'vision tokens')
    if used_tokens is None:
        return False
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_vision_tokens(used_tokens),
    )

def record_tts_request(usage, config, user_id, text_length, tts_model):
    text_length = _positive_int_usage(text_length, 'TTS characters')
    if text_length is None:
        return False
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_tts_request(text_length, tts_model),
    )

def record_transcription_seconds(usage, config, user_id, seconds):
    seconds = _positive_int_usage(seconds, 'transcription seconds')
    if seconds is None:
        return False
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_transcription_seconds(seconds),
    )

def get_reply_to_message_id(config, update: Update):
    """
    Returns the message id of the message to reply to
    :param config: Bot configuration object
    :param update: Telegram update object
    :return: Message id of the message to reply to, or None if quoting is disabled
    """
    if config['enable_quoting'] or is_group_chat(update):
        message = update.message or (update.callback_query.message if update.callback_query else None) or update.effective_message
        if message:
            return message.message_id
    return None

def compute_scope_key(chat_id=None, user_id=None) -> str:
    """
    Build the canonical plugin-state scope key used by skills, agent_tools, and
    routing helpers. Prefers chat scope, falls back to user scope, then global.
    Both ids are coerced to int when possible so that "42" and 42 yield the same key.
    """
    def _to_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    chat_id = _to_int(chat_id)
    if chat_id is not None:
        return f"chat:{chat_id}"
    user_id = _to_int(user_id)
    if user_id is not None:
        return f"user:{user_id}"
    return "global"


def is_direct_result(response: any) -> bool:
    """
    Checks if the dict contains a structurally valid direct_result payload that can be
    sent directly to the user. Requires a dict with a non-empty `kind` field.
    """
    return direct_result_payload(response) is not None


def direct_result_inline_fallback_text(response: any, unavailable_message: str, *, max_chars: int = 3500) -> str:
    if type(response) is not dict:
        try:
            response = json.loads(response)
        except Exception:
            return str(response)[:max_chars]
    result = response.get('direct_result') if isinstance(response, dict) else None
    if not isinstance(result, dict):
        return str(response)[:max_chars]

    def _clip(value: str) -> str:
        value = str(value or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars - 20].rstrip() + "\n... [truncated]"

    def _artifact_line(item: dict) -> str | None:
        kind = str(item.get("kind") or "artifact")
        value = str(
            item.get("value")
            or item.get("file_path")
            or item.get("path")
            or item.get("output_path")
            or item.get("artifact_path")
            or item.get("url")
            or ""
        ).strip()
        if not value:
            return None
        if item.get("format") == "path" or os.path.isabs(value):
            value = os.path.basename(value)
        return f"- {kind}: {value}"

    kind = result.get("kind")
    if kind == "final":
        text = str(result.get("text") or "").strip()
        artifact_lines = [
            line for line in (_artifact_line(item) for item in result.get("artifacts") or [])
            if line
        ]
        parts = []
        if text:
            parts.append(text)
        if artifact_lines:
            parts.append("Artifacts produced, but inline mode cannot attach files:\n" + "\n".join(artifact_lines))
        return _clip("\n\n".join(parts) or unavailable_message)

    if kind == "text":
        return _clip(result.get("add_value") or result.get("value") or unavailable_message)

    artifact_line = _artifact_line(result)
    if artifact_line:
        return _clip(f"{artifact_line}\n\n{unavailable_message}")
    return _clip(unavailable_message)

def escape_markdown(text: str, exclude_code_blocks: bool = True) -> str:
    """
    Экранирует специальные символы Markdown.
    :param text: Исходный текст
    :param exclude_code_blocks: Исключать ли блоки кода из экранирования
    :return: Экранированный текст
    """
    _non_code_escape = frozenset(['_', '*', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'])

    if not exclude_code_blocks:
        escape_chars = _non_code_escape | {'`'}
        return ''.join('\\' + char if char in escape_chars else char for char in text)

    # Segment-based approach: split on backtick-delimited paired runs.
    # Find paired backtick spans; unpaired remainder is treated as plain text.
    segments = []  # list of (text, is_code)
    remaining = text
    while remaining:
        tick_pos = remaining.find('`')
        if tick_pos == -1:
            # No more backticks — rest is plain text
            segments.append((remaining, False))
            break
        # Text before the backtick
        if tick_pos > 0:
            segments.append((remaining[:tick_pos], False))
        remaining = remaining[tick_pos:]
        # Find the closing backtick
        close_pos = remaining.find('`', 1)
        if close_pos == -1:
            # Unpaired backtick — treat the rest as plain text
            segments.append((remaining, False))
            break
        # Code span including both backticks
        segments.append((remaining[:close_pos + 1], True))
        remaining = remaining[close_pos + 1:]

    result = []
    for seg_text, is_code in segments:
        if is_code:
            result.append(seg_text)
        else:
            result.append(''.join('\\' + c if c in _non_code_escape else c for c in seg_text))
    return ''.join(result)

def get_image_size(image_path: str) -> tuple[int, int]:
    """
    Получает размеры изображения
    """
    with Image.open(image_path) as img:
        return img.size

def resize_image_if_needed(image_path: str, max_dimension: int = 10000) -> tuple[io.BytesIO, str]:
    """
    Проверяет размеры изображения и изменяет их при необходимости.
    
    Args:
        image_path: Путь к изображению
        max_dimension: Максимальный размер стороны изображения
        
    Returns:
        Tuple[BytesIO, str]: (объект BytesIO с изображением, формат изображения)
    """
    with Image.open(image_path) as img:
        # Получаем формат изображения
        format = img.format.lower()
        
        # Проверяем размеры
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            # Вычисляем новые размеры с сохранением пропорций
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Изменяем размер
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Сохраняем в BytesIO
        output = io.BytesIO()
        img.save(output, format=format)
        output.seek(0)
        
        return output, format

async def handle_direct_result(config, update: Update, response: any, *, bot=None):
    """
    Handles a direct result from a plugin
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response.get('direct_result') if isinstance(response, dict) else None
    if not isinstance(result, dict):
        logging.warning(
            "handle_direct_result called without a direct_result payload response=%s",
            log_json_shape(response),
        )
        return []
    kind = result.get('kind')
    if not kind:
        logging.warning(
            "handle_direct_result payload missing 'kind' payload=%s",
            log_json_shape(result),
        )
        return []
    result_format = result.get('format')
    value = (
        result.get('value')
        or result.get('file_path')
        or result.get('path')
        or result.get('output_path')
        or result.get('artifact_path')
        or result.get('url')
    )
    add_value = result.get('add_value', None)
    logging.info(
        "Handling direct result - kind=%s format=%s value_shape=%s add_value_shape=%s",
        kind,
        result_format,
        log_value_shape(value, key="value"),
        log_value_shape(add_value, key="add_value"),
    )

    message = update.effective_message or (update.callback_query.message if update.callback_query else None)
    if not message:
        logging.error("No message available to send direct result")
        return

    common_args = {
        'message_thread_id': get_thread_id(update),
        'reply_to_message_id': get_reply_to_message_id(config, update),
    }
    sent_messages = []

    if kind == 'final':
        for artifact in result.get("artifacts") or []:
            if not isinstance(artifact, dict):
                raise ValueError(
                    "final artifact must be an object "
                    f"artifact_shape={log_value_shape(artifact, key='value')}"
                )
            artifact_payload = dict(artifact)
            artifact_payload["preserve_after_delivery"] = True
            artifact_messages = await handle_direct_result(config, update, {"direct_result": artifact_payload}, bot=bot)
            if not artifact_messages:
                raise RuntimeError(
                    "final artifact was not delivered "
                    f"artifact_shape={log_json_shape({'artifact': artifact_payload})}"
                )
            sent_messages.extend(artifact_messages)

        text = str(result.get('text') or "").strip()
        if text:
            text_result = {
                "direct_result": {
                    "kind": "text",
                    "format": result.get("text_format", "markdown"),
                    "value": text,
                    "force_html_file": True,
                }
            }
            text_messages = await handle_direct_result(config, update, text_result, bot=bot)
            if text_messages:
                sent_messages.extend(text_messages)
        return sent_messages

    caption = result.get('caption')
    caption_kwargs = {'caption': str(caption)} if caption else {}

    if kind == 'photo':
        if result_format == 'url':
            sent_messages.append(await message.reply_photo(**common_args, **caption_kwargs, photo=value))
        elif result_format == 'path':
            try:
                if get_image_size(value)[0] > 10000 or get_image_size(value)[1] > 10000:
                    # Пробуем отправить как документ
                    with open(value, 'rb') as fh:
                        sent_messages.append(await message.reply_document(**common_args, **caption_kwargs, document=fh))
                else:
                    # Пробуем отправить как фото
                    sent_messages.append(await message.reply_photo(**common_args, **caption_kwargs, photo=value))
            except Exception as e:
                logging.error(
                    "Error handling photo direct result error=%s value_shape=%s",
                    log_exception_shape(e),
                    log_value_shape(value, key="value"),
                )
                # Проверяем и изменяем размеры изображения при необходимости
                photo_file, photo_format = resize_image_if_needed(value)
                sent_messages.append(await message.reply_photo(**common_args, **caption_kwargs, photo=photo_file))
    elif kind == 'gif':
        if result_format == 'url':
            sent_messages.append(await message.reply_animation(**common_args, **caption_kwargs, animation=value))
        elif result_format == 'path':
            with open(value, 'rb') as fh:
                sent_messages.append(await message.reply_animation(**common_args, **caption_kwargs, animation=fh))
    elif kind == 'file':
        if result_format == 'url':
            sent_messages.append(await message.reply_document(**common_args, **caption_kwargs, document=value))
        elif result_format == 'path':
            with open(value, 'rb') as fh:
                sent_messages.append(await message.reply_document(**common_args, **caption_kwargs, document=fh))
    elif kind == 'dice':
        sent_messages.append(await message.reply_dice(**common_args, emoji=value))
    elif kind == 'reaction':
        target_message = getattr(message, 'reply_to_message', None)
        set_reaction = getattr(target_message, 'set_reaction', None) if target_message else None
        if set_reaction:
            try:
                if await set_reaction(reaction=value):
                    return sent_messages
            except Exception as e:
                logging.warning(
                    "Could not set reaction direct result error=%s",
                    log_exception_shape(e),
                )
        sent_messages.append(await message.reply_text(
            message_thread_id=get_thread_id(update),
            reply_to_message_id=get_reply_to_message_id(config, update),
            text=localized_text(
                "direct_result_reaction",
                config.get("bot_language", "en")
            ).format(value=value),
            parse_mode=None
        ))
        return sent_messages

    if add_value or kind == 'text':
        # Split long messages into chunks
        text = str((add_value if add_value else value) or "")
        chunks = split_into_chunks(text)
        rich_messages = []
        if result_format == 'markdown':
            rich_messages = await try_send_rich_markdown_response(
                config,
                bot=bot,
                message=message,
                chat_id=getattr(getattr(update, "effective_chat", None), "id", None),
                text=text,
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(config, update),
                fallback_label="direct_result markdown",
            )
            message_parts = [] if rich_messages else render_markdown_message_entities(text)
            sent_messages.extend(rich_messages)
        else:
            message_parts = [(chunk, None) for chunk in chunks]

        # Отправляем как файл если: 
        # - ответ больше 3х частей ИЛИ 
        # - (ответ больше одной части И содержит вставки кода)
        if rich_messages:
            pass
        elif should_send_text_as_file(
            text,
            chunks,
            force_html_file=bool(result.get("force_html_file")),
        ):
            # Получаем имя текущей сессии
            session_name = text[:10]
            
            sent_message = await send_long_response_as_file(config, update, text, session_name)
            if sent_message:
                sent_messages.append(sent_message)
        else:
            for i, (chunk, entities) in enumerate(message_parts):
                # Only reply to original message for first chunk
                reply_to = get_reply_to_message_id(config, update) if i == 0 else None
                try:
                    sent_messages.append(await message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=reply_to,
                        text=chunk,
                        parse_mode=None,
                        entities=entities,
                    ))
                except telegram.error.BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        logging.warning(
                            "Telegram entities error in handle_direct_result; retrying without formatting "
                            "error=%s chunk_shape=%s",
                            log_exception_shape(e),
                            log_value_shape(chunk, key="value"),
                        )
                        sent_messages.append(await message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=reply_to,
                            text=chunk,
                            parse_mode=None
                        ))
                    else:
                        # Для других BadRequest ошибок просто убираем форматирование
                        sent_messages.append(await message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=reply_to,
                            text=chunk,
                            parse_mode=None
                        ))
                except Exception as e:
                    logging.error(
                        "Unexpected error in handle_direct_result error=%s chunk_shape=%s",
                        log_exception_shape(e),
                        log_value_shape(chunk, key="value"),
                    )
                    # В случае любой другой ошибки отправляем без форматирования
                    sent_messages.append(await message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=reply_to,
                        text=chunk,
                        parse_mode=None
                    ))

    if result_format == 'path':
        cleanup_intermediate_files(response)
    return sent_messages

def cleanup_intermediate_files(response: any):
    """
    Deletes intermediate files created by plugins
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response.get('direct_result') if isinstance(response, dict) else None
    if not isinstance(result, dict):
        return
    if result.get("kind") == "final":
        for artifact in result.get("artifacts") or []:
            if isinstance(artifact, dict):
                cleanup_intermediate_files({"direct_result": artifact})
        return
    format = result.get('format')
    value = (
        result.get('value')
        or result.get('file_path')
        or result.get('path')
        or result.get('output_path')
        or result.get('artifact_path')
    )

    if format == 'path' and value and not result.get("preserve_after_delivery"):
        if os.path.exists(value):
            os.remove(value)

# Function to encode the image
def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'

def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,'):]
    return base64.b64decode(image)

async def send_long_response_as_file(config, update: Update, response: str, session_name: str = 'response'):
    """
    Отправляет длинный ответ в виде HTML-файла с сохранением форматирования
    
    :param config: Конфигурация бота
    :param update: Объект обновления Telegram
    :param response: Текст ответа для отправки
    :param session_name: Базовое имя файла (по умолчанию 'response')
    """
    # Генерируем уникальный идентификатор сессии
    session_id = str(uuid.uuid4())[:8]
    
    # Используем HTMLVisualizer для создания HTML-файла
    visualizer = HTMLVisualizer(
        output_dir=config.get("output_dir") or None,
        data_dir=config.get("data_dir") or None,
        plots_dir=config.get("plots_dir") or None,
    )
    output_path = await asyncio.to_thread(visualizer.advanced_visualization, response, session_id)
    if not output_path:
        logging.error("HTML visualization did not produce an output path")
        return None
    
    # Получаем содержимое созданного файла
    try:
        file_content = await asyncio.to_thread(_read_file_bytes, output_path)
    except OSError as e:
        logging.error(
            "Failed to read generated HTML file output_shape=%s error=%s",
            log_value_shape(output_path, key="path"),
            log_exception_shape(e),
        )
        return None
    
    # Создаем файл с ответом для отправки
    response_file = io.BytesIO(file_content)
    
    # Формируем имя файла
    import re
    from datetime import datetime
    
    safe_session_name = re.sub(r'[^\w\-_\.]', '_', session_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{safe_session_name}_{timestamp}.html"
    
    # Отправляем файл пользователю
    sent_message = await update.effective_message.reply_document(
        message_thread_id=get_thread_id(update),
        reply_to_message_id=get_reply_to_message_id(config, update),
        document=response_file,
        filename=filename,
        caption=localized_text("full_response_caption", config.get("bot_language", "en")),
        parse_mode=constants.ParseMode.HTML
    )
    
    # Удаляем созданный файл после отправки
    try:
        os.remove(output_path)
    except Exception as e:
        logging.warning(
            "Failed to delete generated HTML file output_shape=%s error=%s",
            log_value_shape(output_path, key="path"),
            log_exception_shape(e),
        )

    return sent_message
