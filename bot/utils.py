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
from telegram import Message, MessageEntity, Update, ChatMember, constants
from telegram.ext import CallbackContext, ContextTypes
from .i18n import localized_text

from .usage_tracker import UsageTracker
from .html_utils import HTMLVisualizer

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
        chat_member = await context.bot.get_chat_member(chat.id, user_id)
        return chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
    except telegram.error.BadRequest as e:
        if str(e) == "User not found":
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
                logging.warning(f"Failed to delete busy status message: {e}")
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
            logging.warning(f"Failed to send busy status message: {e}")
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
            logging.warning(f"Failed to edit busy status message: {e}")
        except Exception as e:
            logging.warning(f"Failed to edit busy status message: {e}")

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
        except Exception:
            logging.debug("plan_provider raised", exc_info=True)
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

def split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a given size while preserving Markdown formatting.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
    
    Returns:
        List of chunks with preserved Markdown formatting
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Предварительная обработка очень длинных строк
    max_line_length = 3800  # Немного меньше чем chunk_size для обеспечения безопасности
    processed_lines = []
    
    for line in text.split('\n'):
        if len(line) > max_line_length:
            # Разбиваем длинную строку на части с переносами
            parts = [line[i:i+max_line_length] for i in range(0, len(line), max_line_length)]
            processed_lines.extend(parts)
        else:
            processed_lines.append(line)
    
    chunks = []
    current_chunk = ""
    markdown_stack = []  # Стек для отслеживания открытых Markdown-элементов
    
    # Используем предварительно обработанные строки
    for line in processed_lines:
        # Если текущая строка с переносом превысит размер чанка
        if len(current_chunk) + len(line) + 1 > chunk_size:
            # Закрываем все открытые Markdown-элементы
            for md in reversed(markdown_stack):
                if not current_chunk.endswith(md):
                    current_chunk += md
            
            chunks.append(current_chunk.strip())
            current_chunk = ""
            
            # Открываем Markdown-элементы для нового чанка
            for md in markdown_stack:
                current_chunk += md
        
        # Добавляем строку к текущему чанку
        if current_chunk:
            current_chunk += '\n'
        current_chunk += line
        
        # Отслеживаем Markdown-элементы в строке
        for i, char in enumerate(line):
            if char in ['*', '_', '`']:
                # Проверяем, является ли это открывающим или закрывающим элементом
                if markdown_stack and markdown_stack[-1][-1] == char:
                    markdown_stack.pop()
                else:
                    # Определяем тип элемента (одиночный или двойной)
                    if i + 1 < len(line) and line[i + 1] == char:
                        markdown_stack.append(char * 2)
                    else:
                        markdown_stack.append(char)
    
    # Добавляем последний чанк
    if current_chunk:
        # Закрываем все открытые Markdown-элементы
        for md in reversed(markdown_stack):
            if not current_chunk.endswith(md):
                current_chunk += md
        chunks.append(current_chunk.strip())
    
    return chunks


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

async def wrap_with_indicator(update: Update, context: CallbackContext, coroutine,
                            chat_action: constants.ChatAction = "", is_inline=False):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    task = context.application.create_task(coroutine(), update=update)
    try:
        # Keep long-running model/tool requests alive while still bounding stuck tasks.
        async with asyncio.timeout(2000):
            while not task.done():
                if not is_inline:
                    try:
                        await update.effective_chat.send_action(
                            chat_action, 
                            message_thread_id=get_thread_id(update)
                        )
                    except Exception as e:
                        logging.warning(f"Error sending chat action: {e}")
                try:
                    await asyncio.wait_for(asyncio.shield(task), 4.5)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    logging.error(f"Error in wrap_with_indicator: {e}")
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
    :param markdown: Whether to use markdown parse mode
    :param is_inline: Whether the message to edit is an inline message
    :return: None
    """
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id) if not is_inline else None,
            inline_message_id=message_id if is_inline else None,
            text=text,
            parse_mode=constants.ParseMode.MARKDOWN if markdown else None,
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
            logging.warning(f'Failed to edit message: {str(e)}')
            raise e

    except Exception as e:
        logging.warning(str(e))
        raise e

async def error_handler(_: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles errors in the telegram-python-bot library.
    """
    logging.error(f'Exception while handling an update: {context.error}')

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
    allowed_user_ids = config['allowed_user_ids'].split(',')
    # Check if user is allowed
    if str(user_id) in allowed_user_ids:
        return True
    # Check if it's a group a chat with at least one authorized member
    if not is_inline and is_group_chat(update):
        admin_user_ids = config['admin_user_ids'].split(',')
        for user in itertools.chain(allowed_user_ids, admin_user_ids):
            if not user.strip():
                continue
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

    admin_user_ids = config['admin_user_ids'].split(',')

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

    user_budgets = config['user_budgets'].split(',')
    if config['allowed_user_ids'] == '*':
        # same budget for all users, use value in first position of budget list
        if len(user_budgets) > 1:
            logging.warning('multiple values for budgets set with unrestricted user list '
                            'only the first value is used as budget for everyone.')
        return float(user_budgets[0])

    allowed_user_ids = config['allowed_user_ids'].split(',')
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
        usage[user_id] = UsageTracker(user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map.get(budget_period, "cost_month")]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
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

def add_chat_request_to_usage_tracker(usage, config, user_id, used_tokens):
    """
    Add chat request to usage tracker
    :param usage: The usage tracker object
    :param config: The bot configuration object
    :param user_id: The user id
    :param used_tokens: The number of tokens used
    """
    try:
        if int(used_tokens) == 0:
            logging.warning('No tokens used. Not adding chat request to usage tracker.')
            return False
        # add chat request to users usage tracker
        usage[user_id].add_chat_tokens(used_tokens, config['token_price'])
        # add guest chat request to guest usage tracker
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            usage["guests"].add_chat_tokens(used_tokens, config['token_price'])
        return True
    except Exception as e:
        logging.warning(f'Failed to add tokens to usage_logs: {str(e)}')
        return False

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
    if type(response) is not dict:
        try:
            response = json.loads(response)
        except Exception:
            return False
    if not isinstance(response, dict):
        return False
    direct_result = response.get('direct_result')
    if not isinstance(direct_result, dict):
        return False
    return bool(direct_result.get('kind'))


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
        value = str(item.get("value") or item.get("file_path") or item.get("url") or "").strip()
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
    if not exclude_code_blocks:
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        return ''.join('\\' + char if char in escape_chars else char for char in text)
    
    result = []
    is_code = False
    current_text = ""
    
    for char in text:
        if char == '`':
            if current_text:
                if not is_code:
                    # Экранируем текст вне блоков кода
                    current_text = ''.join('\\' + c if c in ['_', '*', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'] else c for c in current_text)
                result.append(current_text)
                current_text = ""
            result.append(char)
            is_code = not is_code
        else:
            current_text += char
    
    if current_text:
        if not is_code:
            # Экранируем оставшийся текст вне блоков кода
            current_text = ''.join('\\' + c if c in ['_', '*', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'] else c for c in current_text)
        result.append(current_text)
    
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

async def handle_direct_result(config, update: Update, response: any):
    """
    Handles a direct result from a plugin
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response.get('direct_result') if isinstance(response, dict) else None
    if not isinstance(result, dict):
        logging.warning("handle_direct_result called without a direct_result payload: %r", response)
        return []
    kind = result.get('kind')
    if not kind:
        logging.warning("handle_direct_result payload missing 'kind': %r", result)
        return []
    result_format = result.get('format')
    value = result.get('value')
    add_value = result.get('add_value', None)
    logging.info(f"Handling direct result - kind: {kind}, format: {result_format}, value: {value}, add_value: {str(add_value)[:200]}")

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
            text_messages = await handle_direct_result(config, update, text_result)
            if text_messages:
                sent_messages.extend(text_messages)

        for artifact in result.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            artifact_messages = await handle_direct_result(config, update, {"direct_result": artifact})
            if artifact_messages:
                sent_messages.extend(artifact_messages)
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
                logging.error(f"Error handling photo: {e}")
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
                logging.warning(f"Could not set reaction direct result: {e}", exc_info=True)
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
        text = add_value if add_value else value
        chunks = split_into_chunks(text)
        if result_format == 'markdown':
            parse_mode = constants.ParseMode.MARKDOWN
        else:
            parse_mode = None

        # Отправляем как файл если: 
        # - ответ больше 3х частей ИЛИ 
        # - (ответ больше одной части И содержит вставки кода)
        if should_send_text_as_file(
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
            for i, chunk in enumerate(chunks):
                # Only reply to original message for first chunk
                reply_to = get_reply_to_message_id(config, update) if i == 0 else None
                try:
                    sent_messages.append(await message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=reply_to,
                        text=chunk,
                        parse_mode=parse_mode
                    ))
                except telegram.error.BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        logging.warning(f"Markdown parsing error in handle_direct_result: {e}. Retrying without markdown formatting.")
                        # Убираем проблемные символы и пытаемся снова
                        try:
                            # Экранируем специальные символы для markdown
                            escaped_chunk = escape_markdown(chunk, exclude_code_blocks=False)
                            sent_messages.append(await message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=reply_to,
                                text=escaped_chunk,
                                parse_mode=parse_mode
                            ))
                        except Exception:
                            # Если все еще не получается, отправляем без форматирования
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
                    logging.error(f"Unexpected error in handle_direct_result: {e}")
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
    value = result.get('value') or result.get('file_path')

    if format == 'path' and value:
        if os.path.exists(value):
            os.remove(value)

# Function to encode the image
def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'

def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,'):]
    return base64.b64decode(image)

# Функция для конвертации markdown в HTML
def markdown_to_html(text):
    import re
    
    # Экранирование HTML-символов
    def escape_html(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Последовательность преобразований
    conversions = [
        # Блоки кода с подсветкой языка
        (r'```(\w*)\n(.*?)```', 
            lambda m: f'<pre><code class="language-{m.group(1)}">{escape_html(m.group(2))}</code></pre>', 
            re.DOTALL),
        
        # Моноширный текст (код)
        (r'`(.*?)`', 
            lambda m: f'<code>{escape_html(m.group(1))}</code>'),
        
        # Жирный текст (должен идти до курсива)
        (r'\*\*(.*?)\*\*', 
            lambda m: f'<b>{m.group(1)}</b>'),
        (r'__(.*?)__', 
            lambda m: f'<b>{m.group(1)}</b>'),
        
        # Курсив
        (r'\*(.*?)\*', 
            lambda m: f'<i>{m.group(1)}</i>'),
        (r'_(.*?)_', 
            lambda m: f'<i>{m.group(1)}</i>'),
        
        # Зачеркнутый текст
        (r'~~(.*?)~~', 
            lambda m: f'<s>{m.group(1)}</s>'),
        
        # Ссылки с титлом
        (r'\[(.*?)\]\((.*?)(?:\s+"(.*?)")?\)', 
            lambda m: f'<a href="{m.group(2)}" title="{m.group(3) or ""}">{m.group(1)}</a>'),
        
        # Списки
        (r'^(\s*[-*+])\s*(.*)', 
            lambda m: f'<li>{m.group(2)}</li>', 
            re.MULTILINE),
        
        # Заголовки
        (r'^(#{1,6})\s*(.*)', 
            lambda m: f'<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>', 
            re.MULTILINE)
    ]
    
    # Применяем преобразования
    for pattern, repl, *flags in conversions:
        flag = flags[0] if flags else 0
        text = re.sub(pattern, repl, text, flags=flag)
    
    # Переносы строк и абзацы
    text = re.sub(r'\n\n+', '</p><p>', text)
    text = re.sub(r'\n', '<br>', text)
    
    # Оборачиваем в параграфы, если нет других блочных элементов
    if not re.search(r'<(h\d|pre|ul|ol|blockquote)', text):
        text = f'<p>{text}</p>'
    
    return text

async def send_long_response_as_file(config, update: Update, response: str, session_name: str = 'response'):
    """
    Отправляет длинный ответ в виде HTML-файла с сохранением форматирования
    
    :param config: Конфигурация бота
    :param update: Объект обновления Telegram
    :param response: Текст ответа для отправки
    :param session_name: Базовое имя файла (по умолчанию 'response')
    """
    # Создаем директории output, data и plots, если их нет
    os.makedirs('output', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Генерируем уникальный идентификатор сессии
    session_id = str(uuid.uuid4())[:8]
    
    # Используем HTMLVisualizer для создания HTML-файла
    visualizer = HTMLVisualizer()
    output_path = visualizer.advanced_visualization(response, session_id)
    
    # Получаем содержимое созданного файла
    with open(output_path, 'rb') as f:
        file_content = f.read()
    
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
        logging.warning(f"Не удалось удалить временный файл {output_path}: {e}")

    return sent_message
