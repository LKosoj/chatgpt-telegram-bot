from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import io
import base64

import telegram
from telegram import Message, MessageEntity, Update, ChatMember, constants
from telegram.ext import CallbackContext, ContextTypes

from .usage_tracker import UsageTracker

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
        chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
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

async def wrap_with_indicator(update: Update, context: CallbackContext, coroutine,
                            chat_action: constants.ChatAction = "", is_inline=False):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    task = context.application.create_task(coroutine(), update=update)
    try:
        # Increase timeout to 380 seconds
        async with asyncio.timeout(380):
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

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    if is_admin(config, user_id):
        return True
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
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
        "daily": "cost_today",
        "all-time": "cost_all_time"
    }

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map[budget_period]]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
    cost = usage['guests'].get_current_cost()[budget_cost_map[budget_period]]
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
    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)
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
        return update.message.message_id
    return None

def is_direct_result(response: any) -> bool:
    """
    Checks if the dict contains a direct result that can be sent directly to the user
    :param response: The response value
    :return: Boolean indicating if the result is a direct result
    """
    if type(response) is not dict:
        try:
            json_response = json.loads(response)
            return json_response.get('direct_result', False)
        except:
            return False
    else:
        return response.get('direct_result', False)

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

async def handle_direct_result(config, update: Update, response: any):
    """
    Handles a direct result from a plugin
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response['direct_result']
    kind = result['kind']
    format = result['format']
    value = result['value']
    add_value = result.get('add_value', None)
    logging.info(f"Handling direct result - kind: {kind}, format: {format}, value: {value}")

    common_args = {
        'message_thread_id': get_thread_id(update),
        'reply_to_message_id': get_reply_to_message_id(config, update),
    }

    if kind == 'photo':
        if format == 'url':
            await update.effective_message.reply_photo(**common_args, photo=value)
        elif format == 'path':
            await update.effective_message.reply_photo(**common_args, photo=open(value, 'rb'))
    elif kind == 'gif' or kind == 'file':
        if format == 'url':
            await update.effective_message.reply_document(**common_args, document=value)
        if format == 'path':
            await update.effective_message.reply_document(**common_args, document=open(value, 'rb'))
    elif kind == 'dice':
        await update.effective_message.reply_dice(**common_args, emoji=value)

    if add_value or kind == 'text':
        # Split long messages into chunks
        if add_value:
            chunks = split_into_chunks(add_value)
        else:
            chunks = split_into_chunks(value)
        if format == 'markdown':
            parse_mode = constants.ParseMode.MARKDOWN
        else:
            parse_mode = None
        for i, chunk in enumerate(chunks):
            # Only reply to original message for first chunk
            reply_to = get_reply_to_message_id(config, update) if i == 0 else None
            try:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=reply_to,
                    text=chunk,
                    parse_mode=parse_mode
                )
            except Exception as e:
                logging.error(f"Unexpected error in handle_direct_result: {e}")
                # В случае любой другой ошибки отправляем без форматирования
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=reply_to,
                    text=chunk,
                    parse_mode=None
                )

    if format == 'path':
        cleanup_intermediate_files(response)

def cleanup_intermediate_files(response: any):
    """
    Deletes intermediate files created by plugins
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response['direct_result']
    format = result['format']
    value = result['value']

    if format == 'path':
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
    # Конвертируем ответ в HTML
    formatted_response = markdown_to_html(response)

    # Создаем полноценный HTML-документ
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>{session_name}</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f4f4f4;
    }}
    pre {{
        background-color: #f1f1f1;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
    code {{
        background-color: #f1f1f1;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: monospace;
    }}
    a {{
        color: #0066cc;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 1em;
        margin-bottom: 0.5em;
    }}
    li {{
        margin-bottom: 0.5em;
    }}
</style>
</head>
<body>
<h1>Полный ответ</h1>
{formatted_response}
</body>
</html>"""
    
    # Создаем файл с ответом
    response_file = io.BytesIO()
    response_file.write(html_content.encode('utf-8'))
    response_file.seek(0)
    
    # Формируем имя файла, заменяя недопустимые символы
    import re
    from datetime import datetime
    
    safe_session_name = re.sub(r'[^\w\-_\.]', '_', session_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{safe_session_name}_{timestamp}.html"
    
    await update.effective_message.reply_document(
        message_thread_id=get_thread_id(update),
        reply_to_message_id=get_reply_to_message_id(config, update),
        document=response_file,
        filename=filename,
        caption="Полный ответ:",
        parse_mode=constants.ParseMode.HTML
    )

