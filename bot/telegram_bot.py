from __future__ import annotations

import asyncio
import logging
import os
import io
import requests
import json
import sys
import yaml
from typing import Dict

from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
from PIL import Image

from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files
from openai_helper import GPT_3_16K_MODELS, GPT_3_MODELS, GPT_4_128K_MODELS, GPT_4_32K_MODELS, GPT_4_MODELS, GPT_4_VISION_MODELS, GPT_4O_MODELS, OpenAIHelper, localized_text, O1_MODELS, GPT_ALL_MODELS, ANTHROPIC, GOOGLE, MISTRALAI
from usage_tracker import UsageTracker
from database import Database

#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


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
        bot_language = self.config['bot_language']
        self.commands = [
            BotCommand(command='help', description=localized_text('help_description', bot_language)),
            BotCommand(command='reset', description=localized_text('reset_description', bot_language)),
            BotCommand(command='stats', description=localized_text('stats_description', bot_language)),
            BotCommand(command='resend', description=localized_text('resend_description', bot_language)),
            BotCommand(command='model', description=localized_text('change_model', bot_language)),
            BotCommand(command='animate', description='Convert last uploaded image to video animation'),
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
        self.buffer_lock = asyncio.Lock()  # Добавьте блокировку для потокобезопасности
        self.application = None
        # Add new tracking for last image
        self.last_image = {}  # {chat_id: file_id}
        self.db = Database()  # Инициализация базы данных

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        bot_language = self.config['bot_language']
        #tool_list = "\n".join([f"- {tool['name']}" for tool in TOOLS])
        help_text = (
                localized_text('help_text', bot_language)[0] +
                '\n\n' +
                '\n'.join(commands_description) +
                '\n\n' +
                localized_text('help_text', bot_language)[1] +
                '\n\n' +
                localized_text('help_text', bot_language)[2] +
                '\n\n' +
                'Ты можешь попросить дать транскрипцию видео с youtube, указав адрес с видео (например, https://youtu.be/dQw4w9WgXcQ)' +
                '\n\n' +
                'Ты можешь автомтатически опимизировать промт, пример вызова: "Помоги написать план путешествия. Используй Prompt Perfect"' +
                '\n\n' +
                'Ты можешь попросить перевести текст на любой язык. Ты можешь указать адрес конкретной страницы в интернете и попросить перевести ее на любой язык' +
                '\n\n' +
                'Ты можешь попросить нарисовать картинку по твоему запросу' +
                '\n\n' +
                'Ты можешь попросить саммари страницы в интернете. Если добавить - переведи ее в речь, получишь саммари в голосовом сообщении' +
                '\n\n' +
                'Ты можешь попросить установить напоминание на любое время. Так же можешь попросить показать напоминания и удалить напоминание по ID' +
                '\n\n' +
                'Ты можешь загрузить текстовый документ (.txt, .docx, .pdf, .rtf, .doc) и задавать вопросы по его содержимому, используя команду /ask_question. ' +
                'Для удаления документа используй команду /delete_document'
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     'requested their usage statistics')

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

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
        bot_language = self.config['bot_language']

        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ""
        if self.config.get('enable_image_generation', False):
            text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"

        text_today_vision = ""
        if self.config.get('enable_vision', False):
            text_today_vision = f"{vision_today} {localized_text('stats_vision', bot_language)}\n"

        text_today_tts = ""
        if self.config.get('enable_tts_generation', False):
            text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"

        text_today = (
            f"*{localized_text('usage_today', bot_language)}:*\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_today_images}"  # Include the image statistics for today if applicable
            f"{text_today_vision}"
            f"{text_today_tts}"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
            "----------------------------\n"
        )

        text_month_images = ""
        if self.config.get('enable_image_generation', False):
            text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"

        text_month_vision = ""
        if self.config.get('enable_vision', False):
            text_month_vision = f"{vision_month} {localized_text('stats_vision', bot_language)}\n"

        text_month_tts = ""
        if self.config.get('enable_tts_generation', False):
            text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = (
            f"*{localized_text('usage_month', bot_language)}:*\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_month_images}"  # Include the image statistics for the month if applicable
            f"{text_month_vision}"
            f"{text_month_tts}"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
        )

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

        usage_text = text_current_conversation + text_today + text_month + text_budget
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
            logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            ' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            ' does not have anything to resend')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language'])
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Change model using inline keyboard buttons.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to change model')
            await self.send_disallowed_message(update, context)
            return

        user_id = update.message.from_user.id
        # Получаем модель из базы данных
        current_model = self.db.get_user_model(user_id) or self.openai.config['model']

        # Создаем кнопки для каждой группы моделей
        keyboard = []
        model_groups = [
            ("GPT-4O", GPT_4O_MODELS),
            ("O1", O1_MODELS),
            ("Anthropic", ANTHROPIC),
            ("Google", GOOGLE),
            ("Mistral", MISTRALAI)
        ]

        for group_name, _ in model_groups:
            keyboard.append([InlineKeyboardButton(
                text=group_name,
                callback_data=f"modelgroup:{group_name}"
            )])

        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Current model: {current_model}\n\nSelect model group:",
            reply_markup=reply_markup
        )

    async def handle_model_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle model selection from inline keyboard.
        """
        query = update.callback_query
        await query.answer()
        
        data = query.data.split(':')
        action = data[0]
        value = data[1]
        
        user_id = query.from_user.id
        # Получаем текущую модель из базы данных
        current_model = self.db.get_user_model(user_id) or self.openai.config['model']
        
        if action == "modelgroup":
            # Показываем модели выбранной группы
            keyboard = []
            model_groups = [
                ("GPT-4O", GPT_4O_MODELS),
                ("O1", O1_MODELS),
                ("Anthropic", ANTHROPIC),
                ("Google", GOOGLE),
                ("Mistral", MISTRALAI)
            ]
            
            selected_models = None
            for group_name, models in model_groups:
                if group_name == value:
                    selected_models = models
                    break
            
            if selected_models:
                # Создаем кнопки для моделей
                for model_name in selected_models:
                    button_text = f"✓ {model_name}" if model_name == current_model else model_name
                    keyboard.append([InlineKeyboardButton(
                        text=button_text,
                        callback_data=f"model:{model_name}"
                    )])
                
                # Добавляем кнопку "Назад"
                keyboard.append([InlineKeyboardButton(
                    text="« Back to groups",
                    callback_data="modelback:main"
                )])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"Current model: {current_model}\n\nSelect model from {value} group:",
                    reply_markup=reply_markup
                )
                
        elif action == "model":
            # Обработка выбора конкретной модели
            model_name = value
            if model_name in GPT_ALL_MODELS:
                # Сохраняем выбранную модель в базу данных
                self.db.save_user_model(user_id, model_name)
                await query.edit_message_text(
                    f"Model changed to: {model_name}"
                )
            else:
                await query.edit_message_text(
                    f"Invalid model selection: {model_name}"
                )
                
        elif action == "modelback":
            # Возврат к списку групп
            keyboard = []
            model_groups = [
                ("GPT-4O", GPT_4O_MODELS),
                ("O1", O1_MODELS),
                ("Anthropic", ANTHROPIC),
                ("Google", GOOGLE),
                ("Mistral", MISTRALAI)
            ]
            
            for group_name, _ in model_groups:
                keyboard.append([InlineKeyboardButton(
                    text=group_name,
                    callback_data=f"modelgroup:{group_name}"
                )])
                
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f"Current model: {current_model}\n\nSelect model group:",
                reply_markup=reply_markup
            )

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error = False):
        """
        Reset the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                          'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})...')

        if error:
            # Сброс из-за ошибки
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text='Ошибка. Сбрасываю контекст...'
            )
            return

        try:
            # Загружаем режимы из файла
            current_dir = os.path.dirname(os.path.abspath(__file__))
            chat_modes_path = os.path.join(current_dir, 'chat_modes.yml')
            
            with open(chat_modes_path, 'r', encoding='utf-8') as file:
                chat_modes = yaml.safe_load(file)

            # Проверяем, есть ли текст после команды reset
            if context.args:
                # Получаем текст после команды
                mode_query = ' '.join(context.args).lower()
                
                # Ищем режим по имени (регистронезависимо)
                found_mode = None
                for mode_key, mode_data in chat_modes.items():
                    if mode_data.get('name', '').lower() == mode_query or mode_key.lower() == mode_query:
                        found_mode = (mode_key, mode_data)
                        break
                
                if found_mode:
                    mode_key, mode_data = found_mode
                    # Сбрасываем историю чата с новым промптом
                    reset_content = mode_data.get('prompt_start', '')
                    self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
                                        
                    # Отправляем приветственное сообщение
                    welcome_message = mode_data.get('welcome_message', 'Режим успешно изменен')
                    parse_mode = mode_data.get('parse_mode', 'HTML')
                    
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=f"Режим изменен на: {mode_data['name']}\n\n{welcome_message}",
                        parse_mode=parse_mode
                    )
                    return
                else:
                    # Если режим не найден, отправляем сообщение об ошибке
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=f"Режим '{mode_query}' не найден. Выберите режим из списка:",
                    )
                    # Продолжаем выполнение и показываем список режимов

            # Группируем режимы по group
            mode_groups = {}
            for mode_key, mode_data in chat_modes.items():
                group = mode_data.get('group', 'Другое')
                if group not in mode_groups:
                    mode_groups[group] = []
                mode_groups[group].append((mode_key, mode_data))

            # Создаем клавиатуру с группами
            keyboard = []
            # Добавляем кнопку сброса контекста в начало списка
            keyboard.append([InlineKeyboardButton(
                text="🔄 Сбросить контекст",
                callback_data="promptgroup:reset_context"
            )])
            # Добавляем остальные группы
            for group_name in sorted(mode_groups.keys()):
                keyboard.append([InlineKeyboardButton(
                    text=group_name,
                    callback_data=f"promptgroup:{group_name}"
                )])

            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Выберите группу режимов:",
                reply_markup=reply_markup
            )
        except Exception as e:
            logging.error(f"Error in reset: {str(e)}", exc_info=True)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Произошла ошибка при сбросе контекста. Пожалуйста, попробуйте еще раз через несколько секунд."
            )

    async def handle_prompt_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обрабатывает выбор промпта пользователем
        """
        query = update.callback_query
        await query.answer()
        
        data = query.data.split(':')
        action = data[0]
        value = data[1]
        
        # Загружаем промпты из файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chat_modes_path = os.path.join(current_dir, 'chat_modes.yml')
        
        with open(chat_modes_path, 'r', encoding='utf-8') as file:
            chat_modes = yaml.safe_load(file)

        if action == "promptgroup":
            if value == "reset_context":
                # Просто сбрасываем контекст без смены роли
                chat_id = update.effective_chat.id
                self.openai.reset_chat_history(chat_id=chat_id)
                                
                await query.edit_message_text(
                    text="Контекст диалога сброшен. Можете начать новый диалог."
                )
                return
                
            # Показываем режимы выбранной группы
            keyboard = []
            for mode_key, mode_data in chat_modes.items():
                if mode_data.get('group', 'Другое') == value:
                    keyboard.append([InlineKeyboardButton(
                        text=mode_data.get('name', mode_key),
                        callback_data=f"prompt:{mode_key}"
                    )])
            
            # Добавляем кнопку "Назад"
            keyboard.append([InlineKeyboardButton(
                text="« Назад к группам",
                callback_data="promptback:main"
            )])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text=f"Выберите режим из группы {value}:",
                reply_markup=reply_markup
            )
            
        elif action == "promptback":
            # Возврат к списку групп
            mode_groups = {}
            for mode_key, mode_data in chat_modes.items():
                group = mode_data.get('group', 'Другое')
                if group not in mode_groups:
                    mode_groups[group] = []
                mode_groups[group].append((mode_key, mode_data))

            # Создаем клавиатуру с группами
            keyboard = []
            # Добавляем кнопку сброса контекста в начало списка
            keyboard.append([InlineKeyboardButton(
                text="🔄 Сбросить контекст",
                callback_data="promptgroup:reset_context"
            )])
            # Добавляем остальные группы
            for group_name in sorted(mode_groups.keys()):
                keyboard.append([InlineKeyboardButton(
                    text=group_name,
                    callback_data=f"promptgroup:{group_name}"
                )])

            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text="Выберите группу режимов:",
                reply_markup=reply_markup
            )
            
        elif action == "prompt":
            mode = value
            if mode in chat_modes:
                chat_id = update.effective_chat.id
                mode_data = chat_modes[mode]
                
                # Сбрасываем историю чата с новым промптом
                reset_content = mode_data.get('prompt_start', '')
                self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
                
                # Сохраняем новый контекст в базу данных
                self.db.save_conversation_context(chat_id, {
                    'messages': self.openai.conversations[chat_id],
                }, mode_data.get('parse_mode', 'HTML'), mode_data.get('temperature', self.openai.config['temperature']))
                                        
                # Отправляем приветственное сообщение
                welcome_message = mode_data.get('welcome_message', 'Режим успешно изменен')
                parse_mode = mode_data.get('parse_mode', 'HTML')
                
                await query.edit_message_text(
                    text=f"Режим изменен на: {mode_data['name']}\n\n{welcome_message}",
                    parse_mode=parse_mode
                )
            else:
                await query.edit_message_text(
                    text="Произошла ошибка при выборе режима. Попробуйте еще раз."
                )

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Перезапускает бота. Доступно только администраторам.
        """
        if not is_admin(self.config, update.message.from_user.id):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                          'tried to restart the bot but is not admin')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Эта команда доступна только администраторам бота."
            )
            return

        logging.info(f'Restarting bot by admin {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})...')
        
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text="Перезапуск бота..."
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
            logging.warning(f"Failed to restart systemd service: {e}")
        
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

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                if self.config['image_receive_mode'] == 'photo':
                    await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        photo=image_url
                    )
                elif self.config['image_receive_mode'] == 'document':
                    await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        document=image_url
                    )
                else:
                    raise Exception(
                        f"env variable IMAGE_RECEIVE_MODE has invalid value {self.config['image_receive_mode']}")
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
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

        logging.info(f'New speech generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

                await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file
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
                logging.exception(e)
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
            logging.info('Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['bot_language']
            try:
                media_file = await self.application.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
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
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text('media_type_fail', bot_language)
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            try:
                transcript = await self.openai.transcribe(filename_mp3)

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
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=transcript)

                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Interpret image using vision model.
        """
        if not self.config['enable_vision'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        prompt = update.message.caption
        
        logging.info(f"Vision handler called for chat_id: {chat_id}")

        # Store the last image file_id for potential tool use
        if len(update.message.photo) > 0:
            file_id = update.message.photo[-1].file_id
            logging.info(f"Storing photo file_id: {file_id}")
            self.last_image[chat_id] = file_id
            self.openai.set_last_image_file_id(chat_id, file_id)
        elif update.message.document and update.message.document.mime_type.startswith('image/'):
            file_id = update.message.document.file_id
            logging.info(f"Storing document file_id: {file_id}")
            self.last_image[chat_id] = file_id
            self.openai.set_last_image_file_id(chat_id, file_id)

        logging.info(f"Stored last_image: {self.last_image}")
        logging.info(f"Stored in OpenAI helper: {self.openai.last_image_file_ids}")
        # Only proceed with vision if there's a caption or it's a valid vision request
        if prompt or (is_group_chat(update) and not self.config['ignore_group_vision']):
            if is_group_chat(update):
                if self.config['ignore_group_vision']:
                    logging.info('Vision coming from group chat, ignoring...')
                    return
                else:
                    trigger_keyword = self.config['group_trigger_keyword']
                    if (prompt is None and trigger_keyword != '') or \
                            (prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())):
                        logging.info('Vision coming from group chat with wrong keyword, ignoring...')
                        return

            image = update.message.effective_attachment[-1]

            async def _execute():
                bot_language = self.config['bot_language']
                try:
                    media_file = await self.application.bot.get_file(image.file_id)
                    temp_file = io.BytesIO(await media_file.download_as_bytearray())
                except Exception as e:
                    logging.exception(e)
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
                    logging.info(f'New vision request received from user {update.message.from_user.name} '
                                f'(id: {update.message.from_user.id})')

                except Exception as e:
                    logging.exception(e)
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
                            return await handle_direct_result(self.config, update, content)

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
                                logging.exception(e)
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                                    parse_mode=constants.ParseMode.MARKDOWN
                                )
                    except Exception as e:
                        logging.exception(e)
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update),
                            text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                            parse_mode=constants.ParseMode.MARKDOWN
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
                text="I've received your image. You can now ask me to process it or animate it."
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
        prompt = message_text(update.message)
        message_id = update.message.message_id

        logging.info(f"Prompt handler called for chat_id: {chat_id}")
        logging.info(f"Last image in bot: {self.last_image.get(chat_id)}")
        logging.info(f"Last image in OpenAI helper: {self.openai.get_last_image_file_id(chat_id)}")

        # Add the last image file_id to the context if available
        if chat_id in self.last_image:
            self.openai.set_last_image_file_id(chat_id, self.last_image[chat_id])
            # Add image context to the conversation
            #if not hasattr(self.openai, 'conversations'):
            #    self.openai.conversations = {}
            #if chat_id not in self.openai.conversations:
            #    self.openai.conversations[chat_id] = []
            # Add image context as system message
            #self.openai.conversations[chat_id].append({
            #    "role": "user",
            #    "content": f"User has uploaded an image with file_id: {self.last_image[chat_id]}"
            #})


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
                        logging.error(f"Error processing message: {e}")
                        continue

                    await asyncio.sleep(0.1)  # Prevent flooding

        except Exception as e:
            logging.error(f"Error in process_buffer: {e}")
        finally:
            # Reset processing flag
            async with self.buffer_lock:
                if chat_id in self.message_buffer:
                    self.message_buffer[chat_id]['processing'] = False
                    
    async def process_message(self, prompt: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обрабатывает полное сообщение
        """
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        message_id = update.message.message_id
        self.last_message[chat_id] = prompt
        request_id = f"{chat_id}_{message_id}"
            
        logging.info(
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
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            total_tokens = 0

            # Получаем модель из базы данных
            model_to_use = self.db.get_user_model(user_id) or self.openai.config['model']
            
            if self.config['stream'] and model_to_use not in (O1_MODELS + ANTHROPIC + GOOGLE + MISTRALAI):

                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )

                # Store message_id in openai object for this request
                self.openai.message_ids = getattr(self.openai, 'message_ids', {})
                self.openai.message_ids[request_id] = message_id

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt, request_id=request_id)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

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
                            # Получаем parse_mode из контекста
                            #chat_context, parse_mode, temperature = self.db.get_conversation_context(chat_id) or {}
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                                #parse_mode=parse_mode
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
                async def _reply():
                    nonlocal total_tokens
                    # Store message_id in openai object for this request
                    self.openai.message_ids = getattr(self.openai, 'message_ids', {})
                    self.openai.message_ids[request_id] = message_id

                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt, request_id=request_id)

                    if is_direct_result(response):
                        analytics_plugin = self.openai.plugin_manager.get_plugin('ConversationAnalytics')
                        if analytics_plugin:
                            message_data = {
                                'text': prompt,
                                'tokens': total_tokens,
                                'user_id': user_id
                            }
                            analytics_plugin.update_stats(str(chat_id), message_data)
                        return await handle_direct_result(self.config, update, response)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

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

            analytics_plugin = self.openai.plugin_manager.get_plugin('ConversationAnalytics')
            if analytics_plugin:
                message_data = {
                    'text': prompt,
                    'tokens': total_tokens,
                    'user_id': user_id
                }
                analytics_plugin.update_stats(str(chat_id), message_data)

            result = add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)
            if not result:
                await self.reset(update, context, True)

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
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
            logging.error(f'An error occurred while generating the result card for inline query {e}')

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

                unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
                model_to_use = self.openai.user_models.get(str(user_id), self.openai.config['model'])
                if self.config['stream'] and model_to_use not in (O1_MODELS + ANTHROPIC + GOOGLE + MISTRALAI):
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

                        logging.info(f'Generating response for inline query by {name}')
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
            logging.error(f'Failed to respond to an inline query via button callback: {e}')
            logging.exception(e)
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
        name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
        user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
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
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                #parse_mode=parse_mode,
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
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.budget_limit_message,
                #parse_mode=parse_mode
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.commands)
        await application.bot.set_my_commands(
            self.group_commands,
            scope=BotCommandScopeAllGroupChats()
        )

        # Регистрируем команды от плагинов
        plugin_commands = self.openai.plugin_manager.get_plugin_commands()
        for cmd in plugin_commands:
            handler = CommandHandler(
                cmd['command'],
                lambda update, context, cmd=cmd: self.handle_plugin_command(update, context, cmd),
                filters=filters.COMMAND
            )
            application.add_handler(handler)
            # Добавляем команду в список команд бота
            self.commands.append(BotCommand(
                command=cmd['command'],
                description=cmd['description']
            ))
        
        # Обновляем команды бота
        await application.bot.set_my_commands(self.commands)
        await application.bot.set_my_commands(
            self.group_commands + [BotCommand(cmd['command'], cmd['description']) for cmd in plugin_commands],
            scope=BotCommandScopeAllGroupChats()
        )

    async def handle_plugin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, cmd: Dict):
        """Обработчик команд плагинов"""
        try:
            # Проверяем права доступа
            if not await is_allowed(self.config, update, context):
                await self.send_disallowed_message(update, context)
                return

            # Получаем аргументы команды
            args = context.args
            kwargs = cmd['handler_kwargs'].copy()
            
            # Если команда требует аргументы, но они не предоставлены
            if cmd.get('args') and not args:
                await update.message.reply_text(
                    f"Использование: /{cmd['command']} {cmd.get('args', '')}\n"
                    f"Описание: {cmd['description']}"
                )
                return

            # Добавляем chat_id и аргументы в kwargs
            kwargs['chat_id'] = str(update.effective_chat.id)
            kwargs['update'] = update  # Добавляем update в kwargs
            if cmd.get('args'):
                kwargs['query'] = ' '.join(args)  # Для команд, требующих текстовый запрос
                if '<document_id>' in cmd.get('args'):
                    kwargs['document_id'] = args[0]  # Для команд, требующих ID документа

            # Вызываем обработчик команды
            result = await cmd['handler'](kwargs['function_name'], self.openai, **{k:v for k,v in kwargs.items() if k != 'function_name'})
            
            # Если результат содержит direct_result, обрабатываем его
            if is_direct_result(result):
                await handle_direct_result(self.config, update, result)
            elif isinstance(result, dict) and 'error' in result:
                await update.message.reply_text(f"Ошибка: {result['error']}")
            else:
                await update.message.reply_text(str(result))

        except Exception as e:
            logging.error(f"Ошибка при обработке команды плагина: {e}")
            await update.message.reply_text(f"Произошла ошибка при выполнении команды: {str(e)}")

    async def cleanup(self):
        """
        Comprehensive cleanup:
        - Cancel all timers
        - Clear message buffers
        - Stop background tasks
        - Close any open connections/resources
        """
        try:
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
                        logging.error(f"Error cancelling task: {e}")

            # Cancel all running tasks except current
            current_task = asyncio.current_task()
            running_tasks = [t for t in asyncio.all_tasks() 
                           if t is not current_task and not t.done()]
            
            if running_tasks:
                logging.info(f"Cancelling {len(running_tasks)} remaining tasks...")
                for task in running_tasks:
                    task.cancel()
                
                await asyncio.gather(*running_tasks, return_exceptions=True)

            # Close any open resources
            if hasattr(self, 'openai'):
                await self.openai.close()

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise

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
                logging.error(f"Error in buffer data checker: {e}")
            
            await asyncio.sleep(1)

    async def start_reminder_checker(self, plugin_manager):
        reminders_plugin = plugin_manager.get_plugin('reminders')
        if reminders_plugin:
            # Continuously check reminders
            while True:
                try:
                    await reminders_plugin.check_reminders(self.application.bot)
                except Exception as e:
                    logging.error(f"Error in reminder checker: {e}")
                
                # Sleep for a minute between checks to avoid excessive processing
                await asyncio.sleep(60)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик для загруженных документов
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                          'is not allowed to upload documents')
            await self.send_disallowed_message(update, context)
            return

        try:
            document = update.message.document
            logging.info(f"Получен документ: {document.file_name}, mime_type: {document.mime_type}")
            
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
            
            logging.info(f"Проверка типа файла: mime_type={document.mime_type}, extension={file_extension}")
            
            # Проверяем сначала MIME-тип, потом расширение
            if document.mime_type not in supported_mimes and file_extension not in supported_extensions:
                await update.message.reply_text(
                    "Пожалуйста, загрузите текстовый документ в одном из следующих форматов:\n" +
                    ", ".join(supported_extensions)
                )
                logging.warning(f"Файл отклонен: неподдерживаемый формат {document.mime_type} / {file_extension}")
                return

            logging.info("Начинаем скачивание файла...")
            # Скачиваем файл
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            logging.info(f"Файл успешно скачан, размер: {len(file_content)} байт")
            
            # Вызываем плагин для обработки документа
            plugin = self.openai.plugin_manager.get_plugin('text_document_qa')
            if not plugin:
                logging.error("Плагин text_document_qa не найден")
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text="Document processing is not available. The plugin is not enabled."
                )
                return

            logging.info("Передаем файл в плагин для обработки...")
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
                logging.error(f"Ошибка от плагина: {result['error']}")
                await update.message.reply_text(f"Ошибка: {result['error']}")
            else:
                try:
                    logging.info("Файл успешно обработан, отправляем результат")
                    await handle_direct_result(self.config, update, result)
                except Exception as e:
                    logging.error(f"Error handling direct result: {e}")
                    await update.message.reply_text(str(result))

        except Exception as e:
            error_text = f"Произошла ошибка при обработке документа: {str(e)}"
            logging.error(error_text)
            await update.message.reply_text(error_text)


    def run(self):
        """
        Runs the bot indefinitely.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            application = ApplicationBuilder() \
                .token(self.config['token']) \
                .proxy_url(self.config['proxy']) \
                .get_updates_proxy_url(self.config['proxy']) \
                .post_init(self.post_init) \
                .concurrent_updates(True) \
                .build()

            self.application = application
            self.openai.bot = application.bot
            loop.create_task(self.buffer_data_checker())
            loop.create_task(self.start_reminder_checker(self.openai.plugin_manager))

            application.add_handler(CommandHandler('restart', self.restart))
            application.add_handler(CommandHandler('reset', self.reset))
            application.add_handler(CommandHandler('help', self.help))
            application.add_handler(CommandHandler('image', self.image))
            application.add_handler(CommandHandler('tts', self.tts))
            application.add_handler(CommandHandler('start', self.help))
            application.add_handler(CommandHandler('stats', self.stats))
            application.add_handler(CommandHandler('resend', self.resend))
            application.add_handler(CommandHandler('model', self.model))
            application.add_handler(CommandHandler('animate', self.animate))
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
            application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))

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

            application.add_handler(CallbackQueryHandler(self.handle_model_callback, pattern="^model|modelgroup|modelback"))
            application.add_handler(CallbackQueryHandler(self.handle_prompt_selection, pattern="^prompt|promptgroup|promptback"))
            application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))

            application.add_error_handler(error_handler)

            application.run_polling()
        finally:
            # Завершаем выполнение задач в текущем событийном цикле
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(self.cleanup())

    async def animate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Convert the last uploaded image to video animation.
        """
        if not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        animation_prompt = message_text(update.message)

        if chat_id not in self.last_image:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Please upload an image first, then use the /animate command."
            )
            return

        logging.info(f'New animation request received from user {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})')

        async def _execute():
            try:
                # Get the plugin instance
                plugin = self.openai.plugin_manager.get_plugin('haiper_image_to_video')
                if not plugin:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text="Image to video conversion is not available. The plugin is not enabled."
                    )
                    return

                # Execute the plugin function directly
                response = await plugin.execute(
                    'convert_image_to_video',
                    self.openai,
                    last_image_file_id=self.last_image[chat_id],
                    prompt=animation_prompt
                )

                if isinstance(response, str):
                    response = json.loads(response)

                if isinstance(response, dict) and 'error' in response:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=f"Error creating animation: {response['error']}"
                    )
                    return

                # Handle the response based on its format
                if isinstance(response, dict) and 'direct_result' in response:
                    result = response['direct_result']
                    if result['kind'] == 'video' and result['format'] == 'path':
                        # Send the video file
                        with open(result['value'], 'rb') as video_file:
                            await update.effective_message.reply_video(
                                video=video_file,
                                message_thread_id=get_thread_id(update),
                                caption="Here's your animated video!"
                            )
                        # Clean up the temporary file
                        os.remove(result['value'])
                else:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text="Unexpected response format from the animation service."
                    )

            except Exception as e:
                logging.error(f"Error in animation process: {e}")
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"Error creating animation: {str(e)}"
                )

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.UPLOAD_VIDEO)
        
