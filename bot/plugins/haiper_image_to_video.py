#haiper_image_to_video.py
import os
import logging
import tempfile
import aiohttp
import io
import base64
import json
import asyncio
from typing import Dict, List, Optional
from PIL import Image
from datetime import datetime, timedelta
import contextlib
from asyncio import Queue, Task
from dataclasses import dataclass
from enum import Enum
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    ForceReply, 
    ReplyKeyboardRemove
)
from telegram.ext import ContextTypes, ConversationHandler, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, filters
import hashlib
import telegram
from telegram import constants

from .plugin import Plugin
from ..utils import escape_markdown
# API Configuration
API_URL = "https://api.vsegpt.ru/v1/video"
MAX_RETRIES = 4
RETRY_DELAY = 10
STATUS_CHECK_INTERVAL = 10
TIMEOUT_MINUTES = 45

logger = logging.getLogger(__name__)

# Состояния диалога
WAITING_PROMPT = 1

class TempFileManager:
    """
    Context manager for handling temporary files with automatic cleanup.
    
    Attributes:
        suffix (Optional[str]): File extension for the temporary file
        temp_file: Temporary file object
        path (Optional[str]): Path to the temporary file
    """

    def __init__(self, suffix: Optional[str] = None) -> None:
        """
        Initialize the temporary file manager.
        
        Args:
            suffix: Optional file extension for the temporary file
        """
        self.suffix = suffix
        self.temp_file = None
        self.path = None

    def __enter__(self) -> 'TempFileManager':
        """
        Create and open a temporary file.
        
        Returns:
            TempFileManager: Self reference for context manager
        """
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self.suffix)
        self.path = self.temp_file.name
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """
        Clean up the temporary file when exiting the context.
        
        Args:
            exc_type: Type of the exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Traceback of the exception that occurred, if any
        """
        if self.temp_file:
            self.temp_file.close()
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {self.path}: {e}")

    def write(self, data: bytes) -> None:
        """
        Write data to the temporary file.
        
        Args:
            data: Bytes to write to the file
        """
        if self.temp_file:
            self.temp_file.write(data)
            self.temp_file.flush()

class TaskStatus(Enum):
    """
    Enumeration of possible task statuses.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class VideoTask:
    """
    Data class representing a video generation task.
    
    Attributes:
        task_id (str): Unique identifier for the task
        user_id (int): Telegram user ID
        chat_id (int): Telegram chat ID
        file_id (str): Telegram file ID of the source image
        prompt (str): Text prompt for video generation
        status (TaskStatus): Current status of the task
        result (Optional[Dict]): Task result data if completed
        error (Optional[str]): Error message if task failed
    """
    task_id: str
    user_id: int
    chat_id: int
    file_id: str
    prompt: str
    status: TaskStatus
    result: Optional[Dict] = None
    error: Optional[str] = None

class HaiperImageToVideoPlugin(Plugin):
    def __init__(self):
        self.haiper_token = None
        self.headers = None
        self.status_headers = None
        self.task_queue: Queue[VideoTask] = Queue()
        self.active_tasks: Dict[str, VideoTask] = {}
        self.worker_task: Optional[Task] = None
        self.user_settings: Dict[int, Dict[str, str]] = {}  # Для хранения настроек пользователей
        self.openai = None
        self.bot = None

    def get_source_name(self) -> str:
        return "HaiperImageToVideo"

    def get_spec(self) -> List[Dict]:
        """Возвращает спецификацию плагина"""
        return [
            {
                "name": "convert_image_to_video",
                "description": "Конвертирует изображение в видео с анимацией",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "prompt": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            },
        ]

    def get_commands(self) -> List[Dict]:
        """
        Возвращает список команд, поддерживаемых плагином.
        """
        return [
            {
                "command": "animate",
                "description": "Создать видео из последнего отправленного изображения",
                "handler": self.handle_animate_command,
                "help": self.handle_animate_help_command,
                "handler_kwargs": {},
                "plugin_name": "HaiperImageToVideoPlugin",
            },
            {
                "command": "animate_prompt",
                "description": "Открыть конструктор промптов для создания видео",
                "handler": self.handle_prompt_constructor,
                "plugin_name": "HaiperImageToVideoPlugin",
                "handler_kwargs": {},
                "add_to_menu": True
            },
            {
                "command": "animate_help",
                "description": "Показать помощь по команде animate",
                "handler": self.handle_animate_help_command,
                "plugin_name": "HaiperImageToVideoPlugin",
                "handler_kwargs": {},
                "add_to_menu": True
            },
            {
                # Обработчик для всех callback_query плагина, кроме apply_settings
                "callback_query_handler": self.handle_callback_query,
                "callback_pattern": "^haiper_",
                "plugin_name": "HaiperImageToVideoPlugin",
                "handler_kwargs": {}
            }
        ]

    def get_inline_handlers(self) -> List[Dict]:
        """Возвращает список обработчиков inline-запросов"""
        return [
            {
                "handler": self.handle_inline_query,
                "handler_kwargs": {}
            }
        ]

    def get_message_handlers(self) -> List[Dict]:
        """Возвращает список обработчиков сообщений"""
        return [
            {
                "handler": ConversationHandler(
                    entry_points=[
                        CommandHandler("animate_prompt", self.handle_prompt_constructor)
                    ],
                    states={
                        None: [
                            CallbackQueryHandler(self.apply_settings, pattern="^haiper_apply_settings$")
                        ],
                        WAITING_PROMPT: [
                            MessageHandler(
                                filters.TEXT & ~filters.COMMAND & filters.REPLY,
                                self.handle_prompt_reply
                            ),
                        ]
                    },
                    fallbacks=[
                        CommandHandler("cancel", self.cancel_prompt)
                    ],
                    name="haiper_conversation",
                    persistent=False,
                    map_to_parent=True
                ),
                "handler_kwargs": {}
            },
            {
                "filters": "filters.PHOTO",
                "handler": self.handle_photo_message,
                "handler_kwargs": {}
            }
        ]

    async def start_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        while True:
            try:
                task = await self.task_queue.get()
                if task.status != TaskStatus.PENDING:
                    continue

                task.status = TaskStatus.PROCESSING
                try:
                    result = await self._process_video_task(task)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    
                    # Отправляем видео пользователю
                    if task.result and 'url' in task.result:
                        try:
                            await self.bot.send_message(
                                chat_id=task.chat_id,
                                text="✨ Ваше видео готово! Загружаю..."
                            )
                            await self.bot.send_video(
                                chat_id=task.chat_id,
                                video=task.result['url'],
                                caption=f"🎬 Видео создано с промптом:\n{task.prompt}"
                            )
                        except Exception as e:
                            logging.error(f"Error sending video to user: {e}")
                            await self.bot.send_message(
                                chat_id=task.chat_id,
                                text=f"❌ Произошла ошибка при отправке видео: {str(e)}"
                            )
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    logging.error(f"Error processing task {task.task_id}: {e}")
                    # Отправляем сообщение об ошибке пользователю
                    try:
                        await self.bot.send_message(
                            chat_id=task.chat_id,
                            text=f"❌ Произошла ошибка при создании видео: {str(e)}"
                        )
                    except Exception as send_error:
                        logging.error(f"Error sending error message to user: {send_error}")

                self.task_queue.task_done()
            except Exception as e:
                logging.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)

    async def _process_video_task(self, task: VideoTask) -> Dict:
        """Обработка задачи создания видео"""
        try:
            # Get and process image
            file = await self.bot.get_file(task.file_id)
            image_bytes = await file.download_as_bytearray()
            
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            model_id = "img2vid-kling/standart16"
            
            self.haiper_token = self.openai.api_key
            self.headers = {
                "Authorization": f"Bearer {self.haiper_token}",
                "Content-Type": "application/json"
            }
            self.status_headers = {
                "Authorization": f"Bearer {self.haiper_token}"
            }

            payload = {
                "model": model_id,
                "action": "generate", 
                "aspect_ratio": "16:9",
                "prompt": task.prompt,
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            }

            start_time = datetime.now()
            timeout_time = start_time + timedelta(minutes=TIMEOUT_MINUTES)

            async with aiohttp.ClientSession() as session:
                # Initial generation request with retry logic
                response_data = None
                for retry in range(MAX_RETRIES):
                    try:
                        async with session.post(
                            f"{API_URL}/generate",
                            headers=self.headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            response_text = await response.text()
                            logging.info(f"API Response status: {response.status}, text: {response_text}")
                            
                            if response.status == 429:  # Rate limit
                                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
                                logging.warning(f"Rate limit reached. Waiting {retry_after} seconds...")
                                await asyncio.sleep(retry_after)
                                continue
                            elif response.status == 402:  # Payment required
                                raise Exception("API key has expired or payment is required")
                            elif response.status == 413:  # Payload too large
                                raise Exception("Изображение слишком большое. Пожалуйста, используйте изображение меньшего размера")
                            elif response.status != 200:
                                raise Exception(f"API request failed with status {response.status}: {response_text}")
                            
                            try:
                                response_data = json.loads(response_text)
                                logging.info(f"Parsed response data: {response_data}")
                            except json.JSONDecodeError:
                                raise Exception(f"Invalid JSON response: {response_text}")
                            
                            if response_data.get("status") == "error":
                                raise Exception(f"API error: {response_data.get('reason', 'Unknown error')}")
                            
                            break  # Success, exit retry loop
                    except aiohttp.ClientError as e:
                        if retry == MAX_RETRIES - 1:  # Last retry
                            raise Exception(f"Failed to connect to API after {MAX_RETRIES} retries: {str(e)}")
                        await asyncio.sleep(RETRY_DELAY)
                        continue

                if not response_data:
                    raise Exception("Failed to get response from API")

                task_id = response_data.get("request_id")
                if not task_id:
                    logging.error(f"No request_id in response. Full response: {response_data}")
                    raise Exception("No request_id in response")

                # Poll for task completion
                status_headers = {
                    "Authorization": f"Key {self.haiper_token}"
                }
                while datetime.now() < timeout_time:
                    async with session.get(
                        f"{API_URL}/status?request_id={task_id}",
                        headers=status_headers
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Status check failed: {await response.text()}")
                        
                        status_data = await response.json()
                        task_status = status_data.get("status")
                        #logging.info(f"Task {task_id} status: {task_status}")
                        
                        if task_status == "COMPLETED":
                            video_url = status_data.get('url')
                            if not video_url:
                                raise Exception("No video URL in completed task response")
                            return {"url": video_url}
                        elif task_status == "FAILED":
                            raise Exception(f"Task failed: {status_data.get('reason', 'Unknown error')}")
                        elif task_status in ["pending", "processing", "IN_QUEUE", "IN_PROGRESS"]:
                            await asyncio.sleep(STATUS_CHECK_INTERVAL)
                            logging.info(f"Task {task_id} status: {task_status}")
                        else:
                            raise Exception(f"Unknown task status: {task_status}")

                raise Exception("Task timed out")

        except Exception as e:
            logging.error(f"Error processing video task: {e}")
            raise

    def initialize(self, helper):
        """Инициализация плагина"""
        self.haiper_token = helper.api_key
        self.headers = {
            "Authorization": f"Bearer {self.haiper_token}",
            "Content-Type": "application/json",
            "X-Title": "tgBot"
        }
        self.status_headers = {
            "Authorization": f"Bearer {self.haiper_token}"
        }
        self.bot = helper.bot
        self.openai = helper

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """Execute plugin functionality"""
        self.initialize(helper)

        try:
            logging.info(f"haiper_image_to_video execute called with kwargs: {kwargs}")
            prompt = kwargs.get('prompt', "оживи картинку")
            chat_id = kwargs.get('chat_id')
            logging.info(f"animation prompt: {prompt}")

            # Получаем file_id из сохраненного изображения
            file_id = kwargs.get('image_path')
            if not file_id:
                raise ValueError("No image file_id provided")

            logging.info(f"Found image {file_id} for user {chat_id}")

            # Создаем новую задачу
            task = VideoTask(
                task_id=self.get_file_id_hash(file_id),
                user_id=chat_id,
                chat_id=chat_id,
                file_id=file_id,
                prompt=prompt,
                status=TaskStatus.PENDING
            )

            # Запускаем обработчик очереди, если он еще не запущен
            await self.start_worker()

            # Добавляем задачу в очередь
            await self.task_queue.put(task)

            return {"message": "Ваш запрос добавлен в очередь на обработку. Это может занять несколько минут."}

        except Exception as e:
            logging.error(f"Error in execute: {e}")
            raise

    async def handle_animate_command(self, update: Update, context) -> None:
        """Обработчик команды /animate"""
        if not update or not update.message:
            return
            
        message = update.message
        chat_id = message.chat.id
        user_id = message.from_user.id
        logging.info(f"handle_animate_command called with chat_id: {chat_id}, user_id: {user_id}")
        
        if not self.openai:
            await message.reply_text("Ошибка: не инициализирован OpenAI helper")
            return
        
        try:
            # Получаем промпт из сообщения (убираем команду /animate)
            prompt = message.text[8:].strip() if message.text else "оживи картинку"
            
            # Проверяем, является ли сообщение ответом на фотографию
            if message.reply_to_message and (message.reply_to_message.photo or 
                (message.reply_to_message.document and 
                 message.reply_to_message.document.mime_type.startswith('image/'))):
                # Берем file_id из фото или документа
                if message.reply_to_message.photo:
                    file_id = message.reply_to_message.photo[-1].file_id
                else:
                    file_id = message.reply_to_message.document.file_id
                    
                logging.info(f"Processing animation for replied photo with file_id: {file_id}")
                # Сразу обрабатываем фото с промптом
                await self._process_animate_command(message, file_id, prompt)
                return

            # Если это не ответ на фото, продолжаем стандартную логику
            # Получаем последние изображения пользователя (максимум 1)
            user_images = self.openai.db.get_user_images(user_id, chat_id, limit=1)
            if not user_images:
                await message.reply_text("Пожалуйста, сначала отправьте изображение, а затем используйте команду /animate")
                return

            # Если есть только одно изображение, используем его
            if len(user_images) == 1:
                await self._process_animate_command(message, user_images[0]['file_id'], prompt)
                return

            # Создаем клавиатуру с кнопками для выбора изображения
            keyboard = []
            for idx, img in enumerate(user_images, 1):
                created_at = datetime.fromisoformat(img['created_at'].replace('Z', '+00:00'))
                time_str = created_at.strftime("%H:%M:%S")
                keyboard.append([InlineKeyboardButton(
                    f"Изображение {idx} (загружено в {time_str})",
                    callback_data=f"animate_{img['file_id_hash']}"  # Используем хеш из БД
                )])

            reply_markup = InlineKeyboardMarkup(keyboard)
            await message.reply_text(
                "У вас несколько загруженных изображений. Выберите, какое использовать для создания видео:",
                reply_markup=reply_markup
            )

        except Exception as e:
            logging.error(f"Error in handle_animate_command: {e}")
            await message.reply_text(
                "❌ Произошла ошибка при обработке команды\n\n"
                f"Детали ошибки: {str(e)}\n\n"
                "Пожалуйста, попробуйте позже или обратитесь к администратору."
            )

    async def _process_animate_command(self, message, file_id, prompt=None):
        """Внутренний метод для обработки команды animate с конкретным изображением"""
        temp_file = None
        start_time = datetime.now()
        #status_message = await message.reply_text(
        #    "🎬 анимация  готова.\n"
        #    "🎯 Промпт: {prompt}"
        #)
        #return
        
        try:
            # Create and queue new task
            task = VideoTask(
                task_id=f"{message.chat.id}_{datetime.now().timestamp()}",
                user_id=message.from_user.id,
                chat_id=message.chat.id,
                file_id=file_id,
                prompt=prompt if prompt else "оживи картинку",
                status=TaskStatus.PENDING
            )

            await self.task_queue.put(task)
            self.active_tasks[task.task_id] = task

            # Ensure worker is running
            await self.start_worker()

            # Отправляем более информативное сообщение
            status_message = await message.reply_text(
                "🎬 Начинаю создание анимации...\n\n"
                f"🎯 Промпт: {task.prompt}\n"
                "⏳ Статус: в очереди на обработку\n\n"
                "Это может занять несколько минут. Я сообщу, когда анимация будет готова."
            )

            # Обновляем статус задачи каждые 30 секунд
            while True:
                # Получаем актуальный статус задачи из active_tasks
                current_task = self.active_tasks.get(task.task_id)
                if not current_task:
                    raise ValueError("Задача не найдена в списке активных задач")
                
                if current_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
                    
                await asyncio.sleep(30)
                elapsed_time = datetime.now() - start_time
                elapsed_minutes = elapsed_time.total_seconds() / 60

                if current_task.status == TaskStatus.PROCESSING:
                    await status_message.edit_text(
                        "🎬 Создаю анимацию...\n\n"
                        f"🎯 Промпт: {current_task.prompt}\n"
                        f"⏳ Статус: генерация видео\n"
                        f"⌛️ Прошло времени: {elapsed_minutes:.1f} мин.\n\n"
                        "Пожалуйста, подождите..."
                    )
                elif elapsed_minutes >= TIMEOUT_MINUTES:
                    current_task.status = TaskStatus.FAILED
                    current_task.error = "Превышено время ожидания"
                    break

            current_task = self.active_tasks.get(task.task_id)
            if current_task.status == TaskStatus.COMPLETED and current_task.result:
                video_url = current_task.result.get("url")
                if not video_url:
                    raise ValueError("URL видео не найден в ответе API")

                # Скачиваем видео во временный файл
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                async with aiohttp.ClientSession() as session:
                    async with session.get(video_url) as response:
                        if response.status != 200:
                            raise ValueError(f"Не удалось скачать видео. Статус: {response.status}")
                        temp_file.write(await response.read())
                temp_file.close()

                await status_message.edit_text(
                    "✅ Анимация успешно создана!\n\n"
                    f"🎯 Промпт: {current_task.prompt}\n"
                    f"⌛️ Время создания: {elapsed_minutes:.1f} мин."
                )
                # Отправляем видео из временного файла
                with open(temp_file.name, 'rb') as video_file:
                    await message.reply_video(
                        video=video_file,
                        caption=f"🎨 Анимация по промпту: {current_task.prompt}"
                    )
                
                # Удаляем задачу из списка активных
                self.active_tasks.pop(task.task_id, None)
                
            elif current_task.status == TaskStatus.FAILED:
                error_message = current_task.error or "Неизвестная ошибка"
                elapsed_time = datetime.now() - start_time
                elapsed_minutes = elapsed_time.total_seconds() / 60
                await status_message.edit_text(
                    "❌ Не удалось создать анимацию\n\n"
                    f"🎯 Промпт: {current_task.prompt}\n"
                    f"❗️ Ошибка: {error_message}\n"
                    f"⌛️ Прошло времени: {elapsed_minutes:.1f} мин."
                )
                
                # Удаляем задачу из списка активных
                self.active_tasks.pop(task.task_id, None)

        except Exception as e:
            elapsed_time = datetime.now() - start_time
            elapsed_minutes = elapsed_time.total_seconds() / 60
            logging.error(f"Error in _process_animate_command: {e}")
            await message.reply_text(
                "❌ Произошла ошибка при обработке команды\n\n"
                f"Детали ошибки: {str(e)}\n\n"
                f"⌛️ Прошло времени: {elapsed_minutes:.1f} мин.\n\n"
                "Пожалуйста, попробуйте позже или обратитесь к администратору."
            )
            
            # Удаляем задачу из списка активных в случае ошибки
            if task.task_id in self.active_tasks:
                self.active_tasks.pop(task.task_id, None)
                
        finally:
            # Удаляем временный файл, если он был создан
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logging.error(f"Error deleting temporary file: {e}")

    async def handle_animate_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs) -> None:
        """Обработчик команды /animate_help"""
        help_text = """
*Команда /animate - создание анимации из изображения*

*Использование:*
1. Отправьте изображение в чат
2. Используйте команду /animate [prompt]

*Параметры:*
- prompt (необязательно) - текст, описывающий желаемую анимацию
  Например: `/animate сделай плавное движение`
  Если не указан, будет использован промпт по умолчанию: "оживи картинку"
Вы можете ответить на фотографию, чтобы использовать её для анимации. Ответ должен быть в формате:
/animate [prompt]

*Конструктор анимации:*
/animate_prompt
Используется для создания анимации по заданному промпту. Выбранный промпт будет применен к последнему загруженному изображению.

*Примечания:*
- Обработка может занять несколько минут
- Изображение должно быть отправлено до использования команды
- Поддерживаются форматы: JPEG, PNG
"""
        await update.message.reply_text(escape_markdown(help_text), parse_mode=constants.ParseMode.MARKDOWN_V2)

    async def handle_photo_message(self, update: Update, context) -> None:
        """Обработчик сообщений с фотографиями"""
        try:
            message = update.message
            # Получаем caption сообщения
            caption = message.caption
            if caption and caption.lower().startswith('/animate'):
                # Если в caption есть команда /animate, обрабатываем её
                prompt = caption[8:].strip()  # Убираем '/animate ' из начала
                await self.handle_animate_command(update, context)
            else:
                # Если нет команды, показываем кнопку для анимации
                file_id = message.photo[-1].file_id
                # Получаем хеш из БД
                user_images = self.openai.db.get_user_images(
                    message.from_user.id,
                    str(message.chat.id),
                    limit=1
                )
                if user_images:
                    file_id_hash = user_images[0]['file_id_hash']
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton(
                            "🎬 Создать анимацию",
                            callback_data=f"animate_{file_id_hash}"
                        )]
                    ])
                    await message.reply_text(
                        "Хотите создать анимацию из этого изображения?",
                        reply_markup=keyboard
                    )
        except Exception as e:
            logger.error(f"Error handling photo message: {e}")

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик всех callback-запросов"""
        query = update.callback_query
        user_id = query.from_user.id
        
        try:
            if not query.data.startswith('haiper_'):
                return

            data = query.data.replace('haiper_', '')
            #logging.info(f"Получен callback_query: {data} от пользователя {user_id}")
            #logging.info(f"Текущие настройки пользователя: {self.user_settings.get(user_id, {})}")
            
            # Обработка закрытия меню
            if data == "close_menu":
                #logging.info("Закрытие меню конструктора")
                await query.answer("Меню закрыто")
                await query.message.delete()
                return
                
            # Сначала обрабатываем специфичные паттерны
            if data.startswith('style_'):
                style_id = data.replace('style_', '')
                if user_id not in self.user_settings:
                    self.user_settings[user_id] = {}
                self.user_settings[user_id]['style'] = style_id
                await query.answer(f"✅ Выбран стиль: {self.get_style_name(style_id)}")
                await self.show_main_menu_with_selections(query.message, user_id)
                return

            if data.startswith('effect_'):
                effect_id = data.replace('effect_', '')
                if user_id not in self.user_settings:
                    self.user_settings[user_id] = {}
                self.user_settings[user_id]['effect'] = effect_id
                await query.answer(f"✅ Выбран эффект: {self.get_effect_name(effect_id)}")
                await self.show_main_menu_with_selections(query.message, user_id)
                return

            if data.startswith('preset_'):
                preset_id = data.replace('preset_', '')
                if user_id not in self.user_settings:
                    self.user_settings[user_id] = {}
                self.user_settings[user_id]['preset'] = preset_id
                await query.answer(f"✅ Выбран пресет: {self.get_preset_name(preset_id)}")
                await self.show_main_menu_with_selections(query.message, user_id)
                return

            if data.startswith('animate_'):
                await self.handle_animate_button(query)
                return

            # Затем обрабатываем простые команды
            if data == "separator":
                await query.answer()
                return

            if data == "prompt_restart":
                logging.info("Обработка кнопки сброса")
                await query.answer("🔄 Настройки сброшены")
                if user_id in self.user_settings:
                    self.user_settings[user_id] = {}
                logging.info("Настройки пользователя сброшены")
                await self.show_main_menu_with_selections(query.message, user_id)
                return

            if data == "back_to_main":
                logging.info("Обработка кнопки 'Назад'")
                await query.answer()
                await self.show_main_menu_with_selections(query.message, user_id)
                return

            if data == "show_styles":
                await query.answer()
                await self.show_style_selection(query.message)
                return

            if data == "show_effects":
                await query.answer()
                await self.show_effects_selection(query.message)
                return

            if data == "show_presets":
                await query.answer()
                await self.show_presets_selection(query.message)
                return

            if data == "apply_settings":
                await self.apply_settings(query)
                return

            logging.warning(f"Получен неизвестный callback_data: {data}")
            await query.answer("⚠️ Неизвестная команда")

        except Exception as e:
            logging.error(f"Ошибка в handle_callback_query: {e}", exc_info=True)
            await query.message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже."
            )

    async def show_style_selection(self, message):
        """Показывает меню выбора стиля"""
        try:
            # Получаем текущие настройки пользователя
            user_id = message.chat.id
            settings = self.user_settings.get(user_id, {})
            current_style = settings.get('style')
            
            styles = [
                ("realistic", "🎬 Реалистичный", "плавные естественные движения"),
                ("cartoon", "🎨 Мультяшный", "игривая анимация"),
                ("artistic", "🖼 Художественный", "креативные эффекты"),
                ("cinematic", "🎥 Кино", "драматические движения"),
                ("abstract", "🌈 Абстрактный", "необычные переходы"),
                ("anime", "🎌 Аниме", "в стиле японской анимации"),
                ("pixel", "👾 Пиксельный", "ретро-игровая стилистика"),
                ("watercolor", "🎨 Акварельный", "нежные размытия"),
                ("neon", "💡 Неоновый", "яркие светящиеся эффекты"),
                ("vintage", "📷 Винтаж", "старое кино"),
                ("minimalist", "⚪️ Минималистичный", "простые чистые линии"),
                ("cyberpunk", "🤖 Киберпанк", "футуристический стиль"),
                ("comic", "💭 Комикс", "как в графических новеллах"),
                ("glitch", "⚡️ Глитч", "цифровые искажения"),
                ("surreal", "🎭 Сюрреализм", "необычные трансформации")
            ]
            
            keyboard = []
            for i in range(0, len(styles), 2):
                row = []
                for style_id, style_name, _ in styles[i:i+2]:
                    # Добавляем галочку к названию, если стиль выбран
                    button_text = f"{style_name} ✓" if style_id == current_style else style_name
                    row.append(InlineKeyboardButton(
                        button_text,
                        callback_data=f"haiper_style_{style_id}"
                    ))
                keyboard.append(row)
            
            keyboard.append([
                InlineKeyboardButton("⬅️ Назад", callback_data="haiper_back_to_main")
            ])
            
            style_descriptions = "\n".join(
                f"{name} - {desc}" for _, name, desc in styles
            )
            
            await message.edit_text(
                "*🎨 Выберите стиль анимации:*\n\n"
                f"{style_descriptions}",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except telegram.error.BadRequest as e:
            if "Message is not modified" not in str(e):
                logging.error(f"Ошибка при обновлении меню стилей: {e}")
                raise
            else:
                logging.info("Сообщение не было изменено (это нормально)")
        except Exception as e:
            logging.error(f"Неожиданная ошибка в show_style_selection: {e}")
            raise

    async def show_effects_selection(self, message):
        """Показывает меню выбора эффектов"""
        try:
            # Получаем текущие настройки пользователя
            user_id = message.chat.id
            settings = self.user_settings.get(user_id, {})
            current_effect = settings.get('effect')
            
            effects = [
                ("zoom", "🔍 Зум", "плавное приближение"),
                ("pan", "↔️ Панорама", "плавное движение"),
                ("rotate", "🔄 Поворот", "вращение"),
                ("morph", "🎭 Морфинг", "трансформация"),
                ("wave", "🌊 Волны", "волновой эффект"),
                ("blur", "🌫 Размытие", "плавное размытие"),
                ("shake", "📳 Тряска", "вибрация изображения"),
                ("glitch", "⚡️ Глитч", "цифровые помехи"),
                ("bounce", "🏀 Отскок", "пружинящее движение"),
                ("spiral", "🌀 Спираль", "спиральное вращение"),
                ("fade", "🌅 Затухание", "плавное исчезновение"),
                ("flash", "💫 Вспышка", "яркие вспышки"),
                ("mirror", "🪞 Зеркало", "зеркальные отражения"),
                ("ripple", "💧 Рябь", "эффект волн на воде"),
                ("swing", "🎭 Качание", "маятниковое движение"),
                ("float", "🎈 Парение", "невесомое движение"),
                ("pulse", "💓 Пульсация", "ритмичные изменения"),
                ("scatter", "✨ Рассеивание", "разлетающиеся частицы"),
                ("stretch", "↔️ Растяжение", "эластичная деформация"),
                ("fold", "📄 Сворачивание", "эффект складывания"),
                ("kaleidoscope", "🎨 Калейдоскоп", "зеркальные узоры"),
                ("pixelate", "🔲 Пикселизация", "эффект пикселей"),
                ("dissolve", "💨 Растворение", "постепенное исчезновение"),
                ("shatter", "💔 Разбивание", "эффект осколков"),
                ("neon", "💡 Неон", "светящиеся контуры")
            ]
            
            keyboard = []
            for i in range(0, len(effects), 2):
                row = []
                for effect_id, effect_name, _ in effects[i:i+2]:
                    # Добавляем галочку к названию, если эффект выбран
                    button_text = f"{effect_name} ✓" if effect_id == current_effect else effect_name
                    row.append(InlineKeyboardButton(
                        button_text,
                        callback_data=f"haiper_effect_{effect_id}"
                    ))
                keyboard.append(row)
            
            keyboard.append([
                InlineKeyboardButton("⬅️ Назад", callback_data="haiper_back_to_main")
            ])
            
            effect_descriptions = "\n".join(
                f"{name} - {desc}" for _, name, desc in effects
            )
            
            await message.edit_text(
                "*✨ Выберите эффект:*\n\n"
                f"{effect_descriptions}",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except telegram.error.BadRequest as e:
            if "Message is not modified" not in str(e):
                raise

    async def show_presets_selection(self, message):
        """Показывает меню выбора пресетов"""
        try:
            # Получаем текущие настройки пользователя
            user_id = message.chat.id
            settings = self.user_settings.get(user_id, {})
            current_preset = settings.get('preset')
            
            presets = [
                ("art", "🎨 Художественный", "креативная анимация"),
                ("realistic", "🎬 Реалистичный", "естественное движение"),
                ("wave", "🌊 Волновой", "эффект течения"),
                ("emotion", "🎭 Эмоциональный", "передача настроения"),
                ("nature", "🌿 Природный", "органическое движение"),
                ("tech", "🤖 Технологичный", "футуристические эффекты"),
                ("magic", "✨ Магический", "волшебные трансформации"),
                ("retro", "📺 Ретро", "винтажная анимация"),
                ("cosmic", "🌌 Космический", "космические эффекты"),
                ("dream", "💫 Сновидение", "сюрреалистичные переходы"),
                ("dynamic", "⚡️ Динамичный", "энергичное движение"),
                ("gentle", "🍃 Нежный", "мягкие переходы"),
                ("horror", "👻 Хоррор", "жуткие трансформации"),
                ("party", "🎉 Праздничный", "весёлая анимация"),
                ("romantic", "💝 Романтичный", "нежные переливы"),
                ("sport", "🏃 Спортивный", "динамичные движения"),
                ("fantasy", "🐉 Фэнтези", "магические превращения"),
                ("steampunk", "⚙️ Стимпанк", "механические движения"),
                ("underwater", "🐠 Подводный", "плавные течения"),
                ("fire", "🔥 Огненный", "пламенные эффекты"),
                ("winter", "❄️ Зимний", "морозные узоры"),
                ("space", "🚀 Космический", "межзвёздные эффекты"),
                ("rainbow", "🌈 Радужный", "яркие переливы"),
                ("matrix", "💻 Матрица", "цифровой дождь")
            ]
            
            keyboard = []
            for i in range(0, len(presets), 2):
                row = []
                for preset_id, preset_name, _ in presets[i:i+2]:
                    # Добавляем галочку к названию, если пресет выбран
                    button_text = f"{preset_name} ✓" if preset_id == current_preset else preset_name
                    callback_data = f"haiper_preset_{preset_id}"
                    row.append(InlineKeyboardButton(
                        button_text,
                        callback_data=callback_data
                    ))
                keyboard.append(row)
            
            keyboard.append([
                InlineKeyboardButton("⬅️ Назад", callback_data="haiper_back_to_main")
            ])
            
            preset_descriptions = "\n".join(
                f"{name} - {desc}" for _, name, desc in presets
            )
            
            await message.edit_text(
                "*🎬 Выберите пресет:*\n\n"
                f"{preset_descriptions}",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except telegram.error.BadRequest as e:
            if "Message is not modified" not in str(e):
                logging.error(f"Ошибка при обновлении меню пресетов: {e}")
                raise
            else:
                logging.info("Сообщение не было изменено (это нормально)")
        except Exception as e:
            logging.error(f"Неожиданная ошибка в show_presets_selection: {e}")
            raise

    def get_preset_name(self, preset_id: str) -> str:
        """Возвращает читаемое название пресета"""
        presets = {
            "art": "🎨 Художественный",
            "realistic": "🎬 Реалистичный",
            "wave": "🌊 Волновой",
            "emotion": "🎭 Эмоциональный",
            "nature": "🌿 Природный",
            "tech": "🤖 Технологичный",
            "magic": "✨ Магический",
            "retro": "📺 Ретро",
            "cosmic": "🌌 Космический",
            "dream": "💫 Сновидение",
            "dynamic": "⚡️ Динамичный",
            "gentle": "🍃 Нежный",
            "horror": "👻 Хоррор",
            "party": "🎉 Праздничный",
            "romantic": "💝 Романтичный",
            "sport": "🏃 Спортивный",
            "fantasy": "🐉 Фэнтези",
            "steampunk": "⚙️ Стимпанк",
            "underwater": "🐠 Подводный",
            "fire": "🔥 Огненный",
            "winter": "❄️ Зимний",
            "space": "🚀 Космический",
            "rainbow": "🌈 Радужный",
            "matrix": "💻 Матрица"
        }
        return presets.get(preset_id, preset_id)

    async def apply_settings(self, query):
        """Применяет выбранные настройки и запрашивает дополнительный текст для промпта"""
        user_id = query.from_user.id
        logger.info(f"[ConversationHandler] apply_settings начало: user_id={user_id}")
        logger.info(f"Вызван apply_settings для пользователя {user_id}")
        
        settings = self.user_settings.get(user_id, {})
        logger.info(f"Текущие настройки пользователя {user_id}: {settings}")
        
        if not settings:
            logger.warning(f"Настройки не найдены для пользователя {user_id}")
            await query.answer("⚠️ Нет выбранных настроек")
            return
            
        # Получаем последнее изображение
        user_images = self.openai.db.get_user_images(
            user_id,
            str(query.message.chat.id),
            limit=1
        )
        
        if not user_images:
            logger.warning(f"Изображения не найдены для пользователя {user_id}")
            await query.message.reply_text(
                "⚠️ Не найдено изображение для анимации. Отправьте изображение заново."
            )
            return
            
        # Формируем базовый промпт на основе настроек
        prompt_parts = []
        if settings.get('style'):
            prompt_parts.append(f"стиль: {settings['style']}")
        if settings.get('effect'):
            prompt_parts.append(f"эффект: {settings['effect']}")
        if settings.get('preset'):
            prompt_parts.append(f"пресет: {settings['preset']}")
            
        base_prompt = f"создай анимацию с параметрами: {', '.join(prompt_parts)}"
        logger.info(f"Сформирован базовый промпт для пользователя {user_id}: {base_prompt}")
        
        # Сохраняем базовый промпт и file_id в настройках пользователя
        settings['base_prompt'] = base_prompt
        settings['file_id'] = user_images[0]['file_id']
        self.user_settings[user_id] = settings
        logger.info(f"Обновлены настройки пользователя {user_id}: {settings}")
        
        # Закрываем меню конструктора
        await query.message.delete()
        
        # Отправляем сообщение с запросом дополнительного текста
        sent_message = await query.message.reply_text(
            "💭 Промпт сгенерирован\n\n"
            f"Чтобы его применить, отправьте в ответ на фото `/animate {base_prompt}`\n"
            "Вы можете добавить дополнительные параметры в промпт, чтобы получить более точную анимацию.\n"
            f"Или просто отправьте `/animate {base_prompt}`, будет обработано последнее изображение в чате.",
            parse_mode='Markdown'
        )
        
        logger.info(f"[ConversationHandler] apply_settings возвращает WAITING_PROMPT для user_id={user_id}")
        return WAITING_PROMPT

    async def handle_animate_button(self, query):
        """Обработка кнопки создания анимации"""
        try:
            parts = query.data.split('_')
            file_id_hash = parts[1]
            
            # Получаем file_id из БД по хешу
            user_images = self.openai.db.get_user_images(
                query.from_user.id,
                str(query.message.chat.id),
                limit=5
            )
            
            file_id = None
            for img in user_images:
                if img['file_id_hash'] == file_id_hash:
                    file_id = img['file_id']
                    break
                    
            if not file_id:
                await query.message.reply_text(
                    "⚠️ Не удалось найти изображение. Отправьте его заново."
                )
                return
                
            await self._process_animate_command(query.message, file_id)
            
        except Exception as e:
            logging.error(f"Error in handle_animate_button: {e}")
            await query.message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже."
            )

    def generate_prompt(self, params: Dict) -> str:
        """Генерирует промпт на основе выбранных параметров"""
        effect = params.get('effect', '')
        
        prompts = {
            'zoom': 'создай плавное приближение, сохраняя детали изображения',
            'pan': 'сделай плавное панорамирование по изображению',
            'rotate': 'добавь плавное вращение, сохраняя композицию',
            'morph': 'создай плавную трансформацию элементов',
            'wave': 'добавь волновой эффект, создающий ощущение движения'
        }
        
        return prompts.get(effect, 'оживи картинку')

    async def handle_inline_query(self, inline_query):
        """Обработчик inline-запросов"""
        query = inline_query.query.strip()
        
        # Предлагаем варианты промптов
        results = []
        
        # Добавляем базовые эффекты
        effects = {
            "zoom": "Плавное приближение",
            "pan": "Панорамирование",
            "rotate": "Вращение",
            "morph": "Морфинг",
            "wave": "Волновой эффект"
        }
        
        for effect_id, effect_name in effects.items():
            if not query or query.lower() in effect_name.lower():
                prompt = self.generate_prompt({"effect": effect_id})
                results.append(
                    InlineQueryResultArticle(
                        id=f"effect_{effect_id}",
                        title=effect_name,
                        description=f"Промпт: {prompt}",
                        input_message_content=InputTextMessageContent(
                            message_text=f"/animate {prompt}"
                        )
                    )
                )
        
        # Если есть конкретный запрос, добавляем его как отдельный вариант
        if query and len(query) >= 3:
            results.insert(0, InlineQueryResultArticle(
                id="custom",
                title="Использовать ваш текст",
                description=f"Промпт: {query}",
                input_message_content=InputTextMessageContent(
                    message_text=f"/animate {query}"
                )
            ))
        
        await inline_query.answer(
            results,
            cache_time=300,
            is_personal=True
        )

    async def handle_prompt_constructor(self, function_name: str, openai, update: Update = None, **kwargs) -> None:
        """
        Обработчик для конструктора промптов
        """
        try:
            user_id = update.message.from_user.id if update and update.message else None
            logger.info(f"[ConversationHandler] handle_prompt_constructor вызван: user_id={user_id}")
            logger.info("Вызван handle_prompt_constructor")
            if not update or not update.message:
                logger.error("update или message отсутствуют")
                return
                
            message = update.message
            chat_id = str(message.chat.id)
            user_id = message.from_user.id
            logger.info(f"Конструктор промптов запущен для пользователя {user_id} в чате {chat_id}")
            
            self.openai = openai
            self.bot = openai.bot
            
            # Проверяем наличие изображений
            user_images = self.openai.db.get_user_images(user_id, chat_id, limit=1)
            if not user_images:
                logger.warning(f"Изображения не найдены для пользователя {user_id}")
                await message.reply_text(
                    "⚠️ Сначала отправьте изображение, а затем используйте конструктор анимации"
                )
                return

            logger.info(f"Найдены изображения для пользователя {user_id}")
            # Создаем компактное меню с эмодзи для визуального разделения
            settings = self.user_settings.get(user_id, {})
            logger.info(f"Текущие настройки пользователя {user_id}: {settings}")
            
            # Добавляем отметки к названиям кнопок, если что-то выбрано
            style_text = "🎨 Стиль ✓" if settings.get('style') else "🎨 Стиль"
            effect_text = "✨ Эффекты ✓" if settings.get('effect') else "✨ Эффекты"
            preset_text = "🎬 Пресеты ✓" if settings.get('preset') else "🎬 Пресеты"
            
            keyboard = [
                [
                    InlineKeyboardButton(style_text, callback_data="haiper_show_styles"),
                    InlineKeyboardButton(effect_text, callback_data="haiper_show_effects")
                ],
                [
                    InlineKeyboardButton(preset_text, callback_data="haiper_show_presets"),
                    InlineKeyboardButton("🔄 Сброс", callback_data="haiper_prompt_restart")
                ]
            ]
            
            # Добавляем кнопку закрытия
            keyboard.append([
                InlineKeyboardButton("❌ Закрыть", callback_data="haiper_close_menu")
            ])
            
            # Формируем текст с выбранными настройками
            menu_text = "*🎬 Конструктор анимации*\n\n"
            if settings:
                menu_text += "*Выбранные настройки:*\n"
                if settings.get('style'):
                    style_name = self.get_style_name(settings['style'])
                    menu_text += f"• Стиль: {style_name}\n"
                if settings.get('effect'):
                    effect_name = self.get_effect_name(settings['effect'])
                    menu_text += f"• Эффект: {effect_name}\n"
                if settings.get('preset'):
                    preset_name = self.get_preset_name(settings['preset'])
                    menu_text += f"• Пресет: {preset_name}\n"
                menu_text += "\n"
            
            menu_text += "Выберите тип анимации:\n\n"
            menu_text += "🎨 *Стиль* - реалистичный, мультяшный, художественный\n"
            menu_text += "✨ *Эффекты* - зум, панорама, вращение\n"
            menu_text += "🎬 *Пресеты* - готовые настройки"
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await message.reply_text(
                menu_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logging.error(f"Error in handle_prompt_constructor: {e}")
            if update and update.message:
                await update.message.reply_text(
                    "❌ Произошла ошибка. Попробуйте позже."
                )
    
    def get_file_id_hash(self, file_id: str) -> str:
        """Генерирует короткий хеш для file_id"""
        hash_object = hashlib.md5(file_id.encode())
        return hash_object.hexdigest()[:8]  # Берем первые 8 символов хеша
    
    async def show_main_menu_with_selections(self, message, user_id: int):
        """Показывает главное меню с отметками о выбранных настройках"""
        try:
            settings = self.user_settings.get(user_id, {})
            logging.info(f"Текущие настройки пользователя: {settings}")
            
            # Формируем текст с выбранными настройками
            menu_text = "*🎬 Конструктор анимации*\n\n"
            if settings:
                menu_text += "*Выбранные настройки:*\n"
                if settings.get('style'):
                    style_name = self.get_style_name(settings['style'])
                    menu_text += f"• Стиль: {style_name}\n"
                if settings.get('effect'):
                    effect_name = self.get_effect_name(settings['effect'])
                    menu_text += f"• Эффект: {effect_name}\n"
                if settings.get('preset'):
                    preset_name = self.get_preset_name(settings['preset'])
                    menu_text += f"• Пресет: {preset_name}\n"
                menu_text += "\n"
            
            # Добавляем отметки к названиям кнопок, если что-то выбрано
            style_text = "🎨 Стиль ✓" if settings.get('style') else "🎨 Стиль"
            effect_text = "✨ Эффекты ✓" if settings.get('effect') else "✨ Эффекты"
            preset_text = "🎬 Пресеты ✓" if settings.get('preset') else "🎬 Пресеты"
            
            keyboard = [
                [
                    InlineKeyboardButton(style_text, callback_data="haiper_show_styles"),
                    InlineKeyboardButton(effect_text, callback_data="haiper_show_effects")
                ],
                [
                    InlineKeyboardButton(preset_text, callback_data="haiper_show_presets"),
                    InlineKeyboardButton("🔄 Сброс", callback_data="haiper_prompt_restart")
                ]
            ]
            
            # Добавляем кнопку применения, если есть выбранные настройки
            if settings:
                keyboard.append([
                    InlineKeyboardButton("✅ Применить настройки", callback_data="haiper_apply_settings")
                ])
            
            # Добавляем кнопку закрытия
            keyboard.append([
                InlineKeyboardButton("❌ Закрыть", callback_data="haiper_close_menu")
            ])
            
            menu_text += "Выберите тип анимации:\n\n"
            menu_text += "🎨 *Стиль* - реалистичный, мультяшный, художественный\n"
            menu_text += "✨ *Эффекты* - зум, панорама, вращение\n"
            menu_text += "🎬 *Пресеты* - готовые настройки"
            
            await message.edit_text(
                menu_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except telegram.error.BadRequest as e:
            if "Message is not modified" not in str(e):
                logging.error(f"Ошибка при обновлении главного меню: {e}")
                raise
            else:
                logging.info("Сообщение не было изменено (это нормально)")
        except Exception as e:
            logging.error(f"Неожиданная ошибка в show_main_menu_with_selections: {e}")
            raise

    def get_style_name(self, style_id: str) -> str:
        """Возвращает читаемое название стиля"""
        styles = {
            "realistic": "🎬 Реалистичный",
            "cartoon": "🎨 Мультяшный",
            "artistic": "🖼 Художественный",
            "cinematic": "🎥 Кино",
            "abstract": "🌈 Абстрактный",
            "anime": "🎌 Аниме",
            "pixel": "👾 Пиксельный",
            "watercolor": "🎨 Акварельный",
            "neon": "💡 Неоновый",
            "vintage": "📷 Винтаж",
            "minimalist": "⚪️ Минималистичный",
            "cyberpunk": "🤖 Киберпанк",
            "comic": "💭 Комикс",
            "glitch": "⚡️ Глитч",
            "surreal": "🎭 Сюрреализм"
        }
        return styles.get(style_id, style_id)

    def get_effect_name(self, effect_id: str) -> str:
        """Возвращает читаемое название эффекта"""
        effects = {
            "zoom": "🔍 Зум",
            "pan": "↔️ Панорама",
            "rotate": "🔄 Поворот",
            "morph": "🎭 Морфинг",
            "wave": "🌊 Волны",
            "blur": "🌫 Размытие",
            "shake": "📳 Тряска",
            "glitch": "⚡️ Глитч",
            "bounce": "🏀 Отскок",
            "spiral": "🌀 Спираль",
            "fade": "🌅 Затухание",
            "flash": "💫 Вспышка",
            "mirror": "🪞 Зеркало",
            "ripple": "💧 Рябь",
            "swing": "🎭 Качание",
            "float": "🎈 Парение",
            "pulse": "💓 Пульсация",
            "scatter": "✨ Рассеивание",
            "stretch": "↔️ Растяжение",
            "fold": "📄 Сворачивание",
            "kaleidoscope": "🎨 Калейдоскоп",
            "pixelate": "🔲 Пикселизация",
            "dissolve": "💨 Растворение",
            "shatter": "💔 Разбивание",
            "neon": "💡 Неон"
        }
        return effects.get(effect_id, effect_id)
    
    async def handle_prompt_reply(self, update: Update, context: CallbackContext) -> int:
        """Обработчик ответа на запрос дополнительного текста промпта"""
        user_id = update.message.from_user.id
        logger.info(f"[ConversationHandler] handle_prompt_reply вызван: user_id={user_id}")
        logger.info(f"[ConversationHandler] Текст сообщения: {update.message.text}")
        
        # Проверяем, что это ответ на сообщение от нашего бота
        if not update.message.reply_to_message:
            logger.warning("[ConversationHandler] Ответ не является ответом на сообщение бота")
            return
            
        # Проверяем, что это ответ на правильное сообщение
        reply_text = update.message.reply_to_message.text
        logger.info(f"[ConversationHandler] Текст reply_to_message: {reply_text}")
        
        if not reply_text or "Хотите добавить что-то к промпту?" not in reply_text:
            logger.warning("[ConversationHandler] Ответ не на правильное сообщение")
            return

        settings = self.user_settings.get(user_id, {})

        if not settings or not settings.get('file_id'):
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте заново через /animate_prompt")
            logger.error("Настройки пользователя не найдены или отсутствует file_id")
            return ConversationHandler.END

        # Проверяем ответ пользователя
        if update.message.text.lower() == "нет":
            prompt = settings.get('base_prompt', "оживи картинку")
        else:
            base_prompt = settings.get('base_prompt', "оживи картинку")
            prompt = f"{base_prompt}, {update.message.text}"

        # Запускаем анимацию
        if settings.get('file_id'):
            await self._process_animate_command(update.message, settings['file_id'], prompt)
            logger.info(f"Анимация запущена с промптом: {prompt}")
        else:
            logger.error("file_id отсутствует в настройках пользователя")
            await update.message.reply_text("⚠️ Не удалось запустить анимацию. Попробуйте заново.")

        # Очищаем настройки после применения
        if user_id in self.user_settings:
            del self.user_settings[user_id]
            logger.info("Настройки пользователя очищены после применения")

        return ConversationHandler.END

    async def cancel_prompt(self, update: Update, context: CallbackContext) -> int:
        """Отменяет текущий диалог"""
        user_id = update.message.from_user.id
        logger.info(f"[ConversationHandler] cancel_prompt вызван: user_id={user_id}")
        if user_id in self.user_settings:
            del self.user_settings[user_id]
            
        await update.message.reply_text(
            "Конструктор промптов отменен. Можете начать заново с помощью /animate_prompt"
        )
        return ConversationHandler.END
    