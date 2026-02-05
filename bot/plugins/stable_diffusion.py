import os
import requests
import numpy as np
import cv2
import string
import random
import logging
import asyncio
import tempfile
import json
import io
from PIL import Image
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from .plugin import Plugin
from huggingface_hub import InferenceClient

# Настройка логирования
logger = logging.getLogger(__name__)

# Конфигурация генерации
#MODEL = "black-forest-labs/FLUX.1-dev"  # Модель по умолчанию
MODEL = "HiDream-ai/HiDream-I1-Full"
MAX_RETRIES = 5  # Максимальное количество попыток
RETRY_DELAY = 3  # Задержка между попытками в секундах
STATUS_CHECK_INTERVAL = 10  # Интервал проверки статуса в секундах
TIMEOUT_MINUTES = 5  # Таймаут для генерации изображения

class TaskStatus(Enum):
    """
    Статусы задачи генерации изображения
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ImageTask:
    """
    Задача генерации изображения
    """
    task_id: str
    user_id: int
    chat_id: int
    prompt: str
    status: TaskStatus
    result: Optional[Dict] = None
    error: Optional[str] = None

class StableDiffusionPlugin(Plugin):
    """
    Плагин для генерации изображений с использованием Hugging Face Inference API
    """

    def __init__(self):
        self.hf_token = os.getenv('STABLE_DIFFUSION_TOKEN')
        if not self.hf_token:
            raise ValueError('HUGGINGFACE_TOKEN environment variable must be set to use StableDiffusionPlugin')
        
        # Инициализация клиента Hugging Face
        self.client = InferenceClient(
            provider="fal-ai",
            api_key=self.hf_token,
        )
        
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.worker_task = None
        self.bot = None
        self.openai = None

    def get_source_name(self) -> str:
        return "Image Generator"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "stable_diffusion",
            "description": "Generate an image from a textual prompt",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text prompt for generating the image"}
                },
                "required": ["prompt"],
            },
        }]

    async def start_worker(self):
        """Запускает обработчик очереди задач с улучшенным контролем"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._process_queue())
            self.worker_task.add_done_callback(self._handle_worker_done)

    def _handle_worker_done(self, task):
        """Обработчик завершения воркера"""
        try:
            task.result()  # Проверка на наличие необработанных исключений
        except Exception as e:
            logger.error(f"Worker task completed with error: {e}")
            # Можно добавить логику перезапуска

    async def stop_worker(self):
        """Мягко останавливает обработчик очереди"""
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                logger.info("Worker task was successfully cancelled")

    def is_queue_empty(self) -> bool:
        """Проверяет, пуста ли очередь задач"""
        return self.task_queue.empty()

    async def _process_queue(self):
        """Обработчик очереди задач с улучшенной обработкой ошибок"""
        try:
            while True:
                try:
                    # Используем get_nowait() вместо wait_for()
                    try:
                        task = self.task_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        # Если очередь пуста, ждем некоторое время
                        await asyncio.sleep(10)  # Уменьшаем частоту логирования
                        continue

                    if task.status != TaskStatus.PENDING:
                        self.task_queue.task_done()
                        continue

                    task.status = TaskStatus.PROCESSING
                    try:
                        result = await self._process_image_task(task)
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        logger.error(f"Error processing task {task.task_id}: {e}", exc_info=True)
                    
                    self.task_queue.task_done()
                
                except Exception as inner_e:
                    logger.error(f"Unexpected error in queue processing: {inner_e}", exc_info=True)
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            logger.info("Queue processing was cancelled")
        except Exception as e:
            logger.error(f"Critical error in queue processing: {e}", exc_info=True)
        finally:
            logger.info("Queue processing has stopped")

    async def _process_image_task(self, task: ImageTask) -> Dict:
        """Обработка задачи генерации изображения"""
        start_time = datetime.now()
        
        for attempt in range(MAX_RETRIES):
            try:
                # Генерация изображения через Hugging Face API
                image = await self._generate_image(task.prompt)
                
                # Сохранение изображения
                output_dir = os.path.join(tempfile.gettempdir(), 'image_generation')
                os.makedirs(output_dir, exist_ok=True)
                image_file_path = os.path.join(output_dir, f"{self.generate_random_string()}.png")

                # Сохранение изображения
                image.save(image_file_path, format="PNG")

                return {
                    "path": image_file_path,
                    "prompt": task.prompt
                }

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to generate image after {MAX_RETRIES} attempts: {str(e)}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    async def _generate_image(self, prompt: str) -> Image.Image:
        """Генерирует изображение с использованием Hugging Face Inference API"""
        loop = asyncio.get_event_loop()
        
        def query():
            # Используем InferenceClient для генерации изображения
            return self.client.text_to_image(
                prompt,
                model=MODEL,
            )
        
        # Выполняем блокирующий запрос в отдельном потоке
        return await loop.run_in_executor(None, query)

    @staticmethod
    def generate_random_string(length: int = 15) -> str:
        """Генерирует случайную строку"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """Выполняет генерацию изображения"""
        try:
            self.openai = helper
            self.bot = helper.bot

            prompt = kwargs.get("prompt")
            if not prompt:
                return {"result": "Error: Prompt is required."}

            chat_id = kwargs.get("chat_id")
            if not chat_id:
                return {"result": "Error: Chat ID is required."}

            user_id = kwargs.get('user_id', helper.user_id)
            logger.info(f"Generating image for chat_id: {chat_id}, user_id: {user_id}, prompt: {prompt}")

            # Проверяем существование чата
            try:
                chat = await self.bot.get_chat(chat_id)
                logger.info(f"Successfully verified chat existence: {chat.id}")
            except Exception as e:
                logger.error(f"Error verifying chat existence: {e}")
                return {"result": f"Error: Cannot access chat {chat_id}: {str(e)}"}

            # Создание новой задачи
            task = ImageTask(
                task_id=f"{chat_id}_{datetime.now().timestamp()}",
                user_id=user_id,
                chat_id=int(chat_id),  # Преобразуем в int для корректной работы с Telegram API
                prompt=prompt,
                status=TaskStatus.PENDING
            )

            # Добавление задачи в очередь
            await self.task_queue.put(task)
            self.active_tasks[task.task_id] = task
            logger.info(f"Added task to queue: {task.task_id}")

            # Запуск обработчика
            await self.start_worker()

            # Запускаем мониторинг статуса в отдельной задаче
            asyncio.create_task(self._monitor_task_status(task))

            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t("sd_queue_added"),
                }
            }

        except Exception as e:
            logger.error(f"Unexpected error in execute: {e}", exc_info=True)
            if 'task' in locals() and task.task_id in self.active_tasks:
                self.active_tasks.pop(task.task_id, None)
            return {"result": f"Error: {str(e)}", "traceback": str(e.__traceback__)}

    async def _monitor_task_status(self, task: ImageTask):
        """Асинхронный мониторинг статуса задачи"""
        try:
            # Отправка начального сообщения
            try:
                status_message = await self.bot.send_message(
                    chat_id=task.chat_id,
                    text=self.t(
                        "sd_start_message",
                        prompt=task.prompt
                    )
                )
                if status_message:
                    logger.info(f"Sent initial status message: {status_message.message_id}")
                else:
                    logger.error("send_message returned None")
                    self.active_tasks.pop(task.task_id, None)
                    return
            except Exception as e:
                logger.error(f"Error sending initial message: {e}", exc_info=True)
                self.active_tasks.pop(task.task_id, None)
                return

            # Мониторинг статуса задачи
            start_time = datetime.now()
            while True:
                current_task = self.active_tasks.get(task.task_id)
                if not current_task:
                    logger.error("Task not found in active tasks")
                    return

                if current_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break

                await asyncio.sleep(STATUS_CHECK_INTERVAL)
                elapsed_time = datetime.now() - start_time
                elapsed_minutes = elapsed_time.total_seconds() / 60

                if current_task.status == TaskStatus.PROCESSING:
                    try:
                        await status_message.edit_text(
                            self.t(
                                "sd_processing_message",
                                prompt=current_task.prompt,
                                elapsed_minutes=f"{elapsed_minutes:.1f}"
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Could not update status message: {e}")

                elif elapsed_minutes >= TIMEOUT_MINUTES:
                    current_task.status = TaskStatus.FAILED
                    current_task.error = self.t("sd_timeout_error")
                    break

            current_task = self.active_tasks.get(task.task_id)
            if current_task.status == TaskStatus.COMPLETED and current_task.result:
                image_path = current_task.result.get("path")
                if not image_path:
                    raise ValueError("Путь к изображению не найден в результате")

                try:
                    await status_message.edit_text(
                        self.t(
                            "sd_completed_message",
                            prompt=current_task.prompt,
                            elapsed_minutes=f"{elapsed_minutes:.1f}"
                        )
                    )

                    # Отправка изображения
                    with open(image_path, 'rb') as image_file:
                        await self.bot.send_photo(
                            chat_id=task.chat_id,
                            photo=image_file,
                            caption=self.t(
                                "sd_caption",
                                prompt=self._truncate_prompt(current_task.prompt)
                            )
                        )
                except Exception as e:
                    logger.error(f"Error sending result: {e}")

                finally:
                    # Удаление временного файла
                    try:
                        if os.path.exists(image_path):
                            os.unlink(image_path)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file: {e}")

            elif current_task.status == TaskStatus.FAILED:
                error_message = current_task.error or self.t("sd_unknown_error")
                try:
                    await status_message.edit_text(
                        self.t(
                            "sd_failed_message",
                            prompt=current_task.prompt,
                            error=error_message,
                            elapsed_minutes=f"{elapsed_minutes:.1f}"
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not update error message: {e}")

            # Удаление задачи из списка активных
            self.active_tasks.pop(task.task_id, None)

        except Exception as e:
            logger.error(f"Error in task monitoring: {e}", exc_info=True)
            try:
                self.active_tasks.pop(task.task_id, None)
            except:
                pass

    @staticmethod
    def _truncate_prompt(prompt: str, max_length: int = 100) -> str:
        """
        Усекает промпт до указанной максимальной длины
        
        Args:
            prompt (str): Исходный промпт
            max_length (int): Максимальная длина промпта
        
        Returns:
            str: Усеченный промпт
        """
        if len(prompt) <= max_length:
            return prompt
        return prompt[:max_length] + "..."
