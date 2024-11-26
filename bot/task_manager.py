#task_manager.py
import asyncio
import logging
from typing import Dict, Any
from collections import defaultdict

from utils import handle_direct_result

class GenerationTaskManager:
    def __init__(self):
        # Словарь для хранения задач генерации
        self._tasks: Dict[str, Dict[str, Any]] = {}
        # Словарь для отслеживания активных задач пользователей
        self._user_tasks = defaultdict(set)
        # Семафор для ограничения одновременных задач на пользователя
        self._user_semaphores = defaultdict(lambda: asyncio.Semaphore(3))
        # Глобальный семафор для ограничения общего количества задач
        self._global_semaphore = asyncio.Semaphore(10)
        
    async def create_task(self, task_id: str, generation_coroutine, context: Dict):
        """
        Создает новую задачу генерации с учетом ограничений пользователя
        
        :param task_id: Уникальный идентификатор задачи
        :param generation_coroutine: Корутина для генерации
        :param context: Контекст для последующей обработки результата
        """
        try:
            user_id = context.get('user_id')
            
            # Получаем семафор для пользователя
            user_sem = self._user_semaphores[user_id]
            
            async with self._global_semaphore:
                async with user_sem:
                    # Create the coroutine if it's a function
                    if callable(generation_coroutine) and not asyncio.iscoroutine(generation_coroutine):
                        generation_task = generation_coroutine()
                    else:
                        generation_task = generation_coroutine
                        
                    # Создаем Future для асинхронного ожидания результата
                    future = asyncio.create_task(generation_task)
                    
                    # Сохраняем задачу в словарях
                    self._tasks[task_id] = {
                        'future': future,
                        'context': context,
                        'created_at': asyncio.get_event_loop().time(),
                        'user_id': user_id
                    }
                    self._user_tasks[user_id].add(task_id)
                    
                    return task_id
        
        except Exception as e:
            logging.error(f"Error creating generation task: {e}")
            return None
        
    async def get_task_result(self, task_id: str):
        """
        Получает результат задачи генерации и очищает ресурсы
        
        :param task_id: Идентификатор задачи
        :return: Результат задачи или None
        """
        if task_id not in self._tasks:
            logging.warning(f"Task {task_id} not found")
            return None
        
        try:
            task_data = self._tasks[task_id]
            result = await task_data['future']
            
            # Очищаем задачу из всех словарей
            user_id = task_data['user_id']
            self._user_tasks[user_id].remove(task_id)
            if not self._user_tasks[user_id]:
                del self._user_tasks[user_id]
            
            del self._tasks[task_id]
            
            return result
        
        except Exception as e:
            logging.error(f"Error getting task result for {task_id}: {e}")
            self._cleanup_task(task_id)
            return None
    
    def _cleanup_task(self, task_id: str):
        """
        Очищает все ресурсы, связанные с задачей
        """
        if task_id in self._tasks:
            task_data = self._tasks[task_id]
            user_id = task_data['user_id']
            
            # Отменяем future если он еще не завершен
            if not task_data['future'].done():
                task_data['future'].cancel()
            
            # Очищаем из всех словарей
            self._user_tasks[user_id].remove(task_id)
            if not self._user_tasks[user_id]:
                del self._user_tasks[user_id]
            
            del self._tasks[task_id]
    
    def cleanup_old_tasks(self, max_age: float = 300):
        """
        Очищает старые задачи
        
        :param max_age: Максимальный возраст задачи в секундах
        """
        current_time = asyncio.get_event_loop().time()
        old_tasks = [
            task_id for task_id, task_data in self._tasks.items()
            if current_time - task_data['created_at'] > max_age
        ]
        
        for task_id in old_tasks:
            self._cleanup_task(task_id)

    async def cleanup_generation_tasks(self):
        """
        Периодическая очистка старых задач генерации
        """
        while True:
            try:
                self.cleanup_old_tasks()
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Error in generation tasks cleanup: {e}")

    async def check_generation_tasks(self, config, result_handler):
        """
        Периодически проверяет статус задач генерации
        """
        while True:
            try:
                for task_id in list(self._tasks.keys()):
                    result = await self.get_task_result(task_id)
                    if result:
                        await handle_direct_result(config, result)
            except Exception as e:
                logging.error(f"Error processing generation task: {e}")
            
            await asyncio.sleep(5)

    def get_active_tasks_count(self, user_id: str = None) -> int:
        """
        Возвращает количество активных задач
        :param user_id: ID пользователя (опционально)
        :return: Количество активных задач
        """
        if user_id:
            return len(self._user_tasks.get(user_id, set()))
        return len(self._tasks)