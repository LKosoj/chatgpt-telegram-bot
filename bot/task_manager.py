import asyncio
import logging
from typing import Dict, Any

from utils import handle_direct_result

class GenerationTaskManager:
    def __init__(self):
        # Словарь для хранения задач генерации
        self._tasks: Dict[str, Dict[str, Any]] = {}
        
    async def create_task(self, task_id: str, generation_coroutine, context: Dict):
        """
        Создает новую задачу генерации
        
        :param task_id: Уникальный идентификатор задачи
        :param generation_coroutine: Корутина для генерации
        :param context: Контекст для последующей обработки результата
        """
        try:
            # Создаем Future для асинхронного ожидания результата
            future = asyncio.create_task(generation_coroutine)
            
            # Сохраняем задачу в словаре
            self._tasks[task_id] = {
                'future': future,
                'context': context,
                'created_at': asyncio.get_event_loop().time()
            }
            
            return task_id
        
        except Exception as e:
            logging.error(f"Error creating generation task: {e}")
            return None
    
    async def get_task_result(self, task_id: str):
        """
        Получает результат задачи генерации
        
        :param task_id: Идентификатор задачи
        :return: Результат задачи или None
        """
        if task_id not in self._tasks:
            logging.warning(f"Task {task_id} not found")
            return None
        
        try:
            task_data = self._tasks[task_id]
            result = await task_data['future']
            
            # Очищаем задачу после получения результата
            del self._tasks[task_id]
            
            return result
        
        except Exception as e:
            logging.error(f"Error getting task result for {task_id}: {e}")
            del self._tasks[task_id]
            return None
    
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
            task_data = self._tasks[task_id]
            if not task_data['future'].done():
                task_data['future'].cancel()
            del self._tasks[task_id]


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
                for task_id, task_data in list(self._tasks.items()):
                    try:
                        result = await self.get_task_result(task_id)
                        if result:
                            # Используйте переданный обработчик результата
                            handle_direct_result(config, result)
                    except Exception as e:
                        logging.error(f"Error processing generation task: {e}")

                await asyncio.sleep(5)
