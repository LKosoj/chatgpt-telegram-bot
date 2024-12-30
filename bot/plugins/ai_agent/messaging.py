"""
Система управления сообщениями между агентами
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from .models import AgentRole, AgentMessage


@dataclass
class MessageMetadata:
    """Метаданные сообщения"""
    message_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    requires_ack: bool = True
    ack_timeout: float = 30.0  # в секундах
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


class MessageQueue:
    """Асинхронная очередь сообщений с приоритетами"""
    
    def __init__(self):
        """Инициализация очереди сообщений"""
        self.queues: Dict[AgentRole, asyncio.PriorityQueue] = {
            role: asyncio.PriorityQueue() for role in AgentRole
        }
        self.pending_acks: Dict[str, asyncio.Event] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Настройка логгера"""
        import logging
        logger = logging.getLogger('MessageQueue')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def put_message(
        self,
        message: AgentMessage,
        priority: int = 0,
        expires_in: Optional[float] = None,
        requires_ack: bool = True
    ) -> str:
        """
        Отправка сообщения в очередь
        
        Args:
            message: сообщение для отправки
            priority: приоритет сообщения (меньше = выше приоритет)
            expires_in: время жизни сообщения в секундах
            requires_ack: требуется ли подтверждение доставки
            
        Returns:
            str: ID сообщения
        """
        message_id = str(uuid.uuid4())
        now = datetime.now()
        
        metadata = MessageMetadata(
            message_id=message_id,
            created_at=now,
            expires_at=now + timedelta(seconds=expires_in) if expires_in else None,
            priority=priority,
            requires_ack=requires_ack
        )
        
        # Добавляем метаданные к сообщению
        if message.metadata is None:
            message.metadata = {}
        message.metadata.update(metadata.to_dict())
        
        # Создаем событие для подтверждения доставки
        if requires_ack:
            self.pending_acks[message_id] = asyncio.Event()
        
        # Добавляем сообщение в очередь получателя
        await self.queues[message.to_role].put((priority, message))
        
        # Логируем отправку
        self.logger.info(
            f"Message {message_id} sent from {message.from_role.value} "
            f"to {message.to_role.value} with priority {priority}"
        )
        
        # Сохраняем в историю
        self.message_history.append({
            "message_id": message_id,
            "from_role": message.from_role.value,
            "to_role": message.to_role.value,
            "content": message.content,
            "metadata": metadata.to_dict(),
            "status": "sent"
        })
        
        return message_id

    async def get_message(self, role: AgentRole) -> Optional[AgentMessage]:
        """
        Получение сообщения из очереди
        
        Args:
            role: роль агента, получающего сообщение
            
        Returns:
            Optional[AgentMessage]: сообщение или None, если очередь пуста
        """
        try:
            # Получаем сообщение из очереди
            priority, message = await self.queues[role].get()
            
            # Проверяем срок действия
            metadata = MessageMetadata(**message.metadata)
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                self.logger.warning(f"Message {metadata.message_id} expired")
                self._update_message_status(metadata.message_id, "expired")
                return None
            
            self.logger.info(
                f"Message {metadata.message_id} received by {role.value}"
            )
            self._update_message_status(metadata.message_id, "received")
            
            return message
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            self.logger.error(f"Error getting message: {str(e)}")
            return None

    async def acknowledge_message(self, message_id: str):
        """
        Подтверждение получения сообщения
        
        Args:
            message_id: ID сообщения
        """
        if message_id in self.pending_acks:
            self.pending_acks[message_id].set()
            self.logger.info(f"Message {message_id} acknowledged")
            self._update_message_status(message_id, "acknowledged")

    async def wait_for_ack(self, message_id: str, timeout: float = None) -> bool:
        """
        Ожидание подтверждения доставки
        
        Args:
            message_id: ID сообщения
            timeout: таймаут ожидания в секундах
            
        Returns:
            bool: True если получено подтверждение, False если таймаут
        """
        if message_id not in self.pending_acks:
            return False
            
        try:
            await asyncio.wait_for(
                self.pending_acks[message_id].wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            self.logger.warning(f"Acknowledgement timeout for message {message_id}")
            self._update_message_status(message_id, "timeout")
            return False

    def _update_message_status(self, message_id: str, status: str):
        """
        Обновление статуса сообщения в истории
        
        Args:
            message_id: ID сообщения
            status: новый статус
        """
        for message in self.message_history:
            if message["message_id"] == message_id:
                message["status"] = status
                message["updated_at"] = datetime.now().isoformat()
                break

    def get_message_status(self, message_id: str) -> Optional[str]:
        """
        Получение статуса сообщения
        
        Args:
            message_id: ID сообщения
            
        Returns:
            Optional[str]: статус сообщения или None, если не найдено
        """
        for message in self.message_history:
            if message["message_id"] == message_id:
                return message["status"]
        return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Получение статистики очередей
        
        Returns:
            Dict[str, Any]: статистика по очередям
        """
        stats = {
            "queues": {
                role.value: self.queues[role].qsize() 
                for role in AgentRole
            },
            "pending_acks": len(self.pending_acks),
            "total_messages": len(self.message_history),
            "status_distribution": {}
        }
        
        # Подсчет распределения статусов
        status_counts = {}
        for message in self.message_history:
            status = message["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        stats["status_distribution"] = status_counts
        
        return stats

    async def cleanup_expired_messages(self):
        """Очистка истекших сообщений из очередей"""
        for role in AgentRole:
            queue = self.queues[role]
            new_queue = asyncio.PriorityQueue()
            
            while not queue.empty():
                priority, message = await queue.get()
                metadata = MessageMetadata(**message.metadata)
                
                if not metadata.expires_at or datetime.now() <= metadata.expires_at:
                    await new_queue.put((priority, message))
                else:
                    self._update_message_status(metadata.message_id, "expired")
            
            self.queues[role] = new_queue

    def cleanup_history(self, max_age_days: int = 7):
        """
        Очистка старых сообщений из истории
        
        Args:
            max_age_days: максимальный возраст сообщений в днях
        """
        threshold = datetime.now() - timedelta(days=max_age_days)
        self.message_history = [
            message for message in self.message_history
            if datetime.fromisoformat(message["metadata"]["created_at"]) > threshold
        ]


class MessageAcknowledgement:
    """Система подтверждений доставки сообщений"""
    
    def __init__(self, queue: MessageQueue):
        """
        Инициализация системы подтверждений
        
        Args:
            queue: очередь сообщений
        """
        self.queue = queue
        self.retry_delays = [1, 5, 15]  # задержки между повторными попытками в секундах
        self.logger = self.queue.logger

    async def wait_for_ack(
        self,
        message_id: str,
        timeout: float = 30.0,
        retry_count: int = 3
    ) -> bool:
        """
        Ожидание подтверждения с повторными попытками
        
        Args:
            message_id: ID сообщения
            timeout: таймаут ожидания в секундах
            retry_count: количество повторных попыток
            
        Returns:
            bool: True если получено подтверждение, False если все попытки неудачны
        """
        for attempt in range(retry_count):
            if await self.queue.wait_for_ack(message_id, timeout):
                return True
                
            if attempt < retry_count - 1:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                self.logger.warning(
                    f"Retry {attempt + 1}/{retry_count} for message {message_id} "
                    f"after {delay}s"
                )
                await asyncio.sleep(delay)
        
        return False

    def get_ack_stats(self) -> Dict[str, Any]:
        """
        Получение статистики подтверждений
        
        Returns:
            Dict[str, Any]: статистика подтверждений
        """
        stats = {
            "total_messages": 0,
            "acknowledged": 0,
            "timeout": 0,
            "pending": 0
        }
        
        for message in self.queue.message_history:
            if message["metadata"].get("requires_ack"):
                stats["total_messages"] += 1
                status = message["status"]
                if status == "acknowledged":
                    stats["acknowledged"] += 1
                elif status == "timeout":
                    stats["timeout"] += 1
                elif status in ["sent", "received"]:
                    stats["pending"] += 1
        
        return stats 