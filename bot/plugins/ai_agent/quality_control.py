"""
Модуль для контроля качества
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from .agent_logger import AgentLogger
from .models import ResearchResult, Task, ActionPlan
from .validation import ResultValidator


@dataclass
class QualityMetrics:
    """Метрики качества"""
    accuracy: float
    relevance: float
    completeness: float
    timeliness: float
    user_satisfaction: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class QualityControl:
    """Контроль качества"""
    
    def __init__(
        self,
        logger: Optional[AgentLogger] = None,
        validator: Optional[ResultValidator] = None
    ):
        """
        Инициализация контроля качества
        
        Args:
            logger: логгер (опционально)
            validator: валидатор результатов (опционально)
        """
        self.logger = logger or AgentLogger("logs")
        self.validator = validator or ResultValidator(self.logger)
        
        # История оценок качества
        self.quality_history: Dict[str, List[QualityMetrics]] = defaultdict(list)
        
        # Обратная связь пользователей
        self.user_feedback: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Пороговые значения
        self.quality_thresholds = {
            "accuracy": 0.8,
            "relevance": 0.7,
            "completeness": 0.8,
            "timeliness": 0.9,
            "user_satisfaction": 0.7
        }
        
    async def check_answer_quality(
        self,
        answer: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, QualityMetrics]:
        """
        Проверка качества ответа
        
        Args:
            answer: ответ для проверки
            expected: ожидаемый ответ (опционально)
            context: контекст ответа (опционально)
            
        Returns:
            Tuple[bool, QualityMetrics]: результат проверки и метрики качества
        """
        metrics = QualityMetrics(
            accuracy=0.0,
            relevance=0.0,
            completeness=0.0,
            timeliness=0.0,
            user_satisfaction=0.0
        )
        
        # Проверяем точность, если есть ожидаемый ответ
        if expected:
            metrics.accuracy = self._calculate_accuracy(answer, expected)
            
        # Проверяем релевантность контексту
        if context:
            metrics.relevance = self._calculate_relevance(answer, context)
            
        # Проверяем полноту ответа
        metrics.completeness = self._calculate_completeness(answer)
        
        # Проверяем своевременность
        metrics.timeliness = self._calculate_timeliness(
            context.get("start_time") if context else None
        )
        
        # Получаем удовлетворенность пользователей
        metrics.user_satisfaction = self._get_user_satisfaction(
            context.get("task_id") if context else None
        )
        
        # Проверяем соответствие пороговым значениям
        is_quality_sufficient = all(
            getattr(metrics, metric) >= threshold
            for metric, threshold in self.quality_thresholds.items()
        )
        
        # Сохраняем метрики в историю
        if context and "task_id" in context:
            self.quality_history[context["task_id"]].append(metrics)
            
        return is_quality_sufficient, metrics
        
    async def evaluate_result_relevance(
        self,
        result: Union[ResearchResult, Dict[str, Any]],
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Оценка релевантности результата
        
        Args:
            result: результат для оценки
            query: исходный запрос
            context: контекст запроса (опционально)
            
        Returns:
            float: оценка релевантности (0.0 - 1.0)
        """
        if isinstance(result, ResearchResult):
            result_data = result.to_dict()
        else:
            result_data = result
            
        # Проверяем валидность результата
        is_valid, error = await self.validator.validate_research_result(result_data)
        if not is_valid:
            self.logger.warning(f"Invalid result: {error}")
            return 0.0
            
        # Оцениваем релевантность содержимого
        content_relevance = self._calculate_relevance(
            json.dumps(result_data.get("data", {})),
            {"query": query, **(context or {})}
        )
        
        # Учитываем источник данных
        source_weight = 0.8 if result_data.get("source") else 0.5
        
        # Учитываем время получения результата
        time_weight = self._calculate_timeliness(
            context.get("start_time") if context else None
        )
        
        # Вычисляем итоговую релевантность
        relevance = (
            content_relevance * 0.6 +
            source_weight * 0.2 +
            time_weight * 0.2
        )
        
        return min(max(relevance, 0.0), 1.0)
        
    async def analyze_user_satisfaction(
        self,
        task_id: str,
        period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Анализ удовлетворенности пользователей
        
        Args:
            task_id: идентификатор задачи
            period: период анализа
            
        Returns:
            Dict[str, Any]: результаты анализа
        """
        feedback = self.user_feedback.get(task_id, [])
        
        if period:
            threshold = datetime.now() - period
            feedback = [
                f for f in feedback
                if datetime.fromisoformat(f["timestamp"]) >= threshold
            ]
            
        if not feedback:
            return {
                "average_rating": 0.0,
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "common_issues": [],
                "suggestions": []
            }
            
        # Анализируем оценки
        ratings = [f["rating"] for f in feedback]
        avg_rating = sum(ratings) / len(ratings)
        
        # Считаем позитивные и негативные отзывы
        positive = sum(1 for r in ratings if r >= 4)
        negative = sum(1 for r in ratings if r <= 2)
        
        # Анализируем комментарии
        comments = [f["comment"] for f in feedback if "comment" in f]
        issues = self._analyze_feedback_comments(
            [c for c in comments if any(
                neg in c.lower()
                for neg in ["problem", "issue", "error", "bug", "wrong"]
            )]
        )
        
        suggestions = self._analyze_feedback_comments(
            [c for c in comments if any(
                sug in c.lower()
                for sug in ["suggest", "improve", "better", "would", "could"]
            )]
        )
        
        return {
            "average_rating": avg_rating,
            "total_feedback": len(feedback),
            "positive_feedback": positive,
            "negative_feedback": negative,
            "common_issues": issues[:5],  # Топ-5 проблем
            "suggestions": suggestions[:5]  # Топ-5 предложений
        }
        
    async def process_user_feedback(
        self,
        task_id: str,
        rating: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Обработка обратной связи от пользователя
        
        Args:
            task_id: идентификатор задачи
            rating: оценка (0.0 - 5.0)
            comment: комментарий (опционально)
            metadata: дополнительные данные (опционально)
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "rating": min(max(rating, 0.0), 5.0),
            "metadata": metadata or {}
        }
        
        if comment:
            feedback["comment"] = comment
            
        self.user_feedback[task_id].append(feedback)
        
        # Логируем обратную связь
        self.logger.info(
            f"Received user feedback for task {task_id}",
            extra={
                "task_id": task_id,
                "rating": rating,
                "has_comment": bool(comment)
            }
        )
        
        # Проверяем необходимость автоматической корректировки
        if rating <= 2.0:  # Низкая оценка
            await self._trigger_auto_correction(task_id, feedback)
            
    async def get_quality_trends(
        self,
        task_id: Optional[str] = None,
        period: Optional[timedelta] = None,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получение трендов качества
        
        Args:
            task_id: идентификатор задачи (опционально)
            period: период анализа (опционально)
            metric_name: название метрики (опционально)
            
        Returns:
            Dict[str, Any]: тренды качества
        """
        history = self.quality_history
        if task_id:
            history = {task_id: history.get(task_id, [])}
            
        if period:
            threshold = datetime.now() - period
            history = {
                tid: [
                    m for m in metrics
                    if m.timestamp >= threshold
                ]
                for tid, metrics in history.items()
            }
            
        trends = {}
        for tid, metrics in history.items():
            if not metrics:
                continue
                
            task_trends = {}
            metric_names = (
                [metric_name] if metric_name
                else ["accuracy", "relevance", "completeness", "timeliness", "user_satisfaction"]
            )
            
            for name in metric_names:
                values = [getattr(m, name) for m in metrics]
                if not values:
                    continue
                    
                task_trends[name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._calculate_trend(values)
                }
                
            if task_trends:
                trends[tid] = task_trends
                
        return trends
        
    async def auto_correct_quality(
        self,
        task_id: str,
        feedback: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Автоматическая корректировка качества
        
        Args:
            task_id: идентификатор задачи
            feedback: обратная связь
            
        Returns:
            Optional[Dict[str, Any]]: результаты корректировки
        """
        try:
            # Анализируем проблему
            issues = self._analyze_feedback_comments([feedback.get("comment", "")])
            if not issues:
                return None
                
            corrections = {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "original_feedback": feedback,
                "identified_issues": issues,
                "corrections": []
            }
            
            # Применяем корректировки на основе проблем
            for issue in issues:
                correction = await self._apply_correction(task_id, issue)
                if correction:
                    corrections["corrections"].append(correction)
                    
            if corrections["corrections"]:
                self.logger.info(
                    f"Applied auto-corrections for task {task_id}",
                    extra=corrections
                )
                return corrections
                
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error in auto-correction for task {task_id}",
                error=str(e)
            )
            return None
            
    def _calculate_accuracy(self, answer: str, expected: str) -> float:
        """Расчет точности ответа"""
        if not answer or not expected:
            return 0.0
            
        # Простое сравнение на основе общих слов
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
            
        common_words = answer_words & expected_words
        return len(common_words) / len(expected_words)
        
    def _calculate_relevance(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> float:
        """Расчет релевантности контенту"""
        if not content or not context:
            return 0.0
            
        # Проверяем наличие ключевых слов из запроса
        query = context.get("query", "").lower()
        content = content.lower()
        
        if not query:
            return 0.5  # Нет запроса для сравнения
            
        query_words = set(query.split())
        content_words = set(content.split())
        
        # Находим пересечение слов
        common_words = query_words & content_words
        
        # Базовая релевантность на основе пересечения слов
        base_relevance = len(common_words) / len(query_words) if query_words else 0.0
        
        # Учитываем дополнительные факторы из контекста
        context_weight = 1.0
        if "importance" in context:
            context_weight *= float(context["importance"])
        if "priority" in context:
            context_weight *= float(context["priority"])
            
        return min(base_relevance * context_weight, 1.0)
        
    def _calculate_completeness(self, content: str) -> float:
        """Расчет полноты контента"""
        if not content:
            return 0.0
            
        # Проверяем длину контента
        words = content.split()
        if len(words) < 10:
            return 0.3  # Слишком короткий ответ
        if len(words) < 50:
            return 0.7  # Средний ответ
        return 0.9  # Полный ответ
        
    def _calculate_timeliness(
        self,
        start_time: Optional[str]
    ) -> float:
        """Расчет своевременности"""
        if not start_time:
            return 1.0  # Нет информации о времени
            
        try:
            start = datetime.fromisoformat(start_time)
            duration = (datetime.now() - start).total_seconds()
            
            # Оцениваем своевременность на основе длительности
            if duration <= 1:
                return 1.0  # Мгновенный ответ
            elif duration <= 5:
                return 0.9  # Быстрый ответ
            elif duration <= 30:
                return 0.7  # Приемлемое время
            else:
                return 0.5  # Медленный ответ
                
        except (ValueError, TypeError):
            return 1.0  # Ошибка парсинга времени
            
    def _get_user_satisfaction(
        self,
        task_id: Optional[str]
    ) -> float:
        """Получение оценки удовлетворенности пользователей"""
        if not task_id:
            return 0.5  # Нет данных
            
        feedback = self.user_feedback.get(task_id, [])
        if not feedback:
            return 0.5  # Нет обратной связи
            
        # Вычисляем среднюю оценку
        ratings = [f["rating"] for f in feedback]
        return sum(ratings) / (len(ratings) * 5.0)  # Нормализуем до 0.0-1.0
        
    def _analyze_feedback_comments(
        self,
        comments: List[str]
    ) -> List[str]:
        """Анализ комментариев обратной связи"""
        if not comments:
            return []
            
        # Простой анализ на основе частоты слов
        word_freq = defaultdict(int)
        for comment in comments:
            words = comment.lower().split()
            for word in words:
                word_freq[word] += 1
                
        # Находим наиболее частые слова
        common_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Формируем список проблем/предложений
        results = []
        for word, freq in common_words[:10]:  # Топ-10 слов
            # Ищем предложения с этим словом
            for comment in comments:
                if word in comment.lower():
                    results.append(comment.strip())
                    break
                    
        return list(set(results))  # Убираем дубликаты
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Расчет тренда на основе значений"""
        if len(values) < 2:
            return "stable"
            
        # Используем простую линейную регрессию
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = (
            (len(values) * (x * y).sum() - x.sum() * y.sum()) /
            (len(values) * (x * x).sum() - x.sum() * x.sum())
        )
        
        if abs(slope) < 0.01:
            return "stable"
        return "increasing" if slope > 0 else "decreasing"
        
    async def _trigger_auto_correction(
        self,
        task_id: str,
        feedback: Dict[str, Any]
    ):
        """Запуск автоматической корректировки"""
        corrections = await self.auto_correct_quality(task_id, feedback)
        if corrections:
            self.logger.info(
                f"Auto-correction triggered for task {task_id}",
                extra=corrections
            )
            
    async def _apply_correction(
        self,
        task_id: str,
        issue: str
    ) -> Optional[Dict[str, Any]]:
        """
        Применение корректировки
        
        Args:
            task_id: идентификатор задачи
            issue: проблема
            
        Returns:
            Optional[Dict[str, Any]]: результат корректировки
        """
        # TODO: Реализовать конкретные стратегии корректировки
        # на основе типа проблемы
        return {
            "issue": issue,
            "correction_type": "notification",
            "timestamp": datetime.now().isoformat()
        } 