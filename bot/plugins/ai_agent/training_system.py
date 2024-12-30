"""
Модуль для системы обучения
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from .logging import AgentLogger
from .documentation_system import DocSection


@dataclass
class TrainingModule:
    """Обучающий модуль"""
    title: str
    description: str
    content: str
    difficulty: str
    prerequisites: List[str]
    exercises: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingProgress:
    """Прогресс обучения"""
    module_id: str
    completed_exercises: List[str]
    quiz_scores: Dict[str, float]
    feedback_given: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


class TrainingSystem:
    """Система управления обучением"""
    
    def __init__(
        self,
        training_dir: str = "training",
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация системы обучения
        
        Args:
            training_dir: директория для обучающих материалов
            logger: логгер (опционально)
        """
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or AgentLogger("training")
        self.modules: Dict[str, TrainingModule] = {}
        self.progress: Dict[str, TrainingProgress] = {}
        
    def create_training_module(
        self,
        title: str,
        description: str,
        content: str,
        difficulty: str,
        prerequisites: List[str],
        exercises: List[Dict[str, Any]],
        examples: List[Dict[str, Any]]
    ) -> TrainingModule:
        """
        Создание обучающего модуля
        
        Args:
            title: название модуля
            description: описание модуля
            content: содержимое модуля
            difficulty: сложность модуля
            prerequisites: предварительные требования
            exercises: упражнения
            examples: примеры
            
        Returns:
            TrainingModule: обучающий модуль
        """
        module = TrainingModule(
            title=title,
            description=description,
            content=content,
            difficulty=difficulty,
            prerequisites=prerequisites,
            exercises=exercises,
            examples=examples
        )
        
        self.modules[title] = module
        return module
        
    def add_exercise(
        self,
        module: TrainingModule,
        title: str,
        description: str,
        task: str,
        solution: str,
        hints: List[str]
    ):
        """
        Добавление упражнения
        
        Args:
            module: обучающий модуль
            title: название упражнения
            description: описание упражнения
            task: задание
            solution: решение
            hints: подсказки
        """
        exercise = {
            "title": title,
            "description": description,
            "task": task,
            "solution": solution,
            "hints": hints
        }
        module.exercises.append(exercise)
        
    def add_example(
        self,
        module: TrainingModule,
        title: str,
        code: str,
        description: str,
        output: Optional[str] = None
    ):
        """
        Добавление примера
        
        Args:
            module: обучающий модуль
            title: название примера
            code: код примера
            description: описание примера
            output: вывод примера (опционально)
        """
        example = {
            "title": title,
            "code": code,
            "description": description,
            "output": output
        }
        module.examples.append(example)
        
    def process_feedback(
        self,
        module: TrainingModule,
        rating: int,
        comment: str,
        suggestions: Optional[List[str]] = None
    ):
        """
        Обработка обратной связи
        
        Args:
            module: обучающий модуль
            rating: оценка
            comment: комментарий
            suggestions: предложения (опционально)
        """
        feedback = {
            "rating": rating,
            "comment": comment,
            "suggestions": suggestions or [],
            "timestamp": datetime.now()
        }
        module.feedback.append(feedback)
        
        # Анализируем обратную связь для улучшения
        if rating < 3:
            self.logger.warning(
                "Low rating feedback received",
                module=module.title,
                rating=rating,
                comment=comment
            )
            
    def track_progress(
        self,
        module_id: str,
        exercise_id: str,
        score: Optional[float] = None,
        feedback: Optional[Dict[str, Any]] = None
    ):
        """
        Отслеживание прогресса
        
        Args:
            module_id: идентификатор модуля
            exercise_id: идентификатор упражнения
            score: оценка (опционально)
            feedback: обратная связь (опционально)
        """
        if module_id not in self.progress:
            self.progress[module_id] = TrainingProgress(
                module_id=module_id,
                completed_exercises=[],
                quiz_scores={},
                feedback_given=[]
            )
            
        progress = self.progress[module_id]
        
        # Добавляем выполненное упражнение
        if exercise_id not in progress.completed_exercises:
            progress.completed_exercises.append(exercise_id)
            
        # Добавляем оценку
        if score is not None:
            progress.quiz_scores[exercise_id] = score
            
        # Добавляем обратную связь
        if feedback is not None:
            progress.feedback_given.append({
                **feedback,
                "exercise_id": exercise_id,
                "timestamp": datetime.now()
            })
            
    def update_training_materials(
        self,
        module: TrainingModule,
        updates: Dict[str, Any]
    ):
        """
        Обновление обучающих материалов
        
        Args:
            module: обучающий модуль
            updates: обновления
        """
        # Обновляем основные поля
        for field, value in updates.items():
            if hasattr(module, field):
                setattr(module, field, value)
                
        # Обновляем timestamp
        module.last_updated = datetime.now()
        
        self.logger.info(
            "Training module updated",
            module=module.title,
            updates=list(updates.keys())
        )
        
    def save_module(self, module: TrainingModule):
        """
        Сохранение модуля
        
        Args:
            module: обучающий модуль
        """
        try:
            # Создаем словарь для сериализации
            data = {
                "title": module.title,
                "description": module.description,
                "content": module.content,
                "difficulty": module.difficulty,
                "prerequisites": module.prerequisites,
                "exercises": module.exercises,
                "examples": module.examples,
                "feedback": module.feedback,
                "last_updated": module.last_updated.isoformat()
            }
            
            # Сохраняем в файл
            module_file = self.training_dir / f"{module.title.lower().replace(' ', '_')}.json"
            with open(module_file, "w") as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Training module saved to {module_file}")
            
        except Exception as e:
            self.logger.error(
                "Error saving training module",
                error=str(e),
                module=module.title
            )
            raise
            
    def load_module(self, filename: str) -> TrainingModule:
        """
        Загрузка модуля
        
        Args:
            filename: имя файла
            
        Returns:
            TrainingModule: обучающий модуль
        """
        try:
            module_file = self.training_dir / filename
            
            with open(module_file) as f:
                data = json.load(f)
                
            module = TrainingModule(
                title=data["title"],
                description=data["description"],
                content=data["content"],
                difficulty=data["difficulty"],
                prerequisites=data["prerequisites"],
                exercises=data["exercises"],
                examples=data["examples"],
                feedback=data["feedback"],
                last_updated=datetime.fromisoformat(data["last_updated"])
            )
            
            self.modules[module.title] = module
            return module
            
        except Exception as e:
            self.logger.error(
                "Error loading training module",
                error=str(e),
                filename=filename
            )
            raise
            
    def export_to_markdown(self, module: TrainingModule, filename: str):
        """
        Экспорт модуля в Markdown
        
        Args:
            module: обучающий модуль
            filename: имя файла
        """
        try:
            content = []
            
            # Добавляем заголовок и описание
            content.append(f"# {module.title}\n")
            content.append(f"{module.description}\n")
            
            # Добавляем информацию о сложности и требованиях
            content.append("## Информация\n")
            content.append(f"- Сложность: {module.difficulty}")
            content.append("- Предварительные требования:")
            for req in module.prerequisites:
                content.append(f"  - {req}")
            content.append("")
            
            # Добавляем основной контент
            content.append("## Содержание\n")
            content.append(module.content)
            content.append("")
            
            # Добавляем примеры
            content.append("## Примеры\n")
            for example in module.examples:
                content.append(f"### {example['title']}\n")
                content.append(example['description'])
                content.append(f"\n```python\n{example['code']}\n```")
                if example.get('output'):
                    content.append(f"\nВывод:\n```\n{example['output']}\n```")
                content.append("")
                
            # Добавляем упражнения
            content.append("## Упражнения\n")
            for exercise in module.exercises:
                content.append(f"### {exercise['title']}\n")
                content.append(exercise['description'])
                content.append(f"\nЗадание:\n{exercise['task']}")
                content.append("\nПодсказки:")
                for hint in exercise['hints']:
                    content.append(f"- {hint}")
                content.append("")
                
            # Сохраняем файл
            doc_file = self.training_dir / filename
            with open(doc_file, "w") as f:
                f.write("\n".join(content))
                
            self.logger.info(f"Training module exported to {doc_file}")
            
        except Exception as e:
            self.logger.error(
                "Error exporting training module",
                error=str(e),
                filename=filename
            )
            raise 