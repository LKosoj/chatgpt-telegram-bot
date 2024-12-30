"""
Модуль для валидации данных
"""

import re
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from datetime import datetime
from .exceptions import ValidationError
from .constants import Limits
from .agent_logger import AgentLogger


class Validator:
    """Базовый класс валидатора"""
    
    def __init__(self, field_name: str):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
        """
        self.field_name = field_name
        
    def __call__(self, value: Any) -> Any:
        """
        Валидация значения
        
        Args:
            value: значение
            
        Returns:
            Any: валидное значение
            
        Raises:
            ValidationError: если значение невалидно
        """
        raise NotImplementedError


class Required(Validator):
    """Валидатор обязательного поля"""
    
    def __call__(self, value: Any) -> Any:
        if value is None:
            raise ValidationError(
                f"Field '{self.field_name}' is required",
                field=self.field_name,
                value=value
            )
        return value


class TypeValidator(Validator):
    """Валидатор типа"""
    
    def __init__(self, field_name: str, type_: Type):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            type_: ожидаемый тип
        """
        super().__init__(field_name)
        self.type = type_
        
    def __call__(self, value: Any) -> Any:
        if not isinstance(value, self.type):
            raise ValidationError(
                f"Field '{self.field_name}' must be of type {self.type.__name__}",
                field=self.field_name,
                value=value
            )
        return value


class String(TypeValidator):
    """Валидатор строки"""
    
    def __init__(
        self,
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            min_length: минимальная длина
            max_length: максимальная длина
            pattern: регулярное выражение
        """
        super().__init__(field_name, str)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern and re.compile(pattern)
        
    def __call__(self, value: Any) -> str:
        value = super().__call__(value)
        
        if self.min_length is not None and len(value) < self.min_length:
            raise ValidationError(
                f"Field '{self.field_name}' must be at least {self.min_length} characters long",
                field=self.field_name,
                value=value
            )
            
        if self.max_length is not None and len(value) > self.max_length:
            raise ValidationError(
                f"Field '{self.field_name}' must be at most {self.max_length} characters long",
                field=self.field_name,
                value=value
            )
            
        if self.pattern and not self.pattern.match(value):
            raise ValidationError(
                f"Field '{self.field_name}' must match pattern {self.pattern.pattern}",
                field=self.field_name,
                value=value
            )
            
        return value


class Number(TypeValidator):
    """Валидатор числа"""
    
    def __init__(
        self,
        field_name: str,
        type_: Type[Union[int, float]],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            type_: тип числа
            min_value: минимальное значение
            max_value: максимальное значение
        """
        super().__init__(field_name, type_)
        self.min_value = min_value
        self.max_value = max_value
        
    def __call__(self, value: Any) -> Union[int, float]:
        value = super().__call__(value)
        
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Field '{self.field_name}' must be greater than or equal to {self.min_value}",
                field=self.field_name,
                value=value
            )
            
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Field '{self.field_name}' must be less than or equal to {self.max_value}",
                field=self.field_name,
                value=value
            )
            
        return value


class Integer(Number):
    """Валидатор целого числа"""
    
    def __init__(
        self,
        field_name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            min_value: минимальное значение
            max_value: максимальное значение
        """
        super().__init__(field_name, int, min_value, max_value)


class Float(Number):
    """Валидатор числа с плавающей точкой"""
    
    def __init__(
        self,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            min_value: минимальное значение
            max_value: максимальное значение
        """
        super().__init__(field_name, float, min_value, max_value)


class Boolean(TypeValidator):
    """Валидатор логического значения"""
    
    def __init__(self, field_name: str):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
        """
        super().__init__(field_name, bool)


class DateTime(TypeValidator):
    """Валидатор даты и времени"""
    
    def __init__(
        self,
        field_name: str,
        min_value: Optional[datetime] = None,
        max_value: Optional[datetime] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            min_value: минимальное значение
            max_value: максимальное значение
        """
        super().__init__(field_name, datetime)
        self.min_value = min_value
        self.max_value = max_value
        
    def __call__(self, value: Any) -> datetime:
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValidationError(
                    f"Field '{self.field_name}' must be a valid ISO format datetime",
                    field=self.field_name,
                    value=value
                )
                
        value = super().__call__(value)
        
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Field '{self.field_name}' must be after {self.min_value}",
                field=self.field_name,
                value=value
            )
            
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Field '{self.field_name}' must be before {self.max_value}",
                field=self.field_name,
                value=value
            )
            
        return value


class List(TypeValidator):
    """Валидатор списка"""
    
    def __init__(
        self,
        field_name: str,
        item_validator: Optional[Validator] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            item_validator: валидатор элементов
            min_length: минимальная длина
            max_length: максимальная длина
        """
        super().__init__(field_name, list)
        self.item_validator = item_validator
        self.min_length = min_length
        self.max_length = max_length
        
    def __call__(self, value: Any) -> List[Any]:
        value = super().__call__(value)
        
        if self.min_length is not None and len(value) < self.min_length:
            raise ValidationError(
                f"Field '{self.field_name}' must have at least {self.min_length} items",
                field=self.field_name,
                value=value
            )
            
        if self.max_length is not None and len(value) > self.max_length:
            raise ValidationError(
                f"Field '{self.field_name}' must have at most {self.max_length} items",
                field=self.field_name,
                value=value
            )
            
        if self.item_validator:
            for i, item in enumerate(value):
                try:
                    value[i] = self.item_validator(item)
                except ValidationError as e:
                    raise ValidationError(
                        f"Invalid item at index {i} in field '{self.field_name}'",
                        field=f"{self.field_name}[{i}]",
                        value=item
                    ) from e
                    
        return value


class Dict(TypeValidator):
    """Валидатор словаря"""
    
    def __init__(
        self,
        field_name: str,
        schema: Optional[Dict[str, Validator]] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            schema: схема валидации
        """
        super().__init__(field_name, dict)
        self.schema = schema
        
    def __call__(self, value: Any) -> Dict[str, Any]:
        value = super().__call__(value)
        
        if self.schema:
            result = {}
            for key, validator in self.schema.items():
                try:
                    if key in value:
                        result[key] = validator(value[key])
                    elif isinstance(validator, Required):
                        raise ValidationError(
                            f"Required field '{key}' is missing",
                            field=f"{self.field_name}.{key}",
                            value=None
                        )
                except ValidationError as e:
                    raise ValidationError(
                        f"Invalid value for field '{key}'",
                        field=f"{self.field_name}.{key}",
                        value=value.get(key)
                    ) from e
                    
            return result
            
        return value


class Enum(Validator):
    """Валидатор перечисления"""
    
    def __init__(self, field_name: str, values: List[Any]):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            values: допустимые значения
        """
        super().__init__(field_name)
        self.values = values
        
    def __call__(self, value: Any) -> Any:
        if value not in self.values:
            raise ValidationError(
                f"Field '{self.field_name}' must be one of {self.values}",
                field=self.field_name,
                value=value
            )
        return value


class Email(String):
    """Валидатор email"""
    
    def __init__(self, field_name: str):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
        """
        super().__init__(
            field_name,
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )


class URL(String):
    """Валидатор URL"""
    
    def __init__(self, field_name: str):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
        """
        super().__init__(
            field_name,
            pattern=r"^https?://[^\s/$.?#].[^\s]*$"
        )


class Custom(Validator):
    """Пользовательский валидатор"""
    
    def __init__(
        self,
        field_name: str,
        validator: Callable[[Any], Any]
    ):
        """
        Инициализация валидатора
        
        Args:
            field_name: название поля
            validator: функция валидации
        """
        super().__init__(field_name)
        self.validator = validator
        
    def __call__(self, value: Any) -> Any:
        try:
            return self.validator(value)
        except Exception as e:
            raise ValidationError(
                str(e),
                field=self.field_name,
                value=value
            ) from e 


class ResultValidator:
    """Валидатор результатов"""
    
    def __init__(self, logger: Optional[AgentLogger] = None):
        """
        Инициализация валидатора
        
        Args:
            logger: логгер (опционально)
        """
        self.logger = logger or AgentLogger("logs")
        
        # Схемы валидации
        self.research_schema = Dict("research_result", {
            "query": Required(String("query", max_length=1000)),
            "data": Required(Dict("data")),
            "source": String("source"),
            "relevance": Float("relevance", min_value=0.0, max_value=1.0)
        })
        
        self.plan_schema = Dict("action_plan", {
            "plan_id": Required(String("plan_id")),
            "task_id": Required(String("task_id")),
            "steps": Required(List("steps", min_length=1)),
            "status": Required(Enum("status", [s.value for s in TaskStatus])),
            "created_at": Required(DateTime("created_at")),
            "updated_at": DateTime("updated_at"),
            "metadata": Dict("metadata")
        })
        
        self.task_schema = Dict("task", {
            "task_id": Required(String("task_id")),
            "description": Required(String("description", max_length=5000)),
            "status": Required(Enum("status", [s.value for s in TaskStatus])),
            "priority": Required(Enum("priority", [p.value for p in TaskPriority])),
            "assigned_to": Enum("assigned_to", [r.value for r in AgentRole]),
            "parent_task_id": String("parent_task_id"),
            "created_at": Required(DateTime("created_at")),
            "started_at": DateTime("started_at"),
            "completed_at": DateTime("completed_at"),
            "metadata": Dict("metadata")
        })
        
    async def validate_research_result(
        self,
        result: Union[ResearchResult, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Валидация результата исследования
        
        Args:
            result: результат исследования
            
        Returns:
            Tuple[bool, Optional[str]]: результат валидации и сообщение об ошибке
        """
        try:
            data = result.to_dict() if isinstance(result, ResearchResult) else result
            validated = self.research_schema(data)
            
            # Дополнительные проверки
            if validated["data"] and not isinstance(validated["data"], dict):
                return False, "Field 'data' must be a dictionary"
                
            if validated.get("source") and len(validated["source"]) > 500:
                return False, "Field 'source' is too long"
                
            return True, None
            
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Error validating research result: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    async def validate_action_plan(
        self,
        plan: Union[ActionPlan, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Валидация плана действий
        
        Args:
            plan: план действий
            
        Returns:
            Tuple[bool, Optional[str]]: результат валидации и сообщение об ошибке
        """
        try:
            data = plan.to_dict() if isinstance(plan, ActionPlan) else plan
            validated = self.plan_schema(data)
            
            # Проверяем шаги плана
            for step in validated["steps"]:
                if not isinstance(step, dict):
                    return False, "Each step must be a dictionary"
                    
                required_step_fields = ["step_id", "description", "status"]
                if not all(field in step for field in required_step_fields):
                    return False, f"Step is missing required fields: {required_step_fields}"
                    
                if step["status"] not in [s.value for s in TaskStatus]:
                    return False, f"Invalid step status: {step['status']}"
                    
            # Проверяем временные метки
            if validated.get("updated_at"):
                if validated["updated_at"] < validated["created_at"]:
                    return False, "updated_at cannot be earlier than created_at"
                    
            return True, None
            
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Error validating action plan: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    async def validate_task(
        self,
        task: Union[Task, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Валидация задачи
        
        Args:
            task: задача
            
        Returns:
            Tuple[bool, Optional[str]]: результат валидации и сообщение об ошибке
        """
        try:
            data = task.to_dict() if isinstance(task, Task) else task
            validated = self.task_schema(data)
            
            # Проверяем временные метки
            if validated.get("started_at"):
                if validated["started_at"] < validated["created_at"]:
                    return False, "started_at cannot be earlier than created_at"
                    
            if validated.get("completed_at"):
                if validated["completed_at"] < validated["created_at"]:
                    return False, "completed_at cannot be earlier than created_at"
                if validated.get("started_at") and validated["completed_at"] < validated["started_at"]:
                    return False, "completed_at cannot be earlier than started_at"
                    
            # Проверяем статус и временные метки
            if validated["status"] == TaskStatus.COMPLETED.value and not validated.get("completed_at"):
                return False, "Completed task must have completed_at timestamp"
                
            if validated["status"] == TaskStatus.IN_PROGRESS.value and not validated.get("started_at"):
                return False, "In progress task must have started_at timestamp"
                
            return True, None
            
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Error validating task: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    async def generate_validation_report(
        self,
        results: List[Union[ResearchResult, ActionPlan, Task]]
    ) -> Dict[str, Any]:
        """
        Генерация отчета о валидации
        
        Args:
            results: список результатов для валидации
            
        Returns:
            Dict[str, Any]: отчет о валидации
        """
        report = {
            "total": len(results),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "validation_time": datetime.now().isoformat(),
            "details": []
        }
        
        for item in results:
            start_time = time.time()
            
            if isinstance(item, ResearchResult):
                is_valid, error = await self.validate_research_result(item)
                item_type = "research_result"
            elif isinstance(item, ActionPlan):
                is_valid, error = await self.validate_action_plan(item)
                item_type = "action_plan"
            elif isinstance(item, Task):
                is_valid, error = await self.validate_task(item)
                item_type = "task"
            else:
                is_valid, error = False, "Unknown item type"
                item_type = "unknown"
                
            validation_time = time.time() - start_time
            
            if is_valid:
                report["valid"] += 1
            else:
                report["invalid"] += 1
                report["errors"].append(error)
                
            report["details"].append({
                "type": item_type,
                "id": getattr(item, "id", None) or getattr(item, "task_id", None),
                "is_valid": is_valid,
                "error": error,
                "validation_time": validation_time
            })
            
        return report
            
    async def validate_data_format(
        self,
        data: Any,
        expected_format: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Проверка формата данных
        
        Args:
            data: данные для проверки
            expected_format: ожидаемый формат
            
        Returns:
            Tuple[bool, Optional[str]]: результат проверки и сообщение об ошибке
        """
        try:
            # Создаем схему валидации на основе ожидаемого формата
            schema = {}
            for field_name, field_type in expected_format.items():
                if field_type == str:
                    schema[field_name] = String(field_name)
                elif field_type == int:
                    schema[field_name] = Integer(field_name)
                elif field_type == float:
                    schema[field_name] = Float(field_name)
                elif field_type == bool:
                    schema[field_name] = Boolean(field_name)
                elif field_type == datetime:
                    schema[field_name] = DateTime(field_name)
                elif isinstance(field_type, list):
                    schema[field_name] = List(field_name)
                elif isinstance(field_type, dict):
                    schema[field_name] = Dict(field_name)
                else:
                    return False, f"Unsupported type for field {field_name}: {field_type}"
                    
            validator = Dict("root", schema)
            validator(data)
            return True, None
            
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Error validating data format: {str(e)}")
            return False, f"Format validation error: {str(e)}" 