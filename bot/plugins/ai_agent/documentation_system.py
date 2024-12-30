"""
Модуль для системы документации
"""

import os
import re
import json
import inspect
import docstring_parser
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from .logging import AgentLogger


@dataclass
class DocSection:
    """Секция документации"""
    title: str
    content: str
    subsections: List['DocSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class APIEndpoint:
    """Описание API endpoint"""
    name: str
    description: str
    method: str
    path: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]]


class DocumentationSystem:
    """Система управления документацией"""
    
    def __init__(
        self,
        docs_dir: str = "docs",
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация системы документации
        
        Args:
            docs_dir: директория для документации
            logger: логгер (опционально)
        """
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or AgentLogger("docs")
        self.sections: Dict[str, DocSection] = {}
        
    def generate_technical_docs(self, module_path: Union[str, Path]) -> DocSection:
        """
        Генерация технической документации
        
        Args:
            module_path: путь к модулю
            
        Returns:
            DocSection: секция документации
        """
        module_path = Path(module_path)
        
        # Создаем основную секцию
        section = DocSection(
            title=f"Technical Documentation - {module_path.stem}",
            content="",
            subsections=[]
        )
        
        try:
            # Анализируем исходный код
            with open(module_path) as f:
                source = f.read()
                
            # Извлекаем docstrings и сигнатуры
            module = inspect.getsource(source)
            classes = inspect.getmembers(module, inspect.isclass)
            functions = inspect.getmembers(module, inspect.isfunction)
            
            # Добавляем описания классов
            for name, cls in classes:
                class_doc = docstring_parser.parse(cls.__doc__ or "")
                class_section = DocSection(
                    title=f"Class: {name}",
                    content=class_doc.short_description or "",
                    subsections=[]
                )
                
                # Добавляем методы класса
                for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                    method_doc = docstring_parser.parse(method.__doc__ or "")
                    method_section = DocSection(
                        title=f"Method: {method_name}",
                        content=method_doc.short_description or "",
                        metadata={
                            "parameters": [
                                {"name": p.arg_name, "description": p.description}
                                for p in method_doc.params
                            ],
                            "returns": method_doc.returns.description if method_doc.returns else None
                        }
                    )
                    class_section.subsections.append(method_section)
                    
                section.subsections.append(class_section)
                
            # Добавляем описания функций
            for name, func in functions:
                func_doc = docstring_parser.parse(func.__doc__ or "")
                func_section = DocSection(
                    title=f"Function: {name}",
                    content=func_doc.short_description or "",
                    metadata={
                        "parameters": [
                            {"name": p.arg_name, "description": p.description}
                            for p in func_doc.params
                        ],
                        "returns": func_doc.returns.description if func_doc.returns else None
                    }
                )
                section.subsections.append(func_section)
                
            return section
            
        except Exception as e:
            self.logger.error(
                "Error generating technical docs",
                error=str(e),
                module=str(module_path)
            )
            raise
            
    def generate_user_guide(self, title: str, sections: List[Dict[str, Any]]) -> DocSection:
        """
        Генерация руководства пользователя
        
        Args:
            title: название руководства
            sections: секции руководства
            
        Returns:
            DocSection: секция документации
        """
        guide = DocSection(
            title=title,
            content="",
            subsections=[]
        )
        
        for section in sections:
            section_doc = DocSection(
                title=section["title"],
                content=section["content"],
                subsections=[
                    DocSection(
                        title=sub["title"],
                        content=sub["content"]
                    )
                    for sub in section.get("subsections", [])
                ],
                metadata=section.get("metadata", {})
            )
            guide.subsections.append(section_doc)
            
        return guide
        
    def generate_api_docs(self, endpoints: List[APIEndpoint]) -> DocSection:
        """
        Генерация API документации
        
        Args:
            endpoints: список endpoint'ов
            
        Returns:
            DocSection: секция документации
        """
        api_docs = DocSection(
            title="API Documentation",
            content="API endpoints and their usage",
            subsections=[]
        )
        
        for endpoint in endpoints:
            endpoint_section = DocSection(
                title=f"{endpoint.method} {endpoint.path} - {endpoint.name}",
                content=endpoint.description,
                metadata={
                    "parameters": endpoint.parameters,
                    "returns": endpoint.returns,
                    "examples": endpoint.examples
                }
            )
            api_docs.subsections.append(endpoint_section)
            
        return api_docs
        
    def add_usage_example(
        self,
        section: DocSection,
        title: str,
        code: str,
        description: str,
        output: Optional[str] = None
    ):
        """
        Добавление примера использования
        
        Args:
            section: секция документации
            title: название примера
            code: код примера
            description: описание примера
            output: вывод примера (опционально)
        """
        example = DocSection(
            title=title,
            content=description,
            metadata={
                "code": code,
                "output": output
            }
        )
        section.subsections.append(example)
        
    def save_documentation(self, section: DocSection, filename: str):
        """
        Сохранение документации
        
        Args:
            section: секция документации
            filename: имя файла
        """
        try:
            # Преобразуем в Markdown
            content = self._section_to_markdown(section)
            
            # Сохраняем файл
            doc_file = self.docs_dir / filename
            with open(doc_file, "w") as f:
                f.write(content)
                
            self.logger.info(f"Documentation saved to {doc_file}")
            
        except Exception as e:
            self.logger.error(
                "Error saving documentation",
                error=str(e),
                filename=filename
            )
            raise
            
    def _section_to_markdown(self, section: DocSection, level: int = 1) -> str:
        """
        Преобразование секции в Markdown
        
        Args:
            section: секция документации
            level: уровень заголовка
            
        Returns:
            str: Markdown контент
        """
        content = []
        
        # Добавляем заголовок
        content.append(f"{'#' * level} {section.title}\n")
        
        # Добавляем основной контент
        if section.content:
            content.append(f"{section.content}\n")
            
        # Добавляем метаданные
        if section.metadata:
            if "parameters" in section.metadata:
                content.append("\n### Parameters\n")
                for param in section.metadata["parameters"]:
                    content.append(f"- `{param['name']}`: {param['description']}")
                    
            if "returns" in section.metadata and section.metadata["returns"]:
                content.append(f"\n### Returns\n\n{section.metadata['returns']}")
                
            if "code" in section.metadata:
                content.append(f"\n### Code\n\n```python\n{section.metadata['code']}\n```")
                
            if "output" in section.metadata and section.metadata["output"]:
                content.append(f"\n### Output\n\n```\n{section.metadata['output']}\n```")
                
        # Добавляем подсекции
        for subsection in section.subsections:
            content.append(self._section_to_markdown(subsection, level + 1))
            
        return "\n".join(content)
        
    def update_documentation(self, section: DocSection):
        """
        Обновление документации
        
        Args:
            section: секция документации
        """
        section.last_updated = datetime.now()
        self.sections[section.title] = section 