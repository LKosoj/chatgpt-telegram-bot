"""
Тесты для системы документации
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
from ..documentation_system import (
    DocumentationSystem,
    DocSection,
    APIEndpoint
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def docs_dir(tmp_path):
    """Фикстура для директории документации"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return docs_dir


@pytest.fixture
def doc_system(docs_dir, logger):
    """Фикстура для системы документации"""
    return DocumentationSystem(str(docs_dir), logger)


@pytest.fixture
def test_module(docs_dir):
    """Фикстура для тестового модуля"""
    module_path = docs_dir / "test_module.py"
    
    # Создаем тестовый модуль
    content = '''
"""Test module docstring"""

class TestClass:
    """Test class docstring"""
    
    def test_method(self, param1: str, param2: int) -> str:
        """
        Test method docstring
        
        Args:
            param1: First parameter
            param2: Second parameter
            
        Returns:
            str: Return value
        """
        return "test"

def test_function(param: str) -> None:
    """
    Test function docstring
    
    Args:
        param: Test parameter
    """
    pass
'''
    
    with open(module_path, "w") as f:
        f.write(content)
        
    return module_path


def test_documentation_system_initialization(docs_dir, logger):
    """Тест инициализации системы документации"""
    system = DocumentationSystem(str(docs_dir), logger)
    
    assert system.docs_dir == docs_dir
    assert system.logger == logger
    assert isinstance(system.sections, dict)


def test_generate_technical_docs(doc_system, test_module):
    """Тест генерации технической документации"""
    section = doc_system.generate_technical_docs(test_module)
    
    assert isinstance(section, DocSection)
    assert section.title == f"Technical Documentation - {test_module.stem}"
    assert len(section.subsections) > 0
    
    # Проверяем описание класса
    class_section = next(
        s for s in section.subsections
        if s.title == "Class: TestClass"
    )
    assert class_section.content == "Test class docstring"
    
    # Проверяем описание метода
    method_section = class_section.subsections[0]
    assert method_section.title == "Method: test_method"
    assert len(method_section.metadata["parameters"]) == 2
    assert method_section.metadata["returns"] == "Return value"
    
    # Проверяем описание функции
    func_section = next(
        s for s in section.subsections
        if s.title == "Function: test_function"
    )
    assert func_section.content == "Test function docstring"
    assert len(func_section.metadata["parameters"]) == 1


def test_generate_user_guide(doc_system):
    """Тест генерации руководства пользователя"""
    sections = [
        {
            "title": "Section 1",
            "content": "Content 1",
            "subsections": [
                {
                    "title": "Subsection 1.1",
                    "content": "Content 1.1"
                }
            ]
        },
        {
            "title": "Section 2",
            "content": "Content 2"
        }
    ]
    
    guide = doc_system.generate_user_guide("Test Guide", sections)
    
    assert isinstance(guide, DocSection)
    assert guide.title == "Test Guide"
    assert len(guide.subsections) == 2
    
    # Проверяем первую секцию
    section1 = guide.subsections[0]
    assert section1.title == "Section 1"
    assert section1.content == "Content 1"
    assert len(section1.subsections) == 1
    
    # Проверяем подсекцию
    subsection = section1.subsections[0]
    assert subsection.title == "Subsection 1.1"
    assert subsection.content == "Content 1.1"


def test_generate_api_docs(doc_system):
    """Тест генерации API документации"""
    endpoints = [
        APIEndpoint(
            name="Test Endpoint",
            description="Test description",
            method="GET",
            path="/api/test",
            parameters=[
                {
                    "name": "param1",
                    "description": "Parameter 1"
                }
            ],
            returns={
                "type": "string",
                "description": "Return value"
            },
            examples=[
                {
                    "request": "GET /api/test?param1=value",
                    "response": "Response"
                }
            ]
        )
    ]
    
    api_docs = doc_system.generate_api_docs(endpoints)
    
    assert isinstance(api_docs, DocSection)
    assert api_docs.title == "API Documentation"
    assert len(api_docs.subsections) == 1
    
    # Проверяем описание endpoint'а
    endpoint_section = api_docs.subsections[0]
    assert endpoint_section.title == "GET /api/test - Test Endpoint"
    assert endpoint_section.content == "Test description"
    assert len(endpoint_section.metadata["parameters"]) == 1
    assert endpoint_section.metadata["returns"] == {
        "type": "string",
        "description": "Return value"
    }


def test_add_usage_example(doc_system):
    """Тест добавления примера использования"""
    section = DocSection(
        title="Test Section",
        content="Test content"
    )
    
    doc_system.add_usage_example(
        section,
        "Test Example",
        "print('test')",
        "Example description",
        "test"
    )
    
    assert len(section.subsections) == 1
    
    example = section.subsections[0]
    assert example.title == "Test Example"
    assert example.content == "Example description"
    assert example.metadata["code"] == "print('test')"
    assert example.metadata["output"] == "test"


def test_save_documentation(doc_system):
    """Тест сохранения документации"""
    section = DocSection(
        title="Test Documentation",
        content="Test content",
        subsections=[
            DocSection(
                title="Subsection",
                content="Subsection content",
                metadata={
                    "parameters": [
                        {
                            "name": "param",
                            "description": "Parameter description"
                        }
                    ],
                    "returns": "Return description",
                    "code": "test_code()",
                    "output": "test output"
                }
            )
        ]
    )
    
    doc_system.save_documentation(section, "test_doc.md")
    
    # Проверяем файл
    doc_file = doc_system.docs_dir / "test_doc.md"
    assert doc_file.exists()
    
    # Проверяем содержимое
    content = doc_file.read_text()
    assert "# Test Documentation" in content
    assert "## Subsection" in content
    assert "### Parameters" in content
    assert "### Returns" in content
    assert "### Code" in content
    assert "### Output" in content


def test_update_documentation(doc_system):
    """Тест обновления документации"""
    section = DocSection(
        title="Test Section",
        content="Test content"
    )
    
    # Сохраняем и обновляем
    old_timestamp = section.last_updated
    doc_system.update_documentation(section)
    
    assert section.last_updated > old_timestamp
    assert doc_system.sections["Test Section"] == section 