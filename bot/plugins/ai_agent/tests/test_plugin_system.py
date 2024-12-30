"""
Тесты для системы плагинов
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from ..plugin_system import (
    Plugin,
    PluginManager,
    PluginMetadata,
    PluginError
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def plugin_dir(tmp_path):
    """Фикстура для директории плагинов"""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    return plugins_dir


@pytest.fixture
def test_plugin_dir(plugin_dir):
    """Фикстура для тестового плагина"""
    # Создаем директорию плагина
    plugin_dir = plugin_dir / "test_plugin"
    plugin_dir.mkdir()
    
    # Создаем metadata.json
    metadata = {
        "name": "test_plugin",
        "version": "1.0.0",
        "description": "Test plugin",
        "author": "Test Author",
        "dependencies": {},
        "entry_point": "plugin",
        "enabled": True
    }
    
    with open(plugin_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
        
    # Создаем plugin.py
    plugin_code = '''
from bot.plugins.ai_agent.plugin_system import Plugin

class Plugin(Plugin):
    async def initialize(self):
        self.initialized = True
        
    async def shutdown(self):
        self.initialized = False
        
    def configure(self, config):
        self.config = config
'''
    
    with open(plugin_dir / "plugin.py", "w") as f:
        f.write(plugin_code)
        
    return plugin_dir


@pytest.mark.asyncio
async def test_plugin_metadata():
    """Тест метаданных плагина"""
    metadata = PluginMetadata(
        name="test",
        version="1.0.0",
        description="Test plugin",
        author="Test Author"
    )
    
    assert metadata.name == "test"
    assert metadata.version == "1.0.0"
    assert metadata.description == "Test plugin"
    assert metadata.author == "Test Author"
    assert metadata.dependencies == {}
    assert metadata.entry_point == "plugin"
    assert metadata.enabled is True
    
    # Тест сериализации
    data = metadata.to_dict()
    assert isinstance(data, dict)
    assert data["name"] == "test"
    
    # Тест десериализации
    metadata2 = PluginMetadata.from_dict(data)
    assert metadata2.name == metadata.name
    assert metadata2.version == metadata.version


@pytest.mark.asyncio
async def test_plugin_manager_initialization(plugin_dir, logger):
    """Тест инициализации менеджера плагинов"""
    manager = PluginManager(str(plugin_dir), logger)
    
    assert manager.plugins_dir == plugin_dir
    assert manager.logger == logger
    assert isinstance(manager.plugins, dict)
    assert isinstance(manager.metadata, dict)


@pytest.mark.asyncio
async def test_plugin_loading(test_plugin_dir, logger):
    """Тест загрузки плагина"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем плагин
    success = await manager.load_plugin(test_plugin_dir)
    assert success is True
    
    # Проверяем, что плагин загружен
    assert "test_plugin" in manager.plugins
    assert "test_plugin" in manager.metadata
    
    plugin = manager.get_plugin("test_plugin")
    assert plugin is not None
    assert hasattr(plugin, "initialized")
    assert plugin.initialized is True


@pytest.mark.asyncio
async def test_plugin_unloading(test_plugin_dir, logger):
    """Тест выгрузки плагина"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем и выгружаем плагин
    await manager.load_plugin(test_plugin_dir)
    success = await manager.unload_plugin("test_plugin")
    assert success is True
    
    # Проверяем, что плагин выгружен
    assert "test_plugin" not in manager.plugins
    assert "test_plugin" not in manager.metadata


@pytest.mark.asyncio
async def test_plugin_configuration(test_plugin_dir, logger):
    """Тест конфигурации плагина"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем плагин
    await manager.load_plugin(test_plugin_dir)
    
    # Конфигурируем плагин
    config = {"test_setting": "test_value"}
    success = manager.configure_plugin("test_plugin", config)
    assert success is True
    
    # Проверяем конфигурацию
    plugin = manager.get_plugin("test_plugin")
    assert plugin.config == config


@pytest.mark.asyncio
async def test_plugin_version_check(test_plugin_dir, logger):
    """Тест проверки версий плагина"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем первую версию
    await manager.load_plugin(test_plugin_dir)
    
    # Обновляем метаданные на более старую версию
    metadata = {
        "name": "test_plugin",
        "version": "0.9.0",
        "description": "Test plugin",
        "author": "Test Author"
    }
    
    with open(test_plugin_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
        
    # Пытаемся загрузить старую версию
    success = await manager.load_plugin(test_plugin_dir)
    assert success is False
    
    # Проверяем, что осталась старая версия
    plugin_meta = manager.get_metadata("test_plugin")
    assert plugin_meta.version == "1.0.0"


@pytest.mark.asyncio
async def test_plugin_enable_disable(test_plugin_dir, logger):
    """Тест включения/отключения плагина"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем плагин
    await manager.load_plugin(test_plugin_dir)
    
    # Отключаем плагин
    success = await manager.disable_plugin("test_plugin")
    assert success is True
    assert "test_plugin" not in manager.plugins
    
    # Включаем плагин
    success = await manager.enable_plugin("test_plugin")
    assert success is True
    assert "test_plugin" in manager.plugins


@pytest.mark.asyncio
async def test_plugin_list(test_plugin_dir, logger):
    """Тест списка плагинов"""
    manager = PluginManager(str(test_plugin_dir.parent), logger)
    
    # Загружаем плагин
    await manager.load_plugin(test_plugin_dir)
    
    # Получаем список плагинов
    plugins = manager.list_plugins()
    assert len(plugins) == 1
    assert plugins[0]["name"] == "test_plugin"
    assert plugins[0]["version"] == "1.0.0"
    assert plugins[0]["enabled"] is True


@pytest.mark.asyncio
async def test_plugin_error_handling(plugin_dir, logger):
    """Тест обработки ошибок"""
    manager = PluginManager(str(plugin_dir), logger)
    
    # Создаем некорректный плагин
    bad_plugin_dir = plugin_dir / "bad_plugin"
    bad_plugin_dir.mkdir()
    
    # Пытаемся загрузить некорректный плагин
    success = await manager.load_plugin(bad_plugin_dir)
    assert success is False
    
    # Проверяем, что плагин не загружен
    assert "bad_plugin" not in manager.plugins
    assert "bad_plugin" not in manager.metadata 