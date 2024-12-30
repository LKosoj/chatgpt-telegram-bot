"""
Модуль для системы плагинов
"""

import os
import sys
import json
import importlib
import importlib.util
import pkg_resources
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from packaging import version
from .logging import AgentLogger
from .exceptions import AIAgentError


class PluginError(AIAgentError):
    """Ошибка плагина"""
    pass


@dataclass
class PluginMetadata:
    """Метаданные плагина"""
    name: str
    version: str
    description: str
    author: str
    dependencies: Dict[str, str] = field(default_factory=dict)
    entry_point: str = "plugin"
    config_schema: Optional[Dict[str, Any]] = None
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Создание из словаря"""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            dependencies=data.get("dependencies", {}),
            entry_point=data.get("entry_point", "plugin"),
            config_schema=data.get("config_schema"),
            enabled=data.get("enabled", True)
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "enabled": self.enabled
        }


class Plugin:
    """Базовый класс плагина"""
    
    def __init__(self, metadata: PluginMetadata):
        """
        Инициализация плагина
        
        Args:
            metadata: метаданные плагина
        """
        self.metadata = metadata
        self.config: Dict[str, Any] = {}
        
    async def initialize(self):
        """Инициализация плагина"""
        pass
        
    async def shutdown(self):
        """Завершение работы плагина"""
        pass
        
    def configure(self, config: Dict[str, Any]):
        """
        Конфигурация плагина
        
        Args:
            config: конфигурация
        """
        self.config = config


class PluginManager:
    """Менеджер плагинов"""
    
    def __init__(
        self,
        plugins_dir: str = "plugins",
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация менеджера
        
        Args:
            plugins_dir: директория с плагинами
            logger: логгер (опционально)
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or AgentLogger("logs")
        self.plugins: Dict[str, Plugin] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        
    def _load_metadata(self, plugin_dir: Path) -> Optional[PluginMetadata]:
        """
        Загрузка метаданных плагина
        
        Args:
            plugin_dir: директория плагина
            
        Returns:
            Optional[PluginMetadata]: метаданные плагина
        """
        try:
            metadata_file = plugin_dir / "metadata.json"
            if not metadata_file.exists():
                raise PluginError(
                    f"Metadata file not found: {metadata_file}"
                )
                
            with open(metadata_file, "r") as f:
                data = json.load(f)
                
            return PluginMetadata.from_dict(data)
            
        except Exception as e:
            self.logger.error(
                "Error loading plugin metadata",
                error=e,
                extra={"plugin_dir": str(plugin_dir)}
            )
            return None
            
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """
        Проверка зависимостей плагина
        
        Args:
            metadata: метаданные плагина
            
        Returns:
            bool: результат проверки
        """
        try:
            for package, version_spec in metadata.dependencies.items():
                try:
                    pkg = pkg_resources.working_set.by_key[package]
                    if not pkg_resources.working_set.find(
                        pkg_resources.Requirement.parse(
                            f"{package}{version_spec}"
                        )
                    ):
                        raise PluginError(
                            f"Dependency not satisfied: {package}{version_spec}"
                        )
                except KeyError:
                    raise PluginError(
                        f"Dependency not found: {package}"
                    )
                    
            return True
            
        except Exception as e:
            self.logger.error(
                "Error checking dependencies",
                error=e,
                extra={"plugin": metadata.name}
            )
            return False
            
    def _load_plugin_module(
        self,
        plugin_dir: Path,
        metadata: PluginMetadata
    ) -> Optional[Plugin]:
        """
        Загрузка модуля плагина
        
        Args:
            plugin_dir: директория плагина
            metadata: метаданные плагина
            
        Returns:
            Optional[Plugin]: экземпляр плагина
        """
        try:
            # Добавляем директорию плагина в путь импорта
            sys.path.insert(0, str(plugin_dir))
            
            # Импортируем модуль
            module = importlib.import_module(metadata.entry_point)
            
            # Получаем класс плагина
            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class or not issubclass(plugin_class, Plugin):
                raise PluginError(
                    f"Invalid plugin class in {metadata.entry_point}"
                )
                
            # Создаем экземпляр
            return plugin_class(metadata)
            
        except Exception as e:
            self.logger.error(
                "Error loading plugin module",
                error=e,
                extra={"plugin": metadata.name}
            )
            return None
            
        finally:
            # Восстанавливаем путь импорта
            if str(plugin_dir) in sys.path:
                sys.path.remove(str(plugin_dir))
                
    async def load_plugin(self, plugin_dir: Path) -> bool:
        """
        Загрузка плагина
        
        Args:
            plugin_dir: директория плагина
            
        Returns:
            bool: результат загрузки
        """
        try:
            # Загружаем метаданные
            metadata = self._load_metadata(plugin_dir)
            if not metadata:
                return False
                
            # Проверяем версию если плагин уже загружен
            if metadata.name in self.metadata:
                current = version.parse(self.metadata[metadata.name].version)
                new = version.parse(metadata.version)
                if new <= current:
                    self.logger.warning(
                        f"Plugin {metadata.name} version {metadata.version} "
                        f"is not newer than {current}"
                    )
                    return False
                    
            # Проверяем зависимости
            if not self._check_dependencies(metadata):
                return False
                
            # Загружаем модуль
            plugin = self._load_plugin_module(plugin_dir, metadata)
            if not plugin:
                return False
                
            # Инициализируем плагин
            await plugin.initialize()
            
            # Сохраняем плагин
            self.plugins[metadata.name] = plugin
            self.metadata[metadata.name] = metadata
            
            self.logger.info(
                f"Loaded plugin {metadata.name} v{metadata.version}"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Error loading plugin",
                error=e,
                extra={"plugin_dir": str(plugin_dir)}
            )
            return False
            
    async def unload_plugin(self, name: str) -> bool:
        """
        Выгрузка плагина
        
        Args:
            name: название плагина
            
        Returns:
            bool: результат выгрузки
        """
        try:
            if name not in self.plugins:
                return False
                
            # Завершаем работу плагина
            plugin = self.plugins[name]
            await plugin.shutdown()
            
            # Удаляем плагин
            del self.plugins[name]
            del self.metadata[name]
            
            self.logger.info(f"Unloaded plugin {name}")
            return True
            
        except Exception as e:
            self.logger.error(
                "Error unloading plugin",
                error=e,
                extra={"plugin": name}
            )
            return False
            
    async def load_all_plugins(self):
        """Загрузка всех плагинов"""
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                await self.load_plugin(plugin_dir)
                
    async def unload_all_plugins(self):
        """Выгрузка всех плагинов"""
        for name in list(self.plugins.keys()):
            await self.unload_plugin(name)
            
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Получение плагина
        
        Args:
            name: название плагина
            
        Returns:
            Optional[Plugin]: плагин
        """
        return self.plugins.get(name)
        
    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """
        Получение метаданных плагина
        
        Args:
            name: название плагина
            
        Returns:
            Optional[PluginMetadata]: метаданные
        """
        return self.metadata.get(name)
        
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        Список плагинов
        
        Returns:
            List[Dict[str, Any]]: список плагинов
        """
        return [
            {
                "name": name,
                "version": meta.version,
                "description": meta.description,
                "author": meta.author,
                "enabled": meta.enabled
            }
            for name, meta in self.metadata.items()
        ]
        
    async def enable_plugin(self, name: str) -> bool:
        """
        Включение плагина
        
        Args:
            name: название плагина
            
        Returns:
            bool: результат включения
        """
        if name not in self.metadata:
            return False
            
        metadata = self.metadata[name]
        if metadata.enabled:
            return True
            
        metadata.enabled = True
        return await self.load_plugin(
            self.plugins_dir / name
        )
        
    async def disable_plugin(self, name: str) -> bool:
        """
        Отключение плагина
        
        Args:
            name: название плагина
            
        Returns:
            bool: результат отключения
        """
        if name not in self.metadata:
            return False
            
        metadata = self.metadata[name]
        if not metadata.enabled:
            return True
            
        metadata.enabled = False
        return await self.unload_plugin(name)
        
    def validate_plugin_config(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Валидация конфигурации плагина
        
        Args:
            name: название плагина
            config: конфигурация
            
        Returns:
            bool: результат валидации
        """
        metadata = self.metadata.get(name)
        if not metadata or not metadata.config_schema:
            return True
            
        try:
            # TODO: Реализовать валидацию по схеме
            return True
        except Exception as e:
            self.logger.error(
                "Error validating plugin config",
                error=e,
                extra={"plugin": name}
            )
            return False
            
    def configure_plugin(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Конфигурация плагина
        
        Args:
            name: название плагина
            config: конфигурация
            
        Returns:
            bool: результат конфигурации
        """
        plugin = self.plugins.get(name)
        if not plugin:
            return False
            
        if not self.validate_plugin_config(name, config):
            return False
            
        try:
            plugin.configure(config)
            return True
        except Exception as e:
            self.logger.error(
                "Error configuring plugin",
                error=e,
                extra={"plugin": name}
            )
            return False 