import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .plugins.plugin import Plugin

logger = logging.getLogger(__name__)
GOOGLE = ("google/gemini-flash-1.5-8b",)

class PluginManager:
    _instance = None  # Add singleton instance tracker

    def __new__(cls, config, plugins_directory="bot/plugins"):
        if cls._instance is None:
            logger.info("Creating new PluginManager instance")  # Debug line
            cls._instance = super().__new__(cls)
            cls._instance.plugins = {}
            cls._instance.plugin_instances = {}  # Кеш инстансов плагинов
            cls._instance.plugins_directory = plugins_directory
            cls._instance.enabled_plugins = config.get('plugins', [])  # Initialize enabled_plugins here
            cls._instance.openai = None  # Добавляем ссылку на openai

        else:
            logger.info("Reusing existing PluginManager instance")  # Debug line
        return cls._instance

    def __init__(self, config, plugins_directory="plugins"):
        # Обновляем enabled_plugins при каждой инициализации
        self.enabled_plugins = config.get('plugins', [])
        if not hasattr(self, 'plugins'):
            self.plugins = {}
        if not hasattr(self, 'plugin_instances'):
            self.plugin_instances = {}
        if not hasattr(self, 'openai'):
            self.openai = None
        # Получаем абсолютный путь относительно текущей директории
        current_dir = Path(__file__).parent
        self.plugins_directory = current_dir / plugins_directory

        # Всегда вызываем load_plugins при инициализации
        self.load_plugins()

    def set_openai(self, openai):
        """Устанавливает ссылку на openai и обновляет все существующие инстансы плагинов"""
        self.openai = openai
        # Обновляем openai во всех существующих инстансах
        for instance in self.plugin_instances.values():
            instance.openai = openai
            instance.bot = openai.bot

    def load_plugins(self):
        """Загружает все плагины из указанной директории."""
        # Очищаем существующие плагины и их инстансы перед загрузкой
        self.plugins.clear()
        self.plugin_instances.clear()

        plugins_path = Path(self.plugins_directory)
        excluded_files = {'__init__.py', 'plugin.py'}

        for plugin_file in plugins_path.glob("*.py"):
            if plugin_file.name not in excluded_files:
                plugin_name = plugin_file.stem
                if plugin_name in self.enabled_plugins:
                    plugin_module = self.load_plugin_module(plugin_name)
                    if plugin_module:
                        self.register_plugin(plugin_name, plugin_module)

    def load_plugin_module(self, plugin_name):
        """Загружает модуль плагина по имени."""
        try:
            #plugin_path = Path(self.plugins_directory) / f"{plugin_name}.py"
            plugin_path = Path(self.plugins_directory) / f"{plugin_name}.py"
            spec = importlib.util.spec_from_file_location(
                'bot.plugins.' + plugin_name, plugin_path
            )
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            return plugin_module
        except Exception as e:
            logger.info(f"Ошибка при загрузке плагина {plugin_name}: {e}")
            return None

    def register_plugin(self, plugin_name, plugin_module):
        """Регистрирует плагин в менеджере"""
        try:
            # Получаем класс плагина из модуля
            plugin_classes = [cls for name, cls in inspect.getmembers(plugin_module, inspect.isclass)
                            if issubclass(cls, Plugin) and cls != Plugin]

            if not plugin_classes:
                logger.warning(f"No plugin class found in {plugin_name}")
                return

            # Регистрируем первый найденный класс плагина
            self.plugins[plugin_name] = plugin_classes[0]
            logger.info(f"Successfully registered plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Error registering plugin {plugin_name}: {e}")

    async def execute(self, plugin_name, user_id, *args):
        """Единый интерфейс для вызова плагинов."""
        if plugin_name not in self.plugins:
            return f"Плагин {plugin_name} не найден."

        # Используем get_plugin и передаем user_id при создании
        plugin_instance = self.get_plugin(plugin_name)
        if not plugin_instance:
            return f"Не удалось создать экземпляр плагина {plugin_name}."

        if hasattr(plugin_instance, "execute"):
            result = await plugin_instance.execute(*args)
            return result
        else:
            return f"Плагин {plugin_name} не поддерживает метод execute."

    def reinitialize(self):
        """Переинициализирует плагин менеджер (перезагружает плагины)."""
        logger.info("Переинициализация плагинов...")
        self.plugins.clear()  # Очищаем список зарегистрированных плагинов
        self.load_plugins()  # Перезагружаем плагины из директории
        logger.info("Плагины переинициализированы.")

    def get_functions_specs(self, helper, model_to_use, allowed_plugins=None):
        """
        Return the list of function specs that can be called by the model

        :param helper: OpenAIHelper instance
        :param model_to_use: Model name
        :param allowed_plugins: List of allowed plugin names or ['All'] or ['None']
        :return: List of function specifications
        """
        # Если разрешенных плагинов нет или ['None'], возвращаем пустой список
        if not allowed_plugins or allowed_plugins == ['None']:
            return []

        seen_functions = set()
        all_specs = []

        # Перебираем все плагины
        for plugin_name, plugin_class in self.plugins.items():
            # Пропускаем плагин если он не в списке разрешенных (кроме случая ['All'])
            if allowed_plugins != ['All'] and plugin_name not in allowed_plugins:
                continue

            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue

                specs = plugin_instance.get_spec()
                for spec in specs:
                    if spec and spec.get('name') not in seen_functions:
                        seen_functions.add(spec.get('name'))
                        all_specs.append(spec)
            except Exception as e:
                logger.error(f"Error instantiating plugin {plugin_name}: {str(e)}")
                continue

        if model_to_use in (GOOGLE):
            return {"function_declarations": all_specs}
        return [{"type": "function", "function": spec} for spec in all_specs]

    async def call_function(self, function_name, helper, arguments):
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return json.dumps({'error': f'Function {function_name} not found'})

        try:
            logger.debug(f"Пытаемся разобрать аргументы функции {function_name}: {arguments}")
            parsed_args = json.loads(arguments)

            logger.debug(f"Вызываем функцию {function_name} с аргументами: {parsed_args}")
            result = await plugin.execute(function_name, helper, **parsed_args)

            logger.debug(f"Результат выполнения функции {function_name}: {result}")
            return json.dumps(result, default=str, ensure_ascii=False)

        except json.JSONDecodeError as e:
            error_msg = f"Ошибка разбора JSON аргументов функции {function_name}: {e}, Аргументы: {arguments}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg}, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Ошибка выполнения функции {function_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg}, ensure_ascii=False)

    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        """
        Находит плагин по имени функции
        """
        for plugin_name, plugin_class in self.plugins.items():
            plugin_instance = self.get_plugin(plugin_name)
            if not plugin_instance:
                continue

            specs = plugin_instance.get_spec()
            if any(spec.get('name') == function_name for spec in specs):
                return plugin_instance

        return None

    def get_plugin(self, plugin_name):
        """
        Returns the plugin instance with the given name

        :param plugin_name: The name of the plugin
        :return: The plugin instance or None if not found
        """
        # Проверяем кеш инстансов
        if plugin_name in self.plugin_instances:
            instance = self.plugin_instances[plugin_name]
            # Убеждаемся, что у инстанса есть openai
            if not hasattr(instance, 'openai') or not instance.openai:
                instance.openai = self.openai
                instance.bot = self.openai.bot
            return instance

        # Если инстанса нет в кеше, создаем новый
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class:
            instance = plugin_class()
            if self.openai:
                instance.openai = self.openai
                instance.bot = self.openai.bot
            self.plugin_instances[plugin_name] = instance
            return instance
        return None

    def get_all_plugin_descriptions(self) -> list[str]:
        """Get all plugin descriptions from their get_spec methods."""
        descriptions = []

        # Iterate through all registered plugins
        for plugin_name in self.plugins.keys():
            try:
                # Используем get_plugin вместо создания нового экземпляра
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue

                # Get specs from the plugin
                specs = plugin_instance.get_spec()

                # Extract descriptions from each spec
                for spec in specs:
                    if spec and "description" in spec:
                        descriptions.append({
                            "plugin": plugin_name,
                            "function": spec.get("name", "unknown"),
                            "description": spec["description"]
                        })

            except Exception as e:
                logger.error(f"Error getting description from plugin {plugin_name}: {str(e)}")
                continue

        return descriptions

    def get_plugin_spec(self, plugin_name: str) -> List[Dict]:
        """Возвращает спецификацию плагина по имени"""
        if not self.has_plugin(plugin_name):
            return None
        try:
            # Используем get_plugin вместо создания нового экземпляра
            plugin_instance = self.get_plugin(plugin_name)
            if not plugin_instance:
                return None
            return plugin_instance.get_spec()
        except:
            return None

    def has_plugin(self, plugin_name: str) -> bool:
        """Проверяет существование плагина по имени"""
        return plugin_name in self.plugins

    def get_plugin_commands(self) -> List[Dict]:
        """Возвращает список всех команд от всех плагинов"""
        commands = []
        for plugin_name in self.plugins.keys():
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue

                plugin_commands = plugin_instance.get_commands()
                # Добавляем имя плагина в каждую команду
                for cmd in plugin_commands:
                    cmd['plugin_name'] = plugin_name
                commands.extend(plugin_commands)
            except Exception as e:
                logger.error(f"Ошибка при получении команд плагина {plugin_name}: {e}")
        return commands

    def get_message_handlers(self) -> List[Dict]:
        """Возвращает список обработчиков сообщений от всех плагинов."""
        handlers = []
        for plugin_name in self.plugins.items():
            plugin_instance = self.get_plugin(plugin_name)
            if plugin_instance:
                handlers.extend(plugin_instance.get_message_handlers())
        return handlers

