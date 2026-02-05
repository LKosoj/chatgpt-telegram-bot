import importlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
import difflib

from .plugins.plugin import Plugin
from .model_constants import GOOGLE as GOOGLE_MODELS
from .validation import validate_function_args

logger = logging.getLogger(__name__)

class PluginManager:

    def __init__(self, config, plugins_directory="plugins"):
        self.plugins = {}
        self.plugin_instances = {}
        self.openai = None
        self.bot = None
        self.enabled_plugins = [p for p in (config.get('plugins', []) or []) if p]
        self.strict_validation = str(os.getenv("PLUGIN_STRICT_VALIDATION", "false")).lower() == "true"
        self.storage_root = os.getenv("PLUGIN_STORAGE_ROOT")
        if not self.storage_root:
            self.storage_root = str((Path(__file__).parent / "config").resolve())
        os.makedirs(self.storage_root, exist_ok=True)

        current_dir = Path(__file__).parent
        self.plugins_directory = current_dir / plugins_directory

        self.load_plugins()

    def set_openai(self, openai):
        """Устанавливает ссылку на openai и обновляет все существующие инстансы плагинов"""
        self.openai = openai
        self.bot = getattr(openai, "bot", None)
        for instance in self.plugin_instances.values():
            instance.initialize(openai=openai, bot=self.bot, storage_root=self.storage_root)

    def load_plugins(self):
        """Загружает все плагины из указанной директории."""
        # Очищаем существующие плагины и их инстансы перед загрузкой
        self.plugins.clear()
        self.plugin_instances.clear()

        plugins_path = Path(self.plugins_directory)
        excluded_files = {'__init__.py', 'plugin.py'}

        for plugin_file in sorted(plugins_path.glob("*.py")):
            if plugin_file.name not in excluded_files:
                plugin_name = plugin_file.stem
                if self.enabled_plugins and plugin_name not in self.enabled_plugins:
                    continue
                plugin_module = self.load_plugin_module(plugin_name)
                if plugin_module:
                    self.register_plugin(plugin_name, plugin_module)

        self._validate_enabled_plugins()

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

                specs = self._normalize_specs(plugin_instance.get_spec(), plugin_instance)
                for spec in specs:
                    if spec and spec.get('name') not in seen_functions:
                        seen_functions.add(spec.get('name'))
                        all_specs.append(spec)
                    elif spec:
                        msg = f"Duplicate function name detected: {spec.get('name')} in {plugin_name}"
                        if self.strict_validation:
                            raise ValueError(msg)
                        logger.warning(msg)
            except Exception as e:
                logger.error(f"Error instantiating plugin {plugin_name}: {str(e)}")
                continue

        if model_to_use in GOOGLE_MODELS:
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

            spec = self.get_spec_by_function_name(function_name)
            if spec:
                errors = validate_function_args(spec, parsed_args)
                if errors:
                    return json.dumps({'error': f'Invalid args for {function_name}: {errors}'}, ensure_ascii=False)

            logger.debug(f"Вызываем функцию {function_name} с аргументами: {parsed_args}")
            base_name = function_name.split(".", 1)[-1]
            result = await plugin.execute(base_name, helper, **parsed_args)

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

    def get_spec_by_function_name(self, function_name):
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return None
        specs = self._normalize_specs(plugin.get_spec(), plugin)
        for spec in specs:
            if spec.get("name") == function_name:
                return spec
        return None

    def __get_plugin_by_function_name(self, function_name):
        """
        Находит плагин по имени функции
        """
        if "." in function_name:
            prefix = function_name.split(".", 1)[0]
            for plugin_name in self.plugins.keys():
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue
                if plugin_instance.get_function_prefix() == prefix:
                    return plugin_instance

        for plugin_name, plugin_class in self.plugins.items():
            plugin_instance = self.get_plugin(plugin_name)
            if not plugin_instance:
                continue

            specs = self._normalize_specs(plugin_instance.get_spec(), plugin_instance)
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
            if not hasattr(instance, 'openai') or not instance.openai:
                instance.initialize(openai=self.openai, bot=self.bot, storage_root=self.storage_root)
            if not getattr(instance, "plugin_id", None):
                instance.plugin_id = plugin_name
            if not getattr(instance, "function_prefix", None):
                instance.function_prefix = instance.plugin_id
            return instance

        # Если инстанса нет в кеше, создаем новый
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class:
            instance = plugin_class()
            if not getattr(instance, "plugin_id", None):
                instance.plugin_id = plugin_name
            if not getattr(instance, "function_prefix", None):
                instance.function_prefix = instance.plugin_id
            if self.openai or self.storage_root:
                instance.initialize(openai=self.openai, bot=self.bot, storage_root=self.storage_root)
            self.plugin_instances[plugin_name] = instance
            return instance
        return None

    def close_all(self):
        for instance in self.plugin_instances.values():
            try:
                instance.close()
            except Exception as exc:
                logger.warning(f"Error closing plugin {instance}: {exc}")

    def _normalize_specs(self, specs: List[Dict], plugin_instance: Plugin) -> List[Dict]:
        normalized = []
        prefix = plugin_instance.get_function_prefix()
        for spec in specs:
            if not spec or "name" not in spec:
                continue
            name = spec["name"]
            if "." not in name:
                spec = dict(spec)
                spec["name"] = f"{prefix}.{name}"
            normalized.append(spec)
        return normalized

    def _validate_enabled_plugins(self):
        if not self.enabled_plugins:
            return
        missing = [p for p in self.enabled_plugins if p and p not in self.plugins]
        if not missing:
            return

        available_files = self._get_available_plugin_files()
        failed_to_load = [p for p in missing if p in available_files]
        unknown = [p for p in missing if p not in available_files]

        messages = []
        if failed_to_load:
            messages.append(f"Enabled plugins failed to load: {failed_to_load}")
        if unknown:
            suggestions = {
                name: self._suggest_plugin_names(name, available_files)
                for name in unknown
            }
            messages.append(
                "Enabled plugins not found: "
                f"{unknown}. Suggestions: {suggestions}"
            )

        msg = " | ".join(messages)
        if self.strict_validation:
            raise ValueError(msg)
        logger.error(msg)

    def _get_available_plugin_files(self) -> list[str]:
        plugins_path = Path(self.plugins_directory)
        excluded_files = {'__init__.py', 'plugin.py'}
        return [
            p.stem
            for p in plugins_path.glob("*.py")
            if p.name not in excluded_files
        ]

    def _suggest_plugin_names(self, name: str, candidates: list[str]) -> list[str]:
        if not name or not candidates:
            return []
        return difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)

    def filter_allowed_plugins(self, allowed_plugins: List[str] | None) -> List[str]:
        if not allowed_plugins or allowed_plugins == ['None']:
            return []
        if allowed_plugins == ['All']:
            return ['All']
        missing = [p for p in allowed_plugins if p not in self.plugins]
        if missing:
            msg = f"Allowed plugins not found: {missing}"
            if self.strict_validation:
                raise ValueError(msg)
            logger.error(msg)
        return [p for p in allowed_plugins if p in self.plugins]

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
                specs = self._normalize_specs(plugin_instance.get_spec(), plugin_instance)

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
            return self._normalize_specs(plugin_instance.get_spec(), plugin_instance)
        except:
            return None

    def has_plugin(self, plugin_name: str) -> bool:
        """Проверяет существование плагина по имени"""
        return plugin_name in self.plugins

    def get_plugin_commands(self) -> List[Dict]:
        """Возвращает список всех команд от всех плагинов с базовой валидацией."""
        commands = []
        seen_commands = set()
        for plugin_name in self.plugins.keys():
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue

                plugin_commands = plugin_instance.get_commands() or []
                for cmd in plugin_commands:
                    normalized = self._validate_and_normalize_command(cmd, plugin_name)
                    if not normalized:
                        continue
                    if normalized.get("command"):
                        if normalized["command"] in seen_commands:
                            logger.error(
                                f"Duplicate plugin command '{normalized['command']}' from {plugin_name}"
                            )
                            continue
                        seen_commands.add(normalized["command"])
                    commands.append(normalized)
            except Exception as e:
                logger.error(f"Ошибка при получении команд плагина {plugin_name}: {e}")
        return commands

    def build_bot_commands(self) -> Dict[str, List[Dict]]:
        """
        Build plugin command registrations and menu entries.
        Returns:
          - plugin_commands: validated command dicts (for handlers)
          - menu_entries: list of dicts with command/description for menus
        """
        plugin_commands = self.get_plugin_commands()
        menu_entries = []
        for cmd in plugin_commands:
            if cmd.get("add_to_menu") and cmd.get("command") and cmd.get("description"):
                menu_entries.append({
                    "command": cmd["command"],
                    "description": cmd["description"],
                })
        return {"plugin_commands": plugin_commands, "menu_entries": menu_entries}

    def _validate_and_normalize_command(self, cmd: Dict, plugin_name: str) -> Dict | None:
        if not isinstance(cmd, dict):
            logger.error(f"Invalid command definition from {plugin_name}: not a dict")
            return None

        if cmd.get("callback_query_handler") and cmd.get("callback_pattern"):
            cmd = dict(cmd)
            cmd["plugin_name"] = plugin_name
            cmd.setdefault("handler_kwargs", {})
            return cmd

        command = cmd.get("command")
        description = cmd.get("description")
        handler = cmd.get("handler")
        if not command or not description or not handler:
            logger.error(f"Invalid command definition from {plugin_name}: {cmd}")
            return None

        if isinstance(command, str) and command.startswith("/"):
            command = command[1:]

        if not isinstance(command, str) or " " in command:
            logger.error(f"Invalid command name from {plugin_name}: {command}")
            return None

        if not callable(handler):
            logger.error(f"Invalid handler for command '{command}' from {plugin_name}")
            return None

        normalized = dict(cmd)
        normalized["command"] = command
        normalized.setdefault("handler_kwargs", {})
        normalized.setdefault("add_to_menu", True)
        normalized["plugin_name"] = plugin_name
        return normalized

    def get_message_handlers(self) -> List[Dict]:
        """Возвращает список обработчиков сообщений от всех плагинов."""
        handlers = []
        for plugin_name in self.plugins.keys():
            plugin_instance = self.get_plugin(plugin_name)
            if plugin_instance:
                plugin_handlers = plugin_instance.get_message_handlers()
                for handler in plugin_handlers:
                    if 'plugin_name' not in handler:
                        handler['plugin_name'] = plugin_name
                handlers.extend(plugin_handlers)
        return handlers
