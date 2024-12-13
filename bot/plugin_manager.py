import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from plugins.plugin import Plugin

GOOGLE = ("google/gemini-flash-1.5-8b",)

class PluginManager:
    _instance = None  # Add singleton instance tracker
    
    def __new__(cls, config, plugins_directory="plugins"):
        if cls._instance is None:
            logging.info("Creating new PluginManager instance")  # Debug line
            cls._instance = super().__new__(cls)
            cls._instance.plugins = {}
            cls._instance.plugins_directory = plugins_directory
            cls._instance.enabled_plugins = config.get('plugins', [])  # Initialize enabled_plugins here

        else:
            logging.info("Reusing existing PluginManager instance")  # Debug line
        return cls._instance

    def __init__(self, config, plugins_directory="plugins"):
        # Обновляем enabled_plugins при каждой инициализации
        self.enabled_plugins = config.get('plugins', [])
        if not hasattr(self, 'plugins'):
            self.plugins = {}
        # Получаем абсолютный путь относительно текущей директории
        current_dir = Path(__file__).parent
        self.plugins_directory = current_dir / plugins_directory

        # Всегда вызываем load_plugins при инициализации
        self.load_plugins()

    def load_plugins(self):
        """Загружает все плагины из указанной директории."""
        # Удалим проверку на существующие плагины
        self.plugins.clear()  # Очищаем существующие плагины перед загрузкой
                
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
            plugin_path = Path(f"./plugins/{plugin_name}.py")
            logging.info(f"Attempting to load plugin from path: {plugin_path.absolute()}")
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            return plugin_module
        except Exception as e:
            logging.info(f"Ошибка при загрузке плагина {plugin_name}: {e}")
            return None

    def register_plugin(self, plugin_name: str, plugin_module: Any) -> None:
        """Регистрирует плагин, если в нем есть метод execute."""
        try:
            # Add check if plugin is already registered
            if plugin_name in self.plugins:
                logging.info(f"Плагин {plugin_name} уже зарегистрирован.")
                return
                    
            for name, obj in inspect.getmembers(plugin_module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin and  # Skip the base Plugin class
                    hasattr(obj, "execute")):
                    self.plugins[plugin_name] = obj
                    # Create instance and cache it
                    plugin_instance = obj()
                    logging.info(f"Плагин {plugin_name} успешно зарегистрирован.")
                    return

            logging.info(f"Плагин {plugin_name} не имеет метода 'execute' или не наследуется от Plugin.")
        except Exception as e:
            logging.info(f"Ошибка при регистрации плагина {plugin_name}: {str(e)}")

    async def execute(self, plugin_name, user_id, *args):
        """Единый интерфейс для вызова плагинов."""
        if plugin_name not in self.plugins:
            return f"Плагин {plugin_name} не найден."
        
        plugin_class = self.plugins[plugin_name](user_id)
        
        # Вызов метода execute плагина
        if hasattr(plugin_class, "execute"):
            result = await plugin_class.execute(*args)
            return result
        else:
            return f"Плагин {plugin_name} не поддерживает метод execute."

    def reinitialize(self):
        """Переинициализирует плагин менеджер (перезагружает плагины)."""
        logging.info("Переинициализация плагинов...")
        self.plugins.clear()  # Очищаем список зарегистрированных плагинов
        self.load_plugins()  # Перезагружаем плагины из директории
        logging.info("Плагины переинициализированы.")

    def get_functions_specs(self, helper, model_to_use):
        """
        Return the list of function specs that can be called by the model
        """
        seen_functions = set()
        all_specs = []
        for plugin_name, plugin_class in self.plugins.items():
            try:
                # Create an instance of the plugin class and pass the required parameters
                plugin_instance = plugin_class() if hasattr(plugin_class, '__init__') and len(inspect.signature(plugin_class.__init__).parameters) > 1 else plugin_class()
                specs = plugin_instance.get_spec()
                for spec in specs:
                    if spec and spec.get('name') not in seen_functions:
                        seen_functions.add(spec.get('name'))
                        all_specs.append(spec)
            except Exception as e:
                logging.error(f"Error instantiating plugin {plugin_name}: {str(e)}")
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
            parsed_args = json.loads(arguments)
            result = await plugin.execute(function_name, helper, **parsed_args)
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error executing function {function_name}: {str(e)}")
            return json.dumps({'error': f'Error executing function: {str(e)}'})
        
    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        return next((plugin_class() for plugin_class in self.plugins.values()
                    if function_name in map(lambda spec: spec.get('name'), plugin_class().get_spec())), None)

    def get_plugin(self, plugin_name):
        """
        Returns the plugin instance with the given name
        
        :param plugin_name: The name of the plugin
        :return: The plugin instance or None if not found
        """
        # Simply look up the plugin class by name in self.plugins
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class:
            return plugin_class()
        return None
        
    def get_all_plugin_descriptions(self) -> list[str]:
        """Get all plugin descriptions from their get_spec methods."""
        descriptions = []
        
        # Iterate through all registered plugins
        for plugin_name, plugin_class in self.plugins.items():
            try:
                # Create plugin instance
                plugin_instance = plugin_class()
                
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
                logging.error(f"Error getting description from plugin {plugin_name}: {str(e)}")
                continue
                
        return descriptions

    def get_plugin_spec(self, plugin_name: str) -> List[Dict]:
        """Возвращает спецификацию плагина по имени"""
        if not self.has_plugin(plugin_name):
            return None
        try:
            plugin_instance = self.plugins[plugin_name]()
            return plugin_instance.get_spec()
        except:
            return None

    def has_plugin(self, plugin_name: str) -> bool:
        """Проверяет существование плагина по имени"""
        return plugin_name in self.plugins

