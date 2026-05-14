import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List
import difflib

from .plugins.background import BackgroundTask
from .plugins.db_handle import DbHandle
from .plugins.plugin import Plugin
from .model_constants import GOOGLE as GOOGLE_MODELS
from .user_settings import (
    USER_DISABLED_SKILLS_SETTING,
    USER_DISABLED_PLUGINS_SETTING,
    get_user_settings,
    normalize_string_list,
)
from .tool_result import TOOL_METADATA_KEY, normalize_tool_result
from .validation import validate_function_args

logger = logging.getLogger(__name__)
FRAMEWORK_TOOL_ARGS = {"chat_id", "user_id", "message_id", "allowed_plugins", "disclosed_functions"}


@dataclass
class _RequestUserSettingsCache:
    manager_id: int
    user_id: int | None
    loaded: bool = False
    disabled_plugins: frozenset[str] = field(default_factory=frozenset)
    disabled_skills: frozenset[str] = field(default_factory=frozenset)


_request_user_settings_cache: ContextVar[_RequestUserSettingsCache | None] = ContextVar(
    "request_user_settings_cache",
    default=None,
)


class PluginManager:

    def __init__(self, config, plugins_directory="plugins"):
        self.config = dict(config or {})
        self.plugins = {}
        self.plugin_instances = {}
        self.openai = None
        self.bot = None
        self.db = None
        self.db_handle: DbHandle | None = None
        self._background_tasks: dict[str, asyncio.Task] = {}
        self.enabled_plugins = [p for p in (config.get('plugins', []) or []) if p]
        self.strict_validation = str(os.getenv("PLUGIN_STRICT_VALIDATION", "false")).lower() == "true"
        self.storage_root = os.getenv("PLUGIN_STORAGE_ROOT")
        if not self.storage_root:
            self.storage_root = str((Path(__file__).resolve().parent.parent / "data").resolve())
        os.makedirs(self.storage_root, exist_ok=True)

        current_dir = Path(__file__).parent
        self.plugins_directory = current_dir / plugins_directory

        self.load_plugins()

    def set_openai(self, openai):
        """Устанавливает ссылку на openai и обновляет все существующие инстансы плагинов"""
        self.openai = openai
        self.bot = getattr(openai, "bot", None)
        for plugin_name, instance in self.plugin_instances.items():
            if (
                getattr(instance, "openai", None) is openai
                and getattr(instance, "db_handle", self.db_handle) is self.db_handle
            ):
                continue
            self._call_initialize(
                instance,
                openai=openai,
                bot=self.bot,
                storage_root=self.storage_root,
                db=getattr(self, "db_handle", None),
                plugin_config=self._plugin_config_segment(instance),
            )

    def set_db(self, db) -> None:
        """Устанавливает ссылку на БД для чтения пользовательских настроек.

        Идемпотентен: безопасно вызывать повторно (и с тем же `db`, и с новым).
        Также пересоздаёт ``self.db_handle`` (async-фасад) при каждом вызове.
        """
        self.db = db
        self.db_handle = DbHandle(db) if db is not None else None
        if self.openai is not None:
            self.set_openai(self.openai)

    def _call_initialize(self, plugin: Plugin, **kwargs: Any) -> None:
        """Call ``plugin.initialize`` with only the kwargs its signature accepts.

        This is the compatibility shim that lets the framework pass new
        parameters (``db``, ``plugin_config``) without breaking plugins that
        still use the legacy ``(openai, bot, storage_root)`` signature.
        """
        method = getattr(plugin, "initialize", None)
        if method is None:
            return
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            # Builtin / C-implemented — fall back to passing everything and
            # tolerating TypeError at the call boundary.
            try:
                method(**kwargs)
            except TypeError:
                logger.debug(
                    "plugin %s initialize did not accept new kwargs; calling empty",
                    getattr(plugin, "plugin_id", None) or type(plugin).__name__,
                )
                method()
            return

        params = sig.parameters
        accepts_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_keyword:
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in params}

        method(**filtered)

    def _plugin_config_segment(self, plugin: Plugin) -> Dict[str, Any]:
        """Return the config slice for ``plugin``: keys whose name starts with its prefix.

        Keys are kept with their original prefix (plugin slices its own).
        """
        try:
            prefix = plugin.get_config_prefix()
        except Exception:  # noqa: BLE001 — never let plugin code break init
            prefix = None
        if not prefix:
            return {}
        config = getattr(self, "config", None) or {}
        return {k: v for k, v in config.items() if k.startswith(prefix)}

    @contextmanager
    def user_settings_scope(self, user_id: int | None):
        """Cache user settings-derived values for one request/task context."""
        current = _request_user_settings_cache.get()
        if current is not None and current.manager_id == id(self) and current.user_id == user_id:
            yield
            return
        token = _request_user_settings_cache.set(
            _RequestUserSettingsCache(manager_id=id(self), user_id=user_id)
        )
        try:
            yield
        finally:
            _request_user_settings_cache.reset(token)

    def _request_user_settings(self, user_id: int | None) -> _RequestUserSettingsCache | None:
        if self.db is None or user_id is None:
            return None

        cache = _request_user_settings_cache.get()
        if cache is not None and cache.manager_id == id(self) and cache.user_id == user_id:
            if not cache.loaded:
                settings = get_user_settings(self.db, user_id)
                cache.disabled_plugins = frozenset(
                    normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING))
                )
                cache.disabled_skills = frozenset(
                    normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING))
                )
                cache.loaded = True
            return cache

        settings = get_user_settings(self.db, user_id)
        return _RequestUserSettingsCache(
            manager_id=id(self),
            user_id=user_id,
            loaded=True,
            disabled_plugins=frozenset(
                normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING))
            ),
            disabled_skills=frozenset(
                normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING))
            ),
        )

    def disabled_plugins_for_user(self, user_id: int | None) -> set[str]:
        """Возвращает множество имён плагинов, отключённых пользователем."""
        settings = self._request_user_settings(user_id)
        if settings is None:
            return set()
        return set(settings.disabled_plugins)

    def disabled_skills_for_user(self, user_id: int | None) -> set[str]:
        """Возвращает множество skills, отключённых пользователем."""
        settings = self._request_user_settings(user_id)
        if settings is None:
            return set()
        return set(settings.disabled_skills)

    def is_plugin_disabled_for_user(self, plugin_name: str | None, user_id: int | None) -> bool:
        """Отключён ли `plugin_name` для `user_id`. Пустое/None имя плагина → False."""
        return bool(plugin_name) and plugin_name in self.disabled_plugins_for_user(user_id)

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

    def get_functions_specs(self, helper, model_to_use, allowed_plugins=None, disclosed_functions=None):
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
                if self.strict_validation:
                    raise
                logger.error(f"Error instantiating plugin {plugin_name}: {str(e)}")
                continue

        if disclosed_functions is not None:
            disclosed = set(disclosed_functions or ())
            all_specs = [spec for spec in all_specs if spec.get("name") in disclosed]

        provider_specs = [self._strip_internal_spec_metadata(spec) for spec in all_specs]
        return self._format_specs_for_model(provider_specs, model_to_use)

    @staticmethod
    def _strip_internal_spec_metadata(spec):
        return {
            key: value
            for key, value in dict(spec or {}).items()
            if not str(key).startswith("x_")
        }

    def get_tool_metadata(self, function_name: str) -> Dict[str, Any]:
        spec = self.get_spec_by_function_name(function_name)
        metadata = spec.get(TOOL_METADATA_KEY) if isinstance(spec, dict) else None
        return dict(metadata) if isinstance(metadata, dict) else {}

    def get_tool_catalog(self, helper, model_to_use, allowed_plugins=None) -> List[Dict[str, Any]]:
        allowed_plugins = self.filter_allowed_plugins(allowed_plugins or ['All'])
        catalog: List[Dict[str, Any]] = []
        for plugin_name, plugin_class in self.plugins.items():
            if allowed_plugins != ['All'] and plugin_name not in allowed_plugins:
                continue
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue
                for spec in self._normalize_specs(plugin_instance.get_spec(), plugin_instance):
                    name = spec.get("name")
                    if not name:
                        continue
                    catalog.append({
                        "name": name,
                        "plugin": plugin_name,
                        "description": str(spec.get("description") or ""),
                        "metadata": dict(spec.get(TOOL_METADATA_KEY) or {}),
                    })
            except Exception:
                if self.strict_validation:
                    raise
                logger.exception("Error building tool catalog for plugin %s", plugin_name)
        return catalog

    def get_progressive_disclosure_bootstrap_functions(self, allowed_plugins=None) -> set[str]:
        allowed_plugins = self.filter_allowed_plugins(allowed_plugins or ['All'])
        names: set[str] = set()
        for plugin_name, plugin_class in self.plugins.items():
            if allowed_plugins != ['All'] and plugin_name not in allowed_plugins:
                continue
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue
                getter = getattr(plugin_instance, "get_progressive_disclosure_bootstrap_functions", None)
                if not callable(getter):
                    continue
                prefix = plugin_instance.get_function_prefix()
                for name in getter() or []:
                    text = str(name or "").strip()
                    if not text:
                        continue
                    names.add(text if "." in text else f"{prefix}.{text}")
            except Exception:
                if self.strict_validation:
                    raise
                logger.exception("Error resolving progressive bootstrap functions for %s", plugin_name)
        return names

    @staticmethod
    def _format_specs_for_model(specs, model_to_use):
        """
        Wrap function specs in the envelope expected by the target provider.
        Google models use {"function_declarations": [...]}, OpenAI-compatible
        models use the [{"type": "function", "function": {...}}] form.
        """
        if model_to_use in GOOGLE_MODELS:
            return {"function_declarations": specs}
        return [{"type": "function", "function": spec} for spec in specs]

    async def call_function(self, function_name, helper, arguments, request_context=None):
        """
        Call a function based on the name and parameters provided
        """
        started = time.monotonic()
        parsed_args: Dict[str, Any] = {}
        status = "error"
        error_msg = None
        direct_result = False
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            error_msg = f'Function {function_name} not found'
            self._record_tool_call_event(
                helper,
                function_name,
                parsed_args,
                status=status,
                duration_ms=self._elapsed_ms(started),
                error=error_msg,
                direct_result=direct_result,
                request_context=request_context,
            )
            return json.dumps({'error': error_msg})

        try:
            logger.debug(
                "Parsing arguments for function %s args_chars=%s",
                function_name,
                len(arguments or ""),
            )
            parsed_args = json.loads(arguments)

            spec = self.get_spec_by_function_name(function_name)
            if spec:
                params = spec.get("parameters") or {}
                properties = params.get("properties") or {}
                validation_args = {
                    key: value
                    for key, value in parsed_args.items()
                    if key not in FRAMEWORK_TOOL_ARGS or key in properties
                }
                errors = validate_function_args(spec, validation_args)
                if errors:
                    error_msg = f'Invalid args for {function_name}: {errors}'
                    return json.dumps({'error': error_msg}, ensure_ascii=False)

            if request_context is not None:
                parsed_args['request_context'] = request_context

            guard_result = await self._guard_tool_call(function_name, parsed_args, request_context=request_context)
            if guard_result:
                status = "error"
                error_msg = str(guard_result.get("error") or "Tool call blocked by plugin guard")
                return json.dumps(guard_result, ensure_ascii=False)

            logger.debug(
                "Calling function %s with argument_keys=%s",
                function_name,
                sorted(str(key) for key in parsed_args.keys()),
            )
            base_name = function_name.split(".", 1)[-1]
            result = await plugin.execute(base_name, helper, **parsed_args)
            tool_result = normalize_tool_result(
                result,
                tool_name=function_name,
                metadata=self.get_tool_metadata(function_name),
            )

            logger.debug(
                "Function %s completed result_type=%s result_chars=%s",
                function_name,
                type(result).__name__,
                len(tool_result.content),
            )
            status = "success" if tool_result.success else "error"
            direct_result = isinstance(tool_result.direct_result, dict)
            error_msg = tool_result.error
            return json.dumps(result, default=str, ensure_ascii=False)

        except json.JSONDecodeError as e:
            error_msg = (
                f"Ошибка разбора JSON аргументов функции {function_name}: {e}; "
                f"args_chars={len(arguments or '')}"
            )
            logger.error(error_msg)
            return json.dumps({'error': error_msg}, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Ошибка выполнения функции {function_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg}, ensure_ascii=False)
        finally:
            self._record_tool_call_event(
                helper,
                function_name,
                parsed_args,
                status=status,
                duration_ms=self._elapsed_ms(started),
                error=error_msg,
                direct_result=direct_result,
                request_context=request_context,
            )

    async def _guard_tool_call(self, function_name, parsed_args, *, request_context=None):
        for plugin_name in sorted(self.plugins.keys()):
            try:
                plugin = self.get_plugin(plugin_name)
            except Exception:
                logger.debug("Failed to instantiate plugin guard %s", plugin_name, exc_info=True)
                continue
            guard = getattr(plugin, "guard_tool_call", None)
            if not callable(guard):
                continue
            try:
                result = guard(
                    function_name=function_name,
                    arguments=parsed_args,
                    request_context=request_context,
                )
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                logger.exception("Tool-call guard failed in plugin %s", plugin_name)
                continue
            if result:
                return result
        return None

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    def _record_tool_call_event(
        self,
        helper,
        function_name: str,
        parsed_args: Dict[str, Any],
        *,
        status: str,
        duration_ms: int,
        error: str | None,
        direct_result: bool,
        request_context=None,
    ) -> None:
        db = getattr(helper, "db", None) if helper is not None else None
        record = getattr(db, "record_tool_call_event", None)
        if not callable(record):
            return

        chat_id = getattr(request_context, "chat_id", None) if request_context is not None else None
        user_id = getattr(request_context, "user_id", None) if request_context is not None else None
        request_id = getattr(request_context, "request_id", None) if request_context is not None else None
        if chat_id is None:
            chat_id = parsed_args.get("chat_id")
        if user_id is None:
            user_id = parsed_args.get("user_id")

        try:
            record(
                function_name=function_name,
                plugin_name=self.get_plugin_name_by_function_name(function_name),
                status=status,
                duration_ms=duration_ms,
                error=error,
                direct_result=direct_result,
                chat_id=chat_id,
                user_id=user_id,
                request_id=request_id,
            )
        except Exception:
            logger.debug("Failed to record tool-call telemetry for %s", function_name, exc_info=True)

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

    def get_plugin_name_by_function_name(self, function_name):
        for plugin_name in self.plugins.keys():
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue

                specs = self._normalize_specs(plugin_instance.get_spec(), plugin_instance)
                if any(spec.get('name') == function_name for spec in specs):
                    return plugin_name
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error resolving function %s against plugin %s: %s",
                    function_name,
                    plugin_name,
                    exc,
                    exc_info=True,
                )

        return None

    def is_function_allowed(self, function_name, allowed_plugins):
        if allowed_plugins == ['All']:
            return True
        if not allowed_plugins or allowed_plugins == ['None']:
            return False
        plugin_name = self.get_plugin_name_by_function_name(function_name)
        return plugin_name in allowed_plugins

    def get_subagent_function_specs(
        self,
        helper,
        model_to_use,
        parent_allowed_plugins=None,
        blocked_function_names=None,
        disclosed_functions=None,
    ):
        """
        Return tool specs available to a subagent: same allow-list as the parent,
        minus an explicit set of blocked function names. Always returns OpenAI-style
        specs; subagents do not currently support the Google function_declarations form.
        """
        blocked = set(blocked_function_names or ())
        specs = self.get_functions_specs(
            helper,
            model_to_use,
            parent_allowed_plugins or ['All'],
            disclosed_functions=disclosed_functions,
        )
        if isinstance(specs, dict):
            return [], set()
        filtered = []
        allowed_function_names = set()
        for tool in specs or []:
            function_spec = tool.get("function") or {}
            name = function_spec.get("name")
            if not name or name in blocked:
                continue
            filtered.append(tool)
            allowed_function_names.add(name)
        return filtered, allowed_function_names

    def is_subagent_function_allowed(
        self,
        function_name,
        parent_allowed_plugins=None,
        blocked_function_names=None,
    ):
        if blocked_function_names and function_name in blocked_function_names:
            return False
        return self.is_function_allowed(function_name, parent_allowed_plugins or ['All'])

    def __get_plugin_by_function_name(self, function_name):
        """
        Находит плагин по имени функции
        """
        plugin_name = self.get_plugin_name_by_function_name(function_name)
        if not plugin_name:
            return None
        return self.get_plugin(plugin_name)

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
                self._call_initialize(
                    instance,
                    openai=self.openai,
                    bot=self.bot,
                    storage_root=self.storage_root,
                    db=getattr(self, "db_handle", None),
                    plugin_config=self._plugin_config_segment(instance),
                )
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
                self._call_initialize(
                    instance,
                    openai=self.openai,
                    bot=self.bot,
                    storage_root=self.storage_root,
                    db=getattr(self, "db_handle", None),
                    plugin_config=self._plugin_config_segment(instance),
                )
            self.plugin_instances[plugin_name] = instance
            return instance
        return None

    def close_all(self):
        for instance in self.plugin_instances.values():
            try:
                instance.close()
            except Exception as exc:
                logger.warning(f"Error closing plugin {instance}: {exc}")

    async def close_all_async(self) -> None:
        """Await async cleanup on every loaded plugin.

        Mirrors close_all() but for plugins that need async teardown.
        Exceptions per plugin are logged and swallowed — every plugin gets
        its chance to clean up.
        """
        for instance in self.plugin_instances.values():
            try:
                await instance.close_async()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error in close_async for plugin %s: %s",
                    type(instance).__name__,
                    exc,
                )

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

    def get_prompt_handlers(self) -> List[Dict]:
        """Возвращает список pre-chat обработчиков обычного текста от всех плагинов."""
        handlers = []
        for plugin_name in self.plugins.keys():
            plugin_instance = self.get_plugin(plugin_name)
            if plugin_instance:
                plugin_handlers = plugin_instance.get_prompt_handlers()
                for handler in plugin_handlers:
                    if 'plugin_name' not in handler:
                        handler['plugin_name'] = plugin_name
                handlers.extend(plugin_handlers)
        return handlers

    def get_plugin_help_texts(self) -> List[Dict]:
        """Возвращает дополнительные help-тексты от всех плагинов."""
        help_texts = []
        for plugin_name in self.plugins.keys():
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue
                help_text = plugin_instance.get_help_text()
                if help_text:
                    help_texts.append({
                        "plugin_name": plugin_name,
                        "text": help_text,
                    })
            except Exception as e:
                logger.error(f"Ошибка при получении help-текста плагина {plugin_name}: {e}")
        return help_texts

    # ------------------------------------------------------------------ #
    # Hook framework (Stage 0)                                           #
    # ------------------------------------------------------------------ #

    def _active_plugin_instances(self, user_id: int | None) -> List[Plugin]:
        """Return live plugin instances respecting per-user disabled set.

        Order is **sorted by plugin_name** for deterministic sequential hooks.
        """
        disabled = self.disabled_plugins_for_user(user_id)
        result: List[Plugin] = []
        for plugin_name in sorted(self.plugins.keys()):
            if plugin_name in disabled:
                continue
            try:
                instance = self.get_plugin(plugin_name)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "plugin_hook_error",
                    extra={
                        "plugin_id": plugin_name,
                        "hook": "instance",
                        "event": "get_plugin",
                        "exc_class": type(exc).__name__,
                    },
                    exc_info=exc,
                )
                continue
            if instance is not None:
                result.append(instance)
        return result

    def _log_hook_error(
        self, plugin: Plugin, event_name: str, hook_kind: str, exc: BaseException
    ) -> None:
        plugin_id = getattr(plugin, "plugin_id", None) or type(plugin).__name__
        logger.error(
            "plugin_hook_error",
            extra={
                "plugin_id": plugin_id,
                "hook": hook_kind,
                "event": event_name,
                "exc_class": type(exc).__name__,
            },
            exc_info=exc,
        )

    async def dispatch_observe(
        self, event_name: str, payload: Any, *, user_id: int | None = None
    ) -> None:
        """Fan out an observer event to every subscribed plugin concurrently.

        Plugin exceptions are logged and swallowed; they never propagate.
        """
        plugins = self._active_plugin_instances(user_id)
        coros = []
        targets: List[Plugin] = []
        for plugin in plugins:
            method = getattr(plugin, event_name, None)
            if method is None:
                continue
            coros.append(method(payload))
            targets.append(plugin)
        if not coros:
            return
        results = await asyncio.gather(*coros, return_exceptions=True)
        for plugin, result in zip(targets, results):
            if isinstance(result, BaseException):
                self._log_hook_error(plugin, event_name, "observe", result)

    async def dispatch_blocking(
        self, event_name: str, payload: Any, *, user_id: int | None = None
    ) -> None:
        """Run a blocking event sequentially; one plugin's failure does NOT stop others.

        Policy A: every subscriber gets a chance to handle the event; the
        dispatcher returns normally only after all of them have tried.
        """
        plugins = self._active_plugin_instances(user_id)
        for plugin in plugins:
            method = getattr(plugin, event_name, None)
            if method is None:
                continue
            try:
                await method(payload)
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, event_name, "blocking", exc)

    async def collect_fragments(
        self, slot: str, payload: Any, *, user_id: int | None = None
    ) -> List[str]:
        """Collect non-empty string fragments from every subscribed plugin.

        Order matches the deterministic plugin order (sorted by plugin_name).
        Plugins that raise are skipped; plugins returning ``None`` or empty
        strings are skipped.
        """
        plugins = self._active_plugin_instances(user_id)
        fragments: List[str] = []
        for plugin in plugins:
            method = getattr(plugin, "contribute_prompt_fragment", None)
            if method is None:
                continue
            try:
                fragment = await method(slot, payload)
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, slot, "fragment", exc)
                continue
            if fragment:
                fragments.append(fragment)
        return fragments

    async def collect_objects(
        self, slot: str, payload: Any, *, user_id: int | None = None
    ) -> List[Any]:
        """Collect non-None object contributions from every subscribed plugin.

        Order matches the deterministic plugin order (sorted by plugin_name).
        Plugins that raise are skipped. Plugins returning ``None`` are skipped;
        other falsy values (empty list, ``0``) are kept — caller decides semantics.
        """
        plugins = self._active_plugin_instances(user_id)
        objects: List[Any] = []
        for plugin in plugins:
            method = getattr(plugin, "contribute_prompt_fragment", None)
            if method is None:
                continue
            try:
                obj = await method(slot, payload)
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, slot, "object", exc)
                continue
            if obj is not None:
                objects.append(obj)
        return objects

    async def apply_mutators(
        self,
        event_name: str,
        payload: Any,
        value: Any,
        *,
        user_id: int | None = None,
    ) -> Any:
        """Run a mutator chain (currently used for ``on_before_chat_request``).

        Each plugin receives the latest ``value`` (plus ``payload``) and may
        return a replacement. ``None`` means "no change". On exception the
        previous good value is retained and the chain continues.
        """
        plugins = self._active_plugin_instances(user_id)
        for plugin in plugins:
            method = getattr(plugin, event_name, None)
            if method is None:
                continue
            try:
                new_value = await method(value, payload)
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, event_name, "mutator", exc)
                continue
            if new_value is not None:
                value = new_value
        return value

    async def start_background_tasks(self, application: Any) -> None:
        """Start every plugin-declared background task as a supervised asyncio task.

        On exception inside the coroutine factory we log and sleep
        ``interval_seconds`` before retrying — the task is never permanently
        killed by a transient error.
        """
        for plugin_name in sorted(self.plugins.keys()):
            try:
                plugin = self.get_plugin(plugin_name)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error initializing plugin %s background tasks: %s",
                    plugin_name,
                    exc,
                    exc_info=True,
                )
                continue
            if plugin is None:
                continue
            try:
                tasks = plugin.get_background_tasks() or []
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, "get_background_tasks", "background_init", exc)
                continue
            for task in tasks:
                if not isinstance(task, BackgroundTask):
                    logger.warning(
                        "plugin %s returned non-BackgroundTask entry; skipping",
                        plugin_name,
                    )
                    continue
                key = f"{plugin_name}.{task.name}"
                if key in self._background_tasks:
                    logger.warning("background task %s already running; skipping", key)
                    continue
                self._background_tasks[key] = asyncio.create_task(
                    self._background_task_loop(plugin_name, task, application),
                    name=key,
                )

    async def _background_task_loop(
        self,
        plugin_name: str,
        task: BackgroundTask,
        application: Any,
    ) -> None:
        while True:
            try:
                await task.coroutine_factory(application=application)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "background_task_error",
                    extra={
                        "plugin_id": plugin_name,
                        "task": task.name,
                        "exc_class": type(exc).__name__,
                    },
                    exc_info=exc,
                )
            try:
                await asyncio.sleep(task.interval_seconds)
            except asyncio.CancelledError:
                raise

    async def stop_background_tasks(self, timeout: float = 10.0) -> None:
        """Cancel all background tasks and wait up to ``timeout`` seconds."""
        if not self._background_tasks:
            return
        tasks = list(self._background_tasks.values())
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait(tasks, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("error while stopping background tasks: %s", exc)
        self._background_tasks.clear()

    def register_plugin_schemas(self) -> None:
        """Execute DDL declared by every plugin via ``register_schema()``.

        Called from ``bot/__main__.py`` AFTER ``set_db`` and BEFORE
        ``set_openai`` so plugin-owned tables exist before ``initialize`` runs.

        Plugin instances are created here without invoking ``initialize`` —
        ``set_openai`` will run ``initialize`` exactly once per instance later.
        """
        if self.db is None:
            logger.debug("register_plugin_schemas called before set_db; skipping")
            return
        errors: list[tuple[str, BaseException]] = []
        for plugin_name in sorted(self.plugins.keys()):
            plugin = self._get_or_create_bare_instance(plugin_name)
            if plugin is None:
                continue
            try:
                statements = plugin.register_schema() or []
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, "register_schema", "schema", exc)
                errors.append((plugin_name, exc))
                continue
            if not statements:
                continue
            try:
                with self.db.get_connection() as conn:
                    for ddl in statements:
                        conn.execute(ddl)
            except Exception as exc:  # noqa: BLE001
                self._log_hook_error(plugin, "register_schema", "schema", exc)
                errors.append((plugin_name, exc))
        if errors:
            details = "; ".join(
                f"{plugin_name}: {type(exc).__name__}: {exc}"
                for plugin_name, exc in errors
            )
            raise RuntimeError(f"Plugin schema registration failed: {details}")

    def _get_or_create_bare_instance(self, plugin_name: str) -> Plugin | None:
        """Return cached instance or create a new one WITHOUT calling ``initialize``."""
        instance = self.plugin_instances.get(plugin_name)
        if instance is not None:
            return instance
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class is None:
            return None
        instance = plugin_class()
        if not getattr(instance, "plugin_id", None):
            instance.plugin_id = plugin_name
        if not getattr(instance, "function_prefix", None):
            instance.function_prefix = instance.plugin_id
        self.plugin_instances[plugin_name] = instance
        return instance
