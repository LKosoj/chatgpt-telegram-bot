from abc import abstractmethod, ABC
from typing import Any, Dict, Optional, List
from ..i18n import localized_text


class Plugin(ABC):
    """
    A plugin interface which can be used to create plugins for the ChatGPT API.
    """

    plugin_id: str | None = None
    function_prefix: str | None = None

    def get_plugin_id(self) -> str:
        """Return stable plugin id (defaults to class name if not set)."""
        return self.plugin_id or self.__class__.__name__

    def get_function_prefix(self) -> str:
        """Return function namespace prefix (defaults to plugin_id)."""
        return self.function_prefix or self.get_plugin_id()

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        """Optional lifecycle hook for plugin initialization."""
        self.openai = openai
        self.bot = bot
        self.storage_root = storage_root

    def close(self) -> None:
        """Optional lifecycle hook for plugin shutdown."""
        return None

    async def on_startup(self, application: Any) -> None:
        """Optional async hook called once after the Telegram application is ready."""
        return None

    # --- Hook framework (Stage 0): no-op defaults. Plugins override what they need. ---

    async def on_user_message(self, payload: Any) -> None:
        """Observer hook: fired after a user message is accepted by the bot."""
        return None

    async def on_assistant_response(self, payload: Any) -> None:
        """Observer hook: fired after the assistant produces a response."""
        return None

    async def on_session_reset(self, payload: Any) -> None:
        """Observer hook: fired when a chat session is reset."""
        return None

    async def on_session_before_delete(self, payload: Any) -> None:
        """Blocking hook: fired just before a session is deleted."""
        return None

    async def on_before_chat_request(
        self, messages: List[Dict], payload: Any
    ) -> List[Dict]:
        """Mutator hook: may return a modified ``messages`` list for the chat request.

        Default is identity (no modification). Returning ``None`` means "no change".
        """
        return messages

    async def contribute_prompt_fragment(
        self, slot: str, payload: Any
    ) -> str | None:
        """Collector hook: contribute a string fragment for a named prompt slot."""
        return None

    def get_background_tasks(self) -> list:
        """Return a list of :class:`BackgroundTask` instances to run periodically."""
        return []

    def register_schema(self) -> List[str]:
        """Return DDL statements to execute at startup for plugin-owned tables."""
        return []

    def get_config_prefix(self) -> str | None:
        """Return the prefix used to filter ``self.config`` keys for this plugin.

        ``None`` (default) means the plugin does not want a config slice.
        """
        return None

    def get_bot_language(self) -> str:
        if getattr(self, "openai", None) and getattr(self.openai, "config", None):
            return self.openai.config.get("bot_language", "en")
        return "en"

    def t(self, key: str, **kwargs: Any) -> str:
        text = localized_text(key, self.get_bot_language())
        if kwargs:
            return text.format(**kwargs)
        return text

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the name of the source of the plugin.
        """
        pass

    @abstractmethod
    def get_spec(self) -> [Dict]:
        """
        Function specs in the form of JSON schema as specified in the OpenAI documentation:
        https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
        """
        pass

    @abstractmethod
    async def execute(self, function_name: str, helper: Any, **kwargs: Optional[Dict[str, Any]]) -> Dict:
        """
        Execute the plugin and return a JSON serializable response.
        
        :param function_name: Name of the function to execute
        :param helper: Helper object to assist with function execution
        :param kwargs: Optional keyword arguments, can be partial
        :return: JSON serializable response
        """
        pass

    def get_commands(self) -> List[Dict]:
        """
        Возвращает список команд, которые поддерживает плагин.
        Каждая команда должна содержать:
        - command: str - название команды без /
        - description: str - описание команды
        - args: str (опционально) - описание аргументов команды
        - handler: callable - функция-обработчик команды
        - handler_kwargs: dict - аргументы для передачи в handler
        """
        return []
    
    def get_message_handlers(self) -> List[Dict]:
        """
        Возвращает список обработчиков сообщений.
        """
        return []

    def get_prompt_handlers(self) -> List[Dict]:
        """
        Возвращает список обработчиков обычных текстовых сообщений перед стандартным chat flow.
        """
        return []

    def get_help_text(self) -> str | None:
        """
        Возвращает дополнительный текст для /help.
        """
        return None
    
    def get_inline_handlers(self) -> List[Dict]:
        """
        Возвращает список обработчиков inline-запросов.
        """
        return []
