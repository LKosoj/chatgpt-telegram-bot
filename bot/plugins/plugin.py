from abc import abstractmethod, ABC
from typing import Any, Dict, Optional, List


class Plugin(ABC):
    """
    A plugin interface which can be used to create plugins for the ChatGPT API.
    """

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
    
    def get_inline_handlers(self) -> List[Dict]:
        """
        Возвращает список обработчиков inline-запросов.
        """
        return []
