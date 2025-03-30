from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
import httpx
import json
import asyncio
from urllib.parse import urljoin
import os
from pathlib import Path

from .plugin import Plugin

logger = logging.getLogger(__name__)


class MCPServerPlugin(Plugin):
    """
    Плагин для поддержки сторонних MCP (Machine Conversation Protocol) серверов.
    Позволяет взаимодействовать с различными API, реализующими протокол MCP.
    """

    def __init__(self):
        """Инициализация плагина"""
        self.openai = None
        self.bot = None
        self.servers = {}  # Словарь для хранения конфигураций серверов
        self.config_path = self._get_config_path()
        self.load_servers_config()
        self.admin_ids = self._get_admin_ids()
        self.allowed_users = self._get_allowed_users()

    def _get_config_path(self) -> Path:
        """Получает путь к файлу конфигурации MCP серверов"""
        # Используем директорию бота для хранения конфигурации
        bot_dir = Path(__file__).parent.parent
        config_dir = bot_dir / "config"
        os.makedirs(config_dir, exist_ok=True)
        return config_dir / "mcp_servers.json"

    def _get_admin_ids(self) -> List[int]:
        """Получает список ID администраторов из переменной окружения"""
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "")
        if not admin_ids_str or admin_ids_str == "-":
            return []
        
        try:
            return [int(user_id.strip()) for user_id in admin_ids_str.split(",") if user_id.strip()]
        except ValueError:
            logger.error("Некорректный формат ADMIN_USER_IDS, должны быть указаны числовые ID пользователей")
            return []

    def _get_allowed_users(self) -> List[int]:
        """Получает список пользователей, которым разрешено использовать MCP серверы"""
        allowed_users_str = os.getenv("MCP_SERVERS_ALLOWED_USERS", "*")
        if allowed_users_str == "*":
            # Разрешено всем пользователям
            return []
        
        try:
            # Парсим список идентификаторов
            return [int(user_id.strip()) for user_id in allowed_users_str.split(",") if user_id.strip()]
        except ValueError:
            logger.error("Некорректный формат MCP_SERVERS_ALLOWED_USERS, должны быть указаны числовые ID пользователей или *")
            return []

    def is_user_allowed(self, user_id: int) -> bool:
        """
        Проверяет, разрешено ли пользователю использовать MCP серверы
        
        :param user_id: ID пользователя
        :return: True если разрешено, False если нет
        """
        # Администраторам всегда разрешено
        if self.is_admin(user_id):
            return True
        
        # Если список разрешенных пользователей пуст - разрешено всем
        if not self.allowed_users:
            return True
        
        # Иначе проверяем наличие в списке разрешенных
        return user_id in self.allowed_users

    def load_servers_config(self):
        """Загружает конфигурацию серверов из JSON файла"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.servers = json.load(f)
                logger.info(f"Загружена конфигурация {len(self.servers)} MCP серверов")
            else:
                self.servers = {}
                logger.info("Файл конфигурации MCP серверов не найден, используем пустой список")
                
                # Загружаем серверы по умолчанию из переменной окружения
                default_servers = os.getenv("DEFAULT_MCP_SERVERS", "")
                if default_servers:
                    self._load_default_servers(default_servers)
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации MCP серверов: {str(e)}")
            self.servers = {}

    def _load_default_servers(self, default_servers_str: str):
        """Загружает серверы по умолчанию из переменной окружения"""
        try:
            for server_config in default_servers_str.split(","):
                parts = server_config.strip().split(":")
                if len(parts) >= 2:
                    server_name = parts[0].strip()
                    base_url = ":".join(parts[1:]).strip()  # Учитываем, что URL может содержать двоеточия
                    
                    # Инициализируем сервер без инструментов (они будут загружены при первом использовании)
                    self.servers[server_name] = {
                        "base_url": base_url,
                        "api_key": None,
                        "description": f"Сервер {server_name} по умолчанию",
                        "tools": []
                    }
                    logger.info(f"Добавлен сервер по умолчанию: {server_name} -> {base_url}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке серверов по умолчанию: {str(e)}")

    def save_servers_config(self):
        """Сохраняет конфигурацию серверов в JSON файл"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.servers, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранена конфигурация {len(self.servers)} MCP серверов")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации MCP серверов: {str(e)}")

    def is_admin(self, user_id: int) -> bool:
        """Проверяет, является ли пользователь администратором"""
        return user_id in self.admin_ids

    def get_source_name(self) -> str:
        """Возвращает имя источника плагина"""
        return "MCP Server"

    def get_spec(self) -> List[Dict]:
        """
        Возвращает спецификации функций, доступных через MCP серверы.

        Собирает спецификации от всех зарегистрированных MCP серверов.
        """
        specs = []
        
        # Добавляем функции управления серверами
        specs.append({
            "name": "register_mcp_server",
            "description": "Регистрирует новый MCP сервер для использования в качестве инструмента (только для администраторов)",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Уникальное имя сервера для ссылок"
                    },
                    "base_url": {
                        "type": "string", 
                        "description": "Базовый URL MCP сервера"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API ключ для доступа к серверу (если требуется)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Описание сервера и его функций"
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "ID пользователя, выполняющего действие"
                    }
                },
                "required": ["server_name", "base_url", "user_id"]
            }
        })
        
        specs.append({
            "name": "list_mcp_servers",
            "description": "Получить список всех зарегистрированных MCP серверов",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "ID пользователя, выполняющего действие"
                    }
                },
                "required": ["user_id"]
            }
        })
        
        specs.append({
            "name": "remove_mcp_server",
            "description": "Удалить зарегистрированный MCP сервер (только для администраторов)",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Имя сервера для удаления"
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "ID пользователя, выполняющего действие"
                    }
                },
                "required": ["server_name", "user_id"]
            }
        })
        
        # Динамически добавляем функции от зарегистрированных серверов
        for server_name, server_config in self.servers.items():
            if "tools" in server_config and server_config["tools"]:
                for tool in server_config["tools"]:
                    # Модифицируем спецификацию, добавляя серверу префикс
                    tool_spec = tool.copy()
                    tool_spec["name"] = f"{server_name}_{tool['name']}"
                    tool_spec["description"] = f"[{server_name}] {tool['description']}"
                    specs.append(tool_spec)
            else:
                # Если инструменты не загружены, пробуем загрузить их асинхронно
                asyncio.create_task(self._refresh_server_tools(server_name))
        
        return specs

    async def _refresh_server_tools(self, server_name: str):
        """
        Асинхронно обновляет список инструментов для сервера
        
        :param server_name: Имя сервера для обновления
        """
        if server_name not in self.servers:
            return
        
        server_config = self.servers[server_name]
        try:
            # Получаем инструменты с сервера
            tools_data = await self._fetch_server_tools(
                server_config["base_url"], 
                server_config.get("api_key")
            )
            
            if tools_data:
                server_config["tools"] = tools_data
                self.save_servers_config()
                logger.info(f"Обновлены инструменты для сервера {server_name}: {len(tools_data)} инструментов")
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении инструментов сервера {server_name}: {str(e)}")

    async def _fetch_server_tools(self, base_url: str, api_key: Optional[str] = None) -> List[Dict]:
        """
        Получает список инструментов от MCP сервера
        
        :param base_url: URL MCP сервера
        :param api_key: API ключ для авторизации (опционально)
        :return: Список инструментов
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    urljoin(base_url, "/tools"),
                    headers=headers
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Ошибка при получении инструментов с {base_url}: {str(e)}")
            return []

    async def execute(self, function_name: str, helper: Any, **kwargs) -> Dict:
        """
        Выполняет функции для MCP серверов.
        
        :param function_name: Имя функции для выполнения
        :param helper: Вспомогательный объект OpenAIHelper
        :param kwargs: Аргументы для функции
        :return: JSON-совместимый ответ
        """
        try:
            # Получаем ID пользователя (может отсутствовать)
            user_id = kwargs.get("user_id")
            
            # Если user_id отсутствует и это вызов функции управления MCP сервером,
            # возвращаем ошибку авторизации
            if user_id is None:
                if function_name in ["register_mcp_server", "remove_mcp_server"]:
                    return {"error": "Необходимо указать ID пользователя для управления MCP серверами"}
                
                # Если это не функция управления, но нужно проверить доступ, 
                # получаем user_id из helper, если он доступен
                if helper and hasattr(helper, 'user_id'):
                    user_id = helper.user_id
            
            # Функции управления серверами
            if function_name == "register_mcp_server":
                if not user_id or not self.is_admin(user_id):
                    return {"error": "Управление MCP серверами доступно только администраторам"}
                return await self.register_server(**kwargs)
                
            elif function_name == "list_mcp_servers":
                return await self.list_servers()
                
            elif function_name == "remove_mcp_server":
                if not user_id or not self.is_admin(user_id):
                    return {"error": "Управление MCP серверами доступно только администраторам"}
                return await self.remove_server(**kwargs)
            
            # Проверка прав доступа для использования инструментов MCP серверов
            if user_id and not self.is_user_allowed(user_id):
                return {"error": "У вас нет доступа к использованию MCP серверов"}
            
            # Проверяем, является ли это вызовом функции MCP сервера
            for server_name, server_config in self.servers.items():
                prefix = f"{server_name}_"
                if function_name.startswith(prefix):
                    # Извлекаем исходное имя функции
                    original_function_name = function_name[len(prefix):]
                    
                    # Создаем копию kwargs без внутренних параметров плагина
                    filtered_kwargs = kwargs.copy()
                    # Удаляем параметры, которые не должны передаваться внешнему серверу
                    filtered_kwargs.pop('user_id', None)
                    
                    # Вызываем функцию на сервере с отфильтрованными параметрами
                    return await self.call_mcp_function(server_name, original_function_name, **filtered_kwargs)
            
            return {"error": f"Функция {function_name} не найдена"}
        
        except Exception as e:
            logger.error(f"Ошибка при выполнении функции {function_name}: {str(e)}")
            return {"error": str(e)}

    async def register_server(self, server_name: str, base_url: str, user_id: int, api_key: str = None, description: str = "") -> Dict:
        """
        Регистрирует новый MCP сервер и получает его функции.
        
        :param server_name: Уникальное имя сервера
        :param base_url: Базовый URL сервера
        :param user_id: ID пользователя, выполняющего действие
        :param api_key: API ключ (опционально)
        :param description: Описание сервера
        :return: Результат регистрации
        """
        try:
            # Проверяем права доступа
            if not self.is_admin(user_id):
                return {"error": "Регистрация MCP серверов доступна только администраторам"}
            
            # Проверяем, что сервер еще не зарегистрирован
            if server_name in self.servers:
                return {"error": f"Сервер с именем {server_name} уже зарегистрирован"}
            
            # Получаем список инструментов от MCP сервера
            tools_data = await self._fetch_server_tools(base_url, api_key)
            
            if not tools_data:
                return {"error": f"Не удалось получить инструменты с сервера {base_url}"}
            
            # Сохраняем конфигурацию сервера
            self.servers[server_name] = {
                "base_url": base_url,
                "api_key": api_key,
                "description": description,
                "tools": tools_data
            }
            
            # Сохраняем конфигурацию в файл
            self.save_servers_config()
            
            tool_names = [tool["name"] for tool in tools_data]
            
            return {
                "success": True,
                "message": f"Сервер {server_name} успешно зарегистрирован",
                "tools_count": len(tools_data),
                "tools": tool_names
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка при регистрации сервера {server_name}: {e}")
            return {"error": f"Ошибка HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            logger.error(f"Ошибка при регистрации сервера {server_name}: {str(e)}")
            return {"error": str(e)}

    async def list_servers(self) -> Dict:
        """
        Возвращает список всех зарегистрированных MCP серверов.
        
        :return: Список серверов с их описаниями
        """
        result = {
            "servers": []
        }
        
        for server_name, config in self.servers.items():
            server_info = {
                "name": server_name,
                "base_url": config["base_url"],
                "description": config.get("description", ""),
                "tools_count": len(config.get("tools", [])),
                "tools": [tool["name"] for tool in config.get("tools", [])]
            }
            result["servers"].append(server_info)
        
        return result

    async def remove_server(self, server_name: str, user_id: int) -> Dict:
        """
        Удаляет зарегистрированный MCP сервер.
        
        :param server_name: Имя сервера для удаления
        :param user_id: ID пользователя, выполняющего действие
        :return: Результат удаления
        """
        # Проверяем права доступа
        if not self.is_admin(user_id):
            return {"error": "Удаление MCP серверов доступно только администраторам"}
            
        if server_name not in self.servers:
            return {"error": f"Сервер {server_name} не найден"}
        
        del self.servers[server_name]
        
        # Сохраняем обновленную конфигурацию
        self.save_servers_config()
        
        return {
            "success": True,
            "message": f"Сервер {server_name} успешно удален"
        }

    async def call_mcp_function(self, server_name: str, function_name: str, **kwargs) -> Dict:
        """
        Вызывает функцию на удаленном MCP сервере.
        
        :param server_name: Имя сервера
        :param function_name: Имя функции для вызова
        :param kwargs: Аргументы для функции
        :return: Результат выполнения функции
        """
        if server_name not in self.servers:
            return {"error": f"Сервер {server_name} не найден"}
        
        server_config = self.servers[server_name]
        base_url = server_config["base_url"]
        api_key = server_config.get("api_key")
        
        # Подготовка заголовков
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Подготовка данных запроса
        request_data = {
            "name": function_name,
            "arguments": kwargs
        }
        
        # Устанавливаем таймаут из конфигурации
        timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))
        
        try:
            # Выполнение запроса к MCP серверу
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    urljoin(base_url, "/execute"),
                    headers=headers,
                    json=request_data
                )
                
                response.raise_for_status()
                result = response.json()
                
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка при вызове функции {function_name} на сервере {server_name}: {e}")
            return {"error": f"Ошибка HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            logger.error(f"Ошибка при вызове функции {function_name} на сервере {server_name}: {str(e)}")
            return {"error": str(e)}

    def get_commands(self) -> List[Dict]:
        """
        Возвращает список команд, поддерживаемых плагином.
        """
        return [
            {
                "command": "mcp_servers",
                "description": "Управление MCP серверами",
                "handler": self.handle_mcp_servers_command
            }
        ]
    
    async def handle_mcp_servers_command(self, update, context):
        """
        Обработчик команды для управления MCP серверами.
        """
        user_id = update.effective_user.id
        
        # Проверяем права доступа для команды управления
        is_admin = self.is_admin(user_id)
        is_allowed = self.is_user_allowed(user_id)
        
        if not is_allowed:
            return {"text": "У вас нет доступа к использованию MCP серверов.", "parse_mode": "Markdown"}
        
        servers_info = await self.list_servers()
        servers = servers_info.get("servers", [])
        
        if not servers:
            if is_admin:
                return {"text": "Нет зарегистрированных MCP серверов.\n\nДля добавления нового сервера отправьте сообщение: `Зарегистрируй MCP сервер с именем example, URL http://example.com`", "parse_mode": "Markdown"}
            else:
                return {"text": "Нет зарегистрированных MCP серверов.", "parse_mode": "Markdown"}
        
        result = "Зарегистрированные MCP серверы:\n\n"
        for server in servers:
            result += f"• **{server['name']}**\n"
            result += f"  URL: `{server['base_url']}`\n"
            if server.get("description"):
                result += f"  Описание: {server['description']}\n"
            result += f"  Инструментов: {server['tools_count']}\n\n"
        
        if is_admin:
            result += "\n**Управление серверами** (только для администраторов):\n"
            result += "• Для добавления: `Зарегистрируй MCP сервер с именем example, URL http://example.com`\n"
            result += "• Для удаления: `Удали MCP сервер example`"
        
        return {"text": result, "parse_mode": "Markdown"}