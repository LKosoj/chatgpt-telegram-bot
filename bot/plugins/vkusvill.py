from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict

import httpx

from .plugin import Plugin

logger = logging.getLogger(__name__)

VKUSVILL_MCP_URL = "https://mcp001.vkusvill.ru/mcp"
MCP_PROTOCOL_VERSION = "2025-03-26"
INTERNAL_ARGUMENTS = {"chat_id", "user_id", "request_context"}


class VkusVillPlugin(Plugin):
    """
    Adapter plugin for the VkusVill remote MCP server.
    """

    _tools: dict[str, dict[str, Any]] = {
        "shops": {
            "remote_name": "vkusvill_shops",
            "defaults": {
                "page": 1,
                "id_region_filter": 0,
                "id_city_filter": 0,
                "id_subway_filter": 0,
                "id_feature_filter": 0,
            },
            "spec": {
                "name": "shops",
                "description": (
                    "Поиск магазинов ВкусВилл. Возвращает адрес, координаты, контакты, "
                    "режим работы и особенности. Для списка доступных фильтров вызови page=1."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "description": "Номер страницы", "default": 1, "minimum": 1},
                        "id_region_filter": {"type": "integer", "description": "ID региона для фильтра", "default": 0, "minimum": 0},
                        "id_city_filter": {"type": "integer", "description": "ID города для фильтра", "default": 0, "minimum": 0},
                        "id_subway_filter": {"type": "integer", "description": "ID метро для фильтра", "default": 0, "minimum": 0},
                        "id_feature_filter": {"type": "integer", "description": "ID особенности для фильтра", "default": 0, "minimum": 0},
                    },
                },
            },
        },
        "products_search": {
            "remote_name": "vkusvill_products_search",
            "defaults": {"page": 1, "sort": "popularity", "vvonly": 1},
            "spec": {
                "name": "products_search",
                "description": (
                    "Поиск товаров ВкусВилл по текстовому запросу. Возвращает id, xml_id, "
                    "описание, цену, рейтинг, состав, КБЖУ и фото товаров."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Поисковый запрос", "minLength": 1, "maxLength": 255},
                        "page": {"type": "integer", "description": "Номер страницы", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Сортировка товаров",
                            "default": "popularity",
                            "enum": ["price_asc", "price_desc", "rating", "popularity", "new"],
                        },
                        "vvonly": {
                            "type": "integer",
                            "description": "Искать только товары бренда ВкусВилл",
                            "default": 1,
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["q"],
                },
            },
        },
        "product_details": {
            "remote_name": "vkusvill_product_details",
            "defaults": {},
            "spec": {
                "name": "product_details",
                "description": (
                    "Детальная информация о товаре ВкусВилл по id из products_search: "
                    "состав, КБЖУ, фото, рейтинг и цена."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID товара ВкусВилл", "minimum": 1},
                    },
                    "required": ["id"],
                },
            },
        },
        "product_analogs": {
            "remote_name": "vkusvill_product_analogs",
            "defaults": {},
            "spec": {
                "name": "product_analogs",
                "description": "Возвращает аналоги товара ВкусВилл по id из products_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID товара ВкусВилл", "minimum": 1},
                    },
                    "required": ["id"],
                },
            },
        },
        "products_discount": {
            "remote_name": "vkusvill_products_discount",
            "defaults": {"type": "card", "page": 1, "sort": "popularity", "vvonly": 1},
            "spec": {
                "name": "products_discount",
                "description": (
                    "Список акционных товаров ВкусВилл: скидка по карте или скидка за количество. "
                    "Возвращает id, xml_id, описание, цену, рейтинг, состав, КБЖУ и фото."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Тип скидки",
                            "default": "card",
                            "enum": ["card", "quantity"],
                        },
                        "page": {"type": "integer", "description": "Номер страницы", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Сортировка товаров",
                            "default": "popularity",
                            "enum": ["price_asc", "price_desc", "rating", "popularity", "new", "name_asc", "name_desc"],
                        },
                        "vvonly": {
                            "type": "integer",
                            "description": "Возвращать только товары бренда ВкусВилл",
                            "default": 1,
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
            },
        },
        "cart_link_create": {
            "remote_name": "vkusvill_cart_link_create",
            "defaults": {},
            "spec": {
                "name": "cart_link_create",
                "description": (
                    "Создает ссылку на корзину ВкусВилл. Используй xml_id из products_search "
                    "или product_details и количество q для каждого товара."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "description": "Массив товаров для добавления в корзину",
                            "minItems": 1,
                            "maxItems": 30,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "xml_id": {"type": "integer", "description": "xml_id товара", "minimum": 1},
                                    "q": {"type": "number", "description": "Количество товара", "minimum": 0.01, "maximum": 40},
                                },
                                "required": ["xml_id", "q"],
                            },
                        },
                    },
                    "required": ["products"],
                },
            },
        },
        "recipes": {
            "remote_name": "vkusvill_recipes",
            "defaults": {
                "sort": "popularity",
                "page": 1,
                "q": "",
                "id_feature_filter": 0,
                "id_cooking_time_filter": 0,
                "id_cooking_method_filter": 0,
                "id_complexity_filter": 0,
                "id_category_filter": 0,
                "id_exclude_allergens_filter": [],
            },
            "spec": {
                "name": "recipes",
                "description": (
                    "Поиск рецептов ВкусВилл по запросу и фильтрам. Возвращает ингредиенты, "
                    "пищевую ценность, пошаговое приготовление и фото. Для списка фильтров вызови page=1."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Поисковый запрос", "default": "", "maxLength": 255},
                        "page": {"type": "integer", "description": "Номер страницы", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Сортировка рецептов",
                            "default": "popularity",
                            "enum": ["popularity", "new"],
                        },
                        "id_feature_filter": {"type": "integer", "description": "ID особенности для фильтра", "default": 0, "minimum": 0},
                        "id_cooking_time_filter": {"type": "integer", "description": "ID времени готовки", "default": 0, "minimum": 0},
                        "id_cooking_method_filter": {"type": "integer", "description": "ID способа приготовления", "default": 0, "minimum": 0},
                        "id_complexity_filter": {"type": "integer", "description": "ID сложности", "default": 0, "minimum": 0},
                        "id_category_filter": {"type": "integer", "description": "ID категории", "default": 0, "minimum": 0},
                        "id_exclude_allergens_filter": {
                            "type": "array",
                            "description": "ID аллергенов, которые нужно исключить",
                            "default": [],
                            "items": {"type": "integer", "minimum": 1},
                        },
                    },
                },
            },
        },
    }

    def get_source_name(self) -> str:
        return "ВкусВилл"

    def get_spec(self) -> list[Dict]:
        return [copy.deepcopy(tool["spec"]) for tool in self._tools.values()]

    async def execute(self, function_name: str, helper: Any, **kwargs) -> Dict:
        tool = self._tools.get(function_name)
        if not tool:
            return {"error": f"Unknown VkusVill tool: {function_name}"}

        arguments = self._prepare_arguments(tool, kwargs)
        return await self._call_remote_tool(tool["remote_name"], arguments)

    def _prepare_arguments(self, tool: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        arguments = {
            key: value
            for key, value in kwargs.items()
            if key not in INTERNAL_ARGUMENTS and value is not None
        }
        for key, value in tool.get("defaults", {}).items():
            arguments.setdefault(key, copy.deepcopy(value))
        return arguments

    async def _call_remote_tool(self, remote_name: str, arguments: dict[str, Any]) -> Dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": remote_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(VKUSVILL_MCP_URL, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("VkusVill MCP request failed: %s", exc)
            return {"error": f"VkusVill MCP request failed: {str(exc)}"}

        try:
            data = response.json()
        except ValueError:
            return {"error": "VkusVill MCP returned a non-JSON response"}

        if data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            return {"error": f"VkusVill MCP error: {message}"}

        return self._extract_result(data.get("result"))

    def _extract_result(self, result: Any) -> Dict:
        if not isinstance(result, dict):
            return {"result": result}

        content = result.get("content")
        if not isinstance(content, list):
            return result

        values = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                values.append(self._parse_text_content(item.get("text", "")))
            else:
                values.append(item)

        if len(values) == 1:
            value = values[0]
            if isinstance(value, dict):
                return value
            return {"result": value}
        return {"result": values}

    def _parse_text_content(self, text: str) -> Any:
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return text
