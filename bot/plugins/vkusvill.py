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
                    "Search VkusVill (Russian grocery chain) stores. Returns address, coordinates, "
                    "contacts, opening hours, and features. Call with page=1 to get the list of "
                    "available filters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "description": "Page number.", "default": 1, "minimum": 1},
                        "id_region_filter": {"type": "integer", "description": "Region filter ID.", "default": 0, "minimum": 0},
                        "id_city_filter": {"type": "integer", "description": "City filter ID.", "default": 0, "minimum": 0},
                        "id_subway_filter": {"type": "integer", "description": "Metro station filter ID.", "default": 0, "minimum": 0},
                        "id_feature_filter": {"type": "integer", "description": "Store feature filter ID.", "default": 0, "minimum": 0},
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
                    "Search VkusVill products by text query. Returns id, xml_id, description, "
                    "price, rating, ingredients, nutrition (calories, protein, fat, carbs), and photos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query.", "minLength": 1, "maxLength": 255},
                        "page": {"type": "integer", "description": "Page number.", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Sort order.",
                            "default": "popularity",
                            "enum": ["price_asc", "price_desc", "rating", "popularity", "new"],
                        },
                        "vvonly": {
                            "type": "integer",
                            "description": "Restrict to VkusVill own-brand products (1) or include all (0).",
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
                    "Detailed info about a VkusVill product by id from products_search: "
                    "ingredients, nutrition (calories, protein, fat, carbs), photos, rating, and price."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "VkusVill product id (from products_search).", "minimum": 1},
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
                "description": "Return analog products for a VkusVill product by id from products_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "VkusVill product id (from products_search).", "minimum": 1},
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
                    "List VkusVill products on promotion: loyalty-card discount or quantity discount. "
                    "Returns id, xml_id, description, price, rating, ingredients, nutrition, and photos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Discount type: loyalty card or quantity-based.",
                            "default": "card",
                            "enum": ["card", "quantity"],
                        },
                        "page": {"type": "integer", "description": "Page number.", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Sort order.",
                            "default": "popularity",
                            "enum": ["price_asc", "price_desc", "rating", "popularity", "new", "name_asc", "name_desc"],
                        },
                        "vvonly": {
                            "type": "integer",
                            "description": "Restrict to VkusVill own-brand products (1) or include all (0).",
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
                    "Create a VkusVill shopping-cart link. Use xml_id from products_search or "
                    "product_details, and quantity q for each product."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "description": "Array of products to add to the cart.",
                            "minItems": 1,
                            "maxItems": 30,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "xml_id": {"type": "integer", "description": "Product xml_id.", "minimum": 1},
                                    "q": {"type": "number", "description": "Quantity of the product.", "minimum": 0.01, "maximum": 40},
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
                    "Search the VkusVill recipe catalog by query and filters. Returns ingredients, "
                    "nutrition, step-by-step instructions, and photos. Real recipes with ingredients "
                    "linkable to a shopping cart via cart_link_create. Use when the user wants to "
                    "cook from VkusVill products, build a shopping list, or asks for popular/seasonal "
                    "recipes from the catalog. Call with page=1 to get the list of available filters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query.", "default": "", "maxLength": 255},
                        "page": {"type": "integer", "description": "Page number.", "default": 1, "minimum": 1},
                        "sort": {
                            "type": "string",
                            "description": "Sort order.",
                            "default": "popularity",
                            "enum": ["popularity", "new"],
                        },
                        "id_feature_filter": {"type": "integer", "description": "Feature filter ID.", "default": 0, "minimum": 0},
                        "id_cooking_time_filter": {"type": "integer", "description": "Cooking time filter ID.", "default": 0, "minimum": 0},
                        "id_cooking_method_filter": {"type": "integer", "description": "Cooking method filter ID.", "default": 0, "minimum": 0},
                        "id_complexity_filter": {"type": "integer", "description": "Complexity filter ID.", "default": 0, "minimum": 0},
                        "id_category_filter": {"type": "integer", "description": "Category filter ID.", "default": 0, "minimum": 0},
                        "id_exclude_allergens_filter": {
                            "type": "array",
                            "description": "Allergen IDs to exclude.",
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
