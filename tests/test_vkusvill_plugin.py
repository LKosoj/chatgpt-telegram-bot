from unittest.mock import AsyncMock

import pytest

from bot.plugins.vkusvill import VkusVillPlugin


def test_vkusvill_spec_exposes_domain_tools():
    plugin = VkusVillPlugin()

    specs = plugin.get_spec()
    names = {spec["name"] for spec in specs}

    assert {
        "shops",
        "products_search",
        "product_details",
        "product_analogs",
        "products_discount",
        "cart_link_create",
        "recipes",
    } <= names
    product_search = next(spec for spec in specs if spec["name"] == "products_search")
    assert product_search["parameters"]["required"] == ["q"]


@pytest.mark.asyncio
async def test_vkusvill_execute_maps_product_search_to_mcp_tool():
    plugin = VkusVillPlugin()
    plugin._call_remote_tool = AsyncMock(return_value={"ok": True})

    result = await plugin.execute(
        "products_search",
        helper=None,
        q="молоко",
        user_id=42,
        chat_id=100,
        request_context=object(),
    )

    assert result == {"ok": True}
    plugin._call_remote_tool.assert_awaited_once_with(
        "vkusvill_products_search",
        {
            "q": "молоко",
            "page": 1,
            "sort": "popularity",
            "vvonly": 1,
        },
    )


@pytest.mark.asyncio
async def test_vkusvill_execute_adds_recipe_defaults():
    plugin = VkusVillPlugin()
    plugin._call_remote_tool = AsyncMock(return_value={"ok": True})

    await plugin.execute("recipes", helper=None, q="сырники")

    plugin._call_remote_tool.assert_awaited_once_with(
        "vkusvill_recipes",
        {
            "q": "сырники",
            "sort": "popularity",
            "page": 1,
            "id_feature_filter": 0,
            "id_cooking_time_filter": 0,
            "id_cooking_method_filter": 0,
            "id_complexity_filter": 0,
            "id_category_filter": 0,
            "id_exclude_allergens_filter": [],
        },
    )


def test_vkusvill_extract_result_parses_mcp_text_json():
    plugin = VkusVillPlugin()
    result = plugin._extract_result({
        "content": [
            {
                "type": "text",
                "text": '{"ok": true, "data": {"items": [{"name": "Молоко"}]}}',
            }
        ]
    })

    assert result == {"ok": True, "data": {"items": [{"name": "Молоко"}]}}
