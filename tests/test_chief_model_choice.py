"""Which model each ``ChiefPlugin`` LLM call asks for.

``OpenAIHelper.ask()`` defaults to the main model (``bot/openai_helper.py:806``), so a call
that wants the cheap model has to say so explicitly. The two JSON-parsing calls do; the
recipe-enhancement call, whose text goes straight to the user, deliberately does not.
"""

import json

import pytest

from bot.plugins.chief import ChiefPlugin


class FakeHelper:
    """Records the kwargs of every ``ask()`` call and replies with canned JSON."""

    def __init__(self, reply):
        self.config = {"light_model": "cheap-model", "model": "main-model"}
        self.reply = reply
        self.calls = []

    async def ask(self, prompt, user_id, assistant_prompt=None, model=None, json_mode=False):
        self.calls.append({"model": model, "json_mode": json_mode})
        return self.reply, 42


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv("EDAMAM_APP_ID", "id")
    monkeypatch.setenv("EDAMAM_APP_KEY", "key")
    return ChiefPlugin()


async def test_parse_with_retry_uses_light_model(plugin):
    helper = FakeHelper(json.dumps({
        "ingredients": ["chicken"],
        "max_time": 30,
        "meal_type": "dinner",
        "dietary_restrictions": [],
    }))

    result, _tokens = await plugin._parse_with_retry("курица за полчаса", helper, user_id=1)

    assert result["ingredients"] == ["chicken"]
    assert helper.calls == [{"model": "cheap-model", "json_mode": True}]


async def test_parse_menu_preferences_uses_light_model(plugin):
    helper = FakeHelper(json.dumps({
        "dietary_preferences": ["vegan"],
        "excluded_ingredients": [],
        "meal_types": ["breakfast"],
    }))

    result, _tokens = await plugin._parse_menu_preferences("веган", helper, user_id=1, days=3)

    assert result["days"] == 3
    assert helper.calls == [{"model": "cheap-model", "json_mode": True}]


async def test_enhance_recipe_keeps_the_main_model(plugin):
    """Its output is prose shown to the user, so it must not be downgraded."""
    helper = FakeHelper("Совет: посолите в конце.")
    recipe = {"label": "Курица", "ingredientLines": ["курица", "соль"]}

    _text, _tokens = await plugin._enhance_recipe(recipe, "курица", helper, user_id=1)

    assert helper.calls == [{"model": None, "json_mode": False}]


async def test_empty_light_model_falls_back_inside_ask(plugin):
    """An unset LIGHT_MODEL is passed through as-is; ``ask()`` owns the fallback."""
    helper = FakeHelper(json.dumps({
        "ingredients": ["rice"],
        "max_time": 20,
        "meal_type": "lunch",
        "dietary_restrictions": [],
    }))
    helper.config["light_model"] = ""

    await plugin._parse_with_retry("рис", helper, user_id=1)

    assert helper.calls == [{"model": "", "json_mode": True}]
