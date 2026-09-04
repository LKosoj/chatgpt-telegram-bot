import logging

import pytest

from bot.pricing import DEFAULT_MODEL_TOKEN_PRICES, load_model_token_prices, resolve_chat_cost
from tests.test_openai_helper_tool_calls import (
    DummyClient,
    DummyPluginManager,
    FakeResponse,
    FakeToolCall,
    _make_helper,
)


def test_resolve_chat_cost_known_model_with_split_prices_per_direction():
    table = {"modelA": (1.0, 2.0)}

    cost, price_source = resolve_chat_cost(
        model="modelA",
        total_tokens=300,
        prompt_tokens=100,
        completion_tokens=200,
        fallback_price_per_1k=0.002,
        table=table,
    )

    assert price_source == "model_split"
    assert cost == pytest.approx(100 * 1.0 / 1000 + 200 * 2.0 / 1000)


def test_resolve_chat_cost_known_model_without_split_uses_blended_price():
    table = {"modelA": (1.0, 3.0)}

    cost, price_source = resolve_chat_cost(
        model="modelA",
        total_tokens=1000,
        prompt_tokens=None,
        completion_tokens=None,
        fallback_price_per_1k=0.002,
        table=table,
    )

    assert price_source == "model_blended"
    assert cost == pytest.approx(1000 * ((1.0 + 3.0) / 2) / 1000)


def test_resolve_chat_cost_unknown_model_falls_back_to_legacy_price():
    cost, price_source = resolve_chat_cost(
        model="unknown/model",
        total_tokens=1000,
        prompt_tokens=100,
        completion_tokens=200,
        fallback_price_per_1k=0.5,
        table={},
    )

    assert price_source == "legacy_fallback"
    assert cost == pytest.approx(1000 * 0.5 / 1000)


def test_resolve_chat_cost_rounds_to_six_decimals():
    cost, _price_source = resolve_chat_cost(
        model=None,
        total_tokens=1,
        fallback_price_per_1k=0.0000001234,
        table={},
    )

    assert cost == round(1 * 0.0000001234 / 1000, 6)


def test_load_model_token_prices_parses_valid_pairs_and_skips_invalid(caplog):
    with caplog.at_level(logging.WARNING):
        table = load_model_token_prices(
            "modelA=0.5:1.0, modelB=0.2:0.4,bad-entry,modelC=notanumber:1,modelD=1:2:3,modelE=-1:2"
        )

    assert table["modelA"] == (0.5, 1.0)
    assert table["modelB"] == (0.2, 0.4)
    assert "modelC" not in table
    assert "bad-entry" not in table
    assert "modelD" not in table
    assert "modelE" not in table
    assert any("MODEL_TOKEN_PRICES" in record.message for record in caplog.records)


def test_load_model_token_prices_warns_on_duplicate_model_and_keeps_last(caplog):
    with caplog.at_level(logging.WARNING):
        table = load_model_token_prices("modelA=0.5:1.0,modelA=2.0:4.0")

    assert table["modelA"] == (2.0, 4.0)
    assert any("Duplicate MODEL_TOKEN_PRICES" in record.message for record in caplog.records)


def test_load_model_token_prices_empty_env_returns_defaults():
    assert load_model_token_prices("") == dict(DEFAULT_MODEL_TOKEN_PRICES)
    assert load_model_token_prices(None) == dict(DEFAULT_MODEL_TOKEN_PRICES)


@pytest.mark.asyncio
async def test_usage_split_reaches_record_chat_tokens_through_tool_call_round_trip(tmp_path):
    """Split accumulated over a tool-call round trip (chat_run.py's
    ChatRun.run_non_stream, via OpenAIHelper.get_chat_response) must reach
    the point where record_chat_tokens/resolve_chat_cost would price it,
    with price_source='model_split' once a per-model price is configured.
    """
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.list_skills": {"success": True, "skills": []}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall("skills.list_skills", "{}")], content=None),
        FakeResponse(content="final answer"),
    ])
    helper = _make_helper(pm, client=client)

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="use skills",
        user_id=1,
    )

    assert answer == "final answer"
    assert total_tokens == 6

    model = helper.get_last_chat_model(1)
    split = helper.get_last_chat_usage_split(1)
    # Two round trips, each FakeResponse default prompt=1/completion=2.
    assert split == (2, 4)

    cost, price_source = resolve_chat_cost(
        model=model,
        total_tokens=total_tokens,
        prompt_tokens=split[0],
        completion_tokens=split[1],
        fallback_price_per_1k=0.002,
        table={model: (1.0, 2.0)},
    )

    assert price_source == "model_split"
    assert cost == pytest.approx(2 * 1.0 / 1000 + 4 * 2.0 / 1000)


@pytest.mark.asyncio
async def test_usage_split_stays_unknown_when_gateway_omits_split():
    """Regression test for T06: шлюз прислал total_tokens=5, но не прислал
    prompt/completion_tokens. get_last_chat_usage_split должен вернуть
    None (а не (0, 0)), resolve_chat_cost -- посчитать по model_blended
    (а не по model_split с нулевой ценой), и в ответе пользователю не
    должно быть строки "None".
    """
    pm = DummyPluginManager({}, specs=[])
    client = DummyClient([
        FakeResponse(content="final answer", total_tokens=5, prompt_tokens=None, completion_tokens=None),
    ])
    helper = _make_helper(pm, client=client)
    helper.config["enable_functions"] = False
    helper.config["show_usage"] = True

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1, query="hi", user_id=1,
    )

    assert total_tokens == 5
    assert "None" not in answer
    assert helper.get_last_chat_usage_split(1) is None

    model = helper.get_last_chat_model(1)
    cost, price_source = resolve_chat_cost(
        model=model,
        total_tokens=total_tokens,
        prompt_tokens=None,
        completion_tokens=None,
        fallback_price_per_1k=0.002,
        table={model: (1.0, 2.0)},
    )

    assert price_source == "model_blended"
    assert cost == pytest.approx(5 * ((1.0 + 2.0) / 2) / 1000)


@pytest.mark.asyncio
async def test_interpret_image_text_response_omits_split_detail_when_unknown():
    """Regression test for T06 (vision-путь): то же самое, что предыдущий
    тест, но для OpenAIHelper._interpret_image_text_response.
    """
    pm = DummyPluginManager({}, specs=[])
    helper = _make_helper(pm, client=DummyClient([]))
    helper.config["show_usage"] = True
    response = FakeResponse(content="a photo", total_tokens=5, prompt_tokens=None, completion_tokens=None)

    answer, total_tokens = await helper._interpret_image_text_response(
        chat_id=1, response=response, token_accumulator=[],
    )

    assert total_tokens == 5
    assert "None" not in answer
    assert "💰 5" in answer
