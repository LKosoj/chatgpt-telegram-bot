import types

import pytest

from bot.model_constants import (
    GPT_ALL_MODELS,
    LLMGATEWAY_BIG_CONTEXT_MODEL,
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
)
from bot.plugins.web_research import WebResearchPlugin


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def web_research(self, query, **kwargs):
        self.calls.append(("research", query, kwargs))
        return {"output": "regular", "sources": ["s1"], "usage": {"total_tokens": 1}}

    async def web_deep_research(self, query, **kwargs):
        self.calls.append(("deep_research", query, kwargs))
        return {"output": "deep", "sources": ["s2"], "source_urls": ["u2"], "usage": {"total_tokens": 2}}


class FakeChatClient:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.decision)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class FakeHelper:
    def __init__(self, decision):
        self.config = {"light_model": LLMGATEWAY_LIGHT_MODEL}
        self.client = FakeChatClient(decision)
        self.gateway_client = FakeGateway()

    async def chat_completion(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)


def test_llmgateway_models_are_the_only_chat_models():
    assert GPT_ALL_MODELS == (
        LLMGATEWAY_HIGH_MODEL,
        LLMGATEWAY_LIGHT_MODEL,
        LLMGATEWAY_BIG_CONTEXT_MODEL,
    )


@pytest.mark.asyncio
async def test_web_research_uses_light_model_then_regular_research():
    helper = FakeHelper('{"research_type":"research","reason":"narrow"}')
    result = await WebResearchPlugin().execute("research_articles", helper, query="курс доллара сегодня")

    assert helper.client.calls[0]["model"] == LLMGATEWAY_LIGHT_MODEL
    assert helper.gateway_client.calls[0][0] == "research"
    assert result["result"]["output"] == "regular"


@pytest.mark.asyncio
async def test_web_research_uses_light_model_then_deep_research():
    helper = FakeHelper('{"research_type":"deep_research","reason":"broad"}')
    result = await WebResearchPlugin().execute("research_articles", helper, query="рынок AI агентов")

    assert helper.client.calls[0]["model"] == LLMGATEWAY_LIGHT_MODEL
    assert helper.gateway_client.calls[0][0] == "deep_research"
    assert result["result"]["output"] == "deep"


@pytest.mark.asyncio
async def test_reply_intent_classifier_uses_light_model():
    pytest.importorskip("tiktoken")
    from bot.openai_helper import OpenAIHelper

    helper = FakeHelper('{"intent":"image_edit"}')

    intent = await OpenAIHelper.classify_reply_intent(helper, "добавь шапочку", "image")

    assert intent == "image_edit"
    assert helper.client.calls[0]["model"] == LLMGATEWAY_LIGHT_MODEL
    assert helper.client.calls[0]["response_format"] == {"type": "json_object"}
    assert helper.client.calls[0]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_reply_intent_classifier_rejects_invalid_response():
    pytest.importorskip("tiktoken")
    from bot.openai_helper import OpenAIHelper

    helper = FakeHelper("not-json")

    intent = await OpenAIHelper.classify_reply_intent(helper, "что это значит?", "text")

    assert intent == "unknown"


@pytest.mark.asyncio
async def test_reply_intent_classifier_accepts_plain_label():
    pytest.importorskip("tiktoken")
    from bot.openai_helper import OpenAIHelper

    helper = FakeHelper("image_describe")

    intent = await OpenAIHelper.classify_reply_intent(helper, "что на этой картинке?", "image")

    assert intent == "image_describe"


@pytest.mark.asyncio
async def test_reply_intent_classifier_accepts_image_description_alias():
    pytest.importorskip("tiktoken")
    from bot.openai_helper import OpenAIHelper

    helper = FakeHelper('{"intent":"image_description"}')

    intent = await OpenAIHelper.classify_reply_intent(helper, "что на этой картинке?", "image")

    assert intent == "image_describe"
