import types

import pytest

from bot.llm_gateway_client import LLMGatewayClient, LLMGatewayError
from bot.model_constants import LLMGATEWAY_IMAGE_EDIT_MODEL


class FakeResponse:
    status_code = 200
    text = '{"data":[{"url":"https://example.com/edited.png"}]}'

    def json(self):
        return {"data": [{"url": "https://example.com/edited.png"}]}


class FakeAsyncClient:
    def __init__(self):
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append(types.SimpleNamespace(args=args, kwargs=kwargs))
        return FakeResponse()


class FakeErrorResponse:
    status_code = 502
    text = "gateway-error-" + ("x" * 700)

    def json(self):
        return {}


class FakeErrorAsyncClient:
    async def post(self, *args, **kwargs):
        return FakeErrorResponse()


@pytest.mark.asyncio
async def test_image_edit_file_uses_multipart_payload():
    http_client = FakeAsyncClient()
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = http_client

    result = await client.image_edit_file("add a hat", b"image-bytes")

    assert result["data"][0]["url"] == "https://example.com/edited.png"
    call = http_client.calls[0]
    assert call.args[0] == "https://gateway.example/v1/images/edits"
    assert call.kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "X-Title": "tgBot",
    }
    assert call.kwargs["data"] == {
        "model": LLMGATEWAY_IMAGE_EDIT_MODEL,
        "prompt": "add a hat",
    }
    assert call.kwargs["files"] == [("image", ("source.png", b"image-bytes", "image/png"))]


@pytest.mark.asyncio
async def test_post_json_error_includes_full_response_text():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = FakeErrorAsyncClient()

    with pytest.raises(LLMGatewayError) as exc_info:
        await client.post_json("/bad", {})

    assert FakeErrorResponse.text in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_multipart_error_includes_full_response_text():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = FakeErrorAsyncClient()

    with pytest.raises(LLMGatewayError) as exc_info:
        await client.post_multipart("/bad", data={}, files=[])

    assert FakeErrorResponse.text in str(exc_info.value)
