import types

import pytest

from bot.llm_gateway_client import LLMGatewayClient, LLMGatewayError
from bot.plugins.stable_diffusion import StableDiffusionPlugin


class FakeResponse:
    status_code = 200
    text = '{"data":[{"url":"https://example.com/edited.png"}]}'

    def json(self):
        return {"data": [{"url": "https://example.com/edited.png"}]}


class FakeVoiceResponse:
    status_code = 200
    text = '[{"voice":"kseniya","gender":"female","language":"ru"}]'

    def json(self):
        return [{"voice": "kseniya", "gender": "female", "language": "ru"}]


class FakeAsyncClient:
    def __init__(self):
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append(types.SimpleNamespace(args=args, kwargs=kwargs))
        return FakeResponse()

    async def get(self, *args, **kwargs):
        self.calls.append(types.SimpleNamespace(args=args, kwargs=kwargs))
        return FakeVoiceResponse()


class FakeErrorResponse:
    status_code = 502
    text = "gateway-error-" + ("x" * 700)

    def json(self):
        return {}


class FakeErrorAsyncClient:
    async def post(self, *args, **kwargs):
        return FakeErrorResponse()

    async def get(self, *args, **kwargs):
        return FakeErrorResponse()


@pytest.mark.asyncio
async def test_image_edit_file_uses_multipart_payload():
    http_client = FakeAsyncClient()
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = http_client

    result = await client.image_edit_file(
        "add a hat",
        b"image-bytes",
        model="llmgateway/ai-klein-generation",
    )

    assert result["data"][0]["url"] == "https://example.com/edited.png"
    call = http_client.calls[0]
    assert call.args[0] == "https://gateway.example/v1/images/edits"
    assert call.kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "X-Title": "tgBot",
    }
    assert call.kwargs["data"] == {
        "model": "llmgateway/ai-klein-generation",
        "prompt": "add a hat",
    }
    assert call.kwargs["files"] == [("image", ("source.png", b"image-bytes", "image/png"))]


def test_image_edit_requires_model():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")

    with pytest.raises(TypeError):
        client.image_edit("add a hat", ["https://example.com/source.png"])


def test_image_edit_file_requires_model():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")

    with pytest.raises(TypeError):
        client.image_edit_file("add a hat", b"image-bytes")


@pytest.mark.asyncio
async def test_image_edit_file_accepts_configured_model():
    http_client = FakeAsyncClient()
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = http_client

    await client.image_edit_file(
        "add a hat",
        b"image-bytes",
        model="llmgateway/ai-klein-generation",
    )

    call = http_client.calls[0]
    assert call.kwargs["data"]["model"] == "llmgateway/ai-klein-generation"


@pytest.mark.asyncio
async def test_image_edit_accepts_configured_model():
    http_client = FakeAsyncClient()
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = http_client

    await client.image_edit(
        "add a hat",
        ["https://example.com/source.png"],
        model="llmgateway/ai-klein-generation",
    )

    call = http_client.calls[0]
    assert call.kwargs["json"]["model"] == "llmgateway/ai-klein-generation"


@pytest.mark.asyncio
async def test_stable_diffusion_edit_uses_configured_image_model():
    class FakeGateway:
        def __init__(self):
            self.calls = []

        async def image_edit(self, prompt, images, **kwargs):
            self.calls.append((prompt, images, kwargs))
            return {"data": [{"url": "https://example.com/edited.png"}]}

    helper = types.SimpleNamespace(
        config={"image_model": "llmgateway/ai-klein-generation"},
        gateway_client=FakeGateway(),
    )
    plugin = StableDiffusionPlugin()

    result = await plugin._edit_image(helper, "add a hat", "https://example.com/source.png")

    assert result == ("https://example.com/edited.png", "url")
    [(prompt, images, kwargs)] = helper.gateway_client.calls
    assert prompt == "add a hat"
    assert images == ["https://example.com/source.png"]
    assert kwargs["model"] == "llmgateway/ai-klein-generation"


@pytest.mark.asyncio
async def test_post_json_error_includes_full_response_text():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = FakeErrorAsyncClient()

    with pytest.raises(LLMGatewayError) as exc_info:
        await client.post_json("/bad", {})

    assert FakeErrorResponse.text in str(exc_info.value)


@pytest.mark.asyncio
async def test_audio_voices_uses_get_with_model_query():
    http_client = FakeAsyncClient()
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = http_client

    result = await client.audio_voices("llmgateway/silero-tts")

    assert result[0]["voice"] == "kseniya"
    call = http_client.calls[0]
    assert call.args[0] == "https://gateway.example/v1/audio/voices"
    assert call.kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "X-Title": "tgBot",
    }
    assert call.kwargs["params"] == {"model": "llmgateway/silero-tts"}
    assert call.kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_post_multipart_error_includes_full_response_text():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = FakeErrorAsyncClient()

    with pytest.raises(LLMGatewayError) as exc_info:
        await client.post_multipart("/bad", data={}, files=[])

    assert FakeErrorResponse.text in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_json_error_includes_full_response_text():
    client = LLMGatewayClient("https://gateway.example/v1", "test-key")
    client._client = FakeErrorAsyncClient()

    with pytest.raises(LLMGatewayError) as exc_info:
        await client.get_json("/bad")

    assert FakeErrorResponse.text in str(exc_info.value)
