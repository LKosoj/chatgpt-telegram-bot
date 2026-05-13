import httpx
import pytest

from bot.plugins.hindsight_memory import HindsightClient, HindsightError


@pytest.mark.asyncio
async def test_hindsight_error_includes_full_response_text():
    response_text = "hindsight-error-" + ("x" * 700)

    def handler(request):
        return httpx.Response(500, text=response_text)

    client = HindsightClient(
        "http://hindsight.local",
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(HindsightError) as exc_info:
            await client.request("GET", "/bad")
    finally:
        await client.close()

    assert response_text in str(exc_info.value)
