import pytest

import bot.plugins.text_summarizer as text_summarizer
from bot import net_safety
from bot.plugins.text_summarizer import TextSummarizerPlugin


@pytest.mark.asyncio
async def test_extract_text_from_url_uses_safe_get(monkeypatch):
    plugin = TextSummarizerPlugin()
    calls = []

    async def fake_safe_get(url, *, max_bytes, timeout):
        calls.append({"url": url, "max_bytes": max_bytes, "timeout": timeout})
        html = b"<html><body><div class='summary-scroll'>Summarized text</div></body></html>"
        return net_safety.SafeResponse(status_code=200, headers={}, content=html)

    monkeypatch.setattr(text_summarizer.net_safety, "safe_get", fake_safe_get)

    result = await plugin._extract_text_from_url("http://example.test/summary")

    assert calls == [{
        "url": "http://example.test/summary",
        "max_bytes": text_summarizer.MAX_DOWNLOAD_BYTES,
        "timeout": 15.0,
    }]
    assert result == "Summarized text"


@pytest.mark.asyncio
async def test_extract_text_from_url_returns_empty_string_on_unsafe_url(monkeypatch):
    plugin = TextSummarizerPlugin()

    async def fake_safe_get(url, *, max_bytes, timeout):
        raise net_safety.UnsafeURLError("refused")

    monkeypatch.setattr(text_summarizer.net_safety, "safe_get", fake_safe_get)

    result = await plugin._extract_text_from_url("http://169.254.169.254/x")

    assert result == ""
