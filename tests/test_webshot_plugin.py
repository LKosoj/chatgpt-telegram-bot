"""Tests for WebshotPlugin's timeout/to_thread fix (T07).

``execute`` used to call ``requests.get`` directly on the event loop (blocking it)
and the warm-up call had no timeout at all. Both calls must now go through
``asyncio.to_thread`` with an explicit ``timeout``. A separate regression covers
the ``os.remove`` in the ``except`` branch, which used to be able to raise
``FileNotFoundError`` and mask the original error.
"""

from types import SimpleNamespace

import pytest

import bot.plugins.webshot as webshot


class FakeRequests:
    """Records every ``get`` call and returns a canned successful response."""

    def __init__(self, response):
        self.calls = []
        self._response = response

    def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return self._response


async def fake_to_thread(func, *args, **kwargs):
    to_thread_calls.append(getattr(func, "__name__", repr(func)))
    return func(*args, **kwargs)


to_thread_calls = []


@pytest.fixture(autouse=True)
def _reset_to_thread_calls():
    to_thread_calls.clear()
    yield
    to_thread_calls.clear()


@pytest.mark.asyncio
async def test_execute_passes_timeout_and_uses_to_thread(monkeypatch, tmp_path):
    fake_response = SimpleNamespace(status_code=200, content=b"fake-png-bytes")
    fake_requests = FakeRequests(fake_response)

    monkeypatch.setattr(webshot, "requests", fake_requests)
    monkeypatch.setattr(webshot.asyncio, "to_thread", fake_to_thread)
    monkeypatch.chdir(tmp_path)

    plugin = webshot.WebshotPlugin()
    result = await plugin.execute("screenshot_website", helper=None, url="https://example.com")

    assert len(fake_requests.calls) == 2
    assert to_thread_calls == ["get", "get"]

    first_call, second_call = fake_requests.calls
    assert first_call["kwargs"]["timeout"] == 10
    assert second_call["kwargs"]["timeout"] == 30

    assert result["direct_result"]["kind"] == "photo"


@pytest.mark.asyncio
async def test_write_failure_does_not_raise(monkeypatch, tmp_path):
    """Regression: a failure writing the file used to leak FileNotFoundError from
    ``os.remove`` on the never-created path, masking the real error."""
    fake_response = SimpleNamespace(status_code=200, content=b"fake-png-bytes")
    fake_requests = FakeRequests(fake_response)

    monkeypatch.setattr(webshot, "requests", fake_requests)
    monkeypatch.setattr(webshot.asyncio, "to_thread", fake_to_thread)
    monkeypatch.chdir(tmp_path)

    def failing_open(*_args, **_kwargs):
        raise PermissionError("cannot write")

    monkeypatch.setattr(webshot, "open", failing_open, raising=False)

    plugin = webshot.WebshotPlugin()
    result = await plugin.execute("screenshot_website", helper=None, url="https://example.com")

    assert result == {"result": "Unable to screenshot website"}
