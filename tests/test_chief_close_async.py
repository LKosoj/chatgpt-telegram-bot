"""Tests for ChiefPlugin.close_async() (T07).

``_ensure_session()`` lazily creates an ``aiohttp.ClientSession`` that was never
closed, leaving an "Unclosed client session" warning (and an open TCP socket)
at shutdown. ``close_async()`` closes it if present, and is a no-op otherwise.
"""

import pytest

from bot.plugins.chief import ChiefPlugin


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv("EDAMAM_APP_ID", "id")
    monkeypatch.setenv("EDAMAM_APP_KEY", "key")
    return ChiefPlugin()


async def test_close_async_closes_open_session(plugin):
    await plugin._ensure_session()
    assert plugin.session is not None
    assert plugin.session.closed is False

    await plugin.close_async()

    assert plugin.session.closed is True


async def test_close_async_noop_when_session_never_created(plugin):
    assert plugin.session is None

    # Should not raise.
    await plugin.close_async()
