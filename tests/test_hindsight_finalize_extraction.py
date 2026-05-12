from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.plugins.hindsight_memory import HindsightMemoryPlugin, HINDSIGHT_EXTRACTOR_PROMPT


def _build_plugin(**config_overrides) -> HindsightMemoryPlugin:
    plugin = HindsightMemoryPlugin()
    # Bring up an "active" plugin: client present + enabled.
    plugin.initialize(
        openai=SimpleNamespace(
            config={'light_model': 'fake-light'},
            client=SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=AsyncMock())
                )
            ),
        ),
        plugin_config={
            'hindsight_base_url': 'http://x',
            'hindsight_api_token': 't',
            **config_overrides,
        },
    )
    return plugin


def test_session_transcript_skips_system_and_empty():
    plugin = _build_plugin()
    transcript = plugin._session_transcript_for_hindsight([
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "  spaces  "},
        {"role": "assistant", "content": "reply"},
    ])
    assert "ignored" not in transcript
    assert "user: hello" in transcript
    assert "user: spaces" in transcript
    assert "assistant: reply" in transcript


def test_session_transcript_serializes_list_content():
    plugin = _build_plugin()
    transcript = plugin._session_transcript_for_hindsight([
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ])
    assert "user: " in transcript
    assert '"type"' in transcript  # json-serialized


def test_parse_memory_items_extracts_json_and_respects_limit():
    plugin = _build_plugin(hindsight_max_auto_save_items=2)
    raw = '{"items":[{"content":"a"},{"content":"b"},{"content":"c"}]}'
    items = plugin._parse_hindsight_memory_items(raw)
    assert len(items) == 2
    assert items[0]["content"] == "a"


def test_parse_memory_items_drops_sensitive():
    plugin = _build_plugin()
    raw = '{"items":[{"content":"api key abc"},{"content":"clean fact"}]}'
    items = plugin._parse_hindsight_memory_items(raw)
    assert len(items) == 1
    assert items[0]["content"] == "clean fact"


def test_parse_memory_items_returns_empty_on_bad_json():
    plugin = _build_plugin()
    assert plugin._parse_hindsight_memory_items("not json") == []
    assert plugin._parse_hindsight_memory_items("") == []


def test_looks_sensitive_memory_flags_markers():
    plugin = _build_plugin()
    for s in ["My password is x", "Bearer abc", "Содержит пароль", "sk-12345", "AI-SERV-..."]:
        assert plugin._looks_sensitive_memory(s), f"should flag: {s}"
    assert not plugin._looks_sensitive_memory("clean fact about user")


async def test_extract_calls_openai_client_with_extractor_prompt():
    plugin = _build_plugin()
    plugin.openai.client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"items":[{"content":"keep me"}]}'))],
    ))
    items = await plugin._extract_hindsight_memory_items("user: hi\nassistant: hi")
    assert items == [{"content": "keep me", "context": ""}]
    call = plugin.openai.client.chat.completions.create.call_args
    assert call.kwargs["model"] == "fake-light"
    assert call.kwargs["messages"][0]["content"] == HINDSIGHT_EXTRACTOR_PROMPT


async def test_finalize_session_memory_happy_path():
    plugin = _build_plugin()
    plugin._extract_hindsight_memory_items = AsyncMock(
        return_value=[{"content": "fact a", "context": ""}]
    )
    plugin._retain_hindsight_items = AsyncMock()
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    saved = await plugin.finalize_session_memory(123, "sess-1", messages)
    assert saved == 1
    plugin._retain_hindsight_items.assert_awaited_once()


async def test_finalize_session_memory_returns_zero_when_inactive():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={})  # no creds → inactive
    messages = [{"role": "user", "content": "x"}]
    assert await plugin.finalize_session_memory(1, "s", messages) == 0


async def test_finalize_session_memory_returns_zero_for_empty_session_id():
    plugin = _build_plugin()
    assert await plugin.finalize_session_memory(1, "", [{"role": "user", "content": "x"}]) == 0
    assert await plugin.finalize_session_memory(1, None, [{"role": "user", "content": "x"}]) == 0


async def test_finalize_session_memory_swallows_exception_by_default():
    plugin = _build_plugin()
    plugin._extract_hindsight_memory_items = AsyncMock(side_effect=RuntimeError("boom"))
    saved = await plugin.finalize_session_memory(1, "s", [{"role": "user", "content": "x"}])
    assert saved == 0


async def test_finalize_session_memory_raises_when_raise_on_error_true():
    plugin = _build_plugin()
    plugin._extract_hindsight_memory_items = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await plugin.finalize_session_memory(
            1, "s", [{"role": "user", "content": "x"}], raise_on_error=True,
        )
