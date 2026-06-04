from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import MessageEntity

from bot.utils import edit_message_with_retry, render_markdown_message_entities


def test_render_markdown_message_entities_converts_markdown_to_entities():
    parts = render_markdown_message_entities("**hello** with [link](https://example.com)")

    assert len(parts) == 1
    text, entities = parts[0]
    assert text == "hello with link"
    assert all(isinstance(entity, MessageEntity) for entity in entities)
    assert {entity.type for entity in entities} >= {"bold", "text_link"}


@pytest.mark.asyncio
async def test_edit_message_with_retry_uses_entities_without_parse_mode():
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    context = SimpleNamespace(bot=bot)

    await edit_message_with_retry(
        context,
        chat_id=1,
        message_id="2",
        text="**hello** with [link](https://example.com)",
    )

    bot.edit_message_text.assert_awaited_once()
    kwargs = bot.edit_message_text.await_args.kwargs
    assert kwargs["text"] == "hello with link"
    assert kwargs["parse_mode"] is None
    assert all(isinstance(entity, MessageEntity) for entity in kwargs["entities"])
    assert {entity.type for entity in kwargs["entities"]} >= {"bold", "text_link"}
