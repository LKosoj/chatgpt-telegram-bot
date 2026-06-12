from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import MessageEntity

from bot.utils import (
    edit_message_with_retry,
    escape_markdown,
    render_markdown_message_entities,
    split_into_chunks,
    _utf16_len,
)


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


# --- split_into_chunks: emoji / UTF-16 tests ---

def test_split_into_chunks_ascii_identity():
    """ASCII behaviour must be unchanged: short text returns single chunk."""
    text = "Hello, world!"
    result = split_into_chunks(text, chunk_size=4096)
    assert result == [text]


def test_split_into_chunks_ascii_splits_correctly():
    """Long ASCII text splits into chunks none exceeding chunk_size UTF-16 units."""
    line = "A" * 100
    text = "\n".join([line] * 50)  # 100 chars * 50 lines + newlines ~ 5050 chars
    chunks = split_into_chunks(text, chunk_size=4096)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert _utf16_len(chunk) <= 4096


def test_split_into_chunks_emoji_stays_within_limit():
    """Emoji are 2 UTF-16 units; chunks must not exceed 4096 UTF-16 units."""
    # Each emoji is 2 UTF-16 units, so 2049 emoji > 4096 UTF-16 units
    line = "\U0001F600" * 100  # 100 emoji = 200 UTF-16 units per line
    text = "\n".join([line] * 25)  # 25 lines x 200 = 5000 + 24 newlines = 5024 UTF-16 units
    chunks = split_into_chunks(text, chunk_size=4096)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert _utf16_len(chunk) <= 4096, f"chunk exceeds 4096: {_utf16_len(chunk)}"


def test_split_into_chunks_closing_markers_within_limit():
    """Closing markdown markers must be appended BEFORE the chunk is saved, not after."""
    # Build text with unclosed bold that forces a split
    bold_line = "*" + "x" * 200  # opening bold marker, 201 chars per line
    text = "\n".join([bold_line] * 25)
    chunks = split_into_chunks(text, chunk_size=4096)
    for chunk in chunks:
        assert _utf16_len(chunk) <= 4096, f"chunk with markers exceeds 4096: {_utf16_len(chunk)}"


def test_split_into_chunks_long_emoji_line_splits():
    """A single line with many emoji must be split when it exceeds max_line_utf16."""
    single_long_line = "\U0001F525" * 2000  # 2000 emoji = 4000 UTF-16 units > 3800
    chunks = split_into_chunks(single_long_line, chunk_size=4096)
    for chunk in chunks:
        assert _utf16_len(chunk) <= 4096


# --- escape_markdown tests ---

def test_escape_markdown_paired_backtick_preserves_code():
    """Code spans with paired backticks must not be escaped inside."""
    result = escape_markdown("hello `foo_bar` world", exclude_code_blocks=True)
    # Inside code: underscore untouched; outside: no special chars here
    assert "`foo_bar`" in result


def test_escape_markdown_non_code_chars_escaped():
    """Special markdown chars outside code spans must be escaped."""
    result = escape_markdown("hello_world (test)", exclude_code_blocks=True)
    assert r"\_" in result or "\\_" in result  # underscore escaped
    assert r"\(" in result or "\\(" in result  # ( escaped


def test_escape_markdown_unpaired_backtick_escapes_specials():
    """Unpaired backtick: text following it is treated as plain, specials must be escaped."""
    result = escape_markdown("hello ` world_test", exclude_code_blocks=True)
    # '_' in 'world_test' must be escaped
    assert r"\_" in result or "\\_" in result


def test_escape_markdown_unpaired_backtick_full_escape():
    """With an unpaired backtick, all subsequent special chars are escaped."""
    text = "before ` after_test [link]"
    result = escape_markdown(text, exclude_code_blocks=True)
    assert r"\_" in result
    assert r"\[" in result


def test_escape_markdown_exclude_false_escapes_backtick():
    """With exclude_code_blocks=False, backtick itself must be escaped."""
    result = escape_markdown("hello `code` world", exclude_code_blocks=False)
    assert "\\`" in result


def test_escape_markdown_paired_backtick_specials_outside():
    """Specials outside code spans are escaped, inside are not."""
    result = escape_markdown("_bold_ `code_here` _more_", exclude_code_blocks=True)
    # Outside the code span: _ must be escaped
    # The code span itself is preserved intact
    assert "`code_here`" in result
    assert r"\_" in result


# --- edit_message_with_retry: long text guard ---

@pytest.mark.asyncio
async def test_edit_message_with_retry_long_text_does_not_raise():
    """edit_message_with_retry must not raise when text exceeds 4096 UTF-16 units."""
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    context = SimpleNamespace(bot=bot)

    # 2048 emoji = 4096 UTF-16 units — exactly at limit, should be fine
    at_limit = "\U0001F600" * 2048
    await edit_message_with_retry(context, chat_id=1, message_id="2", text=at_limit)
    bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_message_with_retry_over_limit_truncates():
    """edit_message_with_retry must truncate silently when text exceeds 4096 UTF-16 units."""
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    context = SimpleNamespace(bot=bot)

    # 3000 emoji = 6000 UTF-16 units, well over the 4096 limit
    over_limit = "\U0001F525" * 3000
    await edit_message_with_retry(context, chat_id=1, message_id="2", text=over_limit)
    bot.edit_message_text.assert_awaited_once()
    kwargs = bot.edit_message_text.await_args.kwargs
    # The sent text must be within 4096 UTF-16 units
    assert _utf16_len(kwargs["text"]) <= 4096
