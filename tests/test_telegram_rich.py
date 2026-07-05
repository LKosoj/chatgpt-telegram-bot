from types import MappingProxyType

import pytest

from bot.telegram_rich import (
    RICH_MESSAGES_AUTO,
    RICH_MESSAGES_OFF,
    RICH_MESSAGES_REQUIRED,
    build_markdown_rich_message,
    build_send_rich_message_draft_payload,
    build_send_rich_message_payload,
    rich_messages_enabled,
    rich_messages_mode,
    rich_messages_required,
    send_rich_markdown,
    send_rich_markdown_draft,
)


class FakeMarkup:
    def to_dict(self):
        return {"inline_keyboard": [[{"text": "Open", "callback_data": "open"}]]}


class FakeBot:
    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else {"ok": True}

    async def _post(self, endpoint, data=None, **kwargs):
        self.calls.append((endpoint, data, kwargs))
        return self.result


def test_build_markdown_rich_message_uses_markdown_only():
    assert build_markdown_rich_message(
        "**hello**",
        is_rtl=True,
        skip_entity_detection=False,
    ) == {
        "markdown": "**hello**",
        "is_rtl": True,
        "skip_entity_detection": False,
    }


def test_build_send_rich_message_payload_translates_reply_and_markup():
    payload = build_send_rich_message_payload(
        chat_id=123,
        markdown="**hello**",
        message_thread_id=456,
        reply_to_message_id=789,
        reply_markup=FakeMarkup(),
    )

    assert payload == {
        "chat_id": 123,
        "message_thread_id": 456,
        "rich_message": {"markdown": "**hello**"},
        "reply_parameters": {"message_id": 789},
        "reply_markup": {"inline_keyboard": [[{"text": "Open", "callback_data": "open"}]]},
    }
    assert "text" not in payload
    assert "parse_mode" not in payload


def test_build_send_rich_message_payload_omits_none_values():
    assert build_send_rich_message_payload(chat_id=123, markdown="hello") == {
        "chat_id": 123,
        "rich_message": {"markdown": "hello"},
    }


def test_build_send_rich_message_draft_payload_requires_non_zero_draft_id():
    with pytest.raises(ValueError, match="draft_id"):
        build_send_rich_message_draft_payload(
            chat_id=123,
            draft_id=0,
            markdown="draft",
        )


def test_build_send_rich_message_draft_payload_uses_markdown_only():
    payload = build_send_rich_message_draft_payload(
        chat_id=123,
        draft_id=55,
        markdown="_draft_",
        message_thread_id=777,
    )

    assert payload == {
        "chat_id": 123,
        "message_thread_id": 777,
        "draft_id": 55,
        "rich_message": {"markdown": "_draft_"},
    }
    assert "text" not in payload
    assert "parse_mode" not in payload


@pytest.mark.asyncio
async def test_send_rich_markdown_posts_rich_message_payload():
    bot = FakeBot()

    result = await send_rich_markdown(
        bot,
        chat_id=123,
        markdown="**hello**",
        reply_to_message_id=456,
        api_kwargs={"allow_paid_broadcast": False},
    )

    assert result == {"ok": True}
    assert bot.calls == [
        (
            "sendRichMessage",
            {
                "chat_id": 123,
                "rich_message": {"markdown": "**hello**"},
                "reply_parameters": {"message_id": 456},
            },
            {"api_kwargs": {"allow_paid_broadcast": False}},
        )
    ]


@pytest.mark.asyncio
async def test_send_rich_markdown_converts_full_message_result():
    bot = FakeBot(result={
        "message_id": 99,
        "date": 0,
        "chat": {"id": 123, "type": "private"},
        "rich_message": {"blocks": []},
    })

    message = await send_rich_markdown(bot, chat_id=123, markdown="hello")

    assert message.message_id == 99
    assert message.chat_id == 123
    assert message.api_kwargs == MappingProxyType({"rich_message": {"blocks": []}})


@pytest.mark.asyncio
async def test_send_rich_markdown_draft_posts_rich_draft_payload():
    bot = FakeBot()

    result = await send_rich_markdown_draft(
        bot,
        chat_id=123,
        draft_id=55,
        markdown="_draft_",
    )

    assert result == {"ok": True}
    assert bot.calls == [
        (
            "sendRichMessageDraft",
            {
                "chat_id": 123,
                "draft_id": 55,
                "rich_message": {"markdown": "_draft_"},
            },
            {"api_kwargs": None},
        )
    ]


@pytest.mark.parametrize(
    ("config", "mode", "enabled", "required"),
    [
        ({}, RICH_MESSAGES_OFF, False, False),
        (None, RICH_MESSAGES_OFF, False, False),
        ({"telegram_rich_messages": "auto"}, RICH_MESSAGES_AUTO, True, False),
        ({"telegram_rich_messages": "required"}, RICH_MESSAGES_REQUIRED, True, True),
        ({"telegram_rich_messages": "off"}, RICH_MESSAGES_OFF, False, False),
        ({"telegram_rich_messages": "bad"}, RICH_MESSAGES_OFF, False, False),
    ],
)
def test_rich_message_mode_helpers(config, mode, enabled, required):
    assert rich_messages_mode(config) == mode
    assert rich_messages_enabled(config) is enabled
    assert rich_messages_required(config) is required
