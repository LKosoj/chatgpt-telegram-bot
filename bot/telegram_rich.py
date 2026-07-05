from __future__ import annotations

from typing import Any

from telegram import Message

RICH_MESSAGES_AUTO = "auto"
RICH_MESSAGES_REQUIRED = "required"
RICH_MESSAGES_OFF = "off"
RICH_MESSAGES_MODES = {
    RICH_MESSAGES_AUTO,
    RICH_MESSAGES_REQUIRED,
    RICH_MESSAGES_OFF,
}
MAX_RICH_MARKDOWN_BYTES = 32768


def rich_markdown_fits(markdown: str) -> bool:
    return len(str(markdown or "").encode("utf-8")) <= MAX_RICH_MARKDOWN_BYTES


def build_markdown_rich_message(
    markdown: str,
    *,
    is_rtl: bool | None = None,
    skip_entity_detection: bool | None = None,
) -> dict[str, Any]:
    rich_message: dict[str, Any] = {"markdown": str(markdown or "")}
    if is_rtl is not None:
        rich_message["is_rtl"] = bool(is_rtl)
    if skip_entity_detection is not None:
        rich_message["skip_entity_detection"] = bool(skip_entity_detection)
    return rich_message


def _to_api_value(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_api_value(value) for key, value in payload.items() if value is not None}


def build_send_rich_message_payload(
    *,
    chat_id: int | str,
    markdown: str,
    message_thread_id: int | None = None,
    reply_to_message_id: int | None = None,
    reply_markup: Any | None = None,
    is_rtl: bool | None = None,
    skip_entity_detection: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_thread_id": message_thread_id,
        "rich_message": build_markdown_rich_message(
            markdown,
            is_rtl=is_rtl,
            skip_entity_detection=skip_entity_detection,
        ),
        "reply_markup": reply_markup,
    }
    if reply_to_message_id is not None:
        payload["reply_parameters"] = {"message_id": int(reply_to_message_id)}
    return _drop_none(payload)


def build_send_rich_message_draft_payload(
    *,
    chat_id: int,
    draft_id: int,
    markdown: str,
    message_thread_id: int | None = None,
    is_rtl: bool | None = None,
    skip_entity_detection: bool | None = None,
) -> dict[str, Any]:
    if int(draft_id) == 0:
        raise ValueError("draft_id must be non-zero")
    return _drop_none({
        "chat_id": int(chat_id),
        "message_thread_id": message_thread_id,
        "draft_id": int(draft_id),
        "rich_message": build_markdown_rich_message(
            markdown,
            is_rtl=is_rtl,
            skip_entity_detection=skip_entity_detection,
        ),
    })


def _message_from_post_result(result: Any, bot) -> Any:
    if isinstance(result, dict) and "message_id" in result and "chat" in result:
        message = Message.de_json(result, bot)
        return message if message is not None else result
    return result


async def send_rich_markdown(
    bot,
    *,
    chat_id: int | str,
    markdown: str,
    message_thread_id: int | None = None,
    reply_to_message_id: int | None = None,
    reply_markup: Any | None = None,
    is_rtl: bool | None = None,
    skip_entity_detection: bool | None = None,
    api_kwargs: dict[str, Any] | None = None,
    **timeouts,
) -> Any:
    result = await bot._post(
        "sendRichMessage",
        data=build_send_rich_message_payload(
            chat_id=chat_id,
            markdown=markdown,
            message_thread_id=message_thread_id,
            reply_to_message_id=reply_to_message_id,
            reply_markup=reply_markup,
            is_rtl=is_rtl,
            skip_entity_detection=skip_entity_detection,
        ),
        api_kwargs=api_kwargs,
        **timeouts,
    )
    return _message_from_post_result(result, bot)


async def send_rich_markdown_draft(
    bot,
    *,
    chat_id: int,
    draft_id: int,
    markdown: str,
    message_thread_id: int | None = None,
    is_rtl: bool | None = None,
    skip_entity_detection: bool | None = None,
    api_kwargs: dict[str, Any] | None = None,
    **timeouts,
) -> Any:
    return await bot._post(
        "sendRichMessageDraft",
        data=build_send_rich_message_draft_payload(
            chat_id=chat_id,
            draft_id=draft_id,
            markdown=markdown,
            message_thread_id=message_thread_id,
            is_rtl=is_rtl,
            skip_entity_detection=skip_entity_detection,
        ),
        api_kwargs=api_kwargs,
        **timeouts,
    )


def rich_messages_mode(config: dict[str, Any] | None) -> str:
    if not config:
        return RICH_MESSAGES_OFF
    mode = str(config.get("telegram_rich_messages") or RICH_MESSAGES_OFF).strip().lower()
    return mode if mode in RICH_MESSAGES_MODES else RICH_MESSAGES_OFF


def rich_messages_enabled(config: dict[str, Any] | None) -> bool:
    return rich_messages_mode(config) in {
        RICH_MESSAGES_AUTO,
        RICH_MESSAGES_REQUIRED,
    }


def rich_messages_required(config: dict[str, Any] | None) -> bool:
    return rich_messages_mode(config) == RICH_MESSAGES_REQUIRED
