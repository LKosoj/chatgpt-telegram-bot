from __future__ import annotations

import logging
import os
from typing import Any

import telegram
from telegram import constants

from .telegram_rich import (
    MAX_RICH_MARKDOWN_BYTES,
    rich_markdown_fits,
    rich_messages_enabled,
    rich_messages_required,
    send_rich_markdown,
)
from .tool_result import direct_result_payload
from .utils import (
    cleanup_intermediate_files,
    is_direct_result,
    render_markdown_message_entities,
    split_into_chunks,
)


logger = logging.getLogger(__name__)


def _direct_result_payload(response: Any) -> dict | None:
    return direct_result_payload(response)


async def send_text_chunks(
    bot,
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    parse_mode=constants.ParseMode.MARKDOWN,
    config: dict[str, Any] | None = None,
):
    sent = []
    text = str(text or "")
    if parse_mode == constants.ParseMode.MARKDOWN and rich_messages_enabled(config):
        rich_required = rich_messages_required(config)
        if rich_markdown_fits(text):
            try:
                return [await send_rich_markdown(
                    bot,
                    chat_id=chat_id,
                    markdown=text,
                    message_thread_id=message_thread_id,
                    reply_to_message_id=reply_to_message_id,
                )]
            except Exception:
                if rich_required:
                    raise
                logger.warning(
                    "Rich markdown delivery failed; falling back to legacy text delivery",
                    exc_info=True,
                )
        elif rich_required:
            raise ValueError(f"Rich markdown message exceeds {MAX_RICH_MARKDOWN_BYTES} bytes")
        else:
            logger.warning(
                "Rich markdown delivery skipped; falling back to legacy text delivery "
                "text_bytes=%s limit=%s",
                len(text.encode("utf-8")),
                MAX_RICH_MARKDOWN_BYTES,
            )
    if parse_mode == constants.ParseMode.MARKDOWN:
        message_parts = render_markdown_message_entities(text)
    else:
        message_parts = [(chunk, None) for chunk in split_into_chunks(text)]
    for index, (chunk, entities) in enumerate(message_parts):
        kwargs = {
            "chat_id": chat_id,
            "text": chunk or "...",
            "parse_mode": None if entities else parse_mode,
        }
        if entities:
            kwargs["entities"] = entities
        if reply_to_message_id and index == 0:
            kwargs["reply_to_message_id"] = reply_to_message_id
        if message_thread_id:
            kwargs["message_thread_id"] = message_thread_id
        try:
            sent.append(await bot.send_message(**kwargs))
        except telegram.error.BadRequest:
            kwargs["parse_mode"] = None
            kwargs.pop("entities", None)
            sent.append(await bot.send_message(**kwargs))
    return sent


async def send_agent_response(
    bot,
    *,
    chat_id: int,
    response: Any,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    title: str | None = None,
    config: dict[str, Any] | None = None,
):
    if is_direct_result(response):
        payload = _direct_result_payload(response)
        return await _send_direct_payload(
            bot,
            chat_id=chat_id,
            payload=payload or {},
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message_thread_id,
            title=title,
            config=config,
        )
    text = str(response or "").strip()
    if title:
        text = f"{title}\n\n{text}" if text else title
    return await send_text_chunks(
        bot,
        chat_id=chat_id,
        text=text or "Done.",
        reply_to_message_id=reply_to_message_id,
        message_thread_id=message_thread_id,
        config=config,
    )


async def _send_direct_payload(
    bot,
    *,
    chat_id: int,
    payload: dict,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    title: str | None = None,
    config: dict[str, Any] | None = None,
):
    sent = []
    kind = payload.get("kind")
    if kind == "final":
        for artifact in payload.get("artifacts") or []:
            if isinstance(artifact, dict):
                artifact_payload = dict(artifact)
                artifact_payload["preserve_after_delivery"] = True
                sent.extend(await _send_direct_payload(
                    bot,
                    chat_id=chat_id,
                    payload=artifact_payload,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
                    config=config,
                ))
        text = str(payload.get("text") or "").strip()
        if title or text:
            sent.extend(await send_text_chunks(
                bot,
                chat_id=chat_id,
                text=f"{title}\n\n{text}" if title and text else (title or text),
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                parse_mode=constants.ParseMode.MARKDOWN,
                config=config,
            ))
        return sent

    if kind == "text":
        text = str(payload.get("add_value") or payload.get("value") or "").strip()
        if title:
            text = f"{title}\n\n{text}" if text else title
        return await send_text_chunks(
            bot,
            chat_id=chat_id,
            text=text or "Done.",
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message_thread_id,
            parse_mode=constants.ParseMode.MARKDOWN if payload.get("format") == "markdown" else None,
            config=config,
        )

    common = {"chat_id": chat_id}
    if reply_to_message_id:
        common["reply_to_message_id"] = reply_to_message_id
    if message_thread_id:
        common["message_thread_id"] = message_thread_id
    caption = payload.get("caption")
    if caption:
        common["caption"] = str(caption)

    value = (
        payload.get("value")
        or payload.get("file_path")
        or payload.get("path")
        or payload.get("output_path")
        or payload.get("artifact_path")
        or payload.get("url")
    )
    result_format = payload.get("format")
    if not value:
        return sent

    if kind in {"file", "photo", "gif"} and result_format == "path":
        path = os.path.realpath(os.path.expanduser(str(value)))
        if not os.path.isfile(path):
            await send_text_chunks(
                bot,
                chat_id=chat_id,
                text=f"Artifact path is unavailable: {os.path.basename(path)}",
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                parse_mode=None,
            )
            return sent
        with open(path, "rb") as fh:
            if kind == "photo":
                sent.append(await bot.send_photo(**common, photo=fh))
            elif kind == "gif":
                sent.append(await bot.send_animation(**common, animation=fh))
            else:
                sent.append(await bot.send_document(**common, document=fh))
        cleanup_payload = dict(payload)
        cleanup_payload["value"] = path
        cleanup_intermediate_files({"direct_result": cleanup_payload})
        return sent

    if kind == "photo":
        sent.append(await bot.send_photo(**common, photo=value))
    elif kind == "gif":
        sent.append(await bot.send_animation(**common, animation=value))
    elif kind == "file":
        sent.append(await bot.send_document(**common, document=value))
    else:
        sent.extend(await send_text_chunks(
            bot,
            chat_id=chat_id,
            text=str(value),
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message_thread_id,
            parse_mode=None,
        ))
    return sent
