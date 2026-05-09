from __future__ import annotations

import json
import logging
import os
from typing import Any

import telegram
from telegram import constants

from .utils import is_direct_result, split_into_chunks


logger = logging.getLogger(__name__)


def _direct_result_payload(response: Any) -> dict | None:
    if not isinstance(response, dict):
        try:
            response = json.loads(response)
        except Exception:
            return None
    payload = response.get("direct_result") if isinstance(response, dict) else None
    return payload if isinstance(payload, dict) else None


async def send_text_chunks(
    bot,
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    parse_mode=constants.ParseMode.MARKDOWN,
):
    sent = []
    for index, chunk in enumerate(split_into_chunks(str(text or ""))):
        kwargs = {
            "chat_id": chat_id,
            "text": chunk or "...",
            "parse_mode": parse_mode,
        }
        if reply_to_message_id and index == 0:
            kwargs["reply_to_message_id"] = reply_to_message_id
        if message_thread_id:
            kwargs["message_thread_id"] = message_thread_id
        try:
            sent.append(await bot.send_message(**kwargs))
        except telegram.error.BadRequest:
            kwargs["parse_mode"] = None
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
    )


async def _send_direct_payload(
    bot,
    *,
    chat_id: int,
    payload: dict,
    reply_to_message_id: int | None = None,
    message_thread_id: int | None = None,
    title: str | None = None,
):
    sent = []
    kind = payload.get("kind")
    if kind == "final":
        text = str(payload.get("text") or "").strip()
        if title or text:
            sent.extend(await send_text_chunks(
                bot,
                chat_id=chat_id,
                text=f"{title}\n\n{text}" if title and text else (title or text),
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                parse_mode=constants.ParseMode.MARKDOWN,
            ))
            reply_to_message_id = None
        for artifact in payload.get("artifacts") or []:
            if isinstance(artifact, dict):
                sent.extend(await _send_direct_payload(
                    bot,
                    chat_id=chat_id,
                    payload=artifact,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
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
        )

    common = {"chat_id": chat_id}
    if reply_to_message_id:
        common["reply_to_message_id"] = reply_to_message_id
    if message_thread_id:
        common["message_thread_id"] = message_thread_id
    caption = payload.get("caption")
    if caption:
        common["caption"] = str(caption)

    value = payload.get("value") or payload.get("file_path") or payload.get("url")
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
