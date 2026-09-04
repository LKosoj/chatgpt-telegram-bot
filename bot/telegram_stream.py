"""Shared streaming renderer for (content, tokens) chunks from OpenAIHelper.

See docs/remediation_2026-09-04/T12-telegram-stream.md for the design this
module implements. It fixes architecture_code_review_2026-09-04.md §3.6:
an error on the *last* ``edit`` used to make the final chunk disappear (the
``async for`` had already stopped, so ``continue`` did nothing) and left
``total_tokens`` at 0 because the token assignment sat after the failing
``try/except``.

The module owns only the generic part of the streaming loop: chunking by
``chunk_size``, throttling/backoff, and guaranteed delivery of the final
chunk. Everything transport-specific (how to edit/send a message, what a
direct_result means for this call site) is injected as callbacks by the
caller — the module never touches python-telegram-bot directly.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from telegram.error import RetryAfter, TimedOut

from .utils import is_direct_result, log_exception_shape, split_into_chunks

logger = logging.getLogger(__name__)

# Колбэки — единственное, что знает о конкретном транспорте (обычный чат по
# chat_id+message_id, инлайн по inline_message_id, vision-реплай и т.п.).
# Сам модуль ни разу не вызывает python-telegram-bot напрямую.
EditFn = Callable[[Any, str, bool], Awaitable[None]]        # (message_id, text, markdown) -> None
SendFn = Callable[[str, bool], Awaitable[Any]]               # (text, markdown) -> объект с .message_id
DirectResultFn = Callable[[str, int], Awaitable[None]]        # (raw_content, total_tokens) -> None
CutoffFn = Callable[[str], int]                               # (content) -> порог для throttling


@dataclass
class StreamOutcome:
    final_text: str = ""
    total_tokens: int = 0
    message_ids: list = field(default_factory=list)
    direct_result: Any = None   # сырой content, если стрим завершился direct_result'ом
    delivered: bool = True      # False, только если финальный чанк не доставлен даже фолбэком


async def _swallow(coro: Awaitable[None]) -> None:
    try:
        await coro
    except Exception as exc:
        logger.debug("stream_to_telegram: swallowed edit error=%s", log_exception_shape(exc))


async def _swallow_send(
    send: SendFn, text: str, markdown: bool, message_ids: list, current: Any
) -> Any:
    try:
        msg = await send(text, markdown)
        message_ids.append(msg.message_id)
        return msg
    except Exception as exc:
        logger.debug("stream_to_telegram: swallowed send error=%s", log_exception_shape(exc))
        return current


async def stream_to_telegram(
    stream: AsyncIterator[tuple[str, str]],
    *,
    edit: EditFn,
    send: Optional[SendFn],
    cutoff_for: CutoffFn,
    on_direct_result: Optional[DirectResultFn] = None,
    chunk_size: int = 4096,
    max_final_attempts: int = 3,
    retry_backoff_seconds: float = 5.0,
    inter_edit_delay: float = 0.01,
) -> StreamOutcome:
    """Render an OpenAIHelper ``(content, tokens)`` stream into Telegram.

    ``stream`` follows the ``get_chat_response_stream``/``interpret_image_stream``
    contract: ``content`` is the FULL response accumulated so far (not a
    delta), ``tokens`` is ``'not_finished'`` on every chunk except the last,
    where it is a string with the token count. ``content`` may be a
    direct_result payload — handled by ``on_direct_result`` and the loop
    stops immediately.

    ``send is None`` means the caller has no way to open a NEW message
    (inline mode) — the final-delivery guarantee then only retries ``edit``.

    No side effects beyond the injected callbacks: usage accounting, hook
    dispatch, etc. stay with the caller.
    """
    prev = ""
    backoff = 0
    last_published_chunk = 0
    sent_message: Any = None
    inline_opened = False       # инлайн: первый edit плейсхолдера уже прошёл
    final_text = ""             # полный ответ (для StreamOutcome/хуков)
    pending_content = ""        # текст сообщения, которое сейчас редактируется
    total_tokens = 0
    final_delivered = False
    message_ids: list = []

    async for content, tokens in stream:
        if on_direct_result is not None and is_direct_result(content):
            if tokens != "not_finished":
                total_tokens = int(tokens)
            await on_direct_result(content, total_tokens)
            return StreamOutcome(
                total_tokens=total_tokens, message_ids=message_ids,
                direct_result=content, delivered=True,
            )

        if len(content.strip()) == 0:
            continue

        final_text = content
        is_final = tokens != "not_finished"
        if is_final:
            # Токены модели фиксируются независимо от успеха доставки —
            # закрывает "usage=0 при потерянном чанке" из репро-теста.
            total_tokens = int(tokens)
        final_delivered = False

        # --- переполнение chunk_size: закрыть предыдущий чанк отдельным
        # сообщением, продолжить расти в новом ("хвостовой" чанк).
        # Инлайн (send is None) новое сообщение открыть не может — там, как и
        # раньше, весь текст идёт в один edit без нарезки. ---
        if send is not None:
            chunks = split_into_chunks(content, chunk_size)
            if len(chunks) > 1:
                content = chunks[-1]
                if last_published_chunk != len(chunks) - 1:
                    last_published_chunk += 1
                    previous_chunk = chunks[-2]
                    if sent_message is not None:
                        await _swallow(edit(sent_message.message_id, previous_chunk, True))
                    else:
                        sent_message = await _swallow_send(
                            send, previous_chunk or "...", False, message_ids, sent_message
                        )
                    # Хвост открывается новым сообщением; при неудаче
                    # sent_message сбрасывается, и следующий чанк (или
                    # гарантия доставки) откроет его заново.
                    sent_message = await _swallow_send(
                        send, content or "...", False, message_ids, None
                    )
                    pending_content = content
                    continue
        pending_content = content

        cutoff = cutoff_for(content) + backoff

        if send is not None:
            has_open_message = sent_message is not None
        else:
            has_open_message = inline_opened

        if not has_open_message:
            if send is not None:
                try:
                    sent_message = await send(content, is_final)
                    message_ids.append(sent_message.message_id)
                    prev = content
                    final_delivered = is_final
                except Exception as exc:
                    logger.warning(
                        "stream_to_telegram: initial send failed error=%s",
                        log_exception_shape(exc),
                    )
                    # Не поднимаем: следующий чанк модели попробует снова.
                    # Если поток так и закончится без успешной отправки,
                    # сработает гарантия финальной доставки ниже.
            else:
                # Инлайн: "первое сообщение" — это edit уже существующего
                # плейсхолдера (inline_message_id создан заранее).
                try:
                    await edit(None, content, is_final)
                    inline_opened = True
                    prev = content
                    final_delivered = is_final
                except Exception as exc:
                    logger.debug(
                        "stream_to_telegram: initial inline edit failed error=%s",
                        log_exception_shape(exc),
                    )
            continue

        if not (abs(len(content) - len(prev)) > cutoff or is_final):
            continue

        prev = content
        try:
            await edit(sent_message.message_id if sent_message is not None else None, content, is_final)
            final_delivered = is_final
            await asyncio.sleep(inter_edit_delay)
        except RetryAfter as exc:
            backoff += 5
            await asyncio.sleep(exc.retry_after)
        except TimedOut:
            backoff += 5
            await asyncio.sleep(0.5)
        except Exception as exc:
            backoff += 5
            logger.debug(
                "stream_to_telegram: throttled edit failed error=%s",
                log_exception_shape(exc),
            )

    if final_text and not final_delivered:
        # Доставляем текст ТЕКУЩЕГО сообщения (хвост после нарезки), а не
        # полный ответ: иначе при переполнении 4096 хвостовое сообщение
        # перезаписалось бы началом ответа, уже опубликованным ранее.
        final_delivered = await _guarantee_final_delivery(
            sent_message, pending_content,
            edit=edit, send=send, message_ids=message_ids,
            max_attempts=max_final_attempts,
            backoff_seconds=retry_backoff_seconds,
        )

    return StreamOutcome(
        final_text=final_text, total_tokens=total_tokens,
        message_ids=message_ids, direct_result=None, delivered=final_delivered,
    )


async def _guarantee_final_delivery(
    sent_message, final_text, *, edit, send, message_ids, max_attempts, backoff_seconds,
) -> bool:
    """Retry delivery of the final text up to ``max_attempts`` times with
    growing backoff; if a message already existed and every ``edit`` attempt
    failed, try ``send`` once as a new message (when available)."""
    for attempt in range(max_attempts):
        try:
            if sent_message is not None:
                await edit(sent_message.message_id, final_text, True)
            elif send is not None:
                sent_message = await send(final_text, True)
                message_ids.append(sent_message.message_id)
            else:
                await edit(None, final_text, True)  # инлайн без открытого сообщения
            return True
        except Exception as exc:
            logger.warning(
                "stream_to_telegram: final delivery attempt %d/%d failed error=%s",
                attempt + 1, max_attempts, log_exception_shape(exc),
            )
            await asyncio.sleep(backoff_seconds * (attempt + 1))

    if send is not None:
        try:
            fallback = await send(final_text, True)
            message_ids.append(fallback.message_id)
            return True
        except Exception as exc:
            logger.error(
                "stream_to_telegram: fallback send_message also failed error=%s",
                log_exception_shape(exc),
            )
    else:
        logger.error(
            "stream_to_telegram: final text could not be delivered "
            "(inline mode has no send fallback)"
        )
    return False
