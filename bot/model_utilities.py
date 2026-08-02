from __future__ import annotations

import asyncio
import logging
from typing import Any

from .chat_response_utils import (
    first_choice_or_raise,
    required_choice_message_text,
)
from .utils import log_exception_shape

logger = logging.getLogger(__name__)


class ModelUtilities:
    """Thin layer over ``OpenAIHelper.chat_completion`` for one-shot, cheap
    LLM utility calls (classification, titling, summarisation).

    Touches only ``helper.chat_completion(**kwargs)`` and ``helper.config`` --
    nothing else on the helper -- so it works both against the real
    ``OpenAIHelper`` and against the minimal hand-rolled helper doubles used
    in tests (e.g. objects that only implement ``chat_completion``/``config``).

    Model selection is a shared config-key fallback chain
    (``model_fallback_keys``), applied only when ``model`` is not given
    explicitly. Every method applies a single ``asyncio.wait_for(timeout_seconds)``
    and logs a failure once. All methods degrade to ``None`` on error/timeout,
    EXCEPT ``summarize_window``, which re-raises -- ``OpenAIHelper.
    _summarize_and_trim`` relies on that exception to trigger its
    head-preserve fallback path.
    """

    def __init__(self, helper: Any) -> None:
        self._helper = helper

    def _resolve_model(self, model: str | None, model_fallback_keys: tuple[str, ...]) -> str | None:
        if model:
            return model
        config = self._helper.config
        for key in model_fallback_keys:
            value = config.get(key)
            if value:
                return value
        return None

    async def one_shot(
        self,
        *,
        kind: str,
        messages: list,
        timeout_seconds: float,
        model: str | None = None,
        model_fallback_keys: tuple[str, ...] = ('light_model', 'model'),
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra_headers: dict | None = None,
    ) -> Any | None:
        """Generic one-shot chat completion.

        Returns the raw completion response, or ``None`` on timeout or any
        other error (the error is logged once here).
        """
        model_to_use = self._resolve_model(model, model_fallback_keys)
        try:
            return await asyncio.wait_for(
                self._helper.chat_completion(
                    kind=kind,
                    model=model_to_use,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    stream=False,
                    extra_headers=extra_headers or {"X-Title": "tgBot"},
                ),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            logger.error(
                "Model utility call failed kind=%s error=%s",
                kind,
                log_exception_shape(exc),
            )
            return None

    async def classify_json(
        self,
        *,
        kind: str,
        messages: list,
        timeout_seconds: float,
        model: str | None = None,
        model_fallback_keys: tuple[str, ...] = ('light_model', 'model'),
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
    ) -> str | None:
        """One-shot JSON-mode classification call.

        Returns the raw response text (shape/parsing is caller-specific, so
        it is left to the caller), or ``None`` on failure.
        """
        response = await self.one_shot(
            kind=kind,
            messages=messages,
            timeout_seconds=timeout_seconds,
            model=model,
            model_fallback_keys=model_fallback_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_headers=extra_headers,
        )
        if response is None:
            return None
        try:
            return first_choice_or_raise(response).message.content or ""
        except Exception as exc:
            logger.error(
                "Model utility classify_json content extraction failed kind=%s error=%s",
                kind,
                log_exception_shape(exc),
            )
            return None

    async def generate_title(
        self,
        *,
        messages: list,
        timeout_seconds: float,
        model: str | None = None,
        model_fallback_keys: tuple[str, ...] = ('light_model', 'model'),
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
    ) -> tuple[str | None, int]:
        """One-shot title-generation call.

        Returns ``(text, total_tokens)`` on success, or ``(None, 0)`` on
        failure.
        """
        response = await self.one_shot(
            kind='session_name',
            messages=messages,
            timeout_seconds=timeout_seconds,
            model=model,
            model_fallback_keys=model_fallback_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
        )
        if response is None:
            return None, 0
        try:
            text = required_choice_message_text(first_choice_or_raise(response))
        except Exception as exc:
            logger.error(
                "Model utility generate_title content extraction failed error=%s",
                log_exception_shape(exc),
            )
            return None, 0
        tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        return text, tokens

    async def summarize_window(
        self,
        *,
        messages: list,
        timeout_seconds: float,
        model: str | None = None,
        model_fallback_keys: tuple[str, ...] = ('summary_model', 'light_model', 'model'),
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        """One-shot summarisation call.

        Unlike the other methods, this does NOT degrade to ``None`` on
        error -- it re-raises (timeout included), because
        ``OpenAIHelper._summarize_and_trim`` catches the exception itself to
        decide whether to fall back to the head-preserve trim.
        """
        model_to_use = self._resolve_model(model, model_fallback_keys)
        response = await asyncio.wait_for(
            self._helper.chat_completion(
                kind='summary',
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_headers=extra_headers or {"X-Title": "tgBot"},
            ),
            timeout=timeout_seconds,
        )
        summary = first_choice_or_raise(response).message.content or ""
        return summary.strip()
