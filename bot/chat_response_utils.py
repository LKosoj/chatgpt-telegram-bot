from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

EMPTY_MODEL_RESPONSE_ERROR = "Модель вернула пустой ответ"
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
RAW_TOOL_RESULT_RE = re.compile(r"^Function\s+[\w.\-]+\s+returned:\s*", re.IGNORECASE)


def choice_message_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return ""
    content = THINK_BLOCK_RE.sub("", content)
    content = THINK_TAG_RE.sub("\n", content)
    content = content.strip()
    if RAW_TOOL_RESULT_RE.match(content):
        return ""
    return content


def required_choice_message_text(choice: Any) -> str:
    content = choice_message_text(choice)
    if content:
        return content
    message = getattr(choice, "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    logger.warning(
        "Model returned empty assistant content; finish_reason=%s tool_call_count=%s",
        getattr(choice, "finish_reason", None),
        len(tool_calls) if tool_calls else 0,
    )
    raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)


def response_has_message_text(response: Any) -> bool:
    return any(choice_message_text(choice) for choice in getattr(response, "choices", []) or [])


def response_total_tokens(response: Any) -> int:
    tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
    try:
        return int(tokens)
    except (TypeError, ValueError):
        return 0


def first_choice_or_raise(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning("Model response has no choices")
        raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)
    return choices[0]
