from __future__ import annotations

import logging
import re
from typing import Any

from .i18n import localized_text

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


def response_prompt_completion_tokens(response: Any) -> dict[str, int] | None:
    """Prompt/completion split for one response, or None when unavailable.

    Mirrors response_total_tokens' shape but for the per-direction split
    used by bot.pricing.resolve_chat_cost. An honest None (rather than a
    guessed 0/0) is returned when the response has no usable split.
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None or completion_tokens is None:
        return None
    try:
        return {"prompt": int(prompt_tokens), "completion": int(completion_tokens)}
    except (TypeError, ValueError):
        return None


def aggregate_usage_split(token_accumulator, usage_accumulator) -> tuple[int, int] | None:
    """Combine a parallel usage_accumulator (list of {'prompt', 'completion'}
    dicts, appended in lockstep with token_accumulator) into a single
    (prompt_tokens, completion_tokens) split.

    Returns None when usage_accumulator does not have an entry for every
    round trip counted in token_accumulator (e.g. one round trip's response
    had no usage split, or a synthetic extra-tokens entry with no response
    at all) -- an incomplete split is reported as unknown rather than
    guessed.
    """
    if not usage_accumulator or not token_accumulator:
        return None
    if len(usage_accumulator) != len(token_accumulator):
        return None
    prompt_total = sum(entry.get("prompt", 0) for entry in usage_accumulator)
    completion_total = sum(entry.get("completion", 0) for entry in usage_accumulator)
    return prompt_total, completion_total


def first_choice_or_raise(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning("Model response has no choices")
        raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)
    return choices[0]


def leading_system_count(messages: list) -> int:
    """Count the leading contiguous run of ``role == 'system'`` messages.

    Extracted from the identical loop in ``OpenAIHelper._summarize_and_trim``
    and ``OpenAIHelper._fallback_trim_with_summary`` (both protect the
    leading system messages -- assistant_prompt, mode prompt, etc. -- before
    trimming/summarising the rest of the conversation).
    """
    head_end = 0
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            head_end += 1
        else:
            break
    return head_end


async def finalize_chat_answer(
    helper: Any,
    chat_id: int,
    response: Any,
    *,
    plugins_used: tuple = (),
    token_accumulator: list | None = None,
    session_id: str | None = None,
) -> tuple[str, int]:
    """Build the final answer text and append it to conversation history.

    Shared tail of ``bot.chat_run.ChatRun.run_non_stream`` and
    ``OpenAIHelper._interpret_image_text_response``: choice numbering when
    ``n_choices > 1``, writing the assistant message via
    ``helper._add_to_history``, and the usage/plugins-used footer. The
    caller still owns usage-split bookkeeping and AIRunEnd events -- this
    only returns ``(answer, total_tokens)``.
    """
    answer = ""
    if len(response.choices) > 1 and helper.config["n_choices"] > 1:
        for index, choice in enumerate(response.choices):
            content = required_choice_message_text(choice)
            if index == 0:
                await helper._add_to_history(chat_id, role="assistant", content=content, session_id=session_id)
            answer += f"{index + 1}⃣\n"
            answer += content
            answer += "\n\n"
    else:
        answer = required_choice_message_text(first_choice_or_raise(response))
        await helper._add_to_history(chat_id, role="assistant", content=answer, session_id=session_id)

    bot_language = helper.config["bot_language"]
    show_plugins_used = len(plugins_used) > 0 and helper.config["show_plugins_used"]
    plugin_names = tuple(helper.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
    total_tokens = sum(token_accumulator or []) or response_total_tokens(response)
    if helper.config["show_usage"]:
        usage_tokens = response_total_tokens(response)
        answer += (
            "\n\n---\n"
            f"💰 {str(total_tokens)} {localized_text('stats_tokens', bot_language)}"
        )
        usage = response.usage
        if (
            total_tokens == usage_tokens
            and usage is not None
            and usage.prompt_tokens is not None
            and usage.completion_tokens is not None
        ):
            answer += (
                f" ({str(usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                f" {str(usage.completion_tokens)} "
                f"{localized_text('completion', bot_language)})"
            )
        if show_plugins_used:
            answer += f"\n🔌 {', '.join(plugin_names)}"
    elif show_plugins_used:
        answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

    return answer, total_tokens
