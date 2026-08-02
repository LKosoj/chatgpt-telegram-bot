"""Exemplar-тест: структурное свойство компакции истории при отказе LLM-суммаризатора.

В отличие от tests/test_openai_helper_summarize_trim.py (юнит-тесты отдельных
функций), здесь воспроизводится ровно та последовательность вызовов, что
делает реальный вызывающий код в __common_get_chat_response /
__common_get_chat_response_vision (bot/openai_helper.py ~1510-1525):
сначала пробуем `_summarize_and_trim`, а если оно вернуло False (в т.ч. из-за
исключения в суммаризаторе) — накатываем `_fallback_trim_with_summary`.
Проверяется не текст, а СТРУКТУРА результата: ничего не потеряно, размер
ограничен, tool_calls/tool пара не разорвана.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


def _make_helper() -> OpenAIHelper:
    helper = object.__new__(OpenAIHelper)
    helper.config = {
        'summary_enabled': True,
        'summary_min_messages_between_runs': 6,
        'summary_target_keep_ratio': 0.5,
        'max_history_size': 17,
        'light_model': 'cheap-model',
        'model': 'main-model',
    }
    helper.conversations = {}
    helper._last_summary_at = {}
    return helper


def _history_with_dangling_tool_reply_at_naive_cut() -> list:
    """20 несистемных сообщений; при max_history_size=17 наивная точка среза
    (naive_cut = 20 - 17 = 3) указывает ровно на tool-ответ, следующий сразу
    за assistant(tool_calls) на индексе 2. Без защитного сдвига в
    `_safe_cut_index` наивный срез разорвал бы эту пару: tool_calls ушёл бы в
    отбрасываемую половину, а его tool-ответ остался бы в хвосте — история
    стала бы невалидной для API. Ранние сообщения несут маркер
    "SECRET-FACT", по которому проверяется сохранность содержимого.
    """
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "SECRET-FACT: rocket launch is on 2099-01-01"})
    msgs.append({"role": "assistant", "content": "ack"})
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    })
    msgs.append({"role": "tool", "tool_call_id": "t1", "content": "result-of-lookup"})
    for i in range(8):
        msgs.append({"role": "user", "content": f"filler question {i}"})
        msgs.append({"role": "assistant", "content": f"filler answer {i}"})
    return msgs


async def _run_real_callsite_chain(helper: OpenAIHelper, state_key) -> None:
    """Повторяет реальную цепочку вызовов места использования: суммаризация
    падает -> summarized=False -> откат на детерминированный fallback."""
    helper._summarise_window = AsyncMock(side_effect=asyncio.TimeoutError("llm down"))
    summarized = await helper._summarize_and_trim(
        state_key, chat_id=state_key, session_id=None, memory_user_id=state_key,
    )
    assert summarized is False
    helper._fallback_trim_with_summary(state_key)


@pytest.mark.asyncio
async def test_llm_summarize_failure_does_not_lose_discarded_conversation_content():
    helper = _make_helper()
    state_key = 1
    helper.conversations[state_key] = _history_with_dangling_tool_reply_at_naive_cut()

    await _run_real_callsite_chain(helper, state_key)

    new_conv = helper.conversations[state_key]
    prior_summaries = [
        m for m in new_conv
        if isinstance(m.get('content'), str) and m['content'].startswith('[prior_summary]:')
    ]
    assert len(prior_summaries) == 1
    assert "SECRET-FACT" in prior_summaries[0]['content']


@pytest.mark.asyncio
async def test_llm_summarize_failure_keeps_history_within_max_history_size():
    helper = _make_helper()
    state_key = 2
    helper.conversations[state_key] = _history_with_dangling_tool_reply_at_naive_cut()

    await _run_real_callsite_chain(helper, state_key)

    new_conv = helper.conversations[state_key]
    non_system = [m for m in new_conv if m.get('role') != 'system']
    assert len(non_system) <= helper.config['max_history_size']


@pytest.mark.asyncio
async def test_llm_summarize_failure_never_splits_tool_call_from_its_result():
    helper = _make_helper()
    state_key = 3
    helper.conversations[state_key] = _history_with_dangling_tool_reply_at_naive_cut()

    await _run_real_callsite_chain(helper, state_key)

    new_conv = helper.conversations[state_key]
    assistant_present = any(
        m.get('role') == 'assistant' and m.get('tool_calls') for m in new_conv if isinstance(m, dict)
    )
    tool_present = any(
        m.get('role') == 'tool' and m.get('tool_call_id') == 't1' for m in new_conv if isinstance(m, dict)
    )
    # Либо оба сообщения пары выжили в хвосте, либо оба свёрнуты в summary —
    # никогда только один из двух (это и есть "не разорвано").
    assert assistant_present == tool_present
