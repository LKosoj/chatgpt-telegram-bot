"""Exemplar-тест: структурное свойство ремонта истории после обрыва tool_call.

Симулируется рестарт процесса: в БД/памяти осталось сообщение
assistant(tool_calls) с несколькими параллельными вызовами, но ни один
tool-ответ не пришёл (процесс умер посреди цикла инструмента). Проверяется
не текст, а форма результата `_repair_tool_call_history`: каждый оборванный
tool_call_id закрыт РОВНО одним синтетическим tool-сообщением, оно несёт
маркер INTERRUPTED_TOOL_RESULT_NOTICE и не содержит top-level поля "name"
(иначе форма сообщения разойдётся с тем, что ожидает OpenAI-совместимый API).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import INTERRUPTED_TOOL_RESULT_NOTICE, OpenAIHelper


def _make_helper() -> OpenAIHelper:
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    return helper


def _history_with_interrupted_tool_calls() -> list:
    """Рестарт оборвал историю сразу после assistant(tool_calls) с двумя
    параллельными вызовами — ни один tool-ответ не был записан."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "сделай две вещи"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            ],
        },
        # tool-ответов нет — история оборвана рестартом.
        {"role": "user", "content": "ты ещё тут?"},
    ]


def test_each_interrupted_tool_call_id_is_closed_by_exactly_one_tool_message():
    helper = _make_helper()
    state_key = 1
    helper.conversations[state_key] = _history_with_interrupted_tool_calls()

    helper._repair_tool_call_history(state_key)

    repaired = helper.conversations[state_key]
    closed_ids = [m["tool_call_id"] for m in repaired if m.get("role") == "tool"]
    assert sorted(closed_ids) == ["call_1", "call_2"]
    assert len(closed_ids) == len(set(closed_ids))


def test_interrupted_tool_call_repair_content_carries_notice_marker():
    helper = _make_helper()
    state_key = 2
    helper.conversations[state_key] = _history_with_interrupted_tool_calls()

    helper._repair_tool_call_history(state_key)

    repaired = helper.conversations[state_key]
    tool_messages = {m["tool_call_id"]: m for m in repaired if m.get("role") == "tool"}
    for call_id in ("call_1", "call_2"):
        payload = json.loads(tool_messages[call_id]["content"])
        assert payload["error"] == INTERRUPTED_TOOL_RESULT_NOTICE


def test_interrupted_tool_call_repair_message_has_no_top_level_name_field():
    helper = _make_helper()
    state_key = 3
    helper.conversations[state_key] = _history_with_interrupted_tool_calls()

    helper._repair_tool_call_history(state_key)

    repaired = helper.conversations[state_key]
    tool_messages = [m for m in repaired if m.get("role") == "tool"]
    assert tool_messages  # сверка, что фикстура реально породила ремонт
    for m in tool_messages:
        assert set(m.keys()) == {"role", "tool_call_id", "content"}
