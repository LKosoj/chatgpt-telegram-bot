"""Tests for T4: ``_summarize_and_trim`` and ``_safe_cut_index``.

Covers the new summarisation path that replaces the old ``__summarise +
reset_chat_history`` overflow handler in ``OpenAIHelper``:

* successful trim preserves facts in the summary system message
* safe-cut keeps assistant/tool_calls pairs together
* the leading system prompt is never sliced off
* failure of the summary call short-circuits to ``False`` so callers run
  the head-preserve fallback
* throttle skips reruns when the conversation grew by less than
  ``summary_min_messages_between_runs`` since the last summary
* ``_safe_cut_index`` returns 0 on unresolvable tool chains
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


def _base_config() -> dict:
    return {
        'summary_enabled': True,
        'summary_model': 'cheap-model',
        'summary_max_tokens': 400,
        'summary_timeout_seconds': 5.0,
        'summary_min_messages_between_runs': 6,
        'summary_target_keep_ratio': 0.5,
        'max_history_size': 15,
        'light_model': 'cheap-model',
        'model': 'main-model',
    }


def _make_helper(config: dict | None = None) -> OpenAIHelper:
    helper = object.__new__(OpenAIHelper)
    helper.config = config or _base_config()
    helper.conversations = {}
    helper._last_summary_at = {}
    return helper


@pytest.mark.asyncio
async def test_long_history_triggers_summary_and_preserves_facts():
    helper = _make_helper()
    state_key = 42
    msgs = [{"role": "system", "content": "You are an assistant."}]
    # First exchange mentions a key fact — must end up in the summary window.
    msgs.append({"role": "user", "content": "Привет, я живу в Москве и меня зовут Аня."})
    msgs.append({"role": "assistant", "content": "Очень приятно, Аня!"})
    # Filler that pushes the fact past the cut point.
    for i in range(14):
        msgs.append({"role": "user", "content": f"Вопрос номер {i}"})
        msgs.append({"role": "assistant", "content": f"Ответ номер {i}"})
    helper.conversations[state_key] = msgs

    helper._summarise_window = AsyncMock(
        return_value="Пользователь сказал, что живёт в Москве и его зовут Аня."
    )

    original_len = len(helper.conversations[state_key])
    ok = await helper._summarize_and_trim(
        state_key,
        chat_id=state_key,
        session_id=None,
        memory_user_id=state_key,
    )

    assert ok is True
    helper._summarise_window.assert_awaited_once()
    new_conv = helper.conversations[state_key]
    # Trim must shrink the conversation.
    assert len(new_conv) < original_len
    # System prompt still first.
    assert new_conv[0] == {"role": "system", "content": "You are an assistant."}
    # Summary inserted right after head; "Москва" preserved verbatim.
    assert new_conv[1]['role'] == 'system'
    assert new_conv[1]['content'].startswith("[prior_summary]:")
    # The fact "Москва" must survive the trim — accept any inflected form.
    assert "Москв" in new_conv[1]['content']


@pytest.mark.parametrize(
    "head_pairs, tail_pairs, expect_pair_in_kept",
    [
        # Few pairs before the tool call, many after: naive cut lands well
        # past the tool pair → pair stays in to_summarize (both absent from
        # new_conv).
        (1, 8, False),
        # Many pairs before the tool call, few after: naive cut lands well
        # before the tool pair → pair stays in to_keep (both present in
        # new_conv).
        (8, 1, True),
    ],
)
@pytest.mark.asyncio
async def test_cut_shifts_when_tool_call_at_boundary(head_pairs, tail_pairs, expect_pair_in_kept):
    helper = _make_helper()
    state_key = 7
    # Build a conversation where the assistant(tool_calls=[t1]) and the
    # tool reply must end up on the same side of the cut. Varying
    # head_pairs / tail_pairs around the tool pair exercises both
    # "pair in to_summarize" and "pair in to_keep" outcomes.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(head_pairs):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    # Pair that must stay glued: assistant(tool_calls) -> tool result.
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    })
    msgs.append({"role": "tool", "tool_call_id": "t1", "content": "result-of-lookup"})
    for i in range(tail_pairs):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"r{i}"})
    helper.conversations[state_key] = msgs

    helper._summarise_window = AsyncMock(return_value="summary text")

    ok = await helper._summarize_and_trim(
        state_key,
        chat_id=state_key,
        session_id=None,
        memory_user_id=state_key,
    )
    assert ok is True

    new_conv = helper.conversations[state_key]
    # Find the assistant-with-tool_calls and tool reply: they must either
    # both be inside the summary (i.e. absent from new_conv) or both be
    # present in the kept tail. Never split.
    assistant_present = any(
        m.get('role') == 'assistant' and m.get('tool_calls')
        for m in new_conv if isinstance(m, dict)
    )
    tool_present = any(
        m.get('role') == 'tool' and m.get('tool_call_id') == 't1'
        for m in new_conv if isinstance(m, dict)
    )
    assert assistant_present == tool_present, (
        "assistant(tool_calls) and tool result must stay on the same side of the cut"
    )
    # And the side itself must match what this layout was designed to
    # exercise — covers both True == True and False == False.
    assert assistant_present is expect_pair_in_kept


@pytest.mark.asyncio
async def test_system_prompt_not_truncated():
    helper = _make_helper()
    state_key = 99
    system_msg = {"role": "system", "content": "DO-NOT-LOSE-ME"}
    msgs = [system_msg]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    helper.conversations[state_key] = msgs

    helper._summarise_window = AsyncMock(return_value="brief summary")

    ok = await helper._summarize_and_trim(
        state_key,
        chat_id=state_key,
        session_id=None,
        memory_user_id=state_key,
    )
    assert ok is True
    assert helper.conversations[state_key][0] is system_msg
    assert helper.conversations[state_key][0]['content'] == "DO-NOT-LOSE-ME"


@pytest.mark.asyncio
async def test_summary_failure_falls_back_to_head_preserve_trim():
    """When ``_summarise_window`` raises (e.g. timeout), ``_summarize_and_trim``
    must return False so the caller can apply its head-preserve fallback.
    """
    helper = _make_helper()
    state_key = 1
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    helper.conversations[state_key] = list(msgs)

    import asyncio as _asyncio
    helper._summarise_window = AsyncMock(side_effect=_asyncio.TimeoutError())

    ok = await helper._summarize_and_trim(
        state_key,
        chat_id=state_key,
        session_id=None,
        memory_user_id=state_key,
    )
    assert ok is False
    # Conversation untouched by the failed summariser; caller decides on trim.
    assert helper.conversations[state_key] == msgs

    # Simulate the caller's head-preserve fallback exactly as written in
    # __common_get_chat_response / __common_get_chat_response_vision.
    conv = helper.conversations[state_key]
    head = [m for m in conv if isinstance(m, dict) and m.get('role') == 'system']
    tail = conv[-helper.config['max_history_size']:]
    helper.conversations[state_key] = head + [m for m in tail if m not in head]

    new_conv = helper.conversations[state_key]
    assert new_conv[0]['role'] == 'system'
    assert new_conv[0]['content'] == 'sys'
    assert len(new_conv) <= helper.config['max_history_size'] + 1


@pytest.mark.asyncio
async def test_throttle_skips_summary_within_min_messages():
    helper = _make_helper()
    state_key = 'throttle-key'
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    helper.conversations[state_key] = msgs

    # Pretend we summarised very recently — only 2 messages ago.
    helper._last_summary_at[state_key] = len(msgs) - 2
    helper._summarise_window = AsyncMock(return_value="should not be called")

    ok = await helper._summarize_and_trim(
        state_key,
        chat_id=0,
        session_id=None,
        memory_user_id=0,
    )
    assert ok is False
    helper._summarise_window.assert_not_awaited()


@pytest.mark.parametrize(
    "msgs, naive_cut, expected_zero",
    [
        # Dangling tool result at cut: walk forward to consume tool, OK.
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "f"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "r"},
                {"role": "user", "content": "next"},
            ],
            2,  # naive cut points at the tool message
            False,  # resolvable: shift past the tool result
        ),
        # Assistant with tool_calls last in to_summarize, no matching tool
        # within the window -> unresolvable.
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "tX", "function": {"name": "f"}}]},
                {"role": "user", "content": "next"},
            ],
            2,
            True,
        ),
        # Long contiguous run of tool messages -> exceeds bounded retry.
        (
            (
                [{"role": "user", "content": "q"}]
                + [{"role": "tool", "tool_call_id": f"t{i}", "content": "r"} for i in range(15)]
                + [{"role": "user", "content": "end"}]
            ),
            1,  # cut into the run of tools
            True,
        ),
    ],
)
def test_safe_cut_returns_zero_on_unresolvable_tool_chain(msgs, naive_cut, expected_zero):
    cut = OpenAIHelper._safe_cut_index(list(msgs), naive_cut)
    if expected_zero:
        assert cut == 0
    else:
        assert cut > naive_cut
        # The adjusted cut must not split an assistant(tool_calls) from its
        # tool reply: check that for every assistant with tool_calls in the
        # to_summarize half, all its ids are also in that half.
        to_sum = msgs[:cut]
        for i, m in enumerate(to_sum):
            if m.get('role') == 'assistant' and m.get('tool_calls'):
                ids = {tc['id'] for tc in m['tool_calls'] if isinstance(tc, dict) and tc.get('id')}
                closed = {
                    n.get('tool_call_id')
                    for n in to_sum[i + 1:]
                    if isinstance(n, dict) and n.get('role') == 'tool'
                }
                assert ids.issubset(closed)
