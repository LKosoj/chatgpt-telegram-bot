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

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper
from bot.session_logger import clear_trace, set_trace


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


class CaptureSessionLogger:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(dict(event))


def _summary_response(content: str = "compact summary"):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
                finish_reason=None,
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )


@pytest.mark.asyncio
async def test_summarise_window_calls_chat_response_wrapper():
    helper = _make_helper()
    calls = []

    async def fake_completion(*, kind, **kwargs):
        calls.append((kind, kwargs))
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="compact summary")
                )
            ]
        )

    helper._create_chat_response_completion = fake_completion

    summary = await helper._summarise_window([
        {"role": "user", "content": "old user message"},
    ])

    assert summary == "compact summary"
    assert calls[0][0] == "summary"
    assert calls[0][1]["model"] == "cheap-model"
    assert calls[0][1]["stream"] is False


@pytest.mark.asyncio
async def test_summarise_window_uses_provider_wrapper_by_default():
    helper = _make_helper()
    helper.session_logger = CaptureSessionLogger()
    calls = []

    async def fake_sdk_create(*, kind, **kwargs):
        calls.append((kind, kwargs))
        return _summary_response()

    helper._create_chat_completion_with_rate_limit_retry = fake_sdk_create

    token = set_trace(1, "summary-session", "summary-turn")
    try:
        summary = await helper._summarise_window([
            {"role": "user", "content": "old user message"},
        ])
    finally:
        clear_trace(token)

    assert summary == "compact summary"
    assert calls[0][0] == "summary"
    provider_events = [
        event for event in helper.session_logger.events
        if event["type"] == "ai_provider_response"
    ]
    assert [event["kind"] for event in provider_events] == ["summary"]


@pytest.mark.asyncio
async def test_summarise_window_can_roll_back_to_legacy_timed_create():
    config = _base_config()
    config["chat_run_variant_b_enabled"] = False
    helper = _make_helper(config)
    helper.session_logger = CaptureSessionLogger()

    async def fake_sdk_create(*, kind, **_kwargs):
        assert kind == "summary"
        return _summary_response("legacy summary")

    helper._create_chat_completion_with_rate_limit_retry = fake_sdk_create

    token = set_trace(1, "summary-session", "summary-turn")
    try:
        summary = await helper._summarise_window([
            {"role": "user", "content": "old user message"},
        ])
    finally:
        clear_trace(token)

    assert summary == "legacy summary"
    assert not any(
        event["type"] == "ai_provider_response"
        for event in helper.session_logger.events
    )
    assert any(
        event["type"] == "llm_call" and event["kind"] == "summary"
        for event in helper.session_logger.events
    )


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
    must return False so the caller can apply ``_fallback_trim_with_summary``
    — the deterministic (no LLM call) head-preserve trim.
    """
    helper = _make_helper()
    state_key = 1
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    # A tool_calls/tool pair near the naive cut point, so the fallback's
    # tool-pairing safety net is also exercised.
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    })
    msgs.append({"role": "tool", "tool_call_id": "t1", "content": "result-of-lookup"})
    for i in range(20, 23):
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

    # The caller's real fallback, as called from
    # __common_get_chat_response / __common_get_chat_response_vision.
    helper._fallback_trim_with_summary(state_key)

    new_conv = helper.conversations[state_key]
    assert new_conv[0]['role'] == 'system'
    assert new_conv[0]['content'] == 'sys'
    # head (system) + one deterministic-summary system message + kept tail.
    assert len(new_conv) <= helper.config['max_history_size'] + 2

    # Discarded content must survive verbatim inside the deterministic
    # summary, not be dropped outright.
    summary_msgs = [
        m for m in new_conv
        if isinstance(m.get('content'), str) and m['content'].startswith('[prior_summary]:')
    ]
    assert len(summary_msgs) == 1
    assert 'u0' in summary_msgs[0]['content']

    # tool_calls/tool pair must not be split across summary/tail.
    assistant_present = any(
        m.get('role') == 'assistant' and m.get('tool_calls') for m in new_conv
    )
    tool_present = any(
        m.get('role') == 'tool' and m.get('tool_call_id') == 't1' for m in new_conv
    )
    assert assistant_present == tool_present


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


def test_deterministic_summary_text_noop_when_short():
    """Below ``max_chars``, the rendering is returned verbatim — no notice,
    no truncation.
    """
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    rendered = OpenAIHelper._serialize_messages_for_summary(msgs)
    result = OpenAIHelper._deterministic_summary_text(msgs, max_chars=4000, tail_chars=500)
    assert result == rendered
    assert "truncated" not in result


def test_deterministic_summary_text_caps_with_correct_length_marker():
    """Above ``max_chars``, the result is capped and the truncation notice
    reports the exact length of the full (untruncated) rendering.
    """
    msgs = [{"role": "user", "content": f"message number {i} " * 5} for i in range(20)]
    rendered = OpenAIHelper._serialize_messages_for_summary(msgs)
    assert len(rendered) > 200

    result = OpenAIHelper._deterministic_summary_text(msgs, max_chars=100, tail_chars=20)

    assert len(result) <= 100
    notice = f"…[truncated — {len(rendered)} chars]…"
    assert notice in result
    assert result.endswith(rendered[-20:])


@pytest.mark.parametrize(
    "max_chars, tail_chars",
    [
        # notice + tail alone used to overflow max_chars because tail_chars
        # was clamped without accounting for len(notice).
        (300, 500),
        (50, 200),
        # max_chars smaller than the notice text itself: the notice must be
        # hard-truncated rather than the contract being abandoned.
        (10, 5),
    ],
)
def test_deterministic_summary_text_never_exceeds_max_chars(max_chars, tail_chars):
    """``len(result) <= max_chars`` must hold unconditionally, including
    when ``max_chars`` is too small to fit ``notice + tail`` (regression
    test: ``tail_chars`` used to be clamped by ``max_chars`` alone, ignoring
    ``len(notice)``, so the returned text could exceed ``max_chars``).
    """
    msgs = [{"role": "user", "content": f"message number {i} " * 5} for i in range(20)]
    result = OpenAIHelper._deterministic_summary_text(msgs, max_chars=max_chars, tail_chars=tail_chars)
    assert len(result) <= max_chars


@pytest.mark.parametrize(
    "head_pairs, tail_pairs, expect_pair_in_tail",
    [
        # Naive cut (max_history_size=6 kept) lands well past the tool pair
        # -> the pair falls inside the discarded/summarised half.
        (1, 8, False),
        # Naive cut lands well before the tool pair -> the pair is kept in
        # the tail alongside the rest of the recent messages.
        (8, 1, True),
    ],
)
def test_fallback_trim_keeps_tool_call_pair_together(head_pairs, tail_pairs, expect_pair_in_tail):
    helper = _make_helper()
    helper.config['max_history_size'] = 6
    state_key = 7
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

    helper._fallback_trim_with_summary(state_key)

    new_conv = helper.conversations[state_key]
    assistant_present = any(
        m.get('role') == 'assistant' and m.get('tool_calls')
        for m in new_conv if isinstance(m, dict)
    )
    tool_present = any(
        m.get('role') == 'tool' and m.get('tool_call_id') == 't1'
        for m in new_conv if isinstance(m, dict)
    )
    # Never split: either both survive in the tail, or both are gone
    # (rendered into the deterministic summary instead).
    assert assistant_present == tool_present
    assert assistant_present is expect_pair_in_tail


def test_fallback_trim_calls_repair_tool_call_history():
    """The fallback must repair the history afterwards — the cut can land
    inside an assistant/tool_calls -> tool pair.
    """
    helper = _make_helper()
    state_key = 3
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    helper.conversations[state_key] = msgs
    helper._repair_tool_call_history = MagicMock()

    helper._fallback_trim_with_summary(state_key)

    helper._repair_tool_call_history.assert_called_once_with(state_key)


def test_fallback_trim_handles_unresolvable_cut_without_crash():
    """When ``_safe_cut_index`` can't resolve the cut (returns 0), the
    fallback must not raise — it degrades to keeping everything rather than
    corrupting the tool-call chain.
    """
    helper = _make_helper()
    helper.config['max_history_size'] = 1
    state_key = 9
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tX", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "user", "content": "next"},
    ]
    helper.conversations[state_key] = list(msgs)
    # naive_cut = len(conv) - max_history_size = 4 - 1 = 3; the assistant
    # tool_calls message just before it has no matching tool reply anywhere
    # -> _safe_cut_index bails out to 0 (unresolvable).
    assert OpenAIHelper._safe_cut_index(msgs, 3) == 0

    helper._fallback_trim_with_summary(state_key)  # must not raise

    new_conv = helper.conversations[state_key]
    assert new_conv[0]['role'] == 'system'


def test_fallback_trim_stabilizes_over_repeated_failed_summaries():
    """Regression test for unbounded growth when the LLM summariser fails on
    every turn (e.g. sustained outage/throttling) and the fallback runs
    every time.

    Before the fix, ``naive_cut`` was computed from the *full* conversation
    length including the leading system block, and a brand-new
    ``[prior_summary]`` message was appended on every call instead of being
    merged into the existing one. Both the compressible window and the
    ``[prior_summary]`` count then grew roughly linearly with turn count,
    defeating ``max_history_size`` — the exact overflow the feature exists
    to prevent. 30 consecutive failed-summary turns must instead converge to
    a stable length with a single merged summary message.
    """
    helper = _make_helper()
    helper.config['max_history_size'] = 15
    state_key = 'growth-regression'
    helper.conversations[state_key] = [{"role": "system", "content": "sys prompt"}]

    lengths = []
    system_counts = []
    for turn in range(30):
        conv = helper.conversations[state_key]
        conv.append({"role": "user", "content": f"user msg {turn}"})
        conv.append({"role": "assistant", "content": f"assistant reply {turn}"})
        # Simulates _summarize_and_trim returning False on every turn (LLM
        # summariser down/throttled) — only the deterministic fallback runs.
        helper._fallback_trim_with_summary(state_key)
        conv = helper.conversations[state_key]
        lengths.append(len(conv))
        system_counts.append(sum(1 for m in conv if m.get('role') == 'system'))

    conv = helper.conversations[state_key]
    non_system = [m for m in conv if m.get('role') != 'system']
    prior_summaries = [
        m for m in conv
        if isinstance(m.get('content'), str) and m['content'].startswith('[prior_summary]:')
    ]

    # The old bug measured len=39, system=24 by turn 29 (linear growth).
    # Non-system messages must never exceed max_history_size...
    assert len(non_system) <= helper.config['max_history_size']
    # ...and exactly one merged [prior_summary] message, not one per call.
    assert len(prior_summaries) == 1
    assert len(conv) <= helper.config['max_history_size'] + 2

    # Length/system-count must have stopped growing well before turn 29:
    # the last several turns must be identical (steady state), not still
    # climbing turn over turn.
    assert lengths[-10:] == [lengths[-1]] * 10
    assert system_counts[-10:] == [system_counts[-1]] * 10


def test_fallback_trim_deterministic_summary_disabled_by_zero_max_chars():
    """``SUMMARY_DETERMINISTIC_MAX_CHARS=0`` must disable the deterministic
    summary entirely (discarded content dropped silently, matching the
    pre-feature bare-slice trim) instead of being silently replaced by the
    4000-char default via a falsy-``0`` config read.
    """
    helper = _make_helper()
    helper.config['max_history_size'] = 3
    helper.config['summary_deterministic_max_chars'] = 0
    state_key = 'zero-max-chars'
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    helper.conversations[state_key] = msgs

    helper._fallback_trim_with_summary(state_key)

    new_conv = helper.conversations[state_key]
    prior_summaries = [
        m for m in new_conv
        if isinstance(m.get('content'), str) and m['content'].startswith('[prior_summary]:')
    ]
    assert prior_summaries == []
    assert new_conv[0] == {"role": "system", "content": "sys"}
    non_system = [m for m in new_conv if m.get('role') != 'system']
    assert len(non_system) <= helper.config['max_history_size']
