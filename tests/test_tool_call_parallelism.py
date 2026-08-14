"""Two-level bound on parallel tool execution.

``TOOL_CALL_PARALLELISM`` bounds one model batch (i.e. one chat);
``TOOL_CALL_GLOBAL_PARALLELISM`` caps the whole process. The point of the
split is that a single busy chat must not consume every slot in the bot.
"""

import asyncio
from types import SimpleNamespace

import pytest

from bot.openai_tool_handler import (
    _execute_prepared_tool_calls,
    _tool_call_batch_semaphore,
    _tool_call_global_semaphore,
)


class ConcurrencyProbe:
    """Plugin manager double that records peak overlap of tool calls."""

    def __init__(self, hold_seconds=0.02):
        self.hold_seconds = hold_seconds
        self.active = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def call_function(self, name, helper, arguments, request_context=None):
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.hold_seconds)
        finally:
            async with self._lock:
                self.active -= 1
        return "{}"


def _prepared(count):
    return [
        (f"probe.tool_{index}", f"probe_tool_{index}", "{}", "{}", f"call_{index}")
        for index in range(count)
    ]


def _helper(probe):
    return SimpleNamespace(plugin_manager=probe, session_logger=None)


async def _run_batch(helper, probe, count, global_semaphore):
    await _execute_prepared_tool_calls(
        helper,
        _prepared(count),
        None,
        _tool_call_batch_semaphore(),
        global_semaphore,
    )
    return probe.peak


@pytest.mark.asyncio
async def test_batch_limit_bounds_a_single_chat(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_PARALLELISM", "3")
    monkeypatch.setenv("TOOL_CALL_GLOBAL_PARALLELISM", "50")
    probe = ConcurrencyProbe()
    helper = _helper(probe)

    await _run_batch(helper, probe, 9, _tool_call_global_semaphore(helper))

    assert probe.peak == 3


@pytest.mark.asyncio
async def test_one_chat_no_longer_starves_the_others(monkeypatch):
    """Regression: the bound used to be a single process-wide semaphore, so two
    chats together could never exceed TOOL_CALL_PARALLELISM."""
    monkeypatch.setenv("TOOL_CALL_PARALLELISM", "2")
    monkeypatch.setenv("TOOL_CALL_GLOBAL_PARALLELISM", "50")
    probe = ConcurrencyProbe()
    helper = _helper(probe)
    global_semaphore = _tool_call_global_semaphore(helper)

    await asyncio.gather(
        _run_batch(helper, probe, 4, global_semaphore),
        _run_batch(helper, probe, 4, global_semaphore),
    )

    assert probe.peak == 4


@pytest.mark.asyncio
async def test_global_ceiling_caps_the_whole_process(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_PARALLELISM", "4")
    monkeypatch.setenv("TOOL_CALL_GLOBAL_PARALLELISM", "5")
    probe = ConcurrencyProbe()
    helper = _helper(probe)
    global_semaphore = _tool_call_global_semaphore(helper)

    await asyncio.gather(
        *(_run_batch(helper, probe, 4, global_semaphore) for _ in range(3))
    )

    assert probe.peak == 5


@pytest.mark.asyncio
async def test_global_semaphore_is_shared_and_loop_bound(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_GLOBAL_PARALLELISM", "7")
    helper = _helper(ConcurrencyProbe())

    first = _tool_call_global_semaphore(helper)

    assert _tool_call_global_semaphore(helper) is first
    assert helper._tool_call_global_semaphore_bundle[0] is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_invalid_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_PARALLELISM", "not-a-number")
    monkeypatch.setenv("TOOL_CALL_GLOBAL_PARALLELISM", "0")
    probe = ConcurrencyProbe()
    helper = _helper(probe)

    # Batch falls back to 5; the global gate clamps 0 up to 1 and becomes the
    # binding constraint, so a 6-call batch must never overlap.
    await _run_batch(helper, probe, 6, _tool_call_global_semaphore(helper))

    assert probe.peak == 1
