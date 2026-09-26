"""Unit tests for bot.telegram_stream in isolation — no ChatGPTTelegramBot,
no Telegram API, only the injected edit/send callbacks.

See docs/remediation_2026-09-04/T12-telegram-stream.md.
"""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, RetryAfter, TimedOut

from bot import telegram_stream
from bot.telegram_stream import stream_to_telegram


async def _achunks(pairs):
    for item in pairs:
        yield item


def _msg(message_id):
    return SimpleNamespace(message_id=message_id)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(telegram_stream.asyncio, "sleep", AsyncMock())


@pytest.mark.asyncio
async def test_single_message_grows_then_finalizes_via_edit():
    edit = AsyncMock()
    send = AsyncMock(side_effect=[_msg(1)])

    outcome = await stream_to_telegram(
        _achunks([("Hello", "not_finished"), ("Hello world", "5")]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
    )

    send.assert_awaited_once_with("Hello", False)
    edit.assert_awaited_once_with(1, "Hello world", True)
    assert outcome.final_text == "Hello world"
    assert outcome.total_tokens == 5
    assert outcome.delivered is True
    assert outcome.message_ids == [1]


@pytest.mark.asyncio
async def test_overflow_publishes_closed_chunk_and_continues_in_new_message():
    edit = AsyncMock()
    send = AsyncMock(side_effect=[_msg(1), _msg(2)])

    outcome = await stream_to_telegram(
        _achunks([
            ("aaaa", "not_finished"),
            ("aaaa\nbbbb\ncccc", "not_finished"),
            ("aaaa\nbbbb\ncccc\ndddd", "9"),
        ]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        chunk_size=10,
    )

    # первое сообщение — обычный send
    assert send.await_args_list[0].args == ("aaaa", False)
    # переполнение: закрытый чанк "aaaa\nbbbb" уходит в edit первого сообщения,
    # а новый хвостовой чанк "cccc" — новым send
    assert edit.await_args_list[0].args == (1, "aaaa\nbbbb", True)
    assert send.await_args_list[1].args == ("cccc", False)
    # финальный чанк донёсся как edit хвостового сообщения
    assert edit.await_args_list[-1].args == (2, "cccc\ndddd", True)

    assert outcome.final_text == "aaaa\nbbbb\ncccc\ndddd"
    assert outcome.total_tokens == 9
    assert outcome.delivered is True
    assert outcome.message_ids == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [RetryAfter(retry_after=1), TimedOut(), BadRequest("boom"), RuntimeError("boom")],
)
async def test_intermediate_edit_failure_backs_off_and_final_tokens_stay_correct(exc):
    edit = AsyncMock(side_effect=[exc, None])
    send = AsyncMock(side_effect=[_msg(1)])

    outcome = await stream_to_telegram(
        _achunks([
            ("Hi", "not_finished"),
            ("Hi there", "not_finished"),
            ("Hi there final", "5"),
        ]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 0,
    )

    assert edit.await_count == 2
    assert edit.await_args_list[-1].args == (1, "Hi there final", True)
    assert outcome.total_tokens == 5
    assert outcome.delivered is True


@pytest.mark.asyncio
async def test_final_chunk_edit_failure_is_retried_and_recovers_without_send():
    edit = AsyncMock(side_effect=[TimedOut(), RuntimeError("still down"), None])
    send = AsyncMock(side_effect=[_msg(1)])

    outcome = await stream_to_telegram(
        _achunks([("Ans", "not_finished"), ("Answer", "7")]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        max_final_attempts=3,
    )

    # 1 неудачная попытка внутри цикла + 1 неудачная + 1 успешная в гарантии
    assert edit.await_count == 3
    send.assert_awaited_once_with("Ans", False)
    assert outcome.final_text == "Answer"
    assert outcome.total_tokens == 7
    assert outcome.delivered is True


@pytest.mark.asyncio
async def test_final_chunk_all_edit_attempts_fail_falls_back_to_send():
    edit = AsyncMock(side_effect=[RuntimeError("e1"), RuntimeError("e2"), RuntimeError("e3"), RuntimeError("e4")])
    send = AsyncMock(side_effect=[_msg(1), _msg(2)])

    outcome = await stream_to_telegram(
        _achunks([("Ans", "not_finished"), ("Answer", "7")]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        max_final_attempts=3,
    )

    # 1 (в цикле) + 3 (гарантия) = 4 неудачных edit
    assert edit.await_count == 4
    assert send.await_count == 2
    assert send.await_args_list[-1].args == ("Answer", True)
    assert outcome.total_tokens == 7
    assert outcome.delivered is True
    assert outcome.message_ids == [1, 2]


@pytest.mark.asyncio
async def test_final_chunk_delivery_fails_entirely_when_no_send_fallback_available():
    edit = AsyncMock(side_effect=[
        TimedOut(), RuntimeError("g1"), RuntimeError("g2"), RuntimeError("g3"),
    ])

    outcome = await stream_to_telegram(
        _achunks([("Answer", "7")]),
        edit=edit,
        send=None,
        cutoff_for=lambda content: 100,
        max_final_attempts=3,
    )

    assert edit.await_count == 4
    assert outcome.delivered is False
    assert outcome.total_tokens == 7
    assert outcome.final_text == "Answer"
    assert outcome.message_ids == []


@pytest.mark.asyncio
async def test_direct_result_stops_loop_and_reports_total_tokens():
    edit = AsyncMock()
    send = AsyncMock()
    on_direct_result = AsyncMock()
    direct_payload = {"direct_result": {"kind": "text", "value": "done"}}

    outcome = await stream_to_telegram(
        _achunks([
            ("draft", "not_finished"),
            (direct_payload, "9"),
        ]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        on_direct_result=on_direct_result,
    )

    on_direct_result.assert_awaited_once_with(direct_payload, 9)
    edit.assert_not_awaited()
    assert outcome.direct_result is direct_payload
    assert outcome.total_tokens == 9
    assert outcome.delivered is True


@pytest.mark.asyncio
async def test_final_overflow_guarantee_delivers_tail_not_full_text():
    # Финальный чанк впервые пересекает chunk_size: хвост уходит в новое
    # сообщение, а гарантия доставки должна редактировать именно хвост,
    # а не переписывать его полным ответом (дублируя уже опубликованное начало).
    edit = AsyncMock()
    send = AsyncMock(side_effect=[_msg(1), _msg(2)])

    outcome = await stream_to_telegram(
        _achunks([("aaaa", "not_finished"), ("aaaa\nbbbb", "5")]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        chunk_size=6,
    )

    assert send.await_args_list == [(("aaaa", False),), (("bbbb", False),)]
    assert edit.await_args_list[0].args == (1, "aaaa", True)
    assert edit.await_args_list[-1].args == (2, "bbbb", True)
    assert all("aaaa\nbbbb" not in call.args[1] for call in edit.await_args_list)
    assert outcome.final_text == "aaaa\nbbbb"
    assert outcome.total_tokens == 5
    assert outcome.delivered is True
    assert outcome.message_ids == [1, 2]


@pytest.mark.asyncio
async def test_final_overflow_tail_send_failure_is_recovered_by_guarantee():
    edit = AsyncMock()
    send = AsyncMock(side_effect=[_msg(1), RuntimeError("boom"), _msg(2)])

    outcome = await stream_to_telegram(
        _achunks([("aaaa", "not_finished"), ("aaaa\nbbbb", "5")]),
        edit=edit,
        send=send,
        cutoff_for=lambda content: 100,
        chunk_size=6,
    )

    # Хвост не открылся — гарантия доставки открывает его заново, а не
    # перезаписывает закрытый первый чанк.
    assert send.await_args_list[-1].args == ("bbbb", True)
    assert all(call.args[0] != 1 or call.args[1] == "aaaa" for call in edit.await_args_list)
    assert outcome.delivered is True
    assert outcome.message_ids == [1, 2]


@pytest.mark.asyncio
async def test_inline_intermediate_chunks_are_throttled_by_cutoff():
    edit = AsyncMock()

    outcome = await stream_to_telegram(
        _achunks([
            ("a", "not_finished"),
            ("ab", "not_finished"),
            ("abc", "not_finished"),
            ("abcd", "4"),
        ]),
        edit=edit,
        send=None,
        cutoff_for=lambda content: 100,
    )

    # Первый чанк открывает плейсхолдер без throttling, промежуточные
    # пропускаются по cutoff, финальный доставляется всегда.
    assert edit.await_args_list == [((None, "a", False),), ((None, "abcd", True),)]
    assert outcome.delivered is True
    assert outcome.total_tokens == 4


@pytest.mark.asyncio
async def test_inline_failed_first_edit_is_retried_on_next_chunk_unthrottled():
    edit = AsyncMock(side_effect=[RuntimeError("e1"), None, None])

    await stream_to_telegram(
        _achunks([("a", "not_finished"), ("ab", "not_finished"), ("abc", "3")]),
        edit=edit,
        send=None,
        cutoff_for=lambda content: 100,
    )

    assert edit.await_args_list == [
        ((None, "a", False),), ((None, "ab", False),), ((None, "abc", True),),
    ]


@pytest.mark.asyncio
async def test_inline_does_not_split_long_content_into_chunks():
    edit = AsyncMock()

    outcome = await stream_to_telegram(
        _achunks([("aaaa\nbbbb", "5")]),
        edit=edit,
        send=None,
        cutoff_for=lambda content: 100,
        chunk_size=6,
    )

    edit.assert_awaited_once_with(None, "aaaa\nbbbb", True)
    assert outcome.delivered is True


def test_retry_after_seconds_passes_through_int():
    assert telegram_stream.retry_after_seconds(SimpleNamespace(retry_after=3)) == 3.0


def test_retry_after_seconds_converts_timedelta():
    exc = SimpleNamespace(retry_after=timedelta(seconds=2.5))
    assert telegram_stream.retry_after_seconds(exc) == 2.5


def test_retry_after_seconds_converts_real_retry_after_timedelta(monkeypatch):
    monkeypatch.setenv("PTB_TIMEDELTA", "true")
    exc = RetryAfter(retry_after=2)
    assert telegram_stream.retry_after_seconds(exc) == 2.0
