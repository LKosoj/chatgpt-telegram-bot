"""Direct unit tests for the public session API added by T19.

`tests/test_group_session_flow.py` and `tests/test_per_conversation_serialization.py`
only assert that `bot/telegram_bot.py` calls these names on a fake helper. These tests
run the real `OpenAIHelper` implementations
(`bot/openai_helper.py:2967-3050`) instead:

* `history_snapshot` — cold cache, warm cache, and that the caller cannot corrupt the
  cache through the returned list
* `load_session` — image payloads stripped, loaded session id recorded, stripped list
  returned (this is what the session switch/delete branches rely on)
* `replace_system_message` — insert vs replace, session id taken from the loaded-session
  map, temperature default from config
* `evict` — clears every per-chat dict and is idempotent
* `chat_state_scope` — overrides the effective key and restores it, including on error
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


def _make_helper(**config_overrides) -> OpenAIHelper:
    helper = object.__new__(OpenAIHelper)
    helper.config = {'temperature': 0.5, **config_overrides}
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.last_updated = {}
    helper.last_image_file_ids = {}
    helper._chat_request_extra_tokens = {}
    helper._chat_request_models = {}
    helper._chat_request_usage_split = {}
    helper._per_chat_locks = {}
    return helper


def _vision_message() -> dict:
    return {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'что на фото?'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AAAA'}},
        ],
    }


# --- history_snapshot -------------------------------------------------------

def test_history_snapshot_returns_none_on_cold_cache():
    helper = _make_helper()
    assert helper.history_snapshot(42) is None


def test_history_snapshot_returns_cached_history():
    helper = _make_helper()
    helper.conversations[42] = [{'role': 'system', 'content': 'sys'}]
    assert helper.history_snapshot(42) == [{'role': 'system', 'content': 'sys'}]


def test_history_snapshot_result_cannot_corrupt_the_cache():
    helper = _make_helper()
    helper.conversations[42] = [{'role': 'system', 'content': 'sys'}]

    snapshot = helper.history_snapshot(42)
    snapshot.append({'role': 'user', 'content': 'injected'})
    del snapshot[0]

    assert helper.conversations[42] == [{'role': 'system', 'content': 'sys'}]


def test_history_snapshot_distinguishes_cold_cache_from_empty_history():
    helper = _make_helper()
    helper.conversations[42] = []
    assert helper.history_snapshot(42) == []
    assert helper.history_snapshot(43) is None


# --- load_session -----------------------------------------------------------

def test_load_session_strips_image_payloads_and_records_session():
    helper = _make_helper()

    returned = helper.load_session(42, 'sess-1', [
        {'role': 'system', 'content': 'sys'},
        _vision_message(),
    ])

    assert helper.loaded_conversation_sessions[42] == 'sess-1'
    assert helper.conversations[42] is returned
    assert returned[1]['content'] == 'что на фото?\n[image]'


def test_load_session_does_not_mutate_the_caller_list():
    helper = _make_helper()
    original = _vision_message()
    messages = [original]

    helper.load_session(42, 'sess-1', messages)

    assert messages == [original]
    assert isinstance(original['content'], list)


def test_load_session_overwrites_previous_history():
    helper = _make_helper()
    helper.conversations[42] = [{'role': 'user', 'content': 'old'}]
    helper.loaded_conversation_sessions[42] = 'sess-0'

    helper.load_session(42, 'sess-1', [{'role': 'user', 'content': 'new'}])

    assert helper.conversations[42] == [{'role': 'user', 'content': 'new'}]
    assert helper.loaded_conversation_sessions[42] == 'sess-1'


def test_load_session_accepts_none_session_id():
    helper = _make_helper()
    helper.load_session(42, None, [{'role': 'user', 'content': 'hi'}])
    assert helper.loaded_conversation_sessions[42] is None


# --- replace_system_message -------------------------------------------------

@pytest.mark.asyncio
async def test_replace_system_message_inserts_when_absent():
    helper = _make_helper()
    helper._save_conversation_context = AsyncMock(return_value='sess-1')
    helper.load_session(42, 'sess-1', [{'role': 'user', 'content': 'hi'}])

    result = await helper.replace_system_message(42, 'new prompt', mode_key='assistant')

    assert result == 'sess-1'
    assert helper.conversations[42][0] == {
        'role': 'system', 'content': 'new prompt', 'mode_key': 'assistant',
    }
    assert helper.conversations[42][1] == {'role': 'user', 'content': 'hi'}
    args, _kwargs = helper._save_conversation_context.call_args
    assert args[0] == 42
    assert args[1] == {'messages': helper.conversations[42]}
    assert args[5] == 'sess-1'


@pytest.mark.asyncio
async def test_replace_system_message_replaces_existing_leading_system():
    helper = _make_helper()
    helper._save_conversation_context = AsyncMock(return_value='sess-1')
    helper.load_session(42, 'sess-1', [
        {'role': 'system', 'content': 'old prompt', 'mode_key': 'old'},
        {'role': 'user', 'content': 'hi'},
    ])

    await helper.replace_system_message(42, 'new prompt', mode_key='new')

    assert len(helper.conversations[42]) == 2
    assert helper.conversations[42][0] == {
        'role': 'system', 'content': 'new prompt', 'mode_key': 'new',
    }


@pytest.mark.asyncio
async def test_replace_system_message_omits_mode_key_when_not_given():
    helper = _make_helper()
    helper._save_conversation_context = AsyncMock(return_value='sess-1')
    helper.load_session(42, 'sess-1', [])

    await helper.replace_system_message(42, 'plain prompt')

    assert helper.conversations[42][0] == {'role': 'system', 'content': 'plain prompt'}


@pytest.mark.asyncio
async def test_replace_system_message_falls_back_to_config_temperature():
    helper = _make_helper(temperature=0.7)
    helper._save_conversation_context = AsyncMock(return_value='sess-1')
    helper.load_session(42, 'sess-1', [])

    await helper.replace_system_message(42, 'prompt')
    args, _kwargs = helper._save_conversation_context.call_args
    assert args[3] == 0.7

    await helper.replace_system_message(42, 'prompt', temperature=0.1)
    args, _kwargs = helper._save_conversation_context.call_args
    assert args[3] == 0.1


@pytest.mark.asyncio
async def test_replace_system_message_uses_loaded_session_id():
    helper = _make_helper()
    helper._save_conversation_context = AsyncMock(return_value=None)
    helper.conversations[42] = [{'role': 'user', 'content': 'hi'}]

    # load_session was never called for this chat -> no session id is known
    assert await helper.replace_system_message(42, 'prompt') is None
    args, _kwargs = helper._save_conversation_context.call_args
    assert args[5] is None


# --- evict ------------------------------------------------------------------

def test_evict_clears_every_per_chat_dict():
    helper = _make_helper()
    helper.load_session(42, 'sess-1', [{'role': 'user', 'content': 'hi'}])
    helper.last_updated[42] = 'now'
    helper.last_image_file_ids[42] = ['file-1']
    helper._chat_request_extra_tokens[42] = 10
    helper._chat_request_models[42] = 'model'
    helper._chat_request_usage_split[42] = (1, 2)
    helper._per_chat_locks[42] = object()

    helper.evict(42)

    for name in ('conversations', 'loaded_conversation_sessions', 'last_updated',
                 'last_image_file_ids', '_chat_request_extra_tokens',
                 '_chat_request_models', '_chat_request_usage_split', '_per_chat_locks'):
        assert getattr(helper, name) == {}, name
    assert helper.history_snapshot(42) is None


def test_evict_is_idempotent_on_unknown_chat():
    helper = _make_helper()
    helper.evict(999)
    helper.evict(999)
    assert helper.conversations == {}


def test_evict_leaves_other_chats_alone():
    helper = _make_helper()
    helper.load_session(42, 'sess-1', [{'role': 'user', 'content': 'a'}])
    helper.load_session(43, 'sess-2', [{'role': 'user', 'content': 'b'}])

    helper.evict(42)

    assert helper.history_snapshot(42) is None
    assert helper.history_snapshot(43) == [{'role': 'user', 'content': 'b'}]


# --- chat_state_scope -------------------------------------------------------

def test_chat_state_scope_overrides_the_effective_key():
    helper = _make_helper()
    helper.conversations['override'] = [{'role': 'user', 'content': 'scoped'}]

    assert helper.history_snapshot(42) is None
    with helper.chat_state_scope('override'):
        assert helper.history_snapshot(42) == [{'role': 'user', 'content': 'scoped'}]
        helper.load_session(42, 'sess-x', [{'role': 'user', 'content': 'written'}])

    assert helper.history_snapshot(42) is None
    assert helper.conversations['override'] == [{'role': 'user', 'content': 'written'}]


def test_chat_state_scope_restores_the_key_on_exception():
    helper = _make_helper()
    helper.conversations['override'] = [{'role': 'user', 'content': 'scoped'}]

    with pytest.raises(RuntimeError, match='boom'):
        with helper.chat_state_scope('override'):
            raise RuntimeError('boom')

    assert helper.history_snapshot(42) is None


def test_chat_state_scope_nests():
    helper = _make_helper()
    helper.conversations['outer'] = [{'role': 'user', 'content': 'outer'}]
    helper.conversations['inner'] = [{'role': 'user', 'content': 'inner'}]

    with helper.chat_state_scope('outer'):
        with helper.chat_state_scope('inner'):
            assert helper.history_snapshot(0) == [{'role': 'user', 'content': 'inner'}]
        assert helper.history_snapshot(0) == [{'role': 'user', 'content': 'outer'}]
