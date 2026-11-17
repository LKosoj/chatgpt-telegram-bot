import pytest

from tests.test_openai_helper_tool_calls import (
    DummyClient, DummyPluginManager, FakeClosableAsyncStream, FakeStreamItem,
    FakeStreamToolCallDelta, _fake_stream, _make_helper,
)
from tests.test_stream_usage import FakeUsageChunk


@pytest.mark.parametrize('prefix', [None, 'thinking '])
async def test_stream_text_then_tool_runs_once_and_keeps_incremental_output(prefix):
    pm = DummyPluginManager({'p.do': 'tool ok'}, specs=[{'type': 'function', 'function': {'name': 'p.do'}}])
    source = FakeClosableAsyncStream([
        FakeStreamItem(content=prefix),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{"x"')]),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, arguments=': 1}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    helper = _make_helper(pm, client=DummyClient([source, _fake_stream(['done'])]))
    stream = helper.get_chat_response_stream(1, 'hello', user_id=1)
    if prefix:
        assert await anext(stream) == (prefix, 'not_finished')
        assert pm.calls == []
    output = [item async for item in stream]
    assert output[-1][0] == (prefix or '') + 'done'
    assert len(pm.calls) == 1
    assert source.closed
    assert helper._chat_request_usage_split[helper._chat_state_key(1)] is None


async def test_usage_only_chunk_survives_service_prefix_with_tools_enabled():
    helper = _make_helper(DummyPluginManager({}), client=DummyClient([
        FakeClosableAsyncStream([FakeStreamItem(), FakeStreamItem(content='hello'), FakeUsageChunk(40, 7)])
    ]))
    helper.config['stream_include_usage'] = True
    output = [item async for item in helper.get_chat_response_stream(1, 'hello', user_id=1)]
    assert output[-1] == ('hello', '47')
    assert helper._chat_request_usage_split[helper._chat_state_key(1)] == (40, 7)


async def test_late_direct_result_reaches_chat_consumer():
    result = {'direct_result': {'kind': 'text', 'format': 'markdown', 'value': 'ok'}}
    pm = DummyPluginManager({'p.do': result}, specs=[{'type': 'function', 'function': {'name': 'p.do'}}])
    source = FakeClosableAsyncStream([
        FakeStreamItem(content='working '),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    helper = _make_helper(pm, client=DummyClient([source]))
    lock = await helper._chat_lock(1)
    stream = helper.get_chat_response_stream(1, 'hello', user_id=1)
    assert await anext(stream) == ('working ', 'not_finished')
    assert (await anext(stream))[0] == result
    assert not lock.locked()
    assert len(pm.calls) == 1
    await stream.aclose()


async def test_retry_buffer_still_dispatches_late_tool_call():
    pm = DummyPluginManager({'p.do': 'ok'}, specs=[{'type': 'function', 'function': {'name': 'p.do'}}])
    source = FakeClosableAsyncStream([
        FakeStreamItem(content='thinking '),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    helper = _make_helper(pm, client=DummyClient([_fake_stream(['done'])]))
    response, _ = await helper._handle_function_call(1, source, stream=True, user_id=1, retry_plain_text_tool_intent=True)
    from bot.ai_providers.openai_compatible import stream_chunk_text_delta
    text = ''.join([stream_chunk_text_delta(item) async for item in response])
    assert text == 'thinking done'
    assert len(pm.calls) == 1


async def test_text_and_tool_in_same_chunk_are_both_processed():
    pm = DummyPluginManager({'p.do': 'ok'}, specs=[{'type': 'function', 'function': {'name': 'p.do'}}])
    source = FakeClosableAsyncStream([
        FakeStreamItem(content='thinking ', tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    helper = _make_helper(pm, client=DummyClient([source, _fake_stream(['done'])]))
    output = [item async for item in helper.get_chat_response_stream(1, 'hello', user_id=1)]
    assert output[0] == ('thinking ', 'not_finished')
    assert output[-1][0] == 'thinking done'
    assert len(pm.calls) == 1


async def test_stopping_mixed_stream_closes_provider_stream():
    source = FakeClosableAsyncStream([FakeStreamItem(content='partial'), FakeStreamItem(content='later')])
    helper = _make_helper(DummyPluginManager({}))
    response, _ = await helper._handle_function_call(1, source, stream=True, user_id=1)
    iterator = response.__aiter__()
    assert (await anext(iterator)).choices[0].delta.content == 'partial'
    await response.aclose()
    assert source.closed


async def test_agent_delivery_waits_for_late_tool_and_keeps_prelude_internal():
    tool = 'agent_tools.deliver_to_user'
    result = {'direct_result': {'kind': 'final', 'format': 'mixed', 'text': 'finished'}}
    pm = DummyPluginManager({tool: result})
    source = FakeClosableAsyncStream([
        FakeStreamItem(content='internal prelude'),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name=tool, arguments='{}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    client = DummyClient([source])
    helper = _make_helper(pm, client=client)
    helper.conversations[1] = [{"role": "system", "mode_key": "skills_agent", "content": "agent mode"}]

    output = [item async for item in helper.get_chat_response_stream(1, 'hello', user_id=1)]

    assert len(output) == 1
    assert output[0][0] == result
    assert client.calls == 1
    assert [name for name, _ in pm.calls] == [tool]


async def test_stopping_mixed_stream_closes_the_reply_after_the_tool():
    pm = DummyPluginManager({'p.do': 'ok'})
    source = FakeClosableAsyncStream([
        FakeStreamItem(content='working '),
        FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{}')]),
        FakeStreamItem(finish_reason='tool_calls'),
    ])
    continuation = FakeClosableAsyncStream([FakeStreamItem(content='almost '), FakeStreamItem(content='done')])
    helper = _make_helper(pm, client=DummyClient([source, continuation]))
    stream = helper.get_chat_response_stream(1, 'hello', user_id=1)

    assert await anext(stream) == ('working ', 'not_finished')
    assert await anext(stream) == ('working almost ', 'not_finished')
    await stream.aclose()

    assert source.closed
    assert continuation.closed
