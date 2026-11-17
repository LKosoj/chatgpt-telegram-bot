import pytest

from tests.test_openai_helper_tool_calls import (
    DummyClient, DummyPluginManager, FakeResponse, FakeToolCall,
    FakeClosableAsyncStream, FakeStreamItem, FakeStreamToolCallDelta,
    _fake_stream, _make_helper,
)


def spec(name):
    return {'type': 'function', 'function': {'name': name}}


@pytest.mark.parametrize('choice,expected', [
    ('none', []),
    ('auto', ['p.do', 'p.other']),
    ({'type': 'function', 'function': {'name': 'p.do'}}, ['p.do']),
])
async def test_only_tools_offered_to_this_request_are_executed(choice, expected):
    pm = DummyPluginManager({'p.do': 'ok', 'p.other': 'ok'})
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall('p.do', '{}'), FakeToolCall('p.other', '{}')]),
        FakeResponse(content='done'),
    ])
    helper = _make_helper(pm, client=client)
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=False,
        tools=[spec('p.do'), spec('p.other')], tool_choice=choice,
    )
    await helper._handle_function_call(1, response, user_id=1)
    assert [name for name, _ in pm.calls] == expected


async def test_stream_rejects_tool_after_text_if_not_offered():
    pm = DummyPluginManager({'p.do': 'ok'})
    helper = _make_helper(pm, client=DummyClient([
        FakeClosableAsyncStream([
            FakeStreamItem(content='thinking '),
            FakeStreamItem(tool_calls=[FakeStreamToolCallDelta(0, id='call-1', name='p.do', arguments='{}')]),
            FakeStreamItem(finish_reason='tool_calls'),
        ]),
        _fake_stream(['done']),
    ]))
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=True,
        tools=[spec('p.do')], tool_choice='none',
    )
    response, _ = await helper._handle_function_call(1, response, stream=True, user_id=1)
    assert [item async for item in response]
    assert pm.calls == []


async def test_unoffered_tool_after_round_limit_does_not_restart_the_loop():
    pm = DummyPluginManager({'p.do': 'ok'})
    client = DummyClient([FakeResponse(tool_calls=[FakeToolCall('p.do', '{}')])])
    helper = _make_helper(pm, client=client)
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=False,
        tools=[spec('p.do')], tool_choice='none',
    )
    with pytest.raises(RuntimeError, match="tool-call limit"):
        await helper._handle_function_call(
            1, response, times=helper.config['functions_max_consecutive_calls'] + 1, user_id=1,
        )
    assert client.calls == 1
    assert pm.calls == []


async def test_canonical_tool_name_can_match_offered_model_alias():
    pm = DummyPluginManager({'p.do': 'ok'})
    pm.to_canonical_function_name = lambda name: {'p__do': 'p.do'}.get(name, name)
    helper = _make_helper(pm, client=DummyClient([
        FakeResponse(tool_calls=[FakeToolCall('p.do', '{}')]), FakeResponse(content='done'),
    ]))
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=False,
        tools=[spec('p__do')], tool_choice='auto',
    )
    await helper._handle_function_call(1, response, user_id=1)
    assert [name for name, _ in pm.calls] == ['p.do']
