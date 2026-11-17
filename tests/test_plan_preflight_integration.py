import json
import types

import pytest

from tests.test_openai_helper_tool_calls import (
    DummyClient, DummyPluginManager, FakeResponse, FakeToolCall, _make_helper,
)


def spec(name):
    return {'type': 'function', 'function': {'name': name}}


@pytest.mark.asyncio
async def test_discovery_then_execution_requires_plan_before_execution():
    names = ['skills.list_skills', 'p.do', 'agent_tools.manage_plan_tasks']
    planner = types.SimpleNamespace(tasks=[])
    planner.get_plan_tasks = lambda **kwargs: planner.tasks
    pm = DummyPluginManager({name: {'ok': True} for name in names},
                            specs=[spec(name) for name in names], plugins={'agent_tools': planner})
    original_call = pm.call_function

    async def call_function(name, helper, arguments, request_context=None):
        result = await original_call(name, helper, arguments, request_context=request_context)
        if name == 'agent_tools.manage_plan_tasks':
            planner.tasks.append({'id': 'T1'})
        return result

    pm.call_function = call_function
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall('skills.list_skills', '{}')]),
        FakeResponse(tool_calls=[FakeToolCall('p.do', '{}', id='rejected')]),
        FakeResponse(tool_calls=[FakeToolCall('agent_tools.manage_plan_tasks', '{}')]),
        FakeResponse(tool_calls=[FakeToolCall('p.do', '{}', id='allowed')]),
        FakeResponse(content='done'),
    ])
    helper = _make_helper(pm, client=client)
    helper.config['functions_max_consecutive_calls'] = 5
    helper.conversations[1] = [{'role': 'system', 'content': 'agent-mode', 'mode_key': 'skills_agent'}]
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=False,
        tools=pm.specs, tool_choice='auto',
    )
    await helper._handle_function_call(1, response, allowed_plugins=['All'], user_id=1)
    assert [name for name, _ in pm.calls] == ['skills.list_skills', 'agent_tools.manage_plan_tasks', 'p.do']
    forced = client.create_kwargs[2]
    assert forced['tool_choice'] == {'type': 'function', 'function': {'name': 'agent_tools.manage_plan_tasks'}}
    rejection = [m for m in helper.conversations[1] if m.get('tool_call_id') == 'rejected']
    assert rejection and 'plan' in str(rejection[0]['content']).lower()


@pytest.mark.asyncio
async def test_simple_delivery_does_not_require_plan():
    name = 'agent_tools.deliver_to_user'
    planner = types.SimpleNamespace(get_plan_tasks=lambda **kwargs: [])
    pm = DummyPluginManager({name: {'direct_result': {'kind': 'text', 'value': 'done'}}},
                            specs=[spec(name)], plugins={'agent_tools': planner})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(tool_calls=[FakeToolCall(name, json.dumps({'text': 'done'}))])]))
    helper.conversations[1] = [{'role': 'system', 'content': 'agent-mode', 'mode_key': 'skills_agent'}]
    response = await helper._timed_create_via_ai_provider(
        kind='main', model='llmgateway/high', messages=[], stream=False,
        tools=pm.specs, tool_choice='auto',
    )
    await helper._handle_function_call(1, response, allowed_plugins=['All'], user_id=1)
    assert [called for called, _ in pm.calls] == [name]
