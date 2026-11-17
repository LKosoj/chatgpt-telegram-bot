import logging

from bot.session_logger import set_trace, clear_trace
from tests.test_openai_helper_tool_calls import (
    CaptureSessionLogger, DummyClient, DummyPluginManager, FakeResponse, FakeToolCall, _make_helper,
)


async def test_failed_tool_result_has_failed_structured_status_and_no_payload_in_logs(caplog):
    secret = 'sensitive-result-marker'
    pm = DummyPluginManager({'p.do': {'success': False, 'message': secret}})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content='done')]))
    helper.session_logger = CaptureSessionLogger()
    caplog.set_level(logging.INFO)
    token = set_trace(1, 'session', 'request-id')
    try:
        await helper._handle_function_call(1, FakeResponse(tool_calls=[FakeToolCall('p.do', '{"text":"sensitive-argument-marker"}')]), user_id=1)
    finally:
        clear_trace(token)
    assert secret not in caplog.text
    assert 'sensitive-argument-marker' not in caplog.text
    assert 'request-id' in caplog.text
    event = next(event for event in helper.session_logger.events if event.get('type') == 'tool_exec')
    assert event['ok'] is False
    event = next(event for event in helper.session_logger.events if event.get('type') == 'tool_execution_end')
    assert event['result']['ok'] is False


async def test_request_prompt_schema_and_provider_error_are_not_in_ordinary_logs(caplog):
    class FailingClient(DummyClient):
        async def _create(self, **kwargs):
            raise RuntimeError('secret-provider-body')

    helper = _make_helper(DummyPluginManager({}), client=FailingClient())
    caplog.set_level(logging.INFO)
    try:
        await helper._timed_create_via_ai_provider(
            kind='main', model='llmgateway/high', messages=[{'role': 'user', 'content': 'secret-prompt'}],
            tools=[{'type': 'function', 'function': {'name': 'p.do', 'description': 'secret-schema'}}],
        )
    except RuntimeError:
        pass
    assert 'secret-prompt' not in caplog.text
    assert 'secret-schema' not in caplog.text
    assert 'secret-provider-body' not in caplog.text
    assert 'tools_count=1' in caplog.text
