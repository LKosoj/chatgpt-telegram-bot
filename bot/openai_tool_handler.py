from __future__ import annotations

import asyncio
import json
import logging

import openai

from .skill_script_routing import (
    SCRIPT_FILE_CREATION_RE,
    SKILL_SCRIPT_PATH_RE,
    _active_skill_scripts,
    _is_skills_agent_mode,
    _refers_to_active_script,
    _skill_script_routing_error,
    _skill_script_routing_payload,
    _system_message,
)
from .utils import is_direct_result

logger = logging.getLogger(__name__)


async def _prepend_stream_item(first_item, response):
    yield first_item
    async for item in response:
        yield item


def _direct_result_payload(response) -> dict | None:
    if type(response) is not dict:
        try:
            response = json.loads(response)
        except Exception:
            return None
    result = response.get("direct_result")
    return result if isinstance(result, dict) else None


def _should_defer_direct_result(response, defer_direct_results: bool) -> bool:
    if not defer_direct_results:
        return False
    direct_result = _direct_result_payload(response)
    if direct_result and direct_result.get("defer") is False:
        return False
    return True


def _merge_direct_results_into_final(direct_results: list) -> dict:
    text_parts: list[str] = []
    artifacts: list[dict] = []
    cleanup_directives: list[dict] = []
    for tool_response in direct_results:
        payload = tool_response
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        direct_result = payload.get("direct_result")
        if not isinstance(direct_result, dict):
            continue

        single_cleanup = direct_result.get("cleanup_skill")
        if isinstance(single_cleanup, dict):
            cleanup_directives.append(single_cleanup)
        many_cleanup = direct_result.get("cleanup_skills")
        if isinstance(many_cleanup, list):
            cleanup_directives.extend(item for item in many_cleanup if isinstance(item, dict))

        kind = direct_result.get("kind")
        if kind == "final":
            text = direct_result.get("text")
            if text:
                text_parts.append(str(text))
            for artifact in direct_result.get("artifacts") or []:
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
            continue
        if kind == "text":
            value = direct_result.get("value")
            if value:
                text_parts.append(str(value))
            continue
        if kind in ("file", "photo", "gif"):
            artifacts.append({
                key: direct_result.get(key)
                for key in ("kind", "format", "value", "add_value", "file_size")
                if direct_result.get(key) is not None
            })
            continue
        artifacts.append({k: v for k, v in direct_result.items() if k not in ("cleanup_skill", "cleanup_skills")})

    merged: dict = {
        "direct_result": {
            "kind": "final",
            "format": "mixed",
            "defer": False,
            "text": "\n\n".join(text_parts),
            "artifacts": artifacts,
        }
    }
    if cleanup_directives:
        merged["direct_result"]["cleanup_skills"] = cleanup_directives
    return merged


def _extract_legacy_tool_request(content: str | None) -> tuple[str, str] | None:
    """
    Backwards-compatible parsing for legacy prompts that ask the model to output JSON like:
      {"tool_name": "<tool>", ...args }

    Returns (tool_name, arguments_json_str) or None.
    """
    if not content or not isinstance(content, str):
        return None
    s = content.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = s[start : end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    tool_name = obj.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    args = {k: v for k, v in obj.items() if k != "tool_name"}
    try:
        return tool_name.strip(), json.dumps(args, ensure_ascii=False)
    except Exception:
        return None


async def handle_function_call(
    helper,
    chat_id,
    response,
    stream=False,
    times=0,
    tools_used=(),
    allowed_plugins=None,
    user_id=None,
    request_context=None,
):
    tool_calls = []
    try:
        if request_context is not None:
            chat_id = request_context.chat_id
            user_id = request_context.user_id

        if allowed_plugins is None:
            allowed_plugins = ['All']
        allowed_plugins = helper.plugin_manager.filter_allowed_plugins(allowed_plugins)
        if stream:
            try:
                tool_call_parts = {}
                async for item in response:
                    if not item.choices:
                        continue
                    if len(item.choices) > 0:
                        first_choice = item.choices[0]
                        if first_choice.delta and first_choice.delta.tool_calls:
                            logger.info("found tool calls")
                            for tc in first_choice.delta.tool_calls:
                                idx = getattr(tc, "index", 0)
                                entry = tool_call_parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                                if getattr(tc, "id", None):
                                    entry["id"] += tc.id
                                if tc.function.name:
                                    entry["name"] += tc.function.name
                                if tc.function.arguments:
                                    entry["arguments"] += tc.function.arguments
                        elif first_choice.finish_reason and first_choice.finish_reason == 'tool_calls':
                            break
                        else:
                            return _prepend_stream_item(item, response), tools_used
                    else:
                        return _prepend_stream_item(item, response), tools_used
            except openai.APIError as e:
                logger.error(f"API Error in function call streaming: {e}")
                return response, tools_used
            for idx, entry in sorted(tool_call_parts.items(), key=lambda x: x[0]):
                entry["id"] = entry["id"] or f"call_{idx}"
                tool_calls.append(entry)
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                logger.info(f"first_choice = {first_choice}")
                if first_choice.message.tool_calls:
                    logger.info("found tool calls")
                    for tc in first_choice.message.tool_calls:
                        tool_calls.append({
                            "id": getattr(tc, "id", None) or f"call_{len(tool_calls)}",
                            "name": tc.function.name or "",
                            "arguments": tc.function.arguments or "",
                        })
                else:
                    legacy = _extract_legacy_tool_request(getattr(first_choice.message, "content", None))
                    if legacy:
                        tool_name, arguments = legacy
                        logger.info("found legacy tool request in assistant content")
                        tool_calls.append({"id": None, "name": tool_name, "arguments": arguments, "legacy": True})
                    else:
                        return response, tools_used
            else:
                return response, tools_used

        if not tool_calls:
            return response, tools_used

        model_to_use = helper.get_current_model(user_id)
        uses_structured_tool_history = getattr(helper, "_uses_structured_tool_history", lambda _model: False)
        add_assistant_tool_calls_to_history = getattr(helper, "_add_assistant_tool_calls_to_history", None)
        structured_tool_history = (
            uses_structured_tool_history(model_to_use)
            and callable(add_assistant_tool_calls_to_history)
            and all(call.get("id") and not call.get("legacy") for call in tool_calls)
        )
        if structured_tool_history:
            add_assistant_tool_calls_to_history(chat_id, tool_calls)
        defer_direct_results = bool(
            getattr(helper, "_defer_direct_tool_results", lambda _chat_id: False)(chat_id)
        )
        if defer_direct_results:
            logger.info("Direct tool results will be deferred for chat_id=%s", chat_id)

        def add_tool_result(tool_name, tool_response, tool_call_id=None):
            if structured_tool_history:
                helper._add_function_call_to_history(
                    chat_id=chat_id,
                    function_name=tool_name,
                    content=tool_response,
                    tool_call_id=tool_call_id,
                )
            else:
                helper._add_function_call_to_history(
                    chat_id=chat_id,
                    function_name=tool_name,
                    content=tool_response,
                )

        prepared = []
        errors = []
        for call in tool_calls:
            tool_call_id = call.get("id")
            tool_name = call["name"]
            arguments = call["arguments"]
            logger.info(f'Calling tool {tool_name} with arguments {arguments}')
            if not helper.plugin_manager.is_function_allowed(tool_name, allowed_plugins):
                error = f'Tool {tool_name} is not allowed in the current chat mode'
                logger.warning(error)
                errors.append((tool_name, tool_call_id, json.dumps({'error': error}, ensure_ascii=False)))
                continue
            try:
                args = json.loads(arguments)
                if request_context is not None:
                    args['chat_id'] = request_context.plugin_chat_id
                    args['user_id'] = request_context.user_id
                    if request_context.message_id is not None:
                        args['message_id'] = request_context.message_id
                else:
                    args['chat_id'] = str(chat_id)
                    args['user_id'] = user_id if user_id is not None else chat_id
                routing_error = _skill_script_routing_error(helper, chat_id, tool_name, args)
                if routing_error:
                    logger.warning("%s Tool=%s", routing_error.get("error"), tool_name)
                    errors.append((
                        tool_name,
                        tool_call_id,
                        json.dumps(routing_error, ensure_ascii=False),
                    ))
                    continue
                arguments = json.dumps(args, ensure_ascii=False)
                prepared.append((tool_name, arguments, tool_call_id))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments JSON: {arguments}")
                errors.append((tool_name, tool_call_id, json.dumps({'error': f'Invalid arguments for {tool_name}'}, ensure_ascii=False)))

        tasks = [
            helper.plugin_manager.call_function(name, helper, args, request_context=request_context)
            for name, args, _ in prepared
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        direct_results_collected: list = []
        for (tool_name, _, tool_call_id), tool_response in zip(prepared, results):
            if isinstance(tool_response, Exception):
                tool_response = json.dumps({'error': str(tool_response)}, ensure_ascii=False)
            logger.info(f'Function {tool_name} response: {tool_response}')
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            is_dr = is_direct_result(tool_response)
            if is_dr and not _should_defer_direct_result(tool_response, defer_direct_results):
                direct_results_collected.append(tool_response)
                add_tool_result(
                    tool_name,
                    json.dumps({'result': 'Done, the content has been sent to the user.'}),
                    tool_call_id,
                )
            else:
                if is_dr:
                    logger.info("Deferring direct result from tool %s to model reentry", tool_name)
                add_tool_result(tool_name, tool_response, tool_call_id)

        for tool_name, tool_call_id, tool_response in errors:
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            add_tool_result(tool_name, tool_response, tool_call_id)

        if direct_results_collected:
            if len(direct_results_collected) == 1:
                return direct_results_collected[0], tools_used
            return _merge_direct_results_into_final(direct_results_collected), tools_used

        sessions = helper.db.list_user_sessions(user_id, is_active=1)
        active_session = next((s for s in sessions if s['is_active']), None)
        session_id = active_session['session_id'] if active_session else None
        max_tokens_percent = active_session['max_tokens_percent'] if active_session else 80

        logger.info(f'Function calls completed. messages: {helper.conversations[chat_id]} session_id: {session_id}')

        tools = helper.plugin_manager.get_functions_specs(helper, model_to_use, allowed_plugins)

        response = await helper.client.chat.completions.create(
            model=model_to_use,
            messages=helper._messages_with_hindsight_context(chat_id),
            tools=tools,
            tool_choice='auto' if times < helper.config['functions_max_consecutive_calls'] else 'none',
            max_tokens=helper.get_max_tokens(model_to_use, max_tokens_percent, chat_id),
            stream=stream,
            extra_headers={ "X-Title": "tgBot" },
        )
        return await handle_function_call(
            helper,
            chat_id,
            response,
            stream,
            times + 1,
            tools_used,
            allowed_plugins,
            user_id,
            request_context,
        )
    except Exception:
        logger.error('Error in function call handling', exc_info=True)
        raise
