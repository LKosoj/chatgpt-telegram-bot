from __future__ import annotations

import asyncio
import json
import logging

import openai

from .utils import is_direct_result, escape_markdown

logger = logging.getLogger(__name__)

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
                            return response, tools_used
                    else:
                        return response, tools_used
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

        direct_result = None
        for (tool_name, _, tool_call_id), tool_response in zip(prepared, results):
            if isinstance(tool_response, Exception):
                tool_response = json.dumps({'error': str(tool_response)}, ensure_ascii=False)
            logger.info(f'Function {tool_name} response: {tool_response}')
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            if is_direct_result(tool_response) and direct_result is None:
                direct_result = tool_response
            if direct_result:
                add_tool_result(
                    tool_name,
                    json.dumps({'result': 'Done, the content has been sent'
                                'to the user.'}),
                    tool_call_id,
                )
            else:
                add_tool_result(tool_name, tool_response, tool_call_id)

        for tool_name, tool_call_id, tool_response in errors:
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            add_tool_result(tool_name, tool_response, tool_call_id)

        if direct_result:
            return direct_result, tools_used

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
    except Exception as e:
        logger.error(f'Error in function call handling: {str(e)}', exc_info=True)
        bot_language = helper.config['bot_language']
        error_message = escape_markdown(str(e))
        raise Exception(f"⚠️ _{helper._localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e
