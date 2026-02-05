from __future__ import annotations

import asyncio
import json
import logging

import openai

from .utils import is_direct_result, escape_markdown

logger = logging.getLogger(__name__)


async def handle_function_call(helper, chat_id, response, stream=False, times=0, tools_used=(), allowed_plugins=['All'], user_id=None):
    tool_calls = []
    try:
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
                                entry = tool_call_parts.setdefault(idx, {"name": "", "arguments": ""})
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
            for _, entry in sorted(tool_call_parts.items(), key=lambda x: x[0]):
                tool_calls.append(entry)
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                logger.info("found tool calls")
                logger.info(f"first_choice = {first_choice}")
                if first_choice.message.tool_calls:
                    for tc in first_choice.message.tool_calls:
                        tool_calls.append({
                            "name": tc.function.name or "",
                            "arguments": tc.function.arguments or "",
                        })
                else:
                    return response, tools_used
            else:
                return response, tools_used

        if not tool_calls:
            return response, tools_used

        helper.user_id = user_id
        model_to_use = helper.get_current_model(user_id)

        prepared = []
        errors = {}
        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            logger.info(f'Calling tool {tool_name} with arguments {arguments}')
            try:
                args = json.loads(arguments)
                args['chat_id'] = chat_id
                args['user_id'] = user_id if user_id is not None else chat_id
                arguments = json.dumps(args, ensure_ascii=False)
                prepared.append((tool_name, arguments))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments JSON: {arguments}")
                errors[tool_name] = json.dumps({'error': f'Invalid arguments for {tool_name}'}, ensure_ascii=False)

        tasks = [helper.plugin_manager.call_function(name, helper, args) for name, args in prepared]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        direct_result = None
        for (tool_name, _), tool_response in zip(prepared, results):
            if isinstance(tool_response, Exception):
                tool_response = json.dumps({'error': str(tool_response)}, ensure_ascii=False)
            logger.info(f'Function {tool_name} response: {tool_response}')
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            if is_direct_result(tool_response) and direct_result is None:
                direct_result = tool_response
            if direct_result:
                helper._add_function_call_to_history(chat_id=chat_id, function_name=tool_name,
                                                     content=json.dumps({'result': 'Done, the content has been sent'
                                                                     'to the user.'}))
            else:
                helper._add_function_call_to_history(chat_id=chat_id, function_name=tool_name, content=tool_response)

        for tool_name, tool_response in errors.items():
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            helper._add_function_call_to_history(chat_id=chat_id, function_name=tool_name, content=tool_response)

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
            messages=helper.conversations[chat_id],
            tools=tools,
            tool_choice='auto' if times < helper.config['functions_max_consecutive_calls'] else 'none',
            max_tokens=helper.get_max_tokens(model_to_use, max_tokens_percent, chat_id),
            stream=stream,
            extra_headers={ "X-Title": "tgBot" },
        )
        return await handle_function_call(helper, chat_id, response, stream, times + 1, tools_used, allowed_plugins, user_id)
    except Exception as e:
        logger.error(f'Error in function call handling: {str(e)}', exc_info=True)
        bot_language = helper.config['bot_language']
        error_message = escape_markdown(str(e))
        raise Exception(f"⚠️ _{helper._localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e
