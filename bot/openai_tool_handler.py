from __future__ import annotations

import asyncio
import json
import logging
import os

import openai

from .i18n import localized_text
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
from .tool_result import (
    direct_result_payload as _normalized_direct_result_payload,
    normalize_tool_result,
)

logger = logging.getLogger(__name__)


def _tool_call_semaphore(helper) -> asyncio.Semaphore:
    """Per-helper semaphore that bounds parallel tool execution.

    Why: a single batch from the model can request many tool calls; without a
    bound, asyncio.gather fans out unbounded subprocess/HTTP work and starves
    the event loop. Lazy-init keeps the semaphore bound to the current loop.
    """
    sem = getattr(helper, "_tool_call_semaphore", None)
    if sem is None:
        try:
            limit = max(1, int(os.getenv("TOOL_CALL_PARALLELISM", "5")))
        except ValueError:
            limit = 5
        sem = asyncio.Semaphore(limit)
        helper._tool_call_semaphore = sem
    return sem


async def _call_function_bounded(helper, name, args, request_context, semaphore):
    async with semaphore:
        without_chat_lock = getattr(helper, "_without_chat_lock", None)
        tool_chat_id = None
        if callable(without_chat_lock):
            try:
                tool_args = json.loads(args) if isinstance(args, str) else args
                if isinstance(tool_args, dict):
                    tool_chat_id = tool_args.get("chat_id")
            except (TypeError, json.JSONDecodeError):
                tool_chat_id = None
        if callable(without_chat_lock) and tool_chat_id is not None:
            with without_chat_lock(tool_chat_id):
                return await helper.plugin_manager.call_function(
                    name, helper, args, request_context=request_context
                )
        return await helper.plugin_manager.call_function(
            name, helper, args, request_context=request_context
        )


async def _list_user_sessions(db, user_id, *, is_active: int = 0):
    """Use the async DB API when available, fall back to sync for test doubles."""
    fn = getattr(db, "list_user_sessions_async", None)
    if fn is not None:
        return await fn(user_id, is_active=is_active)
    return db.list_user_sessions(user_id, is_active=is_active)


DELIVERY_TOOL_NAME = "agent_tools.deliver_to_user"
DELIVERY_PLUGIN_PREFIX = DELIVERY_TOOL_NAME.rsplit(".", 1)[0] + "."
ASK_USER_TOOL_NAME = DELIVERY_PLUGIN_PREFIX + "ask_telegram_user"
DELIVERY_REPAIR_MAX_ATTEMPTS = 2
DELIVERY_REPAIR_PROMPT = (
    "The previous assistant text was not delivered to the user; treat it as internal progress. "
    "Continue solving the original user task from the current conversation state. Do not narrate "
    "a future action as the final response. Choose the next appropriate action: if the next step "
    "requires a tool, call that tool; if user input is needed, call agent_tools.ask_telegram_user; "
    "if the task is complete or blocked, call agent_tools.deliver_to_user with status, final text, "
    "artifacts if any, and a concise verification_summary or blocked_reason. If previous tool "
    "results created local files, use their paths as artifacts."
)


async def _prepend_stream_item(first_item, response):
    yield first_item
    async for item in response:
        yield item


def _direct_result_payload(response) -> dict | None:
    return _normalized_direct_result_payload(response)


def _should_defer_direct_result(response, defer_direct_results: bool, tool_name: str = "") -> bool:
    if not defer_direct_results:
        return False
    direct_result = _direct_result_payload(response)
    if direct_result and direct_result.get("kind") == "final":
        return False
    if tool_name == ASK_USER_TOOL_NAME:
        return False
    return True


def _agent_delivery_workflow_active(tool_calls: list[dict], tools_used: tuple) -> bool:
    names = [str(name or "") for name in tools_used]
    names.extend(str(call.get("name") or "") for call in tool_calls if isinstance(call, dict))
    return any(
        name.startswith(DELIVERY_PLUGIN_PREFIX) and name != DELIVERY_TOOL_NAME
        for name in names
    )


_TEXT_PREVIEW_LIMIT = 600
_ARTIFACT_MANIFEST_LIMIT = 50


def _artifact_path(value) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip()
    if not path or "\n" in path or "://" in path:
        return None
    return path if os.path.isabs(path) else None


def _append_artifact_entry(manifest: list[dict], seen_paths: set[str], entry: dict) -> None:
    path = _artifact_path(entry.get("path"))
    if not path or path in seen_paths:
        return
    seen_paths.add(path)
    manifest.append({k: v for k, v in entry.items() if v is not None})


def _artifact_manifest_message(manifest: list[dict]) -> str:
    visible = manifest[-_ARTIFACT_MANIFEST_LIMIT:]
    omitted = max(0, len(manifest) - len(visible))
    payload = {
        "current_run_artifacts": visible,
        "instruction": (
            "Use these exact paths for artifacts created in this request. "
            "Do not discover artifacts by broad-listing shared directories. "
            "If inspection is needed, run bounded commands against specific paths from this manifest."
        ),
    }
    if omitted:
        payload["omitted_older_artifacts"] = omitted
    return "Current run artifact manifest:\n" + json.dumps(payload, ensure_ascii=False)


def _compact_deferred_tool_response(response) -> str:
    """
    Replace a deferred direct_result payload with a compact summary suitable for
    LLM history. Keeps file paths and metadata so the model can reference results
    in subsequent steps, but drops large text bodies and binary blobs that would
    otherwise pollute context with thousands of tokens.
    """
    direct_result = _direct_result_payload(response) or {}
    compact: dict = {"result": "deferred"}
    for key in ("kind", "format", "value", "file_path", "url", "mime_type", "caption"):
        value = direct_result.get(key)
        if isinstance(value, str) and value:
            compact[key] = value

    text = direct_result.get("text")
    if isinstance(text, str) and text:
        compact["text_chars"] = len(text)
        if len(text) <= _TEXT_PREVIEW_LIMIT:
            compact["text"] = text
        else:
            compact["text_preview"] = text[:_TEXT_PREVIEW_LIMIT]

    artifacts = direct_result.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        compact["artifacts_count"] = len(artifacts)
        compact["artifact_paths"] = [
            str(item.get("value") or item.get("file_path") or "")
            for item in artifacts
            if isinstance(item, dict) and (item.get("value") or item.get("file_path"))
        ]

    cleanup_skills = direct_result.get("cleanup_skills")
    if isinstance(cleanup_skills, list) and cleanup_skills:
        compact["cleanup_skills_count"] = len(cleanup_skills)

    return json.dumps(compact, ensure_ascii=False)


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


def _delivery_contract_required(
    helper,
    chat_id,
    final_delivery_required: bool,
    *,
    initial_plain_response: bool = False,
) -> bool:
    if _is_skills_agent_mode(helper, chat_id) and (
        final_delivery_required or initial_plain_response
    ):
        return True
    if not final_delivery_required:
        return False
    system_message = _system_message(helper, chat_id)
    if not isinstance(system_message, dict):
        return False
    mode_from_system = getattr(helper, "_mode_from_system_message", None)
    if not callable(mode_from_system):
        return False
    current_mode = mode_from_system(system_message)
    if not isinstance(current_mode, dict):
        return False
    prompt = str(current_mode.get("prompt_start") or "")
    return bool(current_mode.get("defer_direct_results") and DELIVERY_TOOL_NAME in prompt)


def _delivery_tool_is_allowed(helper, allowed_plugins) -> bool:
    is_allowed = getattr(helper.plugin_manager, "is_function_allowed", None)
    if not callable(is_allowed):
        return False
    return bool(is_allowed(DELIVERY_TOOL_NAME, allowed_plugins))


def _delivery_contract_error(helper) -> dict:
    bot_language = (getattr(helper, "config", None) or {}).get("bot_language", "en")
    return {
        "direct_result": {
            "kind": "text",
            "format": "text",
            "value": localized_text("delivery_contract_error", bot_language),
        }
    }


async def _retry_missing_delivery_tool(
    helper,
    chat_id,
    stream,
    times,
    tools_used,
    allowed_plugins,
    user_id,
    request_context,
    model_to_use=None,
    delivery_repair_attempts=0,
    plain_text=None,
    artifact_manifest=None,
):
    if not _delivery_tool_is_allowed(helper, allowed_plugins):
        logger.error("Delivery contract is required, but %s is not allowed", DELIVERY_TOOL_NAME)
        return _delivery_contract_error(helper), tools_used

    plain_text_preview = str(plain_text or "").strip()[:_TEXT_PREVIEW_LIMIT]
    helper.conversations.setdefault(chat_id, []).append({
        "role": "user",
        "content": (
            DELIVERY_REPAIR_PROMPT
            if not plain_text_preview
            else f"{DELIVERY_REPAIR_PROMPT}\n\nUndelivered assistant text: {plain_text_preview}"
        ),
    })
    logger.warning(
        "Retrying final response because skills_agent returned plain text instead of %s",
        DELIVERY_TOOL_NAME,
    )

    model_to_use = model_to_use or helper.get_current_model(user_id)
    sessions = await _list_user_sessions(helper.db, user_id, is_active=1)
    active_session = next((s for s in sessions if s['is_active']), None)
    session_id = active_session['session_id'] if active_session else None
    max_tokens_percent = active_session['max_tokens_percent'] if active_session else 80
    tools = helper.plugin_manager.get_functions_specs(helper, model_to_use, allowed_plugins)

    messages = await helper._apply_before_chat_request_mutators(
        chat_id=chat_id,
        user_id=user_id,
        session_id=session_id,
        request_id=None,
        persist=False,
    )
    response = await helper.chat_completion(
        model=model_to_use,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=helper.get_max_tokens(model_to_use, max_tokens_percent, chat_id),
        stream=stream,
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
        final_delivery_required=True,
        model_to_use=model_to_use,
        delivery_repair_attempts=delivery_repair_attempts + 1,
        artifact_manifest=artifact_manifest,
    )


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
    final_delivery_required=False,
    model_to_use=None,
    delivery_repair_attempts=0,
    artifact_manifest=None,
):
    tool_calls = []
    try:
        artifact_manifest = list(artifact_manifest or [])
        if request_context is not None:
            chat_id = request_context.chat_id
            user_id = request_context.user_id

        if allowed_plugins is None:
            allowed_plugins = ['All']
        allowed_plugins = helper.plugin_manager.filter_allowed_plugins(allowed_plugins)

        async def enforce_delivery_contract_if_needed(plain_text=None):
            if not _delivery_contract_required(
                helper,
                chat_id,
                final_delivery_required,
                initial_plain_response=times == 0 and not tools_used,
            ):
                return None
            if delivery_repair_attempts >= DELIVERY_REPAIR_MAX_ATTEMPTS:
                logger.error("Delivery contract retry failed for chat_id=%s", chat_id)
                return _delivery_contract_error(helper), tools_used
            return await _retry_missing_delivery_tool(
                helper,
                chat_id,
                stream,
                times,
                tools_used,
                allowed_plugins,
                user_id,
                request_context,
                model_to_use,
                delivery_repair_attempts,
                plain_text,
                artifact_manifest,
            )

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
                            enforced = await enforce_delivery_contract_if_needed()
                            if enforced is not None:
                                return enforced
                            return _prepend_stream_item(item, response), tools_used
                    else:
                        enforced = await enforce_delivery_contract_if_needed()
                        if enforced is not None:
                            return enforced
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
                    plain_text = getattr(first_choice.message, "content", None)
                    enforced = await enforce_delivery_contract_if_needed(plain_text)
                    if enforced is not None:
                        return enforced
                    return response, tools_used
            else:
                enforced = await enforce_delivery_contract_if_needed()
                if enforced is not None:
                    return enforced
                return response, tools_used

        if not tool_calls:
            enforced = await enforce_delivery_contract_if_needed()
            if enforced is not None:
                return enforced
            return response, tools_used

        model_to_use = model_to_use or helper.get_current_model(user_id)
        uses_structured_tool_history = getattr(helper, "_uses_structured_tool_history", lambda _model: False)
        add_assistant_tool_calls_to_history = getattr(helper, "_add_assistant_tool_calls_to_history", None)
        structured_tool_history = (
            uses_structured_tool_history(model_to_use)
            and callable(add_assistant_tool_calls_to_history)
            and all(call.get("id") for call in tool_calls)
        )
        if structured_tool_history:
            add_assistant_tool_calls_to_history(chat_id, tool_calls)
        mode_defers_direct_results = bool(
            getattr(helper, "_defer_direct_tool_results", lambda _chat_id: False)(chat_id)
        )
        agent_delivery_workflow = (
            _agent_delivery_workflow_active(tool_calls, tools_used)
            and _delivery_tool_is_allowed(helper, allowed_plugins)
        )
        defer_direct_results = mode_defers_direct_results or agent_delivery_workflow
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
                if not isinstance(args, dict):
                    raise TypeError("tool arguments must be a JSON object")
                if request_context is not None:
                    args['chat_id'] = request_context.plugin_chat_id
                    args['user_id'] = request_context.user_id
                    if request_context.message_id is not None:
                        args['message_id'] = request_context.message_id
                else:
                    args['chat_id'] = int(chat_id) if chat_id is not None else chat_id
                    args['user_id'] = user_id if user_id is not None else args['chat_id']
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
            except TypeError as exc:
                logger.error(f"Invalid arguments for {tool_name}: {exc}")
                errors.append((tool_name, tool_call_id, json.dumps({'error': f'Invalid arguments for {tool_name}'}, ensure_ascii=False)))

        semaphore = _tool_call_semaphore(helper)
        tasks = [
            _call_function_bounded(helper, name, args, request_context, semaphore)
            for name, args, _ in prepared
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        direct_results_collected: list = []
        seen_artifact_paths = {
            entry.get("path") for entry in artifact_manifest if isinstance(entry, dict)
        }
        seen_artifact_paths.discard(None)
        batch_artifact_count = 0
        for (tool_name, _, tool_call_id), tool_response in zip(prepared, results):
            if isinstance(tool_response, Exception):
                tool_response = json.dumps({'error': str(tool_response)}, ensure_ascii=False)
            tool_result = normalize_tool_result(tool_response, tool_name=tool_name)
            logger.info(f'Function {tool_name} response: {tool_result.content}')
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            if tool_result.success:
                final_delivery_required = True
            for entry in tool_result.artifacts:
                before = len(artifact_manifest)
                _append_artifact_entry(artifact_manifest, seen_artifact_paths, entry)
                if len(artifact_manifest) > before:
                    batch_artifact_count += 1
            is_dr = isinstance(tool_result.direct_result, dict)
            if is_dr and not _should_defer_direct_result(
                tool_response,
                defer_direct_results,
                tool_name,
            ):
                direct_results_collected.append(tool_response)
                add_tool_result(
                    tool_name,
                    json.dumps({'result': 'Done, the content has been sent to the user.'}),
                    tool_call_id,
                )
            else:
                if is_dr:
                    logger.info("Deferring direct result from tool %s to model reentry", tool_name)
                    add_tool_result(
                        tool_name,
                        _compact_deferred_tool_response(tool_response),
                        tool_call_id,
                    )
                else:
                    add_tool_result(tool_name, tool_result.content, tool_call_id)

        for tool_name, tool_call_id, tool_response in errors:
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            add_tool_result(tool_name, tool_response, tool_call_id)

        if direct_results_collected:
            if len(direct_results_collected) == 1:
                return direct_results_collected[0], tools_used
            return _merge_direct_results_into_final(direct_results_collected), tools_used

        if batch_artifact_count:
            helper.conversations.setdefault(chat_id, []).append({
                "role": "user",
                "content": _artifact_manifest_message(artifact_manifest),
            })

        sessions = await _list_user_sessions(helper.db, user_id, is_active=1)
        active_session = next((s for s in sessions if s['is_active']), None)
        session_id = active_session['session_id'] if active_session else None
        max_tokens_percent = active_session['max_tokens_percent'] if active_session else 80

        logger.info(f'Function calls completed. messages: {helper.conversations[chat_id]} session_id: {session_id}')

        tools = helper.plugin_manager.get_functions_specs(helper, model_to_use, allowed_plugins)

        messages = await helper._apply_before_chat_request_mutators(
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
            request_id=None,
            persist=False,
        )
        response = await helper.chat_completion(
            model=model_to_use,
            messages=messages,
            tools=tools,
            tool_choice='auto' if times < helper.config['functions_max_consecutive_calls'] else 'none',
            max_tokens=helper.get_max_tokens(model_to_use, max_tokens_percent, chat_id),
            stream=stream,
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
            final_delivery_required,
            model_to_use,
            delivery_repair_attempts,
            artifact_manifest,
        )
    except Exception:
        logger.error('Error in function call handling', exc_info=True)
        raise
