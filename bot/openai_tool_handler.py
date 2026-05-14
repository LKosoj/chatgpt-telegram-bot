from __future__ import annotations

import asyncio
import json
import logging
import os
import time

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
    artifact_entries_from_tool_response,
    direct_result_payload,
    json_dict,
    normalize_tool_result,
    tool_response_succeeded,
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


def _tool_metadata(helper, tool_name: str) -> dict:
    get_metadata = getattr(getattr(helper, "plugin_manager", None), "get_tool_metadata", None)
    if not callable(get_metadata):
        return {}
    try:
        metadata = get_metadata(tool_name)
    except Exception:
        logger.exception("Failed to load metadata for tool %s", tool_name)
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _tool_timeout_seconds(metadata: dict) -> float | None:
    for key in ("timeout_seconds", "timeout"):
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    timeout_ms = metadata.get("timeout_ms")
    if isinstance(timeout_ms, (int, float)) and timeout_ms > 0:
        return float(timeout_ms) / 1000
    return None


def _tool_parallelizable(metadata: dict) -> bool:
    return metadata.get("parallelizable", True) is not False


async def _call_function_bounded(helper, name, args, request_context, semaphore, timeout_seconds=None):
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
        async def call():
            if callable(without_chat_lock) and tool_chat_id is not None:
                with without_chat_lock(tool_chat_id):
                    return await helper.plugin_manager.call_function(
                        name, helper, args, request_context=request_context
                    )
            return await helper.plugin_manager.call_function(
                name, helper, args, request_context=request_context
            )

        if timeout_seconds:
            try:
                return await asyncio.wait_for(call(), timeout=float(timeout_seconds))
            except asyncio.TimeoutError:
                return json.dumps({"error": f"Tool {name} timed out after {timeout_seconds}s"}, ensure_ascii=False)
        return await call()


async def _list_user_sessions(db, user_id, *, is_active: int = 0):
    """Use the async DB API when available, fall back to sync for test doubles."""
    fn = getattr(db, "list_user_sessions_async", None)
    if fn is not None:
        return await fn(user_id, is_active=is_active)
    return db.list_user_sessions(user_id, is_active=is_active)


DELIVERY_TOOL_NAME = "agent_tools.deliver_to_user"
DELIVERY_PLUGIN_PREFIX = DELIVERY_TOOL_NAME.rsplit(".", 1)[0] + "."
ASK_USER_TOOL_NAME = DELIVERY_PLUGIN_PREFIX + "ask_telegram_user"
DISCOVER_TOOLS_NAME = DELIVERY_PLUGIN_PREFIX + "discover_tools"
RUN_SUBAGENTS_NAME = DELIVERY_PLUGIN_PREFIX + "run_subagents"
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
STUCK_TOOL_LOOP_MESSAGE = (
    "Could not finish the task: the model repeated the same failing tool call batch. "
    "Stopped before re-running it."
)


async def _prepend_stream_item(first_item, response):
    yield first_item
    async for item in response:
        yield item


def _direct_result_payload(response) -> dict | None:
    return direct_result_payload(response)


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


def _canonical_tool_arguments(arguments) -> str:
    if not isinstance(arguments, str):
        try:
            return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(arguments)
    try:
        decoded = json.loads(arguments)
    except Exception:
        return arguments
    return json.dumps(decoded, ensure_ascii=False, sort_keys=True)


def _tool_batch_signature(tool_calls: list[dict]) -> tuple:
    return tuple(
        (
            str(call.get("name") or ""),
            _canonical_tool_arguments(call.get("arguments") or ""),
        )
        for call in tool_calls
        if isinstance(call, dict)
    )


def _stuck_tool_loop_error(tool_calls: list[dict], cleanup_skills: list[dict] | None = None) -> dict:
    tool_names = [
        str(call.get("name") or "").strip()
        for call in tool_calls
        if isinstance(call, dict) and str(call.get("name") or "").strip()
    ]
    unique_tool_names = list(dict.fromkeys(tool_names))
    suffix = f" Repeated tools: {', '.join(unique_tool_names)}." if unique_tool_names else ""
    text = STUCK_TOOL_LOOP_MESSAGE + suffix
    direct_result = {
        "kind": "final",
        "format": "mixed",
        "text_format": "text",
        "text": text,
        "artifacts": [],
        "status": "blocked",
        "blocked_reason": text,
        "verification_summary": "Stopped after the model repeated an identical failing tool batch.",
    }
    if cleanup_skills:
        direct_result["cleanup_skills"] = cleanup_skills
    return {
        "direct_result": direct_result
    }


def _cleanup_directives_for_blocked_result(helper, chat_id, user_id) -> list[dict]:
    plugin_manager = getattr(helper, "plugin_manager", None)
    get_plugin = getattr(plugin_manager, "get_plugin", None)
    if not callable(get_plugin):
        return []
    try:
        plugin = get_plugin(DELIVERY_TOOL_NAME.rsplit(".", 1)[0])
    except Exception:
        logger.debug("Failed to load delivery plugin for blocked-result cleanup", exc_info=True)
        return []
    cleanup = getattr(plugin, "cleanup_directives_for_active_skills", None)
    if not callable(cleanup):
        return []
    try:
        result = cleanup(helper, chat_id=chat_id, user_id=user_id)
    except Exception:
        logger.debug("Failed to build blocked-result cleanup directives", exc_info=True)
        return []
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


_TEXT_PREVIEW_LIMIT = 600
_TOOL_HISTORY_COMPACT_LIMIT = 6000
_TOOL_HISTORY_PREVIEW_LIMIT = 2000
_ARTIFACT_MANIFEST_LIMIT = 50


def _json_dict(value) -> dict | None:
    return json_dict(value)


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


def _artifact_entries_from_tool_response(tool_name: str, response) -> list[dict]:
    return list(artifact_entries_from_tool_response(tool_name, response))


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


def _put_compact_string(target: dict, key: str, value: str, *, keep_exact: bool = False) -> None:
    if keep_exact or len(value) <= _TEXT_PREVIEW_LIMIT:
        target[key] = value
        return
    target[f"{key}_preview"] = value[:_TEXT_PREVIEW_LIMIT]
    target[f"{key}_chars"] = len(value)


def _compact_deferred_tool_response(response) -> str:
    """
    Replace a deferred direct_result payload with a compact summary suitable for
    LLM history. Keeps file paths and metadata so the model can reference results
    in subsequent steps, but drops large text bodies and binary blobs that would
    otherwise pollute context with thousands of tokens.
    """
    direct_result = _direct_result_payload(response) or {}
    compact: dict = {"result": "deferred"}
    for key in ("kind", "format", "mime_type"):
        value = direct_result.get(key)
        if isinstance(value, str) and value:
            compact[key] = value

    result_format = direct_result.get("format")
    for key in ("value", "file_path", "url", "caption"):
        value = direct_result.get(key)
        if not isinstance(value, str) or not value:
            continue
        keep_exact = bool(_artifact_path(value) or (key in {"value", "file_path"} and result_format == "path"))
        _put_compact_string(compact, key, value, keep_exact=keep_exact)

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
        artifact_paths = []
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            path = _artifact_path(item.get("value")) or _artifact_path(item.get("file_path"))
            if path:
                artifact_paths.append(path)
        if artifact_paths:
            compact["artifact_paths"] = artifact_paths

    cleanup_skills = direct_result.get("cleanup_skills")
    if isinstance(cleanup_skills, list) and cleanup_skills:
        compact["cleanup_skills_count"] = len(cleanup_skills)

    return json.dumps(compact, ensure_ascii=False)


def _compact_value_for_history(value):
    if isinstance(value, str):
        if len(value) <= _TEXT_PREVIEW_LIMIT:
            return value
        return {"text_preview": value[:_TEXT_PREVIEW_LIMIT], "text_chars": len(value)}
    if isinstance(value, list):
        return {
            "items_preview": [_compact_value_for_history(item) for item in value[:5]],
            "items_count": len(value),
        }
    if isinstance(value, dict):
        return {
            "keys": sorted(str(key) for key in value.keys())[:20],
            "object": True,
        }
    return value


def _compact_tool_response_for_history(response) -> tuple[str, int]:
    content = response if isinstance(response, str) else json.dumps(response, default=str, ensure_ascii=False)
    if len(content) <= _TOOL_HISTORY_COMPACT_LIMIT:
        return content, 0

    payload = _json_dict(response)
    if isinstance(payload, dict):
        compact = {
            "_compacted_tool_response": True,
            "original_chars": len(content),
            "keys": sorted(str(key) for key in payload.keys()),
        }
        for key in ("success", "error", "status", "message", "result", "output"):
            if key in payload:
                compact[key] = _compact_value_for_history(payload.get(key))
        artifact_paths = [
            entry.get("path")
            for entry in _artifact_entries_from_tool_response("", payload)
            if entry.get("path")
        ]
        if artifact_paths:
            compact["artifact_paths"] = artifact_paths
        return json.dumps(compact, ensure_ascii=False), len(content)

    compact = {
        "_compacted_tool_response": True,
        "original_chars": len(content),
        "text_preview": content[:_TOOL_HISTORY_PREVIEW_LIMIT],
    }
    return json.dumps(compact, ensure_ascii=False), len(content)


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


def _tool_response_succeeded(response) -> bool:
    return tool_response_succeeded(response)


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


def _init_tool_run_state(helper, chat_id, user_id, request_context, model_to_use, run_state):
    if run_state is not None:
        return run_state
    request_id = getattr(request_context, "request_id", None) if request_context is not None else None
    return {
        "request_id": request_id,
        "chat_id": chat_id,
        "user_id": user_id,
        "model": model_to_use,
        "started_at": time.monotonic(),
    }


def _record_tool_run_event(
    helper,
    run_state,
    *,
    event_type: str,
    iteration: int,
    status: str,
    tool_count: int = 0,
    success_count: int = 0,
    error_count: int = 0,
    duration_ms: int = 0,
    stop_reason: str | None = None,
    metadata: dict | None = None,
) -> None:
    db = getattr(helper, "db", None)
    record = getattr(db, "record_tool_run_event", None)
    if not callable(record):
        return
    try:
        record(
            request_id=run_state.get("request_id"),
            chat_id=run_state.get("chat_id"),
            user_id=run_state.get("user_id"),
            event_type=event_type,
            iteration=iteration,
            status=status,
            tool_count=tool_count,
            success_count=success_count,
            error_count=error_count,
            duration_ms=duration_ms,
            stop_reason=stop_reason,
            metadata=metadata or {},
        )
    except Exception:
        logger.exception("Failed to record tool run event")


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
    failed_tool_batch_signature=None,
    run_state=None,
    tool_disclosure_state=None,
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
    if tool_disclosure_state is not None:
        tools = helper.plugin_manager.get_functions_specs(
            helper,
            model_to_use,
            allowed_plugins,
            disclosed_functions=tool_disclosure_state.get("disclosed"),
        )
    else:
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
        failed_tool_batch_signature=failed_tool_batch_signature,
        run_state=run_state,
        tool_disclosure_state=tool_disclosure_state,
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
    failed_tool_batch_signature=None,
    run_state=None,
    tool_disclosure_state=None,
):
    tool_calls = []
    try:
        artifact_manifest = list(artifact_manifest or [])
        if request_context is not None:
            chat_id = request_context.chat_id
            user_id = request_context.user_id
        run_state = _init_tool_run_state(helper, chat_id, user_id, request_context, model_to_use, run_state)

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
                failed_tool_batch_signature,
                run_state,
                tool_disclosure_state,
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
                logger.info(
                    "Received function-call response choice has_tool_calls=%s",
                    bool(getattr(first_choice.message, "tool_calls", None)),
                )
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

        current_batch_signature = _tool_batch_signature(tool_calls)
        if failed_tool_batch_signature and current_batch_signature == failed_tool_batch_signature:
            logger.warning(
                "Stopping repeated failing tool batch for chat_id=%s tools=%s",
                chat_id,
                [call.get("name") for call in tool_calls],
            )
            _record_tool_run_event(
                helper,
                run_state,
                event_type="stuck_tool_batch",
                iteration=times,
                status="blocked",
                tool_count=len(tool_calls),
                error_count=len(tool_calls),
                stop_reason="repeated_failing_tool_batch",
            )
            cleanup_skills = _cleanup_directives_for_blocked_result(helper, chat_id, user_id)
            return _stuck_tool_loop_error(tool_calls, cleanup_skills=cleanup_skills), tools_used

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
        tool_successes: list[bool] = []
        for call in tool_calls:
            tool_call_id = call.get("id")
            tool_name = call["name"]
            arguments = call["arguments"]
            logger.info(
                "Calling tool %s args_chars=%s",
                tool_name,
                len(arguments or ""),
            )
            if not helper.plugin_manager.is_function_allowed(tool_name, allowed_plugins):
                error = f'Tool {tool_name} is not allowed in the current chat mode'
                logger.warning(error)
                errors.append((tool_name, tool_call_id, json.dumps({'error': error}, ensure_ascii=False)))
                tool_successes.append(False)
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
                if tool_name == DISCOVER_TOOLS_NAME:
                    args["allowed_plugins"] = allowed_plugins
                if tool_name == RUN_SUBAGENTS_NAME and tool_disclosure_state is not None:
                    args["disclosed_functions"] = sorted(tool_disclosure_state.get("disclosed") or [])
                routing_error = _skill_script_routing_error(helper, chat_id, tool_name, args)
                if routing_error:
                    logger.warning("%s Tool=%s", routing_error.get("error"), tool_name)
                    errors.append((
                        tool_name,
                        tool_call_id,
                        json.dumps(routing_error, ensure_ascii=False),
                    ))
                    tool_successes.append(False)
                    continue
                arguments = json.dumps(args, ensure_ascii=False)
                prepared.append((tool_name, arguments, tool_call_id, _tool_metadata(helper, tool_name)))
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse arguments JSON for tool %s args_chars=%s",
                    tool_name,
                    len(arguments or ""),
                )
                errors.append((tool_name, tool_call_id, json.dumps({'error': f'Invalid arguments for {tool_name}'}, ensure_ascii=False)))
                tool_successes.append(False)
            except TypeError as exc:
                logger.error(f"Invalid arguments for {tool_name}: {exc}")
                errors.append((tool_name, tool_call_id, json.dumps({'error': f'Invalid arguments for {tool_name}'}, ensure_ascii=False)))
                tool_successes.append(False)

        semaphore = _tool_call_semaphore(helper)
        batch_started = time.monotonic()
        batch_parallel = all(_tool_parallelizable(metadata) for _, _, _, metadata in prepared)
        if batch_parallel:
            tasks = [
                _call_function_bounded(
                    helper,
                    name,
                    args,
                    request_context,
                    semaphore,
                    timeout_seconds=_tool_timeout_seconds(metadata),
                )
                for name, args, _, metadata in prepared
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for name, args, _, metadata in prepared:
                result = await _call_function_bounded(
                    helper,
                    name,
                    args,
                    request_context,
                    semaphore,
                    timeout_seconds=_tool_timeout_seconds(metadata),
                )
                results.append(result)
        batch_duration_ms = int((time.monotonic() - batch_started) * 1000)

        direct_results_collected: list = []
        seen_artifact_paths = {
            entry.get("path") for entry in artifact_manifest if isinstance(entry, dict)
        }
        seen_artifact_paths.discard(None)
        batch_artifact_count = 0
        compacted_bytes = 0
        for (tool_name, _, tool_call_id, metadata), tool_response in zip(prepared, results):
            if isinstance(tool_response, Exception):
                tool_response = json.dumps({'error': str(tool_response)}, ensure_ascii=False)
            tool_result = normalize_tool_result(tool_response, tool_name=tool_name, metadata=metadata)
            logger.info(
                "Function %s completed success=%s response_chars=%s",
                tool_name,
                tool_result.success,
                len(tool_result.content),
            )
            tool_success = tool_result.success
            tool_successes.append(tool_success)
            if tool_name == DISCOVER_TOOLS_NAME and tool_disclosure_state is not None:
                disclosed_tools = tool_result.payload.get("disclosed_tools") if isinstance(tool_result.payload, dict) else None
                if isinstance(disclosed_tools, list):
                    tool_disclosure_state.setdefault("disclosed", set()).update(
                        str(name) for name in disclosed_tools if isinstance(name, str)
                    )
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            if tool_success:
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
                    compacted_response, compacted_from = _compact_tool_response_for_history(tool_result.content)
                    compacted_bytes += compacted_from
                    add_tool_result(tool_name, compacted_response, tool_call_id)

        for tool_name, tool_call_id, tool_response in errors:
            if tool_name not in tools_used:
                tools_used += (tool_name,)
            add_tool_result(tool_name, tool_response, tool_call_id)

        success_count = sum(1 for success in tool_successes if success)
        error_count = len(tool_successes) - success_count
        if tool_successes:
            if error_count == 0:
                batch_status = "success"
            elif success_count:
                batch_status = "partial"
            else:
                batch_status = "error"
            _record_tool_run_event(
                helper,
                run_state,
                event_type="tool_batch",
                iteration=times,
                status=batch_status,
                tool_count=len(tool_calls),
                success_count=success_count,
                error_count=error_count,
                duration_ms=batch_duration_ms,
                metadata={
                    "tools": [call.get("name") for call in tool_calls],
                    "parallel": batch_parallel,
                    "compacted_from_chars": compacted_bytes,
                },
            )

        if direct_results_collected:
            _record_tool_run_event(
                helper,
                run_state,
                event_type="direct_result",
                iteration=times,
                status="success",
                tool_count=len(direct_results_collected),
                stop_reason="direct_result",
            )
            if len(direct_results_collected) == 1:
                return direct_results_collected[0], tools_used
            return _merge_direct_results_into_final(direct_results_collected), tools_used

        if batch_artifact_count:
            helper.conversations.setdefault(chat_id, []).append({
                "role": "user",
                "content": _artifact_manifest_message(artifact_manifest),
            })

        next_failed_tool_batch_signature = (
            current_batch_signature
            if tool_successes and len(tool_successes) == len(tool_calls) and not any(tool_successes)
            else None
        )

        sessions = await _list_user_sessions(helper.db, user_id, is_active=1)
        active_session = next((s for s in sessions if s['is_active']), None)
        session_id = active_session['session_id'] if active_session else None
        max_tokens_percent = active_session['max_tokens_percent'] if active_session else 80

        logger.info(
            "Function calls completed chat_id=%s messages_count=%s session_id=%s",
            chat_id,
            len(helper.conversations.get(chat_id, [])),
            session_id,
        )

        if tool_disclosure_state is not None:
            tools = helper.plugin_manager.get_functions_specs(
                helper,
                model_to_use,
                allowed_plugins,
                disclosed_functions=tool_disclosure_state.get("disclosed"),
            )
        else:
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
            next_failed_tool_batch_signature,
            run_state,
            tool_disclosure_state,
        )
    except Exception:
        logger.error('Error in function call handling', exc_info=True)
        raise
