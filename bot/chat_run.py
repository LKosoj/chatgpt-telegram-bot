from __future__ import annotations

import logging
from typing import Any

from .ai_events import AIRetry, AIRunEnd, event_to_log_dict
from .chat_response_utils import (
    aggregate_usage_split,
    finalize_chat_answer,
    response_has_message_text,
    response_prompt_completion_tokens,
    response_total_tokens,
)
from .session_logger import get_trace
from .utils import is_direct_result, log_exception_shape

logger = logging.getLogger(__name__)


class ChatRun:
    """Compatibility run shell for one non-stream chat turn.

    Routes completions through the provider/event adapter and keeps the tool
    loop on the provider-neutral response boundary.
    """

    def __init__(self, helper: Any):
        self.helper = helper

    def _record(self, event: Any) -> None:
        slog = getattr(self.helper, "session_logger", None)
        if slog is not None and get_trace() is not None:
            slog.record(event_to_log_dict(event))

    def _record_retry(self, message: str, *, stage: str) -> None:
        self._record(AIRetry(
            attempt=1,
            max_attempts=1,
            delay_seconds=0.0,
            message=message,
            data={"stage": stage},
        ))

    async def _retry_after_empty_response(
        self, *, chat_id, user_id, session_id, allowed_plugins, request_context,
        model_to_use, token_accumulator, usage_accumulator,
    ) -> tuple[Any | None, tuple]:
        """Retry once with tool specs after an empty model response.

        Returns (response, retry_plugins_used) from _handle_function_call
        when a retry response was available, or (None, ()) if
        _retry_empty_response_with_tools() itself returned nothing -- the
        caller must treat a None first element as "no retry happened" and
        leave its own `response` untouched.
        """
        retry_response = await self.helper._retry_empty_response_with_tools(
            chat_id, user_id, session_id, allowed_plugins, model_to_use=model_to_use,
        )
        if retry_response is None:
            return None, ()
        return await self.helper._handle_function_call(
            chat_id, retry_response, allowed_plugins=allowed_plugins, user_id=user_id,
            request_context=request_context, model_to_use=model_to_use,
            token_accumulator=token_accumulator, usage_accumulator=usage_accumulator,
        )

    async def run_non_stream(
        self,
        *,
        chat_id: int,
        query: str,
        session_id: str | None,
        user_id: int | None,
        request_context: Any,
        **kwargs: Any,
    ):
        helper = self.helper
        try:
            state_key = helper._chat_state_key(chat_id)
            # Reset per-request skills_agent gate flag so it doesn't persist
            # across user-initiated requests.
            helper._gate_fired.pop(state_key, None)
            plugins_used = ()
            response = await helper._common_get_chat_response(
                chat_id,
                query,
                session_id=session_id,
                user_id=user_id,
                **kwargs,
            )
            token_accumulator = []
            usage_accumulator: list[dict[str, int] | None] = []
            extra_tokens = helper._chat_request_extra_tokens.pop(state_key, 0)
            if extra_tokens:
                token_accumulator.append(extra_tokens)

            if helper.config["enable_functions"]:
                plugin_user_id = user_id or chat_id
                allowed_plugins = await helper.resolve_allowed_plugins(chat_id, session_id, plugin_user_id)
                response, plugins_used = await helper._handle_function_call(
                    chat_id,
                    response,
                    allowed_plugins=allowed_plugins,
                    user_id=user_id,
                    request_context=request_context,
                    model_to_use=helper._chat_request_models.get(state_key),
                    token_accumulator=token_accumulator,
                    usage_accumulator=usage_accumulator,
                )
                if is_direct_result(response):
                    logger.debug("Direct result returned, skipping further processing")
                    self._record(AIRunEnd(reason="direct_result"))
                    helper._chat_request_usage_split[state_key] = aggregate_usage_split(
                        token_accumulator, usage_accumulator,
                    )
                    return response, sum(token_accumulator)
                if plugins_used and not response_has_message_text(response):
                    self._record_retry(
                        "empty response after tool calls; retrying with tool specs",
                        stage="after_tool_calls_with_tools",
                    )
                    retry_response, retry_plugins_used = await self._retry_after_empty_response(
                        chat_id=chat_id,
                        user_id=user_id,
                        session_id=session_id,
                        allowed_plugins=allowed_plugins,
                        request_context=request_context,
                        model_to_use=helper._chat_request_models.get(state_key),
                        token_accumulator=token_accumulator,
                        usage_accumulator=usage_accumulator,
                    )
                    if retry_response is not None:
                        response = retry_response
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug("Direct result returned after empty response retry")
                            self._record(AIRunEnd(reason="direct_result_after_retry"))
                            helper._chat_request_usage_split[state_key] = aggregate_usage_split(
                                token_accumulator, usage_accumulator,
                            )
                            return response, sum(token_accumulator)
                    if not response_has_message_text(response):
                        self._record_retry(
                            "empty response after tool retry; retrying for final answer",
                            stage="after_tool_calls_final_answer",
                        )
                        response = await helper._retry_empty_response_after_tools(
                            chat_id,
                            user_id,
                            session_id,
                            model_to_use=helper._chat_request_models.get(state_key),
                        )
                        retry_tokens = response_total_tokens(response)
                        if retry_tokens:
                            token_accumulator.append(retry_tokens)
                        retry_split = response_prompt_completion_tokens(response)
                        if retry_split is not None:
                            usage_accumulator.append(retry_split)
                elif not plugins_used and not response_has_message_text(response):
                    self._record_retry(
                        "empty response before tool calls; retrying with tool specs",
                        stage="before_tool_calls_with_tools",
                    )
                    retry_response, retry_plugins_used = await self._retry_after_empty_response(
                        chat_id=chat_id,
                        user_id=user_id,
                        session_id=session_id,
                        allowed_plugins=allowed_plugins,
                        request_context=request_context,
                        model_to_use=helper._chat_request_models.get(state_key),
                        token_accumulator=token_accumulator,
                        usage_accumulator=usage_accumulator,
                    )
                    if retry_response is not None:
                        response = retry_response
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug("Direct result returned after empty response retry")
                            self._record(AIRunEnd(reason="direct_result_after_retry"))
                            helper._chat_request_usage_split[state_key] = aggregate_usage_split(
                                token_accumulator, usage_accumulator,
                            )
                            return response, sum(token_accumulator)
                        if retry_plugins_used and not response_has_message_text(response):
                            self._record_retry(
                                "empty response after retry tool calls; retrying for final answer",
                                stage="after_retry_tool_calls_final_answer",
                            )
                            response = await helper._retry_empty_response_after_tools(
                                chat_id,
                                user_id,
                                session_id,
                                model_to_use=helper._chat_request_models.get(state_key),
                            )
                            retry_tokens = response_total_tokens(response)
                            if retry_tokens:
                                token_accumulator.append(retry_tokens)
                            retry_split = response_prompt_completion_tokens(response)
                            if retry_split is not None:
                                usage_accumulator.append(retry_split)
            else:
                response_tokens = response_total_tokens(response)
                if response_tokens:
                    token_accumulator.append(response_tokens)
                response_split = response_prompt_completion_tokens(response)
                if response_split is not None:
                    usage_accumulator.append(response_split)

            answer, total_tokens = await finalize_chat_answer(
                helper,
                chat_id,
                response,
                plugins_used=plugins_used,
                token_accumulator=token_accumulator,
                session_id=session_id,
            )
            helper._chat_request_usage_split[state_key] = aggregate_usage_split(
                token_accumulator, usage_accumulator,
            )

            self._record(AIRunEnd(reason="completed"))
            return answer, total_tokens
        except Exception as e:
            helper._chat_request_extra_tokens.pop(helper._chat_state_key(chat_id), None)
            self._record(AIRunEnd(reason="error"))
            logger.error("Error in get_chat_response error=%s", log_exception_shape(e))
            raise
