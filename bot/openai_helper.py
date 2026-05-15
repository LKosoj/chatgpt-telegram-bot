from __future__ import annotations
import datetime
import logging
import os
import asyncio
import uuid
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime as dt

import tiktoken

import openai

from functools import lru_cache
import json
import httpx
import io
from calendar import monthrange
from PIL import Image
import yaml

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .tool_result import tool_result_content
from .utils import is_direct_result, encode_image, decode_image, escape_markdown
from .plugin_manager import PluginManager
from .database import Database
from .model_constants import (
    LLMGATEWAY_BIG_CONTEXT_MODEL,
    LLMGATEWAY_CHAT_MODELS,
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_IMAGE_GENERATION_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
    LLMGATEWAY_TRANSCRIPTION_MODEL,
    LLMGATEWAY_TTS_MODEL,
    GPT_4_VISION_MODELS,
    GPT_4O_MODELS,
    GPT_5_MODELS,
    O_MODELS,
    ANTHROPIC,
    GOOGLE,
    MISTRALAI,
    DEEPSEEK,
    LLAMA,
    PERPLEXITY,
    MOONSHOTAI,
    QWEN,
)
from .llm_gateway_client import LLMGatewayClient, extract_image_result
from .plugins.hooks import (
    BeforeChatRequestPayload,
    HookEvent,
    PromptFragmentPayload,
    SessionBeforeDeletePayload,
)
from .chat_modes_registry import ChatModesRegistry
from .validation import validate_openai_config
from .openai_tool_handler import handle_function_call
from .i18n import get_current_language, language_name, localized_text
from .user_settings import (
    USER_TTS_MODEL_SETTING,
    USER_TTS_VOICE_SETTING,
    get_user_settings,
)

logger = logging.getLogger(__name__)

EMPTY_MODEL_RESPONSE_ERROR = "Модель вернула пустой ответ"
VISION_MAX_ATTEMPTS = 3
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
RAW_TOOL_RESULT_RE = re.compile(r"^Function\s+[\w.\-]+\s+returned:\s*", re.IGNORECASE)
TTS_OPTIONS_CACHE_SECONDS = 300
_CHAT_LOCK_BYPASS_CHAT_ID = ContextVar("openai_helper_chat_lock_bypass_chat_id", default=None)


def _choice_message_text(choice) -> str:
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return ""
    content = THINK_BLOCK_RE.sub("", content)
    content = THINK_TAG_RE.sub("\n", content)
    content = content.strip()
    if RAW_TOOL_RESULT_RE.match(content):
        return ""
    return content


def _required_choice_message_text(choice) -> str:
    content = _choice_message_text(choice)
    if content:
        return content
    message = getattr(choice, "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    logger.warning(
        "Model returned empty assistant content; finish_reason=%s tool_call_count=%s",
        getattr(choice, "finish_reason", None),
        len(tool_calls) if tool_calls else 0,
    )
    raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)


def _response_has_message_text(response) -> bool:
    return any(_choice_message_text(choice) for choice in getattr(response, "choices", []) or [])


def _first_choice_or_raise(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning("Model response has no choices")
        raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)
    return choices[0]

REPLY_INTENT_CLASSIFIER_PROMPT = """Classify a Telegram user's reply intent.
Return only JSON in this exact shape: {"intent":"<one of: image_edit, image_describe, text_reply>"}

Definitions:
- image_edit: the user wants to transform, modify, add to, remove from, redraw, or restyle an image.
- image_describe: the user asks about visual content in an image, including description, analysis, identification, or questions about what is shown.
- text_reply: the user asks or clarifies something about text, conversation context, or anything that should continue as normal chat.

Use replied_message_kind as the source context. Do not choose image_edit or image_describe unless the user intent depends on an image.
If replied_message_kind is "image" and the user asks to transform the image (add, remove, replace, restyle, redraw something in it), classify as image_edit.
If replied_message_kind is "image" and the user asks about the visual content (description, identification, analysis), classify as image_describe.
Otherwise classify as text_reply."""


@lru_cache(maxsize=128)
def default_max_tokens(model: str = None) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    if model == LLMGATEWAY_BIG_CONTEXT_MODEL:
        return 1_000_000
    if model == LLMGATEWAY_LIGHT_MODEL:
        return 128_000
    if model == LLMGATEWAY_HIGH_MODEL:
        return 256_000
    return 200_000


@lru_cache(maxsize=128)
def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    return model in LLMGATEWAY_CHAT_MODELS


# Per-chat state held by OpenAIHelper. Exists as documentation-as-code:
# the four mappings (conversations, conversations_vision, last_updated,
# loaded_conversation_sessions, last_image_file_ids) all key on chat_id and
# must be reset together — see _clear_chat_state.
@dataclass
class ChatState:
    messages: list
    vision: bool = False
    last_updated: datetime.datetime | None = None
    session_id: str | None = None
    last_image_file_id: str | None = None


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager, db: Database):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        :param db: Database instance
        """
        # http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        http_client = httpx.AsyncClient()
        
        if config['openai_base'] != '' :
            openai.api_base = config['openai_base']
        self.api_key = config['api_key']
        client_kwargs = {
            "api_key": config["api_key"],
            "http_client": http_client,
            "timeout": 300.0,
            "max_retries": 3,
        }
        if config["openai_base"]:
            client_kwargs["base_url"] = config["openai_base"]
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.gateway_client = LLMGatewayClient(config.get("openai_base", ""), config["api_key"])
        validate_openai_config(config)
        self.config = dict(config)
        self.plugin_manager = plugin_manager
        self.db = db
        self.conversations: dict[int, list] = {}  # {chat_id: history}
        self.loaded_conversation_sessions: dict[int, str | None] = {}  # {chat_id: session_id}
        self.conversations_vision: dict[int, bool] = {}  # {chat_id: is_vision}
        self._chat_request_models: dict[int, str] = {}
        self.last_updated: dict[int, datetime.datetime] = {}  # {chat_id: last_update_timestamp}
        self.last_image_file_ids = {}
        self._tts_models_cache: tuple[float, list[str]] | None = None
        self._tts_voices_cache: dict[str, tuple[float, list[str]]] = {}
        self.bot = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.chat_modes_registry = ChatModesRegistry(os.path.join(current_dir, 'chat_modes.yml'))
        self.chat_modes_registry.validate_tools(self.plugin_manager)

        # Set default values for optional configuration
        self.config.setdefault('temperature', 0.7)
        self.config.setdefault('presence_penalty', 0.0)
        self.config.setdefault('frequency_penalty', 0.0)
        self.config.setdefault('vision_detail', 'auto')
        self.config.setdefault('n_choices', 1)
        self.config.setdefault('light_model', LLMGATEWAY_LIGHT_MODEL)
        self.config.setdefault('big_model_to_use', LLMGATEWAY_BIG_CONTEXT_MODEL)
        self.config.setdefault('tts_model', LLMGATEWAY_TTS_MODEL)
        self.config.setdefault('tts_voice', 'kseniya')
        self.config.setdefault('tts_response_format', 'wav')
        self.config.setdefault('transcription_model', LLMGATEWAY_TRANSCRIPTION_MODEL)

    async def classify_reply_intent(self, user_text: str, replied_message_kind: str) -> str:
        """
        Classifies a Telegram reply intent using the light model.
        """
        messages = [
            {"role": "system", "content": REPLY_INTENT_CLASSIFIER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "replied_message_kind": replied_message_kind,
                        "user_reply": user_text,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL),
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
            stream=False,
            extra_headers={ "X-Title": "tgBot" },
        )
        content = _first_choice_or_raise(response).message.content or ""
        allowed_intents = {"image_edit", "image_describe", "text_reply"}
        intent_aliases = {
            "image_description": "image_describe",
            "describe_image": "image_describe",
            "image_question": "image_describe",
            "vision": "image_describe",
            "edit_image": "image_edit",
            "image_editing": "image_edit",
            "image_modify": "image_edit",
            "modify_image": "image_edit",
            "redraw_image": "image_edit",
            "image_redraw": "image_edit",
            "normal_chat": "text_reply",
            "text": "text_reply",
        }
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start:end + 1])
                intent = str(data.get("intent", "")).strip().lower()
                if intent in allowed_intents:
                    return intent
                if intent in intent_aliases:
                    return intent_aliases[intent]
            except json.JSONDecodeError:
                pass

        normalized_content = content.strip().lower()
        for intent in allowed_intents:
            if intent in normalized_content:
                return intent
        for alias, intent in intent_aliases.items():
            if alias in normalized_content:
                return intent
        logger.info("Reply intent classifier returned unrecognized content: %r", content[:200])
        return "unknown"

    async def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            await self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list,
        tools: list | None = None,
        tool_choice=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        json_mode: bool = False,
        response_format: dict | None = None,
        extra_headers: dict | None = None,
        **extra,
    ):
        """Low-level chat completion. Returns the raw SDK response object.

        Does not mutate history, does not synthesize messages, does not pick
        the model. For high-level chat interactions use ``ask`` or
        ``get_chat_response`` instead.
        """
        kwargs: dict = {"model": model, "messages": messages, "stream": stream}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_headers"] = (
            extra_headers if extra_headers is not None else {"X-Title": "tgBot"}
        )
        if extra:
            kwargs.update(extra)
        return await self.client.chat.completions.create(**kwargs)

    async def _save_conversation_context(
        self,
        chat_id: int,
        context: dict[str, Any],
        parse_mode: str,
        temperature: float,
        max_tokens_percent: int,
        session_id: str | None,
    ) -> str | None:
        save_async = getattr(self.db, "save_conversation_context_async", None)
        if callable(save_async):
            return await save_async(
                chat_id,
                context,
                parse_mode,
                temperature,
                max_tokens_percent,
                session_id,
                self,
            )
        return self.db.save_conversation_context(
            chat_id,
            context,
            parse_mode,
            temperature,
            max_tokens_percent,
            session_id,
            self,
        )

    async def ask(self, prompt, user_id, assistant_prompt=None, model=None, json_mode: bool = False):
        """
        Send a prompt to OpenAI and get a response.
        """
        model_to_use = model or self.get_current_model(user_id)
        try:
            # Проверяем, инициализирован ли контекст разговора
            if user_id not in self.conversations:
                logger.info(f'Initializing conversation context for user_id={user_id}')
                # Пытаемся загрузить контекст из базы данных
                saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(user_id, None)

                if saved_context and 'messages' in saved_context:
                    # Если есть сохраненный контекст в БД, используем его
                    self.conversations[user_id] = saved_context['messages']
                else:
                    # Если нет контекста в БД, начинаем новый чат
                    await self.reset_chat_history(user_id, session_id=None)

            # Инициализируем conversations_vision для пользователя, если его нет
            if user_id not in self.conversations_vision:
                self.conversations_vision[user_id] = False

            add_prompt1 = f" Текущая дата и время: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
            if assistant_prompt is None:
                assistant_prompt = "Ты помошник, который отвечает на вопросы пользователя. Ты должен использовать все свои знания и навыки для того, чтобы помочь пользователю. " + add_prompt1

            if model:
                model_to_use = model
            else:
                model_to_use = self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL)
            logger.info(f"Используемая модель: {model_to_use}")

            messages = [
                {"role": "system", "content": assistant_prompt},
                {"role": "user", "content": prompt}
            ]

            await self.__add_to_history(user_id, role="user", content=prompt)
            response = await self.chat_completion(
                model=model_to_use,
                messages=messages,
                max_tokens=self.get_max_tokens(model_to_use, 60, user_id),
                temperature=0.6,
                stream=False,
                json_mode=json_mode,
            )
            content = _required_choice_message_text(_first_choice_or_raise(response))
            await self.__add_to_history(user_id, role="assistant", content=content)
            return content, response.usage.total_tokens
        except Exception as e:
            logger.error(f'Error in ask method: {str(e)}', exc_info=True)
            raise

    async def get_chat_response(
        self,
        chat_id: int,
        query: str,
        request_id: str = None,
        session_id: str = None,
        user_id: int = None,
        request_context=None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Gets a full response from the GPT model with optional session support.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param request_id: Optional request identifier
        :param session_id: Optional session identifier
        :param **kwargs: Additional keyword arguments
        :return: The answer from the model and the number of tokens used
        """
        if request_context is not None:
            chat_id = request_context.chat_id
            user_id = request_context.user_id
            if session_id is None:
                session_id = request_context.session_id

        if self._chat_lock_bypass_enabled(chat_id):
            return await self._get_chat_response_locked(
                chat_id=chat_id,
                query=query,
                session_id=session_id,
                user_id=user_id,
                request_context=request_context,
                **kwargs,
            )

        lock = await self._chat_lock(chat_id)
        async with lock:
            return await self._get_chat_response_locked(
                chat_id=chat_id,
                query=query,
                session_id=session_id,
                user_id=user_id,
                request_context=request_context,
                **kwargs,
            )

    async def _get_chat_response_locked(
        self,
        chat_id,
        query,
        session_id,
        user_id,
        request_context,
        **kwargs,
    ):
        try:
            # Add the last image file ID to the context if available
            if chat_id in self.last_image_file_ids:
                # The model can now access this through the function calls
                self.last_image_file_id = self.last_image_file_ids[chat_id]
            plugins_used = ()
            # Вызов с учетом возможного отсутствия session_id
            response = await self.__common_get_chat_response(
                chat_id,
                query,
                session_id=session_id,
                user_id=user_id,
                **kwargs
            )
            
            if self.config['enable_functions']:
                allowed_plugins = await self.resolve_allowed_plugins(chat_id, session_id, user_id)
                response, plugins_used = await self.__handle_function_call(
                    chat_id,
                    response,
                    allowed_plugins=allowed_plugins,
                    user_id=user_id,
                    request_context=request_context,
                    model_to_use=self._chat_request_models.get(chat_id),
                    retry_plain_text_tool_intent=self.conversations_vision.get(chat_id, False),
                )
                if is_direct_result(response):
                    logger.debug('Direct result returned, skipping further processing')
                    return response, '0'
                if plugins_used and not _response_has_message_text(response):
                    retry_response = await self._retry_empty_response_with_tools(
                        chat_id,
                        user_id,
                        session_id,
                        allowed_plugins,
                        model_to_use=self._chat_request_models.get(chat_id),
                    )
                    if retry_response is not None:
                        response, retry_plugins_used = await self.__handle_function_call(
                            chat_id,
                            retry_response,
                            allowed_plugins=allowed_plugins,
                            user_id=user_id,
                            request_context=request_context,
                            model_to_use=self._chat_request_models.get(chat_id),
                        )
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug('Direct result returned after empty response retry')
                            return response, '0'
                    if not _response_has_message_text(response):
                        response = await self._retry_empty_response_after_tools(
                            chat_id,
                            user_id,
                            session_id,
                            model_to_use=self._chat_request_models.get(chat_id),
                        )
                elif not plugins_used and not _response_has_message_text(response):
                    retry_response = await self._retry_empty_response_with_tools(
                        chat_id,
                        user_id,
                        session_id,
                        allowed_plugins,
                        model_to_use=self._chat_request_models.get(chat_id),
                    )
                    if retry_response is not None:
                        response, retry_plugins_used = await self.__handle_function_call(
                            chat_id,
                            retry_response,
                            allowed_plugins=allowed_plugins,
                            user_id=user_id,
                            request_context=request_context,
                            model_to_use=self._chat_request_models.get(chat_id),
                        )
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug('Direct result returned after empty response retry')
                            return response, '0'
                        if retry_plugins_used and not _response_has_message_text(response):
                            response = await self._retry_empty_response_after_tools(
                                chat_id,
                                user_id,
                                session_id,
                                model_to_use=self._chat_request_models.get(chat_id),
                            )

            answer = ''

            if len(response.choices) > 1 and self.config['n_choices'] > 1:
                for index, choice in enumerate(response.choices):
                    content = _required_choice_message_text(choice)
                    if index == 0:
                        await self.__add_to_history(chat_id, role="assistant", content=content, session_id=session_id)
                    answer += f'{index + 1}\u20e3\n'
                    answer += content
                    answer += '\n\n'
            else:
                answer = _required_choice_message_text(_first_choice_or_raise(response))
                await self.__add_to_history(chat_id, role="assistant", content=answer, session_id=session_id)

            bot_language = self.config['bot_language']
            show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
            plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
            if self.config['show_usage']:
                answer += "\n\n---\n" \
                        f"💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}" \
                        f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                        f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
                if show_plugins_used:
                    answer += f"\n🔌 {', '.join(plugin_names)}"
            elif show_plugins_used:
                answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

            return answer, response.usage.total_tokens
        except Exception as e:
            logger.error(f'Error in get_chat_response: {str(e)}', exc_info=True)
            raise
        finally:
            # Clean up after response is generated
            if hasattr(self, 'last_image_file_id'):
                delattr(self, 'last_image_file_id')

    async def get_chat_response_stream(
        self,
        chat_id: int,
        query: str,
        request_id: str = None,
        session_id: str = None,
        user_id: int = None,
        request_context=None,
    ):
        """
        Stream response from the GPT model with optional session support.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param request_id: Optional request identifier
        :param session_id: Optional session identifier
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        if request_context is not None:
            chat_id = request_context.chat_id
            user_id = request_context.user_id
            if session_id is None:
                session_id = request_context.session_id

        if self._chat_lock_bypass_enabled(chat_id):
            async for chunk in self._get_chat_response_stream_locked(
                chat_id=chat_id,
                query=query,
                session_id=session_id,
                user_id=user_id,
                request_context=request_context,
            ):
                yield chunk
            return

        lock = await self._chat_lock(chat_id)
        async with lock:
            async for chunk in self._get_chat_response_stream_locked(
                chat_id=chat_id,
                query=query,
                session_id=session_id,
                user_id=user_id,
                request_context=request_context,
            ):
                yield chunk

    async def _get_chat_response_stream_locked(
        self,
        chat_id,
        query,
        session_id,
        user_id,
        request_context,
    ):
        plugins_used = ()
        try:
            logger.info(f'Starting chat response stream for chat_id={chat_id}')

            # Проверяем, инициализирован ли контекст разговора
            saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
            loaded_session_id = self.loaded_conversation_sessions.get(chat_id)
            session_changed = chat_id in self.loaded_conversation_sessions and loaded_session_id != session_id
            if chat_id not in self.conversations or session_changed:
                logger.info(f'Initializing conversation context for chat_id={chat_id}')
                
                if saved_context and 'messages' in saved_context:
                    # Если есть сохраненный контекст в БД, используем его
                    self.conversations[chat_id] = saved_context['messages']
                else:
                    # Если нет контекста в БД, начинаем новый чат
                    await self.reset_chat_history(chat_id, session_id=session_id)
                self.loaded_conversation_sessions[chat_id] = session_id

            # Инициализируем conversations_vision для чата, если его нет
            if chat_id not in self.conversations_vision:
                self.conversations_vision[chat_id] = False

            logger.info('Getting chat response from model')
            try:
                # Вызов с учетом возможного отсутствия session_id
                response = await self.__common_get_chat_response(
                    chat_id, 
                    query, 
                    stream=True, 
                    session_id=session_id,
                    user_id=user_id
                )
            except Exception as e:
                logger.error(f'Error getting chat response: {str(e)}')
                yield f"Error: {str(e)}", '0'
                return

            if self.config['enable_functions']:
                try:
                    allowed_plugins = await self.resolve_allowed_plugins(chat_id, session_id, user_id)
                    response, plugins_used = await self.__handle_function_call(
                        chat_id,
                        response,
                        stream=True,
                        allowed_plugins=allowed_plugins,
                        user_id=user_id,
                        request_context=request_context,
                        model_to_use=self._chat_request_models.get(chat_id),
                        retry_plain_text_tool_intent=self.conversations_vision.get(chat_id, False),
                    )
                    if is_direct_result(response):
                        yield response, '0'
                        return
                except Exception as e:
                    logger.error(f'Error in function call: {str(e)}')
                    yield f"Error in function call: {str(e)}", '0'
                    return

            answer = ''
                
            try:
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        answer += delta.content
                        yield answer, 'not_finished'
            except Exception as e:
                logger.error(f'Error processing response stream: {str(e)}')
                if answer:
                    yield answer, 'not_finished'
                return

            answer = answer.strip()
            if not answer:
                raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)
            await self.__add_to_history(chat_id, role="assistant", content=answer, session_id=session_id)
            tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

            show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
            plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
            if self.config['show_usage']:
                answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
                if show_plugins_used:
                    answer += f"\n🔌 {', '.join(plugin_names)}"
            elif show_plugins_used:
                answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

            yield answer, tokens_used

        except Exception as e:
            logger.error(f"Error in chat response stream: {e}", exc_info=True)
            # Yield an error message or handle it gracefully
            yield f"Error generating response: {str(e)}", '0'

    def _apply_user_disabled_plugins(self, allowed_plugins, user_id: int | None):
        disabled_plugins = self.plugin_manager.disabled_plugins_for_user(user_id)
        if not disabled_plugins:
            return allowed_plugins
        if allowed_plugins == ['All']:
            return [
                plugin_name
                for plugin_name in self.plugin_manager.plugins.keys()
                if plugin_name not in disabled_plugins
            ]
        return [plugin_name for plugin_name in allowed_plugins if plugin_name not in disabled_plugins]

    async def resolve_allowed_plugins(self, chat_id: int, session_id: str | None = None, user_id: int | None = None):
        saved_context, _, _, _, _ = self.db.get_conversation_context(chat_id, session_id)
        allowed_plugins = ['All']

        if saved_context and 'messages' in saved_context:
            system_message = next(
                (msg for msg in saved_context['messages'] if msg.get('role') == 'system'),
                None
            )

            if not system_message:
                logger.warning(
                    f'System message not found in context for chat_id {chat_id}, '
                    f'session {session_id}. Resetting session.'
                )
                await self.reset_chat_history(chat_id, '', session_id)
                saved_context, _, _, _, _ = self.db.get_conversation_context(chat_id, session_id)
                if saved_context and 'messages' in saved_context:
                    system_message = next(
                        (msg for msg in saved_context['messages'] if msg.get('role') == 'system'),
                        None
                    )

            current_mode = self._mode_from_system_message(system_message)

            if current_mode and 'tools' in current_mode:
                allowed_plugins = current_mode['tools']

        allowed_plugins = self._apply_user_disabled_plugins(allowed_plugins, user_id)
        return self.plugin_manager.filter_allowed_plugins(allowed_plugins)

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False, session_id=None, **kwargs):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param **kwargs: Additional keyword arguments
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        big_context = kwargs.get('big_context', False)
        try:
            logger.info(f'Generating chat response (chat_id={chat_id}, stream={stream})')
            logger.debug(f'Query: {query}')
            
            # Пытаемся загрузить контекст из базы данных
            saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)

            loaded_session_id = self.loaded_conversation_sessions.get(chat_id)
            session_changed = chat_id in self.loaded_conversation_sessions and loaded_session_id != session_id
            if chat_id not in self.conversations or self.__max_age_reached(chat_id) or session_changed:
                if saved_context and 'messages' in saved_context:
                    # Если есть сохраненный контекст в БД, используем его
                    self.conversations[chat_id] = saved_context['messages']
                else:
                    # Если нет контекста в БД, начинаем новый чат
                    await self.reset_chat_history(chat_id, session_id=session_id)
                self.loaded_conversation_sessions[chat_id] = session_id

            # Инициализируем conversations_vision для чата, если его нет
            if chat_id not in self.conversations_vision:
                self.conversations_vision[chat_id] = False

            # Проверяем, является ли это первым сообщением в сессии, если да, то определяем режим работы
            user_messages = [msg for msg in self.conversations[chat_id] if msg['role'] == 'user']
            model_to_use = self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL)
            if len(user_messages) == 0 and self.config['auto_chat_modes']:
                auto_mode_prompt = await self._build_auto_chat_mode_prompt(
                    query, chat_id=chat_id, user_id=kwargs.get('user_id')
                )
                response = await self.chat_completion(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты маршрутизатор режимов работы. Возвращай только ключ режима без пояснений.",
                        },
                        {"role": "user", "content": auto_mode_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=self.get_max_tokens(model_to_use, 20, chat_id),
                    stream=False,
                )
                mode_name = _required_choice_message_text(_first_choice_or_raise(response))
                logger.info(f"🎯 Определен режим для первого сообщения: {mode_name}")
                
                # Ищем режим по имени
                mode_key = mode_name.strip().lower()
                mode_data = self.chat_modes_registry.get_mode_by_key(mode_key)
                if mode_data:
                    # Обновляем системное сообщение
                    new_system_prompt = mode_data.get('prompt_start', '')
                    if new_system_prompt:
                        # Проверяем, что история не пуста
                        if not self.conversations[chat_id]:
                            logger.warning(f'Conversation history is empty for chat_id {chat_id}. Initializing with system message.')
                            self.conversations[chat_id] = [{"role": "system", "content": new_system_prompt, "mode_key": mode_key}]
                        else:
                            # Заменяем системное сообщение в истории
                            self.conversations[chat_id][0]['role'] = 'system'
                            self.conversations[chat_id][0]['content'] = new_system_prompt
                            self.conversations[chat_id][0]['mode_key'] = mode_key
                        logger.info(f"🔄 Режим работы изменен на: {mode_key}")
                        
                        # Сохраняем обновленный контекст
                        await self._save_conversation_context(
                            chat_id,
                            {'messages': self.conversations[chat_id]},
                            parse_mode,
                            temperature,
                            max_tokens_percent,
                            session_id,
                        )
            
            memory_user_id = kwargs.get('user_id') or chat_id

            self.last_updated[chat_id] = datetime.datetime.now()

            await self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)

            user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
            if self.conversations_vision.get(chat_id, False):
                model_to_use = self.config['vision_model']
            else:
                model_to_use = self.get_current_model(memory_user_id or user_id)

            # Рассчитываем максимальное количество токенов с учетом процента
            max_tokens = self.get_max_tokens(model_to_use, max_tokens_percent, chat_id)
            logger.info(f"Model: {model_to_use}, max_tokens: {max_tokens}, max_tokens_percent: {max_tokens_percent}")

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id], model_to_use)
            exceeded_max_tokens = token_count + max_tokens > default_max_tokens(model_to_use) * 0.95
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logger.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1], user_id, session_id)
                    logger.debug(f'Summary: {summary}')
                    # Pre-dispatch: let session-memory subscribers snapshot the
                    # outgoing window before we wipe it via reset_chat_history.
                    await self._dispatch_before_summarise_reset(
                        chat_id=chat_id,
                        user_id=memory_user_id,
                        session_id=session_id,
                    )
                    await self.reset_chat_history(
                        chat_id,
                        self.conversations[chat_id][0]['content'],
                        session_id,
                        prune_old_sessions=False,
                    )
                    await self.__add_to_history(chat_id, role="assistant", content=summary, session_id=session_id)
                    await self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)
                    token_count = self.__count_tokens(self.conversations[chat_id], model_to_use)
                except Exception as e:
                    logger.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            logger.info(f"Model: {model_to_use}")

            # if max_tokens + token_count + 10000 > default_max_tokens(model_to_use):
            #     max_tokens = default_max_tokens(model_to_use) - token_count - 10000
            # Если token_count больше max_tokens или big_context, используем модель из переменной BIG_MODEL_TO_USE
            if (token_count > max_tokens or big_context) and self.config['big_model_to_use']:
                model_to_use = self.config['big_model_to_use']
                max_tokens = self.get_max_tokens(model_to_use, max_tokens_percent, chat_id)
            self._chat_request_models[chat_id] = model_to_use

            messages = await self._apply_before_chat_request_mutators(
                chat_id=chat_id,
                user_id=memory_user_id,
                session_id=session_id,
                request_id=kwargs.get('request_id'),
                parse_mode=parse_mode,
                temperature=temperature,
                max_tokens_percent=max_tokens_percent,
                persist=True,
            )
            common_args = {
                'model': model_to_use, #if not self.conversations_vision[chat_id] else self.config['vision_model'],
                'messages': messages,
                'temperature': temperature,
                'n': 1, # several choices is not implemented yet
                'max_tokens': max_tokens,
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }

            if model_to_use in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + PERPLEXITY + MOONSHOTAI + QWEN):
                stream = False

                #common_args['messages'] = [msg for msg in common_args['messages'] if msg['role'] != 'system']
                common_args['max_completion_tokens'] = max_tokens # o1 series only supports max_completion_tokens
                #common_args['max_tokens'] = max_tokens
         
                # 'temperature', 'top_p', 'n', 'presence_penalty', 'frequency_penalty' are currently fixed and cannot be changed
            else:
                # Parameters for other models
                common_args.update({
                    'temperature': temperature,
                    'n': self.config['n_choices'],
                    'max_tokens': max_tokens,
                    'presence_penalty': self.config['presence_penalty'],
                    'frequency_penalty': self.config['frequency_penalty'],
                    'stream': stream,
                    'extra_headers': { "X-Title": "tgBot" },
                })

            if self.config['enable_functions']:
                allowed_plugins = await self.resolve_allowed_plugins(chat_id, session_id, memory_user_id)
                tools = self.plugin_manager.get_functions_specs(self, model_to_use, allowed_plugins)
                
                if tools and model_to_use not in (O_MODELS + GOOGLE + PERPLEXITY):
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'

            log_args = {
                key: (f"<{len(value)} messages>" if key == "messages" and isinstance(value, list) else value)
                for key, value in common_args.items()
            }
            logger.info(f"common_args = {json.dumps(log_args, ensure_ascii=False)}")
            response = await self.client.chat.completions.create(**common_args)
            
            if stream:
                # For streaming responses, return the stream directly
                return response
            else:
                # For non-streaming responses, log the number of choices and return the response
                logger.debug(f'_______________________Response choices: {len(response.choices)}')
                return response
    
        except openai.RateLimitError as e:
            logger.warning(f'Rate limit error: {str(e)}')
            raise e

        except openai.BadRequestError as e:
            logger.error(f'Bad request error: {str(e)}')
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{error_message}") from e

        except ValueError as e:
            logger.error(f'Configuration error: {str(e)}')
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ Configuration error: {error_message}") from e

        except Exception as e:
            logger.error(f'Unexpected error in chat response generation: {str(e)}', exc_info=True)
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e

    async def __handle_function_call(
        self,
        chat_id,
        response,
        stream=False,
        times=0,
        tools_used=(),
        allowed_plugins=None,
        user_id=None,
        request_context=None,
        model_to_use=None,
        retry_plain_text_tool_intent=False,
    ):
        return await handle_function_call(
            self,
            chat_id,
            response,
            stream,
            times,
            tools_used,
            allowed_plugins,
            user_id,
            request_context,
            model_to_use=model_to_use,
            retry_plain_text_tool_intent=retry_plain_text_tool_intent,
        )

    def _uses_structured_tool_history(self, model_to_use: str) -> bool:
        return model_to_use in (GPT_4O_MODELS + LLMGATEWAY_CHAT_MODELS)

    def _tool_result_content(self, content: Any) -> str:
        return tool_result_content(content)

    def _add_assistant_tool_calls_to_history(self, chat_id: int, tool_calls: list[dict[str, Any]]) -> None:
        self.conversations[chat_id].append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call.get("arguments") or "{}",
                    },
                }
                for call in tool_calls
            ],
        })

    def _mode_from_system_message(self, system_message: dict[str, Any] | None) -> dict[str, Any] | None:
        if not system_message:
            return None

        mode_key = system_message.get("mode_key")
        get_mode_by_key = getattr(self.chat_modes_registry, "get_mode_by_key", None)
        if mode_key and callable(get_mode_by_key):
            current_mode = get_mode_by_key(mode_key)
            if current_mode:
                return current_mode

        content = system_message.get("content", "")
        current_mode = self.chat_modes_registry.get_mode_by_system_prompt(content)
        if current_mode:
            return current_mode

        return None

    def _defer_direct_tool_results(self, chat_id: int) -> bool:
        messages = self.conversations.get(chat_id, [])
        system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
        current_mode = self._mode_from_system_message(system_message)
        return bool(current_mode and current_mode.get("defer_direct_results"))

    def _add_function_call_to_history(
        self,
        chat_id: int,
        function_name: str,
        content: str,
        tool_call_id: str | None = None,
    ) -> None:
        self.__add_function_call_to_history(
            chat_id=chat_id,
            function_name=function_name,
            content=content,
            tool_call_id=tool_call_id,
        )

    def _localized_text(self, key, bot_language):
        return localized_text(key, bot_language)

    async def _retry_empty_response_after_tools(
        self,
        chat_id: int,
        user_id: int | None,
        session_id: str | None,
        model_to_use: str | None = None,
    ):
        logger.warning(
            "Retrying empty model response after tool calls for chat_id=%s user_id=%s session_id=%s",
            chat_id,
            user_id,
            session_id,
        )
        model_to_use = model_to_use or self.get_current_model(user_id)
        session_owner = user_id if user_id is not None else chat_id
        max_tokens_percent = 80
        try:
            list_async = getattr(self.db, "list_user_sessions_async", None)
            if list_async is not None:
                sessions = await list_async(session_owner, is_active=1)
            else:
                sessions = self.db.list_user_sessions(session_owner, is_active=1)
            active_session = next((s for s in sessions if s['is_active']), None)
            if active_session:
                max_tokens_percent = active_session.get('max_tokens_percent') or max_tokens_percent
        except Exception as exc:
            logger.warning("Failed to read active session for empty response retry: %s", exc)

        max_tokens = self.get_max_tokens(model_to_use, max_tokens_percent, chat_id)
        messages = await self._apply_before_chat_request_mutators(
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
            request_id=None,
            persist=False,
        )
        messages = list(messages)
        messages.append({
            "role": "user",
            "content": (
                "Инструменты уже выполнены. Верните непустой финальный ответ пользователю: "
                "результат, краткий статус или конкретный вопрос для продолжения. "
                "Не вызывайте tools в этом ответе."
            ),
        })
        common_args = {
            'model': model_to_use,
            'messages': messages,
            'temperature': self.config['temperature'],
            'n': 1,
            'max_tokens': max_tokens,
            'presence_penalty': self.config['presence_penalty'],
            'frequency_penalty': self.config['frequency_penalty'],
            'stream': False,
            'extra_headers': { "X-Title": "tgBot" },
        }
        if model_to_use in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + PERPLEXITY + MOONSHOTAI + QWEN):
            common_args['max_completion_tokens'] = max_tokens
        return await self._create_empty_response_retry_completion("after_tools", **common_args)

    async def _retry_empty_response_with_tools(
        self,
        chat_id: int,
        user_id: int | None,
        session_id: str | None,
        allowed_plugins,
        model_to_use: str | None = None,
    ):
        model_to_use = model_to_use or self.get_current_model(user_id)
        if model_to_use in (O_MODELS + GOOGLE + PERPLEXITY):
            return None

        tools = self.plugin_manager.get_functions_specs(self, model_to_use, allowed_plugins)
        if not tools:
            return None

        logger.warning(
            "Retrying empty model response with tools for chat_id=%s user_id=%s session_id=%s",
            chat_id,
            user_id,
            session_id,
        )
        max_tokens = self.get_max_tokens(model_to_use, 80, chat_id)
        messages = await self._apply_before_chat_request_mutators(
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
            request_id=None,
            persist=False,
        )
        messages = list(messages)
        messages.append({
            "role": "user",
            "content": (
                "Предыдущий ответ был пустым. Продолжите выполнение задачи: "
                "если нужен инструмент, вызовите подходящий tool; если инструмент не нужен, "
                "верните непустой ответ пользователю."
            ),
        })
        return await self._create_empty_response_retry_completion(
            "with_tools",
            model=model_to_use,
            messages=messages,
            tools=tools,
            tool_choice='auto',
            temperature=self.config['temperature'],
            n=1,
            max_tokens=max_tokens,
            presence_penalty=self.config['presence_penalty'],
            frequency_penalty=self.config['frequency_penalty'],
            stream=False,
            extra_headers={ "X-Title": "tgBot" },
        )

    async def _create_empty_response_retry_completion(self, retry_kind: str, **kwargs):
        logger.warning(
            "Empty response retry request started kind=%s max_tokens=%s tools=%s",
            retry_kind,
            kwargs.get('max_tokens') or kwargs.get('max_completion_tokens'),
            bool(kwargs.get('tools')),
        )
        response = await self.client.chat.completions.create(**kwargs)
        logger.warning(
            "Empty response retry request finished kind=%s choices=%s",
            retry_kind,
            len(getattr(response, "choices", []) or []),
        )
        return response

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Generates an image from the given prompt using DALL·E model.
        :param prompt: The prompt to send to the model
        :return: The image URL and the image size
        """
        bot_language = self.config['bot_language']
        try:
            image_args = {
                "prompt": prompt,
                "n": 1,
                "model": self.config.get("image_model", LLMGATEWAY_IMAGE_GENERATION_MODEL),
                "size": self.config["image_size"],
                "extra_headers": { "X-Title": "tgBot" },
            }
            if not str(image_args["model"]).startswith("llmgateway/"):
                image_args["quality"] = self.config["image_quality"]
                image_args["style"] = self.config["image_style"]

            response = await self.client.images.generate(**image_args)

            if len(response.data) == 0:
                logger.error(f'No response from GPT: {str(response)}')
                raise Exception(
                    f"⚠️ _{localized_text('error', bot_language)}._ "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            image_value, _image_format = extract_image_result(response)
            return image_value, self.config['image_size']
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def edit_telegram_image(self, prompt: str, file_id: str) -> tuple[str, str]:
        """
        Edits a Telegram image by downloading it and sending it to the LLMGateway image edit endpoint.
        """
        bot_language = self.config['bot_language']
        try:
            image_bytes = await self.download_file_as_bytes(file_id)
            response = await self.gateway_client.image_edit_file(
                prompt,
                image_bytes,
                model=self.config.get("image_model", LLMGATEWAY_IMAGE_GENERATION_MODEL),
            )
            return extract_image_result(response)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    @staticmethod
    def _cache_is_fresh(cache_entry: tuple[float, list[str]] | None) -> bool:
        return bool(cache_entry and time.monotonic() - cache_entry[0] < TTS_OPTIONS_CACHE_SECONDS)

    @staticmethod
    def _extract_option_ids(payload: Any, preferred_keys: tuple[str, ...]) -> list[str]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()

        candidates: list[Any] = []
        if isinstance(payload, dict):
            for key in preferred_keys:
                if key in payload:
                    candidates.append(payload[key])
            if "data" in payload:
                candidates.append(payload["data"])
        elif isinstance(payload, list):
            candidates.append(payload)
        else:
            data = getattr(payload, "data", None)
            if data is not None:
                candidates.append(data)

        values: list[str] = []
        index = 0
        while index < len(candidates):
            candidate = candidates[index]
            index += 1
            if isinstance(candidate, dict):
                for key in preferred_keys:
                    nested = candidate.get(key)
                    if nested is not None:
                        candidates.append(nested)
                continue
            if not isinstance(candidate, list):
                continue
            for item in candidate:
                if isinstance(item, str):
                    values.append(item)
                    continue
                if isinstance(item, dict):
                    value = item.get("id") or item.get("name") or item.get("voice")
                else:
                    value = getattr(item, "id", None) or getattr(item, "name", None)
                if value:
                    values.append(str(value))

        return sorted({value.strip() for value in values if value and value.strip()})

    @staticmethod
    def _is_tts_model_id(model_id: str) -> bool:
        lowered = model_id.lower()
        return "tts" in lowered or "speech" in lowered or "silero" in lowered

    async def get_available_tts_models(self) -> list[str]:
        if self._cache_is_fresh(self._tts_models_cache):
            return list(self._tts_models_cache[1])
        try:
            response = await self.client.models.list()
            models = [
                model
                for model in self._extract_option_ids(response, ("models",))
                if self._is_tts_model_id(model)
            ]
        except Exception as exc:
            logger.warning("Failed to load TTS models from API: %s", exc)
            return []
        self._tts_models_cache = (time.monotonic(), models)
        return list(models)

    async def get_available_tts_voices(self, model: str | None = None) -> list[str]:
        model_to_use = model or self.config.get('tts_model', LLMGATEWAY_TTS_MODEL)
        cache_entry = self._tts_voices_cache.get(model_to_use)
        if self._cache_is_fresh(cache_entry):
            return list(cache_entry[1])
        try:
            response = await self.gateway_client.audio_voices(model_to_use)
            voices = self._extract_option_ids(response, ("voices",))
        except Exception as exc:
            logger.warning("Failed to load TTS voices from API: %s", exc)
            return []
        self._tts_voices_cache[model_to_use] = (time.monotonic(), voices)
        return list(voices)

    def get_user_tts_model(self, user_id: int | None = None) -> str:
        settings = get_user_settings(self.db, user_id)
        model = str(settings.get(USER_TTS_MODEL_SETTING) or "").strip()
        return model or self.config['tts_model']

    def get_user_tts_voice(self, user_id: int | None = None) -> str:
        settings = get_user_settings(self.db, user_id)
        voice = str(settings.get(USER_TTS_VOICE_SETTING) or "").strip()
        return voice or str(self.config['tts_voice']).lower()

    async def generate_speech(self, text: str, user_id: int | None = None) -> tuple[any, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.get_user_tts_model(user_id),
                voice=self.get_user_tts_voice(user_id).lower(),
                input=text,
                response_format=self.config.get('tts_response_format', 'wav'),
                extra_headers={ "X-Title": "tgBot" },
            )

            temp_file = io.BytesIO()
            try:
                temp_file.write(response.read())
                temp_file.seek(0)
                return temp_file, len(text)
            except Exception:
                temp_file.close()
                raise
        except Exception as e:
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(
                    model=self.config.get('transcription_model', LLMGATEWAY_TRANSCRIPTION_MODEL),
                    file=audio, 
                    prompt=prompt_text,
                    response_format="text",
                    extra_headers={ "X-Title": "tgBot" },
                )
                return result
        except Exception as e:
            logger.exception(e)
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{error_message}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False, user_id: int | None = None):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        temperature = self.config['temperature']
        try:
            session_id = None
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                await self.reset_chat_history(chat_id)
            else:
                # Загружаем сохраненный контекст из базы данных
                saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id)
                if saved_context and 'messages' in saved_context:
                    self.conversations[chat_id] = saved_context['messages']

            self.last_updated[chat_id] = datetime.datetime.now()
            history_content = self._vision_history_content(content)

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                await self.__add_to_history(chat_id, role="user", content=content)
            else:
                await self.__add_to_history(chat_id, role="user", content=history_content)

            # Summarize the chat history if it's too long to avoid excessive token usage
            vision_model = self.config['vision_model']
            token_count = self.__count_tokens(self.conversations[chat_id], vision_model)
            exceeded_max_tokens = token_count + self.config['vision_max_tokens'] > default_max_tokens(vision_model)
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logger.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:

                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logger.debug(f'Summary: {summary}')
                    # Pre-dispatch: let session-memory subscribers snapshot the
                    # outgoing vision window before we wipe it.
                    await self._dispatch_before_summarise_reset(
                        chat_id=chat_id,
                        user_id=chat_id,
                        session_id=None,
                    )
                    await self.reset_chat_history(
                        chat_id,
                        self.conversations[chat_id][0]['content'],
                        prune_old_sessions=False,
                    )
                    await self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logger.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role':'user', 'content':content}

            common_args = {
                'model': vision_model,
                'messages': self._messages_with_language_instruction(self.conversations[chat_id][:-1] + [message]),
                'temperature': temperature,
                'n': 1, # several choices is not implemented yet
                'max_tokens': self.config['vision_max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }


            if self.config['enable_functions']:
                allowed_plugins = await self.resolve_allowed_plugins(chat_id, session_id, user_id or chat_id)
                tools = self.plugin_manager.get_functions_specs(self, vision_model, allowed_plugins)
                if tools and vision_model not in (O_MODELS + GOOGLE + PERPLEXITY):
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'
            
            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            logger.error(f'Bad request error: {str(e)}')
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{error_message}") from e

        except Exception as e:
            logger.error(f'Error in function call handling: {str(e)}', exc_info=True)
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e

    @staticmethod
    def _vision_history_content(content: list) -> str:
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(str(item.get('text', '')))
        text = ' '.join(part.strip() for part in text_parts if part and part.strip())
        return text or "[image]"


    def _snapshot_chat_state(self, chat_id):
        return {
            "has_conversation": chat_id in self.conversations,
            "conversation": list(self.conversations.get(chat_id, [])),
            "has_vision": chat_id in self.conversations_vision,
            "vision": self.conversations_vision.get(chat_id),
            "has_last_updated": chat_id in self.last_updated,
            "last_updated": self.last_updated.get(chat_id),
            "has_loaded_session": chat_id in self.loaded_conversation_sessions,
            "loaded_session": self.loaded_conversation_sessions.get(chat_id),
        }

    def _restore_chat_state(self, chat_id, snapshot) -> None:
        if snapshot["has_conversation"]:
            self.conversations[chat_id] = list(snapshot["conversation"])
        else:
            self.conversations.pop(chat_id, None)
        if snapshot["has_vision"]:
            self.conversations_vision[chat_id] = snapshot["vision"]
        else:
            self.conversations_vision.pop(chat_id, None)
        if snapshot["has_last_updated"]:
            self.last_updated[chat_id] = snapshot["last_updated"]
        else:
            self.last_updated.pop(chat_id, None)
        if snapshot["has_loaded_session"]:
            self.loaded_conversation_sessions[chat_id] = snapshot["loaded_session"]
        else:
            self.loaded_conversation_sessions.pop(chat_id, None)

    async def interpret_image(self, chat_id, fileobj, prompt=None, user_id=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        lock = await self._chat_lock(chat_id)
        async with lock:
            return await self._interpret_image_locked(chat_id, fileobj, prompt, user_id=user_id)

    async def _interpret_image_locked(self, chat_id, fileobj, prompt, user_id=None):
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        last_error = None
        for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
            snapshot = self._snapshot_chat_state(chat_id)
            try:
                response = await self.__common_get_chat_response_vision(
                    chat_id,
                    content,
                    user_id=user_id,
                )
            except Exception as exc:
                last_error = exc
            else:
                plugins_used = ()
                if self.config['enable_functions']:
                    allowed_plugins = await self.resolve_allowed_plugins(chat_id, user_id=user_id or chat_id)
                    response, plugins_used = await self.__handle_function_call(
                        chat_id,
                        response,
                        allowed_plugins=allowed_plugins,
                        user_id=user_id or chat_id,
                        model_to_use=self.config['vision_model'],
                        retry_plain_text_tool_intent=True,
                    )
                    if is_direct_result(response):
                        return response, '0'
                try:
                    return await self._interpret_image_text_response(chat_id, response, plugins_used)
                except ValueError as exc:
                    last_error = exc
            if attempt < VISION_MAX_ATTEMPTS:
                logger.warning(
                    "Vision request attempt %s/%s failed for chat_id=%s: %s",
                    attempt,
                    VISION_MAX_ATTEMPTS,
                    chat_id,
                    last_error,
                    exc_info=(
                        (type(last_error), last_error, last_error.__traceback__)
                        if isinstance(last_error, Exception)
                        else None
                    ),
                )
                self._restore_chat_state(chat_id, snapshot)
        raise last_error or ValueError(EMPTY_MODEL_RESPONSE_ERROR)

    async def _interpret_image_text_response(self, chat_id, response, plugins_used=()):
        answer = ''
        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = _required_choice_message_text(choice)
                if index == 0:
                    await self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = _required_choice_message_text(_first_choice_or_raise(response))
            await self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}" \
                      f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None, user_id=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        lock = await self._chat_lock(chat_id)
        async with lock:
            async for chunk in self._interpret_image_stream_locked(chat_id, fileobj, prompt, user_id=user_id):
                yield chunk

    async def _interpret_image_stream_locked(self, chat_id, fileobj, prompt, user_id=None):
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True, user_id=user_id)

        

        if self.config['enable_functions']:
            allowed_plugins = await self.resolve_allowed_plugins(chat_id, user_id=user_id or chat_id)
            response, plugins_used = await self.__handle_function_call(
                chat_id,
                response,
                stream=True,
                allowed_plugins=allowed_plugins,
                user_id=user_id or chat_id,
                model_to_use=self.config['vision_model'],
                retry_plain_text_tool_intent=True,
            )
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        if not answer:
            raise ValueError(EMPTY_MODEL_RESPONSE_ERROR)
        await self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        #show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        #plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
        #     if show_plugins_used:
        #         answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    @staticmethod
    def _messages_with_language_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        language = get_current_language()
        language_instruction = {
            "role": "system",
            "content": (
                f"Reply to the user in {language_name(language)} "
                "unless the user explicitly asks for another language."
            ),
        }
        if messages and messages[0].get("role") == "system":
            return [messages[0], language_instruction, *messages[1:]]
        return [language_instruction, *messages]

    async def _apply_before_chat_request_mutators(
        self,
        chat_id: int,
        user_id: int | None,
        session_id: str | None,
        request_id: str | None,
        parse_mode: str | None = None,
        temperature: float | None = None,
        max_tokens_percent: int | None = None,
        persist: bool = False,
    ) -> list[dict[str, Any]]:
        """Build outbound messages for a chat request.

        Pipeline: repair tool-call history → snapshot with language instruction
        → on_before_chat_request mutator chain → optional persistence of any
        session-memory injection so subsequent turns skip recall.
        """
        self._repair_tool_call_history(chat_id)
        plugin = (
            self.plugin_manager.get_plugin("hindsight_memory")
            if getattr(self, "plugin_manager", None)
            else None
        )
        if plugin is not None:
            is_memory_message = getattr(plugin, "is_hindsight_memory_message", None)
            is_disabled_for_user = getattr(self.plugin_manager, "is_plugin_disabled_for_user", None)
            disabled_for_user = (
                callable(is_disabled_for_user)
                and is_disabled_for_user(plugin.get_plugin_id(), user_id)
            )
            can_send_memory = (
                user_id is not None
                and bool(getattr(plugin, "is_active", False))
                and bool(getattr(plugin, "auto_recall_enabled", True))
                and not disabled_for_user
            )
            if callable(is_memory_message) and not can_send_memory:
                original = self.conversations[chat_id]
                cleaned = [msg for msg in original if not is_memory_message(msg)]
                if len(cleaned) != len(original):
                    self.conversations[chat_id] = cleaned
                    if persist:
                        try:
                            await self._save_conversation_context(
                                chat_id,
                                {'messages': cleaned},
                                parse_mode,
                                temperature,
                                max_tokens_percent,
                                session_id,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to persist removal of session-memory message for chat_id=%s: %s",
                                chat_id, exc,
                            )
        base = self._messages_with_language_instruction(self.conversations[chat_id])
        payload = BeforeChatRequestPayload(
            chat_id=chat_id, user_id=user_id, request_id=request_id,
            allow_dynamic_recall=persist,
        )
        mutated = await self.plugin_manager.apply_mutators(
            'on_before_chat_request',
            payload,
            base,
            user_id=user_id,
        )
        if not persist:
            return mutated

        if plugin is None:
            return mutated
        already_in_conv = any(
            plugin.is_hindsight_memory_message(msg)
            for msg in self.conversations[chat_id]
        )
        if already_in_conv:
            return mutated
        injected = next(
            (msg for msg in mutated if plugin.is_hindsight_memory_message(msg)),
            None,
        )
        if injected is None:
            return mutated
        conv = self.conversations[chat_id]
        if conv and conv[0].get("role") == "system":
            conv.insert(1, injected)
        else:
            conv.insert(0, injected)
        try:
            await self._save_conversation_context(
                chat_id,
                {'messages': conv},
                parse_mode,
                temperature,
                max_tokens_percent,
                session_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist session-memory message for chat_id=%s: %s",
                chat_id, exc,
            )
        return mutated

    @staticmethod
    def _tool_call_ids(message: dict[str, Any]) -> list[str]:
        tool_calls = message.get("tool_calls") or []
        ids = []
        for call in tool_calls:
            if isinstance(call, dict) and call.get("id"):
                ids.append(str(call["id"]))
        return ids

    def _repair_tool_call_history(self, chat_id: int) -> None:
        messages = self.conversations.get(chat_id)
        if not isinstance(messages, list):
            return

        repaired = []
        changed = False
        index = 0
        while index < len(messages):
            message = messages[index]
            if not isinstance(message, dict):
                repaired.append(message)
                index += 1
                continue

            expected_ids = self._tool_call_ids(message)
            if message.get("role") == "assistant" and expected_ids:
                repaired.append(message)
                seen_ids = set()
                index += 1

                while index < len(messages):
                    next_message = messages[index]
                    if not isinstance(next_message, dict) or next_message.get("role") != "tool":
                        break
                    raw_tool_call_id = next_message.get("tool_call_id")
                    tool_call_id = str(raw_tool_call_id) if raw_tool_call_id is not None else None
                    if tool_call_id in expected_ids and tool_call_id not in seen_ids:
                        repaired.append(next_message)
                        seen_ids.add(tool_call_id)
                    else:
                        changed = True
                    index += 1

                missing_ids = [tool_call_id for tool_call_id in expected_ids if tool_call_id not in seen_ids]
                for tool_call_id in missing_ids:
                    repaired.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({
                            "error": "Tool result missing because execution was interrupted before a result was recorded."
                        }, ensure_ascii=False),
                    })
                    changed = True
                if missing_ids:
                    logger.warning(
                        "Repaired incomplete tool-call history for chat_id=%s missing_tool_call_ids=%s",
                        chat_id,
                        missing_ids,
                    )
                continue

            if message.get("role") == "tool":
                changed = True
                logger.warning(
                    "Dropping orphan tool result from history for chat_id=%s tool_call_id=%s",
                    chat_id,
                    message.get("tool_call_id"),
                )
                index += 1
                continue

            repaired.append(message)
            index += 1

        if changed:
            self.conversations[chat_id] = repaired

    async def _chat_lock(self, chat_id) -> asyncio.Lock:
        """Per-chat asyncio.Lock guarding mutations of self.conversations.

        Why: telegram_bot.process_message holds a per-conversation lock for
        the prompt path, but callback queries, inline handlers and plugin
        commands may call top-level OpenAIHelper methods independently for
        the same chat_id. Holding this lock around top-level entry points
        (get_chat_response, get_chat_response_stream, interpret_image*) keeps
        mutations of conversations[chat_id] serialized for that chat_id.
        Note: do not acquire this lock in inner helpers — asyncio.Lock is
        not reentrant and the recursive call paths (handle_function_call,
        retry helpers) would deadlock.
        """
        guard = getattr(self, '_chat_locks_guard', None)
        if guard is None:
            guard = asyncio.Lock()
            self._chat_locks_guard = guard
            self._per_chat_locks = {}
        async with guard:
            lock = self._per_chat_locks.get(chat_id)
            if lock is None:
                lock = asyncio.Lock()
                self._per_chat_locks[chat_id] = lock
            return lock

    def _chat_lock_bypass_enabled(self, chat_id) -> bool:
        bypass_chat_id = _CHAT_LOCK_BYPASS_CHAT_ID.get()
        return bypass_chat_id is not None and str(bypass_chat_id) == str(chat_id)

    @contextmanager
    def _without_chat_lock(self, chat_id):
        token = _CHAT_LOCK_BYPASS_CHAT_ID.set(chat_id)
        try:
            yield
        finally:
            _CHAT_LOCK_BYPASS_CHAT_ID.reset(token)

    def _clear_chat_state(self, chat_id) -> None:
        """Atomically drop all per-chat state for chat_id.

        Why: conversations / conversations_vision / last_updated /
        loaded_conversation_sessions / last_image_file_ids share the same
        chat_id key. Clearing only one of them leaves the others stale,
        and __max_age_reached / vision routing read stale flags.
        """
        self.conversations.pop(chat_id, None)
        self.conversations_vision.pop(chat_id, None)
        self.last_updated.pop(chat_id, None)
        self.loaded_conversation_sessions.pop(chat_id, None)
        self.last_image_file_ids.pop(chat_id, None)

    async def _dispatch_before_summarise_reset(
        self,
        *,
        chat_id: int,
        user_id: int | None,
        session_id: str | None,
    ) -> None:
        """Fire ``on_session_before_delete`` for the in-memory conversation
        about to be cleared by summarise-overflow.

        Subscribers get a chance to snapshot the outgoing window
        before we wipe ``self.conversations[chat_id]`` via ``reset_chat_history``.
        Failures are swallowed — summarisation must continue regardless.
        """
        pm = getattr(self, "plugin_manager", None)
        if pm is None:
            return
        conv = self.conversations.get(chat_id) or ()
        messages = tuple(dict(m) for m in conv if isinstance(m, dict))
        if not messages:
            return
        effective_user_id = user_id if user_id is not None else chat_id
        effective_session_id = session_id or ""
        payload = SessionBeforeDeletePayload(
            user_id=effective_user_id,
            session_id=effective_session_id,
            messages=messages,
        )
        try:
            await pm.dispatch_blocking(
                HookEvent.ON_SESSION_BEFORE_DELETE,
                payload,
                user_id=effective_user_id,
            )
        except Exception:
            logger.exception("on_session_before_delete pre-dispatch failed")

    async def _dispatch_before_create_session_prune(self, user_id: int) -> None:
        """Fire ``on_session_before_delete`` for sessions that
        ``Database.create_session`` is about to prune.

        Walks the same selection ``delete_oldest_session`` uses so the hook
        sees exactly the sessions that will be deleted. Failures here must not
        block session creation — caller continues regardless.
        """
        pm = getattr(self, "plugin_manager", None)
        if pm is None:
            return
        try:
            max_sessions = int(self.config.get('max_sessions', 5))
            session_ids = self.db.get_oldest_session_ids_for_limit(
                user_id, max_sessions=max_sessions,
            )
        except Exception:
            logger.exception("Failed to enumerate sessions for pre-prune dispatch")
            return
        for session_id in session_ids:
            try:
                context, _, _, _, _ = self.db.get_conversation_context(user_id, session_id)
                messages = context.get('messages', []) if isinstance(context, dict) else []
                messages = tuple(
                    dict(m) for m in messages if isinstance(m, dict)
                )
            except Exception:
                logger.exception(
                    "Failed to load session %s/%s for pre-prune dispatch",
                    user_id, session_id,
                )
                continue
            payload = SessionBeforeDeletePayload(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
            )
            try:
                await pm.dispatch_blocking(
                    HookEvent.ON_SESSION_BEFORE_DELETE,
                    payload,
                    user_id=user_id,
                )
            except Exception:
                logger.exception(
                    "on_session_before_delete pre-prune dispatch failed for %s/%s",
                    user_id, session_id,
                )

    async def reset_chat_history(
        self,
        chat_id,
        content='',
        session_id=None,
        prune_old_sessions: bool = True,
    ):
        """
        Resets the conversation history.
        :param chat_id: Chat identifier
        :param content: Initial system message content
        :param session_id: Optional session identifier
        :param prune_old_sessions: When True (default) and a new session is
            created, prune the oldest sessions for the user; pass False from
            summarise-overflow paths so we don't cascade old-session deletes.
        """
        try:
            # Получаем или создаем сессию через базу данных
            if not session_id:
                # Pre-dispatch on_session_before_delete for any sessions that
                # will be pruned, so plugin subscribers can snapshot them.
                if prune_old_sessions:
                    await self._dispatch_before_create_session_prune(chat_id)
                session_id = self.db.create_session(
                    chat_id,
                    max_sessions=self.config.get('max_sessions', 5),
                    openai_helper=self,
                    prune_old_sessions=prune_old_sessions,
                )
            
            if not session_id:
                raise ValueError(f"Не удалось создать/получить сессию для пользователя {chat_id}")
            
            # Получаем контекст сессии
            context, parse_mode, temperature, max_tokens_percent, _ = self.db.get_conversation_context(chat_id, session_id)
            # Инициализируем историю чата
            system_message = self.db.get_mode_from_context(context)
            
            self.conversations[chat_id] = [{"role": "system", "content": content or (system_message['content'] if system_message else '')}]
            self.loaded_conversation_sessions[chat_id] = session_id
            
            # Сохраняем обновленный контекст
            await self._save_conversation_context(
                chat_id,
                {'messages': self.conversations[chat_id]},
                parse_mode,
                temperature,
                max_tokens_percent,
                session_id,
            )
            
            logger.info(f'Chat history reset for chat_id={chat_id}, session_id={session_id}')
            
        except Exception as e:
            logger.error(f'Error in reset_chat_history: {str(e)}', exc_info=True)
            raise

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_function_call_to_history(self, chat_id, function_name, content, tool_call_id=None):
        """
        Adds a function call to the conversation history
        """
        # For models that don't support function role, add as a user message
        user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
        model_to_use = self.get_current_model(user_id)
        content = self._tool_result_content(content)

        if tool_call_id and self._uses_structured_tool_history(model_to_use):
            self.conversations[chat_id].append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            })
            return

        # Some providers either don't support (or inconsistently support) the legacy "function" role.
        # For those, we inject tool results as regular user text so the model reliably sees them.
        if model_to_use in (ANTHROPIC + DEEPSEEK):
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[chat_id].append({"role": "user", "content": function_result})
        elif model_to_use in (MISTRALAI + MOONSHOTAI):
            # Mistral и Moonshot используют роль "tool" вместо "function"
            self.conversations[chat_id].append({"role": "tool", "name": function_name, "content": content})
        elif model_to_use in (O_MODELS + GPT_4O_MODELS + LLMGATEWAY_CHAT_MODELS):
            # For all other models (OpenAI-style), use the assistant role instead of deprecated function role
            # The 'function' role is no longer supported in OpenAI API as of 2025
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[chat_id].append({"role": "assistant", "content": function_result})
        else:
            # For OpenAI-style models, use the function role
            self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    async def __add_to_history(self, chat_id, role, content, session_id=None):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        :param session_id: Optional session identifier
        """
        # Дополнительная проверка инициализации conversation
        if chat_id not in self.conversations:
            logger.warning(f'Conversation not initialized for chat_id={chat_id}, initializing now')
            await self.reset_chat_history(chat_id, session_id=session_id)

        self.conversations[chat_id].append({"role": role, "content": content})
        
        # Получаем текущий контекст для сохранения с учетом session_id
        _, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
        self.loaded_conversation_sessions[chat_id] = session_id
        
        # Сохраняем обновленный контекст в базу данных с использованием session_id
        await self._save_conversation_context(
            chat_id, 
            {'messages': self.conversations[chat_id]}, 
            parse_mode, 
            temperature, 
            max_tokens_percent,
            session_id,
        )

    async def __summarise(self, conversation, chat_id=None, session_id=None) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :param chat_id: The chat ID of the conversation
        :param session_id: Optional session identifier
        :return: The summary
        """
        try:
            # Ограничиваем размер входных данных
            model_to_use = self.config['model']
            if chat_id is not None:
                model_to_use = self.get_current_model(chat_id)

            max_tokens = default_max_tokens(model_to_use)
            current_tokens = 0
            truncated_conversation = []
            
            # Подсчитываем токены и обрезаем историю если нужно
            for msg in reversed(conversation):  # Идем с конца, чтобы сохранить последние сообщения
                msg_tokens = self.__count_tokens([msg])
                if current_tokens + msg_tokens > max_tokens:
                    break
                current_tokens += msg_tokens
                truncated_conversation.insert(0, msg)  # Вставляем в начало списка

            messages = [
                {"role": "assistant", "content": "Summarize this conversation in 1000 characters or less"},
                {"role": "user", "content": str(truncated_conversation)}
            ]
            
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.4,
                max_tokens=self.get_max_tokens(model_to_use, 80, chat_id),  # Явно ограничиваем размер ответа
                extra_headers={ "X-Title": "tgBot" },
            )
            
            summary = _first_choice_or_raise(response).message.content

            if chat_id is not None:
                # Получаем текущие настройки из базы данных
                _, parse_mode, temperature, max_tokens_percent, current_session_id = self.db.get_conversation_context(chat_id, session_id)
                
                # Используем session_id из параметров, если он передан, иначе из базы
                session_id = session_id or current_session_id
                
                # Сохраняем обновленный контекст после суммаризации
                await self._save_conversation_context(
                    chat_id, 
                    {'messages': self.conversations[chat_id]}, 
                    parse_mode,
                    temperature,
                    max_tokens_percent,
                    session_id,
                )
                
            return summary
        except Exception as e:
            logger.error(f'Error in summarise: {str(e)}')
            # В случае ошибки возвращаем базовое сообщение
            return "Previous conversation history was too long and has been truncated."

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages, model_to_use = None) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['model']
        if model_to_use is not None:
            model = model_to_use
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        supported_models = (
            GPT_4_VISION_MODELS + GPT_4O_MODELS + O_MODELS + 
            ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + 
            PERPLEXITY + LLAMA + MOONSHOTAI + QWEN + GPT_5_MODELS +
            LLMGATEWAY_CHAT_MODELS
        )
        if model in supported_models:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if value is None:
                        continue
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    elif isinstance(value, list):
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                    else:
                        num_tokens += len(encoding.encode(str(value)))
                elif key == 'tool_calls':
                    num_tokens += len(encoding.encode(json.dumps(value, ensure_ascii=False)))
                else:
                    if value is None:
                        continue
                    num_tokens += len(encoding.encode(str(value)))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # no longer needed

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        """
        Counts the number of tokens for interpreting an image.
        :param image_bytes: image to interpret
        :return: the number of tokens required
        """
        try:
            image_file = io.BytesIO(image_bytes)
            with Image.open(image_file) as image:
                w, h = image.size
                if w > h: 
                    w, h = h, w
                
                # this computation follows https://platform.openai.com/docs/guides/vision and https://openai.com/pricing#gpt-4-turbo
                base_tokens = 85
                detail = self.config.get('vision_detail', 'auto')
                
                if detail == 'low':
                    return base_tokens
                elif detail == 'high' or detail == 'auto': # assuming worst cost for auto
                    f = max(w / 768, h / 2048)
                    if f > 1:
                        w, h = int(w / f), int(h / f)
                    tw, th = (w + 511) // 512, (h + 511) // 512
                    tiles = tw * th
                    num_tokens = base_tokens + tiles * 170
                    return num_tokens
                else:
                    raise ValueError(f"Unknown vision_detail parameter: {detail}")
        except Exception as e:
            logger.error(f"Error processing image for token counting: {e}")
            raise

    async def get_file_data(self, file_id: str) -> bytes:
        """
        Получает данные файла из Telegram по file_id
        """
        if not hasattr(self, 'bot'):
            raise ValueError("Bot instance not available")
        
        try:
            file = await self.bot.get_file(file_id)
            return await file.download_as_bytearray()
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

    async def download_file_as_bytes(self, file_id: str) -> bytes:
        """
        Downloads a Telegram file by file_id as bytes.
        """
        return bytes(await self.get_file_data(file_id))

    def set_last_image_file_id(self, chat_id: int, file_id: str):
        """Store the last image file ID for the chat"""
        self.last_image_file_ids = getattr(self, 'last_image_file_ids', {})
        self.last_image_file_ids[chat_id] = file_id

    def get_last_image_file_id(self, chat_id: int) -> str | None:
        """
        Gets the last image file ID for a specific chat
        """
        return self.last_image_file_ids.get(chat_id)

    async def generate_session_name(
        self, 
        chat_id: int, 
        first_message: str, 
        session_id: Optional[str] = None
    ) -> tuple[str, int]:
        """
        Генерация названия сессии на основе первого запроса
        
        :param chat_id: ID чата
        :param first_message: Первое сообщение пользователя
        :param session_id: Идентификатор сессии
        :return: Сгенерированное название сессии
        """
        try:
            # Создаем промпт для генерации названия
            prompt = f"""
            Сгенери короткое (до 50 символов) название для беседы на основе следующего сообщения:
            "{first_message}"
            
            Правила:
            - Название должно быть лаконичным
            - Отражать суть первого сообщения
            - Использовать существительные или глаголы
            - Избегать слишком общих названий
            """
            
            model_to_use = self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL)
            response = await self.chat_completion(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты лучший специалист по созданию коротких названий для сессий. Верни только название.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=int(self.get_max_tokens(model_to_use, 20, chat_id)),
                stream=False,
            )
            content = _required_choice_message_text(_first_choice_or_raise(response))
            tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0

            # Очищаем и обрезаем название
            session_name = content.strip().strip('"')[:50]
            
            # Если название пустое, используем дефолтное
            return session_name or f"Сессия {dt.now().strftime('%d.%m')}", tokens
        
        except Exception as e:
            logger.error(f"Ошибка генерации названия сессии: {e}")
            return f"Сессия {dt.now().strftime('%d.%m')}", 0

    def get_current_model(self, user_id: int = None) -> str:
        """
        Получает текущую модель с учетом приоритетов:
        1. Модель из активной сессии
        2. Глобальная модель пользователя
        3. Модель по умолчанию из конфига
        
        :param user_id: ID пользователя (опционально)
        :return: Название модели
        """
        if user_id:
            # Получаем активную сессию
            sessions = self.db.list_user_sessions(user_id, is_active=1)
            active_session = next((s for s in sessions if s['is_active']), None)
            
            # Проверяем модель в сессии, но используем только разрешенные llmgateway модели.
            session_model = active_session.get('model', '') if active_session else ''
            if session_model in LLMGATEWAY_CHAT_MODELS:
                logger.info(f"Модель из активной сессии: {session_model}")
                return session_model
            if session_model:
                logger.info(f"Игнорируем неподдерживаемую модель из сессии: {session_model}")
                            
        # Возвращаем модель по умолчанию
        logger.info(f"Модель по умолчанию: {self.config['model']}")
        return self.config['model']
    
    def get_max_tokens(self, model_to_use, max_tokens_percent, chat_id):
        # Получаем максимальное количество токенов для модели
        total_max_tokens = default_max_tokens(model_to_use)
        
        # Рассчитываем текущее количество токенов в контексте
        current_tokens = 0
        if chat_id is not None and chat_id in self.conversations:
            current_tokens = self.__count_tokens(self.conversations[chat_id], model_to_use)
        
        # Резервируем место для системных токенов
        reserved_tokens = 3000  # Увеличено резервирование
        
        # Максимальное количество токенов для генерации
        max_generation_tokens = max(
            1000,  # Минимальное количество токенов для генерации
            min(
                total_max_tokens - current_tokens - reserved_tokens,  # Оставшиеся токены
                total_max_tokens // 2  # Не более половины от общего количества
            )
        )
        
        # Применяем процентное ограничение
        max_generation_tokens = min(
            max_generation_tokens, 
            total_max_tokens * max_tokens_percent / 100
        )
        
        if model_to_use in GPT_4O_MODELS:
            max_generation_tokens = 32768

        logger.info(f"""
        Токены для модели {model_to_use}:
        - Всего: {total_max_tokens}
        - Текущий контекст: {current_tokens}
        - Зарезервировано: {reserved_tokens}
        - Доступно для генерации: {max_generation_tokens}
        """)
        
        return int(max_generation_tokens)

    def get_all_modes(self):
        return self.chat_modes_registry.get_all_modes_list()

    async def _build_auto_chat_mode_prompt(
        self,
        query: str,
        *,
        chat_id: int,
        user_id: int | None,
    ) -> str:
        payload = PromptFragmentPayload(
            slot="auto_mode_priority",
            chat_id=chat_id,
            user_id=user_id,
            query=query,
        )
        fragments = await self.plugin_manager.collect_fragments(
            "auto_mode_priority", payload, user_id=user_id
        )
        priority_block = ""
        if fragments:
            priority_block = "\n\n".join(fragments) + "\n\n"
        return f"""Определи режим работы для сообщения и верни только ключ режима.

{priority_block}Остальные правила выбора:
1. Если задача простая и может быть решена одним коротким ответом или одним очевидным инструментом, выбирай наиболее простой подходящий режим.
2. Если задача сложная, открытая или ожидаемо требует больше двух шагов, выбирай skills_agent, если такой режим есть в списке доступных режимов.
3. Сложная задача - это задача, где нужно построить план и выполнить больше двух связанных шагов: последовательно использовать инструменты, обработать файлы или артефакты, запустить локальные scripts, проверить результат, исправить ошибки или уточнить требования у пользователя.
4. Не выбирай skills_agent по отдельным словам. Выбирай его по структуре задачи: больше двух шагов, неопределенный маршрут, необходимость orchestration или проверки промежуточных результатов.
5. Если сложность не нужна, не выбирай skills_agent.
6. Если ни один режим не подходит, верни assistant.

Сообщение: ^{query}^
Доступные режимы: ^{self.get_all_modes()}^"""

    async def close(self):
        """
        Закрывает HTTP-клиенты и освобождает ресурсы OpenAIHelper.
        
        Этот метод необходим для корректного завершения работы, так как:
        - OpenAI клиент использует httpx.AsyncClient для HTTP-соединений
        - Без закрытия могут остаться висящие соединения
        - Вызывается из telegram_bot.cleanup() при завершении работы бота
        """
        try:
            # Закрываем OpenAI клиент, который автоматически закроет httpx.AsyncClient
            if hasattr(self, 'client') and self.client:
                await self.client.close()
                logger.info("OpenAI client closed successfully")
            if hasattr(self, 'gateway_client') and self.gateway_client:
                await self.gateway_client.close()
                logger.info("LLMGateway client closed successfully")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")
