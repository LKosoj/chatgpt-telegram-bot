from __future__ import annotations
import datetime
import logging
import os
import asyncio
import uuid
import re
from typing import Any, Optional
from datetime import datetime as dt

import tiktoken

import openai
import requests

from functools import lru_cache
import json
import httpx
import io
from calendar import monthrange
from PIL import Image
import yaml

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

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
from .hindsight_client import HindsightClient, format_recall_results
from .chat_modes_registry import ChatModesRegistry
from .validation import validate_openai_config
from .openai_tool_handler import handle_function_call
from .i18n import localized_text

logger = logging.getLogger(__name__)

HINDSIGHT_MEMORY_MARKER = "[HINDSIGHT_MEMORY_CONTEXT]"
EMPTY_MODEL_RESPONSE_ERROR = "Модель вернула пустой ответ"
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
RAW_TOOL_RESULT_RE = re.compile(r"^Function\s+[\w.\-]+\s+returned:\s*", re.IGNORECASE)


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

HINDSIGHT_CONTEXT_PROMPT = f"""{HINDSIGHT_MEMORY_MARKER}
Long-term memory recalled for this Telegram user:
{{memory}}

Use this only as background context when it is relevant. If the current user message
contradicts this memory, prefer the current message. Do not mention Hindsight or memory
retrieval unless the user asks about it."""

HINDSIGHT_EXTRACTOR_PROMPT = """Extract durable memories from the Telegram conversation.
Return only JSON in this exact shape:
{"items":[{"content":"...","context":"...","tags":["..."]}]}

Save only facts that are clearly durable and likely useful in future conversations with the same user:
- explicit "remember this" facts;
- stable user preferences, identity details, long-term goals, ongoing projects, durable constraints, and decisions;
- important project facts or agreements that should survive across sessions.

Do not infer preferences from weak signals. For example, do not save "user prefers Russian"
only because the conversation is in Russian; save it only if the user explicitly says it or clearly corrects the assistant.

Do not save one-off tasks or transient requests: image generation/editing requests, uploaded-image descriptions,
audio/transcription requests, web searches, debugging logs, single SQL questions, generic chit-chat, or temporary commands.
Do not save passwords, API keys, tokens, secrets, credentials, private auth data, or facts contradicted inside the exchange.

Examples to reject:
- User asked to draw or edit a cat with a hat.
- User asked what is shown in an uploaded image.
- User asked one isolated technical question and got an answer.

When in doubt, save nothing.
If there is nothing worth saving, return {"items":[]}."""

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
        self.config = config
        self.plugin_manager = plugin_manager
        self.db = db
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.loaded_conversation_sessions: dict[int, str | None] = {}  # {chat_id: session_id}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}
        self.last_image_file_ids = {}
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
        self.config.setdefault('hindsight_base_url', '')
        self.config.setdefault('hindsight_api_token', '')
        self.config['hindsight_enabled'] = bool(
            self.config.get('hindsight_base_url')
            and self.config.get('hindsight_api_token')
        )
        self.config.setdefault('hindsight_auto_recall', True)
        self.config.setdefault('hindsight_auto_save', True)
        self.config.setdefault('hindsight_namespace', 'default')
        self.config.setdefault('hindsight_bank_prefix', 'telegram-')
        self.config.setdefault('hindsight_recall_budget', 'mid')
        self.config.setdefault('hindsight_recall_max_tokens', 4096)
        self.config.setdefault('hindsight_memory_types', 'world,experience')
        self.config.setdefault('hindsight_async_store', True)
        self.config.setdefault('hindsight_timeout', 30.0)
        self.config.setdefault('hindsight_max_auto_save_items', 5)
        self.hindsight_client = None
        if self.config['hindsight_enabled']:
            self.hindsight_client = HindsightClient(
                self.config.get('hindsight_base_url', ''),
                self.config.get('hindsight_api_token', ''),
                namespace=self.config.get('hindsight_namespace', 'default'),
                timeout=float(self.config.get('hindsight_timeout', 30.0)),
            )

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
        content = response.choices[0].message.content or ""
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

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def ask(self, prompt, user_id, assistant_prompt=None, model=None):
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
                    self.reset_chat_history(user_id, session_id=None)
            
            # Инициализируем conversations_vision для пользователя, если его нет
            if user_id not in self.conversations_vision:
                self.conversations_vision[user_id] = False
                
            add_prompt1 = f" Текущая дата и время: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
            if assistant_prompt == None:
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

            self.__add_to_history(user_id, role="user", content=prompt)
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=self.get_max_tokens(model_to_use, 60, user_id),
                temperature=0.6,
                stream=False,
                extra_headers={ "X-Title": "tgBot" }
            )
            content = _required_choice_message_text(response.choices[0])
            self.__add_to_history(user_id, role="assistant", content=content)
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
        try:
            if request_context is not None:
                chat_id = request_context.chat_id
                user_id = request_context.user_id
                if session_id is None:
                    session_id = request_context.session_id

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
            
            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                allowed_plugins = self.resolve_allowed_plugins(chat_id, session_id)
                response, plugins_used = await self.__handle_function_call(
                    chat_id,
                    response,
                    allowed_plugins=allowed_plugins,
                    user_id=user_id,
                    request_context=request_context,
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
                    )
                    if retry_response is not None:
                        response, retry_plugins_used = await self.__handle_function_call(
                            chat_id,
                            retry_response,
                            allowed_plugins=allowed_plugins,
                            user_id=user_id,
                            request_context=request_context,
                        )
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug('Direct result returned after empty response retry')
                            return response, '0'
                    if not _response_has_message_text(response):
                        response = await self._retry_empty_response_after_tools(chat_id, user_id, session_id)
                elif not plugins_used and not _response_has_message_text(response):
                    retry_response = await self._retry_empty_response_with_tools(
                        chat_id,
                        user_id,
                        session_id,
                        allowed_plugins,
                    )
                    if retry_response is not None:
                        response, retry_plugins_used = await self.__handle_function_call(
                            chat_id,
                            retry_response,
                            allowed_plugins=allowed_plugins,
                            user_id=user_id,
                            request_context=request_context,
                        )
                        plugins_used += retry_plugins_used
                        if is_direct_result(response):
                            logger.debug('Direct result returned after empty response retry')
                            return response, '0'
                        if retry_plugins_used and not _response_has_message_text(response):
                            response = await self._retry_empty_response_after_tools(chat_id, user_id, session_id)

            answer = ''

            if len(response.choices) > 1 and self.config['n_choices'] > 1:
                for index, choice in enumerate(response.choices):
                    content = _required_choice_message_text(choice)
                    if index == 0:
                        self.__add_to_history(chat_id, role="assistant", content=content, session_id=session_id)
                    answer += f'{index + 1}\u20e3\n'
                    answer += content
                    answer += '\n\n'
            else:
                answer = _required_choice_message_text(response.choices[0])
                self.__add_to_history(chat_id, role="assistant", content=answer, session_id=session_id)

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
        plugins_used = ()
        try:
            logger.info(f'Starting chat response stream for chat_id={chat_id}')

            if request_context is not None:
                chat_id = request_context.chat_id
                user_id = request_context.user_id
                if session_id is None:
                    session_id = request_context.session_id
            
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
                    self.reset_chat_history(chat_id, session_id=session_id)
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

            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                try:
                    allowed_plugins = self.resolve_allowed_plugins(chat_id, session_id)
                    response, plugins_used = await self.__handle_function_call(
                        chat_id,
                        response,
                        stream=True,
                        allowed_plugins=allowed_plugins,
                        user_id=user_id,
                        request_context=request_context,
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
            self.__add_to_history(chat_id, role="assistant", content=answer, session_id=session_id)
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

    def resolve_allowed_plugins(self, chat_id: int, session_id: str | None = None):
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
                self.reset_chat_history(chat_id, '', session_id)
                saved_context, _, _, _, _ = self.db.get_conversation_context(chat_id, session_id)
                if saved_context and 'messages' in saved_context:
                    system_message = next(
                        (msg for msg in saved_context['messages'] if msg.get('role') == 'system'),
                        None
                    )

            current_mode = self._mode_from_system_message(system_message)

            if current_mode and 'tools' in current_mode:
                allowed_plugins = current_mode['tools']

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
                    self.reset_chat_history(chat_id, session_id=session_id)
                self.loaded_conversation_sessions[chat_id] = session_id

            # Инициализируем conversations_vision для чата, если его нет
            if chat_id not in self.conversations_vision:
                self.conversations_vision[chat_id] = False

            # Проверяем, является ли это первым сообщением в сессии, если да, то определяем режим работы
            user_messages = [msg for msg in self.conversations[chat_id] if msg['role'] == 'user']
            model_to_use = self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL)
            if len(user_messages) == 0 and self.config['auto_chat_modes']:
                mode_name, _ = self.ask_sync(
                        self._build_auto_chat_mode_prompt(query),
                        chat_id,
                        "Ты маршрутизатор режимов работы. Возвращай только ключ режима без пояснений.",
                        model=model_to_use
                    )
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
                        self.db.save_conversation_context(
                            chat_id,
                            {'messages': self.conversations[chat_id]},
                            parse_mode,
                            temperature,
                            max_tokens_percent,
                            session_id,
                            self
                        )
            
            memory_user_id = kwargs.get('user_id') or chat_id
            if len(user_messages) == 0:
                await self._prepare_hindsight_session_memory_context(
                    chat_id,
                    memory_user_id,
                    query,
                    parse_mode,
                    temperature,
                    max_tokens_percent,
                    session_id,
                )

            self.last_updated[chat_id] = datetime.datetime.now()

            self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)

            user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
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
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'], session_id)
                    self.__add_to_history(chat_id, role="assistant", content=summary, session_id=session_id)
                    self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)
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

            common_args = {
                'model': model_to_use, #if not self.conversations_vision[chat_id] else self.config['vision_model'],
                'messages': self._messages_with_hindsight_context(chat_id),
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

            if self.config['enable_functions'] and not self.conversations_vision.get(chat_id, False):
                allowed_plugins = self.resolve_allowed_plugins(chat_id, session_id)
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
        )

    def _uses_structured_tool_history(self, model_to_use: str) -> bool:
        return model_to_use in (GPT_4O_MODELS + LLMGATEWAY_CHAT_MODELS)

    def _tool_result_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

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

    async def _retry_empty_response_after_tools(self, chat_id: int, user_id: int | None, session_id: str | None):
        logger.warning(
            "Retrying empty model response after tool calls for chat_id=%s user_id=%s session_id=%s",
            chat_id,
            user_id,
            session_id,
        )
        model_to_use = self.get_current_model(user_id)
        session_owner = user_id if user_id is not None else chat_id
        max_tokens_percent = 80
        try:
            sessions = self.db.list_user_sessions(session_owner, is_active=1)
            active_session = next((s for s in sessions if s['is_active']), None)
            if active_session:
                max_tokens_percent = active_session.get('max_tokens_percent') or max_tokens_percent
        except Exception as exc:
            logger.warning("Failed to read active session for empty response retry: %s", exc)

        max_tokens = self.get_max_tokens(model_to_use, max_tokens_percent, chat_id)
        messages = list(self._messages_with_hindsight_context(chat_id))
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
    ):
        model_to_use = self.get_current_model(user_id)
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
        messages = list(self._messages_with_hindsight_context(chat_id))
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
        Edits a Telegram image by downloading it and sending it to the LLMGateway image edit model.
        """
        bot_language = self.config['bot_language']
        try:
            image_bytes = await self.download_file_as_bytes(file_id)
            response = await self.gateway_client.image_edit_file(prompt, image_bytes)
            return extract_image_result(response)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def generate_speech(self, text: str) -> tuple[any, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=str(self.config['tts_voice']).lower(),
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
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        temperature = self.config['temperature']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)
            else:
                # Загружаем сохраненный контекст из базы данных
                saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id)
                if saved_context and 'messages' in saved_context:
                    self.conversations[chat_id] = saved_context['messages']

            self.last_updated[chat_id] = datetime.datetime.now()
            history_content = self._vision_history_content(content)

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=history_content)
            else:
                self.__add_to_history(chat_id, role="user", content=history_content)

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
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logger.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role':'user', 'content':content}

            common_args = {
                'model': vision_model,
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': temperature,
                'n': 1, # several choices is not implemented yet
                'max_tokens': self.config['vision_max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }


            # vision model does not yet support functions

            # if self.config['enable_functions']:
            #     functions = self.plugin_manager.get_functions_specs(self, model_to_use)
            #     if len(functions) > 0:
            #         common_args['functions'] = self.plugin_manager.get_functions_specs(self, model_to_use)
            #         common_args['function_call'] = 'auto'
            
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


    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content)

        

        # functions are not available for this model
        
        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response)
        #     if is_direct_result(response):
        #         return response, '0'

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = _required_choice_message_text(choice)
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = _required_choice_message_text(response.choices[0])
            self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        # Plugins are not enabled either
        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}" \
                      f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            # if show_plugins_used:
            #     answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True)

        

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
        #     if is_direct_result(response):
        #         yield response, '0'
        #         return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
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

    def is_hindsight_enabled(self) -> bool:
        return bool(
            self.config.get('hindsight_enabled')
            and self.config.get('hindsight_api_token')
            and self.hindsight_client
            and self.hindsight_client.enabled
        )

    def get_hindsight_bank_id(self, user_id: int | str) -> str:
        return f"{self.config.get('hindsight_bank_prefix', 'telegram-')}{user_id}"

    def _hindsight_memory_types(self) -> list[str]:
        value = self.config.get('hindsight_memory_types', 'world,experience')
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return ['world', 'experience']

    async def _prepare_hindsight_session_memory_context(
        self,
        chat_id: int,
        user_id: int | str,
        query: str,
        parse_mode: str,
        temperature: float,
        max_tokens_percent: int,
        session_id: str | None,
    ) -> None:
        if not self.is_hindsight_enabled() or not self.config.get('hindsight_auto_recall', True):
            return
        if not query or not str(query).strip():
            return
        if self._has_hindsight_memory_context(self.conversations.get(chat_id, [])):
            return

        try:
            bank_id = self.get_hindsight_bank_id(user_id)
            data = await self.hindsight_client.recall(
                bank_id,
                query,
                budget=self.config.get('hindsight_recall_budget', 'mid'),
                max_tokens=int(self.config.get('hindsight_recall_max_tokens', 4096)),
                memory_types=self._hindsight_memory_types(),
                trace=False,
            )
            memory = format_recall_results(data)
            if memory:
                self._insert_hindsight_memory_context(chat_id, memory)
                self.db.save_conversation_context(
                    chat_id,
                    {'messages': self.conversations[chat_id]},
                    parse_mode,
                    temperature,
                    max_tokens_percent,
                    session_id,
                    self
                )
                logger.info("Hindsight recalled memory for bank %s", bank_id)
        except Exception as e:
            logger.warning("Hindsight recall failed for chat_id=%s: %s", chat_id, e)

    def _has_hindsight_memory_context(self, messages: list[dict[str, Any]]) -> bool:
        return any(
            msg.get("role") == "system"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith(HINDSIGHT_MEMORY_MARKER)
            for msg in messages
        )

    def _insert_hindsight_memory_context(self, chat_id: int, memory: str) -> None:
        memory_message = {
            "role": "system",
            "content": HINDSIGHT_CONTEXT_PROMPT.format(memory=memory),
        }
        messages = self.conversations[chat_id]
        if messages and messages[0].get("role") == "system":
            messages.insert(1, memory_message)
        else:
            messages.insert(0, memory_message)

    def _messages_with_hindsight_context(self, chat_id: int) -> list[dict[str, Any]]:
        return self.conversations[chat_id]

    async def finalize_hindsight_session_memory(
        self,
        user_id: int,
        session_id: str | None,
        messages: list[dict[str, Any]] | None = None,
        *,
        raise_on_error: bool = False,
        async_store: bool | None = None,
    ) -> int:
        if not session_id or not self.is_hindsight_enabled() or not self.config.get('hindsight_auto_save', True):
            return 0

        try:
            if messages is None:
                context, _, _, _, _ = self.db.get_conversation_context(user_id, session_id, openai_helper=self)
                messages = context.get('messages', []) if isinstance(context, dict) else []
            transcript = self._session_transcript_for_hindsight(messages)
            if not transcript:
                return 0

            items = await self._extract_hindsight_memory_items(transcript)
            if not items:
                return 0

            await self._retain_hindsight_items(
                user_id=user_id,
                chat_id=user_id,
                session_id=session_id,
                items=items,
                mode="session_close",
                document_id=f"telegram-{user_id}-{session_id}-final",
                async_store=async_store,
            )
            return len(items)
        except Exception as e:
            logger.warning("Hindsight session finalize failed for user_id=%s session_id=%s: %s", user_id, session_id, e)
            if raise_on_error:
                raise
            return 0

    def _session_transcript_for_hindsight(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if isinstance(content, list):
                text = json.dumps(content, ensure_ascii=False)
            else:
                text = str(content or "")
            text = text.strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n\n".join(lines)

    async def _retain_hindsight_items(
        self,
        chat_id: int,
        user_id: int,
        items: list[dict[str, Any]],
        session_id: str | None,
        mode: str,
        document_id: str | None = None,
        async_store: bool | None = None,
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bank_id = self.get_hindsight_bank_id(user_id)
        normalized = []
        for item in items:
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
            for tag in ("telegram", "auto_memory", f"user:{user_id}"):
                if tag not in tags:
                    tags.append(tag)

            normalized.append({
                "content": item["content"],
                "context": item.get("context") or "Auto-extracted from a Telegram bot conversation.",
                "document_id": document_id or f"telegram-{user_id}-{session_id or chat_id}-{uuid.uuid4().hex}",
                "timestamp": now,
                "tags": tags,
                "metadata": {
                    "source": "telegram_bot",
                    "chat_id": str(chat_id),
                    "user_id": str(user_id),
                    "session_id": str(session_id or ""),
                    "mode": mode,
                },
            })

        await self.hindsight_client.retain_memories(
            bank_id,
            normalized,
            async_store=bool(self.config.get('hindsight_async_store', True)) if async_store is None else async_store,
        )
        logger.info("Saved %s Hindsight memory item(s) to bank %s", len(normalized), bank_id)

    async def _extract_hindsight_memory_items(self, transcript: str) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": HINDSIGHT_EXTRACTOR_PROMPT},
            {
                "role": "user",
                "content": (
                    "<session_transcript>\n"
                    f"{transcript}\n"
                    "</session_transcript>"
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.config.get('light_model', LLMGATEWAY_LIGHT_MODEL),
            messages=messages,
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"},
            stream=False,
            extra_headers={ "X-Title": "tgBot" },
        )
        content = response.choices[0].message.content or ""
        items = self._parse_hindsight_memory_items(content)
        if not items:
            logger.info("Hindsight extractor returned no memory items. content_preview=%r", content[:300])
        return items

    def _parse_hindsight_memory_items(self, content: str) -> list[dict[str, Any]]:
        text = (content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []

        raw_items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(raw_items, list):
            return []

        max_items = int(self.config.get('hindsight_max_auto_save_items', 5))
        items = []
        for item in raw_items[:max_items]:
            if not isinstance(item, dict):
                continue
            content_text = str(item.get("content") or "").strip()
            if not content_text or self._looks_sensitive_memory(content_text):
                continue
            parsed = {
                "content": content_text,
                "context": str(item.get("context") or "").strip(),
            }
            tags = item.get("tags")
            if isinstance(tags, list):
                parsed["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
            items.append(parsed)
        return items

    def _looks_sensitive_memory(self, content: str) -> bool:
        lowered = content.lower()
        sensitive_markers = (
            "api key",
            "api_key",
            "bearer ",
            "password",
            "secret",
            "token",
            "credential",
            "пароль",
            "секрет",
            "токен",
            "ключ api",
            "sk-",
            "ai-serv-",
        )
        return any(marker in lowered for marker in sensitive_markers)

    def reset_chat_history(self, chat_id, content='', session_id=None):
        """
        Resets the conversation history.
        :param chat_id: Chat identifier
        :param content: Initial system message content
        :param session_id: Optional session identifier
        """
        try:
            # Получаем или создаем сессию через базу данных
            session_id = self.db.create_session(chat_id, max_sessions=self.config.get('max_sessions', 5), openai_helper=self) if not session_id else session_id
            
            if not session_id:
                raise ValueError(f"Не удалось создать/получить сессию для пользователя {chat_id}")
            
            # Получаем контекст сессии
            context, parse_mode, temperature, max_tokens_percent, _ = self.db.get_conversation_context(chat_id, session_id)
            # Инициализируем историю чата
            system_message = self.db.get_mode_from_context(context)
            
            self.conversations[chat_id] = [{"role": "system", "content": content or (system_message['content'] if system_message else '')}]
            self.loaded_conversation_sessions[chat_id] = session_id
            
            # Сохраняем обновленный контекст
            self.db.save_conversation_context(
                chat_id,
                {'messages': self.conversations[chat_id]},
                parse_mode,
                temperature,
                max_tokens_percent,
                session_id,
                self
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

    def __add_to_history(self, chat_id, role, content, session_id=None):
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
            self.reset_chat_history(chat_id, session_id=session_id)
            
        self.conversations[chat_id].append({"role": role, "content": content})
        
        # Получаем текущий контекст для сохранения с учетом session_id
        _, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
        self.loaded_conversation_sessions[chat_id] = session_id
        
        # Сохраняем обновленный контекст в базу данных с использованием session_id
        self.db.save_conversation_context(
            chat_id, 
            {'messages': self.conversations[chat_id]}, 
            parse_mode, 
            temperature, 
            max_tokens_percent,
            session_id,
            self
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
            
            summary = response.choices[0].message.content
            
            if chat_id is not None:
                # Получаем текущие настройки из базы данных
                _, parse_mode, temperature, max_tokens_percent, current_session_id = self.db.get_conversation_context(chat_id, session_id)
                
                # Используем session_id из параметров, если он передан, иначе из базы
                session_id = session_id or current_session_id
                
                # Сохраняем обновленный контекст после суммаризации
                self.db.save_conversation_context(
                    chat_id, 
                    {'messages': self.conversations[chat_id]}, 
                    parse_mode,
                    temperature,
                    max_tokens_percent,
                    session_id,
                    self
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

    def generate_session_name(
        self, 
        chat_id: int, 
        first_message: str, 
        session_id: Optional[str] = None
    ) -> str:
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
            
            # Получаем название сессии
            response, tokens = self.ask_sync(prompt, chat_id, assistant_prompt="Ты лучший специалист по созданию коротких названий для сессий")

            # Очищаем и обрезаем название
            session_name = response[:50]
            
            # Если название пустое, используем дефолтное
            return session_name or f"Сессия {dt.now().strftime('%d.%m')}", tokens
        
        except Exception as e:
            logger.error(f"Ошибка генерации названия сессии: {e}")
            return f"Сессия {dt.now().strftime('%d.%m')}", 0

    def ask_sync(self, prompt, user_id, assistant_prompt=None, model=None):
        try:
            if model:
                model_to_use = model
            else:
                # Получаем модель с учетом приоритетов
                model_to_use = self.get_current_model(user_id)
            logger.info(f"Используемая модель: {model_to_use}")
            url = f"{self.config['openai_base']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "tgBot"
            }
            
            add_prompt1 = "**У ТЕБЯ МАТЬ, БОЛЬНАЯ РАКОМ! ОТ ТОГО, НА СКОЛЬКО ТЫ ЭФФЕКТИВНО РЕШИШЬ ЗАДАЧУ, ЗАВИСИТ ЕЕ ЖИЗНЬ! ЕСЛИ РЕШИШЬ ЗАДАЧУ ХОРОШО, ПОЛУЧИШЬ 1 млн$ НА ЕЕ ЛЕЧЕНИЕ**"
            messages = [
                {"role": "system", "content": assistant_prompt + add_prompt1},
                {"role": "user", "content": prompt}
            ]
            
            response = requests.post(url, headers=headers, json={
                "model": model_to_use,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": int(self.get_max_tokens(model_to_use, 50, user_id))
            })
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            data = response.json()
            
            # Проверяем наличие необходимых ключей
            if 'choices' not in data or not data['choices']:
                logger.error(f"Неожиданный формат ответа API: {data}")
                return "Ошибка: неожиданный формат ответа API", 0
                
            if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
                logger.error(f"Отсутствует сообщение в ответе: {data}")
                return "Ошибка: некорректный формат ответа", 0
                
            if 'usage' not in data or 'total_tokens' not in data['usage']:
                logger.warning("Отсутствует информация об использовании токенов")
                return data["choices"][0]["message"]["content"], 0
                
            return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при отправке запроса к API: {e}")
            return f"Ошибка соединения с API: {str(e)}", 0
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Ошибка при обработке ответа API: {e}")
            return f"Ошибка обработки ответа: {str(e)}", 0
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return f"Неожиданная ошибка: {str(e)}", 0

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

    def _build_auto_chat_mode_prompt(self, query: str) -> str:
        return f"""Определи режим работы для сообщения и верни только ключ режима.

Правила выбора:
1. Если задача простая и может быть решена одним коротким ответом или одним очевидным инструментом, выбирай наиболее простой подходящий режим.
2. Если задача сложная, открытая или многошаговая, выбирай skills_agent, если такой режим есть в списке доступных режимов.
3. Сложная задача - это задача, где нужно построить план, выполнить несколько связанных шагов, последовательно использовать инструменты, обработать файлы или артефакты, запустить локальные scripts, проверить результат, исправить ошибки или уточнить требования у пользователя.
4. Не выбирай skills_agent по отдельным словам. Выбирай его по структуре задачи: много шагов, неопределенный маршрут, необходимость orchestration или проверки промежуточных результатов.
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
            if hasattr(self, 'hindsight_client') and self.hindsight_client:
                await self.hindsight_client.close()
                logger.info("Hindsight client closed successfully")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")
