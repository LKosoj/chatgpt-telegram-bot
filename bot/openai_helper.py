from __future__ import annotations
import datetime
import logging
import os
from typing import Dict, List, Optional

import tiktoken

import openai
import requests

from functools import lru_cache
import json
import httpx
import io
from datetime import date
from calendar import monthrange
from PIL import Image
import yaml

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager
from database import Database

logger = logging.getLogger(__name__)

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
GPT_4_VISION_MODELS = ("gpt-4-vision-preview",)
GPT_4_128K_MODELS = ("gpt-4-1106-preview","gpt-4-0125-preview","gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o","gpt-4o-mini")
O_MODELS = ("openai/o1", "openai/o1-preview","openai/o1-mini", "openai/o3-mini","openai/o3-mini-high")
ANTHROPIC = ("anthropic/claude-3-5-haiku","anthropic/claude-3.7-sonnet", "anthropic/claude-3.7-sonnet-thinking-high")
GOOGLE = ("google/gemini-flash-1.5-8b","google/gemini-pro-1.5-online","google/gemini-2.0-flash-001")
MISTRALAI = ("mistralai/mistral-nemo",)
DEEPSEEK = ("deepseek/deepseek-chat","deepseek/deepseek-reasoner","deepseek/deepseek-r1-distill-llama-70b")
GPT_ALL_MODELS = GPT_3_MODELS + GPT_3_16K_MODELS + GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + O_MODELS\
    + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK

@lru_cache(maxsize=128)
def default_max_tokens(model: str = None) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1200
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        if model == "gpt-3.5-turbo-1106":
            return 4096
        return base * 4
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_VISION_MODELS:
        return 4096
    elif model in GPT_4_128K_MODELS:
        return 4096
    elif model in GPT_4O_MODELS:
        return 128000
    elif model in O_MODELS:
        return 128000
    elif model in ANTHROPIC:
        return 200000
    elif model in MISTRALAI:
        return 128000
    elif model in GOOGLE:
        return 900000
    elif model in DEEPSEEK:
        return 65536
    else:
        return base * 2


@lru_cache(maxsize=128)
def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Deprecated models
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314"):
        return False
    # Stable models will be updated to support functions on June 27, 2023
    if model in ("gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4", "gpt-4-32k","gpt-4-1106-preview","gpt-4-0125-preview","gpt-4-turbo-preview"):
        return datetime.date.today() > datetime.date(2023, 6, 27)
    # Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
    if model in ("gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"):
        return datetime.date.today() < datetime.date(2024, 6, 13)
    if model == 'gpt-4-vision-preview':
        return False
    if model in DEEPSEEK:
        return False
    return True


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key


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
        http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        
        if config['openai_base'] != '' :
            openai.api_base = config['openai_base']
        self.api_key = config['api_key']
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client, timeout=300.0, max_retries=3)
        self.config = config
        self.plugin_manager = plugin_manager
        self.db = db
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}
        self.user_id = ''
        self.message_id = ''
        self.message_ids: Dict[int, List] = {}
        self.last_image_file_ids = {}
        self.bot = None

        # Set default values for optional configuration
        self.config.setdefault('temperature', 0.7)
        self.config.setdefault('presence_penalty', 0.0)
        self.config.setdefault('frequency_penalty', 0.0)
        self.config.setdefault('vision_detail', 'auto')
        self.config.setdefault('n_choices', 1)

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
            add_prompt1 = f" Текущая дата и время: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
            if assistant_prompt == None:
                assistant_prompt = "Ты помошник, который отвечает на вопросы пользователя. Ты должен использовать все свои знания и навыки для того, чтобы помочь пользователю. " + add_prompt1

            # Получаем модель с учетом приоритетов
            model_to_use = self.get_current_model(user_id)

            if model_to_use == "deepseek/deepseek-r1-distill-llama-70b":
                assistant_prompt += " Think step by step!"
            messages = [
                {"role": "system", "content": assistant_prompt},
                {"role": "user", "content": prompt}
            ]

            self.__add_to_history(user_id, role="user", content=prompt)
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=self.get_max_tokens(model_to_use, 50),
                temperature=0.6,
                stream=False,
                extra_headers={ "X-Title": "tgBot" }
            )
            self.__add_to_history(user_id, role="assistant", content=response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip(), response.usage.total_tokens
        except Exception as e:
            logging.error(f'Error in ask method: {str(e)}', exc_info=True)
            raise

    async def get_chat_response(self, chat_id: int, query: str, request_id: str = None, session_id: str = None) -> tuple[str, str]:
        """
        Gets a full response from the GPT model with optional session support.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param request_id: Optional request identifier
        :param session_id: Optional session identifier
        :return: The answer from the model and the number of tokens used
        """
        try:
            # Add the last image file ID to the context if available
            if chat_id in self.last_image_file_ids:
                # The model can now access this through the function calls
                self.last_image_file_id = self.last_image_file_ids[chat_id]
            plugins_used = ()
            if request_id and hasattr(self, 'message_ids'):
                self.message_id = self.message_ids.get(request_id)
            
            # Вызов с учетом возможного отсутствия session_id
            response = await self.__common_get_chat_response(
                chat_id, 
                query, 
                session_id=session_id
            )
            
            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                response, plugins_used = await self.__handle_function_call(chat_id, response)
                if is_direct_result(response):
                    logging.debug('Direct result returned, skipping further processing')
                    return response, '0'

            answer = ''

            if len(response.choices) > 1 and self.config['n_choices'] > 1:
                for index, choice in enumerate(response.choices):
                    content = choice.message.content.strip()
                    if index == 0:
                        self.__add_to_history(chat_id, role="assistant", content=content, session_id=session_id)
                    answer += f'{index + 1}\u20e3\n'
                    answer += content
                    answer += '\n\n'
            else:
                answer = response.choices[0].message.content.strip()
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
            logging.error(f'Error in get_chat_response: {str(e)}', exc_info=True)
            raise
        finally:
            # Clean up after response is generated
            if hasattr(self, 'last_image_file_id'):
                delattr(self, 'last_image_file_id')

    async def get_chat_response_stream(self, chat_id: int, query: str, request_id: str = None, session_id: str = None):
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
            logging.info(f'Starting chat response stream for chat_id={chat_id}')
            
            # Проверяем, инициализирован ли контекст разговора
            if chat_id not in self.conversations:
                logging.info(f'Initializing conversation context for chat_id={chat_id}')
                # Пытаемся загрузить контекст из базы данных с учетом session_id
                saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
                
                if saved_context and 'messages' in saved_context:
                    # Если есть сохраненный контекст в БД, используем его
                    self.conversations[chat_id] = saved_context['messages']
                else:
                    # Если нет контекста в БД, начинаем новый чат
                    self.reset_chat_history(chat_id, session_id=session_id)
            
            # Инициализируем conversations_vision для чата, если его нет
            if chat_id not in self.conversations_vision:
                self.conversations_vision[chat_id] = False

            if request_id and hasattr(self, 'message_ids'):
                self.message_id = self.message_ids.get(request_id)
                
            logging.info('Getting chat response from model')
            try:
                # Вызов с учетом возможного отсутствия session_id
                response = await self.__common_get_chat_response(
                    chat_id, 
                    query, 
                    stream=True, 
                    session_id=session_id
                )
            except Exception as e:
                logging.error(f'Error getting chat response: {str(e)}')
                yield f"Error: {str(e)}", '0'
                return

            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                try:
                    response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
                    if is_direct_result(response):
                        yield response, '0'
                        return
                except Exception as e:
                    logging.error(f'Error in function call: {str(e)}')
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
                logging.error(f'Error processing response stream: {str(e)}')
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
            logging.error(f"Error in chat response stream: {e}", exc_info=True)
            # Yield an error message or handle it gracefully
            yield f"Error generating response: {str(e)}", '0'

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False, session_id=None):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            logging.info(f'Generating chat response (chat_id={chat_id}, stream={stream})')
            logging.debug(f'Query: {query}')
            
            # Пытаемся загрузить контекст из базы данных
            saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
                        
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                if saved_context and 'messages' in saved_context:
                    # Если есть сохраненный контекст в БД, используем его
                    self.conversations[chat_id] = saved_context['messages']
                else:
                    # Если нет контекста в БД, начинаем новый чат
                    self.reset_chat_history(chat_id, session_id=session_id)

            # Инициализируем conversations_vision для чата, если его нет
            if chat_id not in self.conversations_vision:
                self.conversations_vision[chat_id] = False

            # Проверяем, является ли это первым сообщением в сессии, если да, то определяем режим работы
            user_messages = [msg for msg in self.conversations[chat_id] if msg['role'] == 'user']
            if len(user_messages) == 0 and self.config['auto_chat_modes']:
                mode_name, _ = self.ask_sync(
                        f"Определи режим работы для сообщения, верни только название режима. Сообщение: ^{query}^. Доступные режимы: ^{self.get_all_modes()}^. Если ни один режим не подходит, верни 'assistant'.",
                        chat_id,
                        "Ты специалист по определению режима работы для сообщений.",
                        model="google/gemini-2.0-flash-001"
                    )
                logging.info(f"🎯 Определен режим для первого сообщения: {mode_name}")
                
                # Загружаем режимы из файла
                current_dir = os.path.dirname(os.path.abspath(__file__))
                chat_modes_path = os.path.join(current_dir, 'chat_modes.yml')
                with open(chat_modes_path, 'r', encoding='utf-8') as file:
                    chat_modes = yaml.safe_load(file)
                
                # Ищем режим по имени
                mode_key = mode_name.strip().lower()
                if mode_key in chat_modes:
                    # Обновляем системное сообщение
                    new_system_prompt = chat_modes[mode_key].get('prompt_start', '')
                    if new_system_prompt:
                        # Заменяем системное сообщение в истории
                        self.conversations[chat_id][0]['content'] = new_system_prompt
                        logging.info(f"🔄 Режим работы изменен на: {mode_key}")
                        
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
            
            self.last_updated[chat_id] = datetime.datetime.now()

            self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)

            user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
            model_to_use = self.get_current_model(user_id)

            # Рассчитываем максимальное количество токенов с учетом процента
            max_tokens = self.get_max_tokens(model_to_use, max_tokens_percent)
            logging.info(f"Model: {model_to_use}, max_tokens: {max_tokens}, max_tokens_percent: {max_tokens_percent}")

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id], model_to_use)
            exceeded_max_tokens = token_count + max_tokens > default_max_tokens(model_to_use)
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1], user_id, session_id)
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'], session_id)
                    self.__add_to_history(chat_id, role="assistant", content=summary, session_id=session_id)
                    self.__add_to_history(chat_id, role="user", content=query, session_id=session_id)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            logging.info(f"Model: {model_to_use}")

            common_args = {
                'model': model_to_use, #if not self.conversations_vision[chat_id] else self.config['vision_model'],
                'messages': self.conversations[chat_id],
                'temperature': temperature,
                'n': 1, # several choices is not implemented yet
                'max_tokens': max_tokens,
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }

            if model_to_use in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK):
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
            if model_to_use == "deepseek/deepseek-r1-distill-llama-70b":
                # Находим системное сообщение и добавляем инструкцию
                for msg in self.conversations[chat_id]:
                    if msg['role'] == 'system':
                        msg['content'] += " Think step by step!"
                        break

            if self.config['enable_functions'] and not self.conversations_vision.get(chat_id, False):
                # Получаем текущий режим из контекста
                saved_context, _, _, _, _ = self.db.get_conversation_context(chat_id, session_id)
                allowed_plugins = ['All']  # По умолчанию разрешены все плагины
                
                if saved_context and 'messages' in saved_context:
                    system_message = next(
                        (msg for msg in saved_context['messages'] if msg.get('role') == 'system'),
                        None
                    )
                    
                    # Загружаем режимы из файла
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    chat_modes_path = os.path.join(current_dir, 'chat_modes.yml')
                    
                    with open(chat_modes_path, 'r', encoding='utf-8') as file:
                        chat_modes = yaml.safe_load(file)
                    
                    # Ищем текущий режим по системному сообщению
                    current_mode = None
                    for mode_key, mode_data in chat_modes.items():
                        if mode_data.get('prompt_start', '').strip() == system_message.get('content', '').strip():
                            current_mode = mode_data
                            break
                    
                    # Получаем список разрешенных плагинов из режима
                    if current_mode and 'tools' in current_mode:
                        allowed_plugins = current_mode['tools']
                
                # Получаем спецификации функций с учетом разрешенных плагинов
                tools = self.plugin_manager.get_functions_specs(self, model_to_use, allowed_plugins)
                
                if tools and model_to_use not in (O_MODELS + DEEPSEEK):
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'

            c = json.dumps(common_args, ensure_ascii=False)
            logging.info(f"common_args = {c}")
            response = await self.client.chat.completions.create(**common_args)
            
            if stream:
                # For streaming responses, return the stream directly
                return response
            else:
                # For non-streaming responses, log the number of choices and return the response
                logging.debug(f'_______________________Response choices: {len(response.choices)}')
                return response
    
        except openai.RateLimitError as e:
            logging.warning(f'Rate limit error: {str(e)}')
            raise e

        except openai.BadRequestError as e:
            logging.error(f'Bad request error: {str(e)}')
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except ValueError as e:
            logging.error(f'Configuration error: {str(e)}')
            raise Exception(f"⚠️ Configuration error: {str(e)}") from e

        except Exception as e:
            logging.error(f'Unexpected error in chat response generation: {str(e)}', exc_info=True)
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def __handle_function_call(self, chat_id, response, stream=False, times=0, tools_used=()):
        tool_name = ''
        arguments = ''
        try:
            if stream:
                try:
                    async for item in response:
                        if not item.choices:
                            continue
                        if len(item.choices) > 0:
                            first_choice = item.choices[0]
                            if first_choice.delta and first_choice.delta.tool_calls:
                                #Additional logging
                                logger.info("found tool calls")

                                if first_choice.delta.tool_calls[0].function.name:
                                    tool_name += first_choice.delta.tool_calls[0].function.name
                                if first_choice.delta.tool_calls[0].function.arguments:
                                    arguments += first_choice.delta.tool_calls[0].function.arguments
                            elif first_choice.finish_reason and first_choice.finish_reason == 'tool_calls':
                                break
                            else:
                                return response, tools_used
                        else:
                            return response, tools_used
                except openai.APIError as e:
                    logging.error(f"API Error in function call streaming: {e}")
                    return response, tools_used
            else:
                if len(response.choices) > 0:
                    first_choice = response.choices[0]
                    #Additional logging
                    logger.info("found tool calls")
                    logging.info(f"first_choice = {first_choice}")
                    if first_choice.message.tool_calls:
                        if first_choice.message.tool_calls[0].function.name:
                            tool_name += first_choice.message.tool_calls[0].function.name
                        if first_choice.message.tool_calls[0].function.arguments:
                            arguments += first_choice.message.tool_calls[0].function.arguments
                    else:
                        return response, tools_used
                else:
                    return response, tools_used
            logging.info(f'Calling tool {tool_name} with arguments {arguments}')
            
            # Добавляем chat_id в аргументы
            try:
                args = json.loads(arguments)
                args['chat_id'] = chat_id
                args['user_id'] = chat_id
                arguments = json.dumps(args, ensure_ascii=False)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse arguments JSON: {arguments}")
                return response, tools_used
            
            user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
            self.user_id = user_id
            model_to_use = self.get_current_model(user_id)

            tool_response = await self.plugin_manager.call_function(tool_name, self, arguments)
            logging.info(f'Function {tool_name} response: {tool_response}')

            if tool_name not in tools_used:
                tools_used += (tool_name,)

            if is_direct_result(tool_response):
                self.__add_function_call_to_history(chat_id=chat_id, function_name=tool_name,
                                                    content=json.dumps({'result': 'Done, the content has been sent'
                                                                    'to the user.'}))
                return tool_response, tools_used

            self.__add_function_call_to_history(chat_id=chat_id, function_name=tool_name, content=tool_response)

            # Получаем активную сессию для пользователя
            sessions = self.db.list_user_sessions(user_id, is_active=1)
            active_session = next((s for s in sessions if s['is_active']), None)
            session_id = active_session['session_id'] if active_session else None
            max_tokens_percent = active_session['max_tokens_percent'] if active_session else 80

            logging.info(f'Function {tool_name} arguments: {arguments} messages: {self.conversations[chat_id]} session_id: {session_id}')

            if model_to_use == "deepseek/deepseek-r1-distill-llama-70b":
                # Находим системное сообщение и добавляем инструкцию
                for msg in self.conversations[chat_id]:
                    if msg['role'] == 'system':
                        msg['content'] += " Think step by step!"
                        break

            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=self.conversations[chat_id],
                tools=self.plugin_manager.get_functions_specs(self, model_to_use),
                tool_choice='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
                max_tokens=self.get_max_tokens(model_to_use, max_tokens_percent),
                stream=stream,
                extra_headers={ "X-Title": "tgBot" },
            )
            return await self.__handle_function_call(chat_id, response, stream, times + 1, tools_used)
        except Exception as e:
            logging.error(f'Error in function call handling: {str(e)}', exc_info=True)
            raise

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Generates an image from the given prompt using DALL·E model.
        :param prompt: The prompt to send to the model
        :return: The image URL and the image size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config['image_model'],
                quality=self.config['image_quality'],
                style=self.config['image_style'],
                size=self.config['image_size'],
                extra_headers={ "X-Title": "tgBot" },
            )

            if len(response.data) == 0:
                logging.error(f'No response from GPT: {str(response)}')
                raise Exception(
                    f"⚠️ _{localized_text('error', bot_language)}._ "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            return response.data[0].url, self.config['image_size']
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
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus'
            )

            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(
                    model="stt-openai/whisper-1", 
                    file=audio, 
                    prompt=prompt_text,
                    response_format="text",
                    extra_headers={ "X-Title": "tgBot" },
                )
                return result
        except Exception as e:
            logging.exception(e)
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}") from e

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
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)
            else:
                # Загружаем сохраненный контекст из базы данных
                saved_context, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id)
                if saved_context and 'messages' in saved_context:
                    self.conversations[chat_id] = saved_context['messages']

            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break
                self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > default_max_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role':'user', 'content':content}

            common_args = {
                'model': self.config['vision_model'],
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
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e


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
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
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
            
            logging.info(f'Chat history reset for chat_id={chat_id}, session_id={session_id}')
            
        except Exception as e:
            logging.error(f'Error in reset_chat_history: {str(e)}', exc_info=True)
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

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Adds a function call to the conversation history
        """
        # For models that don't support function role, add as a user message
        user_id = next((uid for uid, conversations in self.conversations.items() if conversations == self.conversations[chat_id]), None)
        model_to_use = self.get_current_model(user_id)

        if model_to_use in (ANTHROPIC):
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[chat_id].append({"role": "user", "content": function_result})
        elif model_to_use in (MISTRALAI):
            #function_result = f"Function {function_name} returned: {content}"
            #self.conversations[chat_id].append({"role": "user", "content": function_result})
            self.conversations[chat_id].append({"role": "tool", "name": function_name, "content": content})
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
        self.conversations[chat_id].append({"role": role, "content": content})
        
        # Получаем текущий контекст для сохранения с учетом session_id
        _, parse_mode, temperature, max_tokens_percent, session_id = self.db.get_conversation_context(chat_id, session_id)
        
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

            max_tokens = default_max_tokens(model_to_use) / 2  # Оставляем место для ответа
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
                max_tokens=self.get_max_tokens(model_to_use, 50),  # Явно ограничиваем размер ответа
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
            logging.error(f'Error in summarise: {str(e)}')
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

        if model in GPT_3_MODELS + GPT_3_16K_MODELS:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    num_tokens += len(encoding.encode(value))
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
            logging.error(f"Error processing image for token counting: {e}")
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
            logging.error(f"Error downloading file: {e}")
            raise

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
            return session_name or f"Сессия {datetime.now().strftime('%d.%m')}", tokens
        
        except Exception as e:
            logging.error(f"Ошибка генерации названия сессии: {e}")
            return f"Сессия {datetime.now().strftime('%d.%m')}", 0

    def ask_sync(self, prompt, user_id, assistant_prompt=None, model=None):
        try:
            # Получаем модель с учетом приоритетов
            model_to_use = model or self.get_current_model(user_id)
            url = f"{self.config['openai_base']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "tgBot"
            }
            
            messages = [
                {"role": "system", "content": assistant_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = requests.post(url, headers=headers, json={
                "model": model_to_use,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": int(self.get_max_tokens(model_to_use, 50))
            })
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            data = response.json()
            
            # Проверяем наличие необходимых ключей
            if 'choices' not in data or not data['choices']:
                logging.error(f"Неожиданный формат ответа API: {data}")
                return "Ошибка: неожиданный формат ответа API", 0
                
            if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
                logging.error(f"Отсутствует сообщение в ответе: {data}")
                return "Ошибка: некорректный формат ответа", 0
                
            if 'usage' not in data or 'total_tokens' not in data['usage']:
                logging.warning("Отсутствует информация об использовании токенов")
                return data["choices"][0]["message"]["content"], 0
                
            return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при отправке запроса к API: {e}")
            return f"Ошибка соединения с API: {str(e)}", 0
        except (KeyError, ValueError, TypeError) as e:
            logging.error(f"Ошибка при обработке ответа API: {e}")
            return f"Ошибка обработки ответа: {str(e)}", 0
        except Exception as e:
            logging.error(f"Неожиданная ошибка: {e}")
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
            
            # Проверяем модель в сессии
            if active_session and active_session.get('model', ''):
                logging.info(f"Модель из активной сессии: {active_session.get('model', '')}")
                return active_session.get('model', '')
                            
        # Возвращаем модель по умолчанию
        logging.info(f"Модель по умолчанию: {self.config['model']}")
        return self.config['model']
    
    @lru_cache(maxsize=128)
    def get_max_tokens(self, model_to_use, max_tokens_percent):
        if max_tokens_percent == 100:
            max_tokens_percent = 80
        if model_to_use in DEEPSEEK:
            return 8192
        elif model_to_use in O_MODELS:
            return 22000
        return default_max_tokens(model_to_use) * max_tokens_percent / 100

    @lru_cache(maxsize=256)
    def get_all_modes(self):
        # Получаем все режимы из chat_modes.yml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'chat_modes.yml')
        with open(config_path, 'r') as file:
            modes = yaml.safe_load(file)
            return [f"name: {mode_key}, welcome_message: {mode_data['welcome_message']}" 
                   for mode_key, mode_data in modes.items() 
                   if isinstance(mode_data, dict) and 'welcome_message' in mode_data]
