import logging
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from .model_constants import (
    LLMGATEWAY_BIG_CONTEXT_MODEL,
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_IMAGE_GENERATION_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
    LLMGATEWAY_TRANSCRIPTION_MODEL,
    LLMGATEWAY_TTS_MODEL,
)
from .plugin_manager import PluginManager
from .openai_helper import OpenAIHelper, default_max_tokens, are_functions_available
from .telegram_bot import ChatGPTTelegramBot
from .database import Database
from .i18n import configured_language


DEFAULT_TELEGRAM_BASE_URL = 'http://localhost:8081/bot'


def parse_bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'n', 'off'):
        return False

    raise ValueError(f'{name} must be a boolean value')


def validate_telegram_base_url(value):
    if not value:
        return ''

    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('TELEGRAM_BASE_URL must be an absolute http(s) URL')

    return value


def main():
    # Read .env file
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Check if the required environment variables are set
    required_values = ['TELEGRAM_BOT_TOKEN', 'OPENAI_API_KEY']
    missing_values = [value for value in required_values if os.environ.get(value) is None]
    if len(missing_values) > 0:
        logging.error(f'The following environment values are missing in your .env: {", ".join(missing_values)}')
        exit(1)

    # Setup configurations
    model = os.environ.get('OPENAI_MODEL', LLMGATEWAY_HIGH_MODEL)
    functions_available = are_functions_available(model=model)
    max_tokens_default = default_max_tokens(model=model)
    api_key = os.environ['OPENAI_API_KEY']
    hindsight_base_url = os.environ.get('HINDSIGHT_BASE_URL', '')
    hindsight_api_token = os.environ.get('HINDSIGHT_API_TOKEN', '')
    bot_language = configured_language(os.environ.get('BOT_LANGUAGE', 'auto'))

    openai_config = {
        'openai_base': os.environ.get('OPENAI_BASE_URL', ''),
        'api_key': api_key,
        'show_usage': os.environ.get('SHOW_USAGE', 'false').lower() == 'true',
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None),
        'proxy_web': os.environ.get('PROXY_WEB', None),
        'max_history_size': int(os.environ.get('MAX_HISTORY_SIZE', 15)),
        'max_conversation_age_minutes': int(os.environ.get('MAX_CONVERSATION_AGE_MINUTES', 180)),
        'assistant_prompt': os.environ.get('ASSISTANT_PROMPT', 'You are a helpful assistant.'),
        'max_tokens': int(os.environ.get('MAX_TOKENS', max_tokens_default)),
        'n_choices': int(os.environ.get('N_CHOICES', 1)),
        'temperature': float(os.environ.get('TEMPERATURE', 1.0)),
        'image_model': os.environ.get('IMAGE_MODEL', LLMGATEWAY_IMAGE_GENERATION_MODEL),
        'image_quality': os.environ.get('IMAGE_QUALITY', 'standard'),
        'image_style': os.environ.get('IMAGE_STYLE', 'vivid'),
        'image_size': os.environ.get('IMAGE_SIZE', '512x512'),
        'auto_chat_modes': os.environ.get('AUTO_CHAT_MODES', 'false').lower() == 'true',
        'model': model,
        'enable_functions': os.environ.get('ENABLE_FUNCTIONS', str(functions_available)).lower() == 'true',
        'functions_max_consecutive_calls': int(os.environ.get('FUNCTIONS_MAX_CONSECUTIVE_CALLS', 10)),
        'presence_penalty': float(os.environ.get('PRESENCE_PENALTY', 0.0)),
        'frequency_penalty': float(os.environ.get('FREQUENCY_PENALTY', 0.0)),
        'bot_language': bot_language,
        'show_plugins_used': os.environ.get('SHOW_PLUGINS_USED', 'false').lower() == 'true',
        'whisper_prompt': os.environ.get('WHISPER_PROMPT', ''),
        'vision_model': os.environ.get('VISION_MODEL', LLMGATEWAY_BIG_CONTEXT_MODEL),
        'enable_vision_follow_up_questions': os.environ.get('ENABLE_VISION_FOLLOW_UP_QUESTIONS', 'true').lower() == 'true',
        'vision_prompt': os.environ.get('VISION_PROMPT', 'What is in this image'),
        'vision_detail': os.environ.get('VISION_DETAIL', 'auto'),
        'vision_max_tokens': int(os.environ.get('VISION_MAX_TOKENS', '1000')),
        'tts_model': os.environ.get('TTS_MODEL', LLMGATEWAY_TTS_MODEL),
        'tts_voice': os.environ.get('TTS_VOICE', 'kseniya').lower(),
        'tts_response_format': os.environ.get('TTS_RESPONSE_FORMAT', 'wav'),
        'transcription_model': os.environ.get('TRANSCRIPTION_MODEL', LLMGATEWAY_TRANSCRIPTION_MODEL),
        'yandex_api_token': os.environ.get('YANDEX_API_TOKEN', ''),
        'assemblyai_api_key': os.environ.get('ASSEMBLYAI_API_KEY', ''),
        'big_model_to_use': os.environ.get('BIG_MODEL_TO_USE', LLMGATEWAY_BIG_CONTEXT_MODEL),
        'light_model': os.environ.get('LIGHT_MODEL', LLMGATEWAY_LIGHT_MODEL),
        'hindsight_enabled': bool(hindsight_base_url and hindsight_api_token),
        'hindsight_base_url': hindsight_base_url,
        'hindsight_api_token': hindsight_api_token,
        'hindsight_namespace': os.environ.get('HINDSIGHT_NAMESPACE', 'default'),
        'hindsight_bank_prefix': os.environ.get('HINDSIGHT_BANK_PREFIX', 'telegram-'),
        'hindsight_auto_recall': os.environ.get('HINDSIGHT_AUTO_RECALL', 'true').lower() == 'true',
        'hindsight_auto_save': os.environ.get('HINDSIGHT_AUTO_SAVE', 'true').lower() == 'true',
        'hindsight_recall_budget': os.environ.get('HINDSIGHT_RECALL_BUDGET', 'mid'),
        'hindsight_recall_max_tokens': int(os.environ.get('HINDSIGHT_RECALL_MAX_TOKENS', '4096')),
        'hindsight_memory_types': os.environ.get('HINDSIGHT_MEMORY_TYPES', 'world,experience'),
        'hindsight_async_store': os.environ.get('HINDSIGHT_ASYNC_STORE', 'true').lower() == 'true',
        'hindsight_timeout': float(os.environ.get('HINDSIGHT_TIMEOUT', '30')),
        'hindsight_max_auto_save_items': int(os.environ.get('HINDSIGHT_MAX_AUTO_SAVE_ITEMS', '5')),
    }

    if openai_config['enable_functions'] and not functions_available:
        logging.error(f'ENABLE_FUNCTIONS is set to true, but the model {model} does not support it. '
                        'Please set ENABLE_FUNCTIONS to false or use a model that supports it.')
        exit(1)
    if os.environ.get('MONTHLY_USER_BUDGETS') is not None:
        logging.warning('The environment variable MONTHLY_USER_BUDGETS is deprecated. '
                        'Please use USER_BUDGETS with BUDGET_PERIOD instead.')
    if os.environ.get('MONTHLY_GUEST_BUDGET') is not None:
        logging.warning('The environment variable MONTHLY_GUEST_BUDGET is deprecated. '
                        'Please use GUEST_BUDGET with BUDGET_PERIOD instead.')

    telegram_local_mode = parse_bool_env('TELEGRAM_LOCAL_MODE', True)
    telegram_base_url = os.environ.get(
        'TELEGRAM_BASE_URL',
        DEFAULT_TELEGRAM_BASE_URL if telegram_local_mode else ''
    )
    telegram_base_url = validate_telegram_base_url(telegram_base_url)

    telegram_config = {
        'openai_base': os.environ.get('OPENAI_BASE_URL', ''),
        'api_key': api_key,
        'token': os.environ['TELEGRAM_BOT_TOKEN'],
        'telegram_local_mode': telegram_local_mode,
        'telegram_base_url': telegram_base_url,
        'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
        'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
        'enable_image_generation': os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true',
        'enable_transcription': os.environ.get('ENABLE_TRANSCRIPTION', 'true').lower() == 'true',
        'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
        'enable_tts_generation': os.environ.get('ENABLE_TTS_GENERATION', 'true').lower() == 'true',
        'budget_period': os.environ.get('BUDGET_PERIOD', 'monthly').lower(),
        'user_budgets': os.environ.get('USER_BUDGETS', os.environ.get('MONTHLY_USER_BUDGETS', '*')),
        'guest_budget': float(os.environ.get('GUEST_BUDGET', os.environ.get('MONTHLY_GUEST_BUDGET', '100.0'))),
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('TELEGRAM_PROXY', None),
        'voice_reply_transcript': os.environ.get('VOICE_REPLY_WITH_TRANSCRIPT_ONLY', 'false').lower() == 'true',
        'voice_reply_prompts': os.environ.get('VOICE_REPLY_PROMPTS', '').split(';'),
        'ignore_group_transcriptions': os.environ.get('IGNORE_GROUP_TRANSCRIPTIONS', 'true').lower() == 'true',
        'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
        'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
        'token_price': float(os.environ.get('TOKEN_PRICE', 0.002)),
        'image_prices': [float(i) for i in os.environ.get('IMAGE_PRICES', "0.016,0.018,0.02").split(",")],
        'vision_token_price': float(os.environ.get('VISION_TOKEN_PRICE', '0.01')),
        'image_receive_mode': os.environ.get('IMAGE_FORMAT', "photo"),
        'tts_model': os.environ.get('TTS_MODEL', LLMGATEWAY_TTS_MODEL),
        'tts_response_format': os.environ.get('TTS_RESPONSE_FORMAT', 'wav'),
        'tts_prices': [float(i) for i in os.environ.get('TTS_PRICES', "0.015,0.030").split(",")],
        'transcription_price': float(os.environ.get('TRANSCRIPTION_PRICE', 0.006)),
        'bot_language': bot_language,
        'max_sessions': int(os.environ.get('MAX_SESSIONS', 5)),
        'assemblyai_api_key': os.environ.get('ASSEMBLYAI_API_KEY', ''),
        'telegram_download_bot_id': os.environ.get('TELEGRAM_DOWNLOAD_BOT_ID', ''),
        'telegram_download_dir': os.environ.get('TELEGRAM_DOWNLOAD_DIR', 'media'),
    }

    plugin_config = {
        'plugins': [p.strip() for p in os.environ.get('PLUGINS', '').split(',') if p.strip()]
    }

    # Setup and run ChatGPT and Telegram bot
    plugin_manager = PluginManager(config=plugin_config)
    db = Database()
    plugin_manager.set_db(db)
    # Stage 0 hook wiring: plugins may declare DDL via Plugin.register_schema().
    # On stage 0 the registry is effectively empty (no plugin overrides it).
    # Must run BEFORE set_openai(): set_openai triggers initialize() for every
    # plugin, and plugins may read their tables from inside initialize.
    plugin_manager.register_plugin_schemas()
    openai_helper = OpenAIHelper(config=openai_config, plugin_manager=plugin_manager, db=db)
    # Make the helper available to plugins explicitly, before the bot is built.
    plugin_manager.set_openai(openai_helper)
    telegram_bot = ChatGPTTelegramBot(config=telegram_config, openai=openai_helper, db=db)
    telegram_bot.run()


if __name__ == '__main__':
    main()
