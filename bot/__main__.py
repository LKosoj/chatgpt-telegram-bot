import logging
import math
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from .model_constants import (
    MAX_OUTPUT_TOKENS,
)
from .pricing import load_model_token_prices
from .plugin_manager import PluginManager
from .openai_helper import OpenAIHelper, default_max_tokens, are_functions_available
from .telegram_bot import ChatGPTTelegramBot
from .database import Database
from .i18n import configured_language
from .utils import log_value_shape


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


def parse_telegram_rich_mode_env(name='TELEGRAM_RICH_MESSAGES', default='auto'):
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ('auto', 'required', 'off'):
        return normalized
    if normalized in ('1', 'true', 'yes', 'on'):
        return 'required'
    if normalized in ('0', 'false', 'no', 'off'):
        return 'off'

    raise ValueError(f'{name} must be one of auto, required, or off')


def _parse_numeric_env(name, default, cast, *, minimum=None):
    """Parse a numeric env var without aborting startup on malformed input.

    Falls back to ``default`` and logs a warning when the value cannot be cast.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == '':
        return default
    try:
        value = cast(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError
        if minimum is not None and value < minimum:
            raise ValueError
        return value
    except (TypeError, ValueError):
        logging.warning(
            'Invalid %s value_shape=%s; falling back to default %s',
            name, log_value_shape(raw, key="value"), default,
        )
        return default


def _parse_numeric_list_env(name, default, cast=float):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return list(default)
    values = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            value = cast(item)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError
            values.append(value)
        except (TypeError, ValueError):
            logging.warning(
                'Invalid %s item_shape=%s; falling back to default %s',
                name, log_value_shape(item, key="value"), default,
            )
            return list(default)
    return values or list(default)


def parse_semicolon_list_env(name):
    raw = os.environ.get(name, '')
    return [item.strip() for item in raw.split(';') if item.strip()]


def parse_model_list_env(name, *, required=False):
    raw = os.environ.get(name, '')
    models = [model.strip() for model in raw.split(',') if model.strip()]
    if required and not models:
        raise ValueError(f'{name} must contain at least one model')
    return models


def first_model_env(name, *, required=False):
    models = parse_model_list_env(name, required=required)
    return models[0] if models else ''


def parse_model_context_windows_env(name='MODEL_CONTEXT_WINDOWS'):
    raw = os.environ.get(name, '')
    windows = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        model, sep, value = item.partition('=')
        model = model.strip()
        value = value.strip()
        if not sep or not model or not value:
            logging.warning(
                'Skipping invalid %s entry_shape=%s',
                name,
                log_value_shape(item, key="value"),
            )
            continue
        try:
            window = int(value)
        except ValueError:
            logging.warning(
                'Skipping invalid %s value for %s value_shape=%s',
                name,
                model,
                log_value_shape(value, key="value"),
            )
            continue
        if window <= 0:
            logging.warning(
                'Skipping non-positive %s value for %s value_shape=%s',
                name,
                model,
                log_value_shape(value, key="value"),
            )
            continue
        windows[model] = window
    return windows


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
    model_choices = parse_model_list_env('OPENAI_MODEL', required=True)
    model = model_choices[0]
    functions_available = are_functions_available(model=model)
    model_context_windows = parse_model_context_windows_env()
    max_tokens_default = model_context_windows.get(model, default_max_tokens(model=model))
    api_key = os.environ['OPENAI_API_KEY']
    bot_language = configured_language(os.environ.get('BOT_LANGUAGE', 'auto'))
    max_sessions = _parse_numeric_env('MAX_SESSIONS', 5, int)
    telegram_rich_messages = parse_telegram_rich_mode_env()
    telegram_rich_drafts = parse_bool_env('TELEGRAM_RICH_DRAFTS', True)
    if os.environ.get('GUEST_BUDGET') is None:
        guest_budget_default = _parse_numeric_env('MONTHLY_GUEST_BUDGET', 100.0, float)
    else:
        guest_budget_default = 100.0

    openai_config = {
        'openai_base': os.environ.get('OPENAI_BASE_URL', ''),
        'api_key': api_key,
        'show_usage': os.environ.get('SHOW_USAGE', 'false').lower() == 'true',
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'stream_include_usage': os.environ.get('STREAM_INCLUDE_USAGE', 'false').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None),
        'proxy_web': os.environ.get('PROXY_WEB', None),
        'telegram_rich_messages': telegram_rich_messages,
        'telegram_rich_drafts': telegram_rich_drafts,
        'chat_run_variant_b_enabled': parse_bool_env('CHAT_RUN_VARIANT_B_ENABLED', True),
        'max_history_size': _parse_numeric_env('MAX_HISTORY_SIZE', 15, int),
        'max_conversation_age_minutes': _parse_numeric_env('MAX_CONVERSATION_AGE_MINUTES', 180, int),
        'assistant_prompt': os.environ.get('ASSISTANT_PROMPT', 'You are a helpful assistant.'),
        'max_tokens': _parse_numeric_env('MAX_TOKENS', max_tokens_default, int),
        'model_context_windows': model_context_windows,
        'output_max_tokens': _parse_numeric_env('OUTPUT_MAX_TOKENS', MAX_OUTPUT_TOKENS, int),
        'n_choices': _parse_numeric_env('N_CHOICES', 1, int),
        'temperature': _parse_numeric_env('TEMPERATURE', 1.0, float),
        'image_model': first_model_env('IMAGE_MODEL'),
        'image_quality': os.environ.get('IMAGE_QUALITY', 'standard'),
        'image_style': os.environ.get('IMAGE_STYLE', 'vivid'),
        'image_size': os.environ.get('IMAGE_SIZE', '512x512'),
        'auto_chat_modes': os.environ.get('AUTO_CHAT_MODES', 'false').lower() == 'true',
        'model': model,
        'model_choices': model_choices,
        'enable_functions': os.environ.get('ENABLE_FUNCTIONS', str(functions_available)).lower() == 'true',
        'functions_max_consecutive_calls': _parse_numeric_env('FUNCTIONS_MAX_CONSECUTIVE_CALLS', 10, int),
        'presence_penalty': _parse_numeric_env('PRESENCE_PENALTY', 0.0, float),
        'frequency_penalty': _parse_numeric_env('FREQUENCY_PENALTY', 0.0, float),
        'bot_language': bot_language,
        'show_plugins_used': os.environ.get('SHOW_PLUGINS_USED', 'false').lower() == 'true',
        'whisper_prompt': os.environ.get('WHISPER_PROMPT', ''),
        'vision_model': first_model_env('VISION_MODEL'),
        'enable_vision_follow_up_questions': os.environ.get('ENABLE_VISION_FOLLOW_UP_QUESTIONS', 'true').lower() == 'true',
        'vision_prompt': os.environ.get('VISION_PROMPT', 'What is in this image'),
        'vision_detail': os.environ.get('VISION_DETAIL', 'auto'),
        'vision_max_tokens': _parse_numeric_env('VISION_MAX_TOKENS', 1000, int),
        'tts_model': first_model_env('TTS_MODEL'),
        'tts_voice': os.environ.get('TTS_VOICE', 'kseniya').lower(),
        'tts_response_format': os.environ.get('TTS_RESPONSE_FORMAT', 'wav'),
        'transcription_model': first_model_env('TRANSCRIPTION_MODEL'),
        'yandex_api_token': os.environ.get('YANDEX_API_TOKEN', ''),
        'assemblyai_api_key': os.environ.get('ASSEMBLYAI_API_KEY', ''),
        'big_model_to_use': first_model_env('BIG_MODEL_TO_USE'),
        'light_model': first_model_env('LIGHT_MODEL'),
        # T4: context summarisation knobs. ``SUMMARY_MODEL`` empty -> helper
        # falls back to ``light_model``/``model``.
        'summary_enabled': parse_bool_env('SUMMARY_ENABLED', True),
        'summary_model': os.environ.get('SUMMARY_MODEL', ''),
        'summary_max_tokens': _parse_numeric_env('SUMMARY_MAX_TOKENS', 400, int),
        'summary_timeout_seconds': _parse_numeric_env('SUMMARY_TIMEOUT_SECONDS', 20.0, float),
        'summary_min_messages_between_runs': _parse_numeric_env('SUMMARY_MIN_MESSAGES_BETWEEN_RUNS', 6, int),
        'summary_target_keep_ratio': _parse_numeric_env('SUMMARY_TARGET_KEEP_RATIO', 0.5, float),
        # hindsight_* keys live in bot/plugins/hindsight_memory.py (Stage 4A migration).
        'session_log_enabled': os.environ.get('SESSION_LOG_ENABLED', 'false').lower() == 'true',
        'session_log_dir': os.environ.get('SESSION_LOG_DIR', './log'),
        'session_log_max_bytes': _parse_numeric_env(
            'SESSION_LOG_MAX_BYTES', 10 * 1024 * 1024, int, minimum=0,
        ),
        'session_log_retention_days': _parse_numeric_env('SESSION_LOG_RETENTION_DAYS', 30, int, minimum=0),
        'session_log_otel_endpoint': os.environ.get('SESSION_LOG_OTEL_ENDPOINT', ''),
        'session_log_otel_service_name': os.environ.get('SESSION_LOG_OTEL_SERVICE_NAME', 'chatgpt-telegram-bot'),
        'session_log_otel_insecure': parse_bool_env('SESSION_LOG_OTEL_INSECURE', True),
        'data_dir': os.environ.get('BOT_DATA_DIR', ''),
        'output_dir': os.environ.get('BOT_OUTPUT_DIR', ''),
        'plots_dir': os.environ.get('BOT_PLOTS_DIR', ''),
        'max_sessions': max_sessions,
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
        'telegram_rich_messages': telegram_rich_messages,
        'telegram_rich_drafts': telegram_rich_drafts,
        'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
        'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
        'enable_image_generation': os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true',
        'enable_transcription': os.environ.get('ENABLE_TRANSCRIPTION', 'true').lower() == 'true',
        'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
        'enable_tts_generation': os.environ.get('ENABLE_TTS_GENERATION', 'true').lower() == 'true',
        'budget_period': os.environ.get('BUDGET_PERIOD', 'monthly').lower(),
        'user_budgets': os.environ.get('USER_BUDGETS', os.environ.get('MONTHLY_USER_BUDGETS', '*')),
        'guest_budget': _parse_numeric_env('GUEST_BUDGET', guest_budget_default, float),
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('TELEGRAM_PROXY', None),
        'voice_reply_transcript': os.environ.get('VOICE_REPLY_WITH_TRANSCRIPT_ONLY', 'false').lower() == 'true',
        'voice_reply_prompts': parse_semicolon_list_env('VOICE_REPLY_PROMPTS'),
        'ignore_group_transcriptions': os.environ.get('IGNORE_GROUP_TRANSCRIPTIONS', 'true').lower() == 'true',
        'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
        'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
        'token_price': _parse_numeric_env('TOKEN_PRICE', 0.002, float),
        'model_token_prices': load_model_token_prices(os.environ.get('MODEL_TOKEN_PRICES', '')),
        'image_prices': _parse_numeric_list_env('IMAGE_PRICES', [0.016, 0.018, 0.02], float),
        'vision_token_price': _parse_numeric_env('VISION_TOKEN_PRICE', 0.01, float),
        'image_receive_mode': os.environ.get('IMAGE_FORMAT', "photo"),
        'tts_model': first_model_env('TTS_MODEL'),
        'tts_response_format': os.environ.get('TTS_RESPONSE_FORMAT', 'wav'),
        'tts_prices': _parse_numeric_list_env('TTS_PRICES', [0.015, 0.030], float),
        'transcription_price': _parse_numeric_env('TRANSCRIPTION_PRICE', 0.006, float),
        'bot_language': bot_language,
        'max_sessions': max_sessions,
        'assemblyai_api_key': os.environ.get('ASSEMBLYAI_API_KEY', ''),
        'telegram_download_bot_id': os.environ.get('TELEGRAM_DOWNLOAD_BOT_ID', ''),
        'telegram_download_dir': os.environ.get('TELEGRAM_DOWNLOAD_DIR', 'media'),
        'retention_cleanup_interval_seconds': _parse_numeric_env(
            'RETENTION_CLEANUP_INTERVAL_SECONDS', 3600, int, minimum=0,
        ),
        'tool_call_event_retention_days': _parse_numeric_env(
            'TOOL_CALL_EVENT_RETENTION_DAYS', 30, int, minimum=0,
        ),
        'image_retention_days': _parse_numeric_env('IMAGE_RETENTION_DAYS', 7, int, minimum=0),
        'usage_retention_days': _parse_numeric_env('USAGE_RETENTION_DAYS', 30, int, minimum=0),
        'data_dir': os.environ.get('BOT_DATA_DIR', ''),
        'output_dir': os.environ.get('BOT_OUTPUT_DIR', ''),
        'plots_dir': os.environ.get('BOT_PLOTS_DIR', ''),
    }

    plugin_config = {
        'plugins': [p.strip() for p in os.environ.get('PLUGINS', '').split(',') if p.strip()]
    }

    # Setup and run ChatGPT and Telegram bot
    plugin_manager = PluginManager(config=plugin_config)
    # Stage 4A: expose openai_config keys to plugins via get_config_prefix().
    plugin_manager.config.update(openai_config)
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
