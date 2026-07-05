import asyncio
import importlib.util
import logging
import sys
import types
from contextlib import suppress

import pytest


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


class _FakeEncoding:
    def encode(self, value):
        return list(value)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: _FakeEncoding()
_tiktoken.get_encoding = lambda _name: _FakeEncoding()
_install_module_if_missing("tiktoken", _tiktoken)

_pydub = types.ModuleType("pydub")
_pydub.AudioSegment = object
_install_module_if_missing("pydub", _pydub)

_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)


def _retry(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_tenacity = types.ModuleType("tenacity")
_tenacity.retry = _retry
_tenacity.stop_after_attempt = lambda *args, **kwargs: None
_tenacity.wait_fixed = lambda *args, **kwargs: None
_tenacity.retry_if_exception_type = lambda *args, **kwargs: None
_install_module_if_missing("tenacity", _tenacity)

from bot import __main__ as bot_main  # noqa: E402
from bot import telegram_bot  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakePluginManager:
    def __init__(self):
        self.config = {}
        self.calls = []

    def set_openai(self, openai):
        self.calls.append("set_openai")
        self.openai = openai

    def set_db(self, db):
        self.calls.append("set_db")
        self.db = db

    def register_plugin_schemas(self):
        return None

    def get_message_handlers(self):
        return []

    def close_all(self):
        return None


class FakeOpenAI:
    def __init__(self):
        self.plugin_manager = FakePluginManager()
        self.bot = None

    async def close(self):
        return None


class CleanupPluginManager:
    def __init__(self, events):
        self.events = events

    async def stop_background_tasks(self, timeout=10.0):
        self.events.append(("stop_background_tasks", timeout))

    async def close_all_async(self):
        self.events.append(("close_all_async", None))

    def close_all(self):
        self.events.append(("close_all", None))


class CleanupOpenAI:
    def __init__(self, events):
        self.events = events
        self.plugin_manager = CleanupPluginManager(events)

    async def close(self):
        self.events.append(("openai_close", None))


class CleanupDb:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append(("db_shutdown", None))


class RetentionDb:
    def __init__(self, *, fail_tool_events=False):
        self.calls = []
        self.fail_tool_events = fail_tool_events

    async def prune_tool_call_events_async(self, *, days):
        self.calls.append(("tool_call_events", days))
        if self.fail_tool_events:
            raise RuntimeError("tool prune failed")
        return 2

    async def prune_old_images_async(self, *, days):
        self.calls.append(("images", days))
        return 3


class RetentionTracker:
    def __init__(self):
        self.calls = []

    def prune_store(self, *, event_days, history_days):
        self.calls.append((event_days, history_days))
        return 4


class RetentionSessionLogger:
    def __init__(self):
        self.calls = 0

    async def cleanup_old_logs(self):
        self.calls += 1
        return 5


class PostInitPluginManager:
    def __init__(self):
        self.start_calls = 0

    async def start_background_tasks(self, application):
        self.start_calls += 1

    def build_bot_commands(self):
        return {"plugin_commands": []}

    def get_message_handlers(self):
        return []


class PostInitAppBot:
    def __init__(self):
        self.set_my_commands_calls = []

    async def set_my_commands(self, *args, **kwargs):
        self.set_my_commands_calls.append((args, kwargs))


class FakeApplication:
    def __init__(self):
        self.bot = object()
        self.handlers = []
        self.error_handlers = []
        self.run_polling_calls = 0
        self.run_polling_kwargs = []
        self.invoke_post_shutdown = False
        self.post_shutdown_callback = None

    def add_handler(self, handler, group=0):
        self.handlers.append((handler, group))

    def add_error_handler(self, handler):
        self.error_handlers.append(handler)

    def run_polling(self, **kwargs):
        self.run_polling_calls += 1
        self.run_polling_kwargs.append(kwargs)
        if self.invoke_post_shutdown and self.post_shutdown_callback is not None:
            asyncio.get_event_loop().run_until_complete(
                self.post_shutdown_callback(self)
            )


class FakeApplicationBuilder:
    def __init__(self, application):
        self.application = application
        self.token_calls = []
        self.local_mode_calls = []
        self.base_url_calls = []

    def token(self, token):
        self.token_calls.append(token)
        return self

    def post_init(self, callback):
        self.post_init_callback = callback
        return self

    def post_shutdown(self, callback):
        self.post_shutdown_callback = callback
        return self

    def concurrent_updates(self, enabled):
        self.concurrent_updates_value = enabled
        return self

    def local_mode(self, enabled):
        self.local_mode_calls.append(enabled)
        return self

    def base_url(self, url):
        self.base_url_calls.append(url)
        return self

    def build(self):
        self.application.post_shutdown_callback = self.post_shutdown_callback
        return self.application


class CapturingTelegramBot:
    instances = []

    def __init__(self, config, openai, db):
        self.config = config
        self.openai = openai
        self.db = db
        self.run_calls = 0
        self.__class__.instances.append(self)

    def run(self):
        self.run_calls += 1


class FakeOpenAIHelper:
    def __init__(self, config, plugin_manager, db):
        self.config = config
        self.plugin_manager = plugin_manager
        self.db = db


def _make_bot(config=None):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "token": "telegram-token",
        "bot_language": "en",
        "enable_image_generation": False,
        "enable_tts_generation": False,
        **(config or {}),
    }
    bot.openai = FakeOpenAI()
    bot.db = object()
    bot.commands = []
    bot.group_commands = []
    bot._background_tasks = []
    bot._plugin_message_handlers_registered = False
    return bot


def _current_policy_loop():
    policy = asyncio.get_event_loop_policy()
    return getattr(getattr(policy, "_local", None), "_loop", None)


def _restore_policy_loop(loop):
    if loop is not None and not loop.is_closed():
        asyncio.set_event_loop(loop)
    else:
        asyncio.set_event_loop(None)


def _run_bot_with_fake_builder(monkeypatch, config=None, *, reuse_current_loop=False):
    application = FakeApplication()
    builder = FakeApplicationBuilder(application)
    monkeypatch.setattr(
        telegram_bot,
        "ApplicationBuilder",
        lambda: builder,
    )
    bot = _make_bot(config)

    async def cleanup():
        return None

    monkeypatch.setattr(bot, "cleanup", cleanup)
    previous_loop = _current_policy_loop()
    if not reuse_current_loop:
        asyncio.set_event_loop(None)
    try:
        bot.run()
    finally:
        if not reuse_current_loop:
            _restore_policy_loop(previous_loop)
    return builder, application


def _set_required_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "llmgateway/high")
    monkeypatch.delenv("TELEGRAM_LOCAL_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_BASE_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_RICH_MESSAGES", raising=False)
    monkeypatch.delenv("TELEGRAM_RICH_DRAFTS", raising=False)


def _run_main_with_fake_dependencies(monkeypatch):
    CapturingTelegramBot.instances.clear()
    monkeypatch.setattr(bot_main, "load_dotenv", lambda: None)
    monkeypatch.setattr(bot_main, "PluginManager", lambda config: FakePluginManager())
    monkeypatch.setattr(bot_main, "Database", lambda: object())
    monkeypatch.setattr(bot_main, "OpenAIHelper", FakeOpenAIHelper)
    monkeypatch.setattr(bot_main, "ChatGPTTelegramBot", CapturingTelegramBot)
    monkeypatch.setattr(
        bot_main,
        "are_functions_available",
        lambda model: True,
    )
    monkeypatch.setattr(bot_main, "default_max_tokens", lambda model: 4096)
    bot_main.main()
    assert len(CapturingTelegramBot.instances) == 1
    return CapturingTelegramBot.instances[0]


def test_default_telegram_builder_uses_local_bot_api(monkeypatch):
    builder, application = _run_bot_with_fake_builder(monkeypatch)

    assert builder.token_calls == ["telegram-token"]
    assert builder.concurrent_updates_value is True
    assert builder.local_mode_calls == [True]
    assert builder.base_url_calls == ["http://localhost:8081/bot"]
    assert application.run_polling_calls == 1
    assert application.run_polling_kwargs == [{"close_loop": True}]


def test_telegram_builder_reuses_existing_current_loop_without_closing_it(monkeypatch):
    previous_loop = _current_policy_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        _builder, application = _run_bot_with_fake_builder(monkeypatch, reuse_current_loop=True)

        assert application.run_polling_calls == 1
        assert application.run_polling_kwargs == [{"close_loop": False}]
        assert not loop.is_closed()
    finally:
        loop.close()
        _restore_policy_loop(previous_loop)


def test_constructor_wires_plugin_db_before_openai():
    openai = FakeOpenAI()
    db = object()

    ChatGPTTelegramBot(
        config={
            "bot_language": "en",
            "enable_image_generation": False,
            "enable_tts_generation": False,
        },
        openai=openai,
        db=db,
    )

    assert openai.plugin_manager.calls == ["set_db", "set_openai"]
    assert openai.plugin_manager.db is db


def test_malformed_plugin_menu_page_size_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("PLUGIN_MENU_PAGE_SIZE", "wide")
    caplog.set_level(logging.WARNING)

    bot = ChatGPTTelegramBot(
        config={
            "bot_language": "en",
            "enable_image_generation": False,
            "enable_tts_generation": False,
        },
        openai=FakeOpenAI(),
        db=object(),
    )

    assert bot.plugin_menu_page_size == 8
    log_text = caplog.text
    assert "Invalid PLUGIN_MENU_PAGE_SIZE value_shape=" in log_text
    assert "redacted" not in log_text
    assert "wide" in log_text


def test_main_defaults_telegram_local_bot_api_config(monkeypatch):
    _set_required_env(monkeypatch)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["telegram_local_mode"] is True
    assert bot.config["telegram_base_url"] == "http://localhost:8081/bot"
    assert bot.config["telegram_rich_messages"] == "auto"
    assert bot.config["telegram_rich_drafts"] is True
    assert bot.openai.config["telegram_rich_messages"] == "auto"
    assert bot.openai.config["telegram_rich_drafts"] is True
    assert bot.run_calls == 1


def test_main_defaults_voice_reply_prompts_to_empty_list(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOICE_REPLY_PROMPTS", raising=False)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["voice_reply_prompts"] == []


def test_main_empty_voice_reply_prompts_to_empty_list(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_REPLY_PROMPTS", "")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["voice_reply_prompts"] == []


def test_main_parses_voice_reply_prompts_as_semicolon_list(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_REPLY_PROMPTS", " ; bot;answer ;;")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["voice_reply_prompts"] == ["bot", "answer"]


def test_main_malformed_numeric_envs_fall_back_without_aborting(monkeypatch, caplog):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("MAX_HISTORY_SIZE", "bad")
    monkeypatch.setenv("TEMPERATURE", "hot")
    monkeypatch.setenv("MAX_SESSIONS", "many")
    monkeypatch.setenv("MONTHLY_GUEST_BUDGET", "free")
    monkeypatch.setenv("TOKEN_PRICE", "cheap")
    monkeypatch.setenv("IMAGE_PRICES", "0.1,broken")
    monkeypatch.setenv("TTS_PRICES", "broken")
    caplog.set_level(logging.WARNING)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["max_history_size"] == 15
    assert bot.openai.config["temperature"] == 1.0
    assert bot.openai.config["max_sessions"] == 5
    assert bot.config["max_sessions"] == 5
    assert bot.config["guest_budget"] == 100.0
    assert bot.config["token_price"] == 0.002
    assert bot.config["image_prices"] == [0.016, 0.018, 0.02]
    assert bot.config["tts_prices"] == [0.015, 0.030]
    log_text = caplog.text
    assert "Invalid MAX_HISTORY_SIZE value_shape=" in log_text
    assert "Invalid TEMPERATURE value_shape=" in log_text
    assert "Invalid MAX_SESSIONS value_shape=" in log_text
    assert "Invalid MONTHLY_GUEST_BUDGET value_shape=" in log_text
    assert "Invalid TOKEN_PRICE value_shape=" in log_text
    assert "Invalid IMAGE_PRICES item_shape=" in log_text
    assert "Invalid TTS_PRICES item_shape=" in log_text
    for raw_value in ("bad", "hot", "many", "free", "cheap", "broken"):
        assert raw_value in log_text


def test_main_non_finite_float_envs_fall_back(monkeypatch, caplog):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TEMPERATURE", "nan")
    monkeypatch.setenv("TOKEN_PRICE", "inf")
    monkeypatch.setenv("IMAGE_PRICES", "0.1,nan")
    caplog.set_level(logging.WARNING)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["temperature"] == 1.0
    assert bot.config["token_price"] == 0.002
    assert bot.config["image_prices"] == [0.016, 0.018, 0.02]
    log_text = caplog.text
    assert "Invalid TEMPERATURE value_shape=" in log_text
    assert "Invalid TOKEN_PRICE value_shape=" in log_text
    assert "Invalid IMAGE_PRICES item_shape=" in log_text
    for raw_value in ("nan", "inf"):
        assert raw_value in log_text


def test_main_uses_same_max_sessions_in_openai_and_telegram_config(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("MAX_SESSIONS", "17")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["max_sessions"] == 17
    assert bot.config["max_sessions"] == 17


def test_main_carries_runtime_path_config_to_openai_and_telegram(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("BOT_DATA_DIR", "/runtime/data")
    monkeypatch.setenv("BOT_OUTPUT_DIR", "/runtime/output")
    monkeypatch.setenv("BOT_PLOTS_DIR", "/runtime/plots")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["data_dir"] == "/runtime/data"
    assert bot.openai.config["output_dir"] == "/runtime/output"
    assert bot.openai.config["plots_dir"] == "/runtime/plots"
    assert bot.config["data_dir"] == "/runtime/data"
    assert bot.config["output_dir"] == "/runtime/output"
    assert bot.config["plots_dir"] == "/runtime/plots"


def test_main_defaults_bot_language_to_auto(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("BOT_LANGUAGE", raising=False)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["bot_language"] == "auto"
    assert bot.openai.config["bot_language"] == "auto"


def test_main_uses_first_openai_model_as_default_and_keeps_choices(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "model-a, model-b,,model-c")
    monkeypatch.setenv("LIGHT_MODEL", "light-a,light-b")
    monkeypatch.setenv("BIG_MODEL_TO_USE", "big-a,big-b")
    monkeypatch.setenv("VISION_MODEL", "vision-a,vision-b")
    monkeypatch.setenv("IMAGE_MODEL", "image-a,image-b")
    monkeypatch.setenv("TTS_MODEL", "tts-a,tts-b")
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "speech-a,speech-b")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["model"] == "model-a"
    assert bot.openai.config["model_choices"] == ["model-a", "model-b", "model-c"]
    assert bot.openai.config["light_model"] == "light-a"
    assert bot.openai.config["big_model_to_use"] == "big-a"
    assert bot.openai.config["vision_model"] == "vision-a"
    assert bot.openai.config["image_model"] == "image-a"
    assert bot.openai.config["tts_model"] == "tts-a"
    assert bot.openai.config["transcription_model"] == "speech-a"
    assert bot.config["tts_model"] == "tts-a"


def test_main_parses_model_context_windows(monkeypatch, caplog):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "model-a")
    monkeypatch.setenv(
        "MODEL_CONTEXT_WINDOWS",
        "model-a=12345, model-b=67890, broken, model-c=nope, model-d=0",
    )
    caplog.set_level(logging.WARNING)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["model_context_windows"] == {
        "model-a": 12345,
        "model-b": 67890,
    }
    assert bot.openai.config["max_tokens"] == 12345
    log_text = caplog.text
    assert "Skipping invalid MODEL_CONTEXT_WINDOWS entry_shape=" in log_text
    assert "Skipping invalid MODEL_CONTEXT_WINDOWS value for model-c value_shape=" in log_text
    assert "Skipping non-positive MODEL_CONTEXT_WINDOWS value for model-d value_shape=" in log_text
    assert "broken" in log_text
    assert "nope" in log_text
    assert "value_shape=0" in log_text


def test_main_parses_session_log_retention_config(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SESSION_LOG_MAX_BYTES", "2048")
    monkeypatch.setenv("SESSION_LOG_RETENTION_DAYS", "7")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["session_log_max_bytes"] == 2048
    assert bot.openai.config["session_log_retention_days"] == 7


def test_main_negative_session_log_retention_config_falls_back(monkeypatch, caplog):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SESSION_LOG_MAX_BYTES", "-1")
    monkeypatch.setenv("SESSION_LOG_RETENTION_DAYS", "-1")
    caplog.set_level(logging.WARNING)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.openai.config["session_log_max_bytes"] == 10 * 1024 * 1024
    assert bot.openai.config["session_log_retention_days"] == 30
    assert any("Invalid SESSION_LOG_RETENTION_DAYS" in rec.message for rec in caplog.records)


def test_main_parses_retention_cleanup_config(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("RETENTION_CLEANUP_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("TOOL_CALL_EVENT_RETENTION_DAYS", "14")
    monkeypatch.setenv("IMAGE_RETENTION_DAYS", "3")
    monkeypatch.setenv("USAGE_RETENTION_DAYS", "21")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["retention_cleanup_interval_seconds"] == 120
    assert bot.config["tool_call_event_retention_days"] == 14
    assert bot.config["image_retention_days"] == 3
    assert bot.config["usage_retention_days"] == 21


def test_main_negative_retention_config_falls_back(monkeypatch, caplog):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("RETENTION_CLEANUP_INTERVAL_SECONDS", "-1")
    monkeypatch.setenv("TOOL_CALL_EVENT_RETENTION_DAYS", "-1")
    monkeypatch.setenv("IMAGE_RETENTION_DAYS", "-1")
    monkeypatch.setenv("USAGE_RETENTION_DAYS", "-1")
    caplog.set_level(logging.WARNING)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["retention_cleanup_interval_seconds"] == 3600
    assert bot.config["tool_call_event_retention_days"] == 30
    assert bot.config["image_retention_days"] == 7
    assert bot.config["usage_retention_days"] == 30
    assert any("Invalid USAGE_RETENTION_DAYS" in rec.message for rec in caplog.records)


def test_missing_openai_model_is_rejected_before_polling(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        _run_main_with_fake_dependencies(monkeypatch)

    assert CapturingTelegramBot.instances == []


def test_main_normalizes_explicit_bot_language(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("BOT_LANGUAGE", "pt_BR")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["bot_language"] == "pt-br"
    assert bot.openai.config["bot_language"] == "pt-br"


def test_main_parses_telegram_local_mode_and_custom_base_url(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_LOCAL_MODE", "true")
    monkeypatch.setenv("TELEGRAM_BASE_URL", "http://telegram-api.local/bot")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["telegram_local_mode"] is True
    assert bot.config["telegram_base_url"] == "http://telegram-api.local/bot"
    assert bot.run_calls == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("required", "required"),
        ("off", "off"),
        ("true", "required"),
        ("on", "required"),
        ("yes", "required"),
        ("1", "required"),
        ("false", "off"),
        ("no", "off"),
        ("0", "off"),
    ],
)
def test_main_parses_explicit_telegram_rich_messages(monkeypatch, raw, expected):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_RICH_MESSAGES", raw)
    monkeypatch.setenv("TELEGRAM_RICH_DRAFTS", "false")

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["telegram_rich_messages"] == expected
    assert bot.config["telegram_rich_drafts"] is False
    assert bot.openai.config["telegram_rich_messages"] == expected
    assert bot.openai.config["telegram_rich_drafts"] is False


def test_telegram_builder_uses_custom_local_base_url(monkeypatch):
    builder, application = _run_bot_with_fake_builder(
        monkeypatch,
        {
            "telegram_local_mode": True,
            "telegram_base_url": "http://telegram-api.local/bot",
        },
    )

    assert builder.local_mode_calls == [True]
    assert builder.base_url_calls == ["http://telegram-api.local/bot"]
    assert application.run_polling_calls == 1


def test_telegram_builder_skips_base_url_when_local_mode_disabled(monkeypatch):
    builder, application = _run_bot_with_fake_builder(
        monkeypatch,
        {
            "telegram_local_mode": False,
            "telegram_base_url": "http://telegram-api.local/bot",
        },
    )

    assert builder.local_mode_calls == [False]
    assert builder.base_url_calls == []
    assert application.run_polling_calls == 1


def test_run_post_shutdown_cleanup_not_repeated_by_finally(monkeypatch):
    application = FakeApplication()
    application.invoke_post_shutdown = True
    builder = FakeApplicationBuilder(application)
    monkeypatch.setattr(
        telegram_bot,
        "ApplicationBuilder",
        lambda: builder,
    )
    bot = _make_bot()
    bot._cleanup_called = False
    calls = []

    async def cleanup():
        if bot._cleanup_called:
            return
        bot._cleanup_called = True
        calls.append("cleanup")

    monkeypatch.setattr(bot, "cleanup", cleanup)
    previous_loop = _current_policy_loop()
    asyncio.set_event_loop(None)
    try:
        bot.run()
    finally:
        _restore_policy_loop(previous_loop)

    assert application.run_polling_kwargs == [{"close_loop": True}]
    assert calls == ["cleanup"]


def test_invalid_telegram_base_url_rejected_before_polling(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_LOCAL_MODE", "true")
    monkeypatch.setenv("TELEGRAM_BASE_URL", "not-a-url")

    with pytest.raises(ValueError, match="TELEGRAM_BASE_URL"):
        _run_main_with_fake_dependencies(monkeypatch)

    assert CapturingTelegramBot.instances == []


def test_invalid_telegram_local_mode_rejected_before_polling(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_LOCAL_MODE", "sometimes")

    with pytest.raises(ValueError, match="TELEGRAM_LOCAL_MODE"):
        _run_main_with_fake_dependencies(monkeypatch)

    assert CapturingTelegramBot.instances == []


def test_invalid_telegram_rich_mode_rejected_before_polling(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_RICH_MESSAGES", "sometimes")

    with pytest.raises(ValueError, match="TELEGRAM_RICH_MESSAGES"):
        _run_main_with_fake_dependencies(monkeypatch)

    assert CapturingTelegramBot.instances == []


@pytest.mark.asyncio
async def test_retention_cleanup_once_prunes_configured_stores():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "tool_call_event_retention_days": 14,
        "image_retention_days": 3,
        "usage_retention_days": 21,
    }
    tracker = RetentionTracker()
    session_logger = RetentionSessionLogger()
    bot.db = RetentionDb()
    bot.usage = {42: tracker}
    bot.openai = types.SimpleNamespace(session_logger=session_logger)

    results = await bot.run_retention_cleanup_once()

    assert results == {
        "tool_call_events": 2,
        "images": 3,
        "usage_trackers": 1,
        "usage_events": 4,
        "session_logs": 5,
    }
    assert bot.db.calls == [("tool_call_events", 14), ("images", 3)]
    assert tracker.calls == [(21, 21)]
    assert session_logger.calls == 1


@pytest.mark.asyncio
async def test_retention_cleanup_once_continues_after_subsystem_failure(caplog):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "tool_call_event_retention_days": 14,
        "image_retention_days": 3,
        "usage_retention_days": 21,
    }
    tracker = RetentionTracker()
    session_logger = RetentionSessionLogger()
    bot.db = RetentionDb(fail_tool_events=True)
    bot.usage = {42: tracker}
    bot.openai = types.SimpleNamespace(session_logger=session_logger)
    caplog.set_level(logging.ERROR)

    results = await bot.run_retention_cleanup_once()

    assert results["tool_call_events"] == 0
    assert results["images"] == 3
    assert results["usage_events"] == 4
    assert results["session_logs"] == 5
    assert any("Failed to prune old tool call events" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_post_init_starts_retention_cleanup_task_when_enabled():
    bot = _make_bot({"retention_cleanup_interval_seconds": 60})
    plugin_manager = PostInitPluginManager()
    bot.openai.plugin_manager = plugin_manager
    application = FakeApplication()
    application.bot = PostInitAppBot()
    started = asyncio.Event()

    async def idle_buffer_checker():
        await asyncio.Event().wait()

    async def idle_retention_loop():
        started.set()
        await asyncio.Event().wait()

    bot.buffer_data_checker = idle_buffer_checker
    bot.retention_cleanup_loop = idle_retention_loop

    try:
        await bot.post_init(application)
        await asyncio.wait_for(started.wait(), timeout=1)

        task_names = {task.get_name() for task in bot._background_tasks}
        assert task_names == {"buffer_data_checker", "retention_cleanup_loop"}
        assert plugin_manager.start_calls == 1
    finally:
        for task in bot._background_tasks:
            task.cancel()
        await asyncio.gather(*bot._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_post_init_skips_retention_cleanup_task_when_disabled():
    bot = _make_bot({"retention_cleanup_interval_seconds": 0})
    plugin_manager = PostInitPluginManager()
    bot.openai.plugin_manager = plugin_manager
    application = FakeApplication()
    application.bot = PostInitAppBot()

    async def idle_buffer_checker():
        await asyncio.Event().wait()

    bot.buffer_data_checker = idle_buffer_checker

    try:
        await bot.post_init(application)

        task_names = {task.get_name() for task in bot._background_tasks}
        assert task_names == {"buffer_data_checker"}
        assert plugin_manager.start_calls == 1
    finally:
        for task in bot._background_tasks:
            task.cancel()
        await asyncio.gather(*bot._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_cancels_only_owned_tasks_and_shuts_db_last():
    events = []
    bot = object.__new__(ChatGPTTelegramBot)
    bot._cleanup_called = False
    bot.openai = CleanupOpenAI(events)
    bot.db = CleanupDb(events)
    bot.buffer_lock = asyncio.Lock()
    bot.media_group_lock = asyncio.Lock()

    async def sleeper(name):
        try:
            await asyncio.sleep(60)
        finally:
            events.append((f"{name}_cancelled", None))

    background = asyncio.create_task(sleeper("background"))
    transient = asyncio.create_task(sleeper("transient"))
    message_timer = asyncio.create_task(sleeper("message_timer"))
    media_timer = asyncio.create_task(sleeper("media_timer"))
    unrelated = asyncio.create_task(sleeper("unrelated"))
    bot._background_tasks = [background]
    bot._transient_tasks = {transient}
    bot.message_buffer = {1: {"timer": message_timer}}
    bot.media_group_buffer = {"g": {"timer": media_timer}}

    try:
        await bot.cleanup()

        assert background.done()
        assert transient.done()
        assert message_timer.done()
        assert media_timer.done()
        assert not unrelated.done()
        assert bot.message_buffer == {}
        assert bot.media_group_buffer == {}
        assert events[0] == ("stop_background_tasks", 10.0)
        assert events[-1] == ("db_shutdown", None)
        assert ("openai_close", None) in events
        assert ("close_all_async", None) in events
        assert ("close_all", None) in events
    finally:
        unrelated.cancel()
        with suppress(asyncio.CancelledError):
            await unrelated
