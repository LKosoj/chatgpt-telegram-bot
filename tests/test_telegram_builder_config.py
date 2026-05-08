import importlib.util
import sys
import types

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
    def set_openai(self, openai):
        self.openai = openai

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


class FakeApplication:
    def __init__(self):
        self.bot = object()
        self.handlers = []
        self.error_handlers = []
        self.run_polling_calls = 0

    def add_handler(self, handler, group=0):
        self.handlers.append((handler, group))

    def add_error_handler(self, handler):
        self.error_handlers.append(handler)

    def run_polling(self):
        self.run_polling_calls += 1


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
    bot._background_tasks = []
    bot._plugin_message_handlers_registered = False
    return bot


def _run_bot_with_fake_builder(monkeypatch, config=None):
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
    bot.run()
    return builder, application


def _set_required_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("TELEGRAM_LOCAL_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_BASE_URL", raising=False)


def _run_main_with_fake_dependencies(monkeypatch):
    CapturingTelegramBot.instances.clear()
    monkeypatch.setattr(bot_main, "load_dotenv", lambda: None)
    monkeypatch.setattr(bot_main, "PluginManager", lambda config: object())
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


def test_main_defaults_telegram_local_bot_api_config(monkeypatch):
    _set_required_env(monkeypatch)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["telegram_local_mode"] is True
    assert bot.config["telegram_base_url"] == "http://localhost:8081/bot"
    assert bot.run_calls == 1


def test_main_defaults_bot_language_to_auto(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("BOT_LANGUAGE", raising=False)

    bot = _run_main_with_fake_dependencies(monkeypatch)

    assert bot.config["bot_language"] == "auto"
    assert bot.openai.config["bot_language"] == "auto"


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
