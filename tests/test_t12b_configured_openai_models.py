"""
Test for T12b cluster B6: ``ChatGPTTelegramBot._configured_openai_models()``'s
fallback block (used when ``self.openai`` has no ``get_model_choices``)
mirrors the shape of ``OpenAIHelper.get_model_choices()`` and is replaced by
a call to the shared ``bot.utils.parse_model_choices`` helper (T12a). The
canonical original does NOT deduplicate entries, so this locks that in.
"""
import importlib.util
import sys
import types
from types import SimpleNamespace

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

from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _make_bot(config):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.openai = SimpleNamespace(config=config)
    return bot


def test_configured_openai_models_uses_helper_get_model_choices_when_available():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.openai = SimpleNamespace(get_model_choices=lambda: ["from-helper"])
    assert bot._configured_openai_models() == ["from-helper"]


def test_configured_openai_models_fallback_splits_and_strips_string_choices():
    bot = _make_bot({"model_choices": "gpt-4, gpt-3.5 ", "model": "gpt-4"})
    assert bot._configured_openai_models() == ["gpt-4", "gpt-3.5"]


def test_configured_openai_models_fallback_inserts_missing_default_at_front():
    bot = _make_bot({"model_choices": "gpt-3.5", "model": "gpt-4"})
    assert bot._configured_openai_models() == ["gpt-4", "gpt-3.5"]


def test_configured_openai_models_fallback_does_not_dedupe():
    """The original fallback loop never deduplicated -- only
    OpenAIHelper.get_model_choices()'s absence triggers it, and duplicate
    config entries must be preserved exactly as configured."""
    bot = _make_bot({"model_choices": "gpt-4, gpt-4, gpt-3.5", "model": "gpt-4"})
    assert bot._configured_openai_models() == ["gpt-4", "gpt-4", "gpt-3.5"]


def test_configured_openai_models_fallback_accepts_list_choices():
    bot = _make_bot({"model_choices": [" gpt-4 ", "gpt-3.5"], "model": "gpt-4"})
    assert bot._configured_openai_models() == ["gpt-4", "gpt-3.5"]
