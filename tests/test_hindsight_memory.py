from __future__ import annotations

import asyncio
import json
import types

import httpx
import pytest

pytest.importorskip("tiktoken")

from bot.hindsight_client import HindsightClient, format_recall_results
from bot.openai_helper import OpenAIHelper
from bot.plugins.hindsight_memory import HINDSIGHT_EXTRACTOR_PROMPT, HindsightMemoryPlugin
from bot.telegram_bot import ChatGPTTelegramBot


class DummyDB:
    def __init__(self):
        self.saved = []
        self.context = {"messages": [{"role": "system", "content": "system prompt"}]}

    def list_user_sessions(self, user_id, is_active=1):
        return [{
            "is_active": True,
            "model": "llmgateway/high",
            "session_id": "session-1",
            "max_tokens_percent": 50,
        }]

    def get_conversation_context(self, *args, **kwargs):
        return (
            self.context,
            "Markdown",
            0.1,
            50,
            "session-1",
        )

    def save_conversation_context(self, user_id, context, parse_mode, temperature, max_tokens_percent, session_id=None, openai_helper=None):
        self.context = context
        self.saved.append(context)


class DummyPluginManager:
    db = None
    _hm_plugin = None

    def has_plugin(self, plugin_name):
        return True

    def get_plugin(self, plugin_name):
        if plugin_name != "hindsight_memory":
            return None
        if self._hm_plugin is None:
            self._hm_plugin = HindsightMemoryPlugin()
            self._hm_plugin.initialize(plugin_config={
                'hindsight_base_url': 'http://x',
                'hindsight_api_token': 't',
            })
        return self._hm_plugin

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return []

    def get_plugin_source_name(self, plugin):
        return plugin

    def set_db(self, db):
        self.db = db

    def disabled_plugins_for_user(self, user_id):
        return set()

    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return False

    async def apply_mutators(self, event_name, payload, value, *, user_id=None):
        plugin = self.get_plugin("hindsight_memory")
        method = getattr(plugin, event_name, None)
        if method is None:
            return value
        try:
            new_value = await method(value, payload)
        except Exception:
            return value
        return value if new_value is None else new_value


class FakeHindsight:
    enabled = True

    def __init__(self):
        self.recall_calls = []
        self.retained = []

    async def recall(self, bank_id, query, **kwargs):
        self.recall_calls.append((bank_id, query, kwargs))
        return {
            "results": [{
                "id": "m1",
                "text": "User prefers concise answers.",
                "type": "world",
                "context": "profile",
            }]
        }

    async def retain_memories(self, bank_id, items, **kwargs):
        self.retained.append((bank_id, items, kwargs))
        return {"success": True, "bank_id": bank_id, "items_count": len(items), "async": True}


class SlowFinalizeOpenAI:
    def __init__(self):
        self.config = {"hindsight_auto_save": True}
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls = []
        self.plugin_manager = DummyPluginManager()
        # Force an active hindsight plugin so the bot enqueues the job.
        self.plugin_manager.get_plugin("hindsight_memory")


class FakeCompletions:
    def __init__(self, content="Answer"):
        self.calls = []
        self.content = content

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.content, tool_calls=None)
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(total_tokens=10, prompt_tokens=6, completion_tokens=4)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def make_helper():
    config = {
        "openai_base": "",
        "api_key": "test",
        "show_usage": False,
        "stream": False,
        "proxy": None,
        "proxy_web": None,
        "max_history_size": 5,
        "max_conversation_age_minutes": 60,
        "assistant_prompt": "hi",
        "max_tokens": 1000,
        "n_choices": 1,
        "temperature": 0.1,
        "image_model": "llmgateway/ai-klein-generation",
        "image_quality": "standard",
        "image_style": "vivid",
        "image_size": "512x512",
        "auto_chat_modes": False,
        "model": "llmgateway/high",
        "enable_functions": False,
        "functions_max_consecutive_calls": 2,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "bot_language": "en",
        "show_plugins_used": False,
        "whisper_prompt": "",
        "vision_model": "llmgateway/high",
        "enable_vision_follow_up_questions": True,
        "vision_prompt": "test",
        "vision_detail": "auto",
        "vision_max_tokens": 1000,
        "tts_model": "llmgateway/silero-tts",
        "tts_voice": "kseniya",
        "tts_response_format": "wav",
        "transcription_model": "llmgateway/whisper-large-v3",
        "yandex_api_token": "",
        "assemblyai_api_key": "",
        "big_model_to_use": "llmgateway/big_context",
        "light_model": "llmgateway/light_model",
        "hindsight_auto_save": False,
    }
    helper = OpenAIHelper(config=config, plugin_manager=DummyPluginManager(), db=DummyDB())
    completions = FakeCompletions()
    helper.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    helper.fake_completions = completions
    helper.conversations[1] = [{"role": "system", "content": "system prompt"}]
    # Wire the hindsight plugin to use this helper as its openai backend so the
    # plugin's _extract_hindsight_memory_items can find the fake completions.
    helper.plugin_manager.get_plugin("hindsight_memory").openai = helper
    return helper


@pytest.mark.asyncio
async def test_hindsight_client_uses_expected_rest_paths_and_auth():
    requests = []

    async def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path.endswith("/memories/recall"):
            assert json.loads(request.content)["query"] == "profile"
            return httpx.Response(200, json={"results": [{"id": "1", "text": "Known fact"}]})
        if request.url.path.endswith("/memories"):
            assert json.loads(request.content)["async"] is True
            return httpx.Response(200, json={"success": True, "bank_id": "telegram-123", "items_count": 1, "async": True})
        if request.url.path.endswith("/clear"):
            return httpx.Response(200, json={"success": True, "bank_id": "telegram-123"})
        return httpx.Response(404, json={"error": "unexpected"})

    client = HindsightClient(
        "http://hindsight.local/hindsight",
        "secret",
        transport=httpx.MockTransport(handler),
    )

    recall = await client.recall("telegram-123", "profile")
    retain = await client.retain_memories("telegram-123", [{"content": "Known fact"}])
    clear = await client.clear_bank("telegram-123")
    await client.close()

    assert recall["results"][0]["text"] == "Known fact"
    assert retain["success"] is True
    assert clear["success"] is True
    assert requests[0].url.path == "/hindsight/v1/default/banks/telegram-123/memories/recall"
    assert requests[1].url.path == "/hindsight/v1/default/banks/telegram-123/memories"
    assert requests[2].url.path == "/hindsight/v1/default/banks/telegram-123/clear"


@pytest.mark.asyncio
async def test_recalled_hindsight_memory_is_persisted_once_per_session():
    helper = make_helper()
    helper.config["hindsight_api_token"] = "secret"
    helper.config["hindsight_enabled"] = True
    plugin = helper.plugin_manager.get_plugin("hindsight_memory")
    fake = FakeHindsight()
    plugin.client = fake

    answer, tokens = await helper.get_chat_response(chat_id=1, query="How should you answer me?", user_id=123)

    assert answer == "Answer"
    assert tokens == 10
    assert fake.recall_calls[0][0] == "telegram-123"
    sent_messages = helper.fake_completions.calls[0]["messages"]
    assert any("User prefers concise answers." in message["content"] for message in sent_messages)
    assert any(
        "User prefers concise answers." in str(message.get("content"))
        for message in helper.conversations[1]
    )
    assert any(
        "User prefers concise answers." in str(message.get("content"))
        for message in helper.db.saved[0]["messages"]
    )

    await helper.get_chat_response(chat_id=1, query="And now?", user_id=123)

    assert len(fake.recall_calls) == 1


def test_hindsight_memory_parser_skips_sensitive_content():
    helper = make_helper()
    plugin = helper.plugin_manager.get_plugin("hindsight_memory")
    items = plugin._parse_hindsight_memory_items(json.dumps({
        "items": [
            {"content": "User prefers concise Python examples.", "context": "profile", "tags": ["preference"]},
            {"content": "The API token is ai-serv-example", "context": "secret"},
        ]
    }))

    assert items == [{
        "content": "User prefers concise Python examples.",
        "context": "profile",
        "tags": ["preference"],
    }]


def test_hindsight_is_disabled_without_token():
    helper = make_helper()
    # Re-initialize plugin without token to ensure it's inactive.
    helper.plugin_manager._hm_plugin = HindsightMemoryPlugin()
    helper.plugin_manager._hm_plugin.initialize(plugin_config={})

    assert helper.plugin_manager.get_plugin("hindsight_memory").is_active is False


def test_hindsight_extractor_prompt_rejects_transient_tasks():
    assert "Do not infer preferences from weak signals" in HINDSIGHT_EXTRACTOR_PROMPT
    assert "image generation/editing requests" in HINDSIGHT_EXTRACTOR_PROMPT
    assert "When in doubt, save nothing" in HINDSIGHT_EXTRACTOR_PROMPT


@pytest.mark.asyncio
async def test_finalize_hindsight_session_memory_saves_extracted_items():
    helper = make_helper()
    helper.config["hindsight_api_token"] = "secret"
    helper.config["hindsight_enabled"] = True
    helper.config["hindsight_auto_save"] = True
    plugin = helper.plugin_manager.get_plugin("hindsight_memory")
    fake = FakeHindsight()
    plugin.client = fake
    helper.fake_completions.content = json.dumps({
        "items": [{
            "content": "User prefers concise Python examples.",
            "context": "session preference",
            "tags": ["preference"],
        }]
    })
    helper.db.context = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "I prefer concise Python examples."},
            {"role": "assistant", "content": "Got it."},
        ]
    }

    saved_count = await plugin.finalize_session_memory(123, "session-1", helper.db.context["messages"])

    assert saved_count == 1
    assert helper.fake_completions.calls[0]["response_format"] == {"type": "json_object"}
    assert helper.fake_completions.calls[0]["max_tokens"] == 4000
    assert fake.retained[0][0] == "telegram-123"
    item = fake.retained[0][1][0]
    assert item["content"] == "User prefers concise Python examples."
    assert item["document_id"] == "telegram-123-session-1-final"
    assert item["metadata"]["mode"] == "session_close"


@pytest.mark.asyncio
async def test_finalize_hindsight_session_memory_uses_provided_snapshot():
    helper = make_helper()
    helper.config["hindsight_api_token"] = "secret"
    helper.config["hindsight_enabled"] = True
    helper.config["hindsight_auto_save"] = True
    plugin = helper.plugin_manager.get_plugin("hindsight_memory")
    fake = FakeHindsight()
    plugin.client = fake
    helper.fake_completions.content = json.dumps({
        "items": [{
            "content": "User prefers short answers.",
            "context": "session preference",
        }]
    })
    helper.db.context = {"messages": []}
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "I prefer short answers."},
        {"role": "assistant", "content": "Understood."},
    ]

    saved_count = await plugin.finalize_session_memory(123, "session-1", messages)

    assert saved_count == 1
    assert fake.retained[0][1][0]["content"] == "User prefers short answers."


@pytest.mark.asyncio
async def test_hindsight_session_finalize_is_enqueued_before_delete():
    from bot.plugins.hooks import HookEvent

    bot = object.__new__(ChatGPTTelegramBot)
    bot.db = DummyDB()
    bot.openai = SlowFinalizeOpenAI()
    dispatched = []

    async def _dispatch_blocking(event_name, payload, *, user_id=None):
        dispatched.append((event_name, payload, user_id))

    bot.openai.plugin_manager.dispatch_blocking = _dispatch_blocking

    result = await bot._dispatch_session_before_delete(123, "session-1")

    assert result == 1
    assert len(dispatched) == 1
    event_name, payload, user_id_kw = dispatched[0]
    assert event_name == HookEvent.ON_SESSION_BEFORE_DELETE
    assert payload.user_id == 123
    assert payload.session_id == "session-1"
    assert list(payload.messages) == bot.db.context["messages"]
    assert user_id_kw == 123
    # The synchronous finalize entry point must not be called by the dispatcher.
    assert bot.openai.calls == []


@pytest.mark.asyncio
async def test_hindsight_plugin_recall_uses_user_bank():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
    })
    fake = FakeHindsight()
    plugin.client = fake

    result = await plugin.execute("recall", types.SimpleNamespace(), user_id=123, query="preferences")

    assert result["bank_id"] == "telegram-123"
    assert fake.recall_calls[0][0] == "telegram-123"
    assert "User prefers concise answers." in result["summary"]


def test_format_recall_results_returns_compact_bullets():
    formatted = format_recall_results({
        "results": [{
            "id": "1",
            "text": "User works on the Telegram bot.",
            "type": "experience",
            "context": "project",
        }]
    })

    assert formatted == "- User works on the Telegram bot. (type=experience; context=project)"
