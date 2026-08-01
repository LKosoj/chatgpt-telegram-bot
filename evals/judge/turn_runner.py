"""Drives one real chat turn through ``OpenAIHelper.get_chat_response`` with a
REAL ``openai.AsyncOpenAI`` client, so the LLM-as-judge cases in
``test_response_quality.py`` exercise exactly the same code path production
traffic uses (chat mode prompt -> model -> tool call -> tool result -> model).

Only the DB and PluginManager are faked. The fakes below are modeled on
``tests/test_openai_helper_tool_calls.py:81`` (``DummyDB``) and
``tests/test_openai_helper_tool_calls.py:220`` (``DummyPluginManager``), but
are NOT imported from that file: ``evals/`` is intentionally outside
``pytest.ini``'s ``testpaths`` (see ``evals/README.md``), so importing private
names from a 159KB test file would let a refactor there break this eval
silently. Each fake here implements only the methods actually invoked along
``OpenAIHelper.get_chat_response`` -> ``ChatRun.run_non_stream`` ->
``handle_function_call`` (bot/chat_run.py, bot/openai_tool_handler.py), plus
``has_plugin`` which ``ChatModesRegistry.validate_tools()`` calls once at
``OpenAIHelper.__init__`` time (bot/chat_modes_registry.py:91).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from bot.openai_helper import OpenAIHelper


@dataclass(frozen=True)
class TurnResult:
    reply: str
    tools_called: tuple[str, ...]
    total_tokens: int


class _FakeDB:
    """Minimal DB stub. OpenAIHelper only talks to the DB through the
    ``*_async`` methods routed by ``OpenAIHelper._db_call``
    (bot/openai_helper.py:711-715); the sync methods are never called on this
    path, so they are intentionally omitted (sample: DummyDB,
    tests/test_openai_helper_tool_calls.py:81-109).

    ``get_conversation_context_async`` returns an empty context on purpose:
    with no saved 'messages', ``OpenAIHelper.resolve_allowed_plugins``
    (bot/openai_helper.py:1270, default set at :1272) short-circuits to ``['All']``
    instead of trying to recover a chat mode from a DB-stored system
    message -- the chat mode here is carried by ``build_real_helper`` seeding
    ``helper.conversations`` directly, not by the DB.
    """

    def __init__(self) -> None:
        self.context: dict = {}

    async def get_conversation_context_async(self, chat_id, session_id=None, *args, **kwargs):
        return self.context, None, 0.1, 80, None

    async def save_conversation_context_async(self, *args, **kwargs):
        # Returning None (falsy) keeps OpenAIHelper._save_conversation_context
        # from spawning the background session-naming task, which would
        # otherwise fire an extra, untracked LLM call
        # (bot/openai_helper.py:696-702).
        return None

    async def list_user_sessions_async(self, user_id, is_active=1):
        return []

    async def get_user_settings_async(self, user_id):
        return None


class _FakePluginManager:
    """Minimal plugin-manager stub (sample: DummyPluginManager,
    tests/test_openai_helper_tool_calls.py:220-278). ``tool_specs`` are
    handed back verbatim from ``get_functions_specs`` regardless of
    ``allowed_plugins`` -- the eval cases decide which tools are on the table
    by constructing ``tool_specs``, not by chat-mode restriction.
    """

    def __init__(self, tool_specs, tool_responses) -> None:
        self.specs = list(tool_specs or [])
        self.responses = dict(tool_responses or {})
        self.plugins: dict = {}
        self.calls_made: list[str] = []
        self.db = None

    def set_db(self, db) -> None:
        self.db = db

    def has_plugin(self, plugin_name) -> bool:
        return True

    def disabled_plugins_for_user(self, user_id):
        return set()

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return self.specs

    def get_plugin_source_name(self, function_name):
        return function_name.split(".", 1)[0]

    def get_plugin(self, plugin_name):
        return self.plugins.get(plugin_name)

    def is_function_allowed(self, function_name, allowed_plugins) -> bool:
        if allowed_plugins == ["All"]:
            return True
        if not allowed_plugins or allowed_plugins == ["None"]:
            return False
        return function_name.split(".", 1)[0] in allowed_plugins

    async def call_function(self, name, helper, arguments, request_context=None):
        self.calls_made.append(name)
        return self.responses[name]

    async def apply_mutators(self, event_name, payload, value, *, user_id=None):
        return value

    async def collect_fragments(self, slot, payload, *, user_id=None):
        return []


def build_real_helper(
    *,
    system_prompt: str,
    tool_specs: list | None = None,
    tool_responses: dict | None = None,
    chat_id: int = 1,
) -> OpenAIHelper:
    """Build an ``OpenAIHelper`` wired to a REAL ``openai.AsyncOpenAI`` client.

    Client construction mirrors ``OpenAIHelper.__init__``
    (bot/openai_helper.py:260-268): ``api_key`` from ``OPENAI_API_KEY``,
    ``base_url`` from ``OPENAI_BASE_URL`` when set (:267-268). Model ids follow the same
    env vars the real bot entrypoint uses (bot/__main__.py:189,248:
    ``OPENAI_MODEL``, ``LIGHT_MODEL``), falling back to a plain default so a
    missing var fails at the (skipped-by-default) network call rather than at
    import time.
    """
    model = (os.environ.get("OPENAI_MODEL", "").split(",")[0] or "gpt-4o-mini").strip()
    light_model = os.environ.get("LIGHT_MODEL", "").split(",")[0].strip()
    config = {
        "openai_base": os.environ.get("OPENAI_BASE_URL", "") or "",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "show_usage": False,
        "stream": False,
        "proxy": None,
        "proxy_web": None,
        "chat_run_variant_b_enabled": True,
        "max_history_size": 15,
        "max_conversation_age_minutes": 180,
        "assistant_prompt": "You are a helpful assistant.",
        "max_tokens": 1000,
        "n_choices": 1,
        "temperature": 0.7,
        "image_model": "",
        "image_quality": "standard",
        "image_style": "vivid",
        "image_size": "512x512",
        "auto_chat_modes": False,
        "model": model,
        "model_choices": [model],
        "enable_functions": True,
        "functions_max_consecutive_calls": 2,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "bot_language": "ru",
        "show_plugins_used": False,
        "whisper_prompt": "",
        "vision_model": "",
        "enable_vision_follow_up_questions": True,
        "vision_prompt": "",
        "vision_detail": "auto",
        "vision_max_tokens": 1000,
        "tts_model": "",
        "tts_voice": "",
        "tts_response_format": "wav",
        "transcription_model": "",
        "yandex_api_token": "",
        "assemblyai_api_key": "",
        "big_model_to_use": "",
        "light_model": light_model,
    }
    plugin_manager = _FakePluginManager(tool_specs, tool_responses)
    db = _FakeDB()
    helper = OpenAIHelper(config=config, plugin_manager=plugin_manager, db=db)
    plugin_manager.set_db(db)
    helper.conversations[chat_id] = [{"role": "system", "content": system_prompt}]
    return helper


async def run_turn(helper: OpenAIHelper, *, user_message: str, chat_id: int = 1, user_id: int = 1) -> TurnResult:
    """Send one user message through the real ``get_chat_response`` path.

    Same entry point production code calls per Telegram message
    (bot/openai_helper.py:828).
    """
    reply, total_tokens = await helper.get_chat_response(
        chat_id=chat_id,
        query=user_message,
        user_id=user_id,
    )
    if not isinstance(reply, str):
        # A dict here means a plugin direct_result short-circuited the model
        # (bot/openai_tool_handler.py:1602, bot/utils.py is_direct_result) --
        # that is a case setup bug (a tool_response shaped as a direct result),
        # not something the judge can score. Fail loudly instead of coercing.
        raise TypeError(
            f"get_chat_response returned non-string reply (direct_result short-circuit?): {reply!r}"
        )
    tools_called = tuple(helper.plugin_manager.calls_made)
    return TurnResult(reply=reply, tools_called=tools_called, total_tokens=int(total_tokens))
