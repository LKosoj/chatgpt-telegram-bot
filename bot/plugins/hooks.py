"""Hook event names and payload dataclasses for the plugin hooks framework.

This module is dependency-free (stdlib only) and is consumed by both
``PluginManager`` (dispatcher) and individual plugins (subscribers).

Payloads are immutable (``frozen=True``) to make them safe to share across
concurrently running observer hooks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum


if sys.version_info >= (3, 11):
    from enum import StrEnum

    class HookEvent(StrEnum):
        ON_USER_MESSAGE = "on_user_message"
        ON_ASSISTANT_RESPONSE = "on_assistant_response"
        ON_SESSION_RESET = "on_session_reset"
        ON_SESSION_BEFORE_DELETE = "on_session_before_delete"
        ON_BEFORE_CHAT_REQUEST = "on_before_chat_request"
else:  # pragma: no cover - safety fallback for older interpreters
    class HookEvent(str, Enum):
        ON_USER_MESSAGE = "on_user_message"
        ON_ASSISTANT_RESPONSE = "on_assistant_response"
        ON_SESSION_RESET = "on_session_reset"
        ON_SESSION_BEFORE_DELETE = "on_session_before_delete"
        ON_BEFORE_CHAT_REQUEST = "on_before_chat_request"


@dataclass(frozen=True, slots=True)
class UserMessagePayload:
    chat_id: int
    user_id: int | None
    request_id: str | None
    text: str
    has_image: bool
    has_voice: bool
    is_command: bool
    ts: float


@dataclass(frozen=True, slots=True)
class AssistantResponsePayload:
    chat_id: int
    user_id: int | None
    request_id: str | None
    text: str
    tokens: int
    model: str
    ts: float


@dataclass(frozen=True, slots=True)
class SessionResetPayload:
    chat_id: int
    user_id: int | None
    reason: str
    terminal_only: bool


@dataclass(frozen=True, slots=True)
class SessionBeforeDeletePayload:
    user_id: int
    session_id: str
    messages: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class BeforeChatRequestPayload:
    chat_id: int
    user_id: int | None
    request_id: str | None


@dataclass(frozen=True, slots=True)
class PromptFragmentPayload:
    slot: str
    chat_id: int
    user_id: int | None
    query: str


@dataclass(frozen=True, slots=True)
class StatsBlockPayload:
    user_id: int | None
    chat_id: int | None
    bot_language: str


@dataclass(frozen=True, slots=True)
class SettingsMenuPayload:
    user_id: int | None
    bot_language: str
