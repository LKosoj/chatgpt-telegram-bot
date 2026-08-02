from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    chat_id: int
    user_id: int
    message_id: int | None = None
    session_id: str | None = None
    request_id: str | None = None
    # True only when explicitly set by an autonomous trigger (e.g. agent_cron). There is no
    # auto-detection: cron reuses the chat's active session, so a session mixes live and
    # autonomous turns; marking the whole session autonomous would risk losing real user facts.
    autonomous: bool = False

    @property
    def plugin_chat_id(self) -> int:
        return int(self.chat_id)


__all__ = ["RequestContext"]
