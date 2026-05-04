from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    chat_id: int
    user_id: int
    message_id: int | None = None
    session_id: str | None = None
    request_id: str | None = None

    @property
    def plugin_chat_id(self) -> str:
        return str(self.chat_id)


__all__ = ["RequestContext"]
