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
    def plugin_chat_id(self) -> int:
        return int(self.chat_id)


__all__ = ["RequestContext"]
