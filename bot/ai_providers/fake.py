from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable

from bot.ai_events import AIEvent, AIMessage, AIResponseEnd, AITextDelta
from bot.ai_provider import AIProviderRequest


class FakeAIProvider:
    def __init__(self, events: Iterable[AIEvent] = ()):
        self._event_batches: deque[tuple[AIEvent, ...]] = deque()
        if events:
            self.queue_events(events)
        self.requests: list[AIProviderRequest] = []

    @classmethod
    def text(cls, content: str) -> "FakeAIProvider":
        provider = cls()
        provider.queue_text(content)
        return provider

    def queue_text(self, content: str) -> None:
        self.queue_events((
            AITextDelta(content),
            AIResponseEnd(message=AIMessage(role="assistant", content=content)),
        ))

    def queue_events(self, events: Iterable[AIEvent]) -> None:
        self._event_batches.append(tuple(events))

    def assert_no_pending_events(self) -> None:
        if self._event_batches:
            raise AssertionError(f"{len(self._event_batches)} queued event batches were not consumed")

    async def stream_response(self, request: AIProviderRequest):
        self.requests.append(request)
        if not self._event_batches:
            raise AssertionError("FakeAIProvider has no queued event batch")
        for event in self._event_batches.popleft():
            await asyncio.sleep(0)
            yield event
