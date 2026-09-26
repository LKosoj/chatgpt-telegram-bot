from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Any

from bot.ai_events import AIEvent, AIMessage, AIResponseEnd, AITextDelta
from bot.ai_provider import AIProviderRequest


class FakeAIProvider:
    def __init__(self, events: Iterable[AIEvent] = ()):
        self._event_batches: deque[tuple[AIEvent, ...]] = deque()
        if events:
            self.queue_events(events)
        self.requests: list[AIProviderRequest] = []
        self.image_calls: list[dict] = []
        self._image_results: deque = deque()
        self.speech_calls: list[dict] = []
        self._speech_results: deque = deque()
        self.transcribe_calls: list[dict] = []
        self._transcribe_results: deque = deque()
        self.models_results: deque = deque()
        self.voices_calls: list[dict] = []
        self._voices_results: deque = deque()

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

    def queue_image(self, value) -> None:
        self._image_results.append(value)

    async def generate_image(self, **kwargs) -> Any:
        self.image_calls.append(kwargs)
        return self._image_results.popleft()

    async def edit_image(self, **kwargs) -> Any:
        self.image_calls.append(kwargs)
        return self._image_results.popleft()

    def queue_speech(self, value) -> None:
        self._speech_results.append(value)

    async def speech(self, **kwargs) -> Any:
        self.speech_calls.append(kwargs)
        return self._speech_results.popleft()

    def queue_transcribe(self, value) -> None:
        self._transcribe_results.append(value)

    async def transcribe(self, **kwargs) -> Any:
        self.transcribe_calls.append(kwargs)
        return self._transcribe_results.popleft()

    def queue_models(self, value) -> None:
        self.models_results.append(value)

    async def list_models(self) -> Any:
        return self.models_results.popleft()

    def queue_voices(self, value) -> None:
        self._voices_results.append(value)

    async def list_voices(self, **kwargs) -> Any:
        self.voices_calls.append(kwargs)
        return self._voices_results.popleft()
