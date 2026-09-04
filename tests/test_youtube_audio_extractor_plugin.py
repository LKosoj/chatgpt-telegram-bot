"""Tests for YouTubeAudioExtractorPlugin's blocking-call fix.

``pytubefix`` downloads over ``urllib.request``, which blocks. Called straight
from ``async execute`` it would freeze the single event-loop thread that serves
every chat of every user for the whole download.
"""

import asyncio
import threading

import pytest

import bot.plugins.youtube_audio_extractor as extractor
from bot.plugins.youtube_audio_extractor import YouTubeAudioExtractorPlugin


class FakeStream:
    def __init__(self, recorder):
        self._recorder = recorder

    def download(self, filename):
        self._recorder["thread"] = threading.current_thread().name
        self._recorder["filename"] = filename


class FakeStreamQuery:
    def __init__(self, stream):
        self._stream = stream

    def filter(self, **kwargs):
        self.filter_kwargs = kwargs
        return self

    def first(self):
        return self._stream


class FakeYouTube:
    def __init__(self, recorder, title='Some / Video: "Title"'):
        self._recorder = recorder
        self.title = title

    def __call__(self, link):
        self._recorder["link"] = link
        return self

    @property
    def streams(self):
        return FakeStreamQuery(FakeStream(self._recorder))


@pytest.mark.asyncio
async def test_download_runs_off_the_event_loop_thread(monkeypatch):
    recorder = {}
    monkeypatch.setattr(extractor, "YouTube", FakeYouTube(recorder))

    loop_thread = threading.current_thread().name
    result = await YouTubeAudioExtractorPlugin().execute(
        "extract_youtube_audio", None, youtube_link="https://youtu.be/x"
    )

    assert recorder["link"] == "https://youtu.be/x"
    assert recorder["thread"] != loop_thread, "download blocked the event loop thread"
    assert result["direct_result"]["kind"] == "file"
    assert result["direct_result"]["value"] == recorder["filename"]
    assert result["direct_result"]["value"] == 'Some _ Video_ _Title_.mp3'


@pytest.mark.asyncio
async def test_event_loop_keeps_running_during_the_download(monkeypatch):
    """A blocking download must not stop other coroutines from making progress."""
    started = threading.Event()
    release = threading.Event()

    class BlockingYouTube(FakeYouTube):
        @property
        def streams(self):
            outer = self

            class Stream(FakeStream):
                def download(self, filename):
                    started.set()
                    release.wait(5)
                    super().download(filename)

            return FakeStreamQuery(Stream(outer._recorder))

    monkeypatch.setattr(extractor, "YouTube", BlockingYouTube({}))

    task = asyncio.ensure_future(
        YouTubeAudioExtractorPlugin().execute(
            "extract_youtube_audio", None, youtube_link="https://youtu.be/x"
        )
    )
    for _ in range(100):
        await asyncio.sleep(0.01)
        if started.is_set():
            break
    assert started.is_set(), "download never started"

    ticks = 0
    for _ in range(5):
        await asyncio.sleep(0)
        ticks += 1
    assert ticks == 5, "event loop was blocked by the download"

    release.set()
    result = await task
    assert result["direct_result"]["kind"] == "file"


@pytest.mark.asyncio
async def test_download_failure_is_reported_not_raised(monkeypatch):
    def boom(link):
        raise RuntimeError("video unavailable")

    monkeypatch.setattr(extractor, "YouTube", boom)

    result = await YouTubeAudioExtractorPlugin().execute(
        "extract_youtube_audio", None, youtube_link="https://youtu.be/x"
    )
    assert result == {"result": "Failed to extract audio"}
