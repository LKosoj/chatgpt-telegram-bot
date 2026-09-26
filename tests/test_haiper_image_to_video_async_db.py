from types import SimpleNamespace

import pytest

import bot.plugins.haiper_image_to_video as haiper_image_to_video
from bot import net_safety
from bot.plugins.haiper_image_to_video import HaiperImageToVideoPlugin, TaskStatus, WAITING_PROMPT


class FakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.async_calls = []
        self.sync_calls = 0

    def get_user_images(self, *args, **kwargs):
        self.sync_calls += 1
        raise AssertionError("sync get_user_images must not be called")

    async def get_user_images_async(self, user_id, chat_id, limit):
        self.async_calls.append((user_id, chat_id, limit))
        return self.rows


class LegacyFakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.sync_calls = []

    def get_user_images(self, user_id, chat_id, limit):
        self.sync_calls.append((user_id, chat_id, limit))
        return self.rows


class FakeMessage:
    def __init__(self, user_id=123, chat_id=456, text="/animate move"):
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.caption = None
        self.photo = []
        self.reply_to_message = None
        self.replies = []
        self.deleted = False

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        return SimpleNamespace(message_id=len(self.replies), text=text, kwargs=kwargs)

    async def delete(self):
        self.deleted = True


class FakeQuery:
    def __init__(self, user_id=123, chat_id=456, data="animate_hash-b"):
        self.from_user = SimpleNamespace(id=user_id)
        self.message = FakeMessage(user_id=user_id, chat_id=chat_id)
        self.data = data
        self.answers = []

    async def answer(self, text=None, **kwargs):
        self.answers.append((text, kwargs))


def make_plugin(rows):
    plugin = HaiperImageToVideoPlugin()
    db = FakeDb(rows)
    plugin.openai = SimpleNamespace(db=db, config={}, bot=SimpleNamespace())
    return plugin, db


def image_row(file_id="file-a", file_id_hash="hash-a"):
    return {
        "file_id": file_id,
        "file_id_hash": file_id_hash,
        "created_at": "2026-07-02T10:00:00Z",
    }


@pytest.mark.asyncio
async def test_handle_animate_command_uses_async_image_lookup(monkeypatch):
    plugin, db = make_plugin([image_row()])
    message = FakeMessage()
    processed = []

    async def fake_process_animate_command(message_arg, file_id, prompt=None):
        processed.append((message_arg, file_id, prompt))

    monkeypatch.setattr(plugin, "_process_animate_command", fake_process_animate_command)

    await plugin.handle_animate_command(SimpleNamespace(message=message), context=None)

    assert db.sync_calls == 0
    assert db.async_calls == [(123, 456, 1)]
    assert processed == [(message, "file-a", "move")]
    assert message.replies == []


@pytest.mark.asyncio
async def test_handle_animate_button_uses_async_image_lookup(monkeypatch):
    plugin, db = make_plugin([image_row(file_id="file-b", file_id_hash="hash-b")])
    query = FakeQuery()
    processed = []

    async def fake_process_animate_command(message_arg, file_id, prompt=None):
        processed.append((message_arg, file_id, prompt))

    monkeypatch.setattr(plugin, "_process_animate_command", fake_process_animate_command)

    await plugin.handle_animate_button(query)

    assert db.sync_calls == 0
    assert db.async_calls == [(123, "456", 5)]
    assert processed == [(query.message, "file-b", None)]
    assert query.message.replies == []


@pytest.mark.asyncio
async def test_handle_photo_message_uses_async_image_lookup():
    plugin, db = make_plugin([image_row()])
    message = FakeMessage()
    message.text = None
    message.caption = None
    message.photo = [SimpleNamespace(file_id="telegram-file")]

    await plugin.handle_photo_message(SimpleNamespace(message=message), context=None)

    assert db.sync_calls == 0
    assert db.async_calls == [(123, "456", 1)]
    assert len(message.replies) == 1


@pytest.mark.asyncio
async def test_apply_settings_uses_async_image_lookup():
    plugin, db = make_plugin([image_row()])
    plugin.user_settings[123] = {"style": "realistic"}
    query = FakeQuery()

    result = await plugin.apply_settings(query)

    assert result == WAITING_PROMPT
    assert db.sync_calls == 0
    assert db.async_calls == [(123, "456", 1)]
    assert plugin.user_settings[123]["file_id"] == "file-a"
    assert query.message.deleted is True
    assert len(query.message.replies) == 1


@pytest.mark.asyncio
async def test_handle_prompt_constructor_uses_async_image_lookup():
    plugin, db = make_plugin([image_row()])
    message = FakeMessage()
    openai = SimpleNamespace(db=db, bot=SimpleNamespace())

    result = await plugin.handle_prompt_constructor(
        "animate_prompt",
        openai,
        update=SimpleNamespace(message=message),
    )

    assert result == WAITING_PROMPT
    assert db.sync_calls == 0
    assert db.async_calls == [(123, "456", 1)]
    assert len(message.replies) == 1


@pytest.mark.asyncio
async def test_get_user_images_falls_back_to_sync_only_without_async_api():
    plugin = HaiperImageToVideoPlugin()
    db = LegacyFakeDb([image_row()])
    plugin.openai = SimpleNamespace(db=db)

    rows = await plugin._get_user_images(123, "456", limit=1)

    assert rows == [image_row()]
    assert db.sync_calls == [(123, "456", 1)]


class FakeStatusMessage:
    def __init__(self):
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append((text, kwargs))


def _make_animate_plugin_with_completed_task(monkeypatch, video_url):
    """Строит плагин + FakeMessage, у которых start_worker() не запускает реальный
    background-воркер (что вызвало бы настоящий сетевой вызов к Haiper API), а
    asyncio.sleep(30) внутри цикла опроса статуса пропатчен так, чтобы на первой же
    "паузе" пометить единственную поставленную в очередь задачу как COMPLETED с
    video_url в result. Один проход цикла сохраняется (а не мгновенный break на
    первой итерации), чтобы elapsed_minutes успевал быть вычислен — как и в реальном
    цикле опроса."""
    plugin, db = make_plugin([image_row()])
    message = FakeMessage()
    status_message = FakeStatusMessage()

    async def fake_reply_text(text, **kwargs):
        message.replies.append((text, kwargs))
        return status_message

    monkeypatch.setattr(message, "reply_text", fake_reply_text)

    async def fake_start_worker():
        return None

    monkeypatch.setattr(plugin, "start_worker", fake_start_worker)

    async def fake_sleep(_seconds):
        for task in plugin.active_tasks.values():
            task.status = TaskStatus.COMPLETED
            task.result = {"url": video_url}

    monkeypatch.setattr(haiper_image_to_video.asyncio, "sleep", fake_sleep)

    return plugin, message, status_message


@pytest.mark.asyncio
async def test_process_animate_command_downloads_video_via_safe_get(monkeypatch):
    plugin, message, status_message = _make_animate_plugin_with_completed_task(
        monkeypatch, "http://example.test/video.mp4"
    )

    calls = []

    async def fake_safe_get(url, *, max_bytes, timeout):
        calls.append({"url": url, "max_bytes": max_bytes, "timeout": timeout})
        return net_safety.SafeResponse(status_code=200, headers={}, content=b"video-bytes")

    monkeypatch.setattr(haiper_image_to_video.net_safety, "safe_get", fake_safe_get)

    sent_videos = []

    async def fake_reply_video(video, caption=None, **kwargs):
        sent_videos.append(video.read())

    monkeypatch.setattr(message, "reply_video", fake_reply_video, raising=False)

    await plugin._process_animate_command(message, "file-a", prompt="move")

    assert calls == [{
        "url": "http://example.test/video.mp4",
        "max_bytes": plugin.max_video_bytes,
        "timeout": 60.0,
    }]
    assert sent_videos == [b"video-bytes"]
    assert plugin.active_tasks == {}


@pytest.mark.asyncio
async def test_process_animate_command_reports_unsafe_video_url(monkeypatch):
    plugin, message, status_message = _make_animate_plugin_with_completed_task(
        monkeypatch, "http://169.254.169.254/video.mp4"
    )

    async def fake_safe_get(url, *, max_bytes, timeout):
        raise net_safety.UnsafeURLError("refused")

    monkeypatch.setattr(haiper_image_to_video.net_safety, "safe_get", fake_safe_get)

    await plugin._process_animate_command(message, "file-a", prompt="move")

    assert len(message.replies) == 2
    assert "refused" in message.replies[-1][0]
    assert plugin.active_tasks == {}
