import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import bot.utils as utils


class FakeMessage:
    def __init__(self):
        self.message_id = 100
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_document = AsyncMock(return_value=SimpleNamespace(message_id=201))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.callback_query = None
        self.effective_chat = SimpleNamespace(type="private")


@pytest.mark.asyncio
async def test_send_long_response_as_file_uses_to_thread_for_render_and_read(tmp_path, monkeypatch):
    output_path = tmp_path / "response.html"
    output_path.write_bytes(b"<html>response</html>")
    to_thread_calls = []
    captured_roots = {}

    class FakeVisualizer:
        def __init__(self, *, output_dir=None, data_dir=None, plots_dir=None):
            captured_roots["output_dir"] = output_dir
            captured_roots["data_dir"] = data_dir
            captured_roots["plots_dir"] = plots_dir

        def advanced_visualization(self, response, session_id):
            return str(output_path)

    async def fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(utils, "HTMLVisualizer", FakeVisualizer)
    monkeypatch.setattr(utils.asyncio, "to_thread", fake_to_thread)

    message = FakeMessage()
    update = FakeUpdate(message)

    sent_message = await utils.send_long_response_as_file(
        {
            "bot_language": "en",
            "enable_quoting": False,
            "output_dir": str(tmp_path / "configured-output"),
            "data_dir": str(tmp_path / "configured-data"),
            "plots_dir": str(tmp_path / "configured-plots"),
        },
        update,
        "long response",
        "session name",
    )

    assert sent_message == message.reply_document.return_value
    assert captured_roots == {
        "output_dir": str(tmp_path / "configured-output"),
        "data_dir": str(tmp_path / "configured-data"),
        "plots_dir": str(tmp_path / "configured-plots"),
    }
    assert to_thread_calls == ["advanced_visualization", "_read_file_bytes"]
    message.reply_document.assert_awaited_once()
    document = message.reply_document.await_args.kwargs["document"]
    assert document.getvalue() == b"<html>response</html>"
    assert not output_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_path", [None, "missing"])
async def test_send_long_response_as_file_skips_delivery_without_readable_output(
    tmp_path,
    monkeypatch,
    output_path,
):
    resolved_path = None if output_path is None else str(tmp_path / output_path)

    class FakeVisualizer:
        def __init__(self, **_kwargs):
            pass

        def advanced_visualization(self, response, session_id):
            return resolved_path

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(utils, "HTMLVisualizer", FakeVisualizer)
    monkeypatch.setattr(utils.asyncio, "to_thread", fake_to_thread)

    message = FakeMessage()
    update = FakeUpdate(message)

    sent_message = await utils.send_long_response_as_file(
        {"bot_language": "en", "enable_quoting": False},
        update,
        "long response",
    )

    assert sent_message is None
    message.reply_document.assert_not_called()


@pytest.mark.asyncio
async def test_send_long_response_as_file_read_failure_logs_path_shape(tmp_path, monkeypatch, caplog):
    secret_path = tmp_path / "secret-response-path.html"

    class FakeVisualizer:
        def __init__(self, **_kwargs):
            pass

        def advanced_visualization(self, response, session_id):
            return str(secret_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(utils, "HTMLVisualizer", FakeVisualizer)
    monkeypatch.setattr(utils.asyncio, "to_thread", fake_to_thread)

    message = FakeMessage()
    update = FakeUpdate(message)

    caplog.set_level(logging.INFO)
    sent_message = await utils.send_long_response_as_file(
        {"bot_language": "en", "enable_quoting": False},
        update,
        "long response",
    )

    assert sent_message is None
    message.reply_document.assert_not_called()
    assert "Failed to read generated HTML file" in caplog.text
    assert "output_shape" in caplog.text
    assert "FileNotFoundError(message_chars=" in caplog.text
    assert "secret-response-path" not in caplog.text


@pytest.mark.asyncio
async def test_send_long_response_as_file_remove_failure_logs_path_shape(tmp_path, monkeypatch, caplog):
    output_path = tmp_path / "secret-remove-path.html"
    output_path.write_bytes(b"<html>response</html>")

    class FakeVisualizer:
        def __init__(self, **_kwargs):
            pass

        def advanced_visualization(self, response, session_id):
            return str(output_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fail_remove(_path):
        raise OSError("secret unlink failure")

    monkeypatch.setattr(utils, "HTMLVisualizer", FakeVisualizer)
    monkeypatch.setattr(utils.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(utils.os, "remove", fail_remove)

    message = FakeMessage()
    update = FakeUpdate(message)

    caplog.set_level(logging.INFO)
    sent_message = await utils.send_long_response_as_file(
        {"bot_language": "en", "enable_quoting": False},
        update,
        "long response",
    )

    assert sent_message == message.reply_document.return_value
    message.reply_document.assert_awaited_once()
    assert "Failed to delete generated HTML file" in caplog.text
    assert "output_shape" in caplog.text
    assert "OSError(message_chars=" in caplog.text
    assert "secret-remove-path" not in caplog.text
    assert "secret unlink failure" not in caplog.text
