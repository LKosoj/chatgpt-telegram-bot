import logging
from types import SimpleNamespace

import pytest
from telegram import MessageEntity

from bot.agent_delivery import send_agent_response, send_text_chunks


class FakeBot:
    def __init__(self, *, rich_exception=None):
        self.calls = []
        self.rich_exception = rich_exception

    async def send_message(self, **kwargs):
        self.calls.append(("message", kwargs))
        return SimpleNamespace(message_id=len(self.calls))

    async def _post(self, endpoint, data=None, **kwargs):
        self.calls.append(("post", {"endpoint": endpoint, "data": data, "kwargs": kwargs}))
        if self.rich_exception is not None:
            raise self.rich_exception
        return SimpleNamespace(message_id=len(self.calls))

    async def send_document(self, **kwargs):
        document = kwargs.get("document")
        kwargs = dict(kwargs)
        kwargs["document_name"] = getattr(document, "name", document)
        self.calls.append(("document", kwargs))
        return SimpleNamespace(message_id=len(self.calls))

    async def send_photo(self, **kwargs):
        self.calls.append(("photo", kwargs))
        return SimpleNamespace(message_id=len(self.calls))

    async def send_animation(self, **kwargs):
        self.calls.append(("animation", kwargs))
        return SimpleNamespace(message_id=len(self.calls))


@pytest.mark.asyncio
async def test_send_agent_response_final_sends_artifacts_before_text(tmp_path):
    artifact = tmp_path / "report.txt"
    artifact.write_text("hello", encoding="utf-8")
    bot = FakeBot()

    sent = await send_agent_response(
        bot,
        chat_id=123,
        response={
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "**Done**",
                "artifacts": [
                    {
                        "kind": "file",
                        "format": "path",
                        "file_path": str(artifact),
                        "caption": "Report",
                    }
                ],
            }
        },
        reply_to_message_id=77,
        message_thread_id=88,
    )

    assert [kind for kind, _kwargs in bot.calls] == ["document", "message"]
    document_kwargs = bot.calls[0][1]
    assert document_kwargs["chat_id"] == 123
    assert document_kwargs["reply_to_message_id"] == 77
    assert document_kwargs["message_thread_id"] == 88
    assert document_kwargs["caption"] == "Report"
    assert document_kwargs["document_name"] == str(artifact)
    message_kwargs = bot.calls[1][1]
    assert message_kwargs["text"] == "Done"
    assert message_kwargs["reply_to_message_id"] == 77
    assert message_kwargs["message_thread_id"] == 88
    assert message_kwargs["parse_mode"] is None
    assert all(isinstance(entity, MessageEntity) for entity in message_kwargs["entities"])
    assert [message.message_id for message in sent] == [1, 2]
    assert artifact.exists()


@pytest.mark.asyncio
async def test_send_agent_response_missing_path_reports_unavailable(tmp_path):
    missing = tmp_path / "missing.txt"
    bot = FakeBot()

    sent = await send_agent_response(
        bot,
        chat_id=123,
        response={
            "direct_result": {
                "kind": "file",
                "format": "path",
                "value": str(missing),
            }
        },
        reply_to_message_id=77,
        message_thread_id=88,
    )

    assert sent == []
    assert [kind for kind, _kwargs in bot.calls] == ["message"]
    kwargs = bot.calls[0][1]
    assert kwargs["text"] == "Artifact path is unavailable: missing.txt"
    assert kwargs["reply_to_message_id"] == 77
    assert kwargs["message_thread_id"] == 88
    assert kwargs["parse_mode"] is None


@pytest.mark.asyncio
async def test_send_agent_response_file_path_cleans_up_after_delivery(tmp_path):
    artifact = tmp_path / "report.txt"
    artifact.write_text("hello", encoding="utf-8")
    bot = FakeBot()

    sent = await send_agent_response(
        bot,
        chat_id=123,
        response={
            "direct_result": {
                "kind": "file",
                "format": "path",
                "value": str(artifact),
            }
        },
    )

    assert [kind for kind, _kwargs in bot.calls] == ["document"]
    assert bot.calls[0][1]["document_name"] == str(artifact)
    assert [message.message_id for message in sent] == [1]
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_send_agent_response_file_path_cleans_up_expanded_home_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    artifact = home / "report.txt"
    artifact.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    bot = FakeBot()

    sent = await send_agent_response(
        bot,
        chat_id=123,
        response={
            "direct_result": {
                "kind": "file",
                "format": "path",
                "value": "~/report.txt",
            }
        },
    )

    assert [kind for kind, _kwargs in bot.calls] == ["document"]
    assert bot.calls[0][1]["document_name"] == str(artifact)
    assert [message.message_id for message in sent] == [1]
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_send_text_chunks_uses_entities_for_markdown():
    bot = FakeBot()

    await send_text_chunks(bot, chat_id=123, text="**hello**")

    kwargs = bot.calls[0][1]
    assert kwargs["text"] == "hello"
    assert kwargs["parse_mode"] is None
    assert all(isinstance(entity, MessageEntity) for entity in kwargs["entities"])


@pytest.mark.asyncio
async def test_send_agent_response_rich_enabled_sends_single_rich_message():
    bot = FakeBot()

    sent = await send_agent_response(
        bot,
        chat_id=123,
        response="**hello**",
        reply_to_message_id=77,
        message_thread_id=88,
        config={"telegram_rich_messages": "auto"},
    )

    assert [kind for kind, _kwargs in bot.calls] == ["post"]
    assert sent[0].message_id == 1
    call = bot.calls[0][1]
    assert call["endpoint"] == "sendRichMessage"
    assert call["data"] == {
        "chat_id": 123,
        "message_thread_id": 88,
        "rich_message": {"markdown": "**hello**"},
        "reply_parameters": {"message_id": 77},
    }
    assert call["kwargs"] == {"api_kwargs": None}


@pytest.mark.asyncio
async def test_send_text_chunks_rich_auto_failure_falls_back_with_warning(caplog):
    bot = FakeBot(rich_exception=RuntimeError("boom"))
    caplog.set_level(logging.WARNING, logger="bot.agent_delivery")

    sent = await send_text_chunks(
        bot,
        chat_id=123,
        text="**hello**",
        config={"telegram_rich_messages": "auto"},
    )

    assert [kind for kind, _kwargs in bot.calls] == ["post", "message"]
    assert sent[0].message_id == 2
    kwargs = bot.calls[1][1]
    assert kwargs["text"] == "hello"
    assert kwargs["parse_mode"] is None
    assert all(isinstance(entity, MessageEntity) for entity in kwargs["entities"])
    assert "Rich markdown delivery failed; falling back to legacy text delivery" in caplog.text


@pytest.mark.asyncio
async def test_send_text_chunks_rich_required_failure_raises():
    bot = FakeBot(rich_exception=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await send_text_chunks(
            bot,
            chat_id=123,
            text="**hello**",
            config={"telegram_rich_messages": "required"},
        )

    assert [kind for kind, _kwargs in bot.calls] == ["post"]
