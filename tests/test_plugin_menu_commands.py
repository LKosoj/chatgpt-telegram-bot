from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telegram import MessageEntity

from bot.plugins.agent_cron import AgentCronPlugin
from bot.plugins.agent_tools import AgentToolsPlugin
from bot.plugins.haiper_image_to_video import HaiperImageToVideoPlugin, WAITING_PROMPT
from bot.plugins.reminders import RemindersPlugin


def make_update(text, *, callback=True):
    user = SimpleNamespace(id=42)
    entity = MessageEntity(MessageEntity.BOT_COMMAND, 0, len(text.split()[0]))
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=999) if callback else user,
        chat=SimpleNamespace(id=123),
        chat_id=123,
        message_id=7,
        text=text,
        reply_to_message=None,
        reply_text=AsyncMock(),
        parse_entities=lambda *_: {entity: text.split()[0]} if text.startswith("/") else {},
    )
    return SimpleNamespace(
        message=message,
        effective_message=message,
        effective_user=user,
        callback_query=SimpleNamespace(from_user=user, message=message) if callback else None,
    )


@pytest.mark.parametrize("callback", [True, False])
async def test_reminders_list_uses_command_user(callback):
    plugin = RemindersPlugin()
    plugin.db_handle = SimpleNamespace(fetch_all=AsyncMock(return_value=[]))
    update = make_update("Плагин: reminders" if callback else "/list_reminders", callback=callback)

    await plugin.handle_prompt_constructor(update, None)

    assert plugin.db_handle.fetch_all.await_args.args[1] == ("42",)
    update.message.reply_text.assert_awaited_once()


@pytest.mark.parametrize("callback", [True, False])
async def test_animate_uses_command_user_and_prompt(callback):
    plugin = HaiperImageToVideoPlugin()
    plugin.openai = SimpleNamespace()
    plugin._get_user_images = AsyncMock(return_value=[{"file_id": "image"}])
    plugin._process_animate_command = AsyncMock()
    update = make_update("Плагин: haiper_image_to_video" if callback else "/animate move", callback=callback)

    await plugin.handle_animate_command(update, None)

    plugin._get_user_images.assert_awaited_once_with(42, 123, limit=1)
    prompt = plugin.t("haiper_default_prompt") if callback else "move"
    plugin._process_animate_command.assert_awaited_once_with(
        update.message, "image", prompt, user_id=42,
    )


async def test_animate_prompt_uses_clicking_users_images_and_settings():
    plugin = HaiperImageToVideoPlugin()
    plugin._get_user_images = AsyncMock(return_value=[{"file_id": "image"}])
    plugin.user_settings[42] = {"style": "abstract"}
    plugin.user_settings[999] = {"style": "realistic"}
    plugin._build_settings_summary_text = Mock(return_value="settings")
    update = make_update("Плагин: haiper_image_to_video")

    result = await plugin.handle_prompt_constructor(
        "animate_prompt", SimpleNamespace(bot=SimpleNamespace()), update=update,
    )

    assert result == WAITING_PROMPT
    plugin._get_user_images.assert_awaited_once_with(42, "123", limit=1)
    plugin._build_settings_summary_text.assert_called_once_with(plugin.user_settings[42])


@pytest.mark.parametrize("callback", [True, False])
async def test_background_opens_list_without_starting_job(callback):
    plugin = AgentToolsPlugin()
    plugin._format_background_jobs = Mock(return_value="jobs")
    plugin._create_background_job = Mock()
    update = make_update("Плагин: agent_tools" if callback else "/background list", callback=callback)

    await plugin.handle_background_command(update, SimpleNamespace())

    plugin._format_background_jobs.assert_called_once_with(123, 42)
    plugin._create_background_job.assert_not_called()
    update.message.reply_text.assert_awaited_once_with("jobs")


@pytest.mark.parametrize("callback", [True, False])
async def test_cron_opens_list(callback):
    plugin = AgentCronPlugin()
    plugin._format_jobs = AsyncMock(return_value="jobs")
    update = make_update("Плагин: agent_cron" if callback else "/cron list", callback=callback)

    await plugin.handle_cron_command(update, None)

    plugin._format_jobs.assert_awaited_once_with(123, 42)
    update.message.reply_text.assert_awaited_once_with("jobs")
