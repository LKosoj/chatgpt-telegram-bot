"""T12d D7: handle_prompt_constructor and show_main_menu_with_selections in
bot/plugins/haiper_image_to_video.py duplicate (a) the settings-summary text
assembly and (b) the style/effect/preset keyboard construction. No test
previously exercised the menu content of either method; this pins down that
both produce the same settings-summary text and the same style/effect/preset
button rows for the same settings, both with and without selections made.
"""
from types import SimpleNamespace

import pytest

from bot.plugins.haiper_image_to_video import HaiperImageToVideoPlugin


class FakeDb:
    async def get_user_images_async(self, user_id, chat_id, limit):
        return [{"file_id": "file-a", "file_id_hash": "hash-a", "created_at": "2026-07-02T10:00:00Z"}]


class FakeMessage:
    def __init__(self, user_id=123, chat_id=456):
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=chat_id)
        self.replies = []
        self.edits = []

    async def reply_text(self, text, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return SimpleNamespace(message_id=1)

    async def edit_text(self, text, **kwargs):
        self.edits.append({"text": text, **kwargs})


def _make_plugin(settings):
    plugin = HaiperImageToVideoPlugin()
    plugin.openai = SimpleNamespace(db=FakeDb(), config={}, bot=SimpleNamespace())
    plugin.user_settings[123] = settings
    return plugin


def _style_effect_rows(reply_markup):
    # First two rows are the style/effect and preset/reset buttons in both menus.
    return [
        [(button.text, button.callback_data) for button in row]
        for row in reply_markup.inline_keyboard[:2]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"style": "abstract", "effect": "blur", "preset": "art"},
    ],
)
async def test_prompt_constructor_and_main_menu_agree_on_summary_text(settings):
    constructor_plugin = _make_plugin(dict(settings))
    message = FakeMessage()
    await constructor_plugin.handle_prompt_constructor(
        "animate_prompt", constructor_plugin.openai, update=SimpleNamespace(message=message),
    )
    constructor_text = message.replies[0]["text"]
    constructor_rows = _style_effect_rows(message.replies[0]["reply_markup"])

    menu_plugin = _make_plugin(dict(settings))
    menu_message = FakeMessage()
    await menu_plugin.show_main_menu_with_selections(menu_message, 123)
    menu_text = menu_message.edits[0]["text"]
    menu_rows = _style_effect_rows(menu_message.edits[0]["reply_markup"])

    assert constructor_text == menu_text
    assert constructor_rows == menu_rows
