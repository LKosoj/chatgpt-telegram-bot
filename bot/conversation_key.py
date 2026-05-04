from __future__ import annotations

from telegram import Update, constants


def get_conversation_key(update: Update) -> int:
    if update.effective_chat and update.effective_chat.type in (
        constants.ChatType.GROUP,
        constants.ChatType.SUPERGROUP,
    ):
        return update.effective_chat.id
    return update.effective_user.id


__all__ = ["get_conversation_key"]
