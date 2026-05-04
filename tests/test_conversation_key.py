from types import SimpleNamespace

from telegram import constants

from bot.conversation_key import get_conversation_key


def _update(chat_id, chat_type, user_id, callback=False):
    user = SimpleNamespace(id=user_id)
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_user=user,
        callback_query=SimpleNamespace(from_user=user) if callback else None,
    )


def test_private_conversation_key_uses_effective_user_id():
    update = _update(chat_id=999, chat_type=constants.ChatType.PRIVATE, user_id=42)

    assert get_conversation_key(update) == 42


def test_group_message_conversation_key_uses_chat_id():
    update = _update(chat_id=-100, chat_type=constants.ChatType.GROUP, user_id=42)

    assert get_conversation_key(update) == -100
    assert get_conversation_key(update) != update.effective_user.id


def test_supergroup_callback_conversation_key_uses_chat_id_not_actor_id():
    update = _update(
        chat_id=-100123,
        chat_type=constants.ChatType.SUPERGROUP,
        user_id=42,
        callback=True,
    )

    assert get_conversation_key(update) == -100123
    assert get_conversation_key(update) != update.callback_query.from_user.id
