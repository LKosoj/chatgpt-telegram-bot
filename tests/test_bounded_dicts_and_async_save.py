"""
Тесты WP-C: bounded dicts и async-save.

Покрывает:
- _BoundedLRU: соблюдение cap, вытеснение старейшего, обновление существующего ключа.
- _conversation_locks: WeakValueDictionary — авто-удаление после освобождения ссылки.
- message_buffer idle-prune: удаление пустых/неактивных записей, сохранение активных.
"""
import asyncio
import gc
import weakref

import pytest

from bot.telegram_bot import ChatGPTTelegramBot, _BoundedLRU


# ---------------------------------------------------------------------------
# _BoundedLRU
# ---------------------------------------------------------------------------

class TestBoundedLRU:
    def test_cap_enforced(self):
        lru = _BoundedLRU(3)
        for i in range(5):
            lru[i] = i
        assert len(lru) == 3

    def test_oldest_evicted(self):
        lru = _BoundedLRU(3)
        lru[0] = "a"
        lru[1] = "b"
        lru[2] = "c"
        lru[3] = "d"  # вытесняет 0
        assert 0 not in lru
        assert 3 in lru

    def test_update_existing_key_moves_to_end(self):
        lru = _BoundedLRU(3)
        lru[0] = "a"
        lru[1] = "b"
        lru[2] = "c"
        # Обновляем старейший ключ 0 — он должен переместиться в конец
        lru[0] = "updated"
        lru[3] = "d"  # должен вытеснить 1, а не 0
        assert 0 in lru
        assert 1 not in lru

    def test_read_promotes_recency(self):
        lru = _BoundedLRU(2)
        lru["a"] = 1
        lru["b"] = 2
        _ = lru.get("a")  # чтение помечает "a" как недавно использованный (LRU)
        lru["c"] = 3      # вытесняет давно не использованный — "b", а не "a"
        assert "a" in lru
        assert "b" not in lru
        assert "c" in lru

    def test_setitem_on_empty(self):
        lru = _BoundedLRU(1)
        lru["x"] = 42
        assert lru["x"] == 42
        assert len(lru) == 1

    def test_maxsize_one(self):
        lru = _BoundedLRU(1)
        lru["a"] = 1
        lru["b"] = 2
        assert len(lru) == 1
        assert "b" in lru
        assert "a" not in lru


# ---------------------------------------------------------------------------
# _conversation_locks — WeakValueDictionary
# ---------------------------------------------------------------------------

def _make_bare_bot() -> ChatGPTTelegramBot:
    """Создаёт объект ChatGPTTelegramBot без вызова __init__."""
    bot = object.__new__(ChatGPTTelegramBot)
    bot._conversation_locks = weakref.WeakValueDictionary()
    bot._conversation_locks_guard = asyncio.Lock()
    return bot


class TestConversationLocksWeakRef:
    def test_is_weak_value_dict(self):
        bot = _make_bare_bot()
        assert isinstance(bot._conversation_locks, weakref.WeakValueDictionary)

    @pytest.mark.asyncio
    async def test_lock_present_while_referenced(self):
        bot = _make_bare_bot()
        lock = asyncio.Lock()
        bot._conversation_locks["key1"] = lock
        assert "key1" in bot._conversation_locks

    @pytest.mark.asyncio
    async def test_lock_disappears_after_gc(self):
        bot = _make_bare_bot()

        async def _store_and_release():
            lock = asyncio.Lock()
            bot._conversation_locks["key_gc"] = lock
            assert "key_gc" in bot._conversation_locks
            # lock выходит из области видимости здесь

        await _store_and_release()
        gc.collect()
        assert "key_gc" not in bot._conversation_locks

    @pytest.mark.asyncio
    async def test_lock_survives_while_in_scope(self):
        bot = _make_bare_bot()
        lock = asyncio.Lock()
        bot._conversation_locks["persistent"] = lock
        gc.collect()
        # strong ref 'lock' ещё жива — ключ должен остаться
        assert "persistent" in bot._conversation_locks


# ---------------------------------------------------------------------------
# message_buffer idle-prune
# ---------------------------------------------------------------------------

def _make_bot_with_buffer() -> ChatGPTTelegramBot:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.message_buffer = {}
    return bot


class TestIdleMessageBufferPrune:
    def _done_task(self) -> asyncio.Task:
        """Вспомогательный метод: создаёт завершённую Task через Future."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        future.set_result(None)
        # asyncio.Task нельзя создать напрямую как done; используем Future как замену —
        # важно только .done() == True
        return future  # type: ignore[return-value]

    def test_prune_removes_idle_entry(self):
        bot = _make_bot_with_buffer()
        # Запись полностью idle: нет сообщений, не обрабатывается, таймер None
        bot.message_buffer[1001] = {'messages': [], 'processing': False, 'timer': None}
        bot._prune_idle_message_buffers()
        assert 1001 not in bot.message_buffer

    def test_prune_removes_idle_with_done_timer(self):
        bot = _make_bot_with_buffer()
        loop = asyncio.new_event_loop()
        try:
            fut = loop.create_future()
            fut.set_result(None)
            bot.message_buffer[1002] = {'messages': [], 'processing': False, 'timer': fut}
            bot._prune_idle_message_buffers()
            assert 1002 not in bot.message_buffer
        finally:
            loop.close()

    def test_prune_keeps_processing_entry(self):
        bot = _make_bot_with_buffer()
        bot.message_buffer[2001] = {'messages': [], 'processing': True, 'timer': None}
        bot._prune_idle_message_buffers()
        assert 2001 in bot.message_buffer

    def test_prune_keeps_entry_with_pending_messages(self):
        bot = _make_bot_with_buffer()
        bot.message_buffer[2002] = {
            'messages': [{'text': 'hello'}],
            'processing': False,
            'timer': None,
        }
        bot._prune_idle_message_buffers()
        assert 2002 in bot.message_buffer

    def test_prune_only_removes_idle_mixed(self):
        bot = _make_bot_with_buffer()
        bot.message_buffer[3001] = {'messages': [], 'processing': False, 'timer': None}
        bot.message_buffer[3002] = {'messages': ['x'], 'processing': False, 'timer': None}
        bot.message_buffer[3003] = {'messages': [], 'processing': True, 'timer': None}
        bot._prune_idle_message_buffers()
        assert 3001 not in bot.message_buffer
        assert 3002 in bot.message_buffer
        assert 3003 in bot.message_buffer

    def test_ensure_message_buffer_prunes_before_creating(self):
        bot = _make_bot_with_buffer()
        # Предзаполняем несколькими idle-записями
        for chat_id in range(10):
            bot.message_buffer[chat_id] = {'messages': [], 'processing': False, 'timer': None}
        # _ensure_message_buffer для нового ключа должна сначала прочистить idle
        result = bot._ensure_message_buffer(999)
        assert result is bot.message_buffer[999]
        # Все idle-записи очищены (999 — новая, живая; остальные были idle)
        assert all(k == 999 for k in bot.message_buffer)
