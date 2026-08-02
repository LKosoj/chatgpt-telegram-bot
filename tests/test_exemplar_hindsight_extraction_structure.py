"""Exemplar-тест: структурные свойства извлечения памяти в HindsightMemoryPlugin.

Три свойства, которые должны держаться независимо от текста, который вернёт
модель-экстрактор:

1. правило провенанса — часть промпта экстрактора (регрессия здесь означает,
   что модель начнёт сохранять предпочтения, которые высказал не сам
   пользователь, а придумал ассистент);
2. autonomous=True добавляет AUTONOMOUS_EXTRACTION_ADDENDUM к системному
   сообщению, не изменяя сам базовый промпт; при дефолте системный контент
   точно равен базовому промпту (без примесей);
3. связка "буфер всплесков -> finalize job -> вызов экстрактора" сворачивает
   N последовательных ходов в ОДИН вызов модели с ОДНИМ транскриптом, где
   присутствуют все N ходов — а не в N отдельных вызовов.

Свойство 2 частично пересекается с tests/test_hindsight_finalize_extraction.py
(narrow юнит-тест на `_extract_hindsight_memory_items` в изоляции); здесь
проверка через точное равенство контента, а не startswith/in — сформулировано
как проверка формы одним pytest.mark.parametrize, но это соседний слой:
свойство 3 идёт по-настоящему сквозь систему (реальная SQLite БД + реальный
буфер + реальный finalize-воркер), что не покрыто существующими тестами.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import (
    AUTONOMOUS_EXTRACTION_ADDENDUM,
    HINDSIGHT_EXTRACTOR_PROMPT,
    HindsightMemoryPlugin,
)
from bot.plugins.hooks import AssistantResponsePayload, UserMessagePayload


def test_provenance_rule_is_part_of_extractor_prompt():
    assert "PROVENANCE" in HINDSIGHT_EXTRACTOR_PROMPT


def _build_plugin_for_direct_extraction(**config_overrides) -> HindsightMemoryPlugin:
    plugin = HindsightMemoryPlugin()
    fake_openai = SimpleNamespace(
        config={'light_model': 'fake-light'},
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"items":[]}'))])
        )))),
    )

    async def _chat_completion(**kwargs):
        return await fake_openai.client.chat.completions.create(**kwargs)

    fake_openai.chat_completion = _chat_completion
    plugin.initialize(
        openai=fake_openai,
        plugin_config={'hindsight_base_url': 'http://x', 'hindsight_api_token': 't', **config_overrides},
    )
    return plugin


@pytest.mark.parametrize(
    "autonomous, expected_system_content",
    [
        (False, HINDSIGHT_EXTRACTOR_PROMPT),
        (True, f"{HINDSIGHT_EXTRACTOR_PROMPT}\n\n{AUTONOMOUS_EXTRACTION_ADDENDUM}"),
    ],
)
@pytest.mark.asyncio
async def test_autonomous_flag_selects_addendum_without_mutating_base_prompt(autonomous, expected_system_content):
    plugin = _build_plugin_for_direct_extraction()

    await plugin._extract_hindsight_memory_items("user: hi", autonomous=autonomous)

    call = plugin.openai.client.chat.completions.create.call_args
    system_content = call.kwargs["messages"][0]["content"]
    assert system_content == expected_system_content


class _RealDbBurstSetup:
    """Поднимает настоящую SQLite БД + DbHandle, как это делает
    test_hindsight_burst_buffer.py::test_burst_flush_writes_kind_burst_row_and_finalize_tick_claims_it,
    но с настраиваемым N и с проходом через реальный `_extract_hindsight_memory_items`
    (мокается только внешняя граница — chat_completion), а не через мок
    `_extract_session_memory_items`."""

    def __init__(self, tmp_path, monkeypatch, n_turns: int):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.n_turns = n_turns
        self.calls: list[dict] = []

    async def __aenter__(self):
        from bot.database import Database
        from bot.plugins.db_handle import DbHandle

        db_path = self.tmp_path / "test.db"
        self.monkeypatch.setenv("DB_PATH", str(db_path))
        Database._reset_singleton()
        self.database = Database()
        plugin_for_ddl = HindsightMemoryPlugin()
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            for stmt in plugin_for_ddl.register_schema():
                cursor.execute(stmt)

        async def _chat_completion(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"items":[]}'))],
            )

        fake_openai = SimpleNamespace(config={'light_model': 'fake-light'}, chat_completion=_chat_completion)

        self.plugin = HindsightMemoryPlugin()
        self.plugin.initialize(
            openai=fake_openai,
            plugin_config={
                'hindsight_base_url': 'http://x',
                'hindsight_api_token': 't',
                'hindsight_auto_save': True,
                'hindsight_burst_max_turns': self.n_turns,
            },
        )
        self.plugin.client = SimpleNamespace(enabled=True)
        self.plugin.db_handle = DbHandle(self.database)
        self.session_id = self.database.create_session(7)
        return self

    async def __aexit__(self, *exc_info):
        from bot.database import Database
        Database._reset_singleton()

    async def run_turns(self):
        for i in range(self.n_turns):
            await self.plugin.on_user_message(UserMessagePayload(
                chat_id=7, user_id=7, request_id=f"r{i}", text=f"question-{i}",
                has_image=False, has_voice=False, is_command=False, ts=0.0,
            ))
            await self.plugin.on_assistant_response(AssistantResponsePayload(
                chat_id=7, user_id=7, request_id=f"r{i}", text=f"answer-{i}",
                tokens=1, model="m", ts=0.0,
            ))


@pytest.mark.asyncio
async def test_n_consecutive_turns_collapse_into_a_single_extraction_call(tmp_path, monkeypatch):
    n_turns = 4
    async with _RealDbBurstSetup(tmp_path, monkeypatch, n_turns) as setup:
        await setup.run_turns()
        await setup.plugin._finalize_tick(application=None)

        # Свойство: N ходов подряд породили ОДИН вызов экстрактора, а не N.
        assert len(setup.calls) == 1


@pytest.mark.asyncio
async def test_single_extraction_call_transcript_contains_every_turn(tmp_path, monkeypatch):
    n_turns = 4
    async with _RealDbBurstSetup(tmp_path, monkeypatch, n_turns) as setup:
        await setup.run_turns()
        await setup.plugin._finalize_tick(application=None)

        transcript = setup.calls[0]["messages"][1]["content"]
        for i in range(n_turns):
            assert f"question-{i}" in transcript
            assert f"answer-{i}" in transcript
