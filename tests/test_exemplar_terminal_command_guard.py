"""Exemplar-тест: структурное свойство guard'а опасных команд в TerminalPlugin.

Проверяется не текст вывода модели, а СТРУКТУРА решения: опасная команда
(в т.ч. обфусцированная кавычками) не должна доходить до spawn'а подпроцесса
и должна возвращать человекочитаемую причину отказа; безобидная команда,
наоборот, не должна блокироваться.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.plugins.terminal import TerminalPlugin


@pytest.mark.asyncio
async def test_recursive_rm_is_blocked_without_spawning_a_process(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr("bot.plugins.terminal.asyncio.create_subprocess_shell", spawn)
    plugin = TerminalPlugin()

    result = await plugin.execute("terminal", helper=None, command="rm -rf /tmp/x")

    spawn.assert_not_awaited()
    assert result["success"] is False
    assert result["error"]  # причина отказа не пустая — читаема человеком


@pytest.mark.asyncio
async def test_obfuscated_recursive_rm_is_still_blocked_without_spawning_a_process(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr("bot.plugins.terminal.asyncio.create_subprocess_shell", spawn)
    plugin = TerminalPlugin()

    # Флаг -rf разбит кавычками вокруг "r", чтобы не совпадать с guard'ом
    # буквально по подстроке "-rf" без нормализации команды.
    result = await plugin.execute("terminal", helper=None, command='rm -"r"f /tmp/x')

    spawn.assert_not_awaited()
    assert result["success"] is False
    assert result["error"]


@pytest.mark.asyncio
async def test_benign_command_is_not_blocked_and_does_spawn_a_process(monkeypatch):
    stdout = asyncio.StreamReader()
    stdout.feed_eof()
    stderr = asyncio.StreamReader()
    stderr.feed_eof()
    fake_process = SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        pid=4242,
        returncode=0,
        wait=AsyncMock(return_value=0),
    )
    spawn = AsyncMock(return_value=fake_process)
    monkeypatch.setattr("bot.plugins.terminal.asyncio.create_subprocess_shell", spawn)
    plugin = TerminalPlugin()

    result = await plugin.execute("terminal", helper=None, command="ls -la /tmp")

    spawn.assert_awaited_once()
    assert "guard" not in result
    assert result["success"] is True
