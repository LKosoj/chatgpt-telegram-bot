import fcntl
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bot import instance_lock
from bot.instance_lock import acquire_instance_lock, default_lock_path

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_instance_lock():
    instance_lock._reset_for_tests()
    yield
    instance_lock._reset_for_tests()


def test_acquire_creates_file_and_returns_handle(tmp_path):
    path = tmp_path / "bot.instance.lock"
    handle = acquire_instance_lock(str(path))
    assert handle is not None
    assert path.exists()


def test_repeated_acquire_same_process_returns_cached_handle(tmp_path):
    path = tmp_path / "bot.instance.lock"
    first = acquire_instance_lock(str(path))
    second = acquire_instance_lock(str(path))
    assert first is second


def test_second_open_same_path_is_denied(tmp_path):
    path = tmp_path / "bot.instance.lock"
    acquire_instance_lock(str(path))

    fd = open(path)
    try:
        with pytest.raises(OSError):
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fd.close()


def test_release_then_reacquire_succeeds(tmp_path):
    path = tmp_path / "bot.instance.lock"
    handle = acquire_instance_lock(str(path))
    handle.close()

    fd = open(path)
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fd.close()


def test_second_process_subprocess_denied(tmp_path):
    path = tmp_path / "bot.instance.lock"
    acquire_instance_lock(str(path))

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from bot.instance_lock import acquire_instance_lock, InstanceLockError\n"
        "try:\n"
        f"    acquire_instance_lock({str(path)!r})\n"
        "except InstanceLockError:\n"
        "    print('LOCKED')\n"
        "    sys.exit(1)\n"
        "else:\n"
        "    print('OK')\n"
        "    sys.exit(0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "LOCKED" in result.stdout


def test_default_lock_path_next_to_db():
    assert default_lock_path("/x/data/user_data.db") == "/x/data/bot.instance.lock"
    assert default_lock_path(None).endswith(os.path.join("bot", "bot.instance.lock"))


def test_missing_fcntl_degrades_to_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(instance_lock, "fcntl", None)
    path = tmp_path / "bot.instance.lock"
    with caplog.at_level(logging.WARNING):
        handle = acquire_instance_lock(str(path))
    assert handle is None
    assert "fcntl unavailable" in caplog.text


def test_main_exits_nonzero_when_lock_held(tmp_path, monkeypatch, caplog):
    lock_path = tmp_path / "bot.instance.lock"
    # Simulate another process already holding the lock via a separate fd (like
    # test_second_open_same_path_is_denied) -- calling acquire_instance_lock() twice
    # from this same test process would hit the idempotent same-process cache instead.
    held_fd = open(lock_path, "a")
    fcntl.flock(held_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    from bot import __main__ as bot_main

    calls = []

    class _StubPluginManager:
        def __init__(self, *args, **kwargs):
            calls.append("PluginManager")

    class _StubDatabase:
        def __init__(self, *args, **kwargs):
            calls.append("Database")

        @staticmethod
        def configure(**kwargs):
            calls.append("Database.configure")

    monkeypatch.setattr(bot_main, "load_dotenv", lambda: None)
    monkeypatch.setattr(bot_main, "PluginManager", _StubPluginManager)
    monkeypatch.setattr(bot_main, "Database", _StubDatabase)
    monkeypatch.setattr(
        bot_main, "OpenAIHelper", lambda *a, **kw: calls.append("OpenAIHelper")
    )
    monkeypatch.setattr(
        bot_main, "ChatGPTTelegramBot", lambda *a, **kw: calls.append("ChatGPTTelegramBot")
    )

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "llmgateway/high")
    monkeypatch.setenv("INSTANCE_LOCK_PATH", str(lock_path))

    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as exc_info:
                bot_main.main()
    finally:
        held_fd.close()

    assert exc_info.value.code == 1
    assert "Another bot instance" in caplog.text
    assert calls == []
