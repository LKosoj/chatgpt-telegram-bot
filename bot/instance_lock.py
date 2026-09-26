"""Однопроцессная блокировка: OS-уровневый advisory-лок (POSIX flock) не даёт второму
процессу бота стартовать против той же БД/токена, пока первый уже работает.

Не распределённый лок — только локальный для хоста, снимается ядром автоматически при
завершении/падении держащего процесса (в т.ч. kill -9/OOM) — после краша повторный
`docker restart`/systemd-рестарт не блокируется "зависшим" локом, чистить вручную не нужно.

fcntl доступен только на POSIX; на Windows модуля нет. Проект документирован и разворачивается
как Linux/Docker-сервис, поэтому при отсутствии fcntl блокировка мягко отключается
(WARNING в лог, старт продолжается без гарантии единственности) вместо падения на импорте.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import IO, Optional

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILENAME = "bot.instance.lock"

_cache_lock = threading.Lock()
_held_handles: dict[str, IO[str]] = {}  # resolved path -> открытый handle этого процесса


class InstanceLockError(RuntimeError):
    """Другой процесс уже держит блокировку этого пути."""


def default_lock_path(db_path: Optional[str]) -> str:
    """Путь по умолчанию — рядом с БД. Зеркалит фолбэк DB_PATH в Database.__new__ (bot/database.py)
    без импорта Database (лок берётся ДО создания Database()); если DB_PATH не задан,
    Database.__new__ использует `<каталог bot/>/user_data.db` — тот же каталог пакета,
    что и у __file__ этого модуля."""
    resolved_db_path = db_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "user_data.db"
    )
    directory = os.path.dirname(os.path.abspath(resolved_db_path)) or "."
    return os.path.join(directory, DEFAULT_LOCK_FILENAME)


def acquire_instance_lock(path: str) -> Optional[IO[str]]:
    """Берёт эксклюзивный неблокирующий лок на `path`, возвращает открытый handle.

    Handle нужно держать живым до конца процесса — закрытие снимает OS-лок. Повторный
    вызов с тем же (после os.path.abspath) путём из ЭТОГО ЖЕ процесса возвращает уже
    закешированный handle, не открывая новый fd и не перевызывая flock — иначе процесс
    отклонил бы сам себя: flock на новом fd того же файла конфликтует с уже взятым. Возвращает None, если fcntl недоступен (Windows) —
    залогировав WARNING один раз на путь.

    Raises:
        InstanceLockError: другой процесс уже держит лок на этом пути.
    """
    if fcntl is None:
        logger.warning(
            "Instance lock skipped: fcntl unavailable on this platform (os.name=%s); "
            "startup continues without single-instance protection.", os.name,
        )
        return None

    resolved = os.path.abspath(path)
    with _cache_lock:
        cached = _held_handles.get(resolved)
        if cached is not None:
            return cached

        directory = os.path.dirname(resolved)
        if directory:
            os.makedirs(directory, exist_ok=True)  # на read-only каталоге упадёт OSError — старт прервётся

        handle = open(resolved, "a")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise InstanceLockError(
                f"Another bot instance already holds the lock at '{resolved}'. "
                "Only one process may run against the same TELEGRAM_BOT_TOKEN/database "
                "at a time -- stop the other instance, or set INSTANCE_LOCK_PATH to a "
                "different path if multiple instances are intentional."
            ) from exc

        _held_handles[resolved] = handle
        return handle


def _reset_for_tests() -> None:
    """Только для tests/test_instance_lock.py: закрывает и чистит кеш между тестами
    (снимает flock через close()), по образцу Database._reset_singleton()."""
    with _cache_lock:
        for handle in _held_handles.values():
            try:
                handle.close()
            except OSError:
                pass
        _held_handles.clear()
