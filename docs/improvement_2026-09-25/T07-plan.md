# T07 — Блокировка одного экземпляра

Владение файлами (по мастер-плану): `bot/__main__.py` (только блокировка), новый
`bot/instance_lock.py`, `tests/test_instance_lock.py`, `AGENTS.md` (раздел Project Shape —
одна фраза), `README.md`, `README.ru.md` (раздел запуска — абзац). Другие файлы не трогать.

## 0. Критичное ограничение, из-за которого весь дизайн такой

`tests/test_telegram_builder_config.py` (владение другой задачи, не трогать) дважды вызывает
`bot_main.main()` **в одном процессе pytest** —
`test_default_telegram_builder_uses_local_bot_api` и
`test_telegram_builder_reuses_existing_current_loop_without_closing_it`, через
`_run_main_with_fake_dependencies` → `bot_main.main()`. `Database`/`PluginManager` там
заменены фейками, но `bot_main.main` — настоящий.

`fcntl.flock()` привязан не к процессу, а к *открытому файловому дескриптору*: если тот же
процесс второй раз открывает тот же файл (новый `open()`) и зовёт `flock(LOCK_EX|LOCK_NB)` —
ядро расценивает это как второго конкурента и отклоняет запрос, даже если оба запроса из
одного процесса (man 2 flock: «If a process uses open(2)… to obtain more than one file
descriptor for the same file, these file descriptors are treated independently»). Значит
наивная реализация (открыть файл + `flock` внутри `main()` без состояния) ломает второй вызов
`main()` в этом чужом тесте — `exit(1)`, `SystemExit`, тест падает. Чинить чужой тест нельзя
(не наше владение), значит `bot/instance_lock.py` обязан быть **идемпотентным для повторного
вызова из того же процесса на тот же путь**: он не решает гипотетическую задачу, а обходит
конкретный, уже существующий в дереве конфликт.

## 1. `bot/instance_lock.py` — API

```python
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
    """Путь по умолчанию — рядом с БД. Зеркалит фолбэк Database (bot/database.py:217-220)
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
    отклонил бы сам себя (см. раздел 0). Возвращает None, если fcntl недоступен (Windows) —
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
            os.makedirs(directory, exist_ok=True)  # см. риск "read-only каталог" ниже

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
```

`_reset_for_tests()` — приватная, только для нашего тест-файла (та же схема, что
`Database._reset_singleton()`, `bot/database.py:239-248`), не публичный API.

## 2. Точка интеграции в `bot/__main__.py`

Импорт вверху рядом с остальными: `from .instance_lock import acquire_instance_lock,
InstanceLockError, default_lock_path`.

Место вызова — сразу после проверки обязательных `TELEGRAM_BOT_TOKEN`/`OPENAI_API_KEY`
(после текущих строк 199-203, до комментария `# Setup configurations` на строке 205),
то есть раньше и `plugin_manager = PluginManager(...)` (строка 381), и `Database.configure`/
`Database()` (строки 384-390) — «до создания компонентов» по мастер-плану. Ставить строго
перед проверкой обязательных env — не обязательно: ошибка конфигурации (нет токена) — более
базовая, чем занятый лок, пусть репортится первой; на итоговую защиту от гонки это не влияет,
т.к. Database()/PluginManager() создаются намного позже.

```python
    lock_path = os.environ.get('INSTANCE_LOCK_PATH') or default_lock_path(
        os.environ.get('DB_PATH')
    )
    try:
        acquire_instance_lock(lock_path)
    except InstanceLockError as exc:
        logging.error(str(exc))
        exit(1)
```

Локальная переменная-результат (`acquire_instance_lock(...)`, без присвоения в код выше)
не обязательна: кадр `main()` живёт весь процесс (блокируется на `telegram_bot.run()`), а
`_held_handles` в `instance_lock.py` и так держит ссылку на handle — двойная защита от GC.
Стиль ошибки — `logging.error(...)` + `exit(1)`, как на строках 202-203 и 304-306 (тот же
файл, тот же паттерн, не придумываем новый).

`INSTANCE_LOCK_PATH`, если задан, используется как путь целиком (не как каталог) — так же,
как `TELEGRAM_BASE_URL` переопределяет URL целиком в этом же файле.

## 3. Решённые вопросы investigation-списка

- **Docker/restart после краша.** `flock` — advisory-лок, привязанный к живому файловому
  дескриптору процесса; ядро освобождает его при любом завершении процесса, включая
  `SIGKILL`/OOM (проверено по семантике `flock(2)`, не по памяти — см. docstring модуля).
  `docker-compose.yml` (`restart: unless-stopped`) создаёт новый процесс в новом PID
  namespace контейнера — старого дескриптора не существует, лок гарантированно свободен.
  Ничего чистить руками не нужно, "заброшенный" `bot.instance.lock` не бывает опасным сам
  по себе (в отличие от, например, PID-файла с ручной проверкой "жив ли PID").
- **Путь при дефолтах.** Docker: `DB_PATH=/app/data/user_data.db` (Dockerfile) →
  `/app/data/bot.instance.lock`, каталог `/app/data` уже `chown`-нут на `bot:bot`
  (Dockerfile:42-54) — пишется без проблем. Локальный dev без `DB_PATH`: `Database`
  дефолтится на `<каталог bot/>/user_data.db` (`bot/database.py:217-220`) → наш
  `default_lock_path(None)` даёт `<каталог bot/>/bot.instance.lock` — тот же каталог,
  обычно доступен на запись при обычном чекауте репозитория.
- **Windows.** `fcntl` не существует. Решение: `try/except ImportError` на уровне модуля,
  `acquire_instance_lock` при `fcntl is None` логирует WARNING и возвращает `None` без
  исключения — старт не падает. Обоснование: README/Dockerfile/AGENTS.md нигде не заявляют
  поддержку Windows, единственная документированная цель — Linux/Docker; жёстко требовать
  fcntl означало бы новый hard-fail на платформе, которую проект и так не поддерживает,
  а мягкая деградация ничего не ломает и не создаёт ложного чувства защиты (WARNING виден).
- **Тесты, вызывающие `main()`.** Единственный такой файл в дереве —
  `tests/test_telegram_builder_config.py` (проверено полнотекстовым поиском
  `bot.__main__`/`from bot import __main__` по `tests/` и `bot/tests/`). Разобран в разделе 0;
  идемпотентный кеш в `instance_lock.py` — единственное изменение, которое делает эту
  чужую тестовую задачу совместимой без правки чужого файла.
- **Read-only каталог.** Если `os.makedirs`/`open` в `acquire_instance_lock` падает по
  правам (не по конфликту лока, а по ФС) — исключение `OSError` **не перехватывается** и
  всплывает из `main()` необработанным: трейсбек + ненулевой код возврата via сам
  интерпретатор. Решение осознанное: это отдельный класс ошибки (misconfiguration, не
  "второй инстанс"), заворачивать его в тот же `InstanceLockError`/тот же текст означало бы
  врать в сообщении ("другой процесс держит лок", хотя каталог просто не пишется). Ловить и
  форматировать отдельно — лишний код без явного запроса в мастер-плане (Simplicity First);
  трейсбек и так информативен для оператора.

## 4. Документация

**AGENTS.md**, раздел «Project Shape», одна строка сразу после «Startup creates
`PluginManager`, `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`
(`bot/__main__.py:366-385`).»:

> `main()` acquires a single-instance file lock (`bot/instance_lock.py`) before any of the
> above are created; a second process against the same lock file logs ERROR and exits
> non-zero (`bot/__main__.py:<фактическая строка после правки>`).

Номер строки проставить по факту после правки (см. `project_agents_md_line_refs_drift` —
не копировать наугад).

**README.md / README.ru.md**, раздел «Quick Start» / «Быстрый старт», один абзац после
абзаца про `OPENAI_BASE_URL` (перед `---`), той же плотности, что соседние абзацы. Должен
покрывать: только один процесс бота может работать на одну БД/токен одновременно; лок-файл
по умолчанию лежит рядом с БД (`<каталог DB_PATH>/bot.instance.lock`), путь переопределяется
`INSTANCE_LOCK_PATH`; при попытке второго запуска — ERROR в лог и ненулевой код выхода;
после краша/`docker restart` лок освобождается автоматически (ОС), ручной чистки не нужно.
Не добавлять `INSTANCE_LOCK_PATH` в `.env.example` — файл не во владении T07.

## 5. Тесты — `tests/test_instance_lock.py`

Фикстура `autouse`, вызывающая `instance_lock._reset_for_tests()` до и после каждого теста
(изоляция между тестами в этом файле). Путь лока — всегда `tmp_path / "bot.instance.lock"`.

1. `test_acquire_creates_file_and_returns_handle` — вызов создаёт файл, `handle` не `None`.
2. `test_repeated_acquire_same_process_returns_cached_handle` — два вызова
   `acquire_instance_lock(path)` подряд из теста → один и тот же объект (`is`), без
   исключения. Это прямая регрессионная защита раздела 0.
3. `test_second_open_same_path_is_denied` — после `acquire_instance_lock(path)`, тест сам,
   в обход нашего API, открывает тот же файл (`open(path)`) и зовёт `fcntl.flock(fd,
   LOCK_EX|LOCK_NB)` напрямую → `BlockingIOError`/`OSError`. Это и есть «через отдельный
   открытый файл» из мастер-плана — эмулирует независимого держателя (второй процесс) без
   реального сабпроцесса.
4. `test_release_then_reacquire_succeeds` — закрыть handle из шага 1 (`handle.close()`),
   затем новый прямой `open()+flock()` (как в шаге 3) — теперь успешен. Подтверждает
   "снимается при завершении/закрытии", основание для докер-рестарта.
5. `test_second_process_subprocess_denied` — в текущем процессе взять лок; затем
   `subprocess.run([sys.executable, "-c", <inline-скрипт>], ...)`, где inline-скрипт делает
   `sys.path.insert(0, <repo_root>); from bot.instance_lock import acquire_instance_lock,
   InstanceLockError` и пытается взять лок на тот же путь, печатая `"LOCKED"`/`"OK"` и
   выходя с 0/1 по результату; assert на код возврата и вывод. `bot/instance_lock.py` не
   тянет тяжёлые зависимости (только stdlib, `bot/__init__.py` пуст) — сабпроцесс лёгкий,
   не нужен весь `bot`-стек.
6. `test_default_lock_path_next_to_db` — `default_lock_path('/x/data/user_data.db') ==
   '/x/data/bot.instance.lock'`; `default_lock_path(None)` заканчивается на
   `.../bot/bot.instance.lock` (тот же каталог, что и `bot/database.py`/`__file__`).
7. `test_missing_fcntl_degrades_to_warning` — `monkeypatch.setattr(instance_lock, 'fcntl',
   None)`; `acquire_instance_lock(path)` возвращает `None`, не бросает, пишет WARNING
   (`caplog`).
8. `test_main_exits_nonzero_when_lock_held` — интеграционный: минимальные monkeypatch
   `bot_main.PluginManager`/`Database`/`OpenAIHelper`/`ChatGPTTelegramBot` на дешёвые
   заглушки (не импортировать реальные — по образцу `_run_main_with_fake_dependencies` в
   `tests/test_telegram_builder_config.py`, но копия минимальна, не полный чужой харнесс),
   плюс `monkeypatch.setenv` обязательных `TELEGRAM_BOT_TOKEN`/`OPENAI_API_KEY`/
   `OPENAI_MODEL` и `INSTANCE_LOCK_PATH=str(tmp_path/...)`. Сначала занять лок тем же путём
   вручную, затем вызвать `bot_main.main()` → `pytest.raises(SystemExit)` с кодом 1, лог
   ERROR через `caplog`, и заглушка `PluginManager`/`Database` НЕ была вызвана (proof, что
   лок берётся до создания компонентов).

## 6. Критерии готовности

- `~/.venvs/ctb/bin/python -m pytest tests/test_instance_lock.py
  tests/test_telegram_builder_config.py -q --no-header -p no:cacheprovider` — всё зелёное
  (второй файл — чужое владение, но обязан остаться зелёным без правки).
- `~/.venvs/ctb/bin/python -m pytest tests -q --no-header -p no:cacheprovider` — не меньше
  1650 тестов проходят (базовая линия мастер-плана), новых красных нет.
- `~/.venvs/ctb/bin/python -m ruff check bot/instance_lock.py bot/__main__.py
  tests/test_instance_lock.py`.
- `python3 -m mypy bot/instance_lock.py bot/__main__.py --python-executable
  ~/.venvs/ctb/bin/python --ignore-missing-imports` — не хуже текущей базовой линии
  (`bot/__main__.py` уже правился в T02 параллельно — координатору свериться перед мержем,
  T07 не трогает ничего в `__main__.py` кроме своего блока).
- Ручная проверка (не обязательна для CI, но полезна): два `python -m bot` подряд в одном
  каталоге без второго токена — второй завершается с ERROR и кодом 1.

## 7. Риски / открытые вопросы для координатора

1. **`.gitignore` не во владении T07.** Дефолтный лок-файл без `DB_PATH`
   (`bot/bot.instance.lock`) не покрыт ни одним существующим паттерном (`bot/user_data.db*`
   — точечные записи, не маска). После первого локального запуска файл будет висеть как
   untracked в `git status`. T07 не может это поправить (файл вне владения, уже правился
   T01). Нужна отдельная строка `bot/bot.instance.lock` (или `*.instance.lock`) — либо
   координатор добавляет её сам, либо это одна строка для следующей волны/финального
   ревью.
2. **`bot/__main__.py` уже в работе у T02** (тот же файл, тот же git status показывает `M`).
   Раздел «Владение файлами» master-плана разрешает T07 трогать в `__main__.py` только
   блок блокировки — вставка должна быть чисто аддитивной (новый импорт + 6 строк в
   `main()`), без сдвига/правки соседних строк T02, чтобы merge не конфликтовал по смыслу.
3. **`.env.example` не во владении T07** — `INSTANCE_LOCK_PATH` документирован только в
   README, не в `.env.example`. Несогласованность между README и `.env.example` уже
   допускается правилами проекта («если расходятся — свериться с кодом»), но стоит
   отметить как желательное follow-up.
4. Кеш `_held_handles` в `instance_lock.py` — межпроцессно ничего не меняет (он
   process-local), только устраняет самоконфликт `main()` с самим собой в рамках одного
   процесса (тесты). В реальном рантайме `main()` вызывается ровно один раз — кеш там
   фактически не используется, это чисто тестовая/defensive гарантия.
