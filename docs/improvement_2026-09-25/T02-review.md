# T02 — ревью: попутные баги + доступ в группах 1.4

## Раунд 1

**Проверено:** `git diff HEAD` по всем файлам из владения T02 (`bot/utils.py` —
`is_allowed`/`_charge_user_and_guest`/`_charge_user_and_guest_async`, `bot/__main__.py`,
`bot/plugins/chief.py`, `bot/telegram_stream.py`, `bot/telegram_bot.py` — только строки
`retry_after`, `README.md`, `README.ru.md`, `.env.example`, тесты
`tests/test_callback_authorization.py`, `tests/test_usage_budget.py`,
`tests/test_chief_model_choice.py`, `tests/test_telegram_stream_core.py`) против
`docs/improvement_2026-09-25/T02-plan.md` и раздела T02 мастер-плана. Diff строго
хирургический, ни один файл вне владения не тронут (`git status --short` до и после
совпадает по остальным файлам).

**Тесты:**
```
~/.venvs/ctb/bin/python -m pytest tests/test_callback_authorization.py tests/test_usage_budget.py \
  tests/test_chief_model_choice.py tests/test_chief_close_async.py tests/test_telegram_stream_core.py \
  -q --no-header -p no:cacheprovider
```
→ 71 passed. Дополнительно прогнаны `tests/test_telegram_builder_config.py`,
`tests/test_telegram_streaming.py` (105 passed) и полный `tests/ bot/tests/`
(1734 passed, 3 warnings — все от PTB deprecation, не от изменений T02).

**ruff:** `bot/utils.py bot/__main__.py bot/plugins/chief.py bot/telegram_stream.py
bot/telegram_bot.py` + затронутые тесты — All checks passed.

**mypy:** прогнан на 5 файлах владения и отфильтрован до строк этих же 5 файлов,
сравнение с `/tmp/impl/mypy_before.keep` построчно по (файл, текст ошибки) без учёта
смещения номеров строк (в utils.py/chief.py номера сместились из-за добавленных
строк — учтено). Результат: **0 новых ошибок**, **4 ошибки исчезли** как следствие
самого фикса —
`bot/telegram_stream.py:199` и `bot/telegram_bot.py:4314`/`:4457`
(`Argument 1 to "sleep" has incompatible type "int | timedelta"; expected "float"`,
ушли после `retry_after_seconds`) и `bot/plugins/chief.py:428`
(`Missing return statement`, ушла после добавления `else: raise ValueError(...)`,
все пути функции теперь либо `return`, либо `raise`).

### Построчная проверка шагов плана

1. `.strip()` в `_charge_user_and_guest`/`_charge_user_and_guest_async`
   (`bot/utils.py:868`, `:882` факт.) — применено дословно по плану, идентично
   парсингу в `is_allowed` (`utils.py:680`). Тесты `test_charge_user_and_guest_*`
   зелёные, проверяют именно то, что заявлено (гостевой трекер не тронут).
2. `chief.py:_parse_menu_preferences` — добавлен `else` с тем же текстом ошибки,
   что в `except`. Проверено чтением: поднятый в `else` `ValueError` не перехватывается
   соседним `except (json.JSONDecodeError, ValidationError)` (не входит в кортеж типов),
   пробрасывается до вызывающего `execute()` (`chief.py:551`) и там ловится общим
   `except Exception as e: return {"error": ...}` (`chief.py:569`) — ровно то поведение,
   которое требовал план. Тест `test_parse_menu_preferences_raises_value_error_when_no_json_found`
   бьёт именно этот путь.
3. `retry_after_seconds(exc) -> float` — добавлена в `bot/telegram_stream.py` после
   `logger = logging.getLogger(__name__)`, используется в `telegram_stream.py:214`
   (сдвиг от заявленной `:199` из-за вставленной функции — ожидаемо) и в
   `telegram_bot.py:4314`/`:4457` (только эти две строки изменены, импорт добавлен
   к существующей строке `from .telegram_stream import ...`). Обработка `int`/`float`/
   `timedelta` корректна. Юнит-тесты на конвертацию + тест на реальном `RetryAfter`
   с `PTB_TIMEDELTA=true` — все проходят.
4. Флаг `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` → `allow_group_members_via_authorized_user`
   в `telegram_config` (`__main__.py:328-330`, через `env_bool`, default `True`).
   `is_allowed` (`utils.py:685-686`): условие расширено до
   `not is_inline and is_group_chat(update) and config.get('allow_group_members_via_authorized_user', True)`
   — при `False` весь блок (парсинг `admin_user_ids`, цикл `is_user_in_group`,
   финальный `logging.info` об отказе) не выполняется, `get_chat_member` физически
   не вызывается — подтверждено тестами
   `test_group_membership_grant_disabled_rejects_non_member_without_api_call` и
   `test_group_callback_rejected_when_membership_grant_disabled`
   (`context.bot.get_chat_member.assert_not_awaited()`). `config.get(..., True)` даёт
   старым конфигам (без ключа) прежнее поведение — подтверждено тем, что три
   существующих групповых теста в файле не изменены и всё равно зелёные.
5. Лог режима при старте (`__main__.py:361-371`) — оба варианта (`True`/`False`)
   присутствуют, текст соответствует плану, размещён после закрытия
   `telegram_config` и до `plugin_config`, использует `logging.info` (логирование уже
   настроено на `INFO` в `main()` до этой точки, `__main__.py:190-193` — сообщения
   реально попадут в лог).
6. Документация: `README.md`, `README.ru.md`, `.env.example` — новая строка про флаг
   и уточнённое описание `GUEST_BUDGET` вставлены ровно в места, указанные планом,
   формулировки соответствуют фактическому поведению кода.

### Замечания

Нет ни одного отклонения от плана/мастер-плана, тесты содержательны (проверяют
поведение через отсутствие вызова `get_chat_member`, через типы исключений, через
конвертацию значений), мёртвого кода и стилевых нарушений не найдено.

Одна пограничная заметка без понижения оценки: тест
`test_group_callback_allowed_member_passes_without_api_call_when_membership_grant_disabled`
(`tests/test_callback_authorization.py`) проверяет пользователя, чей id и так есть в
`allowed_user_ids`, поэтому он прошёл бы `is_allowed` до блока с флагом независимо от
значения `allow_group_members_via_authorized_user` — тест не изолирует именно эффект
флага, а дублирует уже покрытый инвариант «прямо разрешённый id не идёт в Telegram API».
Тест списан дословно из плана (`T02-plan.md:483-493`), поведение не нарушено, отчёт —
не блокирующий.

**Итог: 0 ERROR, 0 WARNING, 0 NIT.**
