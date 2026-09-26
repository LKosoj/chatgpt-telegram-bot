# W9 — ревью группы B (данные/персистентность/промпт-данные)

Ревьюер проверил `git diff HEAD` (HEAD 08bc457) для владеемых файлов: `bot/database.py`,
`bot/plugins/db_handle.py`, `bot/chat_modes_registry.py`, `bot/chat_modes.yml`,
`bot/plugins/agent_cron.py`, `bot/plugins/reminders.py`, `bot/conversation_state.py` (новый),
`bot/session_logger.py`, `bot/session_otel.py`, `bot/conversation_key.py`, `bot/validation.py`,
`bot/skill_script_routing.py`, `bot/plugin_manager.py`, плюс закреплённые тесты. Каждый файл
сверен построчно с финальным (закрытым) состоянием из `T01-review.md`, `T04-review.md`,
`T06-review.md`, `T11-review.md`, `T12ad-review.md`, `T13b-review.md` — расхождений с этими
уже принятыми решениями не найдено. Не переподнимались: удаление `bot/skills/**` без бэкапа,
правки `.env.example`/`.gitignore` (T07), текст skills_agent "Function … returned" (T08),
compat-view свойства с count guard (T11).

## Раунд 1

### Проверено и корректно (без замечаний)

- `bot/plugin_manager.py` — только удаление 3 неиспользуемых методов
  (`is_subagent_function_allowed`, `get_all_plugin_descriptions`, `get_plugin_spec`), как в
  T01-plan; `get_spec()`/tool-specs не тронуты.
- `bot/conversation_key.py:12` — добавлен `assert update.effective_user is not None` перед
  `.id`; под `-O` (assert вырезан) поведение идентично старому (падение с `AttributeError`),
  без `-O` меняется только тип исключения (`AssertionError` вместо `AttributeError`) —
  безопасная типизационная правка, `except AttributeError` рядом с вызовами
  `get_conversation_key(` в `telegram_bot.py` не найден.
- `bot/session_logger.py`, `bot/session_otel.py`, `bot/skill_script_routing.py`,
  `bot/validation.py` — только typing-правки T13b (`assert`, `Optional`/`dict`-аннотации,
  однократный вызов `get_trace()` вместо двух без побочных эффектов между вызовами);
  поведение не меняется.
- `bot/database.py` — миграция 3 (backfill `mode_key` через `LEGACY_PROMPT_FINGERPRINTS`,
  `TARGET_SCHEMA_VERSION=3`, `SHAPE_VERIFIABLE_SCHEMA_VERSION=2`) + typing-правки T13b слоями
  друг на друге, без конфликтов. Независимо пересчитан sha256 хардкод-текста
  `LEGACY_ASSISTANT_PROMPT` из `tests/test_database.py` — совпадает байт-в-байт с записью
  `"assistant"` в `LEGACY_PROMPT_FINGERPRINTS`. Тесты покрывают fresh/v1(legacy без
  session_id)/v2/после-миграции-3 сценарии, идемпотентность (`_migrate_conversation_context_
  backfill_mode_key` вызван повторно вручную) и regression-тест на то, что reconcile не
  откатывает schema_version=3 (`test_reconcile_does_not_roll_back_schema_version_after_
  migration_3`) — все не вакуумны, реально проверяют заявленное.
- `bot/plugins/db_handle.py` — только typing-правки T13b (`assert conn is not None` сразу
  после `_ensure_open()` перед использованием локальной `conn` в замыкании `_run()`); во всех
  методах `_ensure_open()` вызывается раньше захвата `conn`, так что assert не может
  сработать на реально пустом соединении.
- `bot/chat_modes_registry.py` — `_substitute_shared_blocks()` (T06): `shared_blocks` каждый
  раз парсится заново из файла (`self._data = data` перед substitute), не течёт между
  перезагрузками; единственный текущий блок `web_search_rules` не содержит вложенных
  `{{shared:...}}` меток (рекурсии подставлять нечего). Покрыто
  `test_shared_blocks_are_substituted_and_hidden` и
  `test_web_search_rules_shared_block_appears_once_in_real_yaml` (пришпиливает 23 вхождения
  в реальном файле) — не вакуумные тесты.
- `bot/chat_modes.yml` — скриптом сверено: множество ключей режимов совпадает с HEAD;
  `tools`/`parse_mode`/`prompt_markers` байт-в-байт идентичны HEAD для всех 25 режимов (никаких
  иных полей, кроме `prompt_start`, diff не коснулся). Текстовые правки (замена 23× блока
  "ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ…" на `{{shared:web_search_rules}}`, удаление 18×
  дублирующей строки `research_articles`, `optimize_prompt`→`prompt_perfect` в 4 местах,
  смягчение "НЕ ПЫТАЙТЕСЬ"→"Не") соответствуют инвентаризации и решению из `T06-plan.md`
  §1/§4.2 (там же зафиксировано, что ни одно "бытовое" имя в прозе — ни старое, ни новое — не
  совпадает с реальным именем функции, которое видит модель; замена на `tools:`-идентификатор
  плагина — осознанное решение T06, не потеря смысла). Потери смысла в тексте не найдено.
- `bot/plugins/agent_cron.py` — SQLite-миграция T04 (claim-with-lease, `_finish_job`,
  идемпотентный импорт с `.migrated`) в финальном виде из `T04-review.md` (раунд 4, E1/W1/W2/
  N1/N2 фиксы на месте). Точечно проверена замена `chat_id = update.effective_chat.id` →
  `chat_id = message.chat_id` в `handle_cron_command`: по исходнику `telegram.Update`
  (`effective_chat`/`effective_message` properties) оба свойства для команд (`message`/
  `edited_message`/…) резолвятся из одного и того же объекта в одном порядке, поэтому при
  непустом `message` (уже проверено `if not message: return` до этой строки)
  `message.chat_id == update.effective_chat.id` всегда — не регрессия, безопасная mypy-правка
  (T13c). `get_spec()` сверен AST-сравнением исходников HEAD/worktree — идентичен.
- `bot/plugins/reminders.py` — SQLite-миграция T04 (send/delete split для W3, `status !=
  'sent'` фильтр на 3 листинговых SELECT для W4, retry-cleanup top-of-tick, неусечённый
  `now_utc_iso` для N1) в финальном виде из `T04-review.md`; extraction
  `_build_reminders_keyboard()` (T12d) — сверено с `T12ad-review.md`, формат кнопок/
  callback_data не изменился, `tests/test_t12d_reminders_keyboard.py` существует и сравнивает
  реальную клавиатуру через БД-фикстуру. `get_spec()` сверен AST-сравнением — идентичен HEAD.
  Reclaim-после-рестарта: `test_stale_lease_job_reclaimable`/
  `test_stale_processing_lease_reclaimable` явно используют другой `worker_id`
  ("old-worker"→"new-worker"), подтверждая, что переклейм не зависит от pid/worker_id между
  перезапусками процесса, только от истёкшей аренды (`locked_at <= lease_cutoff`).
- `bot/conversation_state.py` (новый файл, 179 строк) — прочитан целиком; совпадает с описанием
  из `T11-review.md` (раунд 2, чисто). `sweep()` корректно уменьшает `over_cap` по мере
  вычищения (LRU от старых к новым), локи пропускаются; `_FieldView.__iter__` материализует
  список без `peek()`/`move_to_end()`, поэтому не ловит "OrderedDict mutated during
  iteration"; `replace()` корректно реализует семантику "весь словарь заменён".
- Прогон `~/.venvs/ctb/bin/python -m pytest tests/test_database.py
  tests/test_chat_modes_registry.py tests/test_agent_cron_plugin.py
  tests/test_agent_cron_storage.py tests/test_reminders_fixes.py tests/test_background_tasks.py
  tests/test_concurrent_tool_state.py tests/test_conversation_state.py
  tests/test_compat_state_views_guard.py tests/test_skills_metadata.py
  tests/test_session_logging_integration.py tests/test_t12d_reminders_keyboard.py -q` — 176
  passed. `ruff check` на всех owned-файлах — чисто. `mypy` на всех owned-файлах — "Success:
  no issues found in 12 source files".

### WARNING

1. **`AGENTS.md` — file:line-ссылки на `bot/database.py` и `bot/plugin_manager.py` устарели
   относительно текущего дерева.** Не регрессия по коду, но нарушает собственное правило
   AGENTS.md "cite file:line" — ссылки сейчас указывают в другое место. Примеры (проверено
   открытием строк): `bot/plugin_manager.py` — `get_plugin_commands()` заявлен на `:939`,
   фактически `:885`; `build_bot_commands()` заявлен на `:966`, фактически `:912`;
   `_active_plugin_instances()` заявлен на `:1067`, фактически `:1013` (все три ровно на 54
   строки выше — это в точности размер удаления T01, т.е. регрессия ссылок напрямую вызвана
   этим changeset'ом). `bot/database.py` — класс `Database` заявлен на `:174`, фактически
   `:89`; `save_conversation_context()` заявлена на `:977`, фактически `:1064`;
   `ensure_session_name_async()` заявлена на `:1109`, фактически `:1196` (смещение от вставок
   T06/T13b). Предложение: один проход массовой правки номеров строк в разделах "Database
   Rules" и "Plugin And Tool Rules" (или замена на диапазоны/якоря без жёсткой привязки к
   номеру строки, если это предпочтительнее для поддержки).

Итог: 0 ERROR, 1 WARNING (устаревшие file:line в AGENTS.md), 0 NIT.

## Раунд 2

### Проверка WARNING из Раунда 1 (AGENTS.md → plugin_manager.py / database.py)

Открыты все процитированные строки в текущем `AGENTS.md` и сверены с фактическим кодом.

- `bot/plugin_manager.py`: `AGENTS.md` заявляет `get_plugin_commands()` на `:939`,
  `build_bot_commands()` на `:966`, `_active_plugin_instances()` на `:1067`. В HEAD
  (`git show HEAD:bot/plugin_manager.py`) это действительно 939/966/1067 — заявленные номера
  описывают HEAD, а не текущее дерево. В текущем worktree это 885/912/1013 (ровно на 54
  строки выше — размер удаления T01). Раунд 1 здесь точен, подтверждено.
- `bot/database.py`: `AGENTS.md` заявляет `__new__()` на `:211`, `_op_lock` на `:222`,
  "class starts" на `:174`, thread-local storage на `:225`, foreign keys на `:299`,
  journal_mode/WAL на `:300-304`, busy_timeout на `:305-306`, `save_conversation_context()`
  на `:977`, `ensure_session_name_async()` на `:1109`. В HEAD все эти номера подтверждены
  байт-в-байт (проверено построчным чтением `git show HEAD:bot/database.py`). В текущем
  worktree — `__new__` :251, `_op_lock` (присвоение в `__new__`) :262, класс :207,
  thread-local :265, foreign keys :339, journal_mode/WAL :340-344, busy_timeout :345-346,
  `save_conversation_context()` :1064, `ensure_session_name_async()` :1196 — то есть Раунд 1
  прав по существу (все ссылки устарели из-за миграции 3 + typing-правок T13b).
  **Уточнение/исправление одного числа из Раунда 1**: там написано «класс `Database`
  заявлен на `:174`, фактически `:89`» — это ошибка Раунда 1. Строка `:89` в текущем
  дереве — это `class DatabaseLockTimeoutError(RuntimeError):` (имя класса начинается с той
  же подстроки "class Database", похоже на ложное совпадение при поиске). Фактическая строка
  `class Database:` в текущем дереве — `:207`, не `:89`. Смещение класса — 33 строки (207-174),
  не как у остальных членов класса (+40) и методов ниже (+87), потому что часть вставок (блок
  `LEGACY_PROMPT_FINGERPRINTS`, аннотации полей) физически лежит между началом класса и его
  более поздними методами. Само существование WARNING (устаревшие ссылки) подтверждено,
  исправлена только одна конкретная цифра внутри него.

### Свежий проход (JSON-импорт, часовые пояса reminders/cron, миграция на HEAD-схеме, ChatStateRegistry)

1. **DB-миграция на БД, реально созданной кодом HEAD** — собран `bot/database.py` из
   `git show HEAD:bot/database.py` как отдельный модуль, им создана БД и сохранена "легаси"
   сессия с системным сообщением, равным реальному (без правок T06) `prompt_start` режима
   `assistant` из `git show HEAD:bot/chat_modes.yml`, без `mode_key`. Затем та же БД открыта
   текущим (worktree) `bot/database.py` — `mode_key` корректно проставлен миграцией 3 в
   `'assistant'`. Отдельно скриптом пересчитан sha256 всех 25 значений
   `LEGACY_PROMPT_FINGERPRINTS` (`bot/database.py`) против РЕАЛЬНОГО `prompt_start` каждого
   режима из `git show HEAD:bot/chat_modes.yml` (не против тестовой константы, как в Раунде 1)
   — все 25 совпали побайтово. Дополнительно на том же HEAD-собранном экземпляре проверены
   сессии с "битым"/частичным JSON (`context` не dict, `messages` не список, пустой список
   `messages`, первое сообщение не `system`) и сессия, где `mode_key` уже стоит — миграция 3
   ни разу не упала и не тронула то, что не должна была трогать. Отдельно вызовом
   `_migrate_conversation_context_backfill_mode_key()` напрямую на INSERT'ах с невалидным
   JSON (`"{not valid json"`, `"null"`), `messages` не списком/строкой, `content` не строкой
   (число), первым сообщением не `system` — исключений нет, строки пропускаются как есть.
   **Не регрессия, проблем не найдено** — миграция 3 устойчива к битым/частичным данным
   лучше, чем требовалось.

2. **ERROR — `bot/plugins/reminders.py:277`, импорт legacy `reminders.json` не валидирует
   `time`, хотя `T04-plan.md` явно требует это (строки 274-275, 345-346: "при импорте
   валидировать `time`/`fire_at_utc` через `datetime.fromisoformat`").** Фактически
   валидируется только `fire_at_utc` (`reminders.py:264-273`, `try/except ValueError` →
   `None` при некорректном значении); `time` берётся как `reminder.get("time") or ""` без
   всякой проверки. Запись из старого `reminders.json` без ключа `"time"` (либо с пустым
   значением) импортируется со `time=''`. `_claim_due_reminders_sync`
   (`reminders.py:305-321`) сравнивает due-статус лексикографически:
   `(fire_at_utc IS NULL AND time <= ?)` — пустая строка лексикографически меньше ЛЮБОЙ
   непустой ISO-строки, поэтому такая запись считается "просроченной" на самом первом же
   тике после апгрейда и немедленно уходит пользователю, вместо того чтобы быть молча
   пропущенной (как делал старый HEAD-код: `reminders.py:262-270` в HEAD ловил
   `ValueError`/`KeyError` при `datetime.fromisoformat(reminder['time'])` и логировал skip,
   запись оставалась в файле навсегда, но никогда не отправлялась). Воспроизведено дважды:
   (а) напрямую вызовом `_import_json_reminders_sync` + `_claim_due_reminders_sync`;
   (б) полным пайплайном через реальные `Database`/`DbHandle`/`RemindersPlugin.initialize()` +
   `check_reminders()` — напоминание с отсутствующим `time` реально доставляется через
   `helper.send_message(...)` на первом тике (`send_message` вызван 1 раз, текст напоминания
   отправлен, строка удалена как "sent").
   Существующий регресс-тест `tests/test_background_tasks.py::test_bad_time_does_not_kill_tick`
   (строки 142-166) не ловит это: он использует `time="NOT-A-DATE"` — непустую строку,
   которая лексикографически БОЛЬШЕ любой ISO-даты (начинается с `N`, ISO-даты начинаются с
   цифры), поэтому "безопасно" никогда не оказывается due — это не тот случай, который дают
   `reminder.get("time") or ""` для отсутствующего ключа. `T04-plan.md` сам ссылается на этот
   тест как на место, где разобран этот edge case (строки 344-352), но тест проверяет только
   "мусор, который случайно сортируется безопасно", а не "отсутствующий/пустой `time`",
   который сортируется опасно. Предлагаемое исправление: валидировать `time` через
   `datetime.fromisoformat` в `_import_json_reminders_sync` так же, как `fire_at_utc`, и при
   ошибке/отсутствии пропускать вставку записи (или логировать и не импортировать), а не
   подставлять `""`.

3. **Проверено — `bot/plugins/agent_cron.py`, импорт `agent_cron_jobs.json` c отсутствующим
   `next_run_at` безопасен.** `job.get("next_run_at")` при отсутствии ключа даёт `None` →
   `NULL` в колонке (schema без `NOT NULL`, `agent_cron.py:87`); due-запрос
   (`agent_cron.py:292-293`) требует `next_run_at IS NOT NULL AND next_run_at <= ?` — задание
   с `NULL` просто никогда не считается просроченным (в отличие от reminders.py, здесь нет
   аналогичного бага). Асимметрия с reminders.py именно в выборе fallback-значения: `NULL`
   у cron безопасен, `''` у reminders — нет.
   Побочно замечено (не регрессия, вне скоупа): `_advance_job()` (`agent_cron.py:554`) падает
   с `TypeError` на `int(job["hour"])`, если `hour`/`minute` не заданы для
   `schedule_type='daily'|'weekly'` (например, из битого legacy-импорта); `_run_job()`
   вызывает `_advance_job()` и в `try`, и в `except`-ветке (`agent_cron.py:372`, `:393`) —
   если первый вызов падает и ловится, второй вызов в `except` не защищён и упадёт с тем же
   исключением, пропустив `_finish_job()`/уведомление пользователя о провале. Идентичная
   структура (`int(job["hour"])` без guard, двойной вызов `_advance_job` в try/except) уже
   присутствует в HEAD (`git show HEAD:bot/plugins/agent_cron.py:373-400`) байт-в-байт — это
   не регрессия этого changeset'а, поэтому не считается в итоге ниже; упомянуто для
   будущих ревьюеров.

4. **Проверено — таймзонная эвристика `fire_at_utc` в `set_reminder` не новая.**
   `reminders.py:438-448` (вычисление `offset = parsed_local_now - utc_now` из необязательного
   `current_time`, `fire_at_utc = reminder_time - offset`) байт-в-байт совпадает с HEAD
   (`git show HEAD:bot/plugins/reminders.py:336-346`) — эта логика не тронута changeset'ом,
   вне скоупа регрессии.

5. **Проверено — `ChatStateRegistry` TTL (`bot/conversation_state.py:106-120`,
   `sweep`/`maybe_sweep`) не расходится по семантике с `max_conversation_age_minutes`
   (`bot/openai_helper.py:3393-3404`, `__max_age_reached`).** Обе стороны используют одно и
   то же значение конфига и одно и то же условие "не трогали дольше max_age минут". Все три
   места вызова `__max_age_reached` (`openai_helper.py:1312`, `:2252`, `:3509`) написаны как
   `state_key not in self.conversations or self.__max_age_reached(state_key) or
   session_changed` — короткое замыкание `or` означает: если запись успела быть вычищена
   `sweep()` до этой проверки, `state_key not in self.conversations` уже истинно и ветка
   "начать заново" срабатывает тем же путём, что и при попадании в `__max_age_reached`;
   `__max_age_reached` для уже вычищенной записи просто не вызывается. Расхождения в
   поведении не найдено — старое (без T11) поведение (`self.last_updated` как обычный dict
   без TTL) заменено на эквивалентное с точки зрения этих трёх call site'ов, плюс появилась
   защита от неограниченного роста памяти (LRU-cap `MAX_CHAT_STATES`), которой раньше не было
   вовсе.

Итог Раунда 2: 1 ERROR (bot/plugins/reminders.py:277 — не валидируется `time` при импорте
legacy-напоминаний, что нарушает T04-plan.md и вызывает немедленную ложную отправку
"просроченного" напоминания без времени), 0 новых WARNING, 1 уточнение внутри WARNING
Раунда 1 (неверная "фактическая" строка `:89` → должно быть `:207`), 0 NIT.

## Раунд 3

### Проверка фикса ERROR из Раунда 2 (`bot/plugins/reminders.py`, `_import_json_reminders_sync`)

Открыт текущий код (`bot/plugins/reminders.py:236-308`, миграция не трогалась, только тело
импорт-функции) и построчно сверен с матрицей "что должно происходить":

- `fire_at_utc` валидируется первым (`:264-273`): если ключ есть и не парсится
  `datetime.fromisoformat`, откатывается в `None` (с комментарием про лексикографическое
  сравнение) — без изменений с Раунда 1, не регрессия.
- Новое: если `fire_at_utc is None` (отсутствовал или оказался битым), `time` теперь ТОЖЕ
  валидируется через `datetime.fromisoformat(str(reminder_time))` (`:283-284`); при `ValueError`
  запись пропускается через `continue` (`:285-290`) — вставки в `rows` не происходит,
  `reminder.get("time") or ""` (источник Раунда-2 бага) в этой ветке больше не исполняется.
  Если `fire_at_utc` валиден, `time` не проверяется вовсе и берётся as-is — корректно, т.к.
  `_claim_due_reminders_sync` (`:333-335`, `fire_at_utc IS NOT NULL AND fire_at_utc <= ?` идёт
  первым условием OR) в этом случае `time` для due-сравнения не использует.
- Полная матрица проверена чтением кода: (а) невалидный/отсутствующий `time` + нет валидного
  `fire_at_utc` → запись пропущена, в БД не попадает, никогда не отправится; (б) валидный
  `fire_at_utc` (даже с битым/отсутствующим `time`) → импортируется, due по `fire_at_utc`;
  (в) валидный `time` (без `fire_at_utc` или с битым `fire_at_utc`) → импортируется, due по
  `time`. Все три ветки соответствуют требованию задания.
- **Сравнение с HEAD (`git show HEAD:bot/plugins/reminders.py:262-270`)**: старый
  `check_reminders` ловил `ValueError`/`KeyError` на `datetime.fromisoformat(reminder['time'])`
  и логировал skip НА КАЖДОМ тике (запись оставалась в JSON-файле навсегда, никогда не
  отправлялась). Новый код логирует `WARNING` ОДИН РАЗ, в момент импорта, и запись физически
  не попадает в таблицу `reminders` (при этом сырые данные не теряются — оригинальный
  `reminders.json` переименовывается в `.migrated`, а не удаляется). Конечный инвариант
  "битая запись никогда не отправляется" сохранён тем же способом, что предложил Раунд 2;
  различается только механизм (drop при импорте вместо бесконечного skip на каждом тике) — это
  сознательное упрощение одноразовой миграции, не регрессия, и даже снижает частоту логов.
- Проверен edge-case "всё в файле битое": даже если после фильтрации `rows` пусто, `os.replace`
  на `:308` выполняется безусловно (не под `if rows:`), поэтому legacy-файл всё равно
  переименовывается в `.migrated` и при следующем рестарте `os.path.exists(self.reminders_file)`
  (`:247`) даст `False` — повторного импорта и повторного WARNING не будет.
- Лог `:286-289` печатает только `reminder_id`/`owner_id` (Telegram user id) — без текста
  сообщения и без сырого значения `time`; это тот же уровень PII-безопасности, что и у
  соседних `logging.warning`/`logging.error` в этом файле (например, `:598` — "giving up on
  reminder %s for user %s" тоже логирует owner/user id).
- Тест `tests/test_reminders_fixes.py::test_import_skips_record_with_missing_time_valid_records_still_imported`
  (`:457-486`) — содержательный, не вакуумный: вызывает реальный `plugin.initialize(db=...,
  storage_root=...)` (тот же путь, что и в проде, `:207`), а не приватный метод напрямую;
  проверяет, что в БД остаётся только `{"ok"}` из двух записей, и ДОПОЛНИТЕЛЬНО прогоняет
  `check_reminders()` и проверяет `helper.sent == []` — то есть закрывает оба конца бага
  (запись не импортирована И ничего не отправлено).
- `get_spec()` `reminders.py`/`agent_cron.py` — AST-сравнение с HEAD, идентичны (форбидден-правило
  про tool-спеки не нарушено).

**Вывод: ERROR из Раунда 2 исправлен корректно, поведение "битая запись никогда не
срабатывает" сохранено, тест содержательный.**

### Тот же класс бага в `bot/plugins/agent_cron.py` (`_import_json_jobs_sync`, `:459-509`)

Due-запрос для cron (`agent_cron.py:290-293`) использует только одно поле:
`next_run_at IS NOT NULL AND next_run_at <= ?`. Проверено, может ли `next_run_at` попасть в
таблицу пустой строкой (единственное значение, которое проходит `IS NOT NULL` и при этом
лексикографически меньше любой реальной ISO-даты — тот же механизм, что вызвал баг в
reminders.py):

- Импорт-код (`:491`) берёт `job.get("next_run_at")` БЕЗ `or ""`-фолбэка (в отличие от
  исходного `reminder_time or ""` в reminders.py, который и был причиной Раунда-2 бага) —
  отсутствующий ключ даёт `None` → `NULL` в колонке, что due-запрос отсекает.
  `или ""`-паттерна нигде в `_import_json_jobs_sync` для `next_run_at` нет.
- Проверено grep'ом по всему файлу (`python3 -c "re.finditer(r'next_run_at\s*=...')"`) — из
  присваиваний `next_run_at` встречаются только SQL-плейсхолдеры (`?`) и (в остальном коде
  плагина) `next_run.isoformat(...)`/явный `None`; строкового литерала `""` не производит ни
  один путь кода, включая собственный writer `git show HEAD:bot/plugins/agent_cron.py`
  (`next_run_at` там тоже всегда либо `None`, либо `.isoformat()`-строка).
- Т.е. пустая строка для `next_run_at` может появиться только если исходный legacy JSON-файл
  был вручную испорчен и уже содержал буквально `"next_run_at": ""` — не то, что производит
  собственный код HEAD или текущий импорт; тот же класс бага, что в reminders.py (default
  suddenly-due), здесь не воспроизводится, т.к. нет соответствующего fallback'а.
- Подтверждает и расширяет уже сделанное в Раунде 2 (п.3) наблюдение про безопасность `NULL`:
  здесь дополнительно закрыт вопрос именно про "пустую строку", который в reminders.py и был
  первопричиной бага.

**Вывод: в `agent_cron.py` баг того же класса не найден, фикс не требуется.**

### NIT

1. **`T04-plan.md`, раздел "Для reminders" (около `_import_json_reminders_sync`) описывает
   устаревший план — импортировать битые по `time` записи "как есть" ("по-прежнему
   импортируются as-is"), а не пропускать их.** Фактически реализован (и корректен, это и
   есть фикс Раунда 2) пропуск таких записей. В отличие от T12 A1 (задокументированная
   deviation note в конце `T12-plan.md`), здесь нет пометки о расхождении плана с реализацией.
   На рантайм не влияет (реализация правильнее плана), это только устаревший текст плана,
   который может сбить с толку будущего читателя, сравнивающего diff с планом. Предложение:
   добавить в `T04-plan.md` короткую deviation-заметку по аналогии с T12 A1, либо пометить
   абзац как superseded.

### Прогон тестов

`~/.venvs/ctb/bin/python -m pytest tests/test_reminders_fixes.py tests/test_agent_cron_storage.py
tests/test_agent_cron_plugin.py tests/test_background_tasks.py -q --no-header -p no:cacheprovider`
— 51 passed.

Итог Раунда 3: 0 ERROR (Раунд-2 ERROR подтверждён исправленным), 0 новых WARNING, 1 NIT
(устаревший текст `T04-plan.md` про импорт "as-is" вместо skip, без deviation-заметки).
