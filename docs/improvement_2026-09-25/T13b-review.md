# T13b — ревью: `bot/database.py`, `bot/plugins/db_handle.py`, `bot/session_otel.py`,
`bot/session_logger.py`, `bot/utils.py`, `bot/plugins/haiper_image_to_video.py`,
`bot/validation.py`, `bot/skill_script_routing.py` (mypy-типизация)

Ревьюер проверял `git diff HEAD` по 8 файлам, перечисленным во владении T13b, против
`docs/improvement_2026-09-25/T13-plan.md` (раздел T13b, строки ~221-352) и мастер-плана.
Диффы этих же файлов содержат hunks других параллельных задач (T02, T03, T05, T06, T12a,
T12d) — они не оценивались по существу, только на предмет того, что не пересекаются с
T13b-типизацией (см. "Вне области ревью"). Главный вопрос по инструкции ревью — меняет ли
какая-либо типовая правка рантайм-поведение — проверялся построчно для каждой рискованной
категории (`assert`, `-> Optional`-расширение, `cast`, return-пути `ConversationHandler`).

## Раунд 1

### Проверено по каждому пункту плана

**`bot/database.py`** (план: 42 ошибки — `attr-defined` 23, `assignment` 13, `has-type` 3,
`return-value` 3).

- Аннотации класса `_local: threading.local`, `db_path: str`, `_op_lock: threading.RLock`,
  `_executor: Optional[concurrent.futures.ThreadPoolExecutor]` (`:212-215`) — только
  объявление типа, `__new__`/thread-local-паттерн не тронут.
- `_executor` (implicit-Optional, план предупреждал: "если код не проверяет None до
  использования — задокументировать, не чинить"). Проверил `_get_executor()`
  (`:418-436`): двойная проверка `if self._executor is not None: return ...` (`:424-425`),
  затем повторная проверка под локом и ленивая инициализация (`:432-436`) — None-safety уже
  была на месте до T13b, латентного бага нет.
- Implicit-Optional `session_id: str = None` → `Optional[str] = None` в
  `save_conversation_context` (`:1064`), `save_conversation_context_async` (`:1168`),
  `get_conversation_context` (`:1228`), `get_conversation_context_async` (`:1712`),
  `create_session` (`:1543`), `create_session_async` (`:1784`) — чистые аннотации, default
  `None` и тело функций не изменились.
- `assert cursor.lastrowid is not None` (`:1337`) перед `return cursor.lastrowid` в
  `save_image` (`:1323`) — `cursor.lastrowid` после `INSERT` в rowid-таблицу с
  `AUTOINCREMENT` гарантированно не `None`; под `python -O` (assert вырезается) код просто
  возвращает то же значение — идентичное поведение что с assert, что без.
- `export_sessions_to_yaml_async` (`:1816`) и `export_sessions_to_yaml` (`:1957`) —
  `-> Optional[str]`. Единственный вызывающий код — `bot/telegram_bot.py:6242-6257` (вне
  владения T13b) — уже делает `if filepath: ... else: ...`; расширение типа документирует
  уже существующее поведение, не добавляет новую ветку.
- `export_data: Dict[str, Any]` (`:1969`) — чистая аннотация.

**`bot/plugins/db_handle.py`** (план: 5 — `attr-defined` 4, `assignment` 1).

- `TransactionScope.execute` (`:37`), `.executemany` (`:48`), `.fetch_one` (`:59`),
  `.fetch_all` (`:76`) — везде `conn = self._conn; assert conn is not None`, вложенная
  `_run()` теперь захватывает локальную `conn`, а не `self._conn`. Прочитал
  `_ensure_open()`/`_Transaction.__aenter__`/`__aexit__` целиком: `self._conn` не
  переприсваивается между `_ensure_open()` и вызовом `_run()` — замена эквивалентна.
- `self._ctx: Any | None`, `self._conn: Any | None`, `self._open_token: contextvars.Token |
  None` (`:110`) — чистые аннотации, `import contextvars` добавлен только ради типа.

**`bot/session_otel.py`** (план: 6 — `operator` 2, `index` 2, `arg-type` 2).

- `_extract_provider_error_attrs` (`:106`) и `_extract_retry_attrs` (`:116`) —
  `raw_data = event.get('data'); data: dict = raw_data if isinstance(raw_data, dict) else
  {}` — то же runtime-значение, что и старое `event.get('data') if isinstance(...) else
  {}`, просто через промежуточную переменную.
- `_safe_attrs` (`:150`) — `etype = event.get('type'); extractor = ... if isinstance(etype,
  str) else None` — эквивалентно старому `.get(event.get('type'))`: `dict.get` с
  нестроковым/`None`-ключом и раньше просто не находил бы совпадения (ключи словаря —
  строки), новое поведение то же.
- `_add_event_on_parent(self, event: dict, etype: str | None, ...)` (`:352`) — тело функции
  уже применяло `etype or 'event'` до этой правки (перечитано полностью) — аннотация
  перестала быть неверной, поведение не изменилось.

**`bot/session_logger.py`** (план: 5 — `union-attr` 3, `return-value` 2).

- `assert self._queue is not None` в `_writer_loop` (`:240-241`) и
  `_release_pending_waiters` (`:263-270`) — `self._queue` устанавливается один раз при
  старте и не сбрасывается в `None` (проверены все присваивания `self._queue =` в файле).
- `trace = get_trace(); self._otel.on_event(..., trace.turn_id if trace else None)`
  (`:294`) — заменяет двойной вызов `get_trace()` (было `get_trace().turn_id if
  get_trace() else None`) на одинарный с сохранением в переменную; `get_trace()` синхронна,
  без побочных эффектов между вызовами — результат идентичен.
- `flush_summary` (`:346`) — `-> bool | None`, ранний `return` → `return None` на ветке
  `not self.enabled`. План прямо допускает такой путь ("либо унифицировать return, либо
  расширить аннотацию") — тело функции дальше по-прежнему может возвращать `bool`.

**`bot/utils.py`** (план: 43 — `attr-defined` 19, `assignment` 10, `arg-type` 5,
`valid-type` 4, `union-attr` 3, `var-annotated` 1, `misc` 1).

- `wrap_with_indicator`'s `chat_action: constants.ChatAction | str = ""` (`:553-554`) — план
  явно предписывал не менять default-значение (`""` может быть намеренной
  PTB-совместимостью), а расширить тип параметра — сделано буквально так.
- `assert update.effective_chat is not None` (`:565`) — стоит перед
  `await update.effective_chat.send_action(...)`, не оборачивает сам вызов с побочным
  эффектом (assert проверяет только доступность объекта, не выполняет действие).
- `User | None`-аннотации в `is_allowed` (`:662`), `get_remaining_budget`,
  `_budget_user_and_name` — чистые аннотации; попутно переименована loop-переменная
  `user` → `candidate_id` в group-membership цикле `is_allowed` (`:691`) — то же значение,
  другое имя, не шадоуит внешний `user` (сравнил оба диффа построчно).
- `resize_image_if_needed` (`:1184-1210`) — `assert img.format is not None` (`:1197`) +
  новая переменная `resized_img: Image.Image = img` (`:1202`), далее
  `resized_img = img.resize(...)` вместо `img = img.resize(...)`, `resized_img.save(...)` в
  конце. Логика идентична: раньше `img` переприсваивалась резайзнутой версией (или
  оставалась исходной), теперь то же хранится в `resized_img`; `img` — переменная
  контекст-менеджера `with Image.open(...) as img`, её не трогают ради ясности, `format`
  (bytes-переменная, не builtin) возвращается тот же.
- `handle_direct_result` (`:1219`): `message = cast(Message, message)` (`:1269`) с явным
  комментарием, почему `cast`, а не `assert`/`isinstance` — тестовые дублёры сообщения не
  являются реальными `telegram.Message`, `isinstance`-guard сломал бы тесты; `cast` —
  статическая аннотация без рантайм-эффекта. Соответствует дизайну из плана (T13-plan.md
  ~строки 130-144: "выбор между isinstance-guard-с-return и assert/cast — на усмотрение
  исполнителя, критерий — не менять поведение при уже-корректном входе").
- 6× `assert value is not None, "..."` (`:1314, 1326, 1346, 1358, 1363, 1375`) в
  photo/gif/file × url/path-ветках — каждый добавлен либо перед прямым использованием
  `value` в вызове API, либо после уже существующей проверки `is_deliverable(...)` (T05,
  не T13b) — под `python -O` код просто продолжает выполняться с тем же `value`; ни один
  assert не оборачивает сам вызов с побочным эффектом.
- `message_parts = [(chunk, None) for chunk in chunks]` → `[(chunk, []) for chunk in
  chunks]` (`:1420`) — проверено в установленном PTB v22.8: `Bot.send_message`'s
  `entities` по умолчанию `None`; `Bot._post` фильтрует из payload значения `None`
  (`{k: v for k, v in data.items() if v is not None}`), но НЕ фильтрует falsy `[]`. Итог:
  `entities=[]` теперь явно попадает в payload вместо отсутствия ключа — для Telegram Bot
  API пустой список entities и отсутствие поля семантически одно и то же (нет
  форматирования). Ни один тест не проверяет форму payload на этом уровне.
- `assert update.effective_message is not None` (`:1573`) в `send_long_response_as_file` —
  стоит до `.reply_document(...)` (следующая строка), не оборачивает вызов.
- `response: any` → `response: Any` в `is_direct_result` (`:1069`),
  `direct_result_inline_fallback_text` (`:1077`), `handle_direct_result` (`:1219`),
  `cleanup_intermediate_files` (`:1490`) — `any` (builtin-функция) как аннотация типа была
  рантайм-безобидной (аннотации не проверяются в рантайме без typechecked-декораторов), но
  семантически неверной; `Any` (typing) — корректная замена без поведенческого эффекта.

**`bot/plugins/haiper_image_to_video.py`** (план: 69 — `union-attr` 54, `return-value` 7,
`assignment` 3, `arg-type` 3, `attr-defined` 1, `func-returns-value` 1). Самый плотный файл
после `telegram_bot.py` (T13a).

- `TempFileManager.temp_file: Optional[IO[bytes]]`, `.path: Optional[str]` (`:63-64`) —
  implicit-Optional паттерн на файловом хендле.
- `assert chat_id is not None, "..."` (`:588`) перед `VideoTask(user_id=chat_id,
  chat_id=chat_id, ...)` — план прямо предписывал None-guard перед конструктором; assert
  стоит до вызова, не внутри него.
- Множественные `assert message.from_user is not None` (`:617, 833, 1323, 1422, 1474`),
  `assert message.reply_to_message.document is not None` (`:637`), `assert query is not
  None` / `assert query.data is not None` (`:857, 861`) — все оборачивают чтения
  атрибутов, ни один не оборачивает вызов с побочным эффектом.
- `msg = cast(Message, query.message)` (`:878` и `:965`, close_menu-ветка и
  exception-хендлер `handle_callback_query`) — с комментарием, объясняющим выбор `cast`
  вместо `assert`/`isinstance` (тестовые дублёры `query` — не реальные PTB-объекты); тот же
  дизайн, что и в `utils.py`.
- **Главный проверяемый по инструкции вопрос**: `handle_prompt_constructor` (`:1310`) —
  сигнатура `update: Update = None, ...) -> None` → `update: Optional[Update] = None,
  ...) -> int`. Функция прочитана целиком: КАЖДЫЙ путь и до, и после правки уже возвращал
  `int` (`ConversationHandler.END` и состояния `WAITING_PROMPT`/т.п.) — старая аннотация
  `-> None` была объявленческой ошибкой, а не описанием реального поведения; план допускает
  такой путь буквально ("если все пути и так возвращают int — унифицировать не пришлось").
- `handle_prompt_reply` (`:1418`) — `-> int` → `-> Optional[int]`. Две ветки, которые
  раньше завершались голым `return` (`if not message.reply_to_message: ... return` и
  `if not reply_text or ...: ... return`), уже существовали ДО T13b — сверено с диффом:
  знак `+` стоит только на добавлении `None` после `return`, сам оператор `return` не
  новый. То есть обе ветки уже неявно возвращали `None`, T13b лишь сделал это явным
  (`return` → `return None`). PTB v22.8
  (`ConversationHandler._update_state`, инспектировано напрямую в установленном пакете)
  трактует `None` как "остаться в текущем состоянии" — предсуществующая, задокументированная
  в PTB семантика; T13b её не создаёт, только делает явной в типах.
- `net_safety.safe_get(...)` (T03) и `_build_settings_summary_text`/
  `_build_style_effect_keyboard` (T12d) присутствуют в диффе — не относятся к T13b, см.
  "Вне области ревью".

**`bot/validation.py`** (план: 2 — `arg-type` 1, `var-annotated` 1).

- `required: Dict[str, Any]` (`:12` в `validate_openai_config`) — план предполагал
  переписать сам вызов `isinstance`; по факту минимальный и достаточный фикс — аннотировать
  словарь как `Dict[str, Any]`, из-за чего `expected` в цикле получает тип `Any`, и mypy
  больше не жалуется на второй аргумент `isinstance(config.get(key), expected)`. Сами
  значения словаря (типы и tuple типов, например `(int, float)`) и вызов `isinstance` не
  менялись.
- `parts: List[str] = []` (`:78`, `_join_path`) — чистая аннотация.

**`bot/skill_script_routing.py`** (план: 2 — `assignment` 2).

- `payload: dict[str, Any] = {...}` в `_skill_script_routing_payload` (`:67`) — план
  описывал ошибку как "переменная выведена как `Sequence[str]`, но присваивается
  `list[dict]`/`dict`"; это ровно два последующих присваивания —
  `payload["available_skill_scripts"] = active_scripts` (список `dict`) и
  `payload["suggested_tool_call"] = {...}` (`dict`), которым до явной аннотации mypy
  сужал тип `payload` по первому вложенному литералу. Словарь и его содержимое не
  изменились.

### Тесты и статический анализ

- `python3 -m mypy` на все 8 файлов группы (`--python-executable ~/.venvs/ctb/bin/python
  --ignore-missing-imports`) — `Success: no issues found in 8 source files`.
- Полный `python3 -m mypy --config-file pyproject.toml --python-executable
  ~/.venvs/ctb/bin/python bot` (85 файлов проекта) — `Success: no issues found in 85 source
  files`; подтверждает отсутствие регрессий на стыке с T13a/T13c (например, изменение
  базового класса в `plugin.py` не создаёт новых ошибок в `haiper_image_to_video.py`).
- `~/.venvs/ctb/bin/python -m ruff check` на все 8 файлов — `All checks passed!`.
- Целевые тесты: `tests/test_database.py` + `tests/test_db_handle.py` — 77 passed;
  `tests/test_session_logger.py` + `tests/test_utils_parse_model_choices.py` +
  `tests/test_t12d_haiper_menu_text.py` + `tests/test_haiper_image_to_video_async_db.py` +
  `tests/test_session_otel.py` + `tests/test_utils_send_long_response_file.py` +
  `tests/test_plugin_arg_validation.py` — 68 passed.
- Полный прогон `tests/ bot/tests/` — 1950 passed, 0 failed, 0 регрессий.
- `# type: ignore` в 8 файлах группы — 0 вхождений (проверено построчным поиском по каждому
  файлу).

### Находки

**ERROR:** нет.

**WARNING:** нет.

**NIT (1):**

1. `get_callback_handlers()` (`bot/plugins/haiper_image_to_video.py:361-375`) — метод не
   вызывается нигде в проекте (проверено полным поиском по всем `.py`-файлам: единственное
   вхождение строки `get_callback_handlers` — само определение метода; `PluginManager`
   регистрирует callback-хендлеры haiper через `get_commands()`'s
   `callback_query_handler: self.handle_callback_query` с паттерном `^haiper_`, где реально
   вызывается `apply_settings(query)` напрямую). T13b добавил типизацию (`-> List[Dict]`,
   `-> None` на вложенном `_apply_settings_from_callback`) мёртвому коду — не ошибка T13b
   (метод был недостижим и до правки, T13b не менял его структуру), но раз он попал в
   диапазон правки, стоит зафиксировать для будущей чистки. Не блокер, вне объёма
   типизации.

### Вне области ревью (замечено, не оценивалось)

- `bot/database.py`: `LEGACY_PROMPT_FINGERPRINTS` (`:21`), `TARGET_SCHEMA_VERSION = 3`,
  `SHAPE_VERIFIABLE_SCHEMA_VERSION = 2`, `_migrate_conversation_context_backfill_mode_key`
  (миграция 3) — это T06 (backfill `mode_key` в старых сессиях), не пересекается с
  T13b-типизацией.
- `bot/utils.py`: `is_deliverable`/`is_protected_path`/`_artifact_scope_for_update` и
  связанные ветки в `handle_direct_result`/`cleanup_intermediate_files` — T05
  (артефакт-пути/доставка файлов); `.strip()` в `allowed_user_ids`/
  `_charge_user_and_guest(_async)` и `config.get('allow_group_members_via_authorized_user',
  True)` — T02; полностью новая `parse_model_choices` (используется
  `bot/telegram_bot.py`'s `_configured_openai_models`, задокументирована как B6 в
  `T12b-review.md`) — T12a.
- `bot/plugins/haiper_image_to_video.py`: `net_safety.safe_get(...)` вместо ручного
  aiohttp-скачивания, `self.max_video_bytes` — T03; `_build_settings_summary_text`/
  `_build_style_effect_keyboard` (вынесение общего кода из `handle_prompt_constructor` и
  соседних методов) — T12d.

### Итог

0 ERROR, 0 WARNING, 1 NIT (не блокирующий, для протокола). Все размеченные в инструкции
ревью риск-категории — каждый новый `assert` (ни один не оборачивает выражение с побочным
эффектом), `-> Optional`-расширения (`export_sessions_to_yaml`, `flush_summary`,
`handle_prompt_reply`), `cast` вместо `assert`/`isinstance` на `query.message`/`message`
(мотивировано тестовыми дублёрами, задокументировано в коде), haiper
`get_callback_handlers`-замыкание (мёртвый код, поведение не меняется в силу
недостижимости), `handle_prompt_constructor`/`handle_prompt_reply` return-пути
(ConversationHandler: `None` = "остаться в состоянии" — предсуществующая PTB-семантика,
не изменена), `entities: None → []` (эквивалентно на уровне Telegram Bot API) — проверены
построчно и подтверждены как не меняющие рантайм-поведение. mypy (8 файлов группы и весь
проект, 85 файлов), ruff и полный pytest (1950 passed) — чисто. Ноль `# type: ignore`.
