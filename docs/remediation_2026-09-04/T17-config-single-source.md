# T17. Один источник конфига

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (Волна 4, T17) и
`docs/architecture_code_review_2026-09-04.md`, раздел 5.3.

**Уточнение по нумерации.** В задании T17 сослались на «П4» из аудита — это неверный номер,
он устарел или спутан: П4 в актуальном тексте — «Публичный API состояния сессии в helper»
(`load_session`/`replace_system_message`/`evict`), это отдельная задача **T19**, не T17.
Пункт, который реально описывает эту работу — **П3 «Конфиг: один источник»**
(`docs/architecture_code_review_2026-09-04.md:406-409`). План ниже написан по П3 и по тексту
самой T17 в плане исправлений, а не по П4. Это ещё один пример дрейфа `file:line`/номеров
в документах аудита (см. `project_agents_md_line_refs_drift` в памяти) — цитаты ниже
перепроверены чтением кода на HEAD `af382fb`, а не переписаны из документа.

## Цель

Убрать дублирование чтения env между `bot/__main__.py`, `bot/openai_helper.py` и
`bot/database.py`, не меняя ни одного значения по умолчанию и ни одного пути парсинга,
который сейчас закреплён тестом. Три независимых цели:

1. **`bot/__main__.py`** — 14 ключей сейчас пишутся дважды (в `openai_config` и
   `telegram_config`); булевы читаются двумя разными стилями. Свести к одному
   вычислению на ключ.
2. **`bot/openai_helper.py`** — 30 `self.config.setdefault(...)` в конструкторе дублируют
   дефолты из `__main__.py` вторым источником истины. Оставить только те, что реально
   нужны (есть тест с минимальным config, который их использует), остальные убрать —
   тогда отсутствие ключа в config будет падать явной `KeyError`, а не тихо подставлять
   значение, которое может разойтись с продакшен-дефолтом.
3. **`bot/database.py`** — 4 значения (`db_path`, `max_sessions`, journal mode,
   default model) читаются из env напрямую внутри модуля БД, а не приходят как конфиг.
   Добавить `Database.configure(...)` как **необязательный** явный канал поверх
   существующего env-фоллбэка — не убирая сам фоллбэк, потому что на нём держится ~20
   тестовых файлов через `monkeypatch.setenv(...)` + `Database._reset_singleton()`.

Никаких pydantic/dataclass-конфигов не добавляется — только `shared = {...}` +
`{**shared, ...}`, одна функция для мягких булевых и один classmethod у `Database`.

## Таблица ключей

Все номера строк — по HEAD `af382fb` (текущее состояние дерева), перепроверены построчным
чтением, не переписаны из документа аудита.

### A. 14 ключей, дублированных между `openai_config` и `telegram_config`

Проверено скриптом: `set(keys(openai_config)) & set(keys(telegram_config))` даёт ровно эти
14 имён (подтверждает цифру из аудита).

| Ключ | env | Дефолт (оба места идентичны) | Потребитель |
|---|---|---|---|
| `openai_base` | `OPENAI_BASE_URL` | `''` | `OpenAIHelper`, `ChatGPTTelegramBot` |
| `api_key` | `OPENAI_API_KEY` (уже прочитан в `api_key` до обоих словарей) | обязателен | оба |
| `proxy` | `PROXY`, затем `OPENAI_PROXY` **либо** `TELEGRAM_PROXY` (разные!) | `None` | оба (см. ниже — не 100% дубль) |
| `telegram_rich_messages` | `TELEGRAM_RICH_MESSAGES` (`parse_telegram_rich_mode_env`) | `'auto'` | оба |
| `telegram_rich_drafts` | `TELEGRAM_RICH_DRAFTS` (`parse_bool_env`) | `True` | оба |
| `stream` | `STREAM` (`.lower()=='true'`) | `True` | оба |
| `bot_language` | `BOT_LANGUAGE` (`configured_language(...)`) | `'auto'`→resolved | оба |
| `max_sessions` | `MAX_SESSIONS` (`_parse_numeric_env`) | `5` | оба, и **ещё раз** `bot/database.py` (см. B) |
| `assemblyai_api_key` | `ASSEMBLYAI_API_KEY` | `''` | оба |
| `tts_model` | `TTS_MODEL` (`first_model_env`) | `''` | оба |
| `tts_response_format` | `TTS_RESPONSE_FORMAT` | `'wav'` | оба |
| `data_dir` | `BOT_DATA_DIR` | `''` | оба |
| `output_dir` | `BOT_OUTPUT_DIR` | `''` | оба |
| `plots_dir` | `BOT_PLOTS_DIR` | `''` | оба |

`proxy` — не чистый дубль: `openai_config['proxy']` = `PROXY or OPENAI_PROXY`
(`bot/__main__.py:210`), `telegram_config['proxy']` = `PROXY or TELEGRAM_PROXY`
(`bot/__main__.py:319`). Общая часть — только первичное чтение `PROXY` (сейчас читается
дважды, `os.environ.get('PROXY', None)` на обеих строчках); вторичный фоллбэк — разный
env для разных потребителей. Не сливать в один общий дефолт — только вынести общее
`PROXY`-чтение в одну переменную.

`plugin_config` (`bot/__main__.py:352-354`) в этот список не входит: он содержит только
`plugins` и не пересекается ни с одним из 14 ключей. `plugin_manager.config.update(openai_config)`
(`bot/__main__.py:359`) уже сейчас домешивает туда всё содержимое `openai_config` целиком —
значит добавлять `shared` в сам `plugin_config` не даёт ничего нового (см. «Дизайн»,
раздел про отклонение от буквальной формулировки).

### B. Ключи, дублированные не между двумя словарями `__main__.py`, а между `__main__.py`
и другим модулем (расхождения дефолтов/логики)

| Ключ / env | Где ещё читается | Дефолт там | Расхождение |
|---|---|---|---|
| `MAX_SESSIONS` | `bot/database.py:1164` (`_coerce_max_sessions_limit`, вызывается из `create_session()` без явного `max_sessions=` — `bot/database.py:782,950,1526,1536`) | `5`, но с `max(1, value)` | При `MAX_SESSIONS=0`: `__main__.py`/`telegram_bot.py` (`self.config.get('max_sessions', 5)`) отдают `0` без клампа, `database.py` — `1` (клампит). Существующее расхождение, не вносится этим планом и не устраняется им (см. «Риски»). |
| `OPENAI_MODEL` | `bot/database.py:26-30` (`_first_openai_model_from_env`, используется как `default_model` при `openai_helper is None`: `bot/database.py:495,823,1239`) | `''` если пусто, без исключения | `__main__.py` требует непустой `OPENAI_MODEL` (`parse_model_list_env(..., required=True)`, иначе процесс падает до создания `Database()`) — так что на практике оба чтения всегда дают одно и то же значение, расхождение не наблюдаемо, но это второй независимый парсер того же env. |
| `temperature` | `bot/openai_helper.py:307` (`setdefault`) | **`0.7`** против **`1.0`** в `__main__.py:225` (`_parse_numeric_env('TEMPERATURE', 1.0, float)`) | **Единственное реальное расхождение значения дефолта** в дереве. Эмпирически подтверждено (см. ниже), что это `setdefault` никогда не срабатывает ни в одном известном вызывающем коде (продакшен, `evals/judge/turn_runner.py`, все тесты всегда передают `temperature` явно) — то есть это мёртвый код с неверным дефолтом внутри. Убрать. |

`DB_PATH`, `SQLITE_JOURNAL_MODE`, `SQLITE_TIMEOUT`, `SQLITE_BUSY_TIMEOUT_MS` — читаются
**только** в `bot/database.py` (`bot/database.py:26-60,79,118,122,127`), в `__main__.py`
не встречаются вообще (`grep` подтверждён на HEAD: ни одно из четырёх имён в файле не
встречается). Значит здесь нет «второго источника» — есть один читатель env, который T17
просит заменить на явный конфиг там, где это возможно (`db_path`, journal mode) — но два из
четырёх (`SQLITE_TIMEOUT`, `SQLITE_BUSY_TIMEOUT_MS`) в тексте T17 не перечислены и не входят
в сигнатуру `Database.configure(db_path, max_sessions, journal_mode, default_model)` — не
трогаем их, чтобы не расширять задачу сверх запрошенного.

### C. 30 `setdefault` в `bot/openai_helper.py:307-340` (конструктор `OpenAIHelper.__init__`)

Определено эмпирически: временно пропатчен `bot/openai_helper.py` (в копии дерева в `/tmp`,
не в рабочей копии) так, чтобы при срабатывании `setdefault` (то есть когда ключа
действительно не было в переданном `config`) писать имя ключа и `PYTEST_CURRENT_TEST` в
файл; прогнан `~/.venvs/ctb/bin/python -m pytest -q tests bot/tests` (1500 passed, 1 skipped,
1 failed — см. «Риски», падение не связано с этим патчем). Патч отброшен, в рабочем дереве
изменений нет.

**19 ключей реально используются** хотя бы одним тестом с «минимальным» config (два
независимых fixture: `_make_helper()` в `tests/test_openai_helper_tool_calls.py:440-484`,
переиспользуемый в `tests/test_pricing.py`, `tests/test_skills_prompt_fragment.py`,
`tests/test_stream_usage.py`; и отдельный `make_helper()` в
`tests/test_hindsight_memory.py:158-199`) — **оставить**:

`stream_include_usage`, `model_choices`, `model_context_windows`, `summary_enabled`,
`summary_model`, `summary_max_tokens`, `summary_timeout_seconds`,
`summary_min_messages_between_runs`, `summary_target_keep_ratio`,
`reply_intent_timeout_seconds`, `session_name_timeout_seconds`, `session_log_enabled`,
`session_log_dir`, `session_log_max_bytes`, `session_log_retention_days`,
`session_log_otel_endpoint`, `session_log_otel_service_name`, `session_log_otel_insecure`,
`chat_run_variant_b_enabled`.

(`model_choices` срабатывает только в `test_hindsight_memory.py::make_helper` — это
единственная фикстура без этого ключа.)

**11 ключей никогда не срабатывают** — оба тестовых fixture, продакшен (`bot/__main__.py`)
и `evals/judge/turn_runner.py:142-182` всегда передают их явно — **убрать**:

`temperature`, `presence_penalty`, `frequency_penalty`, `vision_detail`, `n_choices`,
`light_model`, `big_model_to_use`, `tts_model`, `tts_voice`, `tts_response_format`,
`transcription_model`.

Для 5 из этих 11 есть ещё `self.config.get(key, ...)` в других местах helper'а
(`vision_detail` → `'auto'`, `tts_response_format` → `'wav'`, `light_model`/`tts_model`/
`transcription_model` → без дефолта): дефолты у `.get()` совпадают с убираемыми
`setdefault`, новых расхождений при удалении не появляется.

## Дизайн

### (а) `bot/__main__.py` — `shared` + `env_bool`

`bot/utils.py` проверен (`grep def` по файлу) — там нет ни `env_bool`, ни любого другого
парсера булевых/env; `bot/config_utils.py` не существует. Значит переиспользовать нечего,
но кое-что уже есть **в самом `__main__.py`**: `parse_bool_env(name, default)`
(`bot/__main__.py:23-32`) — уже единая функция, просто использована только в 5 из 22 мест.

**Почему не свести все 22 булевых к одной политике** (буквальная просьба T17 — «одна
функция `env_bool`, единая мягкая политика»): `parse_bool_env` **не мягкая** — она
`raise ValueError` на нераспознанное значение, и это осознанно закреплено тестом
`tests/test_telegram_builder_config.py::test_invalid_telegram_local_mode_rejected_before_polling`
(`pytest.raises(ValueError, match="TELEGRAM_LOCAL_MODE")`, вызывает `TELEGRAM_LOCAL_MODE`,
использующий `parse_bool_env`). Если сделать `parse_bool_env` мягкой (warn + fallback, как
у `_parse_numeric_env`) — этот тест сломается по-настоящему (не косметически). Если вместо
этого прогнать нынешние 17 «сырых» `.lower()=='true'` через существующую строгую
`parse_bool_env` — это новое поведение: сейчас `SHOW_USAGE=maybe` тихо даёт `False`, после
такого объединения — `exit`/необработанный `ValueError` при старте. Это ровно то
поведенческое изменение, которого по инструкции быть не должно.

**Выбранный вариант**: два явных чтения одного смысла, а не одна политика на все 22 ключа:

- `parse_bool_env` (`bot/__main__.py:23-32`) — не трогать, оставить как есть (строгая,
  используется в тех же 5 местах: `TELEGRAM_RICH_DRAFTS`, `CHAT_RUN_VARIANT_B_ENABLED`,
  `SUMMARY_ENABLED`, `SESSION_LOG_OTEL_INSECURE`, `TELEGRAM_LOCAL_MODE`).
- Новая `env_bool(name: str, default: bool) -> bool` — байт-в-байт то же самое, что сейчас
  делает инлайн-идиома `os.environ.get(name, default_str).lower() == 'true'`, только без
  тонкости "строковый дефолт ≠ дефолт при некорректном значении": сейчас, например,
  `enable_vision_follow_up_questions` при отсутствии env даёт `True`, а при **мусорном**
  значении — `False` (потому что `.lower()=='true'` сравнивается с самим мусором, а не с
  дефолтной строкой). `env_bool` обязана воспроизводить это один-в-один:

  ```python
  def env_bool(name: str, default: bool) -> bool:
      """Soft boolean env parsing — replaces the inline ``.lower() == 'true'`` idiom.

      Byte-for-byte equivalent of the historical per-call-site expression:
      unset env -> ``default``; **any** other value (including recognizable
      synonyms like ``'1'``/``'yes'``) -> ``False`` unless it is exactly
      ``'true'`` case-insensitively. Never raises. Do not use this for env
      vars whose invalid-value handling must reject startup — those keep using
      the stricter ``parse_bool_env`` above (see
      ``test_invalid_telegram_local_mode_rejected_before_polling``).
      """
      raw = os.environ.get(name)
      if raw is None:
          return default
      return raw.lower() == 'true'
  ```

  Проверка эквивалентности по каждому из 17 мест — построчно в разделе «Правки».

- `shared = {...}` — считается один раз, используется как `{**shared, ...}` в обоих
  словарях:

  ```python
  proxy_env = os.environ.get('PROXY', None)
  shared = {
      'openai_base': os.environ.get('OPENAI_BASE_URL', ''),
      'api_key': api_key,
      'telegram_rich_messages': telegram_rich_messages,
      'telegram_rich_drafts': telegram_rich_drafts,
      'stream': env_bool('STREAM', True),
      'bot_language': bot_language,
      'max_sessions': max_sessions,
      'assemblyai_api_key': os.environ.get('ASSEMBLYAI_API_KEY', ''),
      'tts_model': first_model_env('TTS_MODEL'),
      'tts_response_format': os.environ.get('TTS_RESPONSE_FORMAT', 'wav'),
      'data_dir': os.environ.get('BOT_DATA_DIR', ''),
      'output_dir': os.environ.get('BOT_OUTPUT_DIR', ''),
      'plots_dir': os.environ.get('BOT_PLOTS_DIR', ''),
  }

  openai_config = {
      **shared,
      'proxy': proxy_env or os.environ.get('OPENAI_PROXY', None),
      'proxy_web': os.environ.get('PROXY_WEB', None),
      ...  # остальные ключи, которых нет в telegram_config, без изменений
  }

  telegram_config = {
      **shared,
      'proxy': proxy_env or os.environ.get('TELEGRAM_PROXY', None),
      'token': os.environ['TELEGRAM_BOT_TOKEN'],
      ...  # остальные ключи, которых нет в openai_config, без изменений
  }
  ```

  `plugin_config` не трогается — он не пересекается ни с одним ключом `shared` (см.
  таблицу A), и `plugin_manager.config.update(openai_config)` (`bot/__main__.py:359`)
  и так уже передаёт плагинам весь `openai_config` целиком, включая `shared`-ключи, через
  их собственный `get_config_prefix()`-срез (`PluginManager._plugin_config_segment`,
  `bot/plugin_manager.py:146-158`) — этот механизм уже реализует «плагинам только
  prefix-сегмент» и трогать его не нужно.

### (б) `bot/openai_helper.py` — сократить `setdefault`-блок

Убрать 11 строк (список — раздел «Таблица ключей», C), оставить 19. После удаления
отсутствие любого из 11 ключей в переданном `config` даёт `KeyError` при первом обращении
(`self.config['temperature']` и т.п.) — то же самое поведение, что уже сегодня для
`self.config['model']`, у которого никогда не было `setdefault`.

### (в) `bot/database.py` — `Database.configure(...)` поверх, не вместо, env-фоллбэка

Прямая замена «читать env» на «требовать аргумент конструктора» невозможна без правки
~20 тестовых файлов, которые сейчас делают `monkeypatch.setenv("DB_PATH", ...)` +
`Database._reset_singleton()` и ни разу не проходят через какой-либо явный конфиг
(список файлов — раздел «Тесты»). Значит `Database()` должен по-прежнему уметь работать
без единого явного аргумента — как сейчас.

Решение: `configure()` — необязательный classmethod, вызываемый **до** первого
`Database()` (из `bot/__main__.py::main()`). Если он не вызван — поведение не отличается
от сегодняшнего ни на бит (все проверки на некорректный env: `test_invalid_journal_mode_falls_back_to_wal`,
`test_invalid_sqlite_numeric_envs_fall_back`, `test_malformed_max_sessions_env_falls_back_for_real_session_paths`
и т.д. — идут по прежнему пути и не видят `configure()`).

```python
# bot/database.py — рядом с текущими module-level функциями (после _numeric_env)

def _normalize_journal_mode(raw_mode: str) -> str:
    mode = (raw_mode or "").strip().upper()
    if mode not in SQLITE_JOURNAL_MODES:
        logger.warning("Invalid SQLITE_JOURNAL_MODE=%r; falling back to WAL", raw_mode)
        return "WAL"
    return mode


def _sqlite_journal_mode_from_env() -> str:
    return _normalize_journal_mode(os.getenv("SQLITE_JOURNAL_MODE", "WAL"))
```

```python
class Database:
    _instance = None
    _lock = threading.Lock()
    _connection_lock = threading.Lock()
    _configured: Dict[str, Any] = {}  # explicit overrides set via configure(), empty by default

    TARGET_SCHEMA_VERSION = 2

    @classmethod
    def configure(
        cls,
        *,
        db_path: Optional[str] = None,
        max_sessions: Optional[int] = None,
        journal_mode: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Explicit config for values otherwise read from env inside this module.

        Call before the first ``Database()`` construction (``bot/__main__.py``
        does this right before ``db = Database()``). Any argument left as
        ``None`` keeps today's env-var fallback for that value untouched —
        this is what the tests that ``monkeypatch.setenv(...)`` and call
        ``_reset_singleton()`` without ever calling ``configure()`` rely on.
        """
        if db_path is not None:
            cls._configured['db_path'] = db_path
        if max_sessions is not None:
            cls._configured['max_sessions'] = int(max_sessions)
        if journal_mode is not None:
            cls._configured['journal_mode'] = _normalize_journal_mode(journal_mode)
        if default_model is not None:
            cls._configured['default_model'] = default_model

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super(Database, cls).__new__(cls)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                instance.db_path = (
                    cls._configured.get('db_path')
                    or os.getenv("DB_PATH")
                    or os.path.join(current_dir, 'user_data.db')
                )
                ...
```

`get_connection()` (`bot/database.py:107-129`) — заменить одну строку:

```python
journal_mode = self._configured.get('journal_mode') or _sqlite_journal_mode_from_env()
```

`_coerce_max_sessions_limit` (`bot/database.py:1162-1169`, `@staticmethod`) — добавить
конфигурированный уровень **между** явным аргументом и env:

```python
@staticmethod
def _coerce_max_sessions_limit(max_sessions: Optional[int] = None) -> int:
    if max_sessions is None:
        max_sessions = Database._configured.get('max_sessions')
    raw_value = os.getenv('MAX_SESSIONS', 5) if max_sessions is None else max_sessions
    ...  # без изменений дальше
```

Три места с `_first_openai_model_from_env()` как фоллбэком при `openai_helper is None`
(`bot/database.py:495,823,1239`) — заменить на:

```python
model = (
    openai_helper.config['model'] if openai_helper
    else (self._configured.get('default_model') or _first_openai_model_from_env())
)
```

(на `bot/database.py:495` это классметод/статикметод миграции без `self` — там нужно
`Database._configured.get('default_model')`.)

`_reset_singleton()` (`bot/database.py:99-104`) — дополнительно очищать `_configured`,
чтобы тест, который случайно вызовет `configure()` в будущем, не «протёк» в соседний тест:

```python
@classmethod
def _reset_singleton(cls) -> None:
    with cls._lock:
        instance = cls._instance
        cls._instance = None
        cls._configured = {}
        if instance is not None:
            instance.shutdown()
            instance._close_db_thread_connection()
```

Сегодня `_configured` и так всегда пуст на момент вызова `_reset_singleton()` (никто не
вызывает `configure()`) — эта строчка не меняет поведение ни одного существующего теста,
это чисто защитная мера на будущее.

`bot/__main__.py::main()` — добавить один вызов перед `db = Database()`
(`bot/__main__.py:360`):

```python
Database.configure(
    db_path=os.environ.get('DB_PATH'),
    max_sessions=max_sessions,
    journal_mode=os.environ.get('SQLITE_JOURNAL_MODE'),
    default_model=model,
)
db = Database()
```

`max_sessions` и `model` уже вычислены раньше (`bot/__main__.py:196,190`) — новых
env-чтений в `__main__.py` для `max_sessions`/`default_model` не требуется, только для
`db_path`/`journal_mode` (которые раньше в `__main__.py` не читались вообще).
`SQLITE_TIMEOUT`/`SQLITE_BUSY_TIMEOUT_MS` — не входят в сигнатуру T17, не добавляются.

## Правки по `file:line`

Строки — HEAD `af382fb`, актуальны на момент написания плана; при реализации искать по
именам, не доверять слепо номеру (см. предупреждение выше).

1. `bot/__main__.py:23-32` — не менять (`parse_bool_env`), только добавить `env_bool` сразу
   после.
2. `bot/__main__.py:189-283` (`openai_config` целиком) — ввести `shared`, заменить 7 сырых
   булевых (`show_usage:207`, `stream:208`→в shared, `stream_include_usage:209`,
   `auto_chat_modes:227`, `enable_functions:230`, `show_plugins_used:235`,
   `enable_vision_follow_up_questions:238`, `session_log_enabled:267`) на `env_bool(...)`,
   собрать словарь как `{**shared, ...}`.
3. `bot/__main__.py:293-350` (`telegram_config` целиком) — то же: `shared`-ключи убрать из
   явного перечисления, 8 сырых булевых (`enable_quoting:310`, `enable_image_generation:311`,
   `enable_transcription:312`, `enable_vision:313`, `enable_tts_generation:314`,
   `stream:318`→уже в shared, `voice_reply_transcript:320`, `ignore_group_transcriptions:322`,
   `ignore_group_vision:323`) на `env_bool(...)`.
4. `bot/__main__.py:356-360` — добавить `Database.configure(...)` перед `db = Database()`.
5. `bot/openai_helper.py:307-340` — удалить 11 строк `setdefault` (см. таблицу C), оставить
   19.
6. `bot/database.py:26-60` — добавить `_normalize_journal_mode`, `_sqlite_journal_mode_from_env`
   становится обёрткой над ней; добавить `Database.configure`, `_configured`.
7. `bot/database.py:79` (`__new__`) — `db_path` смотрит в `cls._configured` первым.
8. `bot/database.py:107-129` (`get_connection`) — `journal_mode` смотрит в `self._configured`
   первым; `SQLITE_TIMEOUT`/`SQLITE_BUSY_TIMEOUT_MS` не трогать.
9. `bot/database.py:99-104` (`_reset_singleton`) — очищать `_configured`.
10. `bot/database.py:495,823,1239` — `default_model` смотрит в `_configured` первым.
11. `bot/database.py:1162-1169` (`_coerce_max_sessions_limit`) — `max_sessions` смотрит в
    `_configured` перед env.

## Тесты

Прогонять существующие — новых тестов план не требует (изменение чисто плюмбинговое), но
разработчик на следующем шаге может добавить точечные unit-тесты на `env_bool` и на
`Database.configure()` (не обязательно, план их не предписывает как критерий готовности).

Файлы, которые напрямую проверяют поведение, которое трогает этот план — прогонять в первую
очередь:

- `tests/test_telegram_builder_config.py` — весь `bot/__main__.py::main()`: в частности
  `test_main_uses_same_max_sessions_in_openai_and_telegram_config`,
  `test_main_carries_runtime_path_config_to_openai_and_telegram`,
  `test_main_defaults_telegram_local_bot_api_config`,
  `test_invalid_telegram_local_mode_rejected_before_polling` (закрепляет строгую
  `parse_bool_env` для `TELEGRAM_LOCAL_MODE` — не должен начать проходить иначе или падать
  по другой причине),
  `test_main_enables_chat_run_variant_b_by_default` / `test_main_can_disable_chat_run_variant_b_flag`,
  `test_telegram_builder_sets_proxy_when_configured` / `test_telegram_builder_skips_proxy_when_not_configured`,
  `test_main_parses_explicit_telegram_rich_messages`.
- `tests/test_database.py` — все тесты, которые ставят env перед `Database._reset_singleton()`:
  `db` fixture (`DB_PATH`, строка 18-21), `test_malformed_max_sessions_env_falls_back_for_real_session_paths`,
  `test_create_session_without_helper_uses_first_openai_model_from_env`,
  `test_invalid_journal_mode_falls_back_to_wal`, `test_valid_journal_mode_is_normalized`,
  `test_invalid_sqlite_numeric_envs_fall_back`, `test_non_finite_sqlite_timeout_env_falls_back`,
  `test_legacy_conversation_context_migrates_before_session_index`,
  `test_failed_migration_old_table_with_more_rows_is_recovered`, и другие тесты этого файла,
  использующие `monkeypatch.setenv` + `_reset_singleton` (полный список имён — вывод
  `grep DB_PATH|MAX_SESSIONS|SQLITE_|OPENAI_MODEL tests/test_database.py`, свыше 15 тестов).
- Файлы, которые также ставят `DB_PATH` перед созданием `Database()` (должны продолжать
  работать без единой правки, т.к. `configure()` не вызывается извне `__main__.py`):
  `tests/test_agent_tools_plan_rule_mutator.py`, `test_agent_tools_plugin.py`,
  `test_agent_tools_replan.py`, `test_agent_tools_schema_registry.py`,
  `test_agent_tools_session_reset.py`, `test_agent_tools_verify.py`, `test_db_handle.py`,
  `test_docker_runtime_config.py`, `test_exemplar_hindsight_extraction_structure.py`,
  `test_hindsight_approval_flow.py`, `test_hindsight_burst_buffer.py`,
  `test_hindsight_dream_worker.py`, `test_hindsight_event_journal.py`,
  `test_hindsight_finalize_worker_integration.py`, `test_hindsight_memory_report.py`,
  `test_hindsight_schema_registry.py`, `test_text_document_qa_anythingllm.py`.
- `tests/test_group_session_flow.py`, `tests/test_callback_authorization.py` — ставят
  `MAX_SESSIONS`/`OPENAI_MODEL` через `telegram_bot.py`/`openai_helper.py` пути, не через
  `Database` напрямую — проверить, что не завязаны на порядок чтения.
- `tests/test_openai_helper_tool_calls.py` (фикстура `_make_helper`, строки 440-484),
  `tests/test_hindsight_memory.py` (`make_helper`, строки 158-199), `tests/test_pricing.py`,
  `tests/test_skills_prompt_fragment.py`, `tests/test_stream_usage.py` — минимальные config
  для `OpenAIHelper`, на них проверяется список 19 сохраняемых `setdefault`.

## Команды проверки

```bash
# 0. rg/grep в этом окружении иногда искажают вывод (экранирование `|` в alias)
#    — для точечных грепов использовать python3 -c с re, не rg/grep напрямую.

# 1. Целевые файлы — до и после правок
~/.venvs/ctb/bin/python -m pytest -q tests/test_telegram_builder_config.py tests/test_database.py

# 2. Фикстуры с минимальным OpenAIHelper.config
~/.venvs/ctb/bin/python -m pytest -q \
  tests/test_openai_helper_tool_calls.py tests/test_hindsight_memory.py \
  tests/test_pricing.py tests/test_skills_prompt_fragment.py tests/test_stream_usage.py

# 3. Всё, что ставит DB_PATH/MAX_SESSIONS напрямую
~/.venvs/ctb/bin/python -m pytest -q \
  tests/test_agent_tools_plan_rule_mutator.py tests/test_agent_tools_plugin.py \
  tests/test_agent_tools_replan.py tests/test_agent_tools_schema_registry.py \
  tests/test_agent_tools_session_reset.py tests/test_agent_tools_verify.py \
  tests/test_db_handle.py tests/test_docker_runtime_config.py \
  tests/test_exemplar_hindsight_extraction_structure.py tests/test_hindsight_approval_flow.py \
  tests/test_hindsight_burst_buffer.py tests/test_hindsight_dream_worker.py \
  tests/test_hindsight_event_journal.py tests/test_hindsight_finalize_worker_integration.py \
  tests/test_hindsight_memory_report.py tests/test_hindsight_schema_registry.py \
  tests/test_text_document_qa_anythingllm.py tests/test_group_session_flow.py \
  tests/test_callback_authorization.py

# 4. Полный прогон без evals/ (testpaths = tests bot/tests из pytest.ini уже это обеспечивает)
~/.venvs/ctb/bin/python -m pytest -q
```

**Известный посторонний фейл**: на чистом HEAD (без единой правки из этого плана)
`tests/test_openai_helper_tool_calls.py::test_tool_execution_rejects_nested_chat_response_even_with_same_chat_lock`
падает и в изолированном прогоне файла, и в полном прогоне сьюта (`DID NOT RAISE
RuntimeError`) — воспроизведено дважды на неизменённом дереве. Это не связано с T17;
если он всё ещё падает после правок — это ожидаемо и не блокирует T17, но **новых**
падений быть не должно.

## Риски

Инструкция требует явно перечислить изменения поведения дефолтов — по плану их **не
должно быть**. Ниже — что проверено на отсутствие изменений и что осталось как есть
осознанно:

- **Нет изменения ни одного дефолтного значения.** `shared` — то же вычисление, просто
  один раз вместо двух; `env_bool` — побайтово та же идиома, что и сегодняшний инлайн
  `.lower()=='true'` (включая нелогичную деталь: мусорное значение даёт `False` даже для
  ключей с дефолтом `True` при отсутствии env — это сохраняется, не «исправляется»).
- **`parse_bool_env` не трогается** — значит `test_invalid_telegram_local_mode_rejected_before_polling`
  и подобные (строгий reject для `TELEGRAM_LOCAL_MODE`/`CHAT_RUN_VARIANT_B_ENABLED`/
  `SUMMARY_ENABLED`/`SESSION_LOG_OTEL_INSECURE`/`TELEGRAM_RICH_DRAFTS`) продолжают падать
  ровно как сегодня на некорректном значении.
- **`temperature`: убираемый `setdefault(0.7)` расходился с продакшен-дефолтом `1.0`.**
  Это не «исправление бага в проде» — значение `0.7` никогда не наблюдалось ни одним
  известным вызывающим кодом (см. таблицу C), значит удаление не меняет поведение ни одного
  реального пути, только убирает мёртвый неверный дефолт.
- **`MAX_SESSIONS=0`-расхождение между `telegram_bot.py` (без клампа) и
  `database.py._coerce_max_sessions_limit` (клампит в 1) — не устраняется этим планом.**
  Это отдельный, более рискованный поведенческий вопрос (какое значение правильное?), не
  входящий в T17 («один источник конфига», а не «унификация клампинга»). Если появится
  желание это исправить — отдельная задача с явным решением, что считать правильным
  поведением при `MAX_SESSIONS=0`.
- **`Database.configure()` — новый метод, но по умолчанию не меняет поведение**, пока
  `__main__.py` не начнёт его вызывать. После того как `__main__.py` его вызывает,
  `db_path`/`max_sessions`/`journal_mode`/`default_model` в продакшене перестают читать env
  внутри `database.py` и берут те же самые уже вычисленные значения — то есть значения не
  меняются, меняется только откуда их берёт `database.py` в момент, когда `openai_helper`
  ещё не создан (миграция при первом старте, где `default_model` берётся из `_configured`
  вместо повторного чтения `OPENAI_MODEL`).
- **Частота логирования warning про `SQLITE_JOURNAL_MODE`** для сконфигурированного пути
  меняется с «на каждое новое потоковое соединение» на «один раз при `configure()`» — это
  затрагивает только `journal_mode`, полученный через `configure()` (то есть только
  продакшен-путь после этой правки), а не env-путь, на котором стоят
  `test_invalid_journal_mode_falls_back_to_wal`/`test_valid_journal_mode_is_normalized`
  (они не вызывают `configure()`, значит не видят разницы). Указано явно как единственное
  наблюдаемое отличие в логах.
- **`plugin_config` не получает `shared`** — осознанное отклонение от буквального «`{**shared,
  ...}` для трёх конфигов» в тексте T17: у `plugin_config` нет пересечения ключей с `shared`
  (таблица A), а `plugin_manager.config.update(openai_config)` и так передаёт плагинам всё
  содержимое `openai_config`. Добавление `shared` в `plugin_config` было бы избыточным кодом
  без эффекта — решено не делать ради простоты.
- **`env_bool` — новая функция, а не расширение `parse_bool_env`** — осознанное отклонение
  от буквального «одна функция `env_bool`»: единая мягкая политика для всех 22 булевых
  меняла бы наблюдаемое поведение (см. выше «Дизайн»). Итог — две функции с разными,
  явно задокументированными контрактами, а не одна с двумя режимами по флагу (сделали бы,
  но `strict=True/False` в одном имени не даёт выигрыша перед двумя отдельными именами и
  хуже читается на 22 местах вызова).

## Критерии готовности

- В `bot/__main__.py` ровно одно вычисление на каждый из 14 ключей таблицы A; `openai_config`
  и `telegram_config` собраны через `{**shared, ...}`.
- 17 сырых `.lower() == 'true'` заменены на `env_bool(...)`; `parse_bool_env` и её 5
  вызовов не изменены.
- `bot/openai_helper.py` содержит 19 `setdefault` из раздела C, остальные 11 удалены.
- `bot/database.py` имеет `Database.configure(...)`; `__main__.py` вызывает его перед
  `db = Database()`; `_reset_singleton()` очищает `_configured`.
- `~/.venvs/ctb/bin/python -m pytest -q` (полный прогон, `testpaths = tests bot/tests` из
  `pytest.ini`) даёт тот же результат, что и до правок: тот же набор `passed`/`skipped`, и
  единственный уже существующий до правок фейл
  (`test_tool_execution_rejects_nested_chat_response_even_with_same_chat_lock`) — без новых
  падений.
- Ни одно значение по умолчанию, наблюдаемое извне (в `openai_config`, `telegram_config`,
  в поведении `Database()` без вызова `configure()`), не изменилось — единственное
  осознанное исключение: убранный неверный `temperature=0.7` внутри `openai_helper.py`,
  который не был достижим ни одним известным вызывающим кодом (см. «Риски»).

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer) ошибок не нашло. Побайтовое сравнение `env_bool` со старой
идиомой `os.environ.get(...).lower() == 'true'` дало полное совпадение на всех кейсах; все 16
вызовов `env_bool` и все 11 убранных `setdefault` проверены на риск `KeyError` в `bot/`, `tests/`,
`evals/` — риска нет. Приоритет `Database._configured` над env одинаков во всех точках чтения.

Единственные предупреждения касались не T17, а состояния дерева в момент ревью: параллельно
шла разработка T16 (удаление констант семейств моделей), и тесты, ссылавшиеся на `O_MODELS`/
`PERPLEXITY`, ещё не были обновлены. После завершения T16 полный прогон зелёный
(`1530 passed`), `ruff check bot tests bot/tests` чист. Также ревьюер отметил протухший
`__pycache__` (устранён удалением кэша байткода, код не менялся).

Известное и не устранённое расхождение (вне периметра): `MAX_SESSIONS=0` в `telegram_bot.py`/
`__main__.py` остаётся `0`, а `database.py` клампит в `1` — поведение не изменилось.
