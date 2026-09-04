# T21. Синхронизация документации с кодом — план (код не менялся)

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md:187-189` (Волна 5, T21):
«README/README.ru: убрать 8 несуществующих env, добавить недокументированные; `.env.example`:
убрать 5 мёртвых ключей, добавить отсутствующие; AGENTS.md: все `file:line` одним проходом по
итоговому коду; `.cli-proxy/.codebase_map/nodes/*.md` `Last reviewed`».

Это **план**, а не правка. Ни один файл, кроме этого, в рамках T21 не менялся. Все номера строк
ниже проверены построчным чтением рабочего дерева (незакоммиченные правки T01–T20 поверх HEAD
`af382fb`) 2026-09-04, а не переписаны из описаний задач T01–T20 или из самого AGENTS.md.

## Словарь терминов (простыми словами)

- **env-переменная (переменная окружения)** — настройка, которую программе передают не в коде,
  а через операционную систему (например, `TELEGRAM_BOT_TOKEN=...`). Читается в Python через
  `os.environ` / `os.getenv`.
- **`file:line`** — способ сослаться на конкретное место в коде: имя файла и номер строки.
  Пример: `bot/database.py:299` значит «строка 299 файла `bot/database.py`».
- **Постскриптум (postscript)** — раздел в конце каждого `T*.md`-плана, где после реального
  ревью и правок записано, что получилось на самом деле (могло отличаться от исходного плана).
- **Хук (hook)** — точка в коде, где плагину дают шанс вмешаться в обработку (например,
  «после ответа модели» или «перед отправкой запроса модели»). Подробнее — раздел «Hooks» в
  `AGENTS.md`.
- **Граф карты кодовой базы (codebase map)** — вспомогательные файлы в `.cli-proxy/.codebase_map/`,
  которые описывают, какие инструкции читать перед правкой той или иной части репозитория
  («узел»/`node` = один такой файл-инструкция на область кода).
- **MCP (Model Context Protocol)** — протокol, по которому бот подключается к внешним
  «серверам инструментов» (MCP-серверам) и вызывает их функции как обычные тулы.
- **Мёртвый ключ** — переменная окружения, которая упомянута в примере конфигурации
  (`.env.example`), но нигде в коде не читается — то есть ни на что не влияет.
- **Известные ограничения (Known limitations)** — раздел документации, где явно фиксируют
  «это не баг, который чиним прямо сейчас, а принятое ограничение текущей реализации», чтобы не
  путать пользователей и не заводить дублирующиеся баг-репорты.

---

## Раздел 1. Переменные окружения (README.md, README.ru.md, `.env.example`)

### 1.0 Метод проверки

Список переменных, которые код реально читает, я собрал скриптом (`python3` со своим regex),
который ищет:
- прямые вызовы `os.environ.get("X", ...)`, `os.getenv("X", ...)`, `os.environ["X"]`;
- вызовы обёрток: `env_bool`, `_parse_numeric_env`, `_parse_numeric_list_env`,
  `parse_semicolon_list_env`, `parse_model_list_env`, `first_model_env`,
  `parse_model_context_windows_env`, `parse_telegram_rich_mode_env`, `_numeric_env`,
  `_positive_int_env`, `_int_env`, `_env_flag`, `_env_int`, `_configured_path` — это локальные
  функции-обёртки над `os.environ`/`os.getenv` в `bot/__main__.py`, `bot/database.py`,
  `bot/telegram_bot.py`, `bot/openai_tool_handler.py`, `bot/plugins/agent_tools.py`,
  `bot/plugins/skills.py`, `bot/runtime_paths.py`.

Итог по всему `bot/**`: 144 уникальных имени переменных. Дальше — построчное сравнение этого
списка с текстом README.md, README.ru.md и `.env.example` (по вхождению точного имени в
backtick-обрамлении для README, по вхождению `ИМЯ=` для `.env.example`).

### 1.1 Калибровка ожиданий аудита: часть «удалить мёртвые» в README/README.ru уже выполнена

Аудит (написан до T01–T20) ожидал «8 несуществующих env» в README.md/README.ru.md. Проверка по
текущему рабочему дереву (после сегодняшних T06/T14/T17 и удаления `CHAT_RUN_VARIANT_B_ENABLED`)
показывает: **0** упомянутых в README.md/README.ru.md имён переменных, которых нет в коде.
Так что для README/README.ru часть «убрать» в T21 закрывать не нужно — актуальная работа только
«добавить недокументированные» (см. 1.3–1.4). Раздел «### Deprecated Variables» /
«### Устаревшие переменные» (README.md:485, README.ru.md:491) уже корректно перечисляет
`MONTHLY_USER_BUDGETS`/`MONTHLY_GUEST_BUDGET` как устаревшие псевдонимы — трогать не нужно.

### 1.2 `.env.example`: 5 мёртвых ключей — подтверждено, совпадает с оценкой аудита

Эти 5 имён встречаются только в `.env.example` и нигде в коде репозитория (проверено
`grep -rn` по `*.py`, `*.yml`, `*.yaml`, `Dockerfile`, `docker-compose.yml`, без учёта самого
`.env.example`):

| Строка | Текущее содержимое | Что сделать |
|---|---|---|
| `.env.example:125` | `# STABLE_DIFFUSION_TOKEN=XXX` | удалить строку |
| `.env.example:132` | `# ENABLE_LOCAL_FILE_SERVER=true` | удалить строку |
| `.env.example:133` | `# LOCAL_FILE_SERVER_HOST=localhost` | удалить строку |
| `.env.example:134` | `# LOCAL_FILE_SERVER_PORT=18080` | удалить строку |
| `.env.example:139` | `# ENABLE_MCP_SERVERS=true` | удалить строку |

Это остатки уже удалённых из кода функций (локальный файловый сервер, отдельный флаг включения
MCP, отдельный токен Stable Diffusion). Строки 132–134 идут одним блоком — удалять их можно
одним диапазоном `.env.example:132-134`. Строка 139 стоит сразу под заголовком
`# MCP Server Configuration` (строка 138) — при удалении строки 139 сам заголовок и следующие
за ним `MCP_SERVERS_ALLOWED_USERS`/`DEFAULT_MCP_SERVERS`/`MCP_REQUEST_TIMEOUT` трогать не нужно,
они актуальны.

### 1.3 Высокий приоритет — переменные из сегодняшнего диффа (T05, T08)

Эти три отсутствуют в README.md **и** README.ru.md (проверено по всему тексту, не только по
секции конфигурации); при этом `TERMINAL_APPROVAL_MODE` и `TERMINAL_COMMAND_POLICY` уже
закомментированы как пример в `.env.example:154-165`, а `TERMINAL_OUTPUT_BYTE_LIMIT` отсутствует
вообще везде:

| Переменная | Где читается в коде | README.md / README.ru.md | `.env.example` |
|---|---|---|---|
| `DB_OP_LOCK_TIMEOUT_SECONDS` | `bot/database.py:310` (`_numeric_env("DB_OP_LOCK_TIMEOUT_SECONDS", 15.0, float, minimum=0.0)`) — таймаут ожидания блокировки БД, введён сегодня в T08 | отсутствует в обоих | отсутствует |
| `TERMINAL_APPROVAL_MODE` | `bot/plugins/terminal.py:36` | отсутствует в обоих (в `.env.example:165` уже есть как пример) | уже есть (`.env.example:165`, закомментировано) |
| `TERMINAL_OUTPUT_BYTE_LIMIT` | `bot/plugins/terminal.py:27` (`max(1024, int(os.getenv("TERMINAL_OUTPUT_BYTE_LIMIT", str(8*1024))))`) | отсутствует в обоих | отсутствует |

Предложение по месту вставки: README.md — новая строка в таблице подраздела
`### Plugins And Storage` (README.md:363) или отдельный под-блок «Terminal plugin» рядом с
описанием terminal-плагина в `### Plugin-Specific Keys` (README.md:451); README.ru.md — то же
самое в `### Плагины и хранилище` (README.ru.md:371) / `### Ключи отдельных плагинов`
(README.ru.md:458). `TERMINAL_COMMAND_POLICY` уже есть в `.env.example`, но тоже отсутствует в
обоих README — стоит добавить туда же, раз редактируем этот блок (та же тема, тот же коммит по
смыслу).
`TERMINAL_OUTPUT_BYTE_LIMIT` в `.env.example` — добавить рядом с существующими
`TERMINAL_COMMAND_POLICY`/`TERMINAL_APPROVAL_MODE` (`.env.example:154-165`), с комментарием
«максимальный размер вывода команды в байтах, минимум 1024, по умолчанию 8192».

### 1.4 Остальной бэклог: не связано с сегодняшним диффом, но раз уж считали — фиксирую

Это не входит в «особое внимание» из постановки задачи и не вызвано правками T01–T20, но
раз в T21 явно указано «добавить недокументированные», ниже — полный список, чтобы не потерять.
Приоритет — низкий (backlog), можно делать отдельным проходом.

**Отсутствуют в README.md и README.ru.md (19 штук, без учёта трёх из 1.3):**

| Переменная | Файл, где читается |
|---|---|
| `CODEX_HOME` | `bot/plugins/skills.py` |
| `HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED` | `bot/plugins/agent_cron.py` |
| `IMAGE_RETENTION_DAYS` | `bot/__main__.py` |
| `MCP_SERVERS_CONFIG_PATH` | `bot/plugins/mcp_server.py` |
| `OUTPUT_MAX_TOKENS` | `bot/__main__.py` |
| `REPLY_INTENT_TIMEOUT_SECONDS` | `bot/__main__.py` |
| `RETENTION_CLEANUP_INTERVAL_SECONDS` | `bot/__main__.py` |
| `SESSION_NAME_TIMEOUT_SECONDS` | `bot/__main__.py` |
| `SUBAGENT_DEFAULT_TOOL_ROUNDS` | `bot/plugins/agent_tools.py` |
| `SUMMARY_DETERMINISTIC_MAX_CHARS` | `bot/__main__.py` |
| `SUMMARY_DETERMINISTIC_TAIL_CHARS` | `bot/__main__.py` |
| `SUMMARY_MAX_TOKENS` | `bot/__main__.py` |
| `SUMMARY_MIN_MESSAGES_BETWEEN_RUNS` | `bot/__main__.py` |
| `SUMMARY_MODEL` | `bot/__main__.py` |
| `SUMMARY_TARGET_KEEP_RATIO` | `bot/__main__.py` |
| `SUMMARY_TIMEOUT_SECONDS` | `bot/__main__.py` |
| `SYSTEMD_SERVICE_NAME` | `bot/telegram_bot.py` |
| `TOOL_CALL_EVENT_RETENTION_DAYS` | `bot/__main__.py` |
| `USAGE_RETENTION_DAYS` | `bot/__main__.py` |

**Дополнительно отсутствуют в `.env.example`, но уже есть хотя бы в README.md (значит, не
недокументированы вовсе, просто нет строки-примера — приоритет ниже, чем у предыдущей таблицы;
34 штуки):**

`ANYTHINGLLM_TIMEOUT`, `ENABLE_FUNCTIONS`, `FUNCTIONS_MAX_CONSECUTIVE_CALLS`, `GOOGLE_API_KEY`,
`GOOGLE_CSE_ID`, `JINA_API_KEY`, `MONTHLY_GUEST_BUDGET`, `MONTHLY_USER_BUDGETS`, `OPENAI_PROXY`,
`PLUGIN_MENU_PAGE_SIZE`, `PLUGIN_STRICT_VALIDATION`, `PROXY_WEB`, `SKILLS_SCRIPT_INTERIM_AFTER_SECONDS`,
`SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, `SQLITE_BUSY_TIMEOUT_MS`,
`SQLITE_JOURNAL_MODE`, `SQLITE_TIMEOUT`, `TELEGRAM_PROXY`, `TOOL_CALL_GLOBAL_PARALLELISM`,
`TOOL_CALL_PARALLELISM`, `TTS_RESPONSE_FORMAT`, `WHISPER_PROMPT` и 11 переменных, уже перечисленных
в таблице выше (`CODEX_HOME`, `DB_OP_LOCK_TIMEOUT_SECONDS`, `IMAGE_RETENTION_DAYS`,
`MCP_SERVERS_CONFIG_PATH`, `OUTPUT_MAX_TOKENS`, `RETENTION_CLEANUP_INTERVAL_SECONDS`,
`SUMMARY_MAX_TOKENS`, `SUMMARY_MIN_MESSAGES_BETWEEN_RUNS`, `SUMMARY_MODEL`,
`SUMMARY_TARGET_KEEP_RATIO`, `SUMMARY_TIMEOUT_SECONDS`, `SYSTEMD_SERVICE_NAME`,
`TERMINAL_OUTPUT_BYTE_LIMIT`, `TOOL_CALL_EVENT_RETENTION_DAYS`, `USAGE_RETENTION_DAYS`).

Не предлагаю переписывать `.env.example` целиком под этот список одним махом — он и так
211 строк; в отдельной задаче стоит решить, какие из них вообще уместны как «пример для
копирования» (часть — учётные данные внешних сервисов вроде `SPOTIFY_CLIENT_ID`, часть — тонкая
настройка вроде `SQLITE_BUSY_TIMEOUT_MS`, которая, возможно, не должна быть на виду у обычного
оператора).

---

## Раздел 2. `AGENTS.md` — проверка всех ссылок `file:line`

Ниже — только ссылки, которые оказались неточными (проверил вообще все `file:line` в файле;
часть совпала — например, `bot/plugin_manager.py:34`, `bot/plugin_manager.py:388` через один шаг,
`bot/validation.py:33`, `bot/telegram_bot.py:5337-5366` для внешнего вызова регистрации, но
об этом ниже отдельно, — их в таблицу не включаю).

### 2.1 Требует переписывания содержания, не только номера строк

**Абзац про «недостижимую Google-ветку»** (`AGENTS.md:74-78`):

> Google model tool specs use `{"function_declarations": specs}` while other models receive
> OpenAI-style `{"type": "function", "function": spec}` entries (`bot/plugin_manager.py:354-355`).
> Note: the Google branch is currently unreachable because `GOOGLE_MODELS` is an alias for
> `GOOGLE` (`bot/plugin_manager.py:20`), which is an empty tuple (`bot/model_constants.py:24`).

Это описывает код, которого больше нет. T16 удалил из `bot/model_constants.py` все константы
семейств моделей (`GOOGLE`, `GOOGLE_MODELS` и другие) — в файле (15 строк) остались только
конкретные имена моделей и `MAX_OUTPUT_TOKENS`. В `bot/plugin_manager.py` слова `GOOGLE` теперь
нет вообще (проверено `grep -n GOOGLE bot/plugin_manager.py` — 0 совпадений). Метод
`_format_specs_for_model` (`bot/plugin_manager.py:388`) больше не содержит ветвления и его
собственный docstring прямо говорит: «Currently only the OpenAI-compatible
`[{"type": "function", "function": {...}}]` form is produced; every model reaches this gateway
as an OpenAI-compatible alias (see `bot/model_constants.py`)».

Предлагаемая замена абзаца:

> Every model reaches this gateway as an OpenAI-compatible alias, so `_format_specs_for_model()`
> (`bot/plugin_manager.py:388`) unconditionally wraps every spec as
> `{"type": "function", "function": {...}}` — there is no Google-specific branch or
> `function_declarations` envelope in the current code (`bot/model_constants.py` keeps only
> individual model-name constants after T16 removed the model-family tuples).

**Дублирующаяся ссылка на `_reentry_tool_choice`** (`AGENTS.md:241` и `AGENTS.md:245`):
в файле `bot/openai_tool_handler.py` эта функция определена ровно один раз — на строке **908**
(`def _reentry_tool_choice(tools, *, times: int, ...)`). Вторая ссылка в тексте
(`AGENTS.md:245`, «`_reentry_tool_choice` (`bot/openai_tool_handler.py:908`)») уже верна.
Первая (`AGENTS.md:241`, `bot/openai_tool_handler.py:868`) — нет: там сейчас другой код
(`name=name,` внутри неродственной функции). Правка: заменить `:868` на `:908` в первой ссылке.

### 2.2 Только номер строки — таблица правок

Формат: было (что сейчас пишет AGENTS.md) → что там на самом деле сейчас → куда поправить.
«На самом деле сейчас» я не всегда цитирую дословно (это не нужно для самой правки), но по
каждой строке я лично открыл файл и убедился, что она не совпадает с описанием.

| # | Утверждение в AGENTS.md | Было (`file:line`) | Стало (`file:line`) |
|---|---|---|---|
| 1 | duplicate-функции: `PLUGIN_STRICT_VALIDATION=true` → raise, иначе — лог | `bot/plugin_manager.py:334` | `bot/plugin_manager.py:372-379` (сообщение — `:376`) |
| 2 | tool-аргументы: JSON-декодинг + валидация перед выполнением плагина | `bot/plugin_manager.py:428` | `bot/plugin_manager.py:532` (`json.loads(arguments)`) и `bot/plugin_manager.py:562` (`validate_function_args(...)`) |
| 3 | `<function_prefix>.<name>` нормализация имён спеков | `bot/plugin_manager.py:749` | `bot/plugin_manager.py:824-834` (сама подстановка — `:832`) |
| 4 | tool-вызовы выполняются через `asyncio.gather` | `bot/openai_tool_handler.py:181` | `bot/openai_tool_handler.py:218` |
| 5 | инъекция `chat_id`/`user_id` в аргументы | `bot/openai_tool_handler.py:1358-1360` | `bot/openai_tool_handler.py:1399-1400` (ветка с `request_context`), `:1404-1405` (ветка без него) |
| 6 | прямой результат помечается и обрывает re-entry (первая ссылка — «где помечается») | `bot/openai_tool_handler.py:416-420` | `bot/openai_tool_handler.py:1519-1525` |
| 7 | прямой результат обрывает re-entry (вторая ссылка — «где используется как short-circuit») | `bot/openai_tool_handler.py:1593-1596` | `bot/openai_tool_handler.py:1643-1646` |
| 8 | «empty/unset PLUGINS → все плагины, непустой → allow-list» | `bot/plugin_manager.py:223`, `bot/plugin_manager.py:234` | `bot/plugin_manager.py:65` (`self.enabled_plugins = ...`), `bot/plugin_manager.py:271` (проверка в `load_plugins`, сам `load_plugins` — `:260`) |
| 9 | `get_plugin_commands()` | `bot/plugin_manager.py:865` | `bot/plugin_manager.py:939` |
| 10 | `build_bot_commands()` | `bot/plugin_manager.py:892` | `bot/plugin_manager.py:966` |
| 11 | `_active_plugin_instances()` фильтрует только по отключённым плагинам пользователя | `bot/plugin_manager.py:993` | `bot/plugin_manager.py:1067` |
| 12 | `register_mcp_server` — вызываемый моделью тул | `bot/plugins/mcp_server.py:183` | `bot/plugins/mcp_server.py:219` (`get_spec`), имя тула — `:229` |
| 13 | инструменты подключённого MCP-сервера становятся полными спеками в рантайме | `bot/plugins/mcp_server.py:275-283` | `bot/plugins/mcp_server.py:321-327` |
| 14 | `hindsight_finalize_jobs` DDL в `register_schema()` | `bot/plugins/hindsight_memory.py:1181-1199` | `bot/plugins/hindsight_memory.py:1179` (def), DDL — `:1182-1201` (диапазон почти совпал, сдвиг на 2 строки — низкий приоритет) |
| 15 | `_consolidate_dream_document` | `bot/plugins/hindsight_memory.py:1906` | `bot/plugins/hindsight_memory.py:1904` (сдвиг на 2 строки — низкий приоритет) |
| 16 | `parse_consolidation_actions` / `apply_consolidation_actions` | `bot/plugins/hindsight_memory.py:238`, `:264` | `bot/plugins/hindsight_memory.py:236`, `:262` (сдвиг на 2 строки — низкий приоритет) |
| 17 | `hindsight_memory.on_before_chat_request` | `bot/plugins/hindsight_memory.py:2437` | `bot/plugins/hindsight_memory.py:2435` (сдвиг на 2 строки — низкий приоритет) |
| 18 | `burst_sweep` фоновая задача | `bot/plugins/hindsight_memory.py:825-845` | `bot/plugins/hindsight_memory.py:823-845` (`get_background_tasks` def — `:823`, сам `BackgroundTask(...)` — `:834-842`; диапазон почти совпал — низкий приоритет) |
| 19 | `auto_mode_priority` вызывается через `collect_fragments` | `bot/openai_helper.py:4272-4274` | `bot/openai_helper.py:4034-4036` (файл целиком — 4087 строк, старая ссылка указывает за пределы содержательного кода; сам метод `_build_auto_chat_mode_prompt` — `:4021`) |
| 20 | режим ограничивает `allowed_plugins` через `tools`, по умолчанию `['All']` | `bot/openai_helper.py:1315` | `bot/openai_helper.py:1155` (внутри `resolve_allowed_plugins`, def — `:1153`) |
| 21 | то же (вторая ссылка, повтор в разделе «Tool And Context Footprint») | `bot/openai_helper.py:1346-1347` | `bot/openai_helper.py:1186-1187` |
| 22 | `_deterministic_summary_text` | `bot/openai_helper.py:3718` | `bot/openai_helper.py:3483` |
| 23 | `_summarize_and_trim` | `bot/openai_helper.py:3769` (обе ссылки, в двух разных абзацах) | `bot/openai_helper.py:3534` |
| 24 | `_fallback_trim_with_summary` | `bot/openai_helper.py:3856` | `bot/openai_helper.py:3621` |
| 25 | `evaluate_command()` | `bot/command_policy.py:434` | `bot/command_policy.py:467` |
| 26 | `load_policy_from_env()` | `bot/command_policy.py:410` | `bot/command_policy.py:443` |
| 27 | Database — singleton с thread-local соединениями и operation `RLock` | `bot/database.py:73`, `bot/database.py:80` | `bot/database.py:211` (`__new__`, сам singleton-паттерн), `bot/database.py:222` (`_op_lock = threading.RLock()`); класс `Database` начинается на `:174`, thread-local (`_local = threading.local()`) — `:225` |
| 28 | новые соединения включают foreign keys, WAL по умолчанию, `busy_timeout` | `bot/database.py:121-128` | `bot/database.py:299` (foreign keys), `:300-304` (journal mode/WAL), `:305-306` (busy_timeout) |
| 29 | `conversation_context.context` — JSON вида `{"messages": [...]}`, не «голый» список | `bot/database.py:476` | `bot/database.py:680` (`default_context = '{"messages": []}'`) |
| 30 | `save_conversation_context()` пишет `message_count` как число сообщений с ролью `user` | `bot/database.py:787` | `bot/database.py:991` (сам `save_conversation_context` — def на `:977`) |
| 31 | `Database.ensure_session_name_async()` — короткий фолбэк | `bot/database.py:902` | `bot/database.py:1109` |
| 32 | инвалидный `filters` в конфиге плагина логируется и пропускается | `bot/telegram_bot.py:5238-5243` | `bot/telegram_bot.py:5221-5224` |
| 33 | сборка `ApplicationBuilder`, локальный режим, `base_url` | `bot/telegram_bot.py:6288-6303` | `bot/telegram_bot.py:6269-6285` (proxy — `:6287-6290`; сам `def run(self):` — `:6248`) |

### 2.3 Минорные (в пределах ±2–6 строк, можно не трогать в первую очередь)

- `stats_block` сборщик: AGENTS.md пишет `bot/telegram_bot.py:1290`, актуально —
  `bot/telegram_bot.py:1277-1281`.
- `settings_menu_buttons` сборщик: AGENTS.md пишет `bot/telegram_bot.py:1583-1587`, актуально —
  `bot/telegram_bot.py:1572-1576`.
- Регистрация обычных команд плагинов в цикле `post_init()`: AGENTS.md пишет
  `bot/telegram_bot.py:5337-5366` — сам вызов `self._authorized_command_handler(` сейчас на
  `:5343`, а весь цикл `for cmd in plugin_commands:` — `:5327-5345`; `post_init` начинается на
  `:5299`.
- `_ensure_session_name_with_llm`: AGENTS.md ссылается на `bot/openai_helper.py:718` — это
  строка внутри docstring метода; сам `async def _ensure_session_name_with_llm(` — на `:712`.
  Можно оставить как есть или поправить на `:712` заодно с остальными правками этого файла.

### 2.4 Содержательная неточность без сдвига номера строки

`NON_PLUGIN_MODULES` (`bot/plugin_manager.py:33-40`) — ссылка `bot/plugin_manager.py:34` на
саму строку с содержимым множества верна. Но текст AGENTS.md перечисляет «the base class plus
the framework modules `background.py`, `db_handle.py`, `hooks.py`» — то есть 4 позиции, а на
самом деле в множестве 5: `'__init__.py', 'plugin.py', 'background.py', 'db_handle.py', 'hooks.py'`.
Пропущен `__init__.py`. Добавить его в перечисление.

---

## Раздел 3. `docs/tutorial/09_менеджер_плагинов.md`

Строки 232–235 (проверено — актуальный текст):

```python
    # Формат зависит от модели: OpenAI или Google
    if model_to_use in GOOGLE_MODELS:
        return {"function_declarations": all_specs}
    return [{"type": "function", "function": s} for s in all_specs]
```

`GOOGLE_MODELS` в коде больше не существует (T16, см. раздел 2.1). Это учебный псевдокод —
переписать так, чтобы он совпадал с реальным `_format_specs_for_model`
(`bot/plugin_manager.py:388-397`):

```python
    # Формат единый для всех моделей — этот шлюз всегда работает по протоколу OpenAI
    return [{"type": "function", "function": s} for s in all_specs]
```

Заодно стоит убрать сопроводительный комментарий «Формат зависит от модели: OpenAI или Google»
(строка 232) как вводящий в заблуждение, и заменить его на «Формат единый для всех моделей».

---

## Раздел 4. Граф карты кодовой базы (`.cli-proxy/.codebase_map/`)

Правила обновления — `.cli-proxy/.codebase_map/INDEX.md` (прочитан целиком): шаг 4 требует
«After changes, update affected node metadata (`When to update`, `Last reviewed`)», шаг 5 —
«If node update fails, run targeted repair for that node». Явной команды/скрипта
«targeted repair» в `INDEX.md` не описано — похоже, это внешний инструмент карты, не файл в
репозитории. План ниже поэтому разделён на «что сделать штатным путём» и «что руками, если
инструмента нет под рукой».

### 4.1 Узлы, чьи файлы менялись сегодня (по `git diff HEAD --name-only`)

| Узел | `source_glob` | Затронутые сегодня файлы | Текущий `Last reviewed` |
|---|---|---|---|
| `nodes/bot.md` | `bot/**` | `bot/__main__.py`, `bot/database.py`, `bot/openai_helper.py`, `bot/openai_tool_handler.py`, `bot/plugin_manager.py`, `bot/model_constants.py`, `bot/command_policy.py`, `bot/telegram_bot.py`, `bot/plugins/*.py` (несколько), `bot/README_MCP.md`, `bot/tests/test_mcp_server.py`, `bot/usage_tracker.py`, `bot/utils.py` | `2026-08-14T23:10:49+03:00` |
| `nodes/tests.md` | `tests/**` | 18 файлов `tests/test_*.py` | `2026-08-01T22:59:18+03:00` |
| `nodes/requirements-txt.md` | `requirements.txt` | `requirements.txt` | `2026-08-01T22:15:45+03:00` |
| `nodes/dockerfile.md` | `Dockerfile` | `Dockerfile` | `2026-07-03T04:00:01+03:00` |
| `nodes/docker-compose-yml.md` | `docker-compose.yml` | `docker-compose.yml` | `2026-07-03T04:00:01+03:00` |

Для первых трёх узлов правка по существу небольшая: обновить `Last reviewed` на дату фактической
правки и, для `nodes/bot.md`, дополнить список «Source of truth» — туда не попал новый файл
`bot/telegram_stream.py` (выделен из `telegram_bot.py` в T12), хотя сам узел `bot.md` по
`source_glob: bot/**` уже покрывает его физически.

Файлы `.github/workflows/*`, `.dockerignore`, `.gitignore`, `README.md`, `README.ru.md`,
`AGENTS.md`, `evals/judge/turn_runner.py` менялись сегодня, но под `source_glob` ни одного узла
не попадают — это ожидаемо (в `INDEX.md` нет узлов уровня «корень репозитория» или «docs/»), в
T21 их можно не трогать.

### 4.2 Узел удалённого файла `IMG_3980.jpg` (`nodes/img-3980-jpg.md`)

Файл `IMG_3980.jpg` удалён в T04 (репозиторная гигиена). Узел `img-3980-jpg.md` описывает файл,
которого больше нет, и ссылается на `bot/telegram_bot.py:1775` как на потребителя — эту ссылку
T04 тоже убрал (thumbnail для inline-режима удалён вместе с файлом, см. постскриптум T04).
Узел упомянут в 4 местах карты:

- `INDEX.md` — строка со ссылкой `[IMG_3980.jpg](nodes/img-3980-jpg.md)` в разделе `## Nodes`.
- `graph.json` — запись узла (`"id": "node:img-3980-jpg"`, строки 40-44), ребро `index → node:img-3980-jpg`
  (строка 209 блока `"to": "node:img-3980-jpg"`), строка в `"tree"` (строка 288).
- `state.json` — запись статуса `"nodes/img-3980-jpg.md": {...}` (строка 209), пустой список
  файлов `"IMG_3980.jpg": []` (строка 302), плюс две строки в текстовых деревьях (`14`, `175`).
- `rules.yaml` — правило `update-nodeimg-3980-jpg` (строки 43-52), которое переоткрывает узел
  при любом изменении `IMG_3980.jpg`.

Рекомендация: если у карты есть штатный инструмент удаления/починки узла (то, что `INDEX.md`
называет «targeted repair»), использовать его — он для того и существует, чтобы синхронно
поправить все 4 места без риска рассинхронизировать `graph.json`/`state.json`/`rules.yaml`
между собой. Ручную правку этих трёх JSON/YAML-файлов вручную не предлагаю делать в T21: они
не являются просто документацией «для человека» (это входные данные какого-то внешнего
процесса), и я не проверял, что именно этот процесс ожидает при их ручном редактировании —
риск сломать формат выше пользы. Если штатного инструмента нет — задача на T21 в этой части:
явно завести её как отдельный шаг «удалить/починить узел `img-3980-jpg` инструментом карты
кодовой базы», а не редактировать JSON руками из этого плана.

Что можно сделать наверняка безопасно и без инструмента: обновить сам `nodes/img-3980-jpg.md`
(если решат его не удалять, а оставить как «архивную» запись) — добавить пометку, что файл
удалён в T04, и убрать мёртвую ссылку на `bot/telegram_bot.py:1775`. Но по сути правильнее
удалить узел целиком, раз файла нет — держать инструкцию про несуществующий файл смысла не имеет.

---

## Раздел 5. Известные ограничения из постскриптума T12

Постскриптум `docs/remediation_2026-09-04/T12-telegram-stream.md` фиксирует два «принятых
отклонения» (то есть не баги для немедленного фикса, а осознанные компромиссы):

1. Если самый первый чанк ответа длиннее 4096 символов, у хвостового сообщения не будет
   `reply_to_message_id` (в HEAD это работало только для этого узкого случая через
   delete+resend; сейчас поведение единообразное для всех случаев переполнения).
2. Ветка потоковой отправки «rich-drafts» (`sendRichMessage`/`sendRichMessageDraft`,
   `bot/telegram_rich.py`) — вне периметра ревью T12, отдельный тикет.

Предложение: завести один раздел **«### Known limitations»** под заголовком `## Telegram UX`
в README.md (после `### Inline Mode`, README.md:546-551, перед разделителем `---` на
строке 552) и зеркально **«### Известные ограничения»** под `## Telegram UX` в README.ru.md
(после `### Inline-режим`, README.ru.md:552-558). Выбрал именно это место, а не
`## Troubleshooting` (README.md:992) — раздел «Troubleshooting» построен как «симптом → что
сделать оператору», а тут не пользовательская ошибка конфигурации, а сознательно принятое
ограничение текущей реализации; и не отдельный docs-файл — ограничений пока всего два, они
относятся конкретно к Telegram-стримингу, у README.md уже есть подходящий раздел `## Telegram
UX` с описанием этого же механизма.

Каждый пункт — одна-две строки со ссылкой на `docs/remediation_2026-09-04/T12-telegram-stream.md`
за подробностями, без дублирования анализа из постскриптума. Пример формулировки (для README.md):

> ### Known limitations
> - If the very first streamed chunk exceeds 4096 characters, the tail message is sent without
>   `reply_to_message_id`. See `docs/remediation_2026-09-04/T12-telegram-stream.md` for details.
> - The rich-drafts streaming path (`sendRichMessage`/`sendRichMessageDraft`) has a known,
>   separately tracked issue — see `docs/remediation_2026-09-04/T12-telegram-stream.md`.

---

## Раздел 6. `bot/README_MCP.md` против постскриптума T02

Постскриптум `docs/remediation_2026-09-04/T02-mcp-v2.md` (раздел «Постскриптум после ревью
(2026-09-04)») зафиксировал исправление: `_mcp_call_result_to_dict`
(`bot/plugins/mcp_server.py:31-56`) раньше терял информацию, если MCP-инструмент вернул только
нетекстовый контент (картинку/аудио/ресурс) без `structured_content`; теперь такой ответ
помечается меткой `omitted_content`, а `result` получает читаемое сообщение вида
«`[N non-text content block(s) omitted: ...]`» вместо тихого `{"result": None}`.

Сегодняшняя правка `bot/README_MCP.md` (строки 284-286) это описывает не полностью:

```
- Результат вызова инструмента (`call_tool`) нормализуется перед возвратом модели: если
  сервер вернул структурированные данные, они передаются как есть; иначе возвращается
  текст ответа, либо сообщение об ошибке, если сервер сигнализировал сбой
```

Здесь перечислены только 2 из 3 веток: «структурированные данные» и «текст/ошибка». Третья —
случай, когда контент есть, но он не текстовый и не структурированный (например, только
картинка) — не упомянута, хотя именно её чинил T02. Предложенная правка (добавить как отдельный
пункт списка сразу после строки 286, перед строкой про завершение процесса):

```
- Если сервер вернул только нетекстовый контент (например, изображение) без структурированных
  данных, модель получает пометку о пропущенном контенте вместо пустого результата
```

---

## Критерии приёмки

1. В README.md и README.ru.md нет ни одной переменной окружения в тексте, которой нет в коде
   (проверяется тем же скриптом-сравнением, что использован в разделе 1.0 — 0 расхождений).
2. `DB_OP_LOCK_TIMEOUT_SECONDS`, `TERMINAL_APPROVAL_MODE`, `TERMINAL_COMMAND_POLICY`,
   `TERMINAL_OUTPUT_BYTE_LIMIT` документированы в README.md и README.ru.md.
3. `.env.example` не содержит 5 ключей из раздела 1.2; содержит `TERMINAL_OUTPUT_BYTE_LIMIT`.
4. Все 33 правки из таблицы раздела 2.2 внесены; абзац про Google-ветку переписан по тексту
   раздела 2.1; вторая ссылка на `_reentry_tool_choice` исправлена на `:908`;
   `NON_PLUGIN_MODULES` в прозе AGENTS.md перечисляет все 5 модулей, включая `__init__.py`.
5. `docs/tutorial/09_менеджер_плагинов.md:232-235` не содержит `GOOGLE_MODELS`.
6. Узел `nodes/img-3980-jpg.md` либо удалён инструментом карты кодовой базы (со всеми
   зависимыми упоминаниями в `INDEX.md`/`graph.json`/`state.json`/`rules.yaml`), либо явно
   помечен как архивный с указанием, что файл удалён в T04; `nodes/bot.md`,
   `nodes/tests.md`, `nodes/requirements-txt.md`, `nodes/dockerfile.md`,
   `nodes/docker-compose-yml.md` имеют обновлённый `Last reviewed`; `nodes/bot.md` упоминает
   `bot/telegram_stream.py` в «Source of truth».
7. В README.md и README.ru.md есть раздел «Known limitations»/«Известные ограничения» под
   `## Telegram UX` с двумя пунктами из раздела 5, каждый со ссылкой на T12-документ.
8. `bot/README_MCP.md:284-287` описывает все три ветки нормализации результата MCP-вызова,
   включая `omitted_content`.

## План проверки (после того как правки будут внесены кем-то другим — не в рамках T21)

1. Повторный запуск скрипта сравнения «env в коде vs env в README/README.ru/.env.example» из
   раздела 1.0 — должен показать 0 недокументированных и 0 мёртвых записей (кроме сознательно
   оставленного бэклога раздела 1.4, если решат не делать его в этом же проходе).
2. Построчная проверка всех `file:line` из AGENTS.md — тем же способом, каким проверялся этот
   план (открыть каждую ссылку и убедиться, что описанная конструкция там есть).
3. `git diff HEAD --name-only` — сверить, что для каждого затронутого сегодня узла карты
   кодовой базы `Last reviewed` обновлён.
4. Визуальный просмотр отрендеренного README.md/README.ru.md (правки только в документации —
   прогон тестов не требуется).
5. Если что-то из документации ссылается на поведение, для которого есть тест (например,
   `tests/test_no_hardcoded_plugin_refs.py` — но правки T21 её не касаются), достаточно
   убедиться, что тест не упоминает изменённый текст документации буквально (маловероятно, но
   быстро проверяется).

## Риски

- **Ручная правка `graph.json`/`state.json`/`rules.yaml`.** Формат этих файлов — вход для
  внешнего инструмента карты кодовой базы, не только «для человека». Правка руками без
  штатного инструмента рискует рассинхронизировать три файла между собой (например, поправить
  `graph.json`, но забыть про `rules.yaml`). Поэтому раздел 4.2 явно рекомендует использовать
  инструмент, а не редактировать эти файлы вручную из документации.
- **Дублирование источника правды по терминальным переменным.** `TERMINAL_COMMAND_POLICY` и
  `TERMINAL_APPROVAL_MODE` сейчас документированы только в `.env.example` (как закомментированный
  пример) — если добавлять их в README/README.ru, стоит убедиться, что формулировка не
  разойдётся с комментарием в `.env.example:156-164` (там уже есть точное описание формата
  JSON и поведения при невалидном значении — проще один раз сослаться, чем пересказывать заново).
- **Объём бэклога раздела 1.4.** 19+34 переменных — это много точечных правок; если делать
  одним PR, велик шанс внести опечатку в само имя переменной. Рекомендация — переносить их из
  таблиц этого документа копипастом, не перепечатывать вручную.
- **AGENTS.md устареет снова.** Без привычки актуализировать `file:line` при каждом крупном
  рефакторинге (как это уже произошло дважды — сначала T01–T20, теперь при написании этого
  плана обнаружены расхождения и в самих «уже поправленных» местах) список снова разъедется.
  Вне периметра T21, но стоит учитывать: возможно, часть ссылок стоит вообще убрать в пользу
  ссылок на имена функций без номера строки — обсуждать отдельно, не в этом плане.

---

## Таблица: файл → число правок → приоритет

| Файл | Число правок | Приоритет |
|---|---|---|
| `AGENTS.md` | 33 правки номеров строк (таблица 2.2) + 1 переписанный абзац (Google-ветка) + 1 правка дублирующей ссылки на `_reentry_tool_choice` + 1 правка перечисления `NON_PLUGIN_MODULES` = 36 | Высокий |
| `.env.example` | 5 удалить + 4 добавить (`DB_OP_LOCK_TIMEOUT_SECONDS`, `TERMINAL_OUTPUT_BYTE_LIMIT` — высокий приоритет; 34 из бэклога — низкий) = 9 высокоприоритетных + 34 бэклог | Высокий (5+4), низкий (34) |
| `README.md` | 3 добавить из сегодняшнего диффа + 1 новый раздел «Known limitations» (2 пункта) + 19 из бэклога | Высокий (4), низкий (19) |
| `README.ru.md` | 3 добавить из сегодняшнего диффа + 1 новый раздел «Известные ограничения» (2 пункта) + 19 из бэклога | Высокий (4), низкий (19) |
| `docs/tutorial/09_менеджер_плагинов.md` | 1 (переписать пример строк 232-235) | Средний |
| `bot/README_MCP.md` | 1 (дополнить список веток нормализации результата) | Средний |
| `.cli-proxy/.codebase_map/nodes/bot.md` | 2 (`Last reviewed` + добавить `bot/telegram_stream.py`) | Средний |
| `.cli-proxy/.codebase_map/nodes/tests.md` | 1 (`Last reviewed`) | Средний |
| `.cli-proxy/.codebase_map/nodes/requirements-txt.md` | 1 (`Last reviewed`) | Низкий |
| `.cli-proxy/.codebase_map/nodes/dockerfile.md` | 1 (`Last reviewed`) | Низкий |
| `.cli-proxy/.codebase_map/nodes/docker-compose-yml.md` | 1 (`Last reviewed`) | Низкий |
| `.cli-proxy/.codebase_map/nodes/img-3980-jpg.md` + `INDEX.md` + `graph.json` + `state.json` + `rules.yaml` | 1 задача (удаление/починка узла инструментом карты, не ручная правка) | Средний |


---

## Постскриптум после реализации

Дата выполнения: 2026-09-04. Ниже — что реально сделано по каждому разделу плана, что
исправлено сверх плана, что осознанно отложено и почему.

### 1. Переменные окружения

- `.env.example`: удалены все 5 мёртвых ключей (`STABLE_DIFFUSION_TOKEN`,
  `ENABLE_LOCAL_FILE_SERVER`, `LOCAL_FILE_SERVER_HOST`, `LOCAL_FILE_SERVER_PORT`,
  `ENABLE_MCP_SERVERS`); добавлены `DB_OP_LOCK_TIMEOUT_SECONDS` и
  `TERMINAL_OUTPUT_BYTE_LIMIT` с комментариями и дефолтами, проверенными по коду.
- `README.md` / `README.ru.md`: добавлена строка `DB_OP_LOCK_TIMEOUT_SECONDS` в таблицу
  тюнинга БД; добавлен новый раздел «Terminal Plugin» / «Терминальный плагин» с
  `TERMINAL_APPROVAL_MODE`, `TERMINAL_OUTPUT_BYTE_LIMIT`, `TERMINAL_COMMAND_POLICY` —
  все три обязательных ключа задачи задокументированы во всех трёх файлах.
- **Отложено осознанно**: низкоприоритетный бэклог — 19 переменных, отсутствующих в
  README/README.ru целиком (`CODEX_HOME`, `HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED`,
  `IMAGE_RETENTION_DAYS`, `MCP_SERVERS_CONFIG_PATH`, `OUTPUT_MAX_TOKENS`,
  `REPLY_INTENT_TIMEOUT_SECONDS`, `RETENTION_CLEANUP_INTERVAL_SECONDS`,
  `SESSION_NAME_TIMEOUT_SECONDS`, `SUBAGENT_DEFAULT_TOOL_ROUNDS`,
  `SUMMARY_DETERMINISTIC_MAX_CHARS`, `SUMMARY_DETERMINISTIC_TAIL_CHARS`,
  `SUMMARY_MAX_TOKENS`, `SUMMARY_MIN_MESSAGES_BETWEEN_RUNS`, `SUMMARY_MODEL`,
  `SUMMARY_TARGET_KEEP_RATIO`, `SUMMARY_TIMEOUT_SECONDS`, `SYSTEMD_SERVICE_NAME`,
  `TOOL_CALL_EVENT_RETENTION_DAYS`, `USAGE_RETENTION_DAYS`), плюс 34 переменные,
  отсутствующие только в `.env.example`. Причина отказа: план прямо разрешает добавлять в
  README/`.env.example` только переменные, лично перепроверенные по коду в рамках этой
  задачи; полная перепроверка всего бэклога — отдельная по объёму задача документирования,
  не входящая в обязательный минимум T21, и её смешивание с этой правкой увеличило бы риск
  ошибок без соответствующей проверки тестами.

### 2. `AGENTS.md`

Все 33 ссылки `file:line` из таблицы плана перепроверены по актуальному коду и исправлены
там, где номера строк сдвинулись; переписан абзац про Google-ветку (`GOOGLE_MODELS`/
`GOOGLE` удалены в T16 — теперь `_format_specs_for_model()` безусловно оборачивает каждый
спек в формат OpenAI); исправлены обе ссылки на `_reentry_tool_choice`; в перечисление
`NON_PLUGIN_MODULES` добавлено пропущенное `__init__.py`.

**Сверх плана** — при систематической регэксп-выборке всех `path.py:line` из файла и
проверке каждой ссылки (а не только перечисленных в таблице плана) найдено и исправлено ещё
5 расхождений, которых не было в таблице плана:

1. `bot/__main__.py:183-186` (заявлено как проверка обязательных env-переменных) на деле
   указывало на валидатор схемы URL `TELEGRAM_BASE_URL`. Исправлено на `:199-203`.
2. `bot/__main__.py:349-362` (заявлено как конструирование PluginManager/Database/
   OpenAIHelper/ChatGPTTelegramBot) указывало на словарь конфигурации цен. Исправлено на
   `:366-385`.
3. `bot/openai_helper.py:305-306` (заявлено как конструирование `ChatModesRegistry` +
   вызов `validate_tools()`) указывало на строки таймаута сводки. Исправлено на `:292-293`.
4. `bot/openai_tool_handler.py:930-931` (заявлено как код принудительного сужения набора
   инструментов до tool доставки на «обычном» пути) указывало на не относящиеся к делу
   строки сигнатуры функции. В файле есть два похожих по форме блока сужения — один внутри
   `_retry_missing_delivery_tool` (обязательный путь доставки), другой внутри
   `handle_function_call` (обычный путь, который и описывает это предложение). Исправлено
   на `:1665-1666`.
5. `bot/plugin_manager.py:927` (заявлено как место, где обрезается ведущий `/` и
   отклоняются пробелы в имени команды плагина) указывало на не связанный код поиска
   `get_plugin`. Реальное место — внутри `_validate_and_normalize_command` (def на `:983`):
   `command = command[1:]` на `:1002`, проверка пробела на `:1004`. Исправлено на
   `:1002-1004`.

**Одна правка плана отклонена как ошибочная**: пункт таблицы плана про
`hindsight_finalize_jobs` предлагал сдвинуть диапазон DDL `bot/plugins/hindsight_memory.py:1181-1199`
на `:1182-1201`. Прямая проверка тройных кавычек DDL-блока показала, что существующий в
AGENTS.md диапазон `:1181-1199` уже верен — предложенный планом сдвиг был бы ошибкой.
Диапазон оставлен без изменений; добавлено лишь уточнение, что сам `register_schema()`
определён на `:1179`.

### 3. `docs/tutorial/09_менеджер_плагинов.md`

Пример с `GOOGLE_MODELS`/`function_declarations` заменён на актуальный код — единый путь
`{"type": "function", "function": s}` без ветвления по модели.

### 4. `bot/README_MCP.md`

Добавлено описание третьей ветки нормализации `_mcp_call_result_to_dict`
(`bot/plugins/mcp_server.py:31-56`): если сервер вернул только нетекстовый контент
(например, изображение) без структурированных данных, модель получает пометку о
пропущенном контенте (`omitted_content`) вместо пустого результата.

### 5. «Известные ограничения» / «Known limitations»

Разделы добавлены в README.md и README.ru.md с двумя пунктами T12 (потеря
`reply_to_message_id` при первом чанке длиннее 4096 символов; отдельно отслеживаемая
проблема стриминга rich-drafts), оба со ссылкой на
`docs/remediation_2026-09-04/T12-telegram-stream.md`.

### 6. Граф карты кода (`.cli-proxy/.codebase_map/`)

Инструмент `update-node`/`repair` в окружении отсутствует: проверено через `which` по
нескольким кандидатным именам и через обход дерева `.cli-proxy/` — найдены только
файлы-данные/отчёты, ни одного исполняемого скрипта с такой функцией. Поэтому правки
сделаны только вручную в файлах узлов (без изменения `graph.json`/`state.json`/
`rules.yaml`), как и разрешает план в этом случае:

- `nodes/bot.md`: `Last reviewed` обновлён на `2026-09-04T22:44:42+03:00`; в список
  «Source of truth» добавлен `bot/telegram_stream.py` (новый файл, выделенный из
  `telegram_bot.py` в T12, ранее отсутствовавший в списке).
- `nodes/tests.md`, `nodes/requirements-txt.md`, `nodes/dockerfile.md`,
  `nodes/docker-compose-yml.md`: только `Last reviewed` обновлён на
  `2026-09-04T22:44:42+03:00`, содержимое без изменений (правок в соответствующих файлах
  не было, план требовал только актуализацию отметки просмотра).
- `nodes/img-3980-jpg.md`: файл `IMG_3980.jpg` удалён из репозитория, и ссылка на него в
  `bot/telegram_bot.py` тоже удалена (грепом подтверждено отсутствие `IMG_3980`,
  `thumb_url`, `thumbnail_url` в файле — `InlineQueryResultArticle` больше не использует
  миниатюру). Узел помечен как устаревший (заголовок и `## Purpose` явно говорят «STALE —
  asset deleted»), раздел «Source of truth» очищен, добавлена инструкция не полагаться на
  этот узел и не восстанавливать актив/ссылку по нему.
  **Отложено**: фактическое удаление узла из графа (`graph.json`/`state.json`) не
  выполнено — по правилам плана и `INDEX.md` ручная правка этих файлов запрещена без
  штатного инструмента карты, а такого инструмента в окружении нет. Требуется удалить узел
  штатным инструментом, когда/если он появится.

### Файлы, изменённые в рамках T21

- `.env.example`
- `README.md`
- `README.ru.md`
- `AGENTS.md`
- `docs/tutorial/09_менеджер_плагинов.md`
- `bot/README_MCP.md`
- `.cli-proxy/.codebase_map/nodes/bot.md`
- `.cli-proxy/.codebase_map/nodes/tests.md`
- `.cli-proxy/.codebase_map/nodes/requirements-txt.md`
- `.cli-proxy/.codebase_map/nodes/dockerfile.md`
- `.cli-proxy/.codebase_map/nodes/docker-compose-yml.md`
- `.cli-proxy/.codebase_map/nodes/img-3980-jpg.md`
- `docs/remediation_2026-09-04/T21-docs-sync.md` (этот постскриптум)

Исходный код (`bot/**.py`, `tests/**.py`, `evals/**`) не менялся.

---

## Постскриптум после ревью

Ревьюер (Sonnet, persona reviewer, read-only) прошёл по всем изменённым документам и
сверил каждую ссылку `file:line` с кодом.

### Ошибки — исправлены

- **7 ссылок в `AGENTS.md` съехали** относительно кода. Причина установлена: T19
  (публичный session-API у `OpenAIHelper`) выполнялся параллельно с T21 и добавил ~78 строк
  в `bot/openai_helper.py` и убрал ~37 строк из `bot/telegram_bot.py` уже после того, как
  T21 сверил номера. Ревьюер прямо рекомендовал не переделывать T21, а перепроверить эти
  два файла после завершения T19 — что и сделано.

  Итоговая правка (18 ссылок, включая накопившиеся сдвиги на ±1 в `agent_tools.py`):

  | было | стало | что это |
  | --- | --- | --- |
  | `telegram_bot.py:6270-6283` | `:6232-6245` | `ApplicationBuilder` (concurrent updates, local mode, base URL) |
  | `agent_tools.py:346` (×2) | `:347` | `on_before_chat_request` |
  | `openai_helper.py:4034-4036` | `:4111-4113` | `collect_fragments("auto_mode_priority", ...)` |
  | `agent_tools.py:2547` | `:2548` | `_manage_plan_tasks` |
  | `agent_tools.py:2599-2602` | `:2600-2603` | переходы статуса при `action=add` |
  | `agent_tools.py:2657-2660` | `:2658-2661` | переходы статуса при `action=update` |
  | `agent_tools.py:2080` | `:2081` | `_apply_plan_runtime_effects` |
  | `agent_tools.py:2103` | `:2104` | `_record_tool_outcome` |
  | `agent_tools.py:38` | `:39` | `describe_plan_lifecycle` |
  | `agent_tools.py:2349` | `:2350` | `_validate_plan_tasks` |
  | `agent_tools.py:26` | `:27` | `TASK_STATUSES` |
  | `agent_tools.py:509` | `:510` | `enum` в схеме `manage_plan_tasks` |
  | `openai_helper.py:3534` (×2) | `:3612` | `_summarize_and_trim` |
  | `openai_helper.py:3621` | `:3699` | `_fallback_trim_with_summary` |
  | `openai_helper.py:3483` | `:3561` | `_deterministic_summary_text` |
  | `telegram_bot.py:5329-5345` | `:5293-5311` | регистрация плагинных команд в `post_init` |
  | `telegram_bot.py:5221-5224` | `:5184-5188` | «Invalid filter … skipped» |

  Проверка автоматизирована: скрипт разбирает все 76 ссылок `file:line` в `AGENTS.md`,
  открывает целевую строку и печатает её содержимое. После правки подозрительных
  (несуществующий файл, выход за пределы, пустая строка) — 0.

### Предупреждения — исправлены

- **`bot/README_MCP.md:284-286` неверно описывал порядок проверок** в
  `_mcp_call_result_to_dict`. В тексте было «сначала структурированные данные, иначе текст,
  либо ошибка», а в коде (`bot/plugins/mcp_server.py:37-44`) первым проверяется `is_error`:
  при ошибке возвращается только `{"error": ...}`, и до `structured_content` дело не доходит.
  Формулировка переписана по факту кода, добавлена ссылка на функцию.
- **README недооценивал серьёзность ограничения rich-drafts.** Было: «есть известная
  проблема, вынесенная в отдельный тикет». Проверено по коду: это **дефолтный** путь для
  приватных чатов (`TELEGRAM_RICH_MESSAGES=auto` + `TELEGRAM_RICH_DRAFTS=true`,
  `bot/__main__.py:214-215`, `bot/telegram_bot.py:653-659`), у него другой лимит (32768
  байт, `MAX_RICH_MARKDOWN_BYTES`), и при сбое отправки режим `required` пробрасывает
  исключение и обрывает ответ, а `auto` поглощает ошибку и переключается на легаси, но
  перепубликует текст только если черновик ещё не был отправлен
  (`bot/telegram_bot.py:4310-4347`, ветка `if sent_message is None`) — иначе у пользователя
  остаётся устаревший черновик. Всё это выписано явно в «Known limitations» /
  «Известные ограничения» обоих README, включая замечание о слабом покрытии тестами
  (большинство тестов стриминга настроены на легаси-режим).
- **Недокументированные переменные окружения.** В `.env.example` добавлены
  `SUMMARY_ENABLED` (проверено: `false` не отключает компактизацию, а переводит её на
  детерминированный fallback — `bot/openai_helper.py:3627-3628` возвращает `False`, и
  вызывающий применяет head-preserve-обрезку), `TELEGRAM_RICH_DRAFTS` и парная к ней
  `TELEGRAM_RICH_MESSAGES` (значения `auto | required | off`,
  `bot/__main__.py:54-68`).

### Добавлено сверх ревью

T19 завершился уже после того, как T21 сверил документацию, поэтому его публичный API
в `AGENTS.md` отсутствовал. Добавлен раздел **«Helper Session API»**: пять методов
(`history_snapshot`, `load_session`, `replace_system_message`, `evict`, `chat_state_scope`,
`bot/openai_helper.py:2967-3044`), явное предупреждение, что `history_snapshot` отдаёт
живой список кэша, а не копию, и упоминание сторожа `tests/test_no_private_helper_access.py`
в списке целевых тестов.

### Замечания без действия

- 8 ссылок в `bot/plugins/agent_tools.py` были смещены ровно на одну строку — источник тот
  же класс проблемы (документ правился, пока код менялся), учтено в таблице выше.
- Удаление устаревшего узла `img-3980-jpg` из графа карты кода по-прежнему требует штатного
  инструмента и остаётся отложенным.

**Файлы, изменённые постскриптумом:** `AGENTS.md`, `README.md`, `README.ru.md`,
`.env.example`, `bot/README_MCP.md`. Код не менялся.
