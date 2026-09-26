# T12 — Копипаста (волна 7): план

Базовая ревизия: `08bc457` (после T02–T11; T11 на ревью, может слегка задеть
state-код `bot/openai_helper.py`). Обнаружение дублей: скользящее окно 6 строк,
нормализация пробелов, хеш SHA1, слияние соседних окон в непрерывные блоки
(`/tmp/dedup_scan.py` → `/tmp/dedup_report.txt`, `/tmp/merge_dedup.py` →
`/tmp/merged_dedup.txt`; 609 групп дублей, 208 склеенных блоков), затем ручная
проверка каждого блока по коду и кросс-сверка с подсказками C1–C15 из
мастер-плана (`docs/improvement_2026-09-25/00-master-plan.md:397-416`).

Правило поведения: **ничего не меняется наблюдаемо**. Шаблонные повторы
(SQL/JSON-схемы, импорты, CSS/JS-литералы) не трогаем. Для кластеров без теста —
сначала пишем тест, затем выносим (правило мастер-плана).

Порядок: **T12a — сначала и последовательно** (создаёт файлы-помощники).
Затем параллельно **T12b / T12c / T12d** со строго непересекающимся владением
файлами. Треки **не** редактируют файлы, созданные/изменённые в T12a (только
импортируют из них).

---

## T12a — общие помощники (выполняется первым, последовательно)

### A1. `bot/env_utils.py` (новый файл) — `env_bool`, `parse_kv_list`

- `env_bool(name: str, default: bool) -> bool` — переносится из
  `bot/__main__.py:36-52` (канонический вариант: unset → `default`; иначе
  `True` только если значение точно `'true'` (без учёта регистра), иначе
  `False`). `bot/__main__.py` заменяет свою копию на `from .env_utils import
  env_bool` (единственная точка правки в `__main__.py`, принадлежит T12d).
  **Не трогать** `parse_bool_env` (`bot/__main__.py:24-33`, кидает исключение
  на невалидном значении) — другая семантика, не сливать.
  **Не трогать** `bot/plugins/mcp_server.py:76` (`MCP_ALLOW_PRIVATE_HOSTS`,
  truthy-множество `{"1","true","yes","on"}`) — другая семантика.
- Дубли `env_bool`-эквивалентной логики (`os.environ.get(NAME,
  'true'|'false').lower() == 'true'`) в `bot/plugins/hindsight_memory.py:538,
  539, 544, 547, 552, 557, 564` — заменить на `env_bool(NAME, default)`
  (T12d, файл `hindsight_memory.py`).
- `parse_kv_list(raw: str, *, value_parser: Callable[[str], T], warn_prefix:
  str) -> dict[str, T]` — общая форма цикла: split по `,` → partition по `=`
  → strip → лог+skip на некорректной записи → `value_parser(value)`.
  Источники с идентичной формой (но разным `value_parser`):
  - `bot/pricing.py:37-52` (`load_model_token_prices`; парсер:
    `value.partition(':')` → пара `float`).
  - `bot/__main__.py:138-154` (`parse_model_context_windows_env`; парсер:
    `int(value)` + проверка на положительность).
  Оба места заменяют локальный цикл на вызов `parse_kv_list` с их
  собственным `value_parser`/сообщением об ошибке; сигнатуры
  `load_model_token_prices`/`parse_model_context_windows_env` не меняются
  (внешнее поведение и тексты логов сохранить).
  **Не найдено** третьего парсера такого вида в `bot/database.py` (в мастер-
  плане database упомянута из-за `env_bool`-подобных числовых геттеров, см.
  ниже, а не из-за `parse_kv_list`).
- Тесты (новый файл): `tests/test_env_utils.py` —
  `env_bool` (unset/`'true'`/`'True'`/`'false'`/мусор), `parse_kv_list`
  (валидная пара, невалидная запись пропускается+лог, дубликат ключа
  перезаписывает предыдущий — как в исходных циклах).
  Существующие `tests/test_pricing.py` и тесты `__main__.py`-парсинга должны
  остаться зелёными без изменений (поведение не меняется).

### A2. `bot/chat_response_utils.py` — `leading_system_count`

- `leading_system_count(messages: list[dict]) -> int` — извлекается из
  идентичного цикла, встречающегося дважды в `bot/openai_helper.py`:
  `_summarize_and_trim` (цикл `head_end = 0; for m in conv: if
  isinstance(m, dict) and m.get('role') == 'system': head_end += 1; else:
  break`, ок. строка 3767 внутри функции с def на `bot/openai_helper.py:3734`)
  и `_fallback_trim_with_summary` (тот же цикл, ок. строка 3850, def на
  `bot/openai_helper.py:3821`). Это единственные два вхождения в репозитории
  (проверено регэкспом по `bot/**/*.py`).
- Тест: `tests/test_chat_response_utils.py` (создать, если файла нет — иначе
  добавить кейсы) — пустой список, список без system, список с N ведущими
  system-сообщениями + не-system дальше, system-сообщение после не-system
  (не считается).
- `bot/openai_helper.py` заменяет оба цикла на
  `leading_system_count(conv)` — правка входит в T12c (файл `openai_helper.py`
  ему принадлежит), но сама функция живёт в T12a-файле и не редактируется
  повторно другими треками.

### A3. `bot/utils.py` — `parse_model_choices`

- `parse_model_choices(raw: str | list[str] | None, default_model: str) ->
  list[str]` — общая форма: если `raw` — строка, сплит по `,`+strip+фильтр
  пустых; если список — strip каждого элемента; затем `default_model`
  добавляется в начало, если его там нет (дедуп с сохранением порядка).
  Источники с этой формой:
  - `bot/openai_helper.py:get_model_choices()` (строки 4086-4096) —
    канонический вид.
  - `bot/telegram_bot.py:_configured_openai_models()` (5934-5949, дублирующий
    fallback-блок — 5939-5949).
  - `bot/plugins/agent_tools.py:_model_choices_for_helper()` (117-132,
    fallback-блок 122-132).
  Три места заменяют свой fallback-блок на вызов `parse_model_choices`;
  правки в `openai_helper.py` (T12c), `telegram_bot.py` (T12b),
  `agent_tools.py` (T12d) — сама функция только в `utils.py` (T12a).
- Тест: `tests/test_utils_parse_model_choices.py` (новый, маленький) —
  `raw=None` → `[default]`; `raw="a,b, a"` → `[default, 'a', 'b']` (без
  дублей); `raw=['a','b']` → аналогично; `default` уже в списке → не
  дублируется.

### A4. `bot/chat_response_utils.py` — `finalize_chat_answer`

Самый крупный подтверждённый дубль T12c: хвост `bot/chat_run.py:
ChatRun.run_non_stream` (строки 200-254, сборка ответа: нумерация choices
при `n_choices>1`, запись в историю, footer использования токенов, footer
использованных плагинов) практически побайтово совпадает с
`bot/openai_helper.py:_interpret_image_text_response` (2514-2550, def на
2513). Различия только косметические (кавычки) плюс: `chat_run.py`
дополнительно пишет `usage_split` и `AIRunEnd`-событие после — это остаётся
на стороне вызывающего кода, не в помощнике.

- `async def finalize_chat_answer(helper, chat_id, response, *,
  plugins_used=(), token_accumulator=None, session_id=None) -> tuple[str,
  int]` — строит `answer`/`total_tokens`, пишет `assistant`-сообщение в
  историю через `helper._add_to_history(...)`, добавляет usage/plugins
  footer через `helper.config`, `helper.plugin_manager`,
  `localized_text`. Возвращает `(answer, total_tokens)`; вызывающий код сам
  решает, что делать дальше (usage-split, событие AIRunEnd, ретраи).
- Тест: `tests/test_chat_response_utils.py` (тот же файл, что A2) —
  однократный choice без usage/plugins, `n_choices>1` с нумерацией,
  `show_usage=True` с полным/неполным usage, `show_plugins_used=True` с/без
  `show_usage`.
- Потребители (обе правки — в T12c): `chat_run.py:run_non_stream` (заменяет
  200-254 на вызов + свои 2 строки usage_split/AIRunEnd) и
  `openai_helper.py:_interpret_image_text_response` (заменяет тело 2514-2550
  на вызов + `return`).

**Владение T12a:** `bot/env_utils.py` (новый), плюс новые/расширенные
записи в `bot/chat_response_utils.py` и `bot/utils.py` (существующие файлы —
T12a трогает только добавляемые функции, ничего больше в них не меняет).
Треки T12b/c/d **не** редактируют `bot/env_utils.py`,
`bot/chat_response_utils.py`, `bot/utils.py` в части, добавленной T12a —
только импортируют.

---

## T12b — `bot/telegram_bot.py` (только этот файл)

### B1 (C4). Markdown → plain fallback

Точный дубль (chunk-разбиение + `parse_mode=MARKDOWN` + `except BadRequest:`
plain fallback):
- `bot/telegram_bot.py:1054-1067` (вызывающий код `interpret_image`)
- `bot/telegram_bot.py:3003-3016` (вызывающий код `interpret_images`)

Извлечь приватный метод `async def _send_markdown_or_plain(self, update,
context, text: str) -> None` (или похожее имя, без коллизии с уже
существующими `try_send_rich_markdown_response`/`send_rich_markdown` из
`bot/utils.py`, которые решают другую задачу — не путать с ними). Третье,
похожее, но структурно иное место (vision-stream, 3259-3284, использует
`render_markdown_message_entities`+entities) **не трогать** — 3-уровневый
вариант, не идентичен первым двум.
Тест: покрыто `tests/test_telegram_streaming.py` /
`tests/test_callback_authorization.py` косвенно; отдельного юнит-теста на
именно этот fallback нет — написать
`tests/test_telegram_markdown_fallback.py` с моком `BadRequest`, проверяющим
оба call site до и после рефактора дают одинаковый вызов `reply_text`.

### B2 (C5). `on_session_reset` / `on_user_message` dispatch

- `on_session_reset`: 6 мест — `bot/telegram_bot.py:707-716` (вариант с
  `getattr`+`reason="final_delivery"`), 2751-2760, 2944-2953, 3137-3146,
  4116-4125, 4852-4861 (эти 5 — `reason="request_start"`).
- `on_user_message`: 4 места — 2930-2943, 3123-3136, 4095-4108, 4696-4711.

Извлечь `async def _dispatch_session_reset(self, chat_id, user_id, *,
reason: str) -> None` и `async def _dispatch_user_message(self, chat_id,
user_id, text, ...) -> None` (сигнатуру согласовать по факту общих
аргументов на всех 4 сайтах — проверить перед вырезкой, что аргументы
действительно идентичны, а не только похожи). Метод с `getattr`-вариантом
(707-716) — либо привести к общей сигнатуре, либо оставить с пометкой, если
`getattr` там неспроста (защита на случай отсутствия dispatcher на раннем
этапе жизни объекта).
Тест: `tests/test_agent_tools_session_reset.py` уже гоняет
`on_session_before_delete`/сброс через hooks — расширить или добавить
`tests/test_telegram_session_reset_dispatch.py`, проверяющий, что каждый из
6/4 call site реально зовёт dispatcher с ожидаемым `reason`.

### B3 (C6). Конструирование busy-статуса

`BusyStatusMessage(...)` с одинаковым набором аргументов
(`plan_provider, plan_interval = self._build_plan_status_provider(chat_id,
user_id); BusyStatusMessage(update, context,
localized_text("busy_status_preparing", ...), config=self.config,
plan_provider=plan_provider, interval=plan_interval)`) дублируется на
`bot/telegram_bot.py:3026-3034`, `3294-3302`, `4479-4487`.
Извлечь `def _build_busy_status(self, update, context, chat_id, user_id) ->
BusyStatusMessage`. Тест: `tests/test_pii_safe_logging.py`,
`tests/test_telegram_streaming.py` уже используют `BusyStatusMessage` —
добавить прямую проверку, что все 3 сайта после рефактора возвращают
одинаково сконструированный объект (сравнить атрибуты, а не просто отсутствие
исключения).

### B4 (C7). Отправка rich markdown

`send_rich_markdown(...)` + проверка `rich_markdown_fits` дублируются на
`bot/telegram_bot.py:4282-4295` и `4349-4362`, внутри одного большого метода
стриминга (~4130-4400+, rich draft vs legacy fallback). Извлечь приватный
helper внутри этого же метода/класса, например `async
def _send_rich_markdown_if_fits(self, ...)`. **Осторожно**: метод сложный —
перед вырезкой перечитать полный диапазон 4130-4400 (не только два
найденных блока), чтобы не потерять различия в окружающем контроле потока
между "rich draft" и "legacy" ветками. Тест: `tests/test_telegram_rich.py`
уже покрывает `send_rich_markdown`/`rich_markdown_fits` изолированно —
добавить кейс, что оба вызывающих места (rich draft / legacy) используют
общий путь.

### B5 (C13, telegram_bot-часть). Разрешение "кто отправил"

`check_allowed_and_within_budget` (`bot/telegram_bot.py:4973-5000`, дубль-ядро
4982-4991) повторяет логику, для которой уже есть канонический помощник
`utils._budget_user_and_name` (`bot/utils.py:814-823`). Заменить дублирующий
блок вызовом `_budget_user_and_name(...)` (импорт уже есть в `utils.py`,
`is_allowed`/`get_remaining_budget_async` там же уже используют его как
пример — `bot/utils.py:825-843`). Никаких новых файлов не создаётся, это
чистая замена внутри `telegram_bot.py`.
Тест: `tests/test_usage_budget.py` — расширить проверкой, что
`check_allowed_and_within_budget` даёт тот же результат, что и
`is_allowed`/`get_remaining_budget_async` для одинаковых входов (regression
на конкретно этот call site).

### B6 (C11/C14). Использование помощников T12a

Заменить локальные вхождения после того, как T12a готов:
- Любые *самостоятельные* повторы `env_bool`-формы внутри `telegram_bot.py`
  (если найдутся при вырезке — специально не искал сверх уже перечисленного,
  т.к. явных совпадений в `telegram_bot.py` не обнаружено при сканировании;
  если найдутся — заменить на `from .env_utils import env_bool`).
- `_configured_openai_models()` (5934-5949) — заменить fallback-блок на
  `parse_model_choices` (см. A3).

### B7 (C15, telegram_bot-часть)

**Plugin command reply** — validation-блок (получить `cmd` через
`self._get_plugin_command(...)`, если нет — ответить `unavailable`+return,
если плагин отключён для пользователя — ответить `settings_plugin_disabled`+
return) дублируется дословно в `handle_plugin_menu_callback`
(`bot/telegram_bot.py:5537-...`) на строках **5589-5601** (ветка `action ==
"input"`) и **5618-5630** (ветка `action == "cmd"`), обе — 13 строк, текст
идентичен. Извлечь:

```python
async def _resolve_plugin_command_or_reply(
    self, query, plugin_name: str, cmd_id: str, user_id, bot_language: str,
) -> dict | None:
    cmd = self._get_plugin_command(plugin_name, cmd_id, user_id=user_id)
    if not cmd:
        await query.edit_message_text(
            localized_text('plugins_menu_command_unavailable', bot_language)
        )
        return None
    if self.openai.plugin_manager.is_plugin_disabled_for_user(plugin_name, user_id):
        await query.edit_message_text(
            localized_text('settings_plugin_disabled', bot_language).format(plugin=plugin_name)
        )
        return None
    return cmd
```

Оба call site: `cmd = await self._resolve_plugin_command_or_reply(query,
plugin_name, cmd_id, user_id, bot_language); if cmd is None: return`.
Тест: `tests/test_plugin_menu_force_reply.py` уже покрывает happy-path обеих
веток (`test_plugin_menu_input_imports_force_reply` для `input`,
`test_plugin_menu_command_usage_view_has_close_button` для `cmd`) — ветки
ошибок (`unavailable`/`disabled`) отдельно не проверены; добавить 2 кейса в
тот же файл перед вырезкой (мастер-план: "для групп без тестов — сначала
тест").

**Mode-group keyboard** — построение `mode_groups` (группировка
`chat_modes` по полю `group`) + клавиатура из отсортированных названий групп
+ кнопка "назад" дублируется в `bot/telegram_bot.py:2373-2394` (ветка
`action == "promptback"`) и `6121-6150` (ветка `action == "change_mode"`).
Тексты кнопки "назад" и заголовка отличаются (`prompt_choose_group` /
`session_back_to_sessions` в первом, `session_choose_mode_group` /
`session_back_to_sessions` во втором — заголовок разный, кнопка "назад"
одинаковая). Извлечь:

```python
def _build_mode_group_keyboard(self, chat_modes: dict) -> InlineKeyboardMarkup:
    mode_groups: dict[str, list] = {}
    for mode_key, mode_data in chat_modes.items():
        group = mode_data.get('group', localized_text('session_group_other', self.config['bot_language']))
        mode_groups.setdefault(group, []).append((mode_key, mode_data))
    keyboard = [
        [InlineKeyboardButton(text=group_name, callback_data=f"promptgroup:{group_name}")]
        for group_name in sorted(mode_groups.keys())
    ]
    keyboard.append([InlineKeyboardButton(
        text=localized_text('session_back_to_sessions', self.config['bot_language']),
        callback_data="session:back",
    )])
    return InlineKeyboardMarkup(keyboard)
```

Оба call site заменяют inline-построение на `reply_markup =
self._build_mode_group_keyboard(chat_modes)`, сохраняя свой собственный
текст в `edit_message_text`. Тест: `tests/test_callback_authorization.py`
уже упоминает `promptback`/`change_mode` косвенно — добавить прямую проверку
содержимого клавиатуры (группы+кнопка назад) для обеих веток.

**interpret_image/_stream, provider error log** — эти два под-пункта C15
относятся к `openai_helper.py`, не к `telegram_bot.py` (см. T12c C-blocks
ниже); в подсказке мастер-плана они перечислены в общем списке C15, но по
файлу принадлежат T12c.

---

## T12c — `bot/openai_helper.py`, `bot/openai_tool_handler.py`,
`bot/chat_run.py`, `bot/tool_result.py`

### C1. `_begin_turn` (turn setup)

Два варианта общей заготовки в `bot/openai_helper.py`:
- Полная форма (reentrant-check + state_key/token/turn_id/trace_token/
  stats_token/slog setup, идентичный текст RuntimeError) в
  `get_chat_response` (802-875, дубль-ядро 828-836) и
  `get_chat_response_stream` (900-974, дубль-ядро 925-933).
- Упрощённая форма (state_key/token/lock/delegate/finally-reset) в
  `interpret_image` (2368-2396), `interpret_images` (2407-2431),
  `interpret_image_stream` (2552-2580).

Предложение: два помощника (не один, формы разные по составу полей) —
`_begin_turn(self, chat_id) -> _TurnState` (полная форма, возвращает объект/
namedtuple с `state_key, token, turn_id, trace_token, stats_token`) и
`_begin_simple_turn(self, chat_id) -> _SimpleTurnState` (упрощённая форма).
Оба — приватные методы `OpenAIHelper`, живут в `openai_helper.py` (не в
T12a, т.к. используются только внутри этого файла). **Важно**: перед
вырезкой перечитать оба варианта целиком (не только 6-8 строк ядра) — есть
риск, что T11 (state-контекст, `ChatStateRegistry`/`_CHAT_STATE_KEY`) слегка
меняет это в необъединённом виде; сверить с актуальным HEAD `openai_helper.py`
на момент старта T12c, не с этим планом.
Тест: `tests/test_openai_helper_tool_calls.py`,
`tests/test_openai_helper_session_api.py` уже гоняют оба пути — добавить
прямую проверку идентичности возвращаемого состояния до/после рефактора.

### C2. `_reentry_completion` (re-entry вызовы)

Общий helper `_reentry_session` (`bot/openai_tool_handler.py:261-311`) уже
существует и уже используется всеми тремя местами
(`_retry_missing_delivery_tool:962-1056`,
`_retry_plain_text_tool_intent:1056-1135`, инлайн в
`handle_function_call:1690-1743`) — **дополнительного выноса не требуется**,
дублирование, найденное сканером на этих строках, — это уже частично общий
код вокруг общего `_reentry_session`, а не необработанный дубль. Оставить
как есть; если при ревью найдётся расхождение вызовов, зафиксировать
отдельным пунктом, но по текущему чтению (1000-1140, 1690-1767) выносить
нечего.

### C3/C10/C12a (`openai_helper.py`-часть). Прочие дубли

- **Персист контекста после изменения** — идентичный вызов
  `await self._save_conversation_context(chat_id, {'messages':
  self.conversations[state_key]}, parse_mode, temperature,
  max_tokens_percent, session_id)` в 4 местах:
  `_maybe_apply_auto_chat_mode` (~1247-1254), `reset_chat_history`
  (~3375-3382), `_add_to_history` (~3477-3484), `record_plugin_exchange`
  (~3526-3533). Извлечь `async def _persist_conversation_context(self,
  chat_id, state_key, parse_mode, temperature, max_tokens_percent,
  session_id) -> None`.
- **Session-load/reset-check** (~12-15 строк, 3 места):
  `_get_chat_response_stream_locked` (992-1007),
  `__common_get_chat_response_vision` (2213-2225), `record_plugin_exchange`
  (3501-3520). Извлечь общий приватный helper — перед вырезкой перечитать
  все 3 целиком (не только совпадающее ядро), т.к. это уже граница с T11
  state-кодом; если правки T11 ещё не влиты на момент старта T12c, сверить
  заново.
- **Vision function-call handling** (2×): `_interpret_images_locked`
  (2467-2484), `_interpret_image_stream_locked` (2605-2626). Извлечь общий
  helper для обработки function-call внутри vision-веток.
- **Provider error log** (C15) — except-блоки `_common_get_chat_response`
  (1468-1489: `ProviderRateLimitError`/`ProviderBadRequestError`/
  `ValueError`/`Exception`) практически совпадают с
  `__common_get_chat_response_vision` (2299-2312: то же самое минус ветка
  `ValueError` — в vision-варианте её нет, это осознанное отличие, не
  сливать целиком, а вынести только общие ветки, оставив `ValueError`
  только в text-варианте).
- **`finalize_chat_answer`** (A4, выше) — заменяет тело
  `_interpret_image_text_response` (2514-2550).

Тест на каждый пункт: `tests/test_openai_helper_tool_calls.py`,
`tests/test_per_conversation_serialization.py`,
`tests/test_telegram_streaming.py` покрывают соответствующие пути косвенно;
для `_persist_conversation_context` явного точечного теста нет — добавить
`tests/test_openai_helper_session_api.py`-кейс, что все 4 call site
одинаково персистят контекст (мок `_save_conversation_context`, проверка
аргументов).

### C15 (`openai_helper.py`-часть). `interpret_image`/`_stream`

Уже покрыто пунктами C1 (упрощённая форма `_begin_simple_turn`) и A4
(`finalize_chat_answer`) выше — отдельного дополнительного выноса не
требуется, это одно и то же дублирование, увиденное с двух ракурсов
мастер-плана.

### C-extra. `bot/chat_run.py` — внутренний дубль retry-after-empty-response

Найдено при точечном чтении файла (мастер-план явно называет `chat_run.py`
как файл T12c, детали не расписывал). Внутри `ChatRun.run_non_stream` блок

```python
retry_response = await helper._retry_empty_response_with_tools(
    chat_id, user_id, session_id, allowed_plugins,
    model_to_use=helper._chat_request_models.get(state_key),
)
if retry_response is not None:
    response, retry_plugins_used = await helper._handle_function_call(
        chat_id, retry_response, allowed_plugins=allowed_plugins, user_id=user_id,
        request_context=request_context,
        model_to_use=helper._chat_request_models.get(state_key),
        token_accumulator=token_accumulator, usage_accumulator=usage_accumulator,
    )
    plugins_used += retry_plugins_used
    if is_direct_result(response):
        logger.debug("Direct result returned after empty response retry")
        self._record(AIRunEnd(reason="direct_result_after_retry"))
        helper._chat_request_usage_split[state_key] = aggregate_usage_split(
            token_accumulator, usage_accumulator,
        )
        return response, sum(token_accumulator)
```

повторяется дословно на `bot/chat_run.py:101-126` и `149-174` (два разных
`elif`-ответвления одного и того же `if enable_functions:` блока). Извлечь
приватный метод класса `ChatRun`:

```python
async def _retry_after_empty_response(
    self, *, chat_id, user_id, session_id, allowed_plugins, request_context,
    model_to_use, token_accumulator, usage_accumulator,
) -> tuple[Any | None, tuple]:
    """Вернуть (response, retry_plugins_used) или (None, ()) если ретрая не было."""
    retry_response = await self.helper._retry_empty_response_with_tools(
        chat_id, user_id, session_id, allowed_plugins, model_to_use=model_to_use,
    )
    if retry_response is None:
        return None, ()
    return await self.helper._handle_function_call(
        chat_id, retry_response, allowed_plugins=allowed_plugins, user_id=user_id,
        request_context=request_context, model_to_use=model_to_use,
        token_accumulator=token_accumulator, usage_accumulator=usage_accumulator,
    )
```

Вызывающий код на обоих местах: получает `(response, retry_plugins_used)`,
сам оставляет себе проверку `is_direct_result`+early return (остаётся в
каждой ветке — вынос control-flow с ранним `return` в helper усложнил бы
код без выгоды). Это чисто внутренний дубль одного файла, T12a не
участвует.
Тест: нет прямого теста на `ChatRun.run_non_stream` retry-путь отдельно от
`tests/test_openai_helper_tool_calls.py` (который гоняет
`_common_get_chat_response`, не `ChatRun`) — написать
`tests/test_chat_run.py` (новый) с фейковым `helper`, проверяющим оба пути
(`plugins_used` пусто/непусто) дают одинаковый результат до/после рефактора.

### C-extra2. `bot/tool_result.py` ↔ `bot/openai_tool_handler.py` —
`_artifact_path`

Функция `_artifact_path(value) -> str | None` (валидация: строка, без
`\n`, без `://`, абсолютный путь) определена дословно одинаково дважды:
`bot/tool_result.py:49-55` (модульная, используется внутри
`artifact_entries_from_tool_response`) и `bot/openai_tool_handler.py:719-724`
(приватная, используется внутри `_append_artifact_entry` для построения
manifest с более широким набором ключей — `bot/openai_tool_handler.py`'s
собственный `ARTIFACT_PATH_KEYS`, отличается от `tool_result.py`'s набора,
это два *разных* по составу списка ключей, сливать их нельзя, а вот саму
функцию-валидатор пути — можно).
Правка: `openai_tool_handler.py` убирает свою копию `_artifact_path` и
импортирует её из `tool_result.py` (`from .tool_result import _artifact_path
as _artifact_path` либо, если приватное имя с подчёркиванием импортировать
не принято в проекте — уточнить у публичного соглашения `tool_result.py` уже
экспортирует `artifact_entries_from_tool_response`/`tool_result_content`
как public — можно по аналогии сделать `_artifact_path` публичным
`artifact_path`, раз он теперь используется за пределами модуля). Оба файла
— T12c, конфликта владения нет.
Тест: `tests/test_tool_result.py` покрывает `tool_result.py`'s версию
(побочно через `artifact_entries_from_tool_response`); для
`openai_tool_handler.py`'s `_artifact_path`/`_append_artifact_entry` прямого
теста не нашлось (`tests/test_artifact_paths.py` тестирует `is_deliverable`,
другую функцию) — написать прямой юнит-тест на `_append_artifact_entry`
(валидный путь, путь с `\n`, относительный путь, URL) перед merge, т.к.
группа без теста.

---

## T12d — плагины и прочее

### D1 (C9). `agent_tools.py` — `_manage_plan_tasks`

Ветки `action="add"` (`bot/plugins/agent_tools.py:2620-2639`) и
`action="update"` (2679-2700) разделяют финализацию ответа
(`_save_scope_plan` + сборка `runtime_effects["replan"/"verify"]` +
`_tasks_response` + условное прикрепление `_agent_runtime_effects`) —
это уже задокументировано в `AGENTS.md` (раздел "Deterministic Routing In
Agent Plugins") с теми же диапазонами строк как источник истины; отдельно
выносить в рамках T12 не требуется сверх того, что уже описано — если при
реализации выяснится расхождение с `AGENTS.md`, зафиксировать это как
находку, а не молча менять.

### D2 (C9/C11, новая находка). Снимок задачи в публичном виде

Идентичное dict-comprehension (нормализация задачи в публичный вид)
дублируется в `bot/plugins/agent_tools.py`:
`get_plan_tasks` (def на 2311, дубль на 2318-2325) и `_tasks_response` (def
на 2741, дубль на 2749-2756):

```python
{
    "id": str(task.get("id") or ""),
    "content": str(task.get("content") or ""),
    "status": str(task.get("status") or "pending"),
    "depends_on": self._normalize_depends_on(task.get("depends_on")),
}
```

Извлечь `def _task_public_view(self, task: dict) -> dict` и заменить оба
list-comprehension на `[self._task_public_view(t) for t in tasks]`.
Тест: `tests/test_agent_tools_plugin.py` уже гоняет `get_plan_tasks`;
`tests/test_agent_tools_verify.py`/`tests/test_agent_tools_replan.py`
косвенно бьют `_tasks_response` — добавить прямое сравнение вывода обеих
функций для одного и того же набора задач (до/после должны совпадать).

### D3 (C9/C11, новая находка). `handle_ask_callback` — очистка markup
после ответа

Внутри одного метода `handle_ask_callback`
(`bot/plugins/agent_tools.py:3898-...`) две ветки (multi-select и
single-select) заканчиваются идентичным хвостом:

```python
await query.answer(self.t("agent_tools_answer_received"))
try:
    await query.edit_message_reply_markup(reply_markup=None)
except Exception:
    logging.debug("Failed to clear ask_user markup", exc_info=True)
```

на строках 3934-3939 (multi-select) и 3974-3978 (single-select). Извлечь
`async def _clear_ask_user_markup(self, query) -> None` (вызывает
`query.answer(...)` + `edit_message_reply_markup(None)` с тем же
try/except). Это внутренний дубль одного файла/метода.
Тест: `tests/test_agent_tools_plugin.py` — проверить, есть ли уже кейс на
`handle_ask_callback` для обеих веток; если нет прямой проверки, что markup
очищается в обоих случаях — добавить.

### D4 (C12, явно из мастер-плана). `skills.py` — git clone

`_clone_git_source` (`bot/plugins/skills.py:1981-2000`) и
`_clone_git_source_branch` (2002-2023) идентичны, кроме вставки `["-b",
branch]` в команду. Слить в одну сигнатуру:

```python
def _clone_git_source(self, source: str, temp_dir: Path, branch: str | None = None) -> tuple[Path | None, str | None]:
    git_path = shutil.which("git")
    if not git_path:
        return None, "git executable not found; use an archive URL instead"
    clone_dir = temp_dir / "git-source"
    command = [git_path, "clone", "--depth", "1"]
    if branch:
        command += ["-b", branch]
    command += [source, str(clone_dir)]
    ...  # остальное тело без изменений
```

Единственный вызывающий сайт `_clone_git_source_branch` —
`bot/plugins/skills.py:1905` — заменить на `self._clone_git_source(clone_url,
temp_dir, branch=branch_value)` (уточнить фактическое имя переменной branch
на месте вызова 1900-1908 перед правкой). Удалить `_clone_git_source_branch`
целиком.
**Тест отсутствует** для branch-варианта: `tests/test_skills_plugin.py`
монки-патчит `plugin._clone_git_source` в трёх местах (1380, 1433, 1470),
но `_clone_git_source_branch` нигде не упоминается отдельно — по правилу
мастер-плана ("для групп без тестов — сначала тест") сначала добавить кейс,
явно бьющий путь с `branch` (проверить, что команда содержит `-b
<branch>`), и только потом сливать функции.

### D5 (C14, database.py — опциональная находка, не мандат мастер-плана)

`bot/database.py` содержит `_numeric_env(name, default, cast,
minimum=None)` (~153-172), структурно параллельный
`bot/telegram_bot.py:_positive_int_env` (~69-88) — **но тела разные**
(один логирует+предупреждает про `minimum`, другой молча клэмпит через
`max(1, ...)`) и **имя `_positive_int_env` уже занято другой, тоже
неодинаковой, функцией в `openai_tool_handler.py:58-62`**. Рекомендация:
**не сливать** — поведенческое расхождение делает это небезопасным для
"поведение не меняется", а мастер-план явно этого не требует (только
упоминает `database.py` в контексте C14, что покрывается уже пунктом A1,
т.к. в `database.py` отдельного `parse_kv_list`-подобного парсера не
найдено). Если реализатор T12d всё же захочет унифицировать — это выходит
за рамки T12 и должно обсуждаться отдельно.

### D6 (C8, html_utils.py — не подтверждено)

Автосканер находит крупные дублирующиеся блоки в
`bot/html_utils.py:HTMLVisualizer.advanced_visualization` (метод ~630-1678)
и пересекающийся код в `bot/plugins/codeinterpreter.py:546-605` /
`bot/plugins/show_me_diagrams.py:216-224`. При точечной проверке
(`html_utils.py:690-730, 740-800, 945-980, 1300-1440`;
`codeinterpreter.py:546-605`) весь найденный дубль оказался встроенными
CSS/JS-строковыми шаблонами (стили страницы, функции `savePng`/
`downloadSvg`) — это явно исключённая категория ("CSS-литералы") из
инструкции задачи. **Вывод: не подтверждено как цель для выноса** этим
сканированием. Метод `advanced_visualization` (~1050 строк) не был
прочитан полностью (только точечные диапазоны) — если у реализатора T12d
будет время, можно догрузить его через `large-file-explorer` с вопросом
"есть ли не-CSS/JS-логика, дублирующаяся между html_utils.py и
codeinterpreter.py/show_me_diagrams.py", но специально закладывать это в
объём T12 не стоит без более сильных доказательств.

### D7 (C15). `haiper_image_to_video.py`

`handle_prompt_constructor` (~1270-1327) и
`show_main_menu_with_selections` (~1342-1400+) в
`bot/plugins/haiper_image_to_video.py` дублируют (а) сборку текста-сводки
настроек (if-блоки 1301-1314 vs 1348-1361) и (б) клавиатуру выбора
стиля/эффекта/пресета с отметкой "✓" (1280-1294 vs 1363-1377). Извлечь два
небольших helper'а: `_build_settings_summary_text(...)` и
`_build_style_effect_keyboard(...)` (точные сигнатуры — по месту, зависят
от локальных переменных обеих функций; перед вырезкой перечитать оба
диапазона целиком, не только совпадающее ядро).
Тест: проверить `tests/` на упоминание `handle_prompt_constructor`/
`show_main_menu_with_selections` — на момент планирования прямого теста не
найдено; написать `tests/test_haiper_menu_text.py` (или расширить
существующий haiper-тест) с проверкой, что оба метода дают одинаковый
текст/клавиатуру для одного набора selections.

### D8 (C15). `reminders.py`

Цикл построения клавиатуры списка напоминаний (кнопки view/delete на
напоминание + кнопка "закрыть") дублируется в
`bot/plugins/reminders.py:~157-186` (команда списка) и `~575-601+`
(callback обновления списка). Извлечь `def _build_reminders_keyboard(self,
reminders: list) -> InlineKeyboardMarkup`.
Тест: проверить существующие reminder-тесты на прямое покрытие клавиатуры;
если нет — добавить кейс, сравнивающий вывод обоих call site для одного и
того же списка напоминаний.

### D9. Дубли, рассмотренные и **отклонённые** (для прозрачности, не для
выноса)

- `bot/plugins/agent_tools.py:534-549` / `860-875` — это буквальный
  JSON-schema литерал tool-спеки (`definition_of_done`), не логика; исключён
  правилом "SQL/JSON-схемы... не трогать".
- `bot/plugins/agent_tools.py` ↔ `bot/plugins/agent_cron.py` (три
  кросс-файловых совпадения: обработчики команд `handle_background_command`
  /`handle_cron_command`, построение `job_id`, вызов `send_agent_response`
  с частично общим набором kwargs) — структурно похожи, но это две
  независимо развивающиеся фичи (разовые background-задачи vs cron), и
  выравнивание в общий helper дало бы абстракцию ради экономии ~6-8 строк
  при разном поведении сразу после общего фрагмента. Решение: **не
  выносить** (принцип "никаких абстракций для одноразового кода").
- `bot/openai_tool_handler.py:_filter_tools_by_name` (529-555) /
  `_filter_tools_to_names` (558-581) — структурно параллельны; возможное
  слияние в `_filter_tools(tools, predicate, plugin_manager=None)`
  рассматривалось, но не включено в этот план как обязательный пункт (не
  входит в C1-C15 подсказки, не было явного дубль-блока в скане ≥8 строк).
  Отдельно отмечено: ветка `isinstance(tools, dict)`/
  `function_declarations` (685-716 region) может быть мёртвым кодом по
  правилу AGENTS.md ("нет Google-специфичной обёртки tool spec") — если
  реализатор T12c столкнётся с этим при чтении файла, зафиксировать
  находку отдельно, не удалять молча (вне объёма T12).

---

## Итоговая сводка по трекам

| Трек | Файлы | Кластеры |
|---|---|---|
| T12a | `bot/env_utils.py` (новый), + функции в `bot/chat_response_utils.py`, `bot/utils.py` | помощники для C11/C14 (`env_bool`, `parse_kv_list`), C1/C15 (`leading_system_count`, `finalize_chat_answer`), parse_model_choices |
| T12b | `bot/telegram_bot.py` | C4, C5, C6, C7, C13 (часть), C11/C14 (потребление), C15 (plugin command reply, mode-group keyboard) |
| T12c | `bot/openai_helper.py`, `bot/openai_tool_handler.py`, `bot/chat_run.py`, `bot/tool_result.py` | C1, C2 (уже сделано, без правки), C3/C10/C12, C15 (interpret_image/_stream, provider error log), + внутренний дубль chat_run.py, + `_artifact_path` |
| T12d | `agent_tools.py`, `hindsight_memory.py`, `skills.py`, `html_utils.py` (не подтверждено), `utils.py` (C13 уже в T12b — здесь не трогать повторно), `__main__.py`, `pricing.py`, `database.py`, `haiper_image_to_video.py`, `reminders.py` | C8 (не подтверждено), C9, C11 (часть), C12 (git clone), C14 (потребление), C15 (haiper, reminders) |

Все треки после T12a: писать/расширять тест **до** выноса там, где
отмечено "тест отсутствует"; иначе прогонять существующий тест до и после
правки как критерий приёмки.

## Отклонение от плана (решение координатора, 2026-09-26)

A1 `parse_kv_list` не внедрён и удалён: циклы разбора в `bot/pricing.py` (`load_model_token_prices`) и
`bot/__main__.py` (`parse_model_context_windows_env`) различаются проверками (предупреждение о дубликатах
ключей, отказ от отрицательных значений) и текстами логов, на которые завязаны тесты. Общий помощник
изменил бы поведение, что запрещено правилом T12 «Поведение не меняется». Дубли оставлены намеренно.
