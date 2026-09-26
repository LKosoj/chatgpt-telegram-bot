# T13c. `openai_helper.py`, `bot/plugins/plugin.py` и остальные bot/-файлы (mypy-cleanup, волна 8) — ревью

## Раунд 1

**Область ревью.** 42 файла из `T13-plan.md` (владение T13c): `bot/openai_helper.py`,
`bot/plugins/plugin.py`, `bot/agent_delivery.py`, `bot/ai_events.py`, `bot/chat_run.py`,
`bot/conversation_key.py`, `bot/html_utils.py`, `bot/openai_tool_handler.py` и 34 плагина
(полный список — `/tmp/t13c_files.txt`, совпадает с планом). `git diff HEAD` по этим файлам
смешивает T13c с хантками T02–T12 (общее рабочее дерево без коммитов на задачу), поэтому
каждый файл разбирался вручную: T13c-хантки отделялись от уже отревьюженных чужих правок по
известным маркерам (SSRF-хардненинг T03/`net_safety`, SQLite-миграция T04 в `agent_cron.py`/
`reminders.py`, dedup-рефакторы T09/T12, `returns_untrusted_content`/`DANGEROUS_TOOL_NAMES`
T08, `ChatStateRegistry`/провайдер T10/T11) и оценивались только они, сверяясь с
`/tmp/impl/mypy_before.keep` (171 T13c-специфичная ошибка, отфильтрована в
`/tmp/impl/t13c_before_errors.txt`) как ground truth "что именно нужно было чинить".

36 из 42 файлов (все небольшие/средние по объёму diff) прочитаны и проверены лично
построчно. Три самых больших и переплетённых с другими задачами файла делегированы фоновым
агентам с точным списком ground-truth ошибок и явно перечисленными точками повышенного риска
(закрытие через `assert`, `cast()`, переименования при "no-redef"); по каждому из них
дополнительно лично перепроверен самый рискованный пункт (trust-but-verify):
`bot/openai_helper.py` (940 строк diff, 27 ground-truth ошибок; лично перепроверено
`self.bot`/`get_file` — см. WARNING ниже), `bot/plugins/hindsight_memory.py` +
`bot/plugins/reminders.py` (~860 строк diff, `assert self.client is not None` — 10+ мест),
`bot/plugins/skills.py` + `bot/plugins/agent_tools.py` (~900 строк diff; лично перечитан
`_manage_plan_tasks`, `bot/plugins/agent_tools.py:2554-2697`, — единственный no-redef-фикс,
трогающий ветвление, а не просто переименование).

**Проверка get_spec() на побайтовую идентичность.** Скрипт `/tmp/impl/verify_get_spec.py`
извлекает тело `get_spec()` из `git show HEAD:<file>` и из текущего файла (регекс на границу
следующего метода, включая `async def`) и сравнивает построчно, кроме сигнатуры (там меняется
только `-> [Dict]:` → `-> List[Dict]:`). Прогнано по всем 35 T13c-плагинам с `get_spec()` —
**все 35 "OK"**, тела byte-identical. Первая версия скрипта ошибочно репортила 10
false-positive "BODY DIFFERS" (регекс `^(\s*)def \w` не матчил `async def`, из-за чего в тело
`get_spec()` затягивался хвост `execute()`) — исправлено, перепрогнано, результат чистый.

**Известные намеренные изменения (из задания) — проверены:**
- `bot/plugins/chief.py:189` (`_parse_with_retry` → `Tuple[Optional[Dict], int]`) — совпадает
  с уже существовавшим `return None, 0` после исчерпания ретраев; единственный вызывающий
  (`:513`, `if not params:`) уже страховался. SAFE.
- `bot/plugins/codeinterpreter.py` (`install_package` → `return False` в except) — обе точки
  вызова используют булев контекст/отбрасывают результат, разницы None/False не видно. SAFE.
- "Unknown function"-фолбэки (`chief.py`, `ask_your_pdf.py`, `weather.py`, `spotify.py`,
  `github_analysis.py`, `text_document_qa.py`, `terminal.py`, `iplocation.py`) — для каждого
  плагина скриптом сверены имена функций в `get_spec()` и ветки диспетчера в `execute()`:
  во всех 8 плагинах набор веток `execute()` покрывает 100% имён из `get_spec()`, т.е. фолбэк
  недостижим при нормальном потоке (`plugin_manager.py:578` вызывает `execute(base_name, ...)`
  только с именем, уже провалидированным против `get_spec()`). Раньше при недостижимой ветке
  функция без явного `return` в конце неявно отдавала `None`; теперь отдаёт
  `{"error": "Unknown function: ..."}` — оба варианта недостижимы в проде, разницы для
  реальных вызывающих нет.
- `bot/plugins/ask_your_pdf.py` — перенос `from .plugin import Plugin` перед try/except
  pypdf/pdfminer плюс `pypdf: Any`/`pdfminer_extract_text: Any` — чистые аннотации/порядок
  импорта, поведение не изменилось.
- `assert process.stdout is not None` / `assert process.stderr is not None`
  (`bot/plugins/terminal.py:217-218`, аналогично `bot/plugins/skills.py:2410-2422`/`:3161-3175`) —
  добавлены сразу после `create_subprocess_*(..., stdout=PIPE, stderr=PIPE)`, где `PIPE`
  гарантирует не-`None` по контракту `asyncio`. До фикса `None` привёл бы к `AttributeError`
  в том же месте (`stream.read(...)`) — сейчас `AssertionError` там же. Событие недостижимо в
  обоих случаях. SAFE.
- `bot/plugins/agent_tools.py:2554-2697` (`_manage_plan_tasks`, no-redef fix) — лично
  перечитана вся функция: `if action == "add": ... return response` (2571-2636) и
  `if action == "update": ... return response` (2638-2697) — два независимых, взаимоисключающих
  блока (не `elif`, но первый всегда завершается `return` раньше, чем мог бы начаться второй).
  Фикс убрал вторую избыточную аннотацию `blocked_transitions: List[str] = []` →
  `blocked_transitions = []` (`:2644-2645`) — не переименование, а просто снятие дублирующей
  аннотации, на которую жаловался mypy ("no-redef" в рамках одной области видимости функции).
  Именования и области видимости append/read не пересекаются между ветками. SAFE.

**Прогон тестов и статического анализа.**
- `~/.venvs/ctb/bin/python -m pytest tests/ bot/tests/ -q` — **1950 passed**, 0 failed,
  3 warnings (все — сторонний `PTBDeprecationWarning`, не относится к T13c).
- `python3 -m mypy --config-file pyproject.toml --python-executable ~/.venvs/ctb/bin/python bot`
  — **Success: no issues found in 85 source files**. Полный ноль, а не только по T13c-файлам.
- `~/.venvs/ctb/bin/python -m ruff check <42 T13c-файла>` — **All checks passed!**
- `# type: ignore` в T13c-владении — **0 вхождений** (проверено скриптом по всем 42 файлам);
  требование "никаких бланкетных игнор-комментариев" выполняется тривиально — их вообще нет.

## Находки

### WARNING: `self.bot` типизирован как `Any` вместо честного `Bot | None` с реальной проверкой
- **Файл:** `bot/openai_helper.py:285` (`self.bot: Any = None`, было `self.bot = None`).
- В `__init__` атрибут всегда стартует как `None`; реальное значение (`application.bot`)
  проставляется только после старта бота — подтверждено в `bot/telegram_bot.py:209`
  (`self.openai.bot = None`, инициализация) и `bot/telegram_bot.py:6319`
  (`self.openai.bot = application.bot`, после запуска). Значит `self.bot` рантайм-опционален
  (`Bot | None`), но T13c вместо канонического паттерна этой же задачи
  (`X | None`-виджнинг + guard) типизировал его как `Any`, что полностью выключает проверку
  mypy для всех обращений к `self.bot`. Единственная существующая защита на месте вызова
  (`bot/openai_helper.py:3977-3989`, `if not hasattr(self, 'bot'): raise ValueError(...)`) не
  спасает: атрибут `bot` присутствует всегда (со значением `None` до старта), так что
  `hasattr` всегда `True`, даже когда `self.bot is None`. Если `get_file_data` вызвать до
  того, как бот стартовал, `self.bot.get_file(file_id)` по-прежнему упадёт с
  `AttributeError: 'NoneType' object has no attribute 'get_file'` — ровно как и до T13c.
  Регрессии нет (поведение побайтово то же), но это единственное место в T13c, где вместо
  реального closing typing gap выбрана заглушка `Any`, скрывающая, а не устраняющая пробел,
  который остальные 41 файл в этой же задаче последовательно закрывают через
  `X | None`/`assert`/`cast()`.
- **Фикс:** `self.bot: "Bot | None" = None` (или без кавычек, т.к. модуль уже пользуется
  отложенными аннотациями) + на месте вызова (`:3977-3989`) заменить `hasattr`-проверку на
  `if self.bot is None: raise ValueError(...)`, чтобы ошибка стала явной и типизированной, а
  не полагалась на побочный эффект `AttributeError`.

### NIT: `reminders.py` — ранний `return` вместо прежнего "громкого" падения в недостижимом на практике None-сценарии
- **Файл:** `bot/plugins/reminders.py:526-604` (`handle_reminder_callback`).
- Добавлено `query = update.callback_query; if query is None: return` в начале функции.
  До фикса при гипотетическом `update.callback_query is None` код упал бы с `AttributeError`
  внутри `try`, был бы пойман `except Exception as e:` (`:599-604`), а затем упал бы ещё раз —
  уже без перехвата — на `query.edit_message_text(...)` внутри самого except-блока (т.к.
  `query` там тоже `None`). Теперь функция тихо завершается `return` до входа в `try`, без
  исключения и без лога. Сценарий недостижим на практике: обработчик зарегистрирован через
  `callback_query_handler` с `callback_pattern: "^reminder:"`, и PTB вызывает такие хендлеры
  только для реальных `callback_query`-апдейтов — так что это не живая регрессия, а
  теоретическое расхождение поведения, зафиксированное для полноты (аналогичный паттерн
  отмечен и в независимом ревью `hindsight_memory.py`/`reminders.py`, делегированном фоновому
  агенту).
- **Фикс:** не требуется — эффект не наблюдаем ни в одном достижимом сценарии.

### NIT: два ground-truth mypy-фикса в `skills.py` не удалось однозначно проследить до конкретных строк текущего кода
- **Файл:** `bot/plugins/skills.py` (ground-truth ошибки до T13c: `:295` — "Incompatible types
  in assignment (dict[str, Any] | None → dict[str, Any])"; `:2081` — "Incompatible return
  value type (got str | int, expected str | None)").
- Обе строки взяты из `/tmp/impl/mypy_before.keep`, снятого после T01–T12, но до T13; сам
  файл в этой области сильно переписан несвязанными с T13c задачами (разбивка prompt-каталога
  на `_build_static_skills_catalog`/`_build_active_skills_catalog`, слияние
  `_clone_git_source`/`_clone_git_source_branch`, SSRF-вынос в `net_safety`), из-за чего номера
  строк уехали далеко от исходных. Проверены вручную все функции файла с сигнатурой
  `-> str | None`/`-> Optional[str]` (`_reject_html_markdown:1965`, `_resolve_safe_ip:2061`,
  `_validate_external_url:2070`, `_planning_warning:3058`, `_resolve_skill_id:3402`) и все с
  `-> Dict[str, Any] | None`/`Optional[Dict` (`_disabled_skill_error:941`,
  `_listed_skill_file:2702`, `_resolve_skill_agent:2915`) — ни в одной нет ветки,
  возвращающей `int` вместо `str` или переприсваивающей заведомо не-`Optional`-переменную.
  Независимое фоновое ревью (делегированное на `skills.py`+`agent_tools.py`) пришло к тому же
  выводу и тем же способом не смогло локализовать оба пункта. Оба пункта подтверждённо
  устранены — полный прогон mypy по всему `bot/` чист (0 ошибок), и среди проверенных
  кандидатов не нашлось паттерна, похожего на "скрытую" правку, отличную по характеру от
  остальных 169 T13c-фиксов в этом файле (все — аннотации/`assert`/переименования).
- **Фикс:** не требуется — риска не выявлено, отмечено как ограничение методологии ревью,
  не как найденная проблема.

## Итог раунда 1

**0 ERROR, 1 WARNING, 2 NIT.**

- WARNING: `bot/openai_helper.py:285` — `self.bot: Any = None` маскирует реальную
  Optional-природу атрибута вместо `Bot | None` + honest-guard (не регрессия, отступление от
  паттерна задачи).
- NIT: `bot/plugins/reminders.py:526-604` — недостижимый на практике silent-return вместо
  прежнего double-AttributeError в гипотетическом `callback_query is None`.
- NIT: `bot/plugins/skills.py:295,2081` (ground-truth номера) — два фикса не локализованы
  построчно из-за сильного сдвига номеров строк чужими задачами; подтверждены устранёнными
  через чистый общий прогон mypy, паттерн риска не найден.

Тесты (1950 passed), mypy (0 ошибок по всему `bot/`), ruff (чисто), `# type: ignore` (0
вхождений в T13c-файлах), get_spec() (35/35 byte-identical) — все проверки задания пройдены.
