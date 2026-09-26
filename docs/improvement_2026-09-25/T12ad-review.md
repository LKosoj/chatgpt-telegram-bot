# T12a + T12d — ревью

## Раунд 1

Ревьюер проверил diff (`git diff HEAD`) для файлов, владение которыми закреплено за
T12a/T12d: `bot/env_utils.py` (новый), `bot/chat_response_utils.py` (только добавленные
`leading_system_count`/`finalize_chat_answer`), `bot/utils.py` (только добавленный
`parse_model_choices`), `bot/__main__.py` (только консолидация `env_bool`),
`bot/plugins/hindsight_memory.py` (только консолидация `env_bool`),
`bot/plugins/agent_tools.py` (`_task_public_view`, `_clear_ask_user_markup`),
`bot/plugins/skills.py` (слияние `_clone_git_source`/`_clone_git_source_branch`),
`bot/plugins/haiper_image_to_video.py` (`_build_settings_summary_text`,
`_build_style_effect_keyboard`), `bot/plugins/reminders.py` (`_build_reminders_keyboard`),
плюс тесты `tests/test_env_utils.py`, `tests/test_chat_response_utils.py`,
`tests/test_utils_parse_model_choices.py`, `tests/test_t12d_*.py`.

Прочие хунки в этих же файлах (SSRF-хардунинг в `skills.py`/`haiper_image_to_video.py`
через `net_safety`, миграция `reminders.py` с JSON-файла на SQLite, правка позиции
динамической вставки Hindsight-памяти `_dynamic_insert_index` с явной пометкой
`T09 step 4` в тесте, артефакт-скоупинг `_normalize_delivery_artifacts` в
`agent_tools.py`, T02/T05-хунки в `bot/utils.py`/`bot/__main__.py`) принадлежат другим
задачам — не проверялись и не оценивались в этом отчёте.

### Проверено и корректно (без замечаний)

- `env_bool` (`bot/env_utils.py:17-31`) — byte-for-byte эквивалент оригинала из
  `bot/__main__.py`; в `bot/plugins/hindsight_memory.py` все 7 вхождений
  `env.get(NAME, 'true'|'false').lower() == 'true'` заменены на `env_bool(NAME, default)`
  с сохранением исходных default'ов (проверено построчно); `env = os.environ`
  (`hindsight_memory.py:530`), так что `env_bool`'s `os.environ.get` эквивалентен. Ни
  одного необновлённого `.lower() == 'true'`-паттерна в файле не осталось.
  `bot/plugins/mcp_server.py:76` (другая truthy-семантика) и `parse_bool_env`
  (`bot/__main__.py`, кидает исключение) корректно не тронуты, как требует план.
- `leading_system_count` (`bot/chat_response_utils.py:102-116`) и `finalize_chat_answer`
  (`:119-177`) — сверены построчно с обоими оригиналами до рефакторинга
  (`_summarize_and_trim`/`_fallback_trim_with_summary` и
  `chat_run.py:ChatRun.run_non_stream` / `openai_helper.py:_interpret_image_text_response`
  на `HEAD`); тела идентичны с точностью до имён. Обе функции реально используются
  (T12c уже подключил: `openai_helper.py:3746,3824,2535`, `chat_run.py:209`) — не сироты.
- `parse_model_choices` (`bot/utils.py`) — сверен с каноническим
  `OpenAIHelper.get_model_choices()` и с `ChatGPTTelegramBot._configured_openai_models()`
  на `HEAD`: оба не дедуплицируют список (`models.insert(0, default)` только по признаку
  отсутствия), поэтому `test_parse_model_choices_string_does_not_dedupe_and_prepends_default`
  (`raw="a,b, a"` → `["default","a","b","a"]`) — это верно воспроизводит исходное
  поведение, хотя формулировка примера в плане (`T12-plan.md` A3: "без дублей") вводит в
  заблуждение — сам код и тест поведение не меняют, это неточность текста плана, не баг
  реализации.
- `_task_public_view` (`agent_tools.py:102-108`) — оба call site (`get_plan_tasks`,
  `_tasks_response`) заменены на `[self._task_public_view(t) for t in tasks]`; тело
  идентично убранному dict-comprehension в обоих местах.
- `_clear_ask_user_markup` (`agent_tools.py`) — оба хвоста `handle_ask_callback`
  (multi-select confirm и single-select) заменены на вызов, тело метода побайтово
  совпадает с убранным дублирующимся кодом на обеих ветках (сверено с `HEAD`). Тест
  `tests/test_t12d_agent_tools_ask_callback.py` реально гоняет обе ветки через
  `plugin.handle_ask_callback` и проверяет и очистку markup, и разрешение вопроса.
- `_clone_git_source`/`_clone_git_source_branch` слияние (`skills.py`) — новая сигнатура
  `_clone_git_source(source, temp_dir, branch=None)` строит ту же команду
  (`git clone --depth 1 [-b branch] source dest`), что и оба оригинала; единственный call
  site (github-tree путь, `skills.py:1901`) передаёт `branch=tree_branch`, что эквивалентно
  прежнему `if tree_branch: ... else: ...`, т.к. внутри функции тоже `if branch:`.
  `_clone_git_source_branch` полностью удалена, оставшихся ссылок нет.
  `tests/test_t12d_skills_clone_git_source.py` пришпиливает реальную команду subprocess
  для обоих случаев (с branch и без) — годный тест.
- `_build_settings_summary_text`/`_build_style_effect_keyboard` (`haiper_image_to_video.py`)
  — тела идентичны убранным дублирующимся блокам в `handle_prompt_constructor_command` и
  `show_main_menu_with_selections`; порядок построения `menu_text`/`keyboard` относительно
  друг друга изменился, но т.к. это независимые переменные — на итоговый текст/клавиатуру
  не влияет. `tests/test_t12d_haiper_menu_text.py` сравнивает текст и первые два ряда
  клавиатуры между обоими методами для пустых и заполненных настроек — годный тест.
- `_build_reminders_keyboard` (`reminders.py`) — используется в обоих местах
  (`handle_prompt_constructor` и delete-ветке `handle_reminder_callback`), формат кнопок
  (текст, `callback_data=f"reminder:view/delete:{r['id']}"`, кнопка закрытия) совпадает с
  убранными дублирующимися циклами. `tests/test_t12d_reminders_keyboard.py` сравнивает
  реальную клавиатуру (тексты кнопок) между списком и обновлением после удаления через
  БД-фикстуру — годный тест.
- Тесты `tests/test_env_utils.py`, `tests/test_chat_response_utils.py`,
  `tests/test_utils_parse_model_choices.py` покрывают кейсы, перечисленные в плане (A1/A2/A3
  /A4), включая граничные случаи (дубликат ключа, невалидная запись, non-dict элемент,
  usage-split несовпадение и т.д.).
- Прогон: `~/.venvs/ctb/bin/python -m pytest tests/test_env_utils.py
  tests/test_chat_response_utils.py tests/test_utils_parse_model_choices.py
  tests/test_t12d_*.py -q` — 36 passed. Плюс регрессия по затронутым плагинам/модулям
  (`test_hindsight_mutator.py`, `test_agent_tools_plugin.py`, `test_agent_tools_verify.py`,
  `test_agent_tools_plan_rule_mutator.py`, `test_skills_plugin.py`,
  `test_haiper_image_to_video_async_db.py`, `test_reminders_fixes.py`, `test_pricing.py`,
  `test_telegram_builder_config.py`, `test_instance_lock.py`) — 308 passed, 0 failed.
  `ruff check` на всех owned-файлах — чисто. mypy: 0 ошибок в `env_utils.py` и
  `chat_response_utils.py` (было 0); в остальных owned-файлах счётчик ошибок по
  `(файл, код)` не вырос относительно `/tmp/impl/mypy_before.txt` (`hindsight_memory.py`
  21→21, `agent_tools.py` 16→16, `haiper_image_to_video.py` 69→69, `reminders.py` 18→18);
  `skills.py` даже снизился (18→16, за счёт стороннего net_safety-рефакторинга). Рост
  `attr-defined` в `bot/utils.py` (16→19) целиком в чужом хунке `handle_direct_result`
  (T02/T05), вне `parse_model_choices` — не относится к этой задаче.
- Tool specs (`get_spec()`) во всех проверенных плагинах не менялись (grep по diff — пусто).

### WARNING

1. **`parse_kv_list` — мёртвый код.** `bot/env_utils.py:34-71` определяет
   `parse_kv_list`, план (A1) прямо требует подключить его в
   `bot/pricing.py:load_model_token_prices` и `bot/__main__.py:
   parse_model_context_windows_env` (оба — файлы T12d). Ни один из циклов не заменён:
   `bot/pricing.py` не изменён вовсе (`git diff HEAD -- bot/pricing.py` пуст), а
   `parse_model_context_windows_env` (`bot/__main__.py`) по-прежнему содержит
   собственный ручной цикл разбора `key=value,...`. `parse_kv_list` используется только
   в своих тестах (`tests/test_env_utils.py`) — сирота. Исправление: либо подключить в
   обоих местах (как требует план — сохранив сигнатуры/тексты логов), либо удалить
   `parse_kv_list` и связанные тест-кейсы `test_parse_kv_list_*` из
   `tests/test_env_utils.py`, если решено не выносить в этом раунде.
2. **`agent_tools.py._model_choices_for_helper` не переведён на `parse_model_choices`.**
   План (A3) явно требует заменить fallback-блок в `_model_choices_for_helper`
   (`bot/plugins/agent_tools.py`, определение начинается сразу после
   `MAX_SUBAGENT_TOOL_ROUNDS`) на вызов `parse_model_choices`. Правка сделана только в
   `bot/telegram_bot.py:5877` (T12b) — в `agent_tools.py` остаётся дословная копия того
   же 10-строчного блока. Не баг (поведение то же), но незавершённый пункт плана T12d;
   стоит либо доделать замену, либо явно задокументировать причину отказа.
3. **Отсутствует тест D2, явно требуемый планом.** План (T12d D2) требует "добавить
   прямое сравнение вывода обеих функций [`get_plan_tasks`/`_tasks_response`] для одного
   и того же набора задач (до/после должны совпадать)". Поиск по всем тестам
   (`grep _task_public_view`) не находит ни одного упоминания вне
   `bot/plugins/agent_tools.py` — такого теста нет ни в `test_agent_tools_plugin.py`, ни
   в новых `test_t12d_*.py` файлах. Сама выборка `_task_public_view` проверена вручную
   (идентична убранному коду), поэтому риска регрессии сейчас нет, но плановое
   требование к тестам не выполнено.

### ERROR

Нет.

### Итог

0 ERROR, 3 WARNING, 0 NIT. Реализация T12a-помощников и T12d-дублей поведенчески
корректна везде, где проверялась вручную (сверка построчно с оригиналами на `HEAD`);
все замечания — про незавершённость отдельных пунктов плана (два неподключённых
консолидирующих вызова + один недостающий регрессионный тест), не про баги в уже
написанном коде.

## Раунд 2

Проверены исправления всех трёх WARNING из раунда 1.

1. **`parse_kv_list` — мёртвый код.** Функция полностью удалена из
   `bot/env_utils.py` (в файле остался только `env_bool`). `grep -rn parse_kv_list`
   по всему дереву (код + тесты) — 0 совпадений; в `tests/test_env_utils.py` больше
   нет `test_parse_kv_list_*`. `bot/pricing.py` и `parse_model_context_windows_env`
   (`bot/__main__.py`) не тронуты — решение "удалить, а не подключать" реализовано
   чисто, сирот не осталось. Исправлено.
2. **`_model_choices_for_helper` не переведён на `parse_model_choices`.**
   `bot/plugins/agent_tools.py:120-125` теперь делает `return parse_model_choices(
   config.get("model_choices"), config.get("model") or "")`; импорт добавлен
   (`from ..utils import ..., parse_model_choices`). Сверка семантики вручную
   (в т.ч. edge-case `model_choices=""` — старый код шёл через `choices = "" or []`
   → `[]`, новый код видит `isinstance("", str) is True` и идёт в ветку
   `"".split(",")` → тоже `[]` после фильтрации пустых элементов — результат
   идентичен, просто другой путь) подтверждает эквивалентность. Новый тест
   `tests/test_t12d_agent_tools_model_choices.py` дополнительно фиксирует это:
   воспроизводит убранный блок как `_legacy_model_choices` и параметризованно
   (12 кейсов: строка/список, пустые/`None`/whitespace-элементы, default уже в
   списке, default пустой/`None`) сравнивает с `_model_choices_for_helper` —
   годный регрессионный тест. Исправлено.
3. **Отсутствует тест D2.** `tests/test_agent_tools_plugin.py::
   test_get_plan_tasks_matches_tasks_response_snapshot` добавлен: строит план из
   двух задач (одна `completed`, одна `in_progress` с `depends_on`), затем
   `assert listed["plan_tasks"]["tasks"] == plugin.get_plan_tasks(...)` — прямое
   сравнение вывода `manage_plan_tasks(action="list")` и `get_plan_tasks()`, как
   требовал план. Исправлено.

**Прогон:** `tests/test_env_utils.py tests/test_chat_response_utils.py
tests/test_utils_parse_model_choices.py tests/test_t12d_*.py
tests/test_agent_tools_plugin.py tests/test_openai_helper_tool_calls.py` — все
зелёные (в составе общего прогона `tests/ bot/tests/` — 1944 passed, 0 failed).
`ruff check` на изменённых файлах — чисто. mypy: 0 ошибок в `bot/env_utils.py`;
в `bot/plugins/agent_tools.py` рядом с `_model_choices_for_helper` (строки
110-130) новых ошибок нет; `(файл, код)`-сравнение с `/tmp/impl/mypy_before.keep`
по всем owned-файлам показывает только уменьшения (следствие параллельной T13,
согласно заметке о том, что T13 одновременно правит аннотации в тех же файлах) —
ни одного нового кода ошибки.

**Итог раунда 2:** 0 ERROR, 0 WARNING, 0 NIT. Все три замечания раунда 1
устранены корректно и с тестами.
