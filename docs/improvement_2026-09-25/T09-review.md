# T09. Промпты в коде — review

## Раунд 1

**Вердикт: ERROR 0, WARNING 0, NIT 1.** Изменения точно соответствуют
`T09-plan.md`, тесты (273/273 таргетных, 1818/1818 полного набора) зелёные,
`ruff` чистый, новых `mypy`-ошибок не найдено.

### Проверенные файлы (diff HEAD, только хунки в зоне владения T09)

- `bot/openai_helper.py`: `ask()` — формат даты и опечатка «помошник»→«помощник»
  (строки 793/795); `_build_auto_chat_mode_prompt` — переформулировка правила 4
  («сильный сигнал») и перестановка хвоста (`Доступные режимы` перед
  `Сообщение`). Остальные хунки в этом файле (`__add_function_call_to_history`
  wrap/role="user") — T08, не в зоне T09, не проверялись по существу.
- `bot/plugins/agent_tools.py`: `_PLAN_RULE_TEXT` (порог «больше двух шагов»,
  убраны предложения про «намерение/оцени»), новая `_dynamic_insert_index`,
  переезд checkpoint/re-plan/verify инжектов на `_dynamic_insert_index` в
  `on_before_chat_request`, переформулировка `_verify_message_body`. Хунки про
  `is_deliverable`/`_allowed_artifact_roots`/`wrap_untrusted_tool_output` для
  tool-результатов — не T09, не проверялись.
- `bot/plugins/hindsight_memory.py`: `_dynamic_insert_index` + перенос
  динамического recall-инжекта на него; baseline recall остался на `insert_at`.
- `bot/plugins/skills.py`: разделение каталога на
  `_build_static_skills_catalog`/`_build_active_skills_catalog`,
  `on_before_chat_request` вставляет статику на `insert_at`, динамику — на
  `_dynamic_insert_index`. SSRF-хунки (`net_safety`, `_safe_open`,
  `_validate_external_url`) — не T09, не проверялись.
- `bot/chat_modes.yml`: правило «Function … returned» (skills_agent) **не
  тронуто** — проверено, что блокирующее условие из плана (шаг 0) всё ещё
  верно: `bot/openai_helper.py` в `__add_function_call_to_history` по-прежнему
  строит `f"Function {function_name} returned: {content}"` (T08 поменял только
  роль на `user`, сам текст не убрал). Значит шаг 5 плана корректно пропущен —
  это не находка, а подтверждение правильного решения разработчика.

### Корректность позиционирования (ключевая проверка по заданию ревью)

- `_dynamic_insert_index` идентична побайтово во всех трёх файлах (проверено
  прямым сравнением исходников).
- Прошёл трассировку всех 3 сайтов вызова `_apply_before_chat_request_mutators`
  в `bot/openai_tool_handler.py` (строки 1021, 1100, 1729 в текущем файле —
  номера из плана устарели из-за параллельных задач, но логика та же):
  во всех трёх местах либо `tool_calls` в этом раунде не было вообще (синтетическое
  `role: user` сообщение-реплай добавляется в историю до вызова мутаторов), либо
  (сайт 1729, обычный re-entry после исполнения тулов) ассистентское сообщение с
  `tool_calls` и все его `tool`-результаты уже дописаны в
  `self.conversations[state_key]` до вызова мутаторов
  (`add_assistant_tool_calls_to_history` на 1383, `add_tool_result`/
  `_add_function_call_to_history` в цикле раньше). Значит `messages[-1]` в
  момент вызова мутатора никогда не оказывается «ассистент с tool_calls без
  результатов» — сценарий разрыва assistant/tool пары невозможен.
- Пустая история: `_dynamic_insert_index([])` → `0` (корректно), и все три
  `on_before_chat_request` либо не доходят до вставки при пустой/без-user
  истории (hindsight требует `last_user`, skills — `if not messages: return
  None`), либо (agent_tools) вставка в индекс 0 безопасна.
- Множественная вставка в одном вызове (agent_tools: checkpoint + re-plan +
  verify одновременно) — порядок сохранён через инкремент `dyn_idx` после
  каждой вставки, как и в плане.
- Роутер: парсер ответа (`bot/openai_helper.py:1235` `_required_choice_message_text
  → .strip().lower() → get_mode_by_key`) не зависит от позиции текста в
  промпте — перестановка `Доступные режимы`/`Сообщение` безопасна (проверено
  чтением кода, не только по утверждению плана).
- Маркеры не тронуты: `_PLAN_RULE_MARKER`, `_REPLAN_TRIGGER_MARKER`,
  `_VERIFY_TRIGGER_MARKER`, `_WORKING_CHECKPOINT_MARKER`, подстрока
  `"manage_plan_tasks"` в `_PLAN_RULE_TEXT` — все на месте.
- Tool specs (`get_spec()`, `tools:` в `chat_modes.yml`, `prompt_markers`) —
  без изменений (grep по diff подтвердил ноль совпадений).

### Тесты

Новые/изменённые тесты (`test_agent_tools_plan_rule_mutator.py`,
`test_hindsight_mutator.py`, `test_skills_plugin.py`,
`test_openai_helper_tool_calls.py`, `test_agent_tools_verify.py`) точно
покрывают оба требуемых сценария (вставка перед последним user при наличии
истории; вставка в конец во время tool-раунда с проверкой, что
assistant(tool_calls)/tool остаются соседними и неизменными) для каждого из
трёх плагинов, плюс тест побайтовой стабильности префикса
(`test_before_chat_request_prefix_stable_across_requests_with_dynamic_change`)
через реальный `AgentToolsPlugin`, подключённый к `_apply_before_chat_request_mutators`.
Прогон целиком зелёный:

```
~/.venvs/ctb/bin/python -m pytest tests/test_agent_tools_plan_rule_mutator.py \
  tests/test_hindsight_mutator.py tests/test_skills_plugin.py \
  tests/test_skills_prompt_fragment.py tests/test_openai_helper_tool_calls.py \
  tests/test_agent_tools_verify.py -q --no-header -p no:cacheprovider
# 273 passed

~/.venvs/ctb/bin/python -m pytest tests bot/tests -q --no-header -p no:cacheprovider
# 1818 passed
```

`ruff check` — чисто. `mypy` на 4 файлах: 80 ошибок против 82 в baseline;
построчное сравнение (нормализация по номеру строки убрана, сравнивались тексты
сообщений) показало, что единственные два «новых» сообщения
(`Name "blocked_transitions"/"completed_transitions" already defined`,
`agent_tools.py`) — это тот же самый долг из baseline
(`mypy_before.keep:390-391`, было на строках 2571/2572, стало 2579/2580) —
просто сдвинулся из-за правок других задач в этом же файле. Итог: **новых
mypy-ошибок от T09 нет**.

### NIT

1. **`_verify_message_body` ссылается на инструмент как `manage_plan_tasks`
   без префикса `agent_tools.`** (`bot/plugins/agent_tools.py:2163-2168`, обе
   фразы — «проверь через manage_plan_tasks», «через
   manage_plan_tasks(action=update)»), тогда как модель видит инструмент как
   `agent_tools.manage_plan_tasks`. Это не регрессия T09 — `_replan_message_body`
   (не тронут) использует тот же безпрефиксный стиль до T09, так что новая
   фраза просто продолжает уже существующий локальный паттерн файла. В
   `chat_modes.yml` (skills_agent) те же инструменты, наоборот, упоминаются с
   полным префиксом (`agent_tools.ask_telegram_user`,
   `agent_tools.deliver_to_user`). Несостыковка стилей предсуществует и не
   входит в объём T09 (план явно дал этот текст дословно) — упоминаю для
   осведомлённости, чинить не нужно.
