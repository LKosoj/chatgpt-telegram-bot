# T12b — ревью: `bot/telegram_bot.py` копипаста-рефакторинг

Ревьюер проверял `git diff HEAD -- bot/telegram_bot.py` против `docs/improvement_2026-09-25/T12-plan.md`
(раздел T12b) и мастер-плана. Диапазон вне T12b (правки `retry_after_seconds`/`RetryAfter` —
это T02) не оценивался по существу, только на предмет того, что он не задевает извлекаемые
T12b-блоки — не задевает.

## Раунд 1

### Проверено по каждому пункту плана

- **B1 `_send_markdown_or_plain`** (`bot/telegram_bot.py:139-156`) — оба call site
  (`_describe_image_from_context` ~1054, media-group `_execute` ~3044) заменены на вызов
  общего хелпера; третий похожий site (vision-stream, entities-based,
  `render_markdown_message_entities`, ~3266) не тронут, как требовал план. Поведение
  идентично (chunk-разбиение, `parse_mode=MARKDOWN` → `except BadRequest` → plain retry,
  `reply_to_message_id` только на первом чанке) — подтверждено построчным сравнением diff.
- **B2 `_dispatch_session_reset`/`_dispatch_user_message`** (`:78-110`) — все 6
  `on_session_reset` и все 4 `on_user_message` call site из плана переведены на общие методы;
  `getattr`-вариант (`_handle_direct_result`, было 707-716) оставлен с `getattr` на
  call site, но значения передаются в общую сигнатуру. Структурная проверка
  (`src.count("SessionResetPayload(") == 1`, `...UserMessagePayload(") == 1`) подтверждает
  отсутствие остаточных копий.
- **B3 `_build_busy_status`** (`:648-657`) — все 3 site (media-group, single vision,
  non-stream reply) заменены; `BusyStatusMessage(` встречается в файле 1 раз,
  `self._build_busy_status(` — 3 раза.
- **B4 `_send_rich_markdown_if_fits`** (`:677-693`) — оба branch (`rich_stream_active`,
  `rich_stream_final_only`) заменены; полный диапазон ~4230-4370 перечитан построчно —
  fall-through на `tokens == 'not_finished'` (когда хелпер возвращает `(None, None)`) ведёт в
  тот же код (`if should_send_draft:` / переход к следующей итерации), что и раньше; блоки
  `except RetryAfter`/`except Exception` вокруг не изменены. Поведение идентично.
- **B5 `_budget_user_and_name`** (`check_allowed_and_within_budget`, `:4928`) — дублирующий
  блок разрешения "кто отправил" (4 варианта: inline/callback/message/effective_user) заменён
  вызовом `utils._budget_user_and_name`, логика 1:1 совпадает с тем, что было инлайново.
- **B6 `parse_model_choices`** (`_configured_openai_models`, `:5871-5877`) — fallback-блок
  заменён; проверены edge-cases (falsy/empty `model_choices`, список vs строка, default model
  уже в списке) — во всех случаях новый вызов даёт тот же результат, что и старый инлайн-код
  (в т.ч. отсутствие дедупликации сохранено).
- **B7** `_resolve_plugin_command_or_reply` (`:585-599`) и `_build_mode_group_keyboard`
  (`:648-... /2377-2400`) — оба паттерна вынесены, оба call site на каждый заменены,
  различающийся текст (`prompt_choose_group` vs `session_choose_mode_group`) сохранён на
  call site, а не в хелпере. Третий похожий site с `_get_plugin_command`
  (`handle_plugin_menu_args_reply`, ~5633) корректно НЕ включён в рефакторинг — он
  структурно другой (нет проверки `is_plugin_disabled_for_user`, другой способ ответа,
  доп. side effect на `plugin_menu_pending`), в плане как дубль не заявлен.

### Тесты и статический анализ

- `tests/test_t12b_*.py` (5 новых файлов) + `tests/test_usage_budget.py` +
  `tests/test_plugin_menu_force_reply.py` + `tests/test_callback_authorization.py` — 83/83
  passed.
- Полный прогон `tests/ bot/tests/` — 1930 passed, без регрессий.
- `ruff check bot/telegram_bot.py` + 5 новых test-файлов — 1 ошибка (см. WARNING ниже).
- `mypy bot/telegram_bot.py` (сравнение с `/tmp/impl/mypy_before.keep` по тексту ошибки без
  номера строки) — новых типов ошибок не появилось; количество вхождений по некоторым типам
  уменьшилось (естественное следствие дедупликации call site, не потеря типобезопасности).

### Находки

**ERROR:** нет.

**WARNING (2):**

1. `tests/test_t12b_configured_openai_models.py:13` — неиспользуемый импорт `import pytest`
   (`ruff` F401: в файле нет ни одной `async`/`@pytest.mark`-функции, plain `def test_*`).
   Реальная ошибка линта, упадёт в CI. Исправление: удалить импорт.
2. Тестовое покрытие B1 (`tests/test_t12b_markdown_fallback.py`) не соответствует плану
   буквально: план просил тест, "проверяющий, что оба call site... дают одинаковый вызов
   reply_text" — файл тестирует call site 1 (`_describe_image_from_context`) и сам хелпер
   `_send_markdown_or_plain` напрямую, но не call site 2 (media-group `_execute`, ~3044)
   впрямую, и не содержит структурной guard-проверки "хелпер определён 1 раз / оба call site
   его вызывают" — в отличие от B2/B3/B4, где такая guard-проверка (`source.count(...)`) есть.
   Код проверен вручную построчно (см. выше) и корректен, поэтому это не баг, а пробел в
   тестовом покрытии/несогласованность с соседними кластерами. Рекомендация: добавить к файлу
   аналогичный `test_no_duplicate_..._remains`-guard (`source.count("self._send_markdown_or_plain(") == 2`)
   и/или прямой тест на media-group site.

**NIT (2):**

1. `_dispatch_session_reset`/`_dispatch_user_message` (`bot/telegram_bot.py:78-93`) не имеют
   docstring, в отличие от соседних новых хелперов B1/B4/B7, которые его имеют. Не мешает
   пониманию (имена самодокументирующие), но немного несогласованно.
2. Импорт `_budget_user_and_name` (приватное имя с `_`) из `bot/utils.py` в
   `bot/telegram_bot.py` — прямо предписано текстом мастер-плана для B5, поэтому не считаю
   это отклонением, просто отмечаю для протокола (граница модуля пересекает обычную PEP8-
   конвенцию "приватное не импортировать наружу").

### Вне области ревью (замечено, не оценивалось)

- `bot/telegram_bot.py`: замена `e.retry_after` → `retry_after_seconds(e)` и импорт
  `retry_after_seconds` из `bot/telegram_stream.py` — это T02, не T12b; не пересекается с
  извлечёнными в T12b блоками.
- `tests/test_usage_budget.py`: тесты `test_charge_user_and_guest_strips_whitespace_in_allowed_ids`
  / `_async` (про `bot/utils.py::_charge_user_and_guest`) не относятся к B5 (`_budget_user_and_name`)
  — похоже на утечку из другой параллельной задачи (T02, "попутные баги"), т.к. `bot/utils.py`
  не входит во владение файлами T12b.
- `tests/test_callback_authorization.py`: тесты про `allow_group_members_via_authorized_user`
  (`test_group_membership_grant_disabled_...`, `test_group_callback_rejected_...`,
  `test_group_callback_allowed_member_...`) не относятся к B7/mode-group-keyboard — тоже
  похоже на утечку из другой параллельной задачи.

### Итог

0 ERROR, 2 WARNING (один тривиальный lint-fix, один тест-coverage gap без функционального
риска — код перепроверен вручную), 2 NIT. Рекомендация: исправить WARNING #1 (unused import)
обязательно перед сдачей; WARNING #2 по желанию (тест-качество, не блокер).

## Раунд 2

Проверены исправления обоих WARNING (NIT не были обязательны к исправлению).

1. **Неиспользуемый импорт `pytest`.** `tests/test_t12b_configured_openai_models.py`
   больше не импортирует `pytest` (файл целиком перечитан — импорт отсутствует, в
   файле только plain `def test_*` без `@pytest.mark`/`async def`, как и раньше).
   `ruff check tests/test_t12b_configured_openai_models.py` — чисто. Исправлено.
2. **Покрытие B1 не соответствовало плану буквально.** В
   `tests/test_t12b_markdown_fallback.py` добавлен
   `test_media_group_execute_sends_markdown_first_then_falls_back_on_bad_request` —
   прямой тест call site 2 (`_process_vision_media_group`'s `_execute`) через
   реальный `bot._process_vision_media_group([item])` с PNG-байтами и
   `BadRequest`-на-первом-чанке/`None`-на-втором, проверяющий тот же
   markdown-then-plain паттерн, что и call site 1. Добавлен также
   `test_no_duplicate_markdown_or_plain_inline_send_remains` — структурная guard-
   проверка (`source.count("def _send_markdown_or_plain(") == 1`,
   `source.count("self._send_markdown_or_plain(") == 2`), в том же стиле, что у
   B2/B3/B4. Исправлено.

Заодно проверены упомянутые в раунде 1 NIT: `_dispatch_session_reset`/
`_dispatch_user_message` (`bot/telegram_bot.py`) теперь имеют докстринги
("Fire the `on_session_reset`/`on_user_message` observer hook with a
`SessionResetPayload`/`UserMessagePayload` built from the given args.") — в стиле
соседних B1/B4/B7 хелперов. NIT #2 (приватный импорт `_budget_user_and_name`) —
не проверялся отдельно, план прямо предписывает такой импорт, замечание раунда 1
было "для протокола", не требовало действия.

**Прогон:** `tests/test_t12b_*.py` (5 файлов) + `tests/test_usage_budget.py` +
`tests/test_plugin_menu_force_reply.py` + `tests/test_callback_authorization.py` —
85 passed (было 83/83 в раунде 1, +2 — новый media-group тест и guard-тест);
полный `tests/ bot/tests/` — 1944 passed, 0 failed, 0 регрессий. `ruff check bot/telegram_bot.py tests/test_t12b_*.py` — чисто.
mypy на `bot/telegram_bot.py`: сравнение по `(файл, код)` с
`/tmp/impl/mypy_before.keep` показывает только уменьшения (`union-attr` 168→164,
`arg-type` 18→15, `assignment` 4→2, `var-annotated` 12→0, `<type>` 2→0) — это
параллельная T13 (mypy-аннотации), не T12b; ни одного нового кода ошибки.

**Итог раунда 2:** 0 ERROR, 0 WARNING, 0 NIT (не блокирующих). Оба замечания
раунда 1 устранены корректно и с тестами.
