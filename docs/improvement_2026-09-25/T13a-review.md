# T13a — ревью: `bot/telegram_bot.py` mypy-типизация

Ревьюер проверял `git diff HEAD -- bot/telegram_bot.py` против `docs/improvement_2026-09-25/T13-plan.md`
(раздел T13a) и мастер-плана. Диапазон вне T13a (правки `_send_markdown_or_plain`,
`_dispatch_session_reset`, `_build_busy_status`, `_send_rich_markdown_if_fits`,
`_budget_user_and_name`, `parse_model_choices`, `_resolve_plugin_command_or_reply`,
`_build_mode_group_keyboard` — это T12b; `retry_after_seconds`/`RetryAfter` — T02) по существу
не оценивался, только на предмет того, что он не пересекается с typing-правками T13a — не
пересекается. Главный вопрос ревью: меняет ли какой-то из typing-фиксов поведение в рантайме,
особенно новые ранние `return` на месте бывших безусловных обращений к `Optional`-атрибутам.

## Раунд 1

### Проверено по каждому пункту плана

- **var-annotated / механика** — не проверялось построчно все 13 мест (низкий риск, чистые
  аннотации типов без логики); `mypy` подтверждает отсутствие новых ошибок этого кода.
- **`require_message`/`require_query`/`require_accessible_message`** (`bot/telegram_bot.py:130-149`)
  — сигнатуры и докстринги соответствуют дизайну плана (`T13-plan.md:87-95,123-128`), кроме
  одного отступления от плана: `require_accessible_message` в проекте реализован через
  `cast()` без рантайм `isinstance`-проверки, тогда как в плане (`T13-plan.md:123-127`) был
  предложен `isinstance(msg, Message)`-guard с `return None`. Сам план (`:135-144`) явно
  оставляет этот выбор на усмотрение исполнителя ("выбор... оставлен исполнителю T13a по
  месту") при условии не глотать тихо реальный `MaybeInaccessibleMessage`-путь — выбор `cast`
  корректен только если такой путь в проекте недостижим; см. NIT №1 ниже про факт. обоснование
  в докстринге.
- **Все call site `require_message(update)` (11 шт.)** перепроверены по одному —
  см. находки ниже. Полный список: `:1220` (help), `:1248` (stats), `:1421` (resend),
  `:1453` (settings), `:2601` (restart), `:2660` (image), `:2714` (tts), `:2777`
  (транскрипция аудио), `:3170` (vision), `:5590` (handle_plugins_menu), `:5765`
  (handle_plugin_menu_args_reply).
  - `:2777` и `:3170` — старый код уже был `if update.edited_message or not update.message:
    return`, новый — `if update.edited_message: return` + `require_message`-guard;
    логически идентично, без изменений.
  - `:5765` — старый код уже был `if not update.message: return`; замена на
    `require_message`-guard без изменений.
  - `:1453` (settings) — **ERROR**, см. ниже.
  - `:1220,1248,1421,2601,2660,2714,5590` — **WARNING** (сгруппировано), см. ниже.
- **Все call site `require_query(update)`/`require_accessible_message(...)`** — проверены.
  Прямые `query = require_query(update)` в `handle_prompt_selection` (`:2472`),
  `handle_busy_message_callback` (`:3830`), `handle_plugin_menu_callback` (`:5634`),
  `handle_session_callback` (`:6013`) — все зарегистрированы как обработчики
  `CallbackQueryHandler`, которые PTB диспетчеризует только когда `update.callback_query`
  истинен (`CallbackQueryHandler.check_update`, проверено по исходнику установленного
  `telegram==22.8`) — ветка `query is None` в рантайме недостижима, это чистая типовая
  узость, поведение не меняется. `handle_callback_inline_query` (`:4890`) использует `assert`
  вместо `if None: return` — тоже поведенчески нейтрально (тот же `CallbackQueryHandler`-
  инвариант, `assert` лишь делает допущение явным). Тот же assert-паттерн — `handle_settings_
  callback` (`:1488-1489`), `handle_plugin_menu_callback` (`:5651-5652`), ещё один
  callback-обработчик (`:5695-5696`): везде `query`/`update.callback_query` уже гарантирован
  предыдущей проверкой, `require_accessible_message`+`assert` лишь делает допущение о
  недостижимости `MaybeInaccessibleMessage`-ветки явным, не меняя обработку. Паттерн
  `update.effective_message or require_accessible_message(...)` (`:5122,5183,5202,5445,5557`)
  — это добавление fallback-ветки, а не её удаление: сначала пробуется `effective_message`,
  `require_accessible_message` — запасной вариант; поведение не сужается.
- **`effective_message` → `message` (локальная переменная) внутри вложенных замыканий**
  (`:3324`, `:4286`) — обе замены прослежены до охватывающей функции: `:3324` внутри
  `vision`-хендлера, где `message = require_message(update)` уже выполнен и логически
  эквивалентен старому guard'у (см. выше); `:4286` внутри `_process_message_locked`
  (`:4111-4118`), где `message = update.message` идёт сразу с `assert message is not None`
  — в старом коде та же строка была `update.message.from_user.id` без guard'а вовсе,
  т.е. падала бы с тем же исходом (необработанное исключение, см. WARNING-группу ниже по
  сути этого паттерна) на 2 строки раньше. В обеих точках `update.message` на момент вызова
  `.reply_text` гарантированно не `None`, и тогда `update.effective_message ==
  update.message` (первый по приоритету в цепочке `effective_message`) — поведение
  идентично.
- **`User | Any | None` (26 шт.)** — точечно не перепроверялось по каждому из 26, но во всех
  просмотренных при построчном чтении диффа (полные 1932 строки, 4 фрагмента) новые `assert`
  везде добавлены на месте прежних безусловных обращений, ни один не встретился как замена
  РАБОТАВШЕГО fallback-пути (кроме кейса `settings`, который использует не `User`, а
  `Message`-guard — см. ERROR).
- **Переименование `_edit` → `_run_image_edit`** (`:4173`, внутри `_process_message_locked`)
  — единственный call site внутри той же функции обновлён (`:4176`), других ссылок на старое
  имя в файле нет (проверено). Соответствует `T13-plan.md:156-167`.
- **`_AuthWrappedCallback` Protocol** (`:152-155`) — используется в `_wrap_authorized_callback`
  (`:5131-5156`) через `cast(_AuthWrappedCallback, authorized_callback)._chatgpt_auth_wrapped =
  True`; заменяет прежние два `attr-defined`-игнора на явную типизацию маркерного атрибута,
  без изменения логики (`getattr(callback, "_chatgpt_auth_wrapped", False)` не тронут).
- **`# type: ignore[union-attr]`** — единственный (`:2788`, `filename = message.
  effective_attachment.file_unique_id  # type: ignore[union-attr]`) сопровождён комментарием
  и подтверждён по регистрации `MessageHandler`-фильтров хендлера (AUDIO/VOICE/
  Document.AUDIO/VIDEO/VIDEO_NOTE/Document.VIDEO — все ветки имеют `file_unique_id`).
  Обоснован. Других игноров в файле не добавлено; бланкетных `# type: ignore` нет.
- **`cast()`** — `require_accessible_message` (`:149`), `_UpdateProxy` в хелпере
  `_wrap_update_with_message` (`:5583`, дак-тайпинг-прокси без общего базового класса с
  `Update` — обоснованно, делегирует через `__getattr__`), `_AuthWrappedCallback` (`:5155`).
  Все поведенчески нейтральны (`cast` — чистая типовая аннотация без рантайм-эффекта).
- **`reset()`** (`:2269-2303`) — контрольный пример корректного паттерна: вместо `require_
  message` использует `assert update.effective_message is not None` (`:2299`), то есть
  сохраняет исходную семантику `effective_message`, а не сужает её. Показывает, что
  правильный подход для мест, где раньше действительно использовался `effective_message`,
  был известен и применялся в том же диффе — что делает пропуск в `settings()` (ERROR ниже)
  необъяснимым отклонением, а не системным решением.

### Тесты и статический анализ

- `python3 -m mypy bot/telegram_bot.py --python-executable ~/.venvs/ctb/bin/python
  --ignore-missing-imports` — `Success: no issues found in 1 source file` (было 244 ошибки в
  этом файле в `/tmp/impl/mypy_before.keep`).
- `~/.venvs/ctb/bin/python -m ruff check bot/telegram_bot.py
  tests/test_t13a_require_helpers.py` — чисто.
- `tests/test_t13a_require_helpers.py` — 6/6 passed (юнит-тесты трёх хелперов на дак-тайпинг
  дублёрах; адекватны для чистых аксессоров, но по конструкции не могут поймать регрессию
  на конкретном call site — она обнаружена только построчным чтением диффа).
- Полный прогон `tests/ bot/tests/` — 1950 passed, без регрессий.

### Находки

**ERROR (1):**

1. **`settings()` (`bot/telegram_bot.py:1449-1463`) — потеряна рабочая ветка ответа для
   апдейтов без `update.message`.** До правки функция ни разу не обращалась к
   `update.message`: `_ensure_allowed` (`:1450`) и `_get_user_language_async`/
   `_detect_user_language` (`:282-307`, `:278-280`) читают только `effective_user` через
   `getattr`, а финальный ответ шёл через `await update.effective_message.reply_text(...)`
   — то есть для апдейта без `update.message` (`edited_message`, `channel_post`,
   `business_message` — Telegram Business API все команды шлёт именно через
   `business_message`, не `message`) код реально доходил до рабочего ответа пользователю.
   После правки (`:1453-1455`): `message = require_message(update)` (`update.message` и
   только он, `:130-133`) → `if message is None: return` — то же самое сообщение (например,
   `/settings`, присланное правкой другого сообщения) теперь тихо ничего не отвечает: ни
   лога, ни ответа. `CommandHandler.check_update` диспетчеризует по `update.effective_
   message`, а не `update.message` (проверено по исходнику `telegram==22.8`), так что путь
   реально достижим; ни один из связанных `CommandHandler` (регистрация — `:6320-6346`,
   `settings`/`resend` идут через `_authorized_command_handler` → `_wrap_authorized_callback`,
   `:5131-5163`, тоже без ограничивающих `filters=`) не отсекает такие апдейты, и `application.
   run_polling(close_loop=created_loop)` не передаёт `allowed_updates`.
   Контрольный пример в том же диффе — `reset()` (`:2296-2303`) — правильно сохраняет
   `effective_message` через `assert`, то есть корректный паттерн для этой ситуации в файле
   уже применялся.
   Это прямое нарушение директивы плана (`T13-plan.md:190-195`): "каждый сайт: перепроверить,
   что там уже фактически есть ранний return/эквивалент — если нет, не изобретать новую ветку
   поведения молча, вынести в отчёт как найденный потенциальный баг" — здесь эквивалента не
   было (был реально работавший `effective_message`-путь), а новая тихая ветка добавлена без
   упоминания где-либо в диффе/плане.
   **Исправление:** вернуть `update.effective_message` (например, тем же паттерном, что и в
   `reset()` — `assert update.effective_message is not None`), не `require_message`.

**WARNING (2):**

1. **7 хендлеров получили новую тихую раннюю остановку там, где раньше не было вообще
   никакой guard-ветки** — `help` (`:1220-1222`), `stats` (`:1248-1250`), `resend`
   (`:1421-1423`), `restart` (`:2601-2603`), `image` (`:2660-2662`), `tts` (`:2714-2716`),
   `handle_plugins_menu` (`:5590-5592`). До правки в каждом из них было безусловное
   обращение к `update.message.*`/`message_text(update.message)`, без единой проверки на
   `None` — при `update.message is None` это падало с `AttributeError` необработанным
   исключением, которое ловит только глобальный `error_handler` (`bot/utils.py:650-660`):
   он логирует (`logging.error(...)`) и никогда не отвечает пользователю (первый параметр
   типизирован как `_: object`, не используется). То есть пользователь и раньше ничего не
   получал — в отличие от `settings()` (ERROR выше), здесь пользовательский результат не
   меняется. Меняется другое: раньше падение оставляло `ERROR`-запись в логе, теперь —
   тихий `return` без единой строки в логе, диагностический след пропадает. Формально это
   тоже новая ветка поведения на месте, где эквивалента не было — то же нарушение директивы
   плана (`T13-plan.md:190-195`), просто с более низким практическим риском (нет потери
   рабочего ответа, только потеря лог-сигнала и молчаливое расхождение с планом). Не
   зафиксировано нигде как осознанное отступление.
   **Рекомендация:** не обязательно откатывать (контролируемый ранний `return` сам по себе
   лучше необработанного `AttributeError`), но стоит явно задокументировать это как решение
   T13a (а не молчаливый побочный эффект) и/или добавить `logger.warning(...)` перед `return`,
   чтобы не терять диагностику.
2. **Ветка "документ-как-изображение" в vision-хендлере тихо превращает падение в пропуск**
   (`:3207`): `elif message.document and message.document.mime_type and message.document.
   mime_type.startswith('image/'):` — было `elif update.message.document and update.message.
   document.mime_type.startswith('image/'):` (без проверки `mime_type` на истинность). Если у
   `Document` `mime_type is None` (клиент Telegram иногда не проставляет его), старый код падал
   с `AttributeError` на `.startswith(None)` — тот же класс исхода, что и в WARNING №1
   (необработанное исключение, залогировано, ответа пользователю и так не было). Новый код
   просто не считает вложение картинкой и пропускает сохранение. Не часть `require_message`-
   рефакторинга (это отдельный `union-attr`-фикс на `Optional[str]`), но тот же паттерн — новая
   тихая ветка на месте, где раньше не было условия. Практический риск низкий.

**NIT (1):**

1. **Докстринг `require_accessible_message` (`:141-147`) содержит фактически неверное
   утверждение про PTB.** Текст: "PTB отдаёт MaybeInaccessibleMessage только когда
   Application собран с Defaults(block=False)". Проверено по исходнику установленного
   `telegram==22.8`: `query.message` типизирован как `MaybeInaccessibleMessage | None`
   потому, что Bot API 7.0 может вернуть недоступное (удалённое/устаревшее) сообщение в
   callback-запросе; `Defaults(block=False)` — не связанная с этим настройка конкурентности
   обработчиков. Сам выбор `cast()` без рантайм-проверки от этого не становится неверным
   (в `post_init` проекта `Defaults` с этим параметром не используется, а реальный источник
   `MaybeInaccessibleMessage` — устаревшие callback'и — в проекте отдельно не обрабатывается
   ни до, ни после правки, то есть поведение не меняется в любом случае), но обоснование
   вводит в заблуждение будущего читателя. Утверждение дословно перекочевало из
   `T13-plan.md:142-144` — тот же факт неточен и там.
   **Рекомендация:** поправить докстринг на точную причину (недоступное/устаревшее сообщение
   по Bot API 7.0), без привязки к `Defaults(block=False)`.

### Итог

1 ERROR (settings() теряет рабочий ответ для edited_message/channel_post/business_message —
исправить обязательно), 2 WARNING (7 сайтов с новой тихой веткой на месте отсутствовавшего
guard'а — нарушение директивы плана про раскрытие находок, низкий пользовательский риск;
mime_type-guard в vision — тот же паттерн, другое место), 1 NIT (неточность в докстринге,
унаследована из плана). Статика и тесты чистые: mypy 0 ошибок в файле, ruff чисто, новые
тесты 6/6, полный прогон 1950/1950 без регрессий.
