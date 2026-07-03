# Аудит архитектуры и кода — chatgpt-telegram-bot

Дата: 2026-07-02. Метод: персоны architect + reviewer (skill dev-experts), параллельный
построчный аудит подсистем + сводный архитектурный анализ.

Статус: **в работе** — разделы заполняются по мере поступления результатов.

## Легенда

- **CRITICAL** — баг/уязвимость, ведёт к неверному поведению, потере данных или дыре в безопасности.
- **WARNING** — реальный риск или ошибка при определённых условиях; стоит исправить.
- **NITPICK** — качество кода/поддерживаемость.

Каждая находка: `файл:строка — severity — суть — предлагаемое исправление`.

## Содержание

1. [Сводная архитектурная оценка](#1-сводная-архитектурная-оценка)
2. [Слой Telegram-обработчиков (telegram_bot.py)](#2-слой-telegram-обработчиков)
3. [LLM-пайплайн (openai_helper.py, llm_gateway_client.py)](#3-llm-пайплайн)
4. [Фреймворк плагинов и tool-calls (plugin_manager.py, openai_tool_handler.py)](#4-фреймворк-плагинов-и-tool-calls)
5. [Персистентность (database.py, usage_tracker.py, session_logger.py)](#5-персистентность)
6. [Крупные плагины (agent_tools, skills, hindsight_memory)](#6-крупные-плагины)
7. [Вспомогательные модули и инфраструктура](#7-вспомогательные-модули-и-инфраструктура)
8. [Предложения по улучшению архитектуры](#8-предложения-по-улучшению-архитектуры)

---

## 1. Сводная архитектурная оценка

### Текущая форма системы

Слои (сверху вниз): `ChatGPTTelegramBot` (`bot/telegram_bot.py`) → `OpenAIHelper`
(`bot/openai_helper.py`) → `PluginManager` + `openai_tool_handler` → плагины
(`bot/plugins/*.py`) → `Database` (SQLite). Сборка — вручную в `bot/__main__.py:220-234`,
порядок чувствителен (комментарий `bot/__main__.py:226-229`: `register_plugin_schemas()`
обязан выполниться до `set_openai()`).

Сильные стороны, которые стоит сохранить:

- Явный фреймворк хуков (observers/blocking/mutators/collectors) с частично
  формализованными payload-датаклассами — ядро постепенно очищено от plugin-specific кода,
  граница закреплена тестом `tests/test_no_hardcoded_plugin_refs.py`.
- Выделенный однопоточный DB-executor вместо `asyncio.to_thread` — осознанное решение
  против расползания thread-local соединений.
- Большой корпус целевых тестов (100+ файлов в `tests/`).

### Ключевые архитектурные проблемы (подтверждены чтением кода)

1. **God-модули.** `bot/telegram_bot.py` — 5491 строка / ~172 функции в одном классе;
   `bot/openai_helper.py` — 3822 / ~112; плагины `agent_tools.py` (3932) и `skills.py`
   (3857) сопоставимы с ядром. Любое изменение проходит через файлы-гиганты; ревью и
   merge-конфликты дорожают.
2. **Конфигурация как рассыпанные dict'ы.** В `bot/__main__.py:104-217` собираются три
   пересекающихся словаря (`openai_config` ~50 ключей, `telegram_config` ~35,
   `plugin_config`), при этом `api_key`, `openai_base`, `stream`, `tts_model`,
   `tts_response_format`, `bot_language`, `max_sessions` продублированы в двух словарях.
   Затем `plugin_manager.config.update(openai_config)` (`bot/__main__.py:222`) сливает
   пространства имён. Политика парсинга непоследовательна: `SUMMARY_*` парсятся мягко с
   fallback (`_parse_numeric_env`, `bot/__main__.py:34`), остальные — жёсткие
   `int()/float()`, роняющие процесс при опечатке в env.
3. **Двунаправленные связи вместо слоёв.** `PluginManager` и `OpenAIHelper` ссылаются друг
   на друга (`plugin_manager.set_openai(openai_helper)`, `bot/__main__.py:232`); `helper`
   передаётся в `Plugin.execute()`, т.е. плагины видят весь LLM-фасад целиком.
4. **Слой БД читает env напрямую.** `bot/database.py:21` (`_first_openai_model_from_env`)
   — персистентность знает про `OPENAI_MODEL`; конфиг должен приходить сверху.
5. **Singleton `Database` + ручная передача экземпляра.** `Database()` — синглтон через
   `__new__` (`bot/database.py:39`), но одновременно инжектится параметром в
   `PluginManager`/`OpenAIHelper`/бота — двойственность: тесты вынуждены дергать
   `_reset_singleton`, а инжекция иллюзорна (все получают один глобальный объект).

### Сквозные темы (проявились во всех подсистемах)

6. **Синхронный I/O на event loop — системная болезнь, не набор точечных промахов.**
   ~30 sync-вызовов Database из хендлеров, SQLite-backed UsageTracker с legacy JSON snapshot
   на каждый usage event, синхронная генерация HTML с subprocess в `send_long_response_as_file`,
   PIL/base64 в подсчёте токенов. При одном event loop и `concurrent_updates(True)` каждое
   из этих мест останавливает всех пользователей сразу; usage corruption/growth уже закрыты,
   но async/offload доля остаётся архитектурным долгом.
7. **Дублирование пайплайнов с дрейфом поведения.** 4 копии chat-пайплайна в
   openai_helper (мутаторы применяются только в одной!), 3 копии стримингового
   state-machine в telegram_bot, 3 пути доставки direct_result, 3 места нормализации
   результата инструмента, 3 копии mermaid-восстановления в html_utils. Копии уже
   разошлись — это не риск, а свершившийся факт.
8. **Мёртвый провайдер-слой.** Все кортежи `bot/model_constants.py:19-30` пусты; десятки
   веток-потребителей мертвы и накапливают latent-баги (см. `:1334` — сломанный
   stream-флаг в мёртвой ветке).
9. **Ретраи с побочными эффектами.** Tenacity поверх SDK-ретраев, тело ретрая включает
   запись в историю/БД (дублирование user-сообщений), retry-циклы не восстанавливают
   состояние на последней попытке.
10. **Граница «ядро ↔ плагины» продырявлена с обеих сторон.** telegram_bot мутирует
    приватные поля OpenAIHelper; openai_tool_handler зовёт приватные методы agent_tools;
    контроль границы (test_no_hardcoded_plugin_refs) не покрывает openai_tool_handler.

### Состояние тестов

Полный прогон (2026-07-02): **7 failed, 931 passed, 1 skipped** + 1 подтверждённый
флаки-тест. Детали — в разделе 7.

## 2. Слой Telegram-обработчиков

### Архитектурные наблюдения

- **God-object на транспортном слое.** `ChatGPTTelegramBot` — ~5490 строк, в одном классе:
  Telegram-роутинг, UI-меню (settings/sessions/plugins), буферизация сообщений,
  собственная модель конкурентности (conversation locks + inflight/parallel sessions),
  учёт usage, диспатч плагинных хуков и доставка direct_result.
- **Нарушение слоёв: бот лезет во внутренности OpenAIHelper.** Прямые мутации приватного
  состояния: `self.openai.conversations[...]`, `loaded_conversation_sessions[...]`
  (`bot/telegram_bot.py:1952-1953,1968-1969,5321-5322,5334-5335`), вызовы приватных
  методов через getattr: `_save_conversation_context` (`:1972`),
  `_messages_without_image_payloads` (`:1947`), `_with_chat_state` (`:3543`),
  `_clear_chat_state` (`:3182`). Инвариант «контекст сессии консистентен» размазан по двум
  классам.
- **Sync-Database в async-слое как системная проблема:** синхронный API БД зовётся с event
  loop в ~30 местах, `asyncio.to_thread` — только дважды (`:1983`, `:5399`). Один
  медленный fsync замораживает всех пользователей; `concurrent_updates(True)` лишь
  маскирует проблему.
- **Стриминговый state-machine скопипащен трижды** (vision `:2740-2836`, chat
  `:3735-3858`, inline `:4246-4301`) с расходящимися ветками backoff/чанкования.
- **Определение chat-mode по сравнению полного текста prompt_start**, хотя в системное
  сообщение уже пишется `mode_key` (`:1959`); блок скопирован трижды (`:908-921`,
  `:1707-1717`, `:1781-1792`) — O(режимы × сессии) сравнений длинных строк на каждый
  `/reset` и `/stats`.
- **Авторизация — opt-in в каждом хендлере**, а не централизованный слой
  (middleware/TypeHandler). Пропуски неизбежны — см. CRITICAL ниже.
- **Три перекрывающихся механизма координации:** conversation-локи в
  `WeakValueDictionary`, буфер сообщений с таймерами + сторожевой поллер, и busy-wait
  `_wait_until_conversation_idle` (шаг 0.1s). Корректность держится на негласном «в
  критических секциях нет await».

### Находки

- `bot/telegram_bot.py:2270` — **CRITICAL** — [баг/NameError]
  `record_transcription_seconds(..., audio_track.duration_seconds)` — переменная
  `audio_track` не определена нигде в файле (единственное вхождение — подтверждено grep;
  при рефакторинге в `asyncio.to_thread` переменная стала локальной `track` внутри
  `_convert_audio`, `:2238`). Каждая успешная транскрипция голосового падает с NameError:
  пользователь получает «transcribe_fail», ответ не генерируется, usage не пишется —
  голосовой флоу мёртв. Fix: возвращать duration из `_convert_audio`.
- `bot/telegram_bot.py:4176-4235` — **CRITICAL** — [авторизация]
  `handle_callback_inline_query` (pattern `^gpt:`) не вызывает ни `is_allowed`, ни
  `check_allowed_and_within_budget` — единственный callback-хендлер без проверки
  (подтверждено grep: ближайшая проверка `:4140` — из другого метода). Кнопку под
  inline-результатом может нажать любой участник чата: запускается `get_chat_response`
  (`:4312`) за счёт API-ключа владельца, в обход allow-list и бюджета. Fix: проверка в
  начале хендлера (`is_allowed` уже умеет callback_query — `bot/utils.py:482`).
- `bot/telegram_bot.py:3150-3158` — WARNING — [asyncio] задача
  `_run_pending_busy_message_in_new_session` создаётся голым `asyncio.create_task` без
  сохранения ссылки — GC-риск, ради которого написан `self._track_task` (`:174-179`).
- `bot/telegram_bot.py:3074-3079` — WARNING — [блокировка] `_store_pending_busy_message`
  делает синхронный DB-вызов под глобальным `buffer_lock`: блокируется и event loop, и
  приём сообщений всех чатов.
- `bot/telegram_bot.py:3570` и ещё ряд мест — WARNING — [блокировка] sync-вызовы БД в
  горячем пути. Частично закрыто в P1-A (2026-07-02): core async facade уже требует
  `*_async`, `reset_chat_history`/pre-prune закреплены тестами, `agent_tools.run_subagents`
  использует `get_current_model_async`, `haiper_image_to_video` использует
  `get_user_images_async` во всех 5 async image-read paths. Остаточный долг:
  синхронные compatibility helpers (`_get_user_language`, public `get_current_model`),
  lifecycle `shutdown`, UsageTracker offload и отдельные Telegram hot-path вызовы, которые
  нужно закрывать точечно с тестами.
- `bot/telegram_bot.py:879,2259,2516,2717,3628,3937,4200` — WARNING — [блокировка]
  `UsageTracker` пишет SQLite event и legacy snapshot синхронно на event loop на каждый
  запрос (corruption/growth закрыты; async/offload остаётся долгом, см. раздел 5).
- `bot/telegram_bot.py:2015` (также `:846,869,877,1062`) — WARNING — [edited_message]
  `restart`/`stats`/`resend`/`help` обращаются к `update.message.*` напрямую, но дефолтный
  фильтр CommandHandler включает EDITED_MESSAGE (проверено в установленном PTB):
  редактирование старого сообщения-команды даёт `update.message = None` → AttributeError.
  Fix: ранний guard, как в `prompt` (`:2920`).
- `bot/telegram_bot.py:2707-2717` — WARNING — [поток управления] в `vision` при падении
  конвертации изображения пользователю отправляется `media_type_fail`, но нет `return` —
  дальше в модель уходит пустой файл (в `transcribe` `:2255` return есть).
- `bot/telegram_bot.py:3633` — WARNING — [баг] `_image_edit_source_file_id(update)` без
  `user_id`/`chat_id` → `_active_image_file_id(None, None)` всегда None: «отредактируй
  последнюю картинку» без reply матчится интентом, но источник не находится. Fix:
  передавать id, как в describe-ветке (`:3643`).
- `bot/telegram_bot.py:3897` (также `:2330`) — WARNING — [группы]
  `list_user_sessions(user_id, ...)` — сессии в группах ключуются по chat_id
  (conversation_key), запрос по личному user_id читает чужой/пустой список. Fix:
  `conversation_key`.
- `bot/telegram_bot.py:5325-5336` — WARNING — [race] удаление сессии по callback идёт под
  локом `conversation_key`, а параллельные busy-задачи — под `(conversation_key,
  session_id)`; защита `_protected_session_ids` применяется только к автопрунингу
  (`:3131-3137`) — ручной delete может удалить сессию, в которую пишет параллельная
  задача. Fix: проверять protected в ветке delete.
- `bot/telegram_bot.py:4754` — WARNING — [race/shared state] `handle_plugins_menu`
  перезаписывает общий `self.plugin_menu_entries` списком, отфильтрованным по
  disabled-плагинам текущего пользователя — другой пользователь через фолбэк
  `_resolve_menu_entries` (`:5029-5034`) получает чужое меню, индексы `cmd_id` в
  callback_data съезжают на другие команды. Fix: только per-user снапшоты (и чистить их —
  `_user_plugin_menu_entries` растёт неограниченно).
- `bot/telegram_bot.py:5212-5217` — WARNING — [callback] `data[1]`/`data[2]` без проверки
  длины, ветка `preview` — до try/except (`:5278`): callback_data `"session"` даёт
  IndexError без ответа пользователю.
- `bot/telegram_bot.py:5239-5275` — WARNING — [лимит 4096] preview сессии не ограничен по
  суммарной длине → BadRequest на `edit_message_text` (`:5272`), не перехваченный. Fix:
  обрезка до ~4000 символов.
- `bot/telegram_bot.py:1036` — WARNING — [parse_mode] `/stats` шлётся с MARKDOWN; имя
  режима из yml (`:918`), модель (`:902`) и плагинные stats-фрагменты (`:1027-1034`) не
  экранированы — любой `_`/`*` в них → BadRequest, статистика не приходит вовсе. Fix:
  экранирование или plain-text retry (как `:791-796`).
- `bot/telegram_bot.py:5086-5094` — WARNING — [shutdown] `cleanup()` отменяет ВСЕ задачи
  loop (`asyncio.all_tasks()`) из `_post_shutdown` — т.е. в середине штатного shutdown
  PTB, включая внутренние задачи Application и плагинные finalize. Fix: отменять только
  собственные (`_background_tasks`, `_transient_tasks`).
- `bot/telegram_bot.py:5487-5491` — WARNING — [мёртвый код] finally после `run_polling()`:
  PTB с `close_loop=True` уже закрыл loop → ветка не выполняется; а выполнилась бы — дала
  второй cleanup/RuntimeError. Fix: убрать (cleanup уже в post_shutdown).
- `bot/telegram_bot.py:4562` — WARNING — [callback-роутинг] паттерн
  `"^prompt|promptgroup|promptback"`: якорь относится только к первой альтернативе —
  матчится всё, что СОДЕРЖИТ `promptgroup`/`promptback` или начинается с `prompt`
  (например плагинная кнопка `promptfoo:x`). Fix: `r"^(prompt|promptgroup|promptback)(:|$)"`.
- `bot/telegram_bot.py:3046-3049` — WARNING — [liveness] `_wait_until_conversation_idle` —
  бесконечный busy-poll 0.1s без таймаута: залипший лок = буфер чата не обрабатывается
  никогда. Fix: таймаут + ожидание лока вместо поллинга.
- `bot/telegram_bot.py:4369` + `bot/utils.py:499-503` — WARNING — [perf/группы]
  `is_allowed` в группах делает `get_chat_member` по каждому id из allow+admin списков на
  каждое сообщение — до N сетевых RTT до начала обработки. Fix: кэш membership с TTL.
- `bot/telegram_bot.py:155,166` — WARNING — [память] `self.usage` и
  `_user_plugin_menu_entries` — неограниченные dict'ы (в отличие от `last_message` =
  `_BoundedLRU(1024)`). Fix: тот же LRU.
- `bot/telegram_bot.py:824-846` — WARNING — [авторизация] `help` (он же `/start`) — без
  `is_allowed`: посторонний получает полный список команд и help всех плагинов. Fix:
  проверка или осознанное сокращение выдачи.
- `bot/telegram_bot.py:2197` — NITPICK — временные файлы транскрипции в `bot/temp` внутри
  пакета; имя — `file_unique_id` в общей директории: один файл в двух чатах одновременно —
  коллизия путей (`:2354-2357`). Fix: `tempfile.mkdtemp()` на запрос.
- `bot/telegram_bot.py:57` — NITPICK — `WAITING_PROMPT = 1` затеняет импорт из
  haiper-плагина (`:39`); неиспользуемые `import subprocess` (`:2037`) и
  `plugin_command_index` (`:164`).
- `bot/telegram_bot.py:1080` — NITPICK — `update.message._unfrozen()` — приватный API PTB;
  `/resend` в группе повторяет последний промпт любого участника от имени нажавшего.
- `bot/telegram_bot.py:1121,4785,5281` — NITPICK — `query.message.delete()` без guard:
  сообщения старше 48 ч / None → необработанный BadRequest.
- `bot/telegram_bot.py:1490,1531` — NITPICK — выбор TTS-модели/голоса по индексу живого
  списка из API: список изменился между отрисовкой и кликом → молча сохранится не то.
  Fix: имя/хеш значения в callback_data.
- `bot/telegram_bot.py:3575-3952` и др. — NITPICK — god-функции:
  `_process_message_locked` ~380 строк, `vision` ~340 (вложенность 5 уровней), `transcribe`
  ~200, `reset` ~195, `_handle_session_callback_locked` ~220. Copy-paste:
  `_image_edit_source_file_id`/`_image_description_source_file_id` идентичны (`:712-736`);
  двойной `set_my_commands` в post_init.
- `bot/telegram_bot.py:2291,113` и др. — NITPICK — магические числа: 60000 (порог
  big_context), 600 (TTL busy), `backoff += 5` в 6 местах, 200 (preview), 100
  (inline-кеш). Fix: именованные константы.
- `bot/telegram_bot.py:5126` — NITPICK — [i18n] задачи `process_buffer` из фонового
  поллера наследуют contextvar языка фоновой задачи, а не пользователя — ответы этого пути
  уходят на дефолтном языке.

## 3. LLM-пайплайн

### Архитектурные наблюдения

- **God-модуль:** `bot/openai_helper.py` (3823 строки) совмещает 4 копии одного пайплайна
  (non-stream chat `:711-866`, stream chat `:934-1049`, vision non-stream `:2387-2503`,
  vision stream `:2535-2614`) плюс TTS/транскрипцию/изображения/суммаризацию/именование
  сессий/skills-gate. Копии разошлись поведенчески: retry пустых ответов есть только в
  non-stream chat; VISION_MAX_ATTEMPTS — только в non-stream vision; **мутаторы
  (`_apply_before_chat_request_mutators`) применяются только в chat-пути** — vision-запрос
  (`:2229`) собирает messages напрямую, т.е. hindsight-память и planning-префикс
  agent_tools в vision-запросы не попадают. Кандидаты на выделение: суммаризация
  (`:3229-3479`), vision (`:2143-2628`), бюджетирование токенов (`:3662-3756`).
- **Тройная система идентификаторов** chat_id / user_id / state_key размазана по коду:
  `reset_chat_history(chat_id)` создаёт сессию по chat_id как user_id (`:3048`), `ask()`
  пишет в `conversations[user_id]` (`:630`), `_repair_tool_call_history` принимает
  state_key в параметре с именем chat_id (`:2753`). Единый объект ConversationState снял
  бы неоднозначность.
- **Пер-запросное состояние в разделяемых dict'ах:** `_chat_request_models`,
  `_chat_request_extra_tokens`, `_gate_fired` ключуются по state_key на инстансе, а не в
  ContextVar/объекте запроса. Работает, только пока `_chat_lock` сериализует запросы;
  режим bypass (`_without_chat_lock`, `:2912-2918`) ломает инвариант — параллельные ходы
  одного chat_id воруют друг у друга `.pop(state_key)` (`:739`, `:986`).
- **Мёртвый слой провайдеров:** все кортежи в `bot/model_constants.py:19-30` пусты (не
  только GOOGLE — также ANTHROPIC/MISTRALAI/O_MODELS/GPT_4O_MODELS), ветки-потребители в
  `:1333-1351`, `:1357`, `:1375`, `:1791`, `:1872`, `:1886`, `:2243`, `:3126-3140`,
  `:3742` мертвы и накапливают latent-баги.
- **Избыточные DB-обращения на ход:** streaming-ход делает 4+ синхронных
  `get_conversation_context`; `resolve_allowed_plugins` вызывается и внутри
  `__common_get_chat_response` (`:1354`), и снаружи (`:744`/`:989`) — результат не
  переиспользуется; полная перезапись JSON-контекста на каждый append (`:3171`).

### Находки

- `bot/openai_helper.py:1196` — **CRITICAL** — [ретраи/история] tenacity-retry на
  `__common_get_chat_response` (3 попытки по RateLimitError, wait 20s) переисполняет всё
  тело метода, включая `__add_to_history(role="user")` на `:1249`: каждая повторная
  попытка дублирует сообщение пользователя в истории и персистит его в БД. То же для
  vision: декоратор `:2137-2142` + `__add_to_history` на `:2184`. (Подтверждено прямым
  чтением `:1196-1249`.) Fix: ретраить только сам LLM-вызов, либо добавлять query в
  историю после успешного create.
- `bot/openai_helper.py:227` — WARNING — [ретраи] SDK-клиент с `max_retries=3` ПОВЕРХ
  tenacity `stop_after_attempt(3)` + `wait_fixed(20)`: до 9 сетевых попыток и 40+ секунд
  ожидания, всё это время удерживается пер-чатовый lock. Fix: один уровень ретраев.
- `bot/openai_helper.py:422` — WARNING — [логи/PII] `logger.info(...payload=%s...)` пишет
  в INFO полный payload каждого запроса: всю историю и base64 data-URL изображений
  (мегабайты на строку). Fix: усечение или DEBUG.
- `bot/openai_helper.py:1151` — WARNING — [ошибки] `_maybe_apply_auto_chat_mode` не
  обёрнут в try/except: сбой роутер-вызова абортирует весь ход и мислейблится как
  «Configuration error» (`:1438-1441`). Fix: fallback «режим не менять».
- `bot/openai_helper.py:1334` — WARNING — [мёртвая ветка + latent-баг] ветка
  `O_MODELS + ANTHROPIC + ...` мертва, но содержит баг: локальный `stream = False` не
  обновляет `common_args['stream']` (`:1329`) — при наполнении кортежей API получит
  stream=True, а код обработает ответ как non-stream. Плюс дублирующий else `:1341-1351`
  и два идентичных возврата `:1421-1427`.
- `bot/openai_helper.py:1232-1233` — RESOLVED — [мёртвое состояние]
  `conversations_vision` удалён; обычный chat-flow всегда выбирает chat-модель через
  `get_current_model_async`, а прямой vision-flow по-прежнему использует `vision_model`.
- `bot/openai_helper.py:726` — RESOLVED — [race/мёртвый код] transient image
  instance-attribute удалён; живой словарь и публичные `set_...`/`get_...` методы
  сохранены для Telegram image flow.
- `bot/openai_helper.py:172` — WARNING — [конфиг] `default_max_tokens` возвращает
  захардкоженные 200_000 для любой модели кроме трёх llmgateway-алиасов: для модели с
  окном 128k проверка переполнения (`:1263`) не сработает до 400-ошибки API, суммаризация
  не запустится вовремя. Fix: конфигурируемый маппинг model→context_window.
- `bot/openai_helper.py:1306` — WARNING — [логика] переключение на `big_model_to_use` по
  `token_count > max_tokens` сравнивает размер КОНТЕКСТА с бюджетом ГЕНЕРАЦИИ: при
  `output_max_tokens=65535` диалог длиннее ~65k токенов навсегда уезжает на big-модель.
  Fix: сравнивать с окном контекста модели минус резерв.
- `bot/openai_helper.py:3737` — WARNING — [конфиг] в `get_max_tokens` процент применяется
  к полному окну контекста, затем клампится `output_max_tokens` (`:3746`): при окне 200k
  любой `max_tokens_percent ≥ 33` — no-op, семантика «процент от выхода» не выполняется.
- `bot/openai_helper.py:1477` — WARNING — [конфиг] `_uses_structured_tool_history` не
  включает `vision_model`/`big_model_to_use` из отдельных env — для них tool-результаты
  пишутся legacy-ветками, включая устаревшую роль `"function"` (`:3142-3147`), которую
  современные API отклоняют.
- `bot/openai_helper.py:630` — WARNING — [race] `ask()` мутирует `conversations[user_id]`
  и персистит контекст без `_chat_lock` — гонка с параллельными locked-путями того же
  ключа (плагины зовут `ask` во время активного хода). Аналогично
  `record_plugin_exchange` (`:3180-3227`) и `get_conversation_stats` (`:351-365`).
- `bot/openai_helper.py:594` — WARNING — [мёртвая работа] `model or
  self.get_current_model(user_id)` — синхронный DB-запрос, результат безусловно
  перезаписывается на `:618-621`.
- `bot/openai_helper.py:758,790` — WARNING — [дублирование] два почти идентичных блока
  retry пустого ответа внутри `_get_chat_response_locked` — уже разошлись (первый не
  передаёт `retry_plain_text_tool_intent`).
- `bot/openai_helper.py:982,1006,1049` — WARNING — [ошибки] stream-генератор превращает
  исключения в текстовые yield'ы `f"Error: ..."`: вызывающий не отличает ошибку от ответа
  модели, строка уходит пользователю, в истории остаётся dangling user-message. Fix:
  пробрасывать или typed-маркер в протоколе чанков.
- `bot/openai_helper.py:1163-1171` — WARNING — [порча данных] `_maybe_apply_auto_chat_mode`
  безусловно перезаписывает `conversations[state_key][0]`: для live-сессии без
  system-сообщения (случай задокументирован на `:1079-1086`) первое сообщение будет
  уничтожено. Fix: перезапись только если `[0].role == 'system'`.
- `bot/openai_helper.py:3492` — WARNING — [tiktoken]
  `encoding_for_model("llmgateway/...")` всегда KeyError → cl100k_base для всех рабочих
  моделей — оценки для Claude/Gemini систематически смещены (влияет на триггер
  суммаризации и usage). Плюс `__count_tokens` декодирует base64 и открывает PIL для
  каждой картинки в истории (`:3513-3575`) и вызывается ≥3 раз за ход. Fix: кэш токенов по
  хэшу сообщения.
- `bot/openai_helper.py:2455` — NITPICK — [vision] retry-цикл восстанавливает snapshot
  состояния только при `attempt < VISION_MAX_ATTEMPTS`: после финальной неудачи в
  истории/БД остаётся мусор последней попытки.
- `bot/openai_helper.py:2189,2232,2535` — NITPICK — [vision] рассинхрон с chat-путём: нет
  маржи 0.95, `max_tokens` не клампится, stream без ретраев.
- `bot/openai_helper.py:220` — NITPICK — `openai.api_base = ...` — legacy-глобал
  openai<1.0, мёртвый при openai>=2.14.
- `bot/openai_helper.py:1325,1344,2474` — NITPICK — `n_choices`: non-stream else-ветка
  включает его, stream и vision всегда n=1, обработчик множественных choices наполовину
  мёртв. Определиться.
- `bot/openai_helper.py:1013` — NITPICK — дублирующая проверка пустых chunk.choices.
- `bot/openai_helper.py:642` — NITPICK — `response.usage.total_tokens` в `ask()` без
  защиты (в других местах `_response_total_tokens`).
- `bot/openai_helper.py:2935` — NITPICK — [локи] `_clear_chat_state` удаляет
  `_per_chat_locks[chat_id]` не проверяя, удерживается ли lock — поздний реентри создаст
  второй lock, две корутины войдут одновременно. Fix: не evict'ить удерживаемый lock.
- `bot/openai_helper.py:2089` — NITPICK — аннотация `tuple[any, int]` — builtin `any`
  вместо `Any`.
- `bot/llm_gateway_client.py:41,60,88` — WARNING — [ошибки] `get_json`/`post_json`/
  `post_multipart` не оборачивают httpx.TimeoutException/ConnectError в LLMGatewayError —
  контракт клиента нарушен, ретраев нет. Fix: `except httpx.HTTPError → raise
  LLMGatewayError(...) from exc`.
- `bot/llm_gateway_client.py:43,62,90` — NITPICK — `detail = response.text` без усечения:
  HTML-страница ошибки гейтвея целиком уходит в сообщение исключения и далее в Telegram.
- `bot/llm_gateway_client.py:172` — NITPICK — `web_deep_research` с timeout=2500s на общем
  AsyncClient — зависший запрос держит соединение 40+ минут.
- `bot/model_constants.py:19-30` — WARNING — [конфиг] все провайдер-кортежи пусты —
  мёртвые ветки-потребители по всему пайплайну (см. наблюдения). Fix: удалить слой или
  наполнить из env.
- `bot/chat_modes_registry.py:33-36` — WARNING — [ошибки] `_load_if_needed` не защищает
  `yaml.safe_load`: синтаксическая ошибка при горячем редактировании chat_modes.yml роняет
  активные запросы. Fix: try/except с сохранением предыдущего `_data`.
- `bot/chat_modes_registry.py:47` — NITPICK — `get_mode_by_system_prompt` — линейный
  проход с сравнением многокилобайтных prompt'ов 2-3 раза за ход; `strip()` упадёт при
  content=None. Fix: кэш prompt→mode; guard.
- `bot/chat_modes_registry.py:39,63,70` — NITPICK — `all_modes()` отдаёт внутренний
  мутабельный dict; isinstance-проверки mode_data неконсистентны.
- `bot/chat_modes.yml:2` — NITPICK — шапка-комментарий перечисляет несуществующие tools
  `wolfram`, `deepl` (реальные: `wolfram_alpha`, `ddg_translate`); активные tools-списки
  валидны.

## 4. Фреймворк плагинов и tool-calls

### Архитектурные наблюдения

- **Ядро `openai_tool_handler.py` насквозь прошито знанием об agent_tools:** константы
  DELIVERY_TOOL_NAME/MANAGE_PLAN_TOOL_NAME/ASK_USER_TOOL_NAME
  (`bot/openai_tool_handler.py:161-164`), прямые вызовы приватных методов плагина
  `_current_in_progress_task_id`/`_record_tool_outcome` (`:1144-1151`, `:1286-1291`),
  delivery-contract state machine и repair-промпты. Граница «ядро не знает про плагины»
  обходится тем, что openai_tool_handler.py не входит в контролируемый список
  test_no_hardcoded_plugin_refs. agent_tools — де-факто часть ядра, а не плагин.
- **Контракт результата инструмента не двойной, а тройной+:** `{"error"}` /
  `{"success": false}` / `{"ok": false}` / direct_result / голые строки. Нормализация
  размазана по трём местам — `bot/tool_result.py:98-121`, `bot/plugin_manager.py:452-456`,
  `bot/openai_helper.py:1516-1522` — и они уже разъехались (см. находку про `ok: false`).
  Типизированного результата на границе `Plugin.execute` нет.
- **`handle_function_call` — ~570-строчная рекурсивная функция с 18 параметрами;**
  состояние размазано между мутируемыми аргументами, замыканиями и полями helper.
  Инварианты непроверяемы локально — прямое следствие: CRITICAL про мёртвый лимит ниже.
  Рекурсия вместо цикла наращивает стек на каждый реентри.
- **Policy enforcement рассредоточен:** user-disabled плагины учитываются в
  hook-диспетчере (`bot/plugin_manager.py:967-993`) и в `openai_helper.py:1058`, но НЕ в
  `get_functions_specs` (`:294`, нет user_id) и НЕ в `_guard_tool_call` (`:479`);
  chat-mode allow-list — в handler; subagent-блокировки — в agent_tools. Единой точки
  «можно ли этому пользователю этот инструмент» нет.
- **Резолв «имя функции → плагин/спек» не индексирован:** полный проход по всем плагинам с
  get_spec() на каждый вызов, ≥4 раза на один tool call. Мёртвый полиморфизм Google-ветки
  расползся по трём файлам.

### Находки

- `bot/openai_tool_handler.py:1184` — **CRITICAL** — [лимиты] `if tool_result.success:
  final_delivery_required = True` для ЛЮБОГО успешного инструмента, а `_reentry_tool_choice`
  (`:641-646`) при `final_delivery_required` возвращает `"auto"` ДО проверки
  `times < max_consecutive_calls`. После первого же успешного tool call лимит
  `functions_max_consecutive_calls` (`:1337`) мёртв — реентри-цикл (рекурсия
  `:1356-1375`) не ограничен ничем, кроме желания модели остановиться: неограниченный
  расход токенов + рост стека. (Подтверждено прямым чтением `:641-646` и `:1184-1185`.)
  Fix: взводить флаг только когда delivery-контракт действительно требуется, и жёсткий
  верхний предел times всегда.
- `bot/openai_tool_handler.py:1097-1101` — WARNING — [безопасность] инъекция перекрывает
  model-supplied `chat_id`/`user_id`, но `message_id` перезаписывается только при
  `request_context.message_id is not None`, а в ветке `request_context is None` — никогда.
  При этом `message_id` в FRAMEWORK_TOOL_ARGS и исключён из schema-валидации — модель
  может подсунуть произвольный message_id, который плагины считают доверенным
  (reply/edit чужого сообщения). Аналогично в subagent-пути
  (`bot/plugins/agent_tools.py:3548` не перезаписывает chat_id при родительском None).
  Fix: всегда `args.pop(...)` до инъекции.
- `bot/plugin_manager.py:437-438` — WARNING — [безопасность]
  `parsed_args['request_context'] = request_context` только при не-None контексте: при
  `request_context=None` модельный аргумент `request_context` проходит валидацию
  (additionalProperties разрешены) и попадает в `plugin.execute(**kwargs)` как
  псевдо-доверенный контекст. Fix: безусловный pop перед инъекцией.
- `bot/plugin_manager.py:424` — WARNING — [валидация] schema-валидация под `if spec:` —
  если `get_spec_by_function_name` вернул None, вызов исполняется вообще без валидации,
  молча. Fix: при найденном плагине и spec=None — ошибка.
- `bot/openai_tool_handler.py:1173` — WARNING — [gather] `isinstance(tool_response,
  Exception)` не ловит `asyncio.CancelledError` (наследник BaseException): отменённый таск
  из `gather(..., return_exceptions=True)` (`:110`) уходит в normalize → success=True,
  content='""' — отмена записывается как УСПЕШНЫЙ пустой результат и вдобавок взводит
  final_delivery_required. Fix: `BaseException` + re-raise CancelledError.
- `bot/openai_helper.py:1608-1611` — WARNING — [сжатие истории]
  `_compact_old_tool_results_history` сжимает только `role == "tool"`; результаты legacy
  веток (`role: "user"` `:3126-3128`, `role: "assistant"` `:3136-3140`) не сжимаются
  никогда. Компакция вызывается только между пользовательскими сообщениями — внутри
  tool-цикла не срабатывает; per-result лимита нет — мегабайтный вывод уходит в модель
  целиком. Fix: компакция перед каждым реентри + cap на одиночный результат.
- `bot/openai_helper.py:1517-1522` — WARNING — [двойной контракт] `_tool_result_summary`
  знает `error` и `success is False`, но не `ok is False` (добавлен в
  `bot/tool_result.py:105-106,119-120`, коммит 3925731): при компакции `{"ok": false}`
  суммаризуется как успешный — сигнал ошибки теряется из старой истории. Fix: зеркальная
  ветка.
- `bot/plugin_manager.py:569-594` — WARNING — [perf] `get_plugin_name_by_function_name` —
  полный проход по плагинам с get_spec(), ≥4 раза на tool call (`:404`, `:423`, `:536`,
  handler `:1078`). Fix: индекс canonical_name → (plugin, spec) в load_plugins.
- `bot/plugin_manager.py:479-502` — WARNING — [мёртвый код/политика] `_guard_tool_call`:
  ни один плагин не реализует `guard_tool_call`, метода нет в базовом Plugin, при этом на
  каждый вызов инстанцируются ВСЕ плагины; семантика fail-open (исключение guard'а =
  пропуск). Fix: удалить или объявить в Plugin и решить fail-open/closed осознанно.
- `bot/plugin_manager.py:1020-1028` — WARNING — [хуки] `dispatch_observe` создаёт
  корутины вне try: плагин с синхронным observer или бросающий до первого await роняет
  весь dispatch (`gather(None, ...)` → TypeError), корутины остальных не await'ятся —
  весь батч обсерверов теряется. В остальных диспетчерах вызов внутри try — асимметрия.
  Fix: per-plugin try + `inspect.isawaitable`.
- `bot/openai_tool_handler.py:950-959` — WARNING — [stream] первый чанк без
  `delta.tool_calls` и без `finish_reason == 'tool_calls'` немедленно трактуется как
  plain-text ответ: провайдер, шлющий лидирующий role-only чанк (обычное дело у
  OpenAI-совместимых прокси), ломает tool-flow — сырые tool calls уйдут пользователю
  текстом. Fix: пропускать чанки с пустой delta.
- `bot/plugin_manager.py:257-265` — WARNING — [хрупкость] `register_plugin` берёт
  `plugin_classes[0]` (алфавитно) среди ВСЕХ Plugin-подклассов модуля, включая
  импортированные из других модулей: первый же cross-импорт молча зарегистрирует чужой
  класс. Fix: фильтр `cls.__module__ == module.__name__`.
- `bot/plugin_manager.py:663-671` — WARNING — [жизненный цикл] `get_plugin` перевызывает
  `_call_initialize` на каждое обращение, пока `instance.openai` falsy — многократная
  инициализация неидемпотентных initialize, вопреки контракту «exactly once»
  (`:1222-1223`). Fix: флаг `_initialized`.
- `bot/plugin_manager.py:61` — WARNING — [баг] `config.get('plugins', ...)` на исходном
  аргументе, а не на `self.config`: защита `dict(config or {})` (`:53`) бесполезна — при
  `config=None` AttributeError. Fix: `self.config.get(...)`.
- `bot/plugin_manager.py:453` — NITPICK — детектор direct_result для телеметрии не требует
  `kind`, а `bot/tool_result.py:46` требует — расхождение детекторов. Fix: единый
  `direct_result_payload()`.
- `bot/tool_result.py:101-106` — NITPICK — дыры распознавания legacy-контракта:
  `{"error": ""}` → success (truthiness), `{"success": 0}`/`{"success": "false"}` →
  success (`is False`), `{"status": "error"}` не распознаётся.
- `bot/openai_tool_handler.py:1295-1311` — NITPICK — при повторном сбое в историю
  добавляются ДВА user-сообщения об одном батче (repeated-failure note + reflection note).
- `bot/openai_tool_handler.py:1176,978` — NITPICK — полный `tool_result.content` и
  first_choice на INFO — мегабайты и PII в боевых логах. Fix: preview + DEBUG.
- `bot/openai_tool_handler.py:540-544` — NITPICK — `_merge_direct_results_into_final`
  молча дропает нераспарсившийся direct-result (`except: continue`) — без единого лога.
- `bot/plugins/db_handle.py:84-92` — NITPICK — [контракт] flush батча идёт через
  `db.get_connection()` (deferred BEGIN), а не `Database.transaction()` с BEGIN IMMEDIATE
  — имя `DbHandle.transaction()` обещает семантику, которой нет; спасают `_op_lock` +
  единственный db-поток. Fix: использовать `db.transaction()` или задокументировать.
- `bot/plugins/plugin.py:116,22` — NITPICK — аннотация `-> [Dict]` (list-литерал вместо
  типа); базовая сигнатура `initialize` без `db`/`plugin_config` — из-за этого живёт
  интроспекционная шима `_call_initialize` (`bot/plugin_manager.py:105-139`).
- `bot/plugins/hooks.py:70` — NITPICK — frozen dataclass с мутабельными dict внутри
  (`SessionBeforeDeletePayload.messages`) — заявленная безопасность шаринга поверхностна.
- `bot/openai_tool_handler.py:1100` — NITPICK — `int(chat_id)` может бросить ValueError,
  который не ловится (только JSONDecodeError/TypeError) → падение всего
  handle_function_call вместо ошибки одного вызова.
- `bot/plugin_manager.py:346-347` — NITPICK — мёртвая Google-ветка
  `function_declarations` + хвосты поддержки dict-спеков в трёх файлах
  (`bot/openai_tool_handler.py:367-377,427-428`, `bot/plugin_manager.py:619-620`).
- `bot/openai_tool_handler.py:67,1091` + `bot/plugin_manager.py:421` — NITPICK — одни и те
  же аргументы JSON-парсятся трижды на вызов. Fix: передавать разобранный dict рядом со
  строкой.

## 5. Персистентность

### Архитектурные наблюдения

- Персистентность размазана по трём несогласованным механизмам: SQLite (сессии, настройки,
  телеметрия), legacy JSON-файлы per-user (`usage_tracker` теперь импортирует их в SQLite) и
  JSONL-каталог (`session_logger` — атомарный summary, но неатомарные конкурентные append).
  Самый выгодный шаг — перенести usage_tracker в SQLite (append-only таблица `usage_events`
  + агрегаты): закрывает corruption, гонки при `concurrent_updates(True)` и O(история)
  перезапись файла на каждый запрос.
- Выделенный db-worker (`max_workers=1`) обесценен тем, что sync-методы `Database`
  по-прежнему зовутся прямо из async-хендлеров event-loop-потока
  (`bot/telegram_bot.py:203`, `:1155`, `:2597`): глобальный `RLock` в `get_connection()`
  блокирует event loop на время чужого запроса. Логичное завершение миграции — сделать
  sync-API приватным, наружу только `*_async`/`DbHandle`.
- `conversation_context` хранит весь диалог одним JSON-блобом и целиком перезаписывает его
  на каждое сообщение: для длинных сессий объём записи растёт квадратично, а
  `list_user_sessions` тянет и парсит полные контексты всех сессий даже для UI-списка.
- Две параллельные телеметрии тул-коллов — таблица `tool_call_events` и session_logger
  JSONL — дублируют друг друга, и ни у одной нет retention/ротации.
- Версионирование схемы хрупкое: миграция-1 сверяется с `TARGET_SCHEMA_VERSION` и сама
  пишет в `schema_version` «2» (`bot/database.py:1290`, `:1348-1351`). При появлении версии
  3 цепочка молча сломается. Нужен явный реестр миграций (номер → функция).

### Находки: bot/database.py

- `bot/database.py:48` — WARNING — [жизненный цикл] `cls._instance = instance`
  присваивается ДО `instance.init_db()` (строка 49; подтверждено прямым чтением). Если
  `init_db` кидает, полуинициализированный объект остаётся закэширован; любой последующий
  `Database()` молча вернёт экземпляр без схемы и без повторной попытки. Fix: присваивать
  `_instance` только после успешного `init_db()`.
- `bot/database.py:328` + `:131` — WARNING — [миграция] `migrate_conversation_context`
  вызывается изнутри `with self.get_connection()` в `init_db` (depth=2), поэтому условие
  `depth == 1` в `transaction()` ложно и `BEGIN IMMEDIATE` НЕ выполняется — вопреки
  докстрингу (`:1280-1283`). В legacy-режиме DDL (`ALTER RENAME` :1297, `CREATE` :1298)
  автокоммитятся сразу, а `INSERT...SELECT` ждёт общего commit — окно краха между RENAME и
  commit реально; спасает только эвристика `_recover_from_failed_migration`. Fix: миграцию
  — в собственное соединение/транзакцию вне `get_connection` init_db.
- `bot/database.py:113-115` — WARNING — [corruption] в `get_connection` нет обработки
  ошибки `commit()`: при сбое (disk full) rollback не вызывается, соединение остаётся с
  открытой транзакцией, следующий `get_connection` того же потока молча продолжит её, а его
  commit зафиксирует смесь частичных записей. Fix: try/except вокруг commit с rollback.
- `bot/database.py:101-118` — WARNING — [конкурентность] `_op_lock` держится на всё время
  операции, при этом sync-методы зовутся из event-loop-потока: loop встаёт на
  `_op_lock.acquire()`, пока db-worker выполняет длинный запрос. Прямого deadlock нет, но
  это глобальный стоп бота на каждый sync-вызов. Fix: горячие вызовы — через
  `_run_in_db_thread`.
- `bot/database.py:778` + `:797-826` — WARNING — [corruption/дизайн]
  `get_conversation_context` аннотирован `-> Optional[Dict]`, фактически возвращает
  5-кортеж; любое исключение (включая битый JSON в `json.loads` :814) молча превращается в
  дефолты — вызывающий решит, что контекста нет, и следующий save перезапишет реальную
  историю. Дефолт `max_tokens_percent` расходится: 100 в success-пути (:817) против 80 в
  fallback; read-метод по пути создаёт сессию (:792-794) — запись в getter'е. Fix:
  различать «нет данных» и «ошибка чтения», выровнять дефолты, поправить аннотацию.
- `bot/database.py:260-277` + `:538-555` — WARNING — [рост] `tool_call_events`: только
  INSERT и SELECT, ни одного DELETE/retention во всей кодовой базе — неограниченный рост на
  каждый тул-колл. Fix: фоновая чистка по `created_at` + индекс.
- `bot/database.py:918-921` + `bot/telegram_bot.py:2597` — WARNING — [perf]
  `cleanup_old_images` — DELETE по `created_at` без индекса, вызывается синхронно в
  vision-хендлере на каждое фото: полный скан таблицы в event-loop-потоке. Нет и индексов
  `images(user_id)`. Fix: индексы + чистку в фоновый таск.
- `bot/database.py:1111-1122` — WARNING — [integrity] inline-прунинг в `create_session`
  удаляет сессии без хука `on_session_before_delete`, когда вызывается из
  `save_conversation_context` (:626) или `get_conversation_context` (:794) — plugin-owned
  данные (hindsight и т.п.) не узнают об удалении. Плюс `_oldest_session_ids_for_limit`
  (:995-1001) не исключает активную сессию — может удалить именно её. Fix:
  `exclude_session_ids` + возврат списка удалённых для диспатча хуков.
- `bot/database.py:1162-1191` — WARNING — [perf] `list_user_sessions` тянет колонку
  `context` (полный JSON истории) для ВСЕХ сессий и `json.loads` каждый; используется для
  UI-списков. Fix: лёгкая проекция без context.
- `bot/database.py:1020-1034` — NITPICK — отдельный oldest-session deletion API был
  мёртвым методом без вызовов в рабочем коде. Status: удалён в remediation cleanup.
- `bot/database.py:93` — NITPICK — f-string `PRAGMA journal_mode = {journal_mode}` из env
  без валидации. Fix: whitelist значений.
- `bot/database.py:1289-1291` + `:1348-1351` — NITPICK — миграция-1 пишет в
  `schema_version` сразу 2; при добавлении версии 3 связка молча сломается.
- `bot/database.py:85-97` — NITPICK — `_connection_lock` не защищает общее состояние
  (у `_local` per-thread семантика), зато держится на время `sqlite3.connect`. Fix: убрать.
- `bot/database.py:144-163` — NITPICK — `__del__`/`_reset_singleton` закрывают только
  соединение текущего потока и db-worker'а; thread-local соединения других потоков не
  закрываются никогда.
- `bot/database.py:364` — NITPICK — избыточный `conn.commit()` внутри `get_connection`.
- `bot/database.py:687-690` — NITPICK — `set_session_name` вне транзакции
  `save_conversation_context` — окно гонки с параллельным переименованием (косметика).
- `bot/database.py:973` — NITPICK — `int(os.getenv('MAX_SESSIONS', 5))`: мусор в env даст
  ValueError глубоко внутри `create_session` → невнятная ошибка. Fix: валидация с fallback.
- `bot/database.py:834-836` — NITPICK — `save_user_model`: без активной сессии UPDATE —
  молчаливый no-op. Fix: логировать `rowcount == 0`.
- `bot/database.py:1395,1445` — NITPICK — `export_sessions_to_yaml` аннотирован `-> str`,
  возвращает None при ошибке; пишет в `os.getcwd()/exports`.

### Находки: bot/usage_tracker.py

- `bot/usage_tracker.py:15-260,357-458` — RESOLVED — [corruption/concurrency/growth]
  новые usage events пишутся в `usage.sqlite3` (`usage_events` + `usage_daily_aggregates`),
  legacy JSON импортируется один раз через `usage_imports`, corrupt JSON переименовывается,
  а публичный `self.usage` остался read-compatible snapshot.
- `bot/usage_tracker.py:699-702,727-735` — RESOLVED — [perf]
  `add_current_costs` и `get_current_cost` больше не используют eager `.get(...,
  initialize_all_time_cost())`; cost читается из SQLite-агрегатов.
- `bot/usage_tracker.py:396,411,420` — NITPICK — `initialize_all_time_cost` игнорирует
  `self.prices` и считает по захардкоженным legacy-ценам.
- `bot/usage_tracker.py:329,338` — NITPICK — прямые `["last_update"]`/`["month"] +=` без
  `.get`: legacy-файл без этих ключей даст KeyError.
- `bot/usage_tracker.py:154-171` — NITPICK — `number_images` фиксированной длины 3: при >3
  ценах в конфиге — IndexError по существующей записи; округление до 2 знаков в
  vision/tts/transcription обнуляет микро-стоимости.
- `bot/usage_tracker.py:92` — NITPICK — `mkdir(exist_ok=True)` без `parents=True`.

### Находки: bot/session_logger.py

- `bot/session_logger.py:180-185` — WARNING — [конкурентность/потеря данных] каждый
  `record()` спавнит отдельный to_thread-таск на append в один файл: порядок строк .jsonl
  не гарантирован; ошибки записи молча теряются; backpressure нет. Fix: одна очередь-writer
  (`asyncio.Queue` + один consumer).
- `bot/session_logger.py:125,149-151,198` — WARNING — [рост/память] `_stats` копится по
  (uid, sid) и очищается только в `flush_summary` (единственный вызов —
  `bot/openai_helper.py:2897`); сессии, не дошедшие до flush, живут в памяти вечно;
  .jsonl-файлы не ротируются. Fix: TTL/LRU + ротация.
- `bot/session_logger.py:112` — NITPICK — tmp-файл `path + '.tmp'` общий для конкурентных
  `flush_summary` одной сессии — гонка last-wins на `os.replace`.
- `bot/session_logger.py:132` — NITPICK — `record()` мутирует переданный event
  (`setdefault('ts')`).

### Находки: bot/user_settings.py

- `bot/user_settings.py:26-32` — NITPICK — синхронный `db.get_user_settings` из
  async-хендлеров (`bot/telegram_bot.py:1241`) — та же блокировка event loop через
  `_op_lock`.

## 6. Крупные плагины

### Находки: bot/plugins/mcp_server.py (проверено лично)

- `bot/plugins/mcp_server.py:50-55` — WARNING — [дизайн/тесты] дефолтный путь конфигурации
  — `<repo>/data/mcp_servers.json`: mutable-состояние живёт в дереве репозитория, а
  `__init__` (`:31-32`) сразу делает `os.makedirs` и читает файл — конструктор с
  файловыми сайд-эффектами. Тесты, создающие плагин без `storage_root`, работали бы с
  живым файлом. Fix: путь только через `storage_root`/env, без repo-дефолта; I/O — из
  `initialize()`, не из конструктора.
- `bot/plugins/mcp_server.py:102-118` — WARNING — [надёжность] `load_servers_config`
  глотает любую ошибку парсинга и молча сбрасывает `self.servers = {}`: битый JSON-файл →
  все зарегистрированные серверы «исчезают», а следующий `save_servers_config` перезапишет
  файл пустым — тихая потеря пользовательских регистраций. Fix: при parse-ошибке — backup
  файла и явная ошибка в лог/пользователю, не сброс.
- `bot/plugins/mcp_server.py:584,645` — NITPICK — [контракт] плагин возвращает legacy-форму
  `{"success": True}` / `{"error": ...}`, тогда как ядро мигрировало на контракт
  `tool_response_*` (`bot/tool_result.py`, коммит «обработка ответов с 'ok': false»).
  Смешение двух контрактов результата в одном дереве плагинов.

### Состояние тестов MCP-плагина

5 из 5 падений `bot/tests/test_mcp_server.py` воспроизводятся на чистом прогоне
(`test_register_server` и др., `KeyError: 'success'`): фикстура `mcp_plugin`
(`bot/tests/test_mcp_server.py:17-37`) подсовывает ПУСТОЙ temp-файл конфигурации →
`json.load` падает ещё в setup, дальше поведение расходится с ожиданиями теста. Тесты
рассинхронизированы с текущим кодом — их нужно чинить вместе с контрактом результата.

### Архитектурные наблюдения (agent_tools / skills / hindsight_memory)

- **Паттерн «плагин-переросток».** Каждый из трёх файлов (agent_tools ~3.9k, skills ~3.9k,
  hindsight ~2.6k строк) смешивает 5-7 ответственностей: спеки тулов, execute-диспетчер,
  sync-SQL слой, фоновые воркеры, subprocess/файловые операции, Telegram-UI, хуки. Каждый
  просится на разбиение по слоям (для hindsight: `client`/`store`/`dream`+`finalize`/`ui`).
- **Кросс-поточная мутация plain-dict состояния без единой дисциплины блокировок:** одни и
  те же словари (`_pending_verify`, `_pending_replan`, `_tool_error_streaks`) мутируются то
  из `to_thread`-воркера без блокировки, то на event loop под `_get_replan_lock()`.
- **«Default-open» модель безопасности + доверие к аргументам от LLM:** skills из коробки
  разрешает установки и запуск скриптов всем (`*`), а «подтверждением человека» служит
  булев аргумент `confirmed`, который выставляет сама модель.
- **Reach-through в приватные внутренности ядра и соседних плагинов:** все три плагина
  лезут в `db_handle.database` за сырым sync-Database в обход async-фасада; agent_tools
  дёргает приватные методы skills (`_disabled_skills_for_user`) и использует приватный
  `helper._tool_result_content`.
- **Полное окружение процесса в дочерние subprocess:** и `npx -y skills`, и пользовательские
  скилл-скрипты запускаются с `os.environ.copy()` — все секреты бота (ключи OpenAI/Telegram)
  утекают в сторонний код.

### Находки: bot/plugins/skills.py

- `bot/plugins/skills.py:147,150,152` — **CRITICAL** — [безопасность/default-open] дефолты
  `SKILLS_SCRIPT_ADMIN_USER_IDS="*"`, `SKILLS_ALLOW_INSTALLS=True`,
  `SKILLS_INSTALL_ADMIN_USER_IDS="*"` (подтверждено прямым чтением): из коробки любой
  пользователь ставит скиллы и запускает скрипты. Fix: default-deny.
- `bot/plugins/skills.py:2520,3712` — **CRITICAL** — [безопасность/утечка секретов] и
  `_run_skills_cli` (`npx -y skills`), и запуск скилл-скриптов используют
  `env=os.environ.copy()` (подтверждено grep) — сторонний npm-пакет и каждый дочерний
  скрипт наследуют `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN` и т.д. Fix: минимальный env
  (PATH + явные переменные скилла); вендорить/пиновать CLI.
- `bot/plugins/skills.py:517,553,792,807,1515-1518` — **CRITICAL** — [безопасность] гейт
  «подтверждения человека» — обязательный булев аргумент спеки `confirmed`, который модель
  выставляет сама. Out-of-band подтверждения нет. Fix: подтверждение через Telegram-кнопку
  (callback).
- `bot/plugins/skills.py:1863-1865` — WARNING — [безопасность/LFI]
  `_skill_install_source_kind`: `Path(source).expanduser().exists()` → любой существующий
  локальный путь от LLM классифицируется как `"local"` и читается при установке. Fix:
  ограничить локальные установки базовым каталогом.
- `bot/plugins/skills.py:3404-3405,3438-3448` — WARNING — [безопасность/prompt-injection]
  `_append_skill_reflection` дописывает присланный моделью `proposal_text` в `SKILL.md`;
  текст попадает в инструкции скилла для будущих промптов — самомодифицирующийся канал
  инъекции без ревью человека. Fix: правки SKILL.md только через одобрение.
- `bot/plugins/skills.py:~2420-2469` — WARNING — [безопасность] извлечение архивов: защита
  от traversal/symlink есть, но нет лимита на суммарный распакованный размер/число файлов
  (decompression bomb). Fix: кап на размер и количество.
- `bot/plugins/skills.py:3005` — WARNING — [связность] прямой вызов
  `agent_tools.execute("run_subagents", ...)` — ломается, если agent_tools отключён. Fix:
  lookup по capability + graceful fallback.
- `bot/plugins/skills.py:913-919,3653-3660` — NITPICK — `_ensure_ready`/`_ensure_paths`
  лениво пере-запускают полный `initialize()` из execute-пути.
- `bot/plugins/skills.py:2089-2187` — POSITIVE — SSRF-защита сделана хорошо (IP-pinning +
  ре-валидация редиректов через `_safe_open`) — не трогать.

### Находки: bot/plugins/agent_tools.py

- `bot/plugins/agent_tools.py:192-193` — WARNING — [кросс-тест/файлы] дефолтные
  `pending_file`/`background_jobs_file` указывают в дерево исходников `bot/plugins/`; голый
  `AgentToolsPlugin()` без `initialize()` (фикстура `test_agent_tools_verify.py:50`)
  пишет туда — вторичный источник кросс-тестового загрязнения и корень флака (см. раздел
  7). Fix: дефолт None/tmp, пути только через `initialize()`.
- `bot/plugins/agent_tools.py:2570` — WARNING — [семантика] `action="clear"` сохраняет
  ОТКРЫТЫЕ задачи (чистит только закрытые), вопреки имени — модель ожидает пустой план.
  Fix: переименовать в `prune_closed` либо действительно очищать всё.
- `bot/plugins/agent_tools.py:1550,2576-2578` vs `:329-331` — WARNING — [конкурентность]
  `_manage_plan_tasks` из `to_thread`-потока popает `_tool_error_streaks`/`_pending_replan`/
  `_pending_verify`; те же словари читает `on_before_chat_request` на loop под
  `_get_replan_lock()` — поп из потока не синхронизирован. Fix: мутировать только на loop.
- `bot/plugins/agent_tools.py:1956-1960` — WARNING — [утечка] `_plan_scope_locks` растёт
  неограниченно (Lock на каждый scope без эвикции); `_pending_*` копятся по всем scope.
  Fix: эвикция при clear/session_reset или LRU.
- `bot/plugins/agent_tools.py:1379-1433` — WARNING — [perf] goal-runs BackgroundTask с
  `interval_seconds=1.0` — SELECT + полный скан каждую секунду. Fix: увеличить интервал /
  событийное пробуждение.
- `bot/plugins/agent_tools.py:1478` — WARNING — [баг] `token_budget` проверяется только
  ПОСЛЕ завершения запуска — один прогон может выйти далеко за бюджет. Fix: инкрементальная
  проверка внутри субагентного цикла.
- `bot/plugins/agent_tools.py:3004-3062,2787-2794` — WARNING — [связность] жёсткая связка с
  skills: чтение `skills_plugin.active_skills`, вызов приватного
  `_disabled_skills_for_user`, хардкод имён `skills.*` в `SUBAGENT_BLOCKED_FUNCTIONS`. Fix:
  публичный API на skills или общий контекст через хуки.
- `bot/plugins/agent_tools.py:3254-3359,3592` — WARNING — [ядро/дублирование]
  `_run_subagent_completion_loop` дублирует core tool-calling loop и зовёт приватный
  `helper._tool_result_content`. Fix: общий публичный хелпер в openai_helper.
- `bot/plugins/agent_tools.py:3617-3698` — WARNING — [баг/UX] `_ask_telegram_user`: одна
  pending-вопрос на чат, future ожидается до 86400с прямо внутри tool-call (держит слот
  сутки); второй параллельный вопрос перетирает первый. Fix: ограничить ожидание,
  ключевать по сообщению.

### Находки: bot/plugins/hindsight_memory.py

- `bot/plugins/hindsight_memory.py:694-739` — **CRITICAL** — [баг/poison-pill]
  `_claim_finalize_jobs_sync`: транзакция claim коммитит `status='processing'` (BEGIN
  IMMEDIATE + UPDATE, `with` завершается ~`:735`), затем `json.loads(row["messages"])` на
  `:739` идёт по-строчно БЕЗ try/except (подтверждено прямым чтением). Битая строка бросает
  ПОСЛЕ захвата батча; исключение всплывает в `_finalize_tick` (`:828-832`), который лишь
  логирует и `return`. `attempts` инкрементируется только в `_mark_finalize_job_failed_sync`
  (`:785`), который в этом пути не достигается → весь батч навсегда `'processing'`,
  пере-захватывается после lease и падает снова — бесконечная блокировка финализации. Fix:
  обернуть парсинг каждой строки в try/except и помечать failed (или парсить до захвата).
- `bot/plugins/hindsight_memory.py:360-374` — WARNING — [связность/ядро] `initialize`
  зеркалит ~20 конфиг-ключей в `openai.config` через `setdefault` — плагин пишет в
  конфиг-неймспейс ядрового хелпера. Fix: держать конфиг в срезе плагина.
- `bot/plugins/hindsight_memory.py:402-444,2355,2385,2525` — WARNING — [ядро] reach-through
  `self.db_handle.database` за сырым sync-Database ради DDL/`BEGIN IMMEDIATE`, минуя
  async-фасад. Fix: добавить DDL/транзакции в DbHandle; убрать `.database`.
- `bot/plugins/hindsight_memory.py:1210-1287` — WARNING — [конкурентность] `_dream_tick`
  держит per-user `_memory_user_lock` ЧЕРЕЗ LLM-вызов (`_extract_dream_documents` →
  `chat_completion`): медленный вызов блокирует все операции памяти пользователя
  (approve/discard/clear). Fix: освобождать lock на время сетевого вызова.
- `bot/plugins/hindsight_memory.py:1835-1900` — WARNING — [perf] auto-recall в
  `on_before_chat_request` может делать ДВА сетевых recall-вызова на запрос (baseline +
  dynamic) синхронно в критическом пути подготовки; медленный Hindsight-сервер добавляет
  латентность каждому чату. Fix: таймаут/бюджет на recall, кэш.
- `bot/plugins/hindsight_memory.py:471-483` — NITPICK — `memory_types`: цепочка `if/if/elif`
  вместо `if/elif/elif` (работает, но хрупко).
- `bot/plugins/hindsight_memory.py:2542-2566` — POSITIVE — `_clear_local_memory_sync`
  корректно удаляет по 5 таблицам под BEGIN IMMEDIATE и бампает `clear_generation` — не
  трогать.

## 7. Вспомогательные модули и инфраструктура

### Состояние тестового набора (прогон 2026-07-02, system python3)

`python3 -m pytest -q`: **7 failed, 931 passed, 1 skipped** (~33 с).

- 5 падений — `bot/tests/test_mcp_server.py` (рассинхрон тестов с кодом, см. раздел 6).
- `tests/test_llm_gateway_client.py::test_extract_image_result_writes_data_image_url_to_path`
  — падение из-за общего каталога `/tmp/llmgateway_images` (см. находку ниже): на
  многопользовательской машине каталог принадлежит другому пользователю → PermissionError.
- `tests/test_agent_tools_verify.py::test_clear_plan_clears_pending_verify` — **флаки**:
  падает в полном прогоне, проходит в изоляции. Истинный корень — в
  `bot/database.py:32,144-163`: `_local = threading.local()` объявлен атрибутом **класса**,
  общим для всех Database-инстансов; `_reset_singleton` подменяет `cls._local`, а `__del__`
  резолвит `self._local` в текущий классовый атрибут и делает `connection.close()`. GC
  устаревшего Database-инстанса в недетерминированный момент закрывает **живое** соединение
  текущего теста → операция `clear` через `to_thread` (`agent_tools.py:1550`) падает на
  закрытом соединении. Усугубляют: голый `AgentToolsPlugin()` в фикстуре, пишущий в дерево
  исходников (`agent_tools.py:192-193`), и `Database._reset_singleton()` в
  setup/teardown. Fix: сделать `_local` инстанс-атрибутом; `__del__` не должен трогать
  общее классовое состояние.
- Предупреждения прогона: `RuntimeError: Event loop is closed` из `asyncio.locks` и
  never-awaited корутина `_remove_pending_busy_message` — признаки неаккуратного teardown
  асинхронных ресурсов в тестах/коде.

### Находки

- `bot/llm_gateway_client.py:216-219` — WARNING — [надёжность/безопасность]
  `_write_base64_image` пишет в общий `tempfile.gettempdir()/llmgateway_images` с
  фиксированным именем каталога: на multi-user хосте каталог, созданный другим
  пользователем, даёт PermissionError; файлы никогда не удаляются (нет GC). Fix:
  per-process `tempfile.mkdtemp(prefix=...)` либо каталог под `DB_PATH`-корнем + фоновая
  чистка.

### Архитектурные наблюдения (utils / html_utils / delivery / i18n)

- **Три параллельных пайплайна доставки текста в Telegram, которые уже разъехались:**
  (1) `bot/utils.py:1002` — telegramify-entities без parse_mode; (2)
  `bot/agent_delivery.py:34` — легаси `ParseMode.MARKDOWN` (V1) с fallback на BadRequest;
  (3) `bot/utils.py:1159` — HTML-файл через `HTMLVisualizer`. У agent-пути нет kind'ов
  `dice`/`reaction`, нет отправки файлом при переполнении и нет
  `cleanup_intermediate_files`. Любая правка формата direct_result требует синхронных
  правок в 2-3 местах.
- **`bot/utils.py` — модуль-свалка (1211 строк):** авторизация и бюджеты, чанкинг
  markdown, PIL-обработка изображений, direct-result delivery, scope-ключи плагинов и
  мёртвый markdown→HTML конвертер.
- **`bot/html_utils.py` — write-only генератор:** ~900 строк CSS/JS как Python-строки, три
  дословные копии цикла восстановления mermaid-плейсхолдеров (`:314-329`, `:416-427`,
  `:437-448`), regex-хирургия поверх распарсенного HTML, ~200 строк мёртвого кода. При
  этом `jinja2>=3.1.2` уже в requirements — шаблон убрал бы половину файла.
- **Скрытая связность через CWD:** `bot/utils.py:1169-1171`, `bot/html_utils.py:33,590,
  594,1812` пишут в относительные `output/`, `data/`, `plots/` — работоспособность зависит
  от каталога запуска. В Docker маскируется bind-mount'ом репозитория
  (`docker-compose.yml:7`), из systemd/другого CWD посыплется.
- **i18n на пределе подхода:** `translations.json` — 775KB, 20 языков × 517 ключей
  (консистентно), загрузка при импорте (`bot/i18n.py:47`). Каждый новый ключ — diff в
  757KB-файле. При росте — разложить на `locales/<lang>.json`.

### Находки: bot/html_utils.py

- `bot/html_utils.py:217,366,578,710,1510` — RESOLVED — [security/injection]
  HTML-фрагмент теперь проходит `_sanitize_html_fragment`, fenced code остаётся текстом,
  Mermaid-код экранируется перед вставкой в `<div class="mermaid">`, а Mermaid работает с
  `securityLevel: "strict"`. Покрыто XSS-регрессиями в `tests/test_html_utils_responsive.py`.
- `bot/utils.py:1177-1182` + `bot/html_utils.py:628` — **CRITICAL** — [blocking I/O]
  `send_long_response_as_file` (async) синхронно вызывает `advanced_visualization`:
  os.listdir/чтение файлов, base64 всех PNG, два прохода BeautifulSoup по документу и
  `subprocess.run(['java', ...])` (`:1697`) — всё в event loop; затем ещё синхронный
  `open/read` результата. При `concurrent_updates(True)` встаёт весь бот. (Подтверждено
  прямым чтением.) Fix: `await asyncio.to_thread(...)`.
- `bot/html_utils.py:1662` — WARNING — [correctness] `replace("\\n", "<br>")` по
  ФИНАЛЬНОМУ HTML целиком, включая `<script>` и код: литеральный `\n` в пользовательском
  коде станет `<br>` внутри `<pre>`. Fix: замена только в пользовательском фрагменте.
- `bot/html_utils.py:1676` — WARNING — [error handling] `finally: clean_data(session_id)`
  — `clean_data` (`:1678`) делает `os.listdir('data')` без guard'а: нет каталога →
  FileNotFoundError из finally маскирует исходную ошибку; на success-пути clean_data
  выполняется дважды (`:1668` и `:1676`). Fix: guard + один вызов.
- `bot/html_utils.py:670` — WARNING — [latent bug] `result.append(...)`, но единственный
  вызывающий (`bot/utils.py:1178`) передаёт `result` строкой; AttributeError глушится
  except'ом (`:673`) — MD-файлы молча не попадают в отчёт. Fix: нормализовать вход.
- `bot/html_utils.py:89,1697` — RESOLVED — [infra] PlantUML оказался live feature:
  `bot/plugins/plantuml.jar` есть в текущем checkout. Добавлены поиск jar в `bot/plugins`,
  Docker runtime packages Java + Graphviz и уточнены README-требования.
- `bot/html_utils.py:281,332` + `:197,229,256` — WARNING — [performance]
  `make_urls_clickable` (полнодокументный re.sub) гоняется дважды; замены срезами с ручным
  offset — O(n²) при большом числе ссылок; BeautifulSoup пересоздаётся минимум трижды.
  Fix: один проход парсером.
- `bot/html_utils.py:628` — RESOLVED — [dead code] ранее перечисленные
  неиспользуемые Mermaid/agent/JSON helper-символы и импортный singleton удалены;
  runtime-путь `HTMLVisualizer.advanced_visualization` сохранён.
- `bot/html_utils.py:3,10` — NITPICK — [hygiene] `import re` дважды; неиспользуемые
  импорты; переменная `html` (`:154`) затеняет stdlib-модуль; `print()` вместо logging по
  всему файлу.

### Находки: bot/utils.py

- `bot/utils.py:312-322` — WARNING — [correctness] стек markdown в `split_into_chunks`
  ломается на fenced-блоках: после каждого ``` в стеке остаётся фантомный бэктик, на
  границе чанка дописывается мусорный `` ` ``. Через `agent_delivery.send_text_chunks`
  (parse_mode=MARKDOWN) — BadRequest и потеря форматирования. Fix: обрабатывать ``` как
  единый токен (или заменить эвристику на telegramify split, как в `:335`).
- `bot/utils.py:636` — WARNING — [correctness/billing] `allowed_user_ids.split(',')` без
  `.strip()`: при `ALLOWED_TELEGRAM_USER_IDS="123, 456"` расход пользователя 456 пишется в
  `guests`, хотя `is_allowed` (`:494`) со strip его пропускает. Fix: единый распарсенный
  список.
- `bot/utils.py:1095-1157` — WARNING — [dead code] старый markdown-конвертер имел ноль
  вызовов, дублировал `html_utils._convert_markdown` и был небезопасен. Status: удалён
  в remediation cleanup.
- `bot/utils.py:951-973` — NITPICK — [perf] картинка открывается PIL'ом дважды; `open()`
  синхронно в async-функции. Fix: один вызов + to_thread.
- `bot/utils.py:909` — NITPICK — [contract] `handle_direct_result` возвращает `None` при
  отсутствии message, но список в остальных ветках. Fix: `return []`.
- `bot/utils.py:379` — NITPICK — `asyncio.timeout(4000)` — «ограничитель» в 66 минут.
  Fix: конфиг с вменяемым дефолтом.

### Находки: __main__ / shutdown / delivery / роутинг

- `bot/__main__.py:111-136,200-207` — WARNING — [robustness] ~15 строгих `int()/float()`
  по env: опечатка роняет старт голым ValueError без имени переменной, хотя рядом (`:34`)
  есть `_parse_numeric_env`. Fix: единая мягкая политика парсинга.
- `bot/telegram_bot.py:5444` + `:5491` — WARNING — [shutdown] `cleanup()` подключён дважды
  (post_shutdown + finally); `Database.shutdown()` не вызывается нигде, кроме
  `__del__`/`_reset_singleton` — worker-поток БД завершается только сборщиком мусора. Fix:
  один путь (post_shutdown) + `db.shutdown()` в нём.
- `bot/agent_delivery.py:86-183` — WARNING — [duplication/drift] параллельная реализация
  direct-result рядом с `utils.handle_direct_result`: без `dice`/`reaction`, без ухода в
  файл при переполнении и без `cleanup_intermediate_files` — артефакты с `format=path`,
  доставленные через agent/cron-путь, навсегда остаются на диске. Fix: единый
  delivery-модуль; минимум — cleanup после отправки.
- `bot/agent_delivery.py:34` — NITPICK — легаси `ParseMode.MARKDOWN` (V1) при основном
  пути на telegramify-entities; вместе с багом чанкера — регулярный BadRequest. Fix:
  переиспользовать `render_markdown_message_entities`.
- `bot/skill_script_routing.py:11-16` — NITPICK — [design] `SCRIPT_FILE_CREATION_RE` —
  эвристика по ключевым словам промпта с межплагинными именами в core; сработает на любой
  промпт «create … python … .py» в 240 символах — ложные отказы по мере роста сценариев.
  Компромисс задокументирован — мониторить.
- `bot/__main__.py:94,162` — NITPICK — builtin `exit(1)` вместо `sys.exit(1)`.
- `bot/__main__.py:196` — NITPICK — `VOICE_REPLY_PROMPTS=''.split(';')` даёт `['']`.

### Находки: инфраструктура

- `requirements.txt:6` — WARNING — [security] `requests==2.31.0` уязвим: CVE-2024-35195
  (fix в 2.32.0), CVE-2024-47081 (fix в 2.32.4). Fix: `requests>=2.32.4`.
- `requirements.txt:4,16` — WARNING — [deps] `openai>=2.14.0` без верхней границы при
  жёстком `httpx==0.27.0` — обновление openai сломает установку. Пиннинг хаотичен: точные
  пины вперемешку со свободными (`duckduckgo_search`, `numpy`, `pandas`, ...) — билд
  невоспроизводим. Fix: lock-файл (uv/pip-tools) + границы версий.
- `Dockerfile:17` — WARNING — [docker] контейнер от root (нет `USER`), нет `HEALTHCHECK`,
  база без digest-пина — при том что бот исполняет terminal/codeinterpreter-плагины. Fix:
  непривилегированный пользователь + healthcheck.
- `docker-compose.yml:7` — WARNING — [docker] bind-mount `.:/app` затеняет образ (COPY
  бессмыслен; `.env` и `.git` попадают в контейнер, минуя .dockerignore), всё, что бот
  пишет, ложится на хост root-owned; `env_file` отсутствует. Fix: сузить маунт до
  data-каталогов, добавить `env_file`.
- `.env` — OK — [secrets] подтверждено без чтения содержимого: в `.gitignore:3`, никогда
  не коммитился (`git log --all -- .env` пуст), исключён из образа. Единственный канал
  утечки — bind-mount выше.
- `requirements.txt:41,60` — NITPICK — `beautifulsoup4` дважды; dev-зависимости (pytest,
  ipython) в runtime-списке. Fix: dedupe + requirements-dev.txt.
- `translations.json` — NITPICK — исполняемый бит на JSON-файле данных. Fix: chmod 644.

### Дыры тестового покрытия (по структуре tests/)

- `bot/html_utils.py` (2064 строки) покрыт одним файлом `test_html_utils_responsive.py`
  (3.4KB) — ни экранирование, ни mermaid-пайплайн не тестируются.
- `bot/agent_delivery.py` — ни одного прямого теста контракта delivery.
- `split_into_chunks` с fenced-блоками не покрыт (баг `bot/utils.py:312` прошёл бы тестом).
- Shutdown-путь (`cleanup`, двойной вызов) не тестируется.

## 8. Предложения по улучшению архитектуры

Разделено на приоритеты. P0 — исправить срочно (баги/дыры, ломающие пользователей или
безопасность). P1 — архитектурный долг с высоким рычагом. P2 — качество/поддерживаемость.

### P0 — исправить в первую очередь (баги и безопасность)

1. **Голосовой флоу мёртв** (`bot/telegram_bot.py:2270`, NameError `audio_track`). Ни один
   тест это не ловит → добавить тест на успешную транскрипцию + фикс. **Один из самых
   дорогих багов: целая функция бота не работает.**
2. **Дыра авторизации в inline-callback** (`bot/telegram_bot.py:4176`): любой участник чата
   тратит API-ключ владельца в обход allow-list и бюджета. Добавить
   `check_allowed_and_within_budget` в начало хендлера.
3. **Security-профиль плагина skills default-open** (`skills.py:147,150,152` + два
   `os.environ.copy()` на `:2520,3712` + `confirmed` как аргумент модели). Три отдельных
   CRITICAL: перейти на default-deny, урезать env дочерних процессов до allow-list,
   вынести подтверждение в Telegram-callback. **Пока не закрыто — бот с включённым skills
   исполняет произвольный код и утекает все секреты.**
4. **Poison-pill в hindsight finalize** (`hindsight_memory.py:694-739`): битая строка
   навсегда блокирует финализацию батча. Обернуть парсинг в try/except с пометкой failed.
5. **RESOLVED: лимит tool-реентри** (`openai_tool_handler.py`): generic successful tools
   после `functions_max_consecutive_calls` получают `tool_choice='none'`; delivery escape
   после лимита сужает specs до `agent_tools.deliver_to_user`, а plain-text tool-intent
   repair не получает tools.
6. **RESOLVED: дублирование user-сообщений при RateLimit** (`openai_helper.py`): retry
   перенесён на SDK create boundary; прямой regression проверяет один user-message в
   memory/DB после RateLimit.
7. **RESOLVED: usage persistence** (`usage_tracker.py:15-260`): usage перенесён в SQLite
   events/aggregates с one-time импортом legacy JSON и compatibility snapshot.
8. **RESOLVED: XSS/инъекция в генерируемом HTML** (`html_utils.py`):
   HTML sanitization, Mermaid escaping и `securityLevel: "strict"` включены; Mermaid
   normalization больше не декодирует произвольный `&lt;...&gt;` в теги, но сохраняет
   encoded arrows/line-breaks.
9. **Флаки-тест = реальный баг жизненного цикла БД** (`database.py:32`): класс-атрибут
   `_local` + `__del__`, закрывающий чужие соединения. Сделать `_local` инстансным.
10. **Устаревший `requests==2.31.0`** (2 CVE) — поднять до `>=2.32.4`.

### P1 — архитектурный долг (высокий рычаг)

**A. Завершить миграцию БД на async-фасад и убрать блокировку event loop.**
**STATUS: частично закрыто в P1-A (2026-07-02).** Выделенный db-worker и async wrappers
теперь закреплены тестами на core reset/pre-prune, subagent model lookup и Haiper image
reads. Пройден reviewer-loop без предупреждений. Дальше закрывать остатки точечно:
   - синхронные compatibility helpers (`_get_user_language`, public `get_current_model`);
   - lifecycle `shutdown` оставить documented exception;
   - UsageTracker offload рассматривать отдельной задачей;
   - новые DB API наружу давать через `*_async`/`DbHandle`, sync-API не расширять без
     явного legacy-сценария.

**B. Ввести объект `ConversationState` вместо разделяемых dict'ов по state_key.**
Сейчас пер-запросное состояние (`conversations`, `_chat_request_models`, `_gate_fired`,
`loaded_conversation_sessions`, `last_image_file_ids`) живёт в нескольких параллельных словарях на
инстансе OpenAIHelper, синхронизируется только `_chat_lock`, а bypass-режим и `ask()` ломают
инвариант. Единый объект состояния под одним локом уберёт целый класс гонок и «тройную
систему идентификаторов chat_id/user_id/state_key».

**C. Формализовать контракт результата инструмента как типизированный объект.**
Сейчас `{"error"}`/`{"success": false}`/`{"ok": false}`/direct_result/строка нормализуются
в трёх местах, которые уже разошлись. Ввести единый dataclass `ToolResult` на границе
`Plugin.execute`, нормализацию — в одну точку. Плагины возвращают его, ядро больше не
угадывает по ключам.

**D. Централизовать авторизацию в PTB-middleware (TypeHandler group=-1).**
Заменить per-handler `is_allowed`-проверки одним слоем: устраняет целый класс пропусков
(inline-callback, help), даёт единую точку для кэша membership групп (сейчас — N сетевых
`get_chat_member` на сообщение).

**E. Убрать мёртвый провайдер-слой `model_constants`.**
Все кортежи пусты; десятки веток-потребителей мертвы и гниют (сломанный stream-флаг в
`openai_helper.py:1334` — уже готовый latent-баг). Либо удалить слой и потребителей, либо
наполнить из конфигурации явно. Заодно — конфигурируемый маппинг `model → context_window`
вместо захардкоженных 200k (`openai_helper.py:172`), чтобы бюджет токенов и суммаризация
считались верно.

**F. Единый модуль доставки в Telegram.**
Три разошедшихся пути (`utils.handle_direct_result`, `agent_delivery`, HTML-файл) с разным
набором фич (agent-путь не чистит артефакты, использует legacy Markdown V1). Свести в один
delivery-слой; починить `split_into_chunks` для fenced-блоков.

**G. Загерметизировать границу «ядро ↔ плагины».**
`openai_tool_handler` зовёт приватные методы agent_tools; telegram_bot мутирует приватные
поля OpenAIHelper; плагины лезут в `db_handle.database`. Ввести публичные API там, где
сейчас reach-through, и расширить `test_no_hardcoded_plugin_refs` на openai_tool_handler.

### P2 — качество и поддерживаемость

- **Разбить god-модули.** `telegram_bot.py` (5.5k) → роутинг / UI-меню / стриминг /
  координация. `openai_helper.py` (3.8k) → вынести суммаризацию, vision, бюджетирование.
  Плагины-переростки (agent_tools/skills/hindsight по 2.6–3.9k) → слои client/store/worker/ui.
- **Дедупликация:** 3 копии стримингового state-machine, 4 копии chat-пайплайна (мутаторы
  применяются только в одной — vision теряет hindsight/planning!), 3 копии
  mermaid-восстановления, 3 места поиска chat-mode по полному тексту prompt.
- **Ротация и retention:** `tool_call_events`, session_logger JSONL, llm_gateway-картинки,
  usage_history — ничто не чистится. Добавить фоновый GC + индексы (`images(created_at)`,
  `images(user_id)`).
- **Конфигурация:** убрать дублирование ключей между `openai_config`/`telegram_config`,
  единая мягкая политика парсинга env (сейчас `SUMMARY_*` мягкие, остальные роняют старт),
  прекратить `plugin_manager.config.update(openai_config)` (слияние неймспейсов).
- **Ротация логов и PII:** полный payload запросов и tool-результаты пишутся на INFO
  (мегабайты base64, история, секреты). Перевести на DEBUG с усечением.
- **Инфраструктура:** lock-файл зависимостей (uv/pip-tools) вместо хаотичного пиннинга;
  непривилегированный пользователь в Dockerfile; сузить bind-mount в docker-compose;
  разложить `translations.json` (775KB) на `locales/<lang>.json`.
- **Мёртвый код:** отдельный oldest-session deletion API, старый markdown-конвертер, dead
  `html_utils` helpers, transient image attribute и `conversations_vision` уже удалены;
  PlantUML live path починен через packaging/path.
- **Тесты:** починить 7 падающих (рассинхрон MCP-тестов с кодом, изоляция llm_gateway
  `/tmp`, флаки БД); закрыть дыры покрытия (html_utils экранирование/mermaid,
  agent_delivery, split_into_chunks, shutdown-путь).

### Порядок действий (рекомендация)

1. Закрыть P0 №1–9 — это баги, которые уже сейчас ломают пользователей или открывают
   исполнение кода/утечку секретов. Каждый — точечная правка с тестом.
2. Затем P1-A (async-БД) и P1-D (middleware авторизации) — максимальный рычаг: убирают
   целые классы проблем (блокировка loop, пропуски авторизации).
3. P1-B/C/E-G и P2 — по мере рефакторинга god-модулей, начиная с самых горячих путей.

---

## Приложение: методология

Аудит выполнен персонами **architect** + **reviewer** из skill **dev-experts**. Кодовая
база (~18.7k строк ядра + ~20k строк плагинов) разбита на 6 подсистем, каждую построчно
прочитал отдельный агент-ревьюер; все CRITICAL-находки перепроверены лично по коду с
указанием `file:line`. Тесты прогонялись на `python3 -m pytest` (`.venv/bin/python`
недоступен из-за прав). Ничего не редактировалось и не коммитилось (правило AGENTS.md).
