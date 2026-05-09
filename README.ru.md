# ChatGPT Telegram Bot

> 🌐 **Язык / Language:** **Русский** · [English](README.md)

📚 **Подробная документация:**
- [Пошаговый туториал (17 глав)](/docs/tutorial)
- [Полная документация сайта](/ai_docs_site)

Самостоятельно разворачиваемая агентная рабочая среда в Telegram для
OpenAI-совместимых моделей, tools, субагентов, skills и долгоживущих
диалогов. Проект сохраняет классические сценарии ChatGPT-бота — чат,
изображения, аудио, транскрипцию, TTS и vision — но больше не позиционируется
как минимальная обёртка вокруг ChatGPT. Это рантайм ассистента поверх
LLMGateway-маршрутизации, управляемых tool calls, именованных SQLite-сессий,
опциональной долговременной памяти Hindsight, локальных skills, доставляемых
артефактов и интеграций Model Context Protocol (MCP).

> **Позиционирование.** Считай этот репозиторий платформой для продвинутого
> персонального ассистента, а не простым шаблоном публичного чат-бота. Кодовая
> база значительно разошлась с upstream-проектом `chatgpt-telegram-bot`:
> дефолтные модели, маршрутизация изображений/аудио, неймспейсинг плагинов,
> реестр чат-режимов, рантайм агентов/skills и расположение хранилища —
> всё это специфично для этого репозитория. README описывает текущее
> поведение. Если возникают сомнения — авторитетным источником считай код
> (`bot/`) и `.env.example`.

## Ключевые возможности

- **Агентная рабочая среда в Telegram**: OpenAI-совместимый chat runtime с
  неймспейсингом инструментов плагинов, JSON-Schema-валидацией аргументов,
  allow-list’ами инструментов на уровне чат-режимов, plan-aware исполнением и
  прямой доставкой артефактов обратно в Telegram.
- **Агенты и сабагенты**: `agent_tools` умеет запускать ограниченных сабагентов,
  задавать пользователю уточняющие вопросы в Telegram, вести план,
  координировать работу tools и доставлять сгенерированные файлы, изображения,
  аудио или текстовые артефакты.
- **Рантайм skills**: локальные Codex-style папки с `SKILL.md` экспонируются
  как tools, с опциональным запуском скриптов, allow-list’ами операторов,
  progress-апдейтами и маршрутизацией через terminal / code-interpreter
  исполнение.
- **Файлы и медиа**: генерация и редактирование изображений, vision,
  транскрипция аудио/видео, TTS, Q&A по документам и reply-to-file сценарии,
  где Telegram-документы скачиваются заново для tool-based правок.
- **MCP и внешние интеграции**: удалённые MCP-серверы можно регистрировать и
  пробрасывать модели как tools вместе со встроенными плагинами для web,
  research, search, translation, документов и утилит.
- **Память и сессии**: именованные SQLite-сессии, чат-режимы,
  пер-разговорная сериализация, бюджеты использования и опциональная
  долговременная память Hindsight с recall / retention.

---

## Оглавление

1. [Возможности](#возможности)
2. [Архитектура](#архитектура)
3. [Требования](#требования)
4. [Быстрый старт](#быстрый-старт)
5. [Конфигурация](#конфигурация)
   - [Обязательные переменные](#обязательные-переменные)
   - [Telegram core](#telegram-core)
   - [LLMGateway / OpenAI core](#llmgateway--openai-core)
   - [Изображения, vision, TTS, транскрипция](#изображения-vision-tts-транскрипция)
   - [Сессии, разговоры, поведение](#сессии-разговоры-поведение)
   - [Бюджеты и тарификация](#бюджеты-и-тарификация)
   - [Плагины и хранилище](#плагины-и-хранилище)
   - [Agent tools / субагенты](#agent-tools--субагенты)
   - [Плагин Skills](#плагин-skills)
   - [Плагин MCP](#плагин-mcp)
   - [Hindsight (долговременная память)](#hindsight-долговременная-память)
   - [Ключи отдельных плагинов](#ключи-отдельных-плагинов)
   - [Настройка БД](#настройка-бд)
   - [Устаревшие переменные](#устаревшие-переменные)
6. [Telegram UX](#telegram-ux)
   - [Слэш-команды](#слэш-команды)
   - [Inline-меню](#inline-меню)
   - [Обработчики медиа](#обработчики-медиа)
   - [Inline-режим](#inline-режим)
7. [Сессии и чат-режимы](#сессии-и-чат-режимы)
8. [Каталог плагинов](#каталог-плагинов)
9. [Рантайм агентов / субагентов](#рантайм-агентов--субагентов)
10. [Рантайм skills](#рантайм-skills)
11. [Интеграция с MCP](#интеграция-с-mcp)
12. [Память Hindsight](#память-hindsight)
13. [Расположение хранилища](#расположение-хранилища)
14. [Разработка](#разработка)
15. [Диагностика](#диагностика)
16. [Авторы и зависимости](#авторы-и-зависимости)

---

## Возможности

- **Чат с именованными SQLite-сессиями**: история на пользователя, настраиваемые
  `MAX_HISTORY_SIZE` / `MAX_CONVERSATION_AGE_MINUTES`, автонейминг сессий,
  inline-меню управления сессиями, опциональное автопредложение чат-режимов.
- **Трёхуровневая маршрутизация моделей** через LLMGateway:
  - `llmgateway/high` — основной ассистент.
  - `llmgateway/light_model` — маршрутизация, классификация, нейминг,
    распознавание намерений.
  - `llmgateway/big_context` — задачи с большим контекстом и фолбэк для vision.
- **Изображения**: генерация через `llmgateway/ai-klein-generation`,
  редактирование через `llmgateway/ai-klein-edit`, intent-routing
  ответов/подписей либо в генерацию, либо в редактирование, либо в vision.
- **Vision**: описание изображений выбранной моделью `VISION_MODEL`,
  опциональные follow-up-вопросы.
- **Аудио**: транскрипция голосовых / видео / видеосообщений / аудиодокументов
  через выбранную модель транскрипции; опциональные голосовые ответы;
  по умолчанию TTS Silero через LLMGateway (голос `kseniya`, формат `wav`).
- **Веб и исследования**: gateway-поиск, чтение страниц, обычный research,
  deep-research; путь YouTube-транскриптов; web-research автоматически выбирает
  обычный или deep-режим через light-модель.
- **Система плагинов** с пространствами имён инструментов
  (`<plugin_id>.<function>`), JSON-Schema-валидацией аргументов, allow-list’ом
  на уровне чат-режима, плагинными слэш-командами и inline-меню, фоновыми
  тасками.
- **Reply-to-file сценарии**: когда пользователь отвечает текстом на
  Telegram-документ, не являющийся изображением, бот скачивает этот файл на
  текущий ход и передаёт модели временный локальный путь, чтобы инструменты
  могли отредактировать, конвертировать или проанализировать файл и вернуть
  новый артефакт. Reply на изображения остаётся в отдельной маршрутизации
  image-edit / vision.
- **Рантайм агентов**: плагин `agent_tools` экспонирует `run_subagents`,
  `ask_telegram_user`, `deliver_to_user` и единый инструмент
  `manage_plan_tasks` для трекинга плана; есть жёсткие лимиты на количество
  раундов, скоуп и размер артефактов.
- **Рантайм skills**: плагин `skills` сканирует локальные папки скилов в стиле
  Codex, экспонирует их как инструменты и (по согласию оператора) выполняет
  их скрипты с allowlist’ом операторов, watchdog-индикатором typing
  и потоковым прогрессом.
- **Интеграция с MCP**: встроенный MCP-клиент-плагин подключается к удалённым
  MCP-серверам и пробрасывает их инструменты модели.
- **Hindsight (долговременная память)**: опциональный авто-recall на первом
  запросе сессии и авто-save при удалении сессии.
- **Бюджеты на пользователя** с периодами monthly / daily / weekly и отдельной
  тарификацией для токенов, изображений, TTS, vision, транскрипции.
- **Telegram local-mode** включён по умолчанию (`http://localhost:8081/bot`);
  хостируемое API Telegram включается переключением `TELEGRAM_LOCAL_MODE=false`.

---

## Архитектура

Точка входа процесса: [`bot/__main__.py`](bot/__main__.py).

Порядок старта:

1. Загрузить `.env` через `python-dotenv`.
2. Проверить `TELEGRAM_BOT_TOKEN` и `OPENAI_API_KEY` (если хоть одна
   отсутствует — процесс завершается).
3. Собрать конфиги (OpenAI, Telegram, плагины).
4. Создать `PluginManager` (находит `bot/plugins/*.py` и грузит то, что
   указано в allow-list’е `PLUGINS`).
5. Создать `Database` (singleton, SQLite, по умолчанию WAL, thread-local
   соединения, foreign keys включены).
6. Создать `OpenAIHelper` (OpenAI-совместимый / LLMGateway-клиент, vision-
   и TTS-хелперы, реестр чат-режимов, опциональный Hindsight-клиент).
7. Создать `ChatGPTTelegramBot` и вызвать `run()` — собирается
   PTB-`Application` (concurrent-updates включены, по умолчанию local-mode).

| Путь | Ответственность |
|---|---|
| `bot/__main__.py` | Загрузка env, сборка конфига, dependency wiring, точка входа процесса |
| `bot/telegram_bot.py` | Telegram-хендлеры, UI сессий, плагинные команды, маршрутизация медиа, пагинация плагин-меню, потоковая выдача |
| `bot/openai_helper.py` | Хелперы chat / vision / image / audio, выбор модели, scheduling Hindsight auto-recall и auto-save, применение чат-режима |
| `bot/openai_tool_handler.py` | Извлечение и выполнение tool-call’ов, батчинг через `asyncio.gather`, deferred / direct-result short-circuit |
| `bot/llm_gateway_client.py` | Эндпоинты gateway (web search/read/research, image edit) |
| `bot/plugin_manager.py` | Discovery плагинов, неймспейсинг, JSON-Schema-валидация, агрегация команд / меню, gating субагентов, инициализация storage root |
| `bot/plugins/plugin.py` | Базовый класс `Plugin` (`get_source_name`, `get_spec`, async `execute`, опциональные команды/хендлеры/`on_startup`) |
| `bot/plugins/*.py` | Встроенные плагины (см. [Каталог плагинов](#каталог-плагинов)) |
| `bot/chat_modes_registry.py` | Загружает `bot/chat_modes.yml`, валидирует ссылки на плагины |
| `bot/chat_modes.yml` | Чат-режимы для пользователя, allow-list плагинов на режим |
| `bot/database.py` | Singleton SQLite, сессии, контекст разговора, image-references, учёт использования |
| `bot/usage_tracker.py` | Учёт стоимости токенов / изображений / TTS / транскрипции и окна бюджетов |
| `bot/hindsight_client.py` | HTTP-клиент Hindsight (recall, retain, list, stats, clear) |
| `bot/request_context.py` | Frozen-dataclass `RequestContext`, передаётся в инструменты при выполнении |
| `bot/skill_script_routing.py` | Роутер, нацеливающий запуск скилов на `terminal` или `codeinterpreter` |
| `bot/utils.py` | `BusyStatusMessage`, `compute_scope_key`, разбиение сообщений, отдача файлов, утилиты |
| `bot/validation.py` | JSON-Schema-подобная валидация аргументов плагинов |
| `bot/conversation_key.py` | Ключ `(chat_id, thread_id)` для пер-чатовой сериализации |
| `bot/i18n.py` | Локализация UI |
| `bot/html_utils.py` | Конвертация Markdown / HTML |
| `bot/model_constants.py` | Дефолтные идентификаторы моделей (`LLMGATEWAY_HIGH_MODEL`, …) |
| `bot/prompts/subagent_system.md` | Системный промпт для субагентов |
| `examples/` | Пример MCP-сервера / клиента (`mcp_server_example.py`, `mcp_stdio_*.py`) |
| `tests/` | Основной набор pytest |
| `bot/tests/` | MCP-специфичные тесты |
| `data/` | Дефолтный storage root для плагинов (создаётся при старте, в git не коммитится) |

В репозитории также лежит сгенерированная карта кодбазы под
`.cli-proxy/.codebase_map/`, используется агентскими инструментами.

---

## Требования

- **Python 3.9+** (текущая разработка идёт на 3.12).
- **Telegram-токен** от @BotFather.
- **LLMGateway-совместимый эндпоинт** (или любой другой OpenAI-совместимый —
  имена `OPENAI_API_KEY` / `OPENAI_BASE_URL` остались по совместимости).
- **`ffmpeg`** для аудио / видео.
- **Java runtime** (только если включён плагин `show_me_diagrams` — он
  вызывает `bot/plugins/plantuml.jar`).
- Рекомендуется **локальный Telegram Bot API** (бот по умолчанию идёт на
  `TELEGRAM_LOCAL_MODE=true`, base URL `http://localhost:8081/bot`).
  Для хостируемого Telegram API — `TELEGRAM_LOCAL_MODE=false`.

---

## Быстрый старт

```bash
git clone <repo>
cd chatgpt-telegram-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# отредактируй .env: задай TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, OPENAI_BASE_URL
python -m bot
```

Также есть Docker Compose:

```bash
docker compose up
```

`OPENAI_BASE_URL` по умолчанию пустой и должен указывать на твой gateway,
например `http://gateway.example/v1`.

---

## Конфигурация

Вся конфигурация — через переменные окружения (обычно файл `.env` в корне,
загружается через `python-dotenv`). Таблицы ниже покрывают каждую переменную,
которую читает рантайм. **Жирные** строки — обязательные.

### Обязательные переменные

| Переменная | Тип | Назначение |
|---|---|---|
| **`TELEGRAM_BOT_TOKEN`** | string | Telegram-токен от @BotFather. Без него процесс не стартует. |
| **`OPENAI_API_KEY`** | string | Ключ LLMGateway / OpenAI-совместимого API. Без него процесс не стартует. |

### Telegram core

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `TELEGRAM_LOCAL_MODE` | `true` | bool | Использовать локальный Telegram Bot API. При `false` бот идёт на хостируемое Telegram API и игнорирует `TELEGRAM_BASE_URL`. |
| `TELEGRAM_BASE_URL` | `http://localhost:8081/bot` (в local-mode) | url | Base URL Bot API. Валидируется как абсолютный http(s). |
| `ADMIN_USER_IDS` | `-` | csv | Telegram user ID администраторов через запятую. `-` — без админов. |
| `ALLOWED_TELEGRAM_USER_IDS` | `*` | csv | Allow-list user ID через запятую. `*` — разрешить всем. |
| `ENABLE_QUOTING` | `true` | bool | Отвечать через reply на исходное сообщение в личных чатах (в группах reply всегда). |
| `GROUP_TRIGGER_KEYWORD` | `` | string | Если задано — групповое сообщение должно содержать ключевое слово, чтобы бот среагировал. |
| `IGNORE_GROUP_TRANSCRIPTIONS` | `true` | bool | Не делать авто-транскрипцию в группах. |
| `IGNORE_GROUP_VISION` | `true` | bool | Не делать vision в группах. |
| `TELEGRAM_DOWNLOAD_BOT_ID` | `` | string | Опциональный ID второго бота-загрузчика для крупных файлов. |
| `TELEGRAM_DOWNLOAD_DIR` | `media` | path | Папка, куда скачиваются медиа. |
| `PROXY` / `TELEGRAM_PROXY` | `` | url | HTTP(S)-прокси для трафика Telegram. |

### LLMGateway / OpenAI core

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `OPENAI_BASE_URL` | `` | url | Base URL gateway / OpenAI-совместимого API (например, `http://gateway.example/v1`). |
| `OPENAI_MODEL` | `llmgateway/high` | string | Основная модель ассистента. |
| `LIGHT_MODEL` | `llmgateway/light_model` | string | Быстрая модель для классификации, маршрутизации, нейминга, распознавания намерений. |
| `BIG_MODEL_TO_USE` | `llmgateway/big_context` | string | Модель для большого контекста (используется при компактизации истории и как фолбэк для vision). |
| `MAX_TOKENS` | зависит от модели | int | Лимит выходных токенов на ответ. |
| `MAX_HISTORY_SIZE` | `15` | int | Сколько сообщений истории держать в памяти до суммаризации. |
| `MAX_CONVERSATION_AGE_MINUTES` | `180` | int | Возраст разговора, после которого он сбрасывается. |
| `TEMPERATURE` | `1.0` | float | Sampling temperature. |
| `PRESENCE_PENALTY` | `0.0` | float | OpenAI presence penalty. |
| `FREQUENCY_PENALTY` | `0.0` | float | OpenAI frequency penalty. |
| `N_CHOICES` | `1` | int | Сколько completion-ов запрашивать. |
| `STREAM` | `true` | bool | Стримить ответы в Telegram. |
| `SHOW_USAGE` | `false` | bool | Дописывать в ответ футер с использованием токенов. |
| `SHOW_PLUGINS_USED` | `false` | bool | Дописывать список вызванных плагинов / инструментов. |
| `ASSISTANT_PROMPT` | `You are a helpful assistant.` | string | Дефолтный системный промпт до применения чат-режима. |
| `WHISPER_PROMPT` | `` | string | Опциональный промпт для модели транскрипции. |
| `PROXY` / `OPENAI_PROXY` | `` | url | HTTP(S)-прокси для трафика OpenAI/LLMGateway. |
| `PROXY_WEB` | `` | url | Прокси для разовых веб-запросов. |
| `ENABLE_FUNCTIONS` | автодетект | bool | Включить tool-calling. Автоопределяется по выбранной модели. |
| `FUNCTIONS_MAX_CONSECUTIVE_CALLS` | `10` | int | Лимит последовательных раундов tool-call’ов в основном цикле (на один ответ ассистента). |

### Изображения, vision, TTS, транскрипция

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `ENABLE_IMAGE_GENERATION` | `true` | bool | Включить генерацию изображений и команду `/image`. |
| `IMAGE_MODEL` | `llmgateway/ai-klein-generation` | string | Модель генерации. |
| `IMAGE_QUALITY` | `standard` | string | `standard` или `hd`. |
| `IMAGE_STYLE` | `vivid` | string | `vivid` или `natural`. |
| `IMAGE_SIZE` | `512x512` | string | Размер, например `1024x1024`. |
| `IMAGE_FORMAT` | `photo` | string | `photo` или `document`. |
| `ENABLE_VISION` | `true` | bool | Включить vision над фото / image-документами. |
| `VISION_MODEL` | `llmgateway/big_context` | string | Vision-модель. |
| `VISION_PROMPT` | `What is in this image` | string | Дефолтная инструкция для vision. |
| `VISION_DETAIL` | `auto` | string | `low`, `high` или `auto`. |
| `VISION_MAX_TOKENS` | `1000` | int | Лимит токенов для vision-ответа. |
| `ENABLE_VISION_FOLLOW_UP_QUESTIONS` | `true` | bool | Разрешить follow-up-вопросы по последнему изображению. |
| `ENABLE_TRANSCRIPTION` | `true` | bool | Включить транскрипцию голосовых / аудио / видео. |
| `TRANSCRIPTION_MODEL` | `llmgateway/whisper-large-v3` | string | Модель транскрипции. |
| `ENABLE_TTS_GENERATION` | `true` | bool | Включить команду `/tts` и TTS-плагины. |
| `TTS_MODEL` | `llmgateway/silero-tts` | string | TTS-модель. |
| `TTS_VOICE` | `kseniya` | string | Голос. |
| `TTS_RESPONSE_FORMAT` | `wav` | string | Формат аудио на выходе (`wav`, `mp3`, …). |
| `VOICE_REPLY_WITH_TRANSCRIPT_ONLY` | `false` | bool | Отвечать на голосовое только транскриптом. |
| `VOICE_REPLY_PROMPTS` | `` | список через `;` | Триггер-фразы, форсирующие путь голосового ответа. |
| `YANDEX_API_TOKEN` | `` | string | Опциональный Yandex-токен для `text_summarizer`. |
| `ASSEMBLYAI_API_KEY` | `` | string | Опциональный AssemblyAI-ключ для путей транскрипции. |

### Сессии, разговоры, поведение

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `MAX_SESSIONS` | `5` | int | Максимум именованных сессий на пользователя. Старые удаляются через `delete_oldest_session()`. |
| `BOT_LANGUAGE` | `en` | string | Язык UI бота (`en`, `ru`, …). |
| `AUTO_CHAT_MODES` | `false` | bool | Автоматически предлагать чат-режим по первому сообщению. |

### Бюджеты и тарификация

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `BUDGET_PERIOD` | `monthly` | string | `daily`, `weekly`, `monthly` или `total`. |
| `USER_BUDGETS` | `*` | csv / `*` | Лимит на пользователя. `*` снимает лимит. |
| `GUEST_BUDGET` | `100.0` | float | Бюджет для гостей в групповых чатах. |
| `TOKEN_PRICE` | `0.002` | float | USD за 1K токенов. |
| `IMAGE_PRICES` | `0.016,0.018,0.02` | csv float | Стоимость для разных размеров (small / medium / large). |
| `TTS_PRICES` | `0.015,0.030` | csv float | Тарифы TTS. |
| `TRANSCRIPTION_PRICE` | `0.006` | float | USD за минуту. |
| `VISION_TOKEN_PRICE` | `0.01` | float | USD за 1K vision-токенов. |

### Плагины и хранилище

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `PLUGINS` | `` | csv | Allow-list имён модулей плагинов (имя файла без `.py`). Пустой — плагины не грузятся. |
| `PLUGIN_STRICT_VALIDATION` | `false` | bool | Если `true`, дублирующиеся имена функций между плагинами роняют старт; если `false` — лог + пропуск второй регистрации. |
| `PLUGIN_STORAGE_ROOT` | `<repo>/data` | path | Корень состояния плагинов (skills workdir, висящие вопросы агента, MCP-конфиг, conversation analytics, …). Создаётся при старте. |
| `PLUGIN_MENU_PAGE_SIZE` | `8` | int | Размер страницы для меню `/plugins`. |

### Agent tools / субагенты

Плагин `agent_tools` хранит ряд жёстких лимитов прямо в коде; они не
конфигурируются через env, кроме указанных ниже. См.
[`bot/plugins/agent_tools.py`](bot/plugins/agent_tools.py).

| Параметр | Значение | Тип | Источник |
|---|---|---|---|
| `AGENT_ASK_USER_TIMEOUT_SECONDS` | `1800` | int (env) | Сколько секунд агент ждёт ответ в `ask_telegram_user`. |
| `MAX_SUBAGENTS` | `5` | константа | Максимум субагентов на один `run_subagents`. |
| `MIN_SUBAGENT_TOOL_ROUNDS` / `MAX_SUBAGENT_TOOL_ROUNDS` | `10` / `50` | константа | Лимит tool-раундов на субагента, кламп в этот диапазон. |
| `DELIVERY_DEDUP_WINDOW_SECONDS` | `60` | константа | Окно идемпотентности для `deliver_to_user`. |
| `DELIVERY_MAX_ARTIFACT_BYTES` | `49 МиБ` | константа | Жёсткий потолок на артефакты в `deliver_to_user`. |
| `TASKS_TTL_SECONDS` | `2 дня` | константа | TTL, после которого задачи плана удаляются. |
| `SUBAGENT_BLOCKED_FUNCTIONS` | 4 self-protection-инструмента | константа | Какие инструменты субагент вызвать не может (`agent_tools.run_subagents`, `…ask_telegram_user`, `…deliver_to_user`, `…cancel_pending_question`). |

### Плагин Skills

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `SKILLS_DIR` | `<storage_root>/skills` | path | Папка, в которой ищутся `SKILL.md`. |
| `SKILLS_WORKDIR` | `<storage_root>/skill_workdir` | path | Рабочая директория для скриптов скилов. |
| `SKILLS_ALLOW_SCRIPTS` | `false` | bool | Включить `skills.run_skill_script`. |
| `SKILLS_ALLOW_INSTALLS` | `true` | bool | Включить `skills.install_skill` для установки пакетов из `npx skills find/add`; поставь `false`, чтобы запретить установки. |
| `SKILLS_INSTALL_ADMIN_USER_IDS` | `*` | csv | User ID, которым разрешено устанавливать новые скилы. `*` — всем пользователям. |
| `SKILLS_INSTALL_TIMEOUT` | `180` | int | Таймаут операций `npx skills` search/install в секундах. |
| `SKILLS_SCRIPT_TIMEOUT` | `120` | int | Таймаут скрипта в секундах. |
| `SKILLS_SCRIPT_OUTPUT_MAX_CHARS` | `12000` | int | Максимум символов вывода скрипта. |
| `SKILLS_SCRIPT_INTERIM_AFTER_SECONDS` | `20` | int | После скольких секунд бот шлёт «всё ещё работаю». |
| `SKILLS_SCRIPT_ADMIN_USER_IDS` | `` | csv | User ID, которым разрешено запускать скрипты. `*` — всем админам. |

> **Безопасность.** Скрипты скилов запускаются с правами процесса бота.
> Включай `SKILLS_ALLOW_SCRIPTS=true` только с доверенными скилами и
> доверенными операторами.
> Установка скилов тоже пишет в файловую систему бота и вызывает
> `npx skills`; ограничь `SKILLS_INSTALL_ADMIN_USER_IDS` или поставь
> `SKILLS_ALLOW_INSTALLS=false`, если установки не должны быть доступны всем.

### Плагин MCP

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `MCP_SERVERS_ALLOWED_USERS` | `*` | csv | User ID, которым разрешено пользоваться MCP-серверами. |
| `DEFAULT_MCP_SERVERS` | `` | csv | Пары `name:url` через запятую, регистрируются при старте. |
| `MCP_REQUEST_TIMEOUT` | `30` | int | Таймаут одного запроса в секундах. |

> Плагин гейтится через `PLUGINS=mcp_server`, отдельного enable-флага нет.

### Hindsight (долговременная память)

Hindsight включается автоматически, когда заданы оба
`HINDSIGHT_BASE_URL` и `HINDSIGHT_API_TOKEN`.

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `HINDSIGHT_BASE_URL` | `` | url | Base URL сервиса Hindsight. |
| `HINDSIGHT_API_TOKEN` | `` | string | API-токен Hindsight. |
| `HINDSIGHT_NAMESPACE` | `default` | string | Namespace в Hindsight. |
| `HINDSIGHT_BANK_PREFIX` | `telegram-` | string | Префикс для имени банка памяти `telegram-<user_id>`. |
| `HINDSIGHT_AUTO_RECALL` | `true` | bool | Recall на первом сообщении сессии. |
| `HINDSIGHT_AUTO_SAVE` | `true` | bool | Запускать извлечение фактов + retain при удалении сессии. |
| `HINDSIGHT_RECALL_BUDGET` | `mid` | string | `low`, `mid` или `high`. |
| `HINDSIGHT_RECALL_MAX_TOKENS` | `4096` | int | Максимум токенов, которые подмешиваются из recall. |
| `HINDSIGHT_MEMORY_TYPES` | `world,experience` | csv | Типы памяти, которые надо recall’ить. |
| `HINDSIGHT_ASYNC_STORE` | `true` | bool | Запускать retain в фоне. |
| `HINDSIGHT_TIMEOUT` | `30` | float | Таймаут одного запроса в секундах. |
| `HINDSIGHT_MAX_AUTO_SAVE_ITEMS` | `5` | int | Максимум фактов, сохраняемых на удаление сессии. |

### Ключи отдельных плагинов

Используются конкретными плагинами и нужны только если соответствующий
плагин включён в `PLUGINS`.

| Переменная | Плагин | Назначение |
|---|---|---|
| `JINA_API_KEY` | `jina_web_search` | Ключ Jina Search API. |
| `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | `google_web_search` | Google Programmable Search. |
| `DEEPL_API_KEY` | `deepl` | Перевод DeepL. |
| `WOLFRAM_APP_ID` | `wolfram_alpha` | App ID WolframAlpha. |
| `WORLDTIME_DEFAULT_TIMEZONE` | `worldtimeapi` | Дефолтный часовой пояс (например, `Europe/Moscow`). |
| `TMDB_API_KEY` | `movie_info` | Ключ TMDb. |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` | `spotify` | OAuth Spotify. |
| `EDAMAM_APP_ID`, `EDAMAM_APP_KEY` | `chief` | Edamam (рецепты). |
| `ANYTHINGLLM_BASE_URL`, `ANYTHINGLLM_API_KEY` | `text_document_qa` | API workspace AnythingLLM. |
| `ANYTHINGLLM_TIMEOUT` (`120`) | `text_document_qa` | Таймаут запроса в секундах. |
| `ANYTHINGLLM_CHAT_MODE` (`query`) | `text_document_qa` | Chat-mode, передаваемый в AnythingLLM. |
| `ANYTHINGLLM_TOP_N` (`6`) | `text_document_qa` | Top-N результатов поиска. |
| `ANYTHINGLLM_SIMILARITY_THRESHOLD` (`0.25`) | `text_document_qa` | Порог сходства. |
| `ANYTHINGLLM_VECTOR_SEARCH_MODE` (`rerank`) | `text_document_qa` | Режим поиска. |
| `ANYTHINGLLM_WORKSPACE_PREFIX` (`telegram-chat`) | `text_document_qa` | Один workspace на чат, имя начинается с этого префикса. |

### Настройка БД

SQLite-слой настраивается через env. См.
[`bot/database.py`](bot/database.py).

| Переменная | По умолчанию | Тип | Назначение |
|---|---|---|---|
| `DB_PATH` | `bot/user_data.db` | path | Файл SQLite. |
| `SQLITE_TIMEOUT` | `5.0` | float | Таймаут открытия соединения в секундах. |
| `SQLITE_JOURNAL_MODE` | `WAL` | string | Журнальный режим, выставляется на каждое новое соединение. |
| `SQLITE_BUSY_TIMEOUT_MS` | `5000` | int | Значение `PRAGMA busy_timeout`. |

### Устаревшие переменные

| Переменная | Замена |
|---|---|
| `MONTHLY_USER_BUDGETS` | `USER_BUDGETS` + `BUDGET_PERIOD=monthly`. |
| `MONTHLY_GUEST_BUDGET` | `GUEST_BUDGET` + `BUDGET_PERIOD=monthly`. |

Обе всё ещё работают (читаются как fallback), но при старте логируется
deprecation warning.

---

## Telegram UX

### Слэш-команды

Регистрируются в [`bot/telegram_bot.py`](bot/telegram_bot.py)
`post_init()`. Финальный список команд, видимый в клиентах Telegram,
выставляется через `Bot.set_my_commands()`.

| Команда | Скоуп | Назначение |
|---|---|---|
| `/start`, `/help` | приват + группы | Показать справку. |
| `/reset` | приват + группы | Открыть меню сессий (preview, switch, delete, change mode, export). |
| `/stats` | приват + группы | Статистика по токенам / изображениям / TTS / транскрипции и инфа по сессии. |
| `/resend` | приват + группы | Переотправить последний промпт. |
| `/plugins` | приват + группы | Открыть пагинированное меню плагинов (команды + кнопки). |
| `/settings` | приват + группы | Открыть пользовательские настройки: язык, TTS-модель/голос, переключатели плагинов и скилов. |
| `/image <prompt>` | приват + группы | Сгенерировать изображение (только при `ENABLE_IMAGE_GENERATION=true`). |
| `/tts <text>` | приват + группы | Синтезировать речь (только при `ENABLE_TTS_GENERATION=true`). |
| `/chat <prompt>` | только группы | Явно адресовать бота в групповом чате. |

Плагины могут регистрировать слэш-команды, обработчики кнопок, message-фильтры,
pre-chat prompt handlers и секции `/help` через plugin contract. Команды с
`add_to_menu=True` также показываются в меню `/plugins`. Плагин
`text_document_qa`, если включён, добавляет `/rag` как единственную UX-команду
RAG.

### Inline-меню

| Меню | Pattern | Назначение |
|---|---|---|
| Меню сессий | `^session` | Кнопки: preview активной, новая сессия, switch, delete, change mode, export, close. |
| Picker режима | `^prompt`, `^promptgroup`, `^promptback` | Двухуровневый выбор (группа → режим), запускается из меню сессий. |
| Меню плагинов | `^pluginmenu:` | Пагинированный список плагинов → их команд. Команда либо запускается сразу, либо запрашивает аргументы через force-reply, который ловит `handle_plugin_menu_args_reply()`. |
| Меню настроек | `^settings` | Пользовательские настройки. TTS-модель и голос загружаются из API; переключатели плагинов и скилов отключают доступ только для этого пользователя. |
| Inline-query response | `^gpt:` | Одна кнопка в результате inline-query, асинхронно подтягивает ответ. |

### Обработчики медиа

Бот регистрирует встроенные tail-фильтры и позволяет включённым плагинам
регистрировать свои message-фильтры до catch-all текстового обработчика:

| Фильтр | Хендлер | Поведение |
|---|---|---|
| `filters.PHOTO`, `filters.Document.IMAGE` | `vision()` | Vision над изображением или роутинг в image-edit, если намерение совпадает. |
| `filters.AUDIO`, `filters.VOICE`, `filters.Document.AUDIO`, `filters.VIDEO`, `filters.VIDEO_NOTE`, `filters.Document.VIDEO` | `transcribe()` | Транскрипция (для видео извлекается аудио). Учитывает `IGNORE_GROUP_TRANSCRIPTIONS`. |
| `filters.Document.*` (txt, doc, docx, pdf, rtf, odt, md) | handler, зарегистрированный плагином | `text_document_qa` регистрирует загрузку документов, когда плагин включён. |
| `filters.REPLY & filters.TEXT` | `handle_plugin_menu_args_reply()` | Ловит ответы на force-reply-промпты плагинов; остальные reply передаёт в `prompt()`. Reply на non-image документы скачивает файл во временный локальный путь для tool-based редактирования / анализа. |
| `filters.TEXT & ~filters.COMMAND & ~filters.REPLY` | `prompt()` | Catch-all обработчик нереплайного текста. |

### Inline-режим

`@<bot> query` триггерит inline-query-ответ с одной кнопкой
«🤖 Answer with ChatGPT». Ответ генерируется асинхронно, и inline-сообщение
обновляется in-place, как только готово.

---

## Сессии и чат-режимы

Состояние разговора хранится в SQLite. Пользователь может держать до
`MAX_SESSIONS` именованных сессий; в каждой:

- активный системный промпт,
- сообщения user / assistant,
- любой Hindsight-контекст, подмешанный в эту сессию,
- счётчик user-сообщений (`message_count`).

Жизненный цикл сессии:

1. Новые сессии создаются автоматически на первом промпте или из меню
   сессий.
2. При достижении лимита `delete_oldest_session()` удаляет самую старую
   до создания новой.
3. Сессии можно переименовывать; долгие сессии получают авто-имя через
   light-модель.
4. Удаление сессии триггерит Hindsight auto-save (если настроен) — меню
   закрывается сразу, а извлечение и retain бегут в фоне.

**Чат-режимы** определены в [`bot/chat_modes.yml`](bot/chat_modes.yml) и
загружаются через [`bot/chat_modes_registry.py`](bot/chat_modes_registry.py).
В каждом режиме указаны:

- отображаемое имя (с эмодзи),
- системный промпт,
- список `tools` (plugin ID или `["All"]`),
- температура,
- опциональная группа для пикера.

`OpenAIHelper` валидирует `tools` против набора загруженных плагинов
при инициализации; отсутствующие имена логируются. Когда готовится запрос,
активный режим ограничивает `allowed_plugins` (по умолчанию `["All"]`,
если режим не задаёт `tools`). Wiring — в `bot/openai_helper.py:520`.
Пользовательские отключения плагинов из `/settings` применяются после allow-list
чат-режима, то есть могут только сузить набор доступных tools.

Текущий каталог чат-режимов сгруппирован так:

- **Общие**: `assistant`, `text_improver`, `travel_guide`, `content_creator`,
  `technical_writer`, `summary_assistant`, `primitives`.
- **Программирование**: `code_assistant`, `sql_assistant`.
- **Обучение**: `artist`, `english_tutor`, `psychologist`, `movie_expert`.
- **Бизнес**: `code_interpreter`, `startup_idea_generator`, `money_maker`,
  `accountant`, `project_manager`, `meta_writer`, `chief_assistant`.
- **Агенты**: `skills_agent`.

Имена в `tools` должны совпадать с **plugin ID** (имена файлов без `.py`),
а не с человекочитаемыми именами из `get_source_name()`.

---

## Каталог плагинов

Плагины живут в `bot/plugins/` и наследуются от
[`bot.plugins.plugin.Plugin`](bot/plugins/plugin.py). Каждый объявляет:

- `plugin_id` (стабильный ID; по умолчанию — имя файла без `.py`),
- `function_prefix` (неймспейс инструментов; по умолчанию равен `plugin_id`),
- `get_source_name()` (отображаемое имя),
- `get_spec()` (список OpenAI-совместимых function-спеков),
- async `execute(function_name, helper, **kwargs)`.

`PluginManager.get_functions_specs()` пробрасывает каждый спек в
`<function_prefix>.<name>`. Аргументы tool-call’ов валидируются по
JSON-Schema до выполнения; чат-режимы могут дополнительно сужать набор.

Включи плагины через allow-list `PLUGINS`:

```env
PLUGINS=ddg_web_search,ddg_image_search,deepl,gtts_text_to_speech,auto_tts,\
website_content,stable_diffusion,github_analysis,youtube_transcript,\
prompt_perfect,reminders,show_me_diagrams,chief,vkusvill,codeinterpreter,\
mcp_server,text_document_qa,hindsight_memory,skills,terminal,agent_tools,\
agent_cron,web_research
```

### Веб и поиск

| Плагин | Инструменты | Заметки |
|---|---|---|
| `ddg_web_search` | `web_search` | Gateway-поиск (`llmgateway/web-search`), поддержка hint’а языка. |
| `ddg_image_search` | `search_images` | Gateway image-search. |
| `google_web_search` | `web_search` | Google Programmable Search; нужны `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`. |
| `jina_web_search` | `web_search` | Jina Search; нужен `JINA_API_KEY`. |
| `website_content` | `website_content` | Gateway-ридер страниц (`llmgateway/web-read`); YouTube тоже идёт через gateway. |
| `web_research` | `research_articles` | Многошаговое исследование; light-модель выбирает `web-research` или `web-deep-research`. |
| `webshot` | `screenshot_website` | Скриншоты через thum.io. |

### Перевод

| Плагин | Инструменты | Заметки |
|---|---|---|
| `deepl` | `translate` | Автодетект DeepL Free/Pro. |
| `ddg_translate` | `translate` | DuckDuckGo translate (gateway-совместимый путь). |

### Изображения, видео, речь

| Плагин | Инструменты | Заметки |
|---|---|---|
| `stable_diffusion` | `stable_diffusion`, `edit_image` | Gateway-генерация и edit (Klein). |
| `haiper_image_to_video` | `generate_video` | Image-to-video через vsegpt.ru. |
| `gtts_text_to_speech` | `google_translate_text_to_speech` | Gateway Silero TTS. |
| `auto_tts` | `translate_text_to_speech` | Авто-TTS для ответов (через OpenAI-совместимое audio API). |

### Знания и данные

| Плагин | Инструменты | Заметки |
|---|---|---|
| `wolfram_alpha` | `answer_with_wolfram_alpha` | Нужен `WOLFRAM_APP_ID`. |
| `crypto` | `get_crypto_rate` | Курсы CoinCap. |
| `iplocation` | `iplocation` | Geo / ASN по IP. |
| `weather` | `get_current_weather`, `get_forecast_weather` | Open Meteo. |
| `worldtimeapi` | `worldtimeapi` | Часовые пояса; для фолбэка нужен `WORLDTIME_DEFAULT_TIMEZONE`. |
| `whois_` | `get_whois` | python-whois. |
| `text_summarizer` | `summarize_text` | Yandex-суммаризация; нужен `YANDEX_API_TOKEN`. |
| `prompt_perfect` | `optimize_prompt` | Переписывает промпты через хелпер. |
| `movie_info` | `get_new_movies`, `get_movie_recommendations` | TMDb; нужен `TMDB_API_KEY`. |
| `spotify` | `spotify_get_currently_playing_song`, `spotify_get_users_top_artists`, `spotify_get_users_top_tracks` | Нужен Spotify OAuth. |

### Документы

| Плагин | Инструменты | Заметки |
|---|---|---|
| `ask_your_pdf` | `analyze_pdf`, `upload_pdf` | Локальное извлечение текста из PDF. |
| `text_document_qa` | `upload_document`, `ask_question`, `ask_workspace`, `list_documents`, `delete_document`, `set_rag_mode`, `get_rag_status` | Один workspace AnythingLLM на чат; `/rag` открывает UX RAG-режима. |
| `youtube_transcript` | `youtube_video_transcript` | Gateway-транскрипты. |
| `youtube_audio_extractor` | `extract_youtube_audio` | Извлечение аудио через pytube. |

`/rag` открывает меню RAG-режима для текущего чата: статус, количество
документов, кнопку списка документов и переключатель включить/выключить.
Переключатель хранится в SQLite как настройка чата. Когда режим включён,
обычные текстовые сообщения в этом Telegram-чате отвечаются через AnythingLLM
workspace этого чата вместо стандартного assistant flow. Загрузка
поддерживаемого документа всегда добавляет его в тот же workspace; выключение
RAG в меню `/rag` возвращает текстовые сообщения к обычному режиму.

### Агент / автоматизация / система

| Плагин | Инструменты | Заметки |
|---|---|---|
| `agent_tools` | `manage_plan_tasks`, `ask_telegram_user`, `cancel_pending_question`, `deliver_to_user`, `run_subagents` | См. [Рантайм агентов](#рантайм-агентов--субагентов); `/background` запускает фоновые agent jobs с доставкой в тот же чат. |
| `agent_cron` | `create_cron_job` | Отдельные scheduled agent tasks через `/cron add <schedule> \| <prompt>`, list/pause/resume/run/remove. |
| `skills` | `list_skills`, `get_skill`, `find_installable_skills`, `install_skill`, `get_skill_status`, `list_active_skills`, `activate_skill`, `deactivate_skill`, `update_skill_progress`, `record_skill_reflection`, `run_skill_script` | См. [Рантайм skills](#рантайм-skills). |
| `mcp_server` | динамические | См. [Интеграция с MCP](#интеграция-с-mcp). |
| `hindsight_memory` | `recall`, `list_memories`, `stats` | Ручная инспекция Hindsight; `/memory` показывает статус, поиск, экспорт и очистку, когда Hindsight настроен. |
| `terminal` | `terminal` | Прямой shell; пара к `skills.run_skill_script`. |
| `codeinterpreter` | `deep_analysis` | Sandboxed Python с pandas / numpy / matplotlib / plotly. |
| `github_analysis` | `analyze_github_code` | Читает и суммирует репозитории GitHub с подсветкой. |
| `show_me_diagrams` | `create_diagram` | Диаграммы PlantUML; нужен Java + `bot/plugins/plantuml.jar`. |
| `chief` | `get_recipe`, `plan_menu` | Рецепты и меню Edamam. |
| `vkusvill` | `shops`, `products_search`, `recipes`, `basket_create_link` | Интеграция с ВкусВиллом. |
| `dice` | `send_dice` | Анимированные кубики Telegram. |
| `reaction` | `react_with_emoji` | Реакции на ответы ассистента. |
| `reminders` | `set_reminder`, `list_reminders`, `delete_reminder` | Напоминания, persistent JSON. |
| `task_management` | `create_task`, `list_tasks`, `update_task` | Пользовательский task-list (отдельный от `agent_tools.manage_plan_tasks`). |
| `language_learning` | `daily_practice`, `track_progress` | Vocabulary / grammar / conversation. |
| `conversation_analytics` | `analyze_conversation`, `get_personalized_recommendations` | Скользящая аналитика разговоров. |

> **Замечание.** Имена tool-функций неймспейснутые. Например,
> `deepl.translate` и `ddg_translate.translate` сосуществуют без коллизии.

---

## Рантайм агентов / субагентов

Плагин `agent_tools` накладывает на обычный function-calling-цикл петлю
субагентов, трекинг плана, идемпотентную доставку артефактов и структурный
механизм «спросить пользователя».

Ключевые инструменты:

- **`run_subagents`** — параллельно запускает до 5 ограниченных субагентов.
  Каждый идёт по собственному tool-циклу с `max_rounds`, заклампленным в
  `[10, 50]`, и использует `bot/prompts/subagent_system.md` как системный
  промпт. Субагенты наследуют `allowed_plugins` родителя и не могут
  рекурсивно вызывать `run_subagents`, `ask_telegram_user`,
  `deliver_to_user`, `cancel_pending_question`. Финальный вывод субагента
  должен кратко возвращать родителю выводы, проверенные evidence, изменённые
  файлы, артефакты и оставшиеся риски.
- **`ask_telegram_user`** — посылает пользователю вопрос и ждёт до
  `AGENT_ASK_USER_TIMEOUT_SECONDS` ответа в Telegram (текст или inline-кнопки).
  Висящие вопросы персистятся под `data/`, поэтому рестарт не теряет промпт.
- **`cancel_pending_question`** — программно закрывает висящий вопрос.
- **`deliver_to_user`** — публикует артефакт в чат пользователя
  (текст, файл, изображение, аудио, …). Плагин обеспечивает:
  - окно идемпотентности 60 секунд, ключ — scope;
  - потолок размера 49 МиБ;
  - whitelist допустимых типов артефактов и MIME.
  При `defer=true` model loop выходит коротко, артефакт доставляется без
  повторного входа в модель.
- **`manage_plan_tasks`** — единый инструмент трекинга плана для текущего
  scope с `action ∈ {add, update, list, clear}`, опциональным Definition of
  Done-контрактом (`goal`, `success_criteria`, `verification`, `constraints`)
  и DAG задач. У каждой задачи есть `id` вида `T1`, свободный `content`,
  `status` и опциональные зависимости `depends_on`. Статусы: `pending`,
  `in_progress`, `completed`, `cancelled`. Runtime-состояние хранится только в
  SQLite (`agent_plan_tasks` и `agent_plan_contracts`). Задачи старше
  `TASKS_TTL_SECONDS` (2 дня) удаляются при загрузке.

Telegram UX:

- `/background <prompt>` запускает agent job вне текущего request path и
  доставляет результат обратно в тот же чат.
- `/background list`, `/background status <job_id>`,
  `/background cancel <job_id>` и `/background clear` управляют такими jobs.

Отдельный плагин `agent_cron` отвечает за scheduled agent tasks:
`/cron add daily at 09:00 | пришли утренний бриф`, `/cron list`,
`/cron pause <job_id>`, `/cron resume <job_id>`, `/cron run <job_id>` и
`/cron remove <job_id>`.

Каждый вызов plugin tool также записывается в SQLite `tool_call_events`:
function name, owning plugin, chat/user/request id при наличии, status,
duration, текст ошибки и direct-result flag. Это операционная telemetry; она
не меняет tool result, который видит модель.

Субагентам также доступен инструмент `agent_tools.internal_publish`
(вкладывается автоматически), через который они отдают находки или
локальные файловые пути в список `published` родительского агента,
ничего не показывая пользователю.

План также рендерится в реальном времени внутри сообщения «Готовлю ответ…»,
которое бот поддерживает на всё время обработки запроса. См.
[`bot/utils.py`](bot/utils.py) — `BusyStatusMessage` принимает callable
`plan_provider`, который проксируется в
[`bot/telegram_bot.py`](bot/telegram_bot.py)
`_build_plan_status_provider()` и тянет задачи активного плагина
`agent_tools` каждые 5 секунд.

---

## Рантайм skills

Плагин `skills` экспонирует локальные скилы в стиле Codex (одна папка на
скил с файлом `SKILL.md`, описывающим намерение, и опциональными
Python-скриптами) как инструменты, которые модель может вызывать.

Инструменты:

- `list_skills` — список доступных скилов с краткими описаниями.
- `get_skill` — полное содержимое `SKILL.md` одного скила.
- `find_installable_skills` — поиск installable skills через
  `npx skills find`.
- `install_skill` — установка подтверждённого пакета через `npx skills add`,
  копирование в `SKILLS_DIR` и refresh локального реестра.
- `get_skill_status` / `list_active_skills` — посмотреть, какие скилы
  активны в текущем chat-scope.
- `activate_skill` / `deactivate_skill` — пометить скил как in-context
  для текущей сессии (используется в чат-режиме `skills_agent`).
- `run_skill_script` — выполнить скрипт `skills/<skill>/scripts/*.py`
  (гейтится через `SKILLS_ALLOW_SCRIPTS=true` и allow-list
  `SKILLS_SCRIPT_ADMIN_USER_IDS`).
- `update_skill_progress` — записать прогресс по скилу для будущих сессий.
- `record_skill_reflection` — накапливает повторяющиеся предложения по
  улучшению после сбоев скила; четвёртое одинаковое предложение добавляется
  в `SKILL.md` как learned clarification.

Watchdog обновляет typing-индикатор каждые 4 секунды, пока крутится
скрипт; через `SKILLS_SCRIPT_INTERIM_AFTER_SECONDS` (по умолчанию 20 с)
бот шлёт «всё ещё работаю».

Если включён плагин `terminal`, `terminal.terminal` — предпочтительный путь
для скриптов скилов, которым нужен реальный shell; `codeinterpreter`
оставлен под аналитические / численные задачи.

В `bot/chat_modes.yml` есть готовый чат-режим `skills_agent`, доводящий
этот сценарий до пользователя.

---

## Интеграция с MCP

Плагин `mcp_server` действует как MCP-**клиент**: подключается к одному или
нескольким удалённым MCP-серверам, забирает их каталог инструментов и
пробрасывает каждый удалённый инструмент модели под неймспейснутым ID
(вид `mcp_<server>_<tool>`).

- Авторегистрация серверов: `DEFAULT_MCP_SERVERS=name1:url1,name2:url2`.
- Доступ ограничивается `MCP_SERVERS_ALLOWED_USERS`.
- Таймаут запроса: `MCP_REQUEST_TIMEOUT`.

Примеры MCP-серверов лежат в [`examples/`](examples/) (`mcp_server_example.py`
и `mcp_stdio_server.py`) — хороший старт. Полный контракт плагина — в
[`bot/README_MCP.md`](bot/README_MCP.md).

---

## Память Hindsight

Когда настроен, Hindsight используется в двух местах:

1. **Auto-recall**: на первом сообщении сессии бот тянет релевантные
   воспоминания из банка `telegram-<user_id>` и кладёт их в контекст
   сессии так, чтобы они пережили компактизации истории.
2. **Auto-save**: при удалении сессии бот делает снапшот удаляемой сессии,
   возвращает управление Telegram и в фоне запускает извлечение фактов и
   retain (с учётом `HINDSIGHT_MAX_AUTO_SAVE_ITEMS`).

Извлечение фактов сознательно строгое: оно сохраняет долгоиграющие
предпочтения, текущие проекты, долгосрочные ограничения и явные «запомни это»,
но пропускает разовые edit’ы изображений, отдельные технические вопросы,
случайный поиск и секреты.

Плагин `hindsight_memory` (`recall`, `list_memories`, `stats`) доступен
как обычный инструмент, когда нужна ручная инспекция памяти. Когда плагин
включён и Hindsight настроен, `/memory` и Settings показывают статус,
количество воспоминаний, ручной поиск, экспорт и очистку.

---

## Расположение хранилища

```
.
├── bot/
│   ├── chat_modes.yml          # Каталог чат-режимов
│   ├── plugins/                # Встроенные плагины (~43 модуля)
│   ├── prompts/
│   │   └── subagent_system.md  # Системный промпт субагентов
│   ├── user_data.db            # SQLite (сессии, история, image-refs, usage)
│   ├── …
├── data/                       # Дефолтный PLUGIN_STORAGE_ROOT
│   ├── skills/                 # Определения скилов (если не переопределён SKILLS_DIR)
│   ├── skill_workdir/          # Workdir для скриптов скилов
│   ├── agent_pending_questions.json
│   ├── conversation_analytics/
│   ├── document_metadata/
│   ├── language_data/
│   ├── pdf_cache/
│   └── …
├── examples/                   # Пример MCP-сервера / клиента
├── tests/                      # Основной набор pytest
├── requirements.txt
├── README.md                   # Английская версия
└── README.ru.md                # Русская версия (этот файл)
```

Оба `bot/user_data.db*` и `data/` исключены через `.gitignore`. Плагины,
которым нужно состояние, должны всегда использовать `self.storage_root`
(выдаётся `PluginManager`), а не писать прямо в `bot/plugins/`.

---

## Разработка

В `pytest.ini` выставлен `asyncio_mode = auto`. Основные тесты лежат в
`tests/`, MCP-специфичные — в `bot/tests/`.

Запустить всё:

```bash
python -m pytest -q tests/
python -m pytest -q bot/tests/
```

Полезные узкие наборы:

| Область | Тесты |
|---|---|
| Реестр плагинов / специ | `tests/test_plugin_manager.py`, `tests/test_plugin_arg_validation.py`, `tests/test_plugin_commands.py` |
| Direct-results плагинов | `tests/test_plugin_direct_results.py` |
| Контракт chat-id плагинов | `tests/test_plugin_chat_id_contract.py` |
| Регистрация хендлеров плагинов | `tests/test_plugin_handlers_registration.py`, `tests/test_plugin_menu_force_reply.py` |
| Маршрутизация tool-call’ов | `tests/test_openai_helper_tool_calls.py`, `tests/test_concurrent_tool_state.py` |
| Pre-conversation сериализация | `tests/test_per_conversation_serialization.py` |
| Стриминг | `tests/test_telegram_streaming.py` |
| Группы и сессии | `tests/test_group_session_flow.py` |
| Авторизация callback’ов | `tests/test_callback_authorization.py` |
| Конфиг Telegram-builder | `tests/test_telegram_builder_config.py` |
| Реестр чат-режимов | `tests/test_chat_modes_registry.py` |
| SQLite-сессии и контекст | `tests/test_database.py` |
| Маршрутизация LLMGateway | `tests/test_llm_gateway_routing.py`, `tests/test_llm_gateway_client.py` |
| Hindsight | `tests/test_hindsight_memory.py`, `tests/test_hindsight_client.py` |
| Async-плагины | `tests/test_async_plugins.py` |
| Контракт `RequestContext` | `tests/test_request_context.py`, `tests/test_conversation_key.py` |
| Agent tools | `tests/test_agent_tools_plugin.py` |
| Skills | `tests/test_skills_plugin.py` |
| `text_document_qa` (AnythingLLM) | `tests/test_text_document_qa_anythingllm.py` |
| `vkusvill` | `tests/test_vkusvill_plugin.py` |
| `ask_your_pdf` | `tests/test_ask_your_pdf.py` |
| `codeinterpreter` | `tests/test_codeinterpreter_plugin.py` |
| MCP | `bot/tests/test_mcp_server.py` |

Быстрая синтаксическая проверка крупных модулей:

```bash
python -m py_compile bot/openai_helper.py bot/telegram_bot.py
```

Общие правила для агентов и кодбазный граф — в [`AGENTS.md`](AGENTS.md)
(на него ссылается `CLAUDE.md`).

---

## Диагностика

### Процесс сразу падает на старте

Задай `TELEGRAM_BOT_TOKEN` и `OPENAI_API_KEY`. Обе обязательны, без них
процесс выходит с понятным сообщением в лог.

### Запросы в gateway падают

- Проверь, что `OPENAI_BASE_URL` указывает на `/v1`-base твоего gateway.
- Проверь, что `OPENAI_API_KEY` — это ключ gateway.
- Проверь, что выбранный model id начинается с `llmgateway/` (или того
  префикса, который требует твой gateway).

### Бот шлёт «Готовлю ответ…», но не отвечает

- Убедись, что локальный Telegram Bot API запущен, или поставь
  `TELEGRAM_LOCAL_MODE=false`, чтобы пойти на хостируемое API.
- Проверь логи на ошибки tool-call’ов. С `SHOW_PLUGINS_USED=true`
  бот сам пишет, какой плагин упал.

### Reply на изображение генерирует новое вместо edit’а

Бот определяет намерение в reply через light-модель. Делай reply прямо
на сообщение Telegram с картинкой, чтобы классификатор увидел
`replied_message_kind="image"` и выбрал между `image_edit` и
`image_describe`.

### Reply на файл говорит, что файл удалён из `/tmp`

Делай reply прямо на Telegram-сообщение с документом. Для reply на non-image
документы бот скачивает файл через Telegram, добавляет временный `local_path`
в запрос к модели и удаляет эту исходную копию после хода. В активном
чат-режиме всё равно должны быть включены инструменты, умеющие работать с
файлами, например `terminal`, `agent_tools`, `skills` или форматный плагин,
чтобы изменить файл и отправить обновлённый артефакт.

### Hindsight видит документы, но без memory

Hindsight хранит сырые документы и сам гонит экстракшн-пайплайн.
Смотри банковские эндпоинты `documents`, `memories/list`, `operations`,
`stats` напрямую. Бот шлёт только извлечённые факты, document→memory
он не контролирует.

### Коллизия схем плагинов

При `PLUGIN_STRICT_VALIDATION=true` регистрация двух плагинов с одинаковым
полным именем функции роняет старт с `ValueError: Duplicate function name …`.
Либо переименуй конфликтующую функцию, либо поставь переменную в `false` —
тогда поведение скатывается к log-and-skip.

### Скрипты скилов не запускаются

- Проверь, что `SKILLS_ALLOW_SCRIPTS=true` и user id вызывающего лежит в
  `SKILLS_SCRIPT_ADMIN_USER_IDS`.
- Проверь, что `SKILLS_DIR` существует и содержит `SKILL.md`.

---

## Авторы и зависимости

Этот проект — сильно модифицированный потомок оригинальной линейки
[`chatgpt-telegram-bot`](https://github.com/n3d1117/chatgpt-telegram-bot).
В числе прочего использует:

- [python-telegram-bot](https://python-telegram-bot.org)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [LLMGateway-compatible OpenAI-style APIs](https://github.com/LKosoj/LLMApiGateway)
- [Hindsight](#память-hindsight) (сервис долговременной памяти)
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) (Q&A по документам)
- [PlantUML](https://plantuml.com/) (диаграммы)
- Экосистему [Model Context Protocol](https://modelcontextprotocol.io/)
