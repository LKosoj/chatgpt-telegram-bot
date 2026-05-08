# ChatGPT Telegram Bot

> 🌐 **Language / Язык:** **English** · [Русский](README.ru.md)

A Telegram bot built around an OpenAI-compatible **LLMGateway** runtime. The
project keeps the chat / image / audio / vision UX of the original
`chatgpt-telegram-bot` lineage but is now centred on gateway-routed models, an
extensive plugin system, named SQLite sessions, optional Hindsight long-term
memory, a subagent / skills runtime, and Model Context Protocol (MCP) support.

> **Status.** The codebase has diverged significantly from the upstream project.
> Default models, image/audio routing, plugin namespacing, the chat-mode
> registry, the agent/skills runtime, and the storage layout are all
> repo-specific. This README documents the current behaviour. Treat the source
> (`bot/`) and `.env.example` as authoritative when in doubt.

---

## Table Of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
   - [Required Variables](#required-variables)
   - [Telegram Core](#telegram-core)
   - [LLMGateway / OpenAI Core](#llmgateway--openai-core)
   - [Image, Vision, TTS, Transcription](#image-vision-tts-transcription)
   - [Sessions, Conversations, Behaviour](#sessions-conversations-behaviour)
   - [Budgets And Pricing](#budgets-and-pricing)
   - [Plugins And Storage](#plugins-and-storage)
   - [Agent Tools / Subagents](#agent-tools--subagents)
   - [Skills Plugin](#skills-plugin)
   - [MCP Plugin](#mcp-plugin)
   - [Hindsight Long-Term Memory](#hindsight-long-term-memory)
   - [Plugin-Specific Keys](#plugin-specific-keys)
   - [Database Tuning](#database-tuning)
   - [Deprecated Variables](#deprecated-variables)
6. [Telegram UX](#telegram-ux)
   - [Slash Commands](#slash-commands)
   - [Inline Menus](#inline-menus)
   - [Media Handlers](#media-handlers)
   - [Inline Mode](#inline-mode)
7. [Sessions And Chat Modes](#sessions-and-chat-modes)
8. [Plugins Catalogue](#plugins-catalogue)
9. [Agent / Subagent Runtime](#agent--subagent-runtime)
10. [Skills Runtime](#skills-runtime)
11. [MCP Integration](#mcp-integration)
12. [Hindsight Memory](#hindsight-memory)
13. [Storage Layout](#storage-layout)
14. [Development](#development)
15. [Troubleshooting](#troubleshooting)
16. [Credits](#credits)

---

## Features

- **Chat with named SQLite sessions**, per-user history, configurable
  `MAX_HISTORY_SIZE` / `MAX_CONVERSATION_AGE_MINUTES`, automatic session naming,
  inline session-management menus, optional auto-suggested chat modes.
- **Three-tier model routing** through LLMGateway:
  - `llmgateway/high` — main assistant.
  - `llmgateway/light_model` — routing, classification, naming, intent detection.
  - `llmgateway/big_context` — large-context tasks and vision fallbacks.
- **Images**: generation via `llmgateway/ai-klein-generation`, edits via
  `llmgateway/ai-klein-edit`, intent-routing of replies/captions to either
  generate, edit, or run vision over the referenced image.
- **Vision**: image description via the configured `VISION_MODEL` with optional
  follow-up Q&A.
- **Audio**: voice / video / video-note / audio-document transcription through
  the configured transcription model; optional voice replies; LLMGateway Silero
  TTS by default (voice `kseniya`, format `wav`).
- **Web & research**: gateway-backed search, page reads, regular research, deep
  research; YouTube transcript path; web research auto-picks regular vs. deep
  using the light model.
- **Plugin system** with namespaced tools (`<plugin_id>.<function>`), JSON-schema
  argument validation, per-mode allow-lists, plugin-contributed slash commands
  and inline menus, and automatic background task plumbing.
- **Reply-to-file workflows**: when the user replies with text to a Telegram
  non-image document, the bot downloads that file for the current turn and gives
  the model a temporary local path so file-capable tools can edit, convert, or
  analyze it and return a new artifact. Image replies keep the dedicated
  image-edit / vision routing.
- **Agent runtime**: `agent_tools` plugin exposes `run_subagents`,
  `ask_telegram_user`, `deliver_to_user`, and a unified `manage_plan_tasks`
  tool for plan tracking, with safety caps on rounds, scope, and artifact size.
- **Skills runtime**: `skills` plugin scans local Codex-style skill folders,
  exposes them as tools, and (opt-in) executes their scripts under operator
  allowlists, with watchdog typing indicators and progress streaming.
- **MCP integration**: built-in MCP client plugin connects to remote MCP servers
  and re-exposes their tools to the model.
- **Hindsight long-term memory**: optional auto-recall on first request of a
  session and auto-save on session deletion.
- **Per-user budgets** with monthly/daily/weekly periods and granular pricing
  for tokens, images, TTS, vision, transcription.
- **Telegram local-mode** is the default (`http://localhost:8081/bot`); hosted
  Telegram API is supported by toggling `TELEGRAM_LOCAL_MODE=false`.

---

## Architecture

Process entrypoint: [`bot/__main__.py`](bot/__main__.py).

Startup order:

1. Load `.env` via `python-dotenv`.
2. Validate `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` (exits if either is
   missing).
3. Build configuration dictionaries (OpenAI, Telegram, plugins).
4. Instantiate `PluginManager` (discovers `bot/plugins/*.py` and loads anything
   in the comma-separated `PLUGINS` allow-list).
5. Instantiate `Database` (singleton SQLite, WAL by default, thread-local
   connections, foreign keys).
6. Instantiate `OpenAIHelper` (LLMGateway/OpenAI-compatible client, vision and
   TTS helpers, chat-mode registry, optional Hindsight client).
7. Instantiate `ChatGPTTelegramBot` and call `run()`, which builds the PTB
   `Application` (concurrent updates enabled, local Telegram mode by default).

| Path | Responsibility |
|---|---|
| `bot/__main__.py` | Env loading, config wiring, dependency assembly, process entry |
| `bot/telegram_bot.py` | Telegram handlers, sessions UI, plugin commands, media routing, plugin/menu pagination, streaming output |
| `bot/openai_helper.py` | Chat / vision / image / audio helpers, model selection, Hindsight auto-recall and auto-save scheduling, chat-mode application |
| `bot/openai_tool_handler.py` | Tool-call extraction and execution loop, batched `asyncio.gather`, deferred / direct-result short-circuits |
| `bot/llm_gateway_client.py` | Gateway-specific endpoints (web search/read/research, image edit) |
| `bot/plugin_manager.py` | Plugin discovery, namespacing, JSON-schema validation, command/menu aggregation, subagent gating, storage root provisioning |
| `bot/plugins/plugin.py` | Base `Plugin` class (`get_source_name`, `get_spec`, async `execute`, optional commands/handlers/on_startup) |
| `bot/plugins/*.py` | Built-in plugins (see [Plugins Catalogue](#plugins-catalogue)) |
| `bot/chat_modes_registry.py` | Loads `bot/chat_modes.yml`, validates plugin tool references |
| `bot/chat_modes.yml` | User-facing chat modes, per-mode allowed plugin IDs |
| `bot/database.py` | SQLite singleton, sessions, conversation context, image refs, usage tracking |
| `bot/usage_tracker.py` | Token / image / TTS / transcription cost tracking and budget windows |
| `bot/hindsight_client.py` | Hindsight HTTP client (recall, retain, list, stats) |
| `bot/request_context.py` | Frozen `RequestContext` dataclass passed through tool execution |
| `bot/skill_script_routing.py` | Router that nudges skill scripts toward `terminal` vs. `codeinterpreter` |
| `bot/utils.py` | `BusyStatusMessage`, `compute_scope_key`, message splitting, file delivery, helpers |
| `bot/validation.py` | Generic JSON-schema-style argument validation for plugin specs |
| `bot/conversation_key.py` | `(chat_id, thread_id)` keying for per-conversation locking |
| `bot/i18n.py` | Bot localization helper |
| `bot/html_utils.py` | Markdown / HTML conversion utilities |
| `bot/model_constants.py` | Default model identifiers (`LLMGATEWAY_HIGH_MODEL`, …) |
| `bot/prompts/subagent_system.md` | System prompt for spawned subagents |
| `examples/` | Sample MCP server / client (`mcp_server_example.py`, `mcp_stdio_*.py`) |
| `tests/` | Top-level pytest suite |
| `bot/tests/` | MCP-specific tests |
| `data/` | Default plugin storage root (created on startup; ignored by git) |

The repository also keeps a generated codebase map under
`.cli-proxy/.codebase_map/` used by agent tooling.

---

## Requirements

- **Python 3.9+** (3.12 is what the project is currently developed against).
- **Telegram bot token** from @BotFather.
- **LLMGateway-compatible API endpoint** (or any OpenAI-compatible endpoint —
  the bot still uses `OPENAI_API_KEY` / `OPENAI_BASE_URL` env names).
- **`ffmpeg`** for audio / video processing.
- **Java runtime** (only if the `show_me_diagrams` plugin is enabled — it
  invokes `bot/plugins/plantuml.jar`).
- A **local Telegram Bot API server** is recommended (the bot defaults to
  `TELEGRAM_LOCAL_MODE=true`, base URL `http://localhost:8081/bot`). To use
  Telegram's hosted API, set `TELEGRAM_LOCAL_MODE=false`.

---

## Quick Start

```bash
git clone <repo>
cd chatgpt-telegram-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, OPENAI_BASE_URL
python -m bot
```

Docker Compose is also available:

```bash
docker compose up
```

The default `OPENAI_BASE_URL` is empty and must point at your gateway, e.g.
`http://gateway.example/v1`.

---

## Configuration

All configuration is environment-driven (typically a `.env` file at the repo
root, loaded via `python-dotenv`). The tables below cover every variable read
by the runtime. **Bold** rows are required.

### Required Variables

| Variable | Type | Purpose |
|---|---|---|
| **`TELEGRAM_BOT_TOKEN`** | string | Telegram Bot API token from @BotFather. The process exits at startup if missing. |
| **`OPENAI_API_KEY`** | string | LLMGateway / OpenAI-compatible API key. The process exits at startup if missing. |

### Telegram Core

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `TELEGRAM_LOCAL_MODE` | `true` | bool | Use the local Telegram Bot API server. When `false`, the bot speaks to Telegram's hosted API and ignores `TELEGRAM_BASE_URL`. |
| `TELEGRAM_BASE_URL` | `http://localhost:8081/bot` (in local mode) | url | Bot API base URL. Validated as absolute http(s). |
| `ADMIN_USER_IDS` | `-` | csv | Comma-separated Telegram user IDs with admin privileges. Use `-` for none. |
| `ALLOWED_TELEGRAM_USER_IDS` | `*` | csv | Comma-separated allow-list of user IDs. `*` allows everyone. |
| `ENABLE_QUOTING` | `true` | bool | Reply to the original message in private chats (group chats always quote). |
| `GROUP_TRIGGER_KEYWORD` | `` | string | If set, group messages must contain this keyword to address the bot. |
| `IGNORE_GROUP_TRANSCRIPTIONS` | `true` | bool | Skip auto-transcription in groups. |
| `IGNORE_GROUP_VISION` | `true` | bool | Skip vision in groups. |
| `TELEGRAM_DOWNLOAD_BOT_ID` | `` | string | Optional secondary download-bot ID for large files. |
| `TELEGRAM_DOWNLOAD_DIR` | `media` | path | Directory where downloaded media is stored. |
| `PROXY` / `TELEGRAM_PROXY` | `` | url | HTTP(S) proxy for Telegram traffic. |

### LLMGateway / OpenAI Core

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `OPENAI_BASE_URL` | `` | url | Gateway / OpenAI-compatible base URL (e.g. `http://gateway.example/v1`). |
| `OPENAI_MODEL` | `llmgateway/high` | string | Main assistant model. |
| `LIGHT_MODEL` | `llmgateway/light_model` | string | Fast model used for classification, routing, naming, intent detection. |
| `BIG_MODEL_TO_USE` | `llmgateway/big_context` | string | Large-context model (used for big history compactions and as a vision fallback). |
| `MAX_TOKENS` | model-dependent | int | Max output tokens per response. |
| `MAX_HISTORY_SIZE` | `15` | int | Max chat-history messages kept in memory before summarisation. |
| `MAX_CONVERSATION_AGE_MINUTES` | `180` | int | Age window after which conversations are reset. |
| `TEMPERATURE` | `1.0` | float | Sampling temperature. |
| `PRESENCE_PENALTY` | `0.0` | float | OpenAI presence penalty. |
| `FREQUENCY_PENALTY` | `0.0` | float | OpenAI frequency penalty. |
| `N_CHOICES` | `1` | int | Number of completions returned. |
| `STREAM` | `true` | bool | Stream chat responses to Telegram. |
| `SHOW_USAGE` | `false` | bool | Append token-usage footer to responses. |
| `SHOW_PLUGINS_USED` | `false` | bool | Append a list of plugins/tools that were called. |
| `ASSISTANT_PROMPT` | `You are a helpful assistant.` | string | Default system prompt before chat-mode application. |
| `WHISPER_PROMPT` | `` | string | Optional prompt forwarded to the transcription model. |
| `PROXY` / `OPENAI_PROXY` | `` | url | HTTP(S) proxy for OpenAI/LLMGateway traffic. |
| `PROXY_WEB` | `` | url | Proxy for ad-hoc web fetches. |
| `ENABLE_FUNCTIONS` | derived | bool | Enable tool-calling. Auto-detected from the chosen model. |
| `FUNCTIONS_MAX_CONSECUTIVE_CALLS` | `10` | int | Cap on consecutive tool-call rounds in the main loop (per assistant turn). |

### Image, Vision, TTS, Transcription

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `ENABLE_IMAGE_GENERATION` | `true` | bool | Toggle image generation and the `/image` command. |
| `IMAGE_MODEL` | `llmgateway/ai-klein-generation` | string | Image generation model. |
| `IMAGE_QUALITY` | `standard` | string | `standard` or `hd`. |
| `IMAGE_STYLE` | `vivid` | string | `vivid` or `natural`. |
| `IMAGE_SIZE` | `512x512` | string | Output size, e.g. `1024x1024`. |
| `IMAGE_FORMAT` | `photo` | string | `photo` or `document`. |
| `ENABLE_VISION` | `true` | bool | Toggle vision over photos / image documents. |
| `VISION_MODEL` | `llmgateway/big_context` | string | Vision model. |
| `VISION_PROMPT` | `What is in this image` | string | Default vision instruction. |
| `VISION_DETAIL` | `auto` | string | `low`, `high`, or `auto`. |
| `VISION_MAX_TOKENS` | `1000` | int | Max tokens for vision answers. |
| `ENABLE_VISION_FOLLOW_UP_QUESTIONS` | `true` | bool | Allow follow-up Q&A on the most recent image. |
| `ENABLE_TRANSCRIPTION` | `true` | bool | Toggle voice / audio / video transcription. |
| `TRANSCRIPTION_MODEL` | `llmgateway/whisper-large-v3` | string | Transcription model. |
| `ENABLE_TTS_GENERATION` | `true` | bool | Toggle the `/tts` command and TTS plugins. |
| `TTS_MODEL` | `llmgateway/silero-tts` | string | TTS model. |
| `TTS_VOICE` | `kseniya` | string | TTS voice. |
| `TTS_RESPONSE_FORMAT` | `wav` | string | TTS output format (`wav`, `mp3`, …). |
| `VOICE_REPLY_WITH_TRANSCRIPT_ONLY` | `false` | bool | Reply to voice messages with the transcript only. |
| `VOICE_REPLY_PROMPTS` | `` | semicolon list | Trigger phrases that force a voice-reply path. |
| `YANDEX_API_TOKEN` | `` | string | Optional Yandex token used by `text_summarizer`. |
| `ASSEMBLYAI_API_KEY` | `` | string | Optional AssemblyAI key for transcription paths. |

### Sessions, Conversations, Behaviour

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `MAX_SESSIONS` | `5` | int | Max named sessions per user. Older sessions are pruned via `delete_oldest_session()`. |
| `BOT_LANGUAGE` | `en` | string | Bot UI language (`en`, `ru`, …). |
| `AUTO_CHAT_MODES` | `false` | bool | Suggest a chat mode automatically based on the first user message. |

### Budgets And Pricing

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `BUDGET_PERIOD` | `monthly` | string | `daily`, `weekly`, `monthly`, or `total`. |
| `USER_BUDGETS` | `*` | csv / `*` | Per-user budget caps. `*` disables the cap. |
| `GUEST_BUDGET` | `100.0` | float | Budget for guests when group chats are addressed by an allowed user. |
| `TOKEN_PRICE` | `0.002` | float | USD per 1K tokens. |
| `IMAGE_PRICES` | `0.016,0.018,0.02` | csv floats | Per-size image costs (small, medium, large). |
| `TTS_PRICES` | `0.015,0.030` | csv floats | TTS cost tiers. |
| `TRANSCRIPTION_PRICE` | `0.006` | float | USD per minute. |
| `VISION_TOKEN_PRICE` | `0.01` | float | USD per 1K vision tokens. |

### Plugins And Storage

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `PLUGINS` | `` | csv | Allow-list of plugin module names (filenames without `.py`). Empty means no plugins are loaded. |
| `PLUGIN_STRICT_VALIDATION` | `false` | bool | When `true`, duplicate function names across plugins raise at registration; when `false`, duplicates are logged and the second registration is dropped. |
| `PLUGIN_STORAGE_ROOT` | `<repo>/data` | path | Root directory for plugin state (skills workdir, agent tasks, MCP config, conversation analytics, etc.). Created on startup if missing. |
| `PLUGIN_MENU_PAGE_SIZE` | `8` | int | Page size for the `/plugins` menu. |

### Agent Tools / Subagents

The `agent_tools` plugin enforces several hard limits in code; they are not
configurable through env vars unless noted. See
[`bot/plugins/agent_tools.py`](bot/plugins/agent_tools.py).

| Setting | Default | Type | Source |
|---|---|---|---|
| `AGENT_ASK_USER_TIMEOUT_SECONDS` | `1800` | int (env) | Seconds the agent waits for a Telegram answer in `ask_telegram_user`. |
| `MAX_SUBAGENTS` | `5` | constant | Max subagents per `run_subagents` call. |
| `MIN_SUBAGENT_TOOL_ROUNDS` / `MAX_SUBAGENT_TOOL_ROUNDS` | `10` / `50` | constants | Per-subagent tool-call cap, clamped to this band. |
| `DELIVERY_DEDUP_WINDOW_SECONDS` | `60` | constant | Idempotency window for `deliver_to_user`. |
| `DELIVERY_MAX_ARTIFACT_BYTES` | `49 MiB` | constant | Hard cap on artifacts pushed via `deliver_to_user`. |
| `TASKS_TTL_SECONDS` | `2 days` | constant | TTL after which agent plan tasks are pruned. |
| `SUBAGENT_BLOCKED_FUNCTIONS` | 4 self-protection tools | constant | Tool names a subagent cannot call (`agent_tools.run_subagents`, `…ask_telegram_user`, `…deliver_to_user`, `…cancel_pending_question`). |

### Skills Plugin

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `SKILLS_DIR` | `<storage_root>/skills` | path | Directory scanned for `SKILL.md` files. |
| `SKILLS_WORKDIR` | `<storage_root>/skill_workdir` | path | Working directory used when running skill scripts. |
| `SKILLS_ALLOW_SCRIPTS` | `false` | bool | Enable `skills.run_skill_script`. |
| `SKILLS_ALLOW_INSTALLS` | `true` | bool | Enable `skills.install_skill` to install packages from `npx skills find/add`; set `false` to disable installs. |
| `SKILLS_INSTALL_ADMIN_USER_IDS` | `*` | csv | User IDs allowed to install new skills. `*` allows everyone. |
| `SKILLS_INSTALL_TIMEOUT` | `180` | int | Timeout in seconds for `npx skills` search/install operations. |
| `SKILLS_SCRIPT_TIMEOUT` | `120` | int | Per-script timeout in seconds. |
| `SKILLS_SCRIPT_OUTPUT_MAX_CHARS` | `12000` | int | Max characters captured from script output. |
| `SKILLS_SCRIPT_INTERIM_AFTER_SECONDS` | `20` | int | Threshold after which a watchdog "still running" notice is sent. |
| `SKILLS_SCRIPT_ADMIN_USER_IDS` | `` | csv | User IDs allowed to invoke skill scripts. `*` allows all admins. |

> **Security.** Skill scripts run with the same OS permissions as the bot
> process. Only enable `SKILLS_ALLOW_SCRIPTS=true` with trusted skills and
> trusted operators.
> Skill installation also writes to the bot filesystem and shells out to
> `npx skills`; restrict `SKILLS_INSTALL_ADMIN_USER_IDS` or set
> `SKILLS_ALLOW_INSTALLS=false` if installs should not be available to everyone.

### MCP Plugin

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `MCP_SERVERS_ALLOWED_USERS` | `*` | csv | User IDs allowed to interact with MCP servers. |
| `DEFAULT_MCP_SERVERS` | `` | csv | Comma-separated `name:url` pairs auto-registered at startup. |
| `MCP_REQUEST_TIMEOUT` | `30` | int | Per-request timeout in seconds. |

> The plugin is gated by `PLUGINS=mcp_server`, not by a separate enable flag.

### Hindsight Long-Term Memory

Hindsight is auto-enabled when both `HINDSIGHT_BASE_URL` and
`HINDSIGHT_API_TOKEN` are set.

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `HINDSIGHT_BASE_URL` | `` | url | Hindsight service base URL. |
| `HINDSIGHT_API_TOKEN` | `` | string | Hindsight API token. |
| `HINDSIGHT_NAMESPACE` | `default` | string | Hindsight namespace. |
| `HINDSIGHT_BANK_PREFIX` | `telegram-` | string | Prefix used to build the per-user memory bank `telegram-<user_id>`. |
| `HINDSIGHT_AUTO_RECALL` | `true` | bool | Recall memories on first user message of a session. |
| `HINDSIGHT_AUTO_SAVE` | `true` | bool | Schedule fact extraction + retain on session deletion. |
| `HINDSIGHT_RECALL_BUDGET` | `mid` | string | `low`, `mid`, or `high`. |
| `HINDSIGHT_RECALL_MAX_TOKENS` | `4096` | int | Max tokens injected from recall. |
| `HINDSIGHT_MEMORY_TYPES` | `world,experience` | csv | Memory types to recall. |
| `HINDSIGHT_ASYNC_STORE` | `true` | bool | Run retain calls in the background. |
| `HINDSIGHT_TIMEOUT` | `30` | float | Per-request timeout in seconds. |
| `HINDSIGHT_MAX_AUTO_SAVE_ITEMS` | `5` | int | Max facts saved per session deletion. |

### Plugin-Specific Keys

These are read by individual plugins and are only required if the corresponding
plugin is enabled in `PLUGINS`.

| Variable | Plugin | Purpose |
|---|---|---|
| `JINA_API_KEY` | `jina_web_search` | Jina Search API key. |
| `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | `google_web_search` | Google Programmable Search. |
| `DEEPL_API_KEY` | `deepl` | DeepL translation. |
| `WOLFRAM_APP_ID` | `wolfram_alpha` | WolframAlpha App ID. |
| `WORLDTIME_DEFAULT_TIMEZONE` | `worldtimeapi` | Fallback time zone (e.g. `Europe/Moscow`). |
| `TMDB_API_KEY` | `movie_info` | TMDb API key. |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` | `spotify` | Spotify OAuth. |
| `EDAMAM_APP_ID`, `EDAMAM_APP_KEY` | `chief` | Edamam recipes. |
| `ANYTHINGLLM_BASE_URL`, `ANYTHINGLLM_API_KEY` | `text_document_qa` | AnythingLLM workspace API. |
| `ANYTHINGLLM_TIMEOUT` (`120`) | `text_document_qa` | Request timeout in seconds. |
| `ANYTHINGLLM_CHAT_MODE` (`query`) | `text_document_qa` | Chat mode passed to AnythingLLM. |
| `ANYTHINGLLM_TOP_N` (`6`) | `text_document_qa` | Top-N retrieval results. |
| `ANYTHINGLLM_SIMILARITY_THRESHOLD` (`0.25`) | `text_document_qa` | Similarity threshold. |
| `ANYTHINGLLM_VECTOR_SEARCH_MODE` (`rerank`) | `text_document_qa` | Search mode. |
| `ANYTHINGLLM_WORKSPACE_PREFIX` (`telegram-chat`) | `text_document_qa` | One workspace per Telegram chat, prefixed with this string. |

### Database Tuning

The SQLite layer is configured via env. See
[`bot/database.py`](bot/database.py).

| Variable | Default | Type | Purpose |
|---|---|---|---|
| `DB_PATH` | `bot/user_data.db` | path | SQLite database file. |
| `SQLITE_TIMEOUT` | `5.0` | float | Connection-open timeout in seconds. |
| `SQLITE_JOURNAL_MODE` | `WAL` | string | Journal mode applied to every new connection. |
| `SQLITE_BUSY_TIMEOUT_MS` | `5000` | int | `PRAGMA busy_timeout` value. |

### Deprecated Variables

| Variable | Replacement |
|---|---|
| `MONTHLY_USER_BUDGETS` | Use `USER_BUDGETS` with `BUDGET_PERIOD=monthly`. |
| `MONTHLY_GUEST_BUDGET` | Use `GUEST_BUDGET` with `BUDGET_PERIOD=monthly`. |

Both still work (they are read as fallbacks) but log a deprecation warning at
startup.

---

## Telegram UX

### Slash Commands

Registered in [`bot/telegram_bot.py`](bot/telegram_bot.py)
`post_init()`. The exact command list visible in Telegram clients is
controlled by `Bot.set_my_commands()`.

| Command | Scope | Purpose |
|---|---|---|
| `/start`, `/help` | private + groups | Show the help menu. |
| `/reset` | private + groups | Open the session-management menu (preview, switch, delete, change mode, export). |
| `/stats` | private + groups | Token / image / TTS / transcription usage and current session info. |
| `/resend` | private + groups | Resend the last user prompt. |
| `/plugins` | private + groups | Open the paginated plugin menu (commands + button actions). |
| `/image <prompt>` | private + groups | Generate an image (only when `ENABLE_IMAGE_GENERATION=true`). |
| `/tts <text>` | private + groups | Synthesise speech (only when `ENABLE_TTS_GENERATION=true`). |
| `/chat <prompt>` | groups only | Address the bot explicitly in a group chat. |

Plugins can contribute their own slash commands and button handlers via
`Plugin.get_commands()` and `Plugin.get_message_handlers()`. They are
discovered through `PluginManager.build_bot_commands()` and registered
alongside the built-in commands. Commands with `add_to_menu=True` also appear
in the `/plugins` menu.

### Inline Menus

| Menu | Pattern | Purpose |
|---|---|---|
| Session menu | `^session` | Buttons: preview active session, new session, switch, delete, change mode, export, close. |
| Mode picker | `^prompt`, `^promptgroup`, `^promptback` | Two-level picker (group → mode), launched from the session menu. |
| Plugin menu | `^pluginmenu:` | Paginated list of plugins → commands. Each command runs directly or asks for arguments via a force-reply prompt captured by `handle_plugin_menu_args_reply()`. |
| Inline-query response | `^gpt:` | Single button on inline-query results, fetches a reply asynchronously. |

### Media Handlers

`_register_message_tail_handlers()` registers the following filters:

| Filter | Handler | Behaviour |
|---|---|---|
| `filters.PHOTO`, `filters.Document.IMAGE` | `vision()` | Vision over the image, or routed to image-edit when intent matches. |
| `filters.AUDIO`, `filters.VOICE`, `filters.Document.AUDIO`, `filters.VIDEO`, `filters.VIDEO_NOTE`, `filters.Document.VIDEO` | `transcribe()` | Transcribe the file (extracts audio for video). Subject to `IGNORE_GROUP_TRANSCRIPTIONS`. |
| `filters.Document.*` (txt, doc, docx, pdf, rtf, odt, md) | `handle_document()` | Upload to the `text_document_qa` plugin (one workspace per chat). |
| `filters.REPLY & filters.TEXT` | `handle_plugin_menu_args_reply()` | Captures replies to plugin force-reply prompts; otherwise delegates to `prompt()`. Replies to non-image documents are downloaded to a temporary local path for tool-based editing / analysis. |
| `filters.TEXT & ~filters.COMMAND & ~filters.REPLY` | `prompt()` | Catch-all non-reply chat handler. |

### Inline Mode

`@<bot> query` triggers an inline-query response with a single "🤖 Answer with
ChatGPT" button. The reply is generated asynchronously and the inline message
is patched in place once ready.

---

## Sessions And Chat Modes

Conversation state is persisted in SQLite. A user can hold up to `MAX_SESSIONS`
named sessions; each session stores:

- the active system prompt,
- user / assistant messages,
- any Hindsight context that was injected for that session,
- a counter of user-role messages (`message_count`).

Session lifecycle:

1. New sessions are created automatically on first prompt or via the session
   menu.
2. When the limit is reached, `delete_oldest_session()` prunes the oldest one
   before the new one is created.
3. Sessions can be renamed; long-running sessions are auto-named by the
   light model.
4. Deleting a session triggers Hindsight auto-save (when configured) — the
   menu closes immediately and the extraction + retain runs in the background.

**Chat modes** are defined in [`bot/chat_modes.yml`](bot/chat_modes.yml) and
loaded by [`bot/chat_modes_registry.py`](bot/chat_modes_registry.py). Each mode
declares:

- a display name (with emoji),
- a system prompt,
- a per-mode `tools` list (plugin IDs, or `["All"]`),
- a sampling temperature,
- optional grouping for the picker.

`OpenAIHelper` validates the `tools` list against the loaded plugin set during
initialisation; missing references are logged. When a request is prepared, the
active mode constrains `allowed_plugins` (defaulting to `["All"]` when the
mode declares no `tools`). See `bot/openai_helper.py:520` for the wiring.

The current chat-mode catalogue is grouped as follows:

- **Общие**: `assistant`, `text_improver`, `travel_guide`, `content_creator`,
  `technical_writer`, `summary_assistant`, `primitives`.
- **Программирование**: `code_assistant`, `sql_assistant`.
- **Обучение**: `artist`, `english_tutor`, `psychologist`, `movie_expert`.
- **Бизнес**: `code_interpreter`, `startup_idea_generator`, `money_maker`,
  `accountant`, `project_manager`, `meta_writer`, `chief_assistant`.
- **Агенты**: `skills_agent`.

Mode tool names must match plugin **IDs** (filenames without `.py`), not the
human-readable plugin names from `get_source_name()`.

---

## Plugins Catalogue

Plugins live in `bot/plugins/` and subclass
[`bot.plugins.plugin.Plugin`](bot/plugins/plugin.py). Each plugin declares:

- `plugin_id` (stable ID; defaults to the filename without `.py`),
- `function_prefix` (tool namespace; defaults to `plugin_id`),
- `get_source_name()` (display label),
- `get_spec()` (list of OpenAI-compatible function specs),
- async `execute(function_name, helper, **kwargs)`.

`PluginManager.get_functions_specs()` namespaces every spec to
`<function_prefix>.<name>`. Tool calls are validated against their JSON schema
before execution; chat modes can further restrict the allowed tool set.

Enable plugins with the `PLUGINS` allow-list:

```env
PLUGINS=ddg_web_search,ddg_image_search,deepl,gtts_text_to_speech,auto_tts,\
website_content,stable_diffusion,github_analysis,youtube_transcript,\
prompt_perfect,reminders,show_me_diagrams,chief,vkusvill,codeinterpreter,\
mcp_server,text_document_qa,hindsight_memory,skills,terminal,agent_tools,\
web_research
```

### Web And Search

| Plugin | Tools | Notes |
|---|---|---|
| `ddg_web_search` | `web_search` | Gateway-backed search (`llmgateway/web-search`), language hint support. |
| `ddg_image_search` | `search_images` | Gateway-backed image search. |
| `google_web_search` | `web_search` | Google Programmable Search; needs `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`. |
| `jina_web_search` | `web_search` | Jina Search; needs `JINA_API_KEY`. |
| `website_content` | `website_content` | Gateway page reader (`llmgateway/web-read`); also handles YouTube via gateway. |
| `web_research` | `research_articles` | Multi-step research; uses the light model to pick `web-research` vs. `web-deep-research`. |
| `webshot` | `screenshot_website` | thum.io snapshots. |

### Translation

| Plugin | Tools | Notes |
|---|---|---|
| `deepl` | `translate` | DeepL Free/Pro auto-detect. |
| `ddg_translate` | `translate` | DuckDuckGo translation library (gateway-backed compatibility). |

### Images, Video, Speech

| Plugin | Tools | Notes |
|---|---|---|
| `stable_diffusion` | `stable_diffusion`, `edit_image` | Gateway image generation/edit (Klein backends). |
| `haiper_image_to_video` | `generate_video` | vsegpt.ru-backed image-to-video. |
| `gtts_text_to_speech` | `google_translate_text_to_speech` | Gateway Silero TTS. |
| `auto_tts` | `translate_text_to_speech` | Auto-TTS for outputs (uses OpenAI-compatible audio API). |

### Knowledge And Data

| Plugin | Tools | Notes |
|---|---|---|
| `wolfram_alpha` | `answer_with_wolfram_alpha` | Needs `WOLFRAM_APP_ID`. |
| `crypto` | `get_crypto_rate` | CoinCap rates. |
| `iplocation` | `iplocation` | IP geolocation / ASN. |
| `weather` | `get_current_weather`, `get_forecast_weather` | Open Meteo. |
| `worldtimeapi` | `worldtimeapi` | Time zone clock; `WORLDTIME_DEFAULT_TIMEZONE` is required for fallbacks. |
| `whois_` | `get_whois` | python-whois. |
| `text_summarizer` | `summarize_text` | Yandex summarisation; needs `YANDEX_API_TOKEN`. |
| `prompt_perfect` | `optimize_prompt` | Self-rewrites prompts via the helper. |
| `movie_info` | `get_new_movies`, `get_movie_recommendations` | TMDb; needs `TMDB_API_KEY`. |
| `spotify` | `spotify_get_currently_playing_song`, `spotify_get_users_top_artists`, `spotify_get_users_top_tracks` | Needs Spotify OAuth credentials. |

### Documents

| Plugin | Tools | Notes |
|---|---|---|
| `ask_your_pdf` | `analyze_pdf`, `upload_pdf` | Local PDF text extraction. |
| `text_document_qa` | `upload_document`, `ask_question`, `list_documents`, `delete_document` | One AnythingLLM workspace per Telegram chat. |
| `youtube_transcript` | `youtube_video_transcript` | Gateway transcript path. |
| `youtube_audio_extractor` | `extract_youtube_audio` | pytube audio extraction. |

### Agent / Automation / System

| Plugin | Tools | Notes |
|---|---|---|
| `agent_tools` | `manage_plan_tasks`, `ask_telegram_user`, `cancel_pending_question`, `deliver_to_user`, `run_subagents` | See [Agent Runtime](#agent--subagent-runtime). |
| `skills` | `list_skills`, `get_skill`, `get_skill_status`, `list_active_skills`, `activate_skill`, `deactivate_skill`, `update_skill_progress`, `record_skill_reflection`, `run_skill_script` | See [Skills Runtime](#skills-runtime). |
| `mcp_server` | dynamic | See [MCP Integration](#mcp-integration). |
| `hindsight_memory` | `recall`, `list_memories`, `stats` | Manual Hindsight inspection. |
| `terminal` | `terminal` | Direct shell execution; pair with `skills.run_skill_script`. |
| `codeinterpreter` | `deep_analysis` | Sandboxed Python with pandas / numpy / matplotlib / plotly. |
| `github_analysis` | `analyze_github_code` | Reads and summarises GitHub repos with syntax highlighting. |
| `show_me_diagrams` | `create_diagram` | PlantUML diagrams; requires Java + `bot/plugins/plantuml.jar`. |
| `chief` | `get_recipe`, `plan_menu` | Edamam recipes / meal planning. |
| `vkusvill` | `shops`, `products_search`, `recipes`, `basket_create_link` | VkusVill grocery integration. |
| `dice` | `send_dice` | Animated Telegram dice. |
| `reaction` | `react_with_emoji` | Reactions on assistant messages. |
| `reminders` | `set_reminder`, `list_reminders`, `delete_reminder` | Persistent JSON-backed reminders. |
| `task_management` | `create_task`, `list_tasks`, `update_task` | User-visible task list (separate from `agent_tools.manage_plan_tasks`). |
| `language_learning` | `daily_practice`, `track_progress` | Vocabulary / grammar / conversation drills. |
| `conversation_analytics` | `analyze_conversation`, `get_personalized_recommendations` | Rolling conversation analytics. |

> **Note.** Tool function names are namespaced. For example, `deepl.translate`
> and `ddg_translate.translate` co-exist without collision.

---

## Agent / Subagent Runtime

The `agent_tools` plugin layers a subagent loop, plan tracking, idempotent
delivery, and a structured "ask the user" mechanism on top of the regular
function-calling flow.

Key tools:

- **`run_subagents`** — spawns up to 5 bounded subagents in parallel. Each
  subagent runs its own tool-call loop with `max_rounds` clamped to
  `[10, 50]` and uses `bot/prompts/subagent_system.md` as system prompt.
  Subagents inherit the parent's `allowed_plugins` and cannot recursively call
  `run_subagents`, `ask_telegram_user`, `deliver_to_user`, or
  `cancel_pending_question`.
- **`ask_telegram_user`** — sends a question to the user and waits up to
  `AGENT_ASK_USER_TIMEOUT_SECONDS` for a Telegram answer (text or inline
  buttons). Pending questions are persisted under `data/`, so a restart does
  not lose the prompt.
- **`cancel_pending_question`** — closes a pending question programmatically.
- **`deliver_to_user`** — posts an artifact to the user's chat (text, file,
  image, audio, …). The plugin enforces:
  - a 60-second idempotency window keyed by scope;
  - a 49 MiB artifact ceiling;
  - whitelisted artifact kinds and MIME types.
  When `defer=true`, the model loop short-circuits the tool result and the
  artifact is delivered without re-entering the model.
- **`manage_plan_tasks`** — single per-scope plan-tracking tool with
  `action ∈ {add, update, list, clear}` and a list of tasks (each with an `id`
  like `T1`, free-form `content`, and a `status`). Statuses are `pending`,
  `in_progress`, `completed`, `cancelled`. Tasks older than
  `TASKS_TTL_SECONDS` (2 days) are pruned on load.

Subagents are also given an `agent_tools.internal_publish` tool (injected
automatically) that lets them hand findings or local file paths back to the
parent agent's `published` list without delivering anything to the user.

The plan is also rendered live inside the "Готовлю ответ…" busy-status message
that the bot maintains while a request is being processed. See
[`bot/utils.py`](bot/utils.py) — `BusyStatusMessage` accepts a `plan_provider`
callable, which is wired in
[`bot/telegram_bot.py`](bot/telegram_bot.py)
`_build_plan_status_provider()` to pull tasks from the active `agent_tools`
plugin every 5 seconds.

---

## Skills Runtime

The `skills` plugin exposes Codex-style local skills (one folder per skill,
with a `SKILL.md` file describing intent and any optional Python scripts) as
tools the model can call.

Tools:

- `list_skills` — list available skills with their summaries.
- `get_skill` — fetch a single skill's full `SKILL.md` content.
- `find_installable_skills` — search skills.sh via `npx skills find`.
- `install_skill` — install a confirmed package via `npx skills add`, copy it
  into `SKILLS_DIR`, and refresh the local registry.
- `get_skill_status` / `list_active_skills` — inspect which skills are
  currently active for the calling chat scope.
- `activate_skill` / `deactivate_skill` — mark a skill as in-context for the
  current session (used by the `skills_agent` chat mode).
- `run_skill_script` — execute a script under `skills/<skill>/scripts/*.py`
  (gated by `SKILLS_ALLOW_SCRIPTS=true` and the `SKILLS_SCRIPT_ADMIN_USER_IDS`
  allowlist).
- `update_skill_progress` — persist a skill's progress note for later runs.
- `record_skill_reflection` — accumulate repeated improvement proposals after
  skill failures; the fourth identical proposal is appended to that skill's
  `SKILL.md` as a learned clarification.

A watchdog refreshes the typing indicator every 4 seconds while a script is
running, and an interim "still running" notice is sent after
`SKILLS_SCRIPT_INTERIM_AFTER_SECONDS` (default 20 s).

When the `terminal` plugin is also enabled, `terminal.terminal` is the
preferred path for skill scripts that need a real shell workflow;
`codeinterpreter` is reserved for analytical / numeric tasks.

A pre-built `skills_agent` chat mode is defined in `bot/chat_modes.yml` to
drive these tools end-to-end.

---

## MCP Integration

The `mcp_server` plugin acts as an MCP **client**: it connects to one or more
remote MCP servers, fetches their tool catalogues, and re-exposes each remote
tool to the model under a namespaced ID (`mcp_<server>_<tool>` style).

- Auto-register servers via `DEFAULT_MCP_SERVERS=name1:url1,name2:url2`.
- Restrict access via `MCP_SERVERS_ALLOWED_USERS`.
- Per-request timeout: `MCP_REQUEST_TIMEOUT`.

The example MCP servers under [`examples/`](examples/) (`mcp_server_example.py`
and `mcp_stdio_server.py`) are a good starting point. For the full plugin
contract see [`bot/README_MCP.md`](bot/README_MCP.md).

---

## Hindsight Memory

When configured, Hindsight is used twice:

1. **Auto-recall**: on the first user message of a session, the bot fetches
   relevant memories from bank `telegram-<user_id>` and persists them into the
   session context so they survive history compactions.
2. **Auto-save**: when a session is deleted, the bot snapshots the deleted
   session, returns control to Telegram, and runs fact extraction + retain
   in the background (subject to `HINDSIGHT_MAX_AUTO_SAVE_ITEMS`).

The fact extractor is intentionally strict: it stores durable preferences,
ongoing projects, long-term constraints, and explicit "remember this"
statements, but skips one-off image edits, single technical questions,
transient web searches, or secrets.

The `hindsight_memory` plugin (`recall`, `list_memories`, `stats`) is also
available as a regular tool when manual memory inspection is needed.

---

## Storage Layout

```
.
├── bot/
│   ├── chat_modes.yml          # Chat modes catalogue
│   ├── plugins/                # Built-in plugins (~43 modules)
│   ├── prompts/
│   │   └── subagent_system.md  # Subagent system prompt
│   ├── user_data.db            # SQLite (sessions, history, image refs, usage)
│   ├── …
├── data/                       # Default PLUGIN_STORAGE_ROOT
│   ├── skills/                 # Skill definitions (when SKILLS_DIR not overridden)
│   ├── skill_workdir/          # Skill script work area
│   ├── agent_tasks.json        # Plan persistence
│   ├── agent_pending_questions.json
│   ├── conversation_analytics/
│   ├── document_metadata/
│   ├── language_data/
│   ├── pdf_cache/
│   └── …
├── examples/                   # Example MCP server / client
├── tests/                      # Top-level pytest suite
├── requirements.txt
├── README.md                   # English (this file)
└── README.ru.md                # Russian translation
```

Both `bot/user_data.db*` and `data/` are gitignored. Plugins that need
persistent state should always use `self.storage_root` (provided by
`PluginManager`) rather than writing under `bot/plugins/`.

---

## Development

`pytest.ini` enables `asyncio_mode = auto`. Top-level tests live in `tests/`,
MCP-specific tests in `bot/tests/`.

Run everything:

```bash
python -m pytest -q tests/
python -m pytest -q bot/tests/
```

Useful focused suites:

| Area | Tests |
|---|---|
| Plugin registry / specs | `tests/test_plugin_manager.py`, `tests/test_plugin_arg_validation.py`, `tests/test_plugin_commands.py` |
| Plugin direct results | `tests/test_plugin_direct_results.py` |
| Plugin chat-id contract | `tests/test_plugin_chat_id_contract.py` |
| Plugin handler registration | `tests/test_plugin_handlers_registration.py`, `tests/test_plugin_menu_force_reply.py` |
| Tool-call routing | `tests/test_openai_helper_tool_calls.py`, `tests/test_concurrent_tool_state.py` |
| Per-conversation locking | `tests/test_per_conversation_serialization.py` |
| Streaming | `tests/test_telegram_streaming.py` |
| Group session flow | `tests/test_group_session_flow.py` |
| Callback authorization | `tests/test_callback_authorization.py` |
| Telegram builder config | `tests/test_telegram_builder_config.py` |
| Chat-mode registry | `tests/test_chat_modes_registry.py` |
| SQLite sessions / context | `tests/test_database.py` |
| LLMGateway routing | `tests/test_llm_gateway_routing.py`, `tests/test_llm_gateway_client.py` |
| Hindsight | `tests/test_hindsight_memory.py`, `tests/test_hindsight_client.py` |
| Async plugins | `tests/test_async_plugins.py` |
| `RequestContext` contract | `tests/test_request_context.py`, `tests/test_conversation_key.py` |
| Agent tools | `tests/test_agent_tools_plugin.py` |
| Skills | `tests/test_skills_plugin.py` |
| `text_document_qa` (AnythingLLM) | `tests/test_text_document_qa_anythingllm.py` |
| `vkusvill` | `tests/test_vkusvill_plugin.py` |
| `ask_your_pdf` | `tests/test_ask_your_pdf.py` |
| `codeinterpreter` | `tests/test_codeinterpreter_plugin.py` |
| MCP | `bot/tests/test_mcp_server.py` |

For syntax-only checks of the largest modules:

```bash
python -m py_compile bot/openai_helper.py bot/telegram_bot.py
```

Project-wide agent rules and the codebase-mapper graph live in
[`AGENTS.md`](AGENTS.md) (also referenced from `CLAUDE.md`).

---

## Troubleshooting

### Startup exits immediately

Set `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`. Both are required and the
process exits with a clear log line if either is missing.

### Gateway calls fail

- Confirm `OPENAI_BASE_URL` points at your gateway's `/v1` base.
- Confirm `OPENAI_API_KEY` is the gateway key.
- Confirm the chosen model id starts with `llmgateway/` (or whatever prefix
  your gateway requires).

### Bot sends "Готовлю ответ…" but never replies

- Ensure the local Telegram Bot API server is running, or set
  `TELEGRAM_LOCAL_MODE=false` to use Telegram's hosted API.
- Inspect logs for tool-call errors. With `SHOW_PLUGINS_USED=true` the bot
  reports which plugin failed.

### Image reply edits a new image instead of editing

The bot uses the light model to classify intent on replies. Reply directly to
the Telegram image message so the classifier sees `replied_message_kind="image"`
and can decide between `image_edit` and `image_describe`.

### File reply says the generated file was deleted from `/tmp`

Reply directly to the Telegram document message. For non-image document replies,
the bot downloads the replied file through Telegram, injects the temporary
`local_path` into the model request, and removes that source copy after the turn.
The active chat mode still needs file-capable tools enabled, such as `terminal`,
`agent_tools`, `skills`, or a format-specific plugin, to modify the file and
deliver the updated artifact.

### Hindsight shows documents but no memories

Hindsight stores raw documents and runs its own extraction pipeline. Inspect
the bank's `documents`, `memories/list`, `operations`, and `stats` endpoints
directly. The bot only sends extracted facts; it does not control the
document-to-memory transformation.

### Plugin schema collisions

When `PLUGIN_STRICT_VALIDATION=true`, registering two plugins with the same
fully-qualified function name raises `ValueError: Duplicate function name …`
at startup. Either rename the offending function or set the env var to
`false` to fall back to logging-and-skip behaviour.

### Skill scripts won't run

- Confirm `SKILLS_ALLOW_SCRIPTS=true` and that the calling user's id is in
  `SKILLS_SCRIPT_ADMIN_USER_IDS`.
- Check `SKILLS_DIR` actually exists and contains `SKILL.md`.

---

## Credits

This project is a heavily-modified descendant of the original
[`chatgpt-telegram-bot`](https://github.com/n3d1117/chatgpt-telegram-bot)
lineage and uses, among others:

- [python-telegram-bot](https://python-telegram-bot.org)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- LLMGateway-compatible OpenAI-style APIs
- [Hindsight](#hindsight-memory) (long-term memory service)
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) (document Q&A)
- [PlantUML](https://plantuml.com/) (diagrams)
- The [Model Context Protocol](https://modelcontextprotocol.io/) ecosystem
