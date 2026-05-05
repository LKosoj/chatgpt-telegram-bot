# ChatGPT Telegram Bot

Telegram bot for an OpenAI-compatible LLMGateway setup. The bot keeps the
Telegram UX from the original project, but the current runtime is centered on
LLMGateway models and gateway-backed tools for chat, web, images, audio, and
long-term memory.

## What It Does

- Chat with session history, named sessions, budgets, usage stats, streaming,
  and per-user access control.
- Use only LLMGateway chat models:
  - `llmgateway/high` for regular assistant work.
  - `llmgateway/light_model` for routing, classification, naming, and small
    background decisions.
  - `llmgateway/big_context` for large-context tasks.
- Generate and edit images through LLMGateway image models.
- Describe images with the configured vision model.
- Transcribe audio through LLMGateway transcription.
- Generate speech through LLMGateway Silero TTS.
- Read web pages, YouTube URLs, search the web, and run web/deep-research
  through LLMGateway.
- Use plugins as model tools, with namespaced function names and per-mode tool
  allow-lists.
- Optionally use Hindsight as per-Telegram-user long-term memory.

## Architecture

Runtime starts in `bot/__main__.py`:

1. Load environment variables.
2. Build `PluginManager`.
3. Build `Database` for SQLite session storage.
4. Build `OpenAIHelper` for LLMGateway/OpenAI-compatible calls.
5. Build and run `ChatGPTTelegramBot`.

Main modules:

| Path | Responsibility |
|---|---|
| `bot/telegram_bot.py` | Telegram handlers, sessions UI, media routing, usage accounting |
| `bot/openai_helper.py` | Chat, vision, image/audio helpers, model selection, Hindsight auto-recall/save |
| `bot/llm_gateway_client.py` | Gateway web, research, image-edit calls |
| `bot/plugin_manager.py` | Plugin discovery, tool specs, validation |
| `bot/openai_tool_handler.py` | Tool-call execution loop |
| `bot/database.py` | SQLite persistence for sessions and image references |
| `bot/chat_modes.yml` | User-facing chat modes and allowed tools |
| `bot/plugins/` | Built-in plugins |

## Requirements

- Python 3.9+.
- Telegram bot token from BotFather.
- LLMGateway-compatible API endpoint.
- API key for the gateway. The bot still reads it from `OPENAI_API_KEY`.
- `ffmpeg` for audio/video processing.

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python -m bot
```

Docker is also available:

```bash
docker compose up
```

## Minimal Configuration

Create `.env` from `.env.example` and set at least:

```env
TELEGRAM_BOT_TOKEN=123456:telegram-token
OPENAI_API_KEY=sk-gateway-token
OPENAI_BASE_URL=http://gateway.example/v1
```

> [!NOTE]
> `OPENAI_API_KEY` should contain the LLMGateway key in the current setup. The
> project still uses OpenAI-compatible clients, so the legacy environment
> variable name remains.

Telegram Bot API access defaults to the local endpoint for backward
compatibility:

```env
TELEGRAM_LOCAL_MODE=true
TELEGRAM_BASE_URL=http://localhost:8081/bot
```

Set `TELEGRAM_LOCAL_MODE=false` to use Telegram's hosted API. In that mode,
`TELEGRAM_BASE_URL` is not passed to the Telegram application builder.

Recommended gateway model settings:

```env
OPENAI_MODEL=llmgateway/high
LIGHT_MODEL=llmgateway/light_model
BIG_MODEL_TO_USE=llmgateway/big_context
IMAGE_MODEL=llmgateway/ai-klein-generation
TTS_MODEL=llmgateway/silero-tts
TTS_VOICE=kseniya
TTS_RESPONSE_FORMAT=wav
TRANSCRIPTION_MODEL=llmgateway/whisper-large-v3
```

Useful feature toggles:

```env
ENABLE_FUNCTIONS=true
ENABLE_IMAGE_GENERATION=true
ENABLE_VISION=true
ENABLE_TRANSCRIPTION=true
ENABLE_TTS_GENERATION=true
MAX_SESSIONS=5
BOT_LANGUAGE=ru
```

## LLMGateway Features

### Chat Models

Only these chat models are expected in the bot:

| Use | Model |
|---|---|
| Main assistant | `llmgateway/high` |
| Fast classification/routing | `llmgateway/light_model` |
| Large context | `llmgateway/big_context` |

Model switching UI from the legacy multi-provider bot is not part of the
current intended workflow.

### Web And Research

Web operations are gateway-backed:

- Web search: `llmgateway/web-search`.
- Web read: `llmgateway/web-read`.
- Web research: `llmgateway/web-research`.
- Deep research: `llmgateway/web-deep-research`.

The web research plugin first uses the light model to decide whether a request
needs regular research or deep research. Deep research can take significantly
longer.

YouTube URLs are handled through gateway web read; the gateway detects YouTube
and returns transcript-like content when available.

### Images

Image generation uses `llmgateway/ai-klein-generation`.

Image editing uses `llmgateway/ai-klein-edit`. The bot can route edit requests
from:

- an image uploaded with a caption;
- a reply to a Telegram image;
- the last generated/sent image when the text clearly refers to it.

Examples:

- "Нарисуй кота" creates a new image.
- Replying to that image with "нарисуй коту шапочку" edits the referenced image.
- Replying to an image with "что на этой картинке?" routes to vision instead of
  image edit.

### Audio

- Text-to-speech uses LLMGateway Silero TTS.
- Default voice is `kseniya`.
- Default response format is `wav`.
- Voice/audio transcription uses the gateway transcription model.

## Hindsight Long-Term Memory

Hindsight is optional and enabled automatically when both values are configured:

```env
HINDSIGHT_BASE_URL=http://hindsight.example/hindsight
HINDSIGHT_API_TOKEN=secret
```

Defaults:

```env
HINDSIGHT_NAMESPACE=default
HINDSIGHT_BANK_PREFIX=telegram-
HINDSIGHT_AUTO_RECALL=true
HINDSIGHT_AUTO_SAVE=true
HINDSIGHT_RECALL_BUDGET=mid
HINDSIGHT_RECALL_MAX_TOKENS=4096
HINDSIGHT_MEMORY_TYPES=world,experience
HINDSIGHT_ASYNC_STORE=true
HINDSIGHT_MAX_AUTO_SAVE_ITEMS=5
```

Memory bank per Telegram user:

```text
telegram-<user_id>
```

Auto-recall is done once for the first user request in a session and persisted
into SQLite conversation history. Auto-save runs when a session is deleted:

1. The bot snapshots the deleted session messages.
2. The Telegram UI closes/updates immediately.
3. Fact extraction and Hindsight retain run in a background task.

The extractor is intentionally strict. It should save only durable facts such
as explicit preferences, ongoing projects, long-term constraints, and explicit
"remember this" statements. It should not save one-off image edits, single
technical questions, transient web searches, uploaded-image descriptions, logs,
or secrets.

## Plugins

Plugins live in `bot/plugins/` and subclass `Plugin`.

Key behavior:

- Tool names are namespaced as `<plugin_id>.<function>`.
- `PLUGINS` is an allow-list. Empty means no explicit allow-list is applied by
  startup config.
- Chat modes can further restrict allowed tools through `bot/chat_modes.yml`.
- Tool arguments are validated against plugin JSON schemas before execution.

Common current plugins include:

| Plugin | Purpose |
|---|---|
| `web_research` | Gateway-backed web/deep research |
| `website_content` | Gateway-backed page/URL reading |
| `ddg_image_search` | Gateway-backed image search compatibility plugin |
| `stable_diffusion` | Gateway-backed image generation compatibility plugin |
| `gtts_text_to_speech` | Gateway-backed TTS compatibility plugin |
| `auto_tts` | Auto TTS responses |
| `youtube_transcript` | YouTube transcript compatibility path |
| `hindsight_memory` | Manual Hindsight recall/list/stats tools |
| `codeinterpreter` | Code execution/analysis helper |
| `text_document_qa` | AnythingLLM-backed document Q&A with one workspace per Telegram chat |
| `reminders` | Reminder management |

Enable selected plugins:

```env
PLUGINS=web_research,website_content,stable_diffusion,gtts_text_to_speech,hindsight_memory
```

For `text_document_qa`, configure AnythingLLM API access:

```env
ANYTHINGLLM_BASE_URL=http://anythingllm.example/api
ANYTHINGLLM_API_KEY=your_anythingllm_api_key
ANYTHINGLLM_WORKSPACE_PREFIX=telegram-chat
```

## Sessions And Chat Modes

The bot stores real conversation history in SQLite:

- system prompt;
- user messages;
- assistant responses;
- Hindsight recall context when it was injected for that session.

Session controls are exposed through Telegram inline menus. Deleting a session
is the point where Hindsight auto-save is scheduled.

Chat modes are defined in `bot/chat_modes.yml`. Keep mode tool names aligned
with plugin ids, not display names.

## Development

Run targeted tests:

```bash
python -m pytest -q tests/test_llm_gateway_routing.py
python -m pytest -q tests/test_hindsight_memory.py
python -m pytest -q tests/test_database.py
```

For syntax-only checks:

```bash
python -m py_compile bot/openai_helper.py bot/telegram_bot.py
```

Useful focused test areas:

| Area | Tests |
|---|---|
| Plugin registry/specs | `tests/test_plugin_manager.py` |
| Plugin arguments | `tests/test_plugin_arg_validation.py` |
| Tool-call routing | `tests/test_openai_helper_tool_calls.py` |
| Chat modes | `tests/test_chat_modes_registry.py` |
| SQLite sessions | `tests/test_database.py` |
| LLMGateway routing | `tests/test_llm_gateway_routing.py` |
| Hindsight memory | `tests/test_hindsight_memory.py` |

## Troubleshooting

### Startup Exits Immediately

Check required environment variables:

```env
TELEGRAM_BOT_TOKEN=...
OPENAI_API_KEY=...
```

### Gateway Calls Fail

Check:

- `OPENAI_BASE_URL` points to the gateway `/v1` base.
- `OPENAI_API_KEY` is the gateway key.
- The selected model starts with `llmgateway/`.

### Hindsight Shows Documents But No Memories

Hindsight can store source documents while still producing zero memory units.
Check the bank's `documents`, `memories/list`, `operations`, and `stats`
endpoints. In this bot, auto-save sends extracted durable facts to Hindsight;
Hindsight then owns the document-to-memory extraction pipeline.

### Image Reply Editing Generates A New Image

Use a reply to the Telegram image message or refer clearly to the last generated
image. The bot classifies reply intent with the light model and has a legacy
fallback for obvious image-edit wording.

## Credits

This project is based on the original `chatgpt-telegram-bot` lineage and uses:

- [python-telegram-bot](https://python-telegram-bot.org)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- LLMGateway-compatible OpenAI API routing
