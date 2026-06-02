from __future__ import annotations

LLMGATEWAY_HIGH_MODEL = "llmgateway/high"
LLMGATEWAY_LIGHT_MODEL = "llmgateway/light_model"
LLMGATEWAY_BIG_CONTEXT_MODEL = "llmgateway/big_context"

LLMGATEWAY_CHAT_MODELS = (
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
    LLMGATEWAY_BIG_CONTEXT_MODEL,
)

# Фолбэк-лимит max_tokens для одного запроса генерации, если в конфиге не задан
# ключ ``output_max_tokens``. Применяется как общий клампинг в get_max_tokens и
# как явное значение там, где параметр иначе не задан.
MAX_OUTPUT_TOKENS = 65535

LLMGATEWAY_IMAGE_GENERATION_MODEL = "llmgateway/ai-klein-generation"
LLMGATEWAY_WEB_SEARCH_MODEL = "llmgateway/web-search"
LLMGATEWAY_WEB_READ_MODEL = "llmgateway/web-read"
LLMGATEWAY_WEB_RESEARCH_MODEL = "llmgateway/web-research"
LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL = "llmgateway/web-deep-research"
LLMGATEWAY_TTS_MODEL = "llmgateway/silero-tts"
LLMGATEWAY_TRANSCRIPTION_MODEL = "llmgateway/whisper-large-v3"

# Provider groups are kept for compatibility with older helper/Telegram checks.
# Model switching now exposes only the llmgateway chat models above.
GPT_4_VISION_MODELS = ()
GPT_4O_MODELS = ()
GPT_5_MODELS = ()
O_MODELS = ()
ANTHROPIC = ()
GOOGLE = ()
MISTRALAI = ()
DEEPSEEK = ()
LLAMA = ()
PERPLEXITY = ()
MOONSHOTAI = ()
QWEN = ()

GPT_ALL_MODELS = LLMGATEWAY_CHAT_MODELS
