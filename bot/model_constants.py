from __future__ import annotations

LLMGATEWAY_HIGH_MODEL = "llmgateway/high"
LLMGATEWAY_LIGHT_MODEL = "llmgateway/light_model"
LLMGATEWAY_BIG_CONTEXT_MODEL = "llmgateway/big_context"

# Фолбэк-лимит max_tokens для одного запроса генерации, если в конфиге не задан
# ключ ``output_max_tokens``. Применяется как общий клампинг в get_max_tokens и
# как явное значение там, где параметр иначе не задан.
MAX_OUTPUT_TOKENS = 65535

LLMGATEWAY_WEB_SEARCH_MODEL = "llmgateway/web-search"
LLMGATEWAY_WEB_READ_MODEL = "llmgateway/web-read"
LLMGATEWAY_WEB_RESEARCH_MODEL = "llmgateway/web-research"
LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL = "llmgateway/web-deep-research"

# Provider groups are kept for compatibility with older helper checks.
# Runtime model selection is configured through OPENAI_MODEL.
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
