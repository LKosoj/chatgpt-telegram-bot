from __future__ import annotations

# Models can be found here: https://platform.openai.com/docs/models/overview
GPT_4_VISION_MODELS = ("gpt-4-vision-preview",)
GPT_4O_MODELS = ("openai/gpt-4.1-nano", "openai/gpt-4.1-mini", "openai/gpt-4.1")
GPT_5_MODELS = ("openai/gpt-5-mini", "openai/gpt-5-chat")
O_MODELS = ("openai/o1", "openai/o1-preview", "openai/o1-mini", "openai/o3-mini", "openai/o3-mini-high")
ANTHROPIC = ("anthropic/claude-3-5-haiku", "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4-thinking-high")
GOOGLE = ("google/gemini-flash-1.5-8b", "google/gemini-pro-1.5-online", "google/gemini-2.5-flash-lite",
          "google/gemini-2.5-flash", "google/gemini-2.5-pro")
MISTRALAI = ("mistralai/mistral-medium-3",)
DEEPSEEK = ("deepseek/deepseek-chat-0324-alt-structured", "deepseek/deepseek-r1-alt",)
LLAMA = ("meta-llama/llama-4-maverick", "meta-llama/llama-4-scout")
PERPLEXITY = ("perplexity/sonar-online",)
MOONSHOTAI = ("moonshotai/kimi-k2",)
QWEN = ("qwen/qwen3-235b-a22b-07-25", "qwen/qwen3-next-80b-a3b")

GPT_ALL_MODELS = (
    GPT_4_VISION_MODELS + GPT_4O_MODELS + O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI +
    DEEPSEEK + PERPLEXITY + LLAMA + MOONSHOTAI + QWEN + GPT_5_MODELS
)
