# T10. Единый интерфейс провайдера — implementation plan

Owner: developer implementing T10, wave 5. File ownership per master plan
(`docs/improvement_2026-09-25/00-master-plan.md:340-367`): `bot/ai_provider.py`,
`bot/ai_providers/**`, `bot/openai_helper.py`, `bot/openai_tool_handler.py` (error
handling only), `bot/plugins/stable_diffusion.py`, `bot/plugin_tool_adapter.py`
(delete), `tests/test_plugin_tool_adapter.py` (delete), provider tests, new AST-guard
test. **Never touch `get_spec()`, tool names/params, or `tools:` lists.**

User decision (binding): retries live only in the SDK (`max_retries=3`, already set at
`bot/openai_helper.py:258` inside `client_kwargs`). Delete the manual retry loop
`_create_chat_completion_with_rate_limit_retry` and the `LLM_RATE_LIMIT_RETRY_*`
constants/env. After the SDK exhausts retries, show a clear user message (today the
rate-limit branch at `bot/openai_helper.py:1492` does a bare `raise e` with **no**
user-facing wrapping — this is a real gap the plan below closes).

Environment verified: `openai==3.8.0` installed in `~/.venvs/ctb`.
`openai.RateLimitError`/`openai.BadRequestError` extend `openai.APIStatusError` →
`openai.APIError` → `openai.OpenAIError` → `Exception`. `AsyncOpenAI.__init__` default
`max_retries=2`; the codebase already overrides it to `3`.

---

## 0. Full inventory of raw `openai`/`self.client.` access in `bot/`

Found by scanning every `.py` under `bot/` for `import openai`, `(?<!\.)\bopenai\.` (bare
module reference, excludes `self.openai.*` and comment URLs like
`platform.openai.com`), and `self\.client\.`:

| file:line | code | classification |
|---|---|---|
| `bot/openai_helper.py:16` | `import openai` | import |
| `bot/openai_helper.py:262` | `self.client = openai.AsyncOpenAI(**client_kwargs)` | client construction |
| `bot/openai_helper.py:469` (`_create_chat_completion_with_rate_limit_retry`, def at :466) | `return await self.client.chat.completions.create(**kwargs)` | chat, inside the loop being deleted |
| `bot/openai_helper.py:470` | `except openai.RateLimitError as exc:` | error handling, inside the loop being deleted |
| `bot/openai_helper.py:1492` (`_common_get_chat_response`, def at :1280) | `except openai.RateLimitError as e:` | error handling — **bare `raise e`, no user message** |
| `bot/openai_helper.py:1496` | `except openai.BadRequestError as e:` | error handling — already wraps with a localized message |
| `bot/openai_helper.py:2021` (`generate_image`, def at :2002) | `await self.client.images.generate(**image_args)` | images generate |
| `bot/openai_helper.py:2114` (`get_available_tts_models`, def at :2110) | `await self.client.models.list()` | models list |
| `bot/openai_helper.py:2172` (`generate_speech`, def at :2164) | `await self.client.audio.speech.create(...)` | audio speech |
| `bot/openai_helper.py:2199` (`transcribe`, def at :2192) | `await self.client.audio.transcriptions.create(...)` | audio transcriptions |
| `bot/openai_helper.py:2315` (`__common_get_chat_response_vision`, def at :2212) | `except openai.RateLimitError as e:` | error handling — bare `raise e` |
| `bot/openai_helper.py:2318` | `except openai.BadRequestError as e:` | error handling — already wraps |
| `bot/openai_helper.py:4176` (`close`, def at :4149) | `await self.client.close()` | cleanup, not a request |
| `bot/openai_tool_handler.py:11` | `import openai` | import |
| `bot/openai_tool_handler.py:1247` (nested closure `retry_stream_plain_text_tool_intent_or_replay`, inside `handle_function_call`, def at :1136) | `except openai.APIError as e:` | error handling, mid-stream buffering |
| `bot/openai_tool_handler.py:1307` (inside `handle_function_call`'s `if stream:` block) | `except openai.APIError as e:` | error handling, mid-stream iteration |
| `bot/plugins/stable_diffusion.py:52` (`_generate_image`, def at :51) | `await helper.client.images.generate(...)` | images generate, **bypasses the helper entirely** |

Two lines matched the raw grep but are **false positives**, verified by reading context:
- `bot/plugins/haiper_image_to_video.py:1264` — `self.bot = openai.bot`: `openai` here is
  a local parameter name of `handle_prompt_constructor(self, function_name, openai,
  update=None, **kwargs)` (`bot/plugins/haiper_image_to_video.py:1243`), i.e. it *is* the
  `OpenAIHelper` instance passed in under that name, not the SDK module. No change.
- `bot/plugins/hindsight_memory.py` (`self.client.*` at :640, :2334, :2554, :3165, :3181,
  :3226, :3319, :3323, :3353, :3361) — `self.client` there is a `HindsightClient`
  (`bot/plugins/hindsight_memory.py:594`, unrelated HTTP client for the memory service).
  Not OpenAI SDK. No change.

`bot/plugins/*` other `gateway_client.*` calls (`ddg_image_search.py:45`,
`ddg_web_search.py:70`, `web_research.py:104/110`, `website_content.py:38`,
`youtube_transcript.py:44`, `stable_diffusion.py:62`) call
`helper.gateway_client.web_*`/`image_edit` directly. `gateway_client` is a distinct
attribute name (`LLMGatewayClient`, `bot/llm_gateway_client.py`, plain `httpx`, raises
`LLMGatewayError`, not `openai.*`) — it does **not** match `.client.` textually or
semantically, so it is correctly outside this task's guard and outside the provider.
This matches master-plan's explicit "gateway_client web_* stays outside the provider".

`LLM_RATE_LIMIT_RETRY_ATTEMPTS`/`LLM_RATE_LIMIT_RETRY_WAIT_SECONDS` (defined
`bot/openai_helper.py:74-75`) appear in exactly two places outside their own
definition/usage: `tests/test_openai_helper_tool_calls.py` and this master plan doc.
**`.env.example`, `README.md`, `README.ru.md`, `bot/__main__.py` have zero references** —
confirmed by direct grep, nothing to update there.

`bot/plugin_tool_adapter.py` (`PluginToolAdapter`) has exactly one production-code
mention, a docstring comment in `bot/ai_events.py:29` ("Provider adapters use the
model-visible tool name. PluginToolAdapter canonicalizes it..."). No code imports or
instantiates it anywhere except its own test file `tests/test_plugin_tool_adapter.py`.
Confirmed dead — safe to delete per master-plan step 6. The `bot/ai_events.py:29`
docstring comment is **not owned by T10** (file not in ownership list) — leave it; it
still describes the design intent, not a specific dead class.

---

## 1. Current call chain (why this is two nested "providers", not one)

```
OpenAIHelper.chat_completion / _common_get_chat_response / vision path
  -> _create_chat_response_completion(kind=..., **kwargs)     [:667]
     -> _timed_create_via_ai_provider(kind=..., **kwargs)     [:594]
        # builds AIProviderRequest from kwargs (:562 _ai_provider_request_from_kwargs)
        # constructs a NEW OpenAICompatibleProvider EVERY CALL:
        provider = OpenAICompatibleProvider(create_chat_completion=<closure calling self._timed_create>, ...)
        -> provider.create_response(request) / provider.stream_response(request)
           -> self._timed_create(kind=..., **kwargs)          [:483]
              # timing + session_logger 'llm_request'/'llm_call'/'llm_error' events
              -> self._create_chat_completion_with_rate_limit_retry(kind=..., **kwargs)  [:466, TO DELETE]
                 # manual retry loop, catches openai.RateLimitError, asyncio.sleep(20)
                 -> self.client.chat.completions.create(**kwargs)   # <-- the ONE real SDK call
```

Two independent logging layers exist on purpose and must both survive: `_timed_create`
records low-level `llm_request`/`llm_call`/`llm_error` (kind, duration_ms, raw
usage/tool_calls) and is **directly unit-tested** (`tests/test_session_logging_integration.py`,
`tests/test_openai_helper_tool_calls.py:522,567`) — it must keep its name, signature, and
externally observable event shape. `_timed_create_via_ai_provider` records the higher-level
`ai_provider_response`/`AIProviderError` events from the `AIProviderResponse` abstraction and
is also directly monkeypatched by tests (`tests/test_openai_helper_tool_calls.py:542`:
`monkeypatch.setattr(helper, "_timed_create_via_ai_provider", fake_provider)`) — it must also
keep its name.

**Critical constraint found while reading tests**: ~90 call sites across
`tests/test_openai_helper_tool_calls.py`, `tests/test_skills_agent_gate.py`,
`tests/test_reflection_on_tool_error.py`, `tests/test_session_logging_integration.py`,
`tests/test_openai_helper_db_offload.py`, `tests/test_hindsight_memory.py` do
`_make_helper(...)` (which runs the **real** `OpenAIHelper.__init__`) and **then**
overwrite `helper.client = SomeFakeClient()` **after construction**
(`tests/test_openai_helper_tool_calls.py:470` inside `_make_helper` itself, plus ~85 more
direct overwrites). Master-plan step 2 ("провайдер создаётся один раз в `__init__`") must
not snapshot `self.client` into the provider at construction time, or all of these tests
break. The design below resolves this by giving the provider a `get_client` **accessor**
(a closure reading `self.client` fresh on every call), not a captured client reference —
this exactly matches `self.client.chat.completions.create(...)`'s current dynamic
attribute lookup, so no test that reassigns `helper.client` needs to change.

---

## 2. Target design

### 2.1 `bot/ai_provider.py` — new error classes

```python
class ProviderError(Exception):
    """Base class for all AI-provider-level errors. Call sites outside
    bot/ai_providers/ catch these, never openai.* directly."""


class ProviderRateLimitError(ProviderError):
    """Rate limit exhausted. The SDK already retried internally
    (max_retries=3); this means retries are exhausted, not that a retry
    should be attempted by caller code."""


class ProviderBadRequestError(ProviderError):
    """Request rejected by the backend (4xx, not a rate limit)."""


class ProviderStreamError(ProviderError):
    """Raised mid-iteration of a streaming response."""
```

Add near the existing dataclasses, before `class AIProvider(Protocol):`. Extend the
Protocol with the five new methods (documentation/typing only — `Protocol` is
structural, this does not force anything at runtime):

```python
class AIProvider(Protocol):
    def stream_response(self, request: AIProviderRequest) -> AsyncIterator[AIEvent]: ...
    async def generate_image(self, **kwargs: Any) -> Any: ...
    async def edit_image(self, **kwargs: Any) -> Any: ...
    async def speech(self, **kwargs: Any) -> Any: ...
    async def transcribe(self, **kwargs: Any) -> Any: ...
    async def list_models(self) -> Any: ...
    async def list_voices(self, **kwargs: Any) -> Any: ...
```

### 2.2 `bot/ai_providers/openai_compatible.py` — where `import openai` moves to

Add, without touching any existing function/method (`OpenAICompatibleProvider.create_response`/
`stream_response`, `_response_events`, `_streaming_events`, `OpenAIStreamEventRecorder`,
`_usage`, etc. stay byte-for-byte identical — they never touched `openai.*` and don't need
to):

```python
import openai

from bot.ai_provider import ProviderBadRequestError, ProviderError, ProviderRateLimitError, ProviderStreamError


def build_openai_client(config: dict, http_client) -> "openai.AsyncOpenAI":
    """Moved from OpenAIHelper.__init__ (bot/openai_helper.py:254-262) verbatim."""
    client_kwargs = {
        "api_key": config["api_key"],
        "http_client": http_client,
        "timeout": 300.0,
        "max_retries": 3,
    }
    if config["openai_base"]:
        client_kwargs["base_url"] = config["openai_base"]
    return openai.AsyncOpenAI(**client_kwargs)


def _translate(exc: Exception) -> Exception:
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, openai.BadRequestError):
        return ProviderBadRequestError(str(exc))
    if isinstance(exc, openai.APIError):
        return ProviderError(str(exc))
    return exc


async def _translate_stream_errors(raw_stream):
    """Wrap a raw SDK stream so mid-iteration openai.* errors surface as
    Provider* to every consumer (_AIProviderStreamProxy, openai_tool_handler)."""
    try:
        async for chunk in raw_stream:
            yield chunk
    except openai.APIError as exc:
        raise ProviderStreamError(str(exc)) from exc


def raw_chat_completion(get_client):
    """Build the CreateChatCompletion callable used by the production
    OpenAICompatibleProvider. `get_client` is called fresh on every request
    (not captured once) so tests that reassign `helper.client` after
    construction keep working unchanged. No retry here — SDK max_retries=3
    (build_openai_client) is the only retry layer."""
    async def _create(**kwargs):
        client = get_client()
        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc
        if kwargs.get("stream"):
            return _translate_stream_errors(response)
        return response
    return _create
```

Extend `OpenAICompatibleProvider` with the five backend methods. Constructor keeps its
existing positional `create_chat_completion` parameter untouched (so
`tests/test_ai_provider.py:124`, `tests/test_openai_compatible_provider.py:81` etc. keep
constructing it exactly as today); add two **optional** accessor kwargs:

```python
class OpenAICompatibleProvider:
    def __init__(
        self,
        create_chat_completion: CreateChatCompletion,
        *,
        provider_name: str = "openai-compatible",
        get_client=None,          # () -> openai.AsyncOpenAI, needed for image/audio/models
        get_gateway_client=None,  # () -> LLMGatewayClient, needed for edit_image/list_voices
    ):
        self._create_chat_completion = create_chat_completion
        self.provider_name = provider_name
        self._get_client = get_client
        self._get_gateway_client = get_gateway_client

    # ... existing stream_response/create_response unchanged ...

    async def generate_image(self, **kwargs):
        try:
            return await self._get_client().images.generate(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def list_models(self):
        try:
            return await self._get_client().models.list()
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def speech(self, **kwargs):
        try:
            return await self._get_client().audio.speech.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def transcribe(self, **kwargs):
        try:
            return await self._get_client().audio.transcriptions.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def edit_image(self, **kwargs):
        return await self._get_gateway_client().image_edit_file(**kwargs)

    async def list_voices(self, **kwargs):
        return await self._get_gateway_client().audio_voices(**kwargs)
```

`edit_image`/`list_voices` are gateway-backed (`LLMGatewayClient`, raises `LLMGatewayError`,
not `openai.*`) — no translation needed for the literal "провайдер переводит openai.* в
них" requirement. **Open design point** (see §6): moving them onto the provider is
requested by master-plan step 3's method list for interface uniformity, but is **not**
required by the AST guard itself (`gateway_client` doesn't match `.client.`). Recommend
doing it anyway (near-zero risk, one-line call-site swap, keeps `OpenAIHelper` talking to
exactly one object for all six backend operations) but flagging it as separable if time is
short — everything else in this plan works without it.

### 2.3 `bot/ai_providers/fake.py` — parity methods

Add queue-based fakes mirroring the existing `queue_text`/`_event_batches` pattern
(43 lines today, stays small):

```python
def __init__(self, events=()):
    ...  # existing
    self.image_calls: list[dict] = []
    self._image_results: deque = deque()
    self.speech_calls: list[dict] = []
    self._speech_results: deque = deque()
    self.transcribe_calls: list[dict] = []
    self._transcribe_results: deque = deque()
    self.models_results: deque = deque()
    self.voices_calls: list[dict] = []
    self._voices_results: deque = deque()

def queue_image(self, value) -> None:
    self._image_results.append(value)

async def generate_image(self, **kwargs):
    self.image_calls.append(kwargs)
    return self._image_results.popleft()

async def edit_image(self, **kwargs):
    self.image_calls.append(kwargs)
    return self._image_results.popleft()

# speech/transcribe/list_models/list_voices follow the same shape
```

(Exact field names are a judgment call for whoever implements this — keep it symmetric
with `queue_text`/`assert_no_pending_events` so it reads consistently; not prescribing
byte-exact code since no existing test currently exercises these fake methods.)

### 2.4 `bot/openai_helper.py` — wiring

**Imports** (`:16`): delete `import openai`. Add
`from .ai_provider import ProviderBadRequestError, ProviderError, ProviderRateLimitError` and
`from .ai_providers.openai_compatible import OpenAICompatibleProvider, build_openai_client,
raw_chat_completion` (the second import already exists partially at `:59` for
`stream_chunk_has_choice, stream_chunk_text_delta` — extend that import line, don't add a
duplicate `from .ai_providers.openai_compatible import ...`).

**Constants** (`:74-75`): delete `LLM_RATE_LIMIT_RETRY_ATTEMPTS` and
`LLM_RATE_LIMIT_RETRY_WAIT_SECONDS`.

**`__init__` client construction** (`:254-263`), before:
```python
client_kwargs = {
    "api_key": config["api_key"],
    "http_client": self._http_client,
    "timeout": 300.0,
    "max_retries": 3,
}
if config["openai_base"]:
    client_kwargs["base_url"] = config["openai_base"]
self.client = openai.AsyncOpenAI(**client_kwargs)
self.gateway_client = LLMGatewayClient(config.get("openai_base", ""), config["api_key"])
```
after:
```python
self.client = build_openai_client(config, self._http_client)
self.gateway_client = LLMGatewayClient(config.get("openai_base", ""), config["api_key"])
self._provider = OpenAICompatibleProvider(
    raw_chat_completion(lambda: self.client),
    provider_name="chat-run-openai-compatible",
    get_client=lambda: self.client,
    get_gateway_client=lambda: self.gateway_client,
)
```
`self.client`/`self.gateway_client` stay exactly as today (same names, same place, same
mutability) — only the RHS construction and one new `self._provider` line change. This is
the "provider constructed once in `__init__`" master-plan requirement, satisfied for the
raw-SDK-touching provider. Note precisely which object this satisfies: `self._provider`
(the singleton wrapping the raw client) — **not** the separate per-call
`OpenAICompatibleProvider(self._timed_create, ...)` still built inside
`_timed_create_via_ai_provider` (see §6, left alone deliberately: it never imports
`openai` or touches `.client.`, so it doesn't trip the guard, and collapsing it into a
single object is a bigger, riskier rewrite not required by any of T10's stated goals).

**`_create_chat_completion_with_rate_limit_retry`** (`:466-481`): delete entirely.

**`_timed_create`** (`:483` onward): the only change is the one line that called the
deleted method:
```python
# before
response = await self._create_chat_completion_with_rate_limit_retry(kind=kind, **kwargs)
# after
response = await self._provider.create_response(self._ai_provider_request_from_kwargs(kwargs))
```
Everything else in `_timed_create` (timing, `llm_request`/`llm_call`/`llm_error` session
events, stats accumulation, the existing `except Exception as exc:` logging+re-raise)
stays untouched — it already re-raises whatever it catches unchanged, so
`ProviderRateLimitError`/`ProviderBadRequestError`/`ProviderError` propagate through it
exactly like `openai.RateLimitError` did before.

**`generate_image`** (`:2002-2035`): `:2021` `await self.client.images.generate(**image_args)`
→ `await self._provider.generate_image(**image_args)`.

**`edit_telegram_image`** (`:2040-2053`, `:2047`): `await self.gateway_client.image_edit_file(...)`
→ `await self._provider.edit_image(...)` — see §2.2's open point; skip if not doing the
gateway-side uniformity pass.

**`get_available_tts_models`** (`:2110-2124`, `:2114`): `await self.client.models.list()`
→ `await self._provider.list_models()`.

**`get_available_tts_voices`** (`:2126-2143`, `:2132`): `await self.gateway_client.audio_voices(model_to_use)`
→ `await self._provider.list_voices(model=model_to_use)` — same open point as edit_image.

**`generate_speech`** (`:2164-2189`, `:2172`): `await self.client.audio.speech.create(...)`
→ `await self._provider.speech(...)`.

**`transcribe`** (`:2192-2211`, `:2199`): `await self.client.audio.transcriptions.create(...)`
→ `await self._provider.transcribe(...)`.

**New public passthrough for `stable_diffusion.py`** (see §2.6 for why): add next to
`generate_image`:
```python
async def raw_generate_image(self, **kwargs):
    """Low-level image-generate call returning the raw SDK response object
    (not the (value, size) tuple generate_image() returns). Exists for
    plugins that need extract_image_result()'s raw url/b64_json/path shape
    directly, e.g. bot/plugins/stable_diffusion.py."""
    return await self._provider.generate_image(**kwargs)
```

**`_common_get_chat_response`** (`:1280` onward, tail at `:1490-1497`), before:
```python
        except openai.RateLimitError as e:
            logger.warning("Rate limit error error=%s", log_exception_shape(e))
            raise e

        except openai.BadRequestError as e:
            logger.error("Bad request error error=%s", log_exception_shape(e))
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{error_message}") from e
```
after (adds the missing user-facing message on rate-limit exhaustion — reuses the
existing generic `'error'` i18n key already used elsewhere in this same file, no new i18n
key/file needed):
```python
        except ProviderRateLimitError as e:
            logger.warning("Rate limit error error=%s", log_exception_shape(e))
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{error_message}") from e

        except ProviderBadRequestError as e:
            logger.error("Bad request error error=%s", log_exception_shape(e))
            error_message = escape_markdown(str(e))
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{error_message}") from e
```

**`__common_get_chat_response_vision`** (`:2212` onward, tail at `:2315-2322`): same
transformation — `except openai.RateLimitError as e: raise e` gets the same wrap, and
`except openai.BadRequestError` becomes `except ProviderBadRequestError`.

**`close`** (`:4149-4183`, the `self.client.close()`/`self.gateway_client.close()` block at
`:4172-4179`): **leave as-is**, put in the AST guard's allow-list (see §3) — pure resource
cleanup on shutdown, no request semantics, no error translation need, not worth adding an
indirection for.

### 2.5 `bot/openai_tool_handler.py`

**Import** (`:11`): delete `import openai`, add `from .ai_provider import ProviderStreamError`.

**`handle_function_call`** (def `:1136`), both sites:
- `:1247` (inside nested `retry_stream_plain_text_tool_intent_or_replay`):
  `except openai.APIError as e:` → `except ProviderStreamError as e:`
- `:1307` (inside the `if stream:` block): same change.

No other lines in this function change — the log messages, the `_replay_stream_items`
fallback, and the `return response, tools_used` behavior are untouched.

### 2.6 `bot/plugins/stable_diffusion.py`

`_generate_image` (`:51-58`), before:
```python
    async def _generate_image(self, helper, prompt: str) -> tuple[str, str]:
        response = await helper.client.images.generate(
            prompt=prompt,
            n=1,
            model=helper.config.get("image_model"),
            size=helper.config.get("image_size", "1024x1024"),
            extra_headers={ "X-Title": "tgBot" },
        )
        return extract_image_result(response)
```
after (only the call target changes):
```python
    async def _generate_image(self, helper, prompt: str) -> tuple[str, str]:
        response = await helper.raw_generate_image(
            prompt=prompt,
            n=1,
            model=helper.config.get("image_model"),
            size=helper.config.get("image_size", "1024x1024"),
            extra_headers={ "X-Title": "tgBot" },
        )
        return extract_image_result(response)
```
Why not call `helper.generate_image(prompt)` (the existing public method)? Checked its
contract: it returns `(image_value, self.config['image_size'])` — the second element is
the configured size string, **not** the "path"/"url" format tag that
`extract_image_result()` returns and that `execute()`
(`bot/plugins/stable_diffusion.py:80` `image_value, image_format = await
self._generate_image(...)`) puts into `direct_result["format"]`. Reusing
`generate_image()` would silently put `"1024x1024"` into the format field and break image
delivery. Hence the new `raw_generate_image()` passthrough in §2.4 instead of reusing the
existing method or reaching into `helper._provider` directly (the latter would also fail
`tests/test_no_private_helper_access.py`'s `test_plugins_do_not_touch_helper_privates`,
which is a generic `name.startswith("_")` AST check over `helper.<attr>` in every
`bot/plugins/*.py` file — confirmed by reading `tests/test_no_private_helper_access.py:52`).

`_edit_image` (`:61-67`, uses `helper.gateway_client.image_edit`) is **not** a guard
violation (`gateway_client` ≠ `.client.` textually) and is unaffected by this plan.

### 2.7 Delete `bot/plugin_tool_adapter.py` and `tests/test_plugin_tool_adapter.py`

Confirmed dead (§0). Plain `rm` of both files, no other edits needed.

---

## 3. AST guard test (new file, e.g. `tests/test_ast_no_raw_openai_access.py`)

Scan every `.py` under `bot/` (matches the guard's stated scope — `stable_diffusion.py`
is a real violation today and must be caught). For each file outside
`bot/ai_providers/`, flag:
1. `ast.Import`/`ast.ImportFrom` where the module is `openai` (or starts with `openai.`).
2. `ast.Attribute` nodes shaped `X.client.Y` — i.e. an `Attribute` whose `.value` is
   itself an `Attribute` with `.attr == "client"`. This matches `self.client.images.generate`,
   `helper.client.images.generate`, etc. but **not** `self.client = ...` (assignment
   target, no further chained attribute) and **not** passing `self.client` as a bare
   argument (also no chained attribute) — both of which the design in §2.4 relies on.

```python
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = REPO_ROOT / "bot"
ALLOWED_DIR = BOT_DIR / "ai_providers"

# (file_relpath, kind) -> reason
ALLOWLIST: dict[tuple[str, str], str] = {
    ("bot/openai_helper.py", "client_attr:close"):
        "bot/openai_helper.py:4176-4178 resource cleanup on shutdown, no request "
        "semantics; not worth an indirection through the provider.",
}

def _violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "openai" or alias.name.startswith("openai.") for alias in node.names):
                out.append((node.lineno, "import_openai"))
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "openai" or node.module.startswith("openai.")):
                out.append((node.lineno, "import_openai"))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "client":
                out.append((node.lineno, f"client_attr:{node.attr}"))
    return out

def test_no_raw_openai_access_outside_ai_providers():
    failures = []
    for path in sorted(BOT_DIR.rglob("*.py")):
        if ALLOWED_DIR in path.parents:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for lineno, kind in _violations(path):
            reason = ALLOWLIST.get((rel, kind))
            if reason is None:
                failures.append(f"{rel}:{lineno}: {kind} (not allow-listed)")
    assert not failures, "\n".join(failures)
```

This is a sketch precise enough to implement directly, not literal final code — the
implementer should run it against the tree once written and confirm the only surviving
hit outside `bot/ai_providers/` is the allow-listed `close()` line.

---

## 4. Migration order (keep tests green at each step)

1. `bot/ai_provider.py`: add `ProviderError`/`ProviderRateLimitError`/
   `ProviderBadRequestError`/`ProviderStreamError` + Protocol methods. No behavior change,
   nothing imports them yet — safe standalone commit-sized step.
2. `bot/ai_providers/openai_compatible.py`: add `build_openai_client`,
   `raw_chat_completion`, `_translate_stream_errors`, extend `OpenAICompatibleProvider`
   with `get_client`/`get_gateway_client` + five methods. Existing methods untouched. Run
   `tests/test_ai_provider.py tests/test_openai_compatible_provider.py` — must still pass
   unchanged (new code is additive).
3. `bot/ai_providers/fake.py`: add the five parity methods. Run the same two test files.
4. `bot/openai_helper.py`: apply all of §2.4 in one pass (imports, constants, `__init__`
   wiring, delete the retry method, `_timed_create`'s one-line swap, the six method
   call-site swaps, `raw_generate_image`, both error-handler rewrites). This is the step
   most likely to need iteration — after it, run
   `tests/test_openai_helper_tool_calls.py` first (largest, most sensitive file) before
   the full suite.
5. `bot/openai_tool_handler.py`: the import + two `except` swaps from §2.5.
6. `bot/plugins/stable_diffusion.py`: the one-line swap from §2.6.
7. Delete `bot/plugin_tool_adapter.py` and `tests/test_plugin_tool_adapter.py`.
8. Update the tests listed in §5 (some must change before step 4/5 will even pass — see
   note below — but keep the diff for source and tests in the same logical step per
   file/behavior so `git diff` stays reviewable).
9. Add the AST guard test from §3 last, once the tree is clean, so it lands green
   immediately instead of red-then-fixed.

Note on ordering 4 vs 8: `tests/test_openai_helper_tool_calls.py` monkeypatches
`openai_helper_module.openai` in three tests (§5) — those three tests will hard-fail
(`AttributeError: module has no attribute`) the moment `import openai` is removed from
`bot/openai_helper.py` in step 4, before any of step 8's rewrites land. This is expected
and unavoidable (can't keep a deleted import "for tests"); do steps 4 and the
corresponding part of 8 as one atomic change, not two separate commits/passes.

---

## 5. Tests to add/update (all found by grepping for `openai_helper_module.openai`,
`openai_tool_handler_module.openai`, `LLM_RATE_LIMIT_RETRY`, `APIError`/`RateLimitError`/
`BadRequestError` across `tests/`)

**Must change (currently red the moment `import openai` is removed):**
- `tests/test_openai_helper_tool_calls.py:499-528`
  (`test_timed_create_retries_rate_limit_at_sdk_boundary`) — this test asserts the exact
  manual-retry behavior being deleted (2 calls, one `asyncio.sleep(20)`). Replace with a
  test asserting the **opposite**: one call, immediate `ProviderRateLimitError`, no sleep.
  Sketch:
  ```python
  async def test_timed_create_does_not_retry_rate_limit_manually(monkeypatch):
      class AlwaysRateLimitedClient(DummyClient):
          async def _create(self, **kwargs):
              self.calls += 1
              raise openai.RateLimitError("limited", response=..., body=None)  # or a minimal fake shaped like openai's real error

      sleep_calls = []
      monkeypatch.setattr(openai_helper_module.asyncio, "sleep", lambda s: sleep_calls.append(s))
      helper = _make_helper(DummyPluginManager({}), client=AlwaysRateLimitedClient())

      with pytest.raises(ProviderRateLimitError):
          await helper._timed_create(kind="unit", model="llmgateway/high", messages=[])

      assert helper.client.calls == 1
      assert sleep_calls == []
  ```
  (Constructing a real `openai.RateLimitError` needs a fake `httpx.Response`/body — check
  `openai._exceptions` for the minimal constructor shape in the installed 3.8.0, or raise
  a `SimpleNamespace`-free minimal subclass instance if the real constructor is awkward in
  a unit test; either way the test must raise something `isinstance(..., openai.RateLimitError)`
  so `raw_chat_completion`'s `_translate` picks it up.)
- `tests/test_openai_helper_tool_calls.py:774-810`
  (`test_get_chat_response_rate_limit_does_not_duplicate_user_message`) — same
  monkeypatch-of-`openai_helper_module.openai` problem, plus it currently asserts
  `pytest.raises(DummyRateLimitError)` (bare, unwrapped) and
  `helper.client.calls == openai_helper_module.LLM_RATE_LIMIT_RETRY_ATTEMPTS` (constant
  being deleted). Update to: raise a real/fake `openai.RateLimitError` from the client,
  expect `pytest.raises(Exception)` with the wrapped user message (now added per §2.4),
  and `helper.client.calls == 1`. The core assertion this test protects — no duplicate
  user message saved to history/DB on failure — must be kept, just re-pointed at the new
  exception type and call count.
- `tests/test_openai_helper_tool_calls.py:2517-2527`
  (`test_streaming_api_error_log_includes_error_text`) and `:2548-2556`
  (`test_streaming_plain_text_tool_intent_buffer_api_error_log_includes_error_text`) —
  both do `monkeypatch.setattr(openai_tool_handler_module.openai, "APIError",
  DummyAPIError)` then raise `DummyAPIError` from a hand-built async generator fed
  directly into `handle_function_call(..., response=response, ...)`. After §2.5, drop the
  monkeypatch entirely and raise `ProviderStreamError("secret streaming api error")`
  directly from the fake generator — simpler than before, no monkeypatch needed since
  `openai_tool_handler.py` now imports the concrete `ProviderStreamError` class from
  `bot.ai_provider`.

**Should verify still pass unchanged (no code references the deleted names, but they
exercise adjacent behavior):**
- `tests/test_openai_helper_tool_calls.py:490-496`
  (`test_common_chat_response_methods_are_not_wrapped_in_method_level_retry`) — checks
  `"@retry" not in inspect.getsource(...)` for a `tenacity`-style decorator; unrelated to
  this refactor, should stay green untouched.
- `tests/test_openai_helper_tool_calls.py:811-836`
  (`test_get_chat_response_provider_failure_logs_debug_values`) — raises a plain
  `RuntimeError`, not an `openai.*`/`Provider*` type; falls through to the generic
  `except Exception` path in `_common_get_chat_response`, unaffected.
- The ~85 other `helper.client = ...` / `helper.client.calls` / `helper.client.create_kwargs`
  sites across `tests/test_openai_helper_tool_calls.py`, `tests/test_skills_agent_gate.py`,
  `tests/test_reflection_on_tool_error.py`, `tests/test_session_logging_integration.py`,
  `tests/test_openai_helper_db_offload.py`, `tests/test_hindsight_memory.py`,
  `tests/test_llm_gateway_routing.py` — none of these hit `openai.RateLimitError`/
  `BadRequestError`/`APIError` paths, so they should be unaffected **by construction** of
  the `get_client=lambda: self.client` design (§1's "critical constraint", §2.4's `__init__`
  wiring). Explicitly run the full set as a regression check (see §7) rather than trusting
  this by inspection alone.

**New tests to add:**
- `bot/ai_providers/openai_compatible.py`: error translation — feed `create_chat_completion`
  callables that raise `openai.RateLimitError`/`openai.BadRequestError`/generic
  `openai.APIError`, assert `ProviderRateLimitError`/`ProviderBadRequestError`/`ProviderError`
  come out of `provider.create_response(...)` (or via `collect_ai_response(provider.stream_response(...))`).
  Also one streaming case: a fake stream that raises `openai.APIError` partway through
  iteration, assert `ProviderStreamError` surfaces from `_translate_stream_errors`.
- `bot/ai_providers/fake.py` new methods: at minimum one queue/consume round-trip per
  method (mirrors `test_fake_provider_consumes_queued_responses_in_order`).
- The AST guard itself (§3).

---

## 6. Explicitly flagged design choices (not silently resolved)

1. **`edit_image`/`list_voices` on the provider** (§2.2, §2.4): master-plan step 3 names
   them alongside the four SDK-backed methods; the AST guard as literally specified does
   not require moving them (different attribute name, `gateway_client` not `client`).
   Recommendation: do it anyway for interface uniformity, low risk (pure passthrough, no
   error-shape change since `LLMGatewayError` already isn't `openai.*`). Alternative: skip
   these two, leave `edit_telegram_image`/`get_available_tts_voices` calling
   `self.gateway_client.*` directly as today — nothing else in this plan depends on it.
2. **The remaining per-call `OpenAICompatibleProvider(self._timed_create, ...)` inside
   `_timed_create_via_ai_provider`** (§2.4's `__init__` note): not collapsed into
   `self._provider`. It's redundant with `self._provider` conceptually (both are
   "callable → AIProviderResponse/event" adapters) but harmless — it never touches
   `openai`/`.client.`, so nothing in the master plan's stated goals (retries, error
   translation, AST guard, provider-owns-raw-access) requires touching it. Collapsing the
   two layers into one is a legitimate follow-up but is a materially bigger, riskier
   rewrite of `_timed_create_via_ai_provider`'s and `_timed_create`'s shared
   responsibilities (two independently-tested logging layers, §1) for zero behavior
   change — recommend leaving it alone in T10.
3. **User-facing rate-limit message wording** (§2.4): reused the existing `'error'`
   i18n key rather than adding a new one (e.g. `'rate_limited'`), since no i18n file is in
   T10's ownership list and the master plan doesn't ask for new translated copy, only "a
   clear message". If a more specific message is wanted later, that's a follow-up touching
   i18n files, out of scope here.

---

## 7. Acceptance commands

```bash
~/.venvs/ctb/bin/python -m mypy bot/ai_provider.py bot/ai_providers/openai_compatible.py \
  bot/ai_providers/fake.py bot/openai_helper.py bot/openai_tool_handler.py \
  bot/plugins/stable_diffusion.py --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports

~/.venvs/ctb/bin/python -m ruff check bot/ai_provider.py bot/ai_providers/ bot/openai_helper.py \
  bot/openai_tool_handler.py bot/plugins/stable_diffusion.py

~/.venvs/ctb/bin/python -m pytest tests/test_ai_provider.py tests/test_openai_compatible_provider.py \
  tests/test_ast_no_raw_openai_access.py -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests/ bot/tests/ -q --no-header -p no:cacheprovider
```

`grep -rn "LLM_RATE_LIMIT_RETRY\|plugin_tool_adapter\|PluginToolAdapter" bot/ tests/`
(via a python inline script, not `rg`/`grep` directly — distorted output in this
environment) should return zero hits after the migration.

---

## 8. Risks

- **Real `openai.RateLimitError`/`BadRequestError` construction in tests**: the SDK's
  exception classes take `response`/`body` constructor args (`APIStatusError` subclasses
  wrap an `httpx.Response`); building one by hand in a unit test needs a minimal fake
  `httpx.Response` or `object.__new__` + manual attribute set. Check the exact constructor
  in the installed `openai==3.8.0` before writing the new tests in §5 — this is the single
  most likely spot to burn time during implementation.
- **`self.gateway_client` reassignment**: unlike `self.client`, no existing test currently
  overwrites `helper.gateway_client` after construction, so `get_gateway_client=lambda:
  self.gateway_client` is precautionary symmetry with `get_client`, not a verified-needed
  requirement — low risk either way since it costs nothing to keep dynamic.
- **`_translate_stream_errors` and cancellation**: wrapping the raw stream in an async
  generator adds one more `aclose()`-forwarding layer between the SDK stream and
  `_AIProviderStreamProxy`. `_AIProviderStreamProxy.aclose()` already does a generic
  `getattr(..., "aclose", None)` probe (`bot/openai_helper.py` in the class, unchanged),
  and async generators support `aclose()` natively, so this should be transparent —
  worth an explicit test (stream closed mid-iteration via `_AIProviderStreamProxy.aclose()`
  still propagates to the underlying raw stream) if one doesn't already exist.
- **`stable_diffusion.py`'s `_edit_image`** stays on `helper.gateway_client.image_edit`
  directly (not `helper.client`) — confirm this doesn't regress if the optional
  `edit_image`/`list_voices` provider methods (§6.1) are implemented, since that plugin
  method is separate code from `edit_telegram_image`'s gateway call and is not touched by
  this plan either way.
