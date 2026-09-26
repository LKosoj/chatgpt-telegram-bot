# W9 Review — GROUP C (LLM core flow) — Раунд 1

Scope: `bot/openai_helper.py` (all hunks except `__add_function_call_to_history`,
group A), `bot/ai_provider.py`, `bot/ai_providers/fake.py`,
`bot/ai_providers/openai_compatible.py`, `bot/ai_events.py`, `bot/chat_run.py`,
`bot/tool_result.py`, `bot/chat_response_utils.py`, `bot/openai_tool_handler.py`
(all except taint/wrapping), `bot/plugins/agent_tools.py` (all except
delivery/subagent-wrapping), `bot/plugins/hindsight_memory.py`,
`bot/plugins/skills.py` (prompt/catalog/mutator/git-clone parts only),
`bot/plugins/stable_diffusion.py`, deleted `bot/plugin_tool_adapter.py`, plus the
~18 listed test files. Diff base: `git diff HEAD` against `08bc457`.

## Method

Read every hunk in scope (`git diff HEAD -- <file>` per file, full function context
via `Read` where needed), cross-checked against `00-master-plan.md` and the
per-task `Txx-plan.md`/`Txx-review.md` docs that already covered this code
(T09, T10, T11, T12ad, T12c, T13c), and independently re-verified — not just
trusted — several specific behavioral-equivalence claims from those reviews by
diffing against `git show HEAD:<file>` originals:

- `_begin_turn`/`_begin_simple_turn` (openai_helper.py): both call sites' removed
  code is byte-identical to what `_begin_turn` now runs; parameters passed
  identically at both sites. Equivalent.
- `finalize_chat_answer` (chat_response_utils.py): body matches both original
  inline copies (`chat_run.py` tail, `_interpret_image_text_response`);
  `sum(token_accumulator or [])` vs. old `sum(token_accumulator)` is a no-op
  difference at the `chat_run.py` call site (token_accumulator is always a list
  there, never None) and the defensive form only matters for the
  `_interpret_image_text_response` site. The `_chat_request_usage_split`
  assignment being moved to after the `finalize_chat_answer` call in
  `chat_run.py` is safe: `_add_to_history` never reads
  `_chat_request_usage_split`, and `token_accumulator`/`usage_accumulator` are
  not mutated inside `finalize_chat_answer`.
- `_retry_after_empty_response` (chat_run.py) vs. the "theoretical asymmetry"
  T12c-review.md round 1 flagged but didn't escalate: traced
  `handle_function_call` (openai_tool_handler.py) end to end — every actual
  return path is a 2-tuple `return response, tools_used` where `response` is
  either the untouched input parameter or the (also-non-None-by-the-same-
  induction) result of an internal retry/repair closure. Confirmed:
  `handle_function_call` never returns `None` as its first element, so
  checking `retry_response is not None` post-`_handle_function_call` is
  equivalent to the old pre-call check. Not a bug.
- `_raise_provider_rate_limit_or_bad_request` (openai_helper.py): the RateLimitError
  path now wraps into a generic user-facing `Exception` instead of a bare
  `raise e` — looked like a behavior change at first read, but `T10-plan.md`
  explicitly specified this wrap for *both* call sites and `T10-review.md`
  confirms it was already in place pre-T12c; T12c only deduplicated two
  already-identical blocks. Not a new regression.
- `_translate_stream_errors`/`ProviderStreamError` (openai_compatible.py) fully
  covers what `except openai.APIError` used to catch in
  `openai_tool_handler.py`'s two streaming call sites (RateLimitError,
  BadRequestError, and any other `openai.APIError` subclass all funnel through
  the same `except openai.APIError as exc: raise ProviderStreamError(...)`).
  Error-translation completeness holds for the streaming path.
- `edit_image`/`list_voices` on `AIProvider`/`OpenAICompatibleProvider`/
  `FakeAIProvider` are dead code in production (`openai_helper.py` still calls
  `self.gateway_client.image_edit_file`/`audio_voices` directly, not
  `self._provider.edit_image`/`list_voices`) — confirmed via grep, and
  confirmed this is an explicitly accepted, documented decision
  (`T10-plan.md` §6.1/§7, `T10-review.md`): gateway-backed methods were added
  for future parity but wiring the two call sites was explicitly left
  optional. Not re-raised.
- `_dynamic_insert_index` (agent_tools.py, hindsight_memory.py, skills.py):
  traced the full insert sequence in `agent_tools.py`'s plan-rule/checkpoint/
  pending/pending_verify injection — `dyn_idx` is recomputed once after the
  (unmoved) plan-rule insert, then incremented per subsequent insert, so the
  three dynamic messages land contiguously right before the trailing user
  message (or at the end mid-tool-round), never splitting an assistant
  `tool_calls` message from its `tool` results. Same helper in
  `hindsight_memory.py`/`skills.py` used identically. Matches T09's design;
  no bug.
- `_clone_git_source`/`_clone_git_source_branch` merge (skills.py): new
  single-function branch handling (`if branch: command += ["-b", branch]`)
  reproduces both original command lines exactly; the one call site passes
  `branch=tree_branch` (falsy when absent), matching the old if/else split.
  Equivalent.
- `_ensure_paths()` return-type widening (`None` → `Path`, skills.py): all
  call sites updated to capture the return value; `self.skills_dir` is always
  set to a real `Path` by `initialize()` before the `assert` fires. No
  behavior change, plus one new call site (`_batch_install`-style path) adds a
  defensive `_ensure_paths()` call that wasn't there before — harmless
  (idempotent) hardening, not a behavior change for the existing paths.
- `env_bool` (hindsight_memory.py): docstring guarantees byte-for-byte
  equivalence to the replaced `.lower() == 'true'` idiom; matches.
- `_artifact_path` dedup: `bot/tool_result.py` has zero diff against HEAD;
  the copy removed from `openai_tool_handler.py` was byte-identical to the
  one already in `tool_result.py`. Clean dedup.
- Ran the full owned test list (`tests/test_openai_helper_tool_calls.py`,
  `tests/test_openai_helper_summarize_trim.py`,
  `tests/test_openai_compatible_provider.py`, `tests/test_ai_provider.py`,
  `tests/test_ast_no_raw_openai_access.py`, `tests/test_agent_tools_plugin.py`,
  `tests/test_agent_tools_plan_rule_mutator.py`,
  `tests/test_agent_tools_verify.py`, `tests/test_hindsight_mutator.py`,
  `tests/test_skills_plugin.py`, `tests/test_reflection_on_tool_error.py`,
  `tests/test_chat_response_utils.py`, `tests/test_no_private_helper_access.py`,
  `tests/test_no_hardcoded_plugin_refs.py`, `tests/test_t12c_*.py`,
  `tests/test_t12d_agent_tools_*.py`, `tests/test_t12d_skills_clone_git_source.py`)
  — 498 passed, 0 failed. Also ran
  `scripts/mypy_baseline.py check` — passed (no new mypy regressions
  project-wide).
- Confirmed `get_spec()`/`chat_modes.yml` `tools:` lists untouched for files in
  scope (no diff hunks touch spec dicts or the YAML tools lists).
- Deletion of `bot/plugin_tool_adapter.py`: confirmed no remaining
  `plugin_tool_adapter`/`PluginToolAdapter` reference anywhere in `*.py`
  except the one flagged below.

## Findings

### WARNING 1 — Stale `PluginToolAdapter` docstring reference after deletion

- **File:** `bot/ai_events.py:29`
- **What:** `AIToolCall`'s class docstring still reads "Provider adapters use
  the model-visible tool name. `PluginToolAdapter` canonicalizes it before
  execution and preserves the raw provider name in `model_name`." The
  `PluginToolAdapter` class (`bot/plugin_tool_adapter.py`) is deleted by this
  changeset. Canonicalization now happens inline via
  `plugin_manager.to_canonical_function_name(...)` at the two call sites
  (`bot/openai_tool_handler.py:527`, `bot/openai_helper.py:1458`), not through
  any adapter class.
- **Why this wasn't already fixed:** `T10-plan.md` (line 75-79) explicitly
  found this exact docstring during T10 (the task that deletes
  `plugin_tool_adapter.py`) and explicitly left it alone with the reasoning
  "not owned by T10 (file not in ownership list)". `T10-review.md` confirms
  the deferral. At the time that was a reasonable scope call, but
  `bot/ai_events.py` **is** in this wave's (GROUP C) ownership list, so the
  deferred cleanup is now in scope and the reference is a real, verifiable
  dangling pointer to a class that no longer exists anywhere in the tree (not
  even as dead code) — not merely "not literally the class, but still
  describes the design intent" as T10-plan characterized it.
- **Failure scenario:** Not a runtime bug (docstrings don't execute), but a
  maintainer reading `AIToolCall` to understand tool-call canonicalization
  will look for `PluginToolAdapter` and find nothing, wasting time or drawing
  wrong conclusions about where canonicalization happens.
- **Suggested fix:** Reword to name the actual current mechanism, e.g.:
  "Provider adapters use the model-visible tool name;
  `PluginManager.to_canonical_function_name` canonicalizes it before
  execution, and the raw provider name is preserved in `model_name`."

## Summary

- **ERROR:** 0
- **WARNING:** 1 (stale docstring, `bot/ai_events.py:29`)
- **NIT:** 0

No forbidden tool-spec changes, no behavior regressions, no concurrency/async
issues, no broken error handling, and no dead code found beyond the one
already-accepted `edit_image`/`list_voices` case (documented above as not
re-raised) and the docstring above. All prior per-task reviews covering this
group's files (T09, T10, T11, T12ad, T12c, T13c) were independently
re-verified rather than trusted at face value, and held up.

## Раунд 2

### Round-1 WARNING verified fixed

- `bot/ai_events.py:29` — `AIToolCall`'s docstring now reads
  "Provider adapters use the model-visible tool name.
  `PluginManager.to_canonical_function_name` canonicalizes it before execution
  and preserves the raw provider name in `model_name`." (confirmed by reading
  the file). No remaining `PluginToolAdapter` reference in any `*.py` file
  (`git grep -i plugin_tool_adapter` only hits pre-existing historical
  planning docs under `docs/`, none in scope, none live code).

### Fresh pass — emphasis areas from the task

- **Streaming mid-iteration error translation**
  (`bot/ai_providers/openai_compatible.py:54-68` `_translate_stream_errors`,
  `bot/openai_helper.py:98-166` `_AIProviderStreamProxy`,
  `bot/openai_tool_handler.py:1235-1304`): `openai.APIError` raised while
  iterating the SDK stream is uniformly wrapped as `ProviderStreamError` and
  propagated through the proxy; both `handle_function_call` catch sites
  (`:1239`, `:1299`) were mechanically retargeted from `except
  openai.APIError` to `except ProviderStreamError` with byte-identical
  surrounding logic — diffed against `git show 08bc457:bot/openai_tool_handler.py`
  to confirm. Not a new regression. (Separately: both catch sites `return
  response, tools_used` with `response` being the now-exhausted stream
  object; this pattern is unchanged from HEAD, so it's a pre-existing
  behavior outside this changeset's diff, not raised as a new finding.)
- **SDK `max_retries=3` on client construction**: exactly one
  `openai.AsyncOpenAI(...)` construction site in the whole tree
  (`bot/ai_providers/openai_compatible.py:30-41` `build_openai_client`,
  confirmed via `git grep -n "AsyncOpenAI("`), with `max_retries: 3` set
  unconditionally. Image (`generate_image`/`list_models`/`speech`/
  `transcribe`) and the `raw_generate_image` path used by
  `stable_diffusion.py:52` all resolve through the same `self._get_client()`
  → `self.client`, so no separate uncovered client exists. `edit_image`/
  `list_voices` still go through `gateway_client` (`LLMGatewayClient`, a
  non-SDK HTTP client, file unchanged/out of scope) — already documented as
  accepted dead-in-production code in Round 1. `max_retries=3` itself
  predates this changeset (present at `git show 08bc457:bot/openai_helper.py:258`
  already); T10's actual change was removing the *duplicate* manual retry
  loop stacked on top of it (`_create_chat_completion_with_rate_limit_retry`,
  confirmed deleted), not introducing the SDK retry. No test asserts
  `max_retries=3` in `client_kwargs`, but since the value/behavior is
  unchanged from HEAD this isn't "risky new logic" — not raised as a
  finding.
- **Router prompt (`_build_auto_chat_mode_prompt`) output parsing**
  (`bot/openai_helper.py:4204-4234` builds the prompt,
  `:1233-1238` parses the reply via `mode_name.strip().lower()` →
  `chat_modes_registry.get_mode_by_key`): parsing code itself is untouched by
  this diff. The prompt-text diff matches T09-plan exactly — user query
  moved after the mode list into the trailing tag (`:4233-4234`), and rule 4
  softened from "не выбирай skills_agent по отдельным словам" to "совпадение
  одного отдельного слова — не сильный сигнал" (confirmed against
  `git diff HEAD -- bot/openai_helper.py`). No mismatch found.
- **Mutator ordering across tool rounds**: confirmed
  `_apply_before_chat_request_mutators(..., persist=False)` is called fresh
  before every LLM re-entry round, not just the first call of a turn — all
  three call sites checked: `bot/openai_tool_handler.py:1013-1019` (delivery
  repair), `:1092-1098` (plain-text tool-intent repair), `:1718-1724` (main
  post-tool-execution re-entry). Only the very first call
  (`bot/openai_helper.py:1394-1403`, inside `_common_get_chat_response`) uses
  `persist=True`. `_dynamic_insert_index` (identical in `agent_tools.py:201`,
  `hindsight_memory.py:494`, `skills.py:84` — verified byte-identical again)
  correctly falls back to "insert at end" once the trailing message is no
  longer `role=="user"` (i.e. mid tool-round, where the tail is a `tool`
  result), so dynamic content never gets misplaced across rounds.
  `hindsight_memory`'s dynamic recall is intentionally gated to
  `persist=True` calls only (`allow_dynamic_recall` payload field, wired at
  `hindsight_memory.py:2477-2479`) — i.e. it re-queries memory once per user
  turn, not once per tool round; baseline recall (`needs_baseline_recall`)
  is correctly suppressed on later rounds because the first round's
  injected message gets persisted into `self.conversations[state_key]`
  before those rounds run. No ordering bug found.
- **Prompt texts** — `_PLAN_RULE_TEXT` (`agent_tools.py:175-181`): diff
  confirms threshold changed from "3+ tool-вызова" to "больше двух шагов" and
  the two "одной строкой укажи намерение/оцени" sentences were removed, both
  matching T09-plan step 2 exactly. `_verify_message_body`
  (`agent_tools.py:2159-2166`): "одной фразой подтверди" (plain-text
  confirmation) replaced with "проверь через manage_plan_tasks" (forces a
  tool call), removing the conflict with skills_agent's "не отвечайте
  промежуточным plain-text" rule (`chat_modes.yml:1709`) that T09-plan
  flagged. `_replan_message_body` (`agent_tools.py:2146-2157`, unchanged by
  this diff) already required a tool call, so no analogous conflict existed
  there. `ask()` prompt (`openai_helper.py:794-796`): typo "помошник" fixed
  to "помощник", date format changed from unreadable `%Y%m%d%H%M%S` to
  `%Y-%m-%d %H:%M:%S UTC` — matches T09-plan step 3 exactly. No typos found
  in the changed text. `chat_modes.yml`'s skills_agent block documents the
  `[re-plan trigger-v1] ` marker explicitly (2c) but never mentions
  `[verify-step-v1] `; checked whether this is a new gap — it is not: the
  verify-trigger mechanism, its marker text, and `chat_modes.yml`'s
  documentation gap around it all predate this changeset unchanged
  (confirmed via `git show 08bc457` on both files), and `chat_modes.yml` is
  outside this wave's ownership except for the one already-accepted
  "Function … returned" line. Not raised as a new finding.

### Verification

- Full owned test subset re-run: `tests/test_openai_helper_tool_calls.py`,
  `tests/test_openai_helper_summarize_trim.py`,
  `tests/test_openai_compatible_provider.py`, `tests/test_ai_provider.py`,
  `tests/test_ast_no_raw_openai_access.py`, `tests/test_agent_tools_plugin.py`,
  `tests/test_agent_tools_plan_rule_mutator.py`,
  `tests/test_agent_tools_verify.py`, `tests/test_hindsight_mutator.py`,
  `tests/test_skills_plugin.py`, `tests/test_reflection_on_tool_error.py`,
  `tests/test_chat_response_utils.py`, `tests/test_no_private_helper_access.py`,
  `tests/test_no_hardcoded_plugin_refs.py` — 470 passed, 0 failed. Plus
  `tests/test_t12c_*.py`, `tests/test_t12d_agent_tools_*.py`,
  `tests/test_t12d_skills_clone_git_source.py` — 28 passed, 0 failed.

### Summary (Раунд 2)

- **ERROR:** 0
- **WARNING:** 0 (round-1's single WARNING confirmed fixed)
- **NIT:** 0

No new findings. Round 1's single WARNING is fixed. The fresh pass targeted
at streaming error translation, `max_retries=3` client construction, router
prompt parsing, cross-round mutator ordering, and prompt-text
typos/contradictions found nothing new — every candidate issue traced back
to either matching the relevant `Txx-plan.md` exactly or predating this
changeset untouched.
