# Agent Instructions

These instructions are project-specific. Apply them together with any higher-level agent
rules from the current session.

## Working Rules

- Before non-trivial edits, state assumptions, success criteria, and a short verification plan.
- Keep changes surgical. Do not refactor adjacent code, reformat unrelated files, or remove
  unrelated dead code.
- Prefer the existing project style over new abstractions. Add abstractions only when they
  remove real duplication or match an established local pattern.
- Use `rg`/`rg --files` for repository search.
- Check `git status --short` before editing and do not revert unrelated user changes.
- Do not make runtime claims from memory. Verify the concrete function or method in code and
  cite `file:line` when explaining behavior.
- Do not call external services during verification unless the task explicitly requires it.
- Не борись с ошибками! Каждый раз, когда ты сталкиваешься с одной и той же ошибкой дважды, изучи веб и найди 3–5 возможных способов её исправления.
  Затем выбери самое эффективное решение и реализуй его.
- Нельзя ничего коммитить и создавать ветки!

## Project Shape

- Runtime is a Python Telegram bot. The process entrypoint is `bot/__main__.py`.
- Required runtime env vars are `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`; startup exits when
  either is missing (`bot/__main__.py:183-186`).
- Startup creates `PluginManager`, `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`
  (`bot/__main__.py:349-362`).
- Telegram polling is owned by `ChatGPTTelegramBot.run()` in `bot/telegram_bot.py`; the current
  builder enables concurrent updates, local Telegram Bot API mode, and
  `http://localhost:8081/bot` as base URL (`bot/telegram_bot.py:6288-6303`).
- Main request flow:
  - Telegram update handling lives mostly in `bot/telegram_bot.py`.
  - OpenAI-compatible chat/image/audio/vision access is in `bot/openai_helper.py`.
  - Tool-call extraction and plugin execution are in `bot/openai_tool_handler.py`.
  - Plugin discovery, tool specs, command metadata, and argument validation are in
    `bot/plugin_manager.py`.
  - SQLite persistence is in `bot/database.py`.
  - Chat mode loading and tool validation are in `bot/chat_modes_registry.py` plus
    `bot/chat_modes.yml`.
- `requirements.txt` is the primary dependency list. `environment.yml` is an alternate Conda
  environment and must not be treated as an exact mirror.
- use .venv

## Plugin And Tool Rules

- Plugins subclass `bot.plugins.plugin.Plugin` and implement `get_source_name()`,
  `get_spec()`, and async `execute(function_name, helper, **kwargs)`
  (`bot/plugins/plugin.py:109-124`).
- Stable plugin identity is `plugin_id`; tool namespace is `function_prefix`, defaulting to
  `plugin_id` (`bot/plugins/plugin.py:11`, `bot/plugins/plugin.py:18`).
- `PluginManager` loads plugin modules from `bot/plugins/*.py`, excluding the files listed in
  `NON_PLUGIN_MODULES` (`bot/plugin_manager.py:34`) — the base class plus the framework
  modules `background.py`, `db_handle.py`, `hooks.py`; empty/unset `PLUGINS` loads all
  plugins, and non-empty `PLUGINS` acts as a comma-separated allow-list
  (`bot/plugin_manager.py:223`, `bot/plugin_manager.py:234`).
- The loader runs a plugin through `exec_module` without registering it in `sys.modules`, so a
  plugin module must not combine `from __future__ import annotations` with `@dataclass`:
  `dataclasses` resolves the resulting string annotations through
  `sys.modules[cls.__module__].__dict__` and the plugin silently fails to load with
  `'NoneType' object has no attribute '__dict__'`. Guarded by
  `tests/test_plugin_manager.py::test_plugin_modules_do_not_combine_future_annotations_with_dataclass`.
- Function specs must be unique after namespacing. Unqualified spec names are normalized to
  `<function_prefix>.<name>` (`bot/plugin_manager.py:749`).
- Duplicate function names are invalid. With `PLUGIN_STRICT_VALIDATION=true`, duplicates raise;
  otherwise they are logged and skipped (`bot/plugin_manager.py:334`).
- Tool arguments are JSON-decoded and validated against the function spec before plugin
  execution (`bot/plugin_manager.py:428`, `bot/validation.py:33`).
- Tool calls may arrive in batches and are executed with `asyncio.gather`; `chat_id` and
  `user_id` are injected into arguments before execution (`bot/openai_tool_handler.py:181`,
  `bot/openai_tool_handler.py:1358-1360`).
- A plugin response marked as a direct result short-circuits model re-entry
  (`bot/openai_tool_handler.py:416-420`, `bot/openai_tool_handler.py:1593-1596`).
- Google model tool specs use `{"function_declarations": specs}` while other models receive
  OpenAI-style `{"type": "function", "function": spec}` entries
  (`bot/plugin_manager.py:354-355`). Note: the Google branch is currently unreachable because
  `GOOGLE_MODELS` is an alias for `GOOGLE` (`bot/plugin_manager.py:20`), which is an empty
  tuple (`bot/model_constants.py:24`).
- Core modules (`bot/openai_helper.py`, `bot/telegram_bot.py`, `bot/database.py`) must not
  introduce new hardcoded plugin-id references. Generic `get_plugin(plugin_id)` reads for UI
  menus and the documented Strategy Z compromise are tracked in
  `tests/test_no_hardcoded_plugin_refs.py` allow-list — bump or lower the entry when changing
  intentionally.

## Hooks: contract and lifecycle

The plugin hook framework lives in `bot/plugins/hooks.py` (events + payloads) and
`bot/plugin_manager.py` (dispatcher). Plugins override no-op defaults from
`bot.plugins.plugin.Plugin`.

There are four kinds of hooks, each with a different dispatch policy:

1. **Observers** (`dispatch_observe`): `on_user_message`, `on_assistant_response`,
   `on_session_reset`. Fire-and-forget; all subscribers run **concurrently** via
   `asyncio.gather(..., return_exceptions=True)`. Plugins return `None`. Exceptions are
   logged and swallowed; one failing plugin does not block others.
2. **Blocking hooks** (`dispatch_blocking`): `on_session_before_delete`. Awaited
   **sequentially** before the action (e.g. session deletion) proceeds. Exceptions are
   logged and swallowed — the action still completes (Policy A: PII delete must not be
   blocked by plugin failure).
3. **Mutators** (`apply_mutators`): `on_before_chat_request`. Plugins are awaited
   **sequentially**; each receives the current value (e.g. `messages: List[Dict]`) and a
   payload; returns a possibly-modified value or `None` (= no change). Identity on failure:
   a raising plugin yields the unchanged value from the previous step. Order is
   deterministic — `sorted(self.plugins.keys())`, i.e. by plugin module name.
   Active mutators in tree:
   - `agent_tools.on_before_chat_request` (`bot/plugins/agent_tools.py:346`) — injects the
     planning-prefix system message that reminds the model to call `manage_plan_tasks`
     before non-trivial work.
   - `hindsight_memory.on_before_chat_request` (`bot/plugins/hindsight_memory.py:2437`) —
     injects a recalled long-term-memory system message when auto-recall is enabled.
4. **Collectors** (`collect_fragments` / `collect_objects`): named slots, called
   **sequentially**. Active slots in tree: `auto_mode_priority` (auto-mode prompt prefix,
   `bot/openai_helper.py:4272-4274`), `stats_block` (`/stats` extra blocks,
   `bot/telegram_bot.py:1290`), `settings_menu_buttons` (extra settings-menu button rows,
   `bot/telegram_bot.py:1583-1587` — only consumer of `collect_objects`). Each plugin's
   `contribute_prompt_fragment(slot, payload)` returns a string fragment (for
   `collect_fragments`) or an arbitrary object (for `collect_objects`) or `None`. Skipped
   on exception. Caller decides composition (e.g. `"\n\n".join(...)`).

Payload classes are frozen dataclasses defined in `bot/plugins/hooks.py`. New events should
add a new `HookEvent` member and a frozen payload class; the dispatcher then routes by event
name.

## Plugin-owned tables

Plugins that need persistent storage declare DDL via `register_schema()` and access the DB
through `self.db_handle` (async `DbHandle` facade: `execute`/`executemany`/`fetch_one`/
`fetch_all`/`transaction()`). `PluginManager` runs `register_schema()` statements at startup
once per plugin; tables created this way live alongside core tables but are owned by the
plugin and are removed from `bot/database.py`.

Examples in tree: `bot/plugins/hindsight_memory.py:1181-1199` (`hindsight_finalize_jobs` DDL in
`register_schema()`), `bot/plugins/agent_tools.py` (`agent_plan_contracts` /
`agent_plan_tasks`). The `kind` column on `hindsight_finalize_jobs` distinguishes
`session_close` from `burst` jobs (see Background tasks); long-term-memory consolidation
(`_consolidate_dream_document`, `bot/plugins/hindsight_memory.py:1906`) merges a new summary
into an existing document via a bounded ADD/DELETE action protocol
(`parse_consolidation_actions` / `apply_consolidation_actions`,
`bot/plugins/hindsight_memory.py:238`, `:264`), not a schema change. Plugins that own
a table without `ON DELETE CASCADE` to a core table are responsible for their own GC if/when
a user-deletion mechanism is introduced.

## Plugin config segments

Plugins declare a config prefix via `get_config_prefix()`. `PluginManager.config` is a single
dict; the plugin reads only its own slice (keys with that prefix) and may mirror defaults
into `openai.config` via `setdefault` during `initialize()` for compatibility with helper
code that hasn't migrated yet. Mirrors are documented (see Stage 4A notes in
`docs/plugin-hooks-migration.md`).

## Background tasks

Plugins return `BackgroundTask(name, interval_seconds, coroutine_factory)` entries from
`get_background_tasks()`. `PluginManager.start_background_tasks(application)` spawns them
with deterministic interval scheduling; `close_async()` cancels them on shutdown. Reminders,
hindsight finalize worker, and agent_tools cleanup all run this way — core code (telegram
bot, openai helper) no longer launches plugin-specific workers.

Hindsight also registers a `burst_sweep` task (`bot/plugins/hindsight_memory.py:825-845`) that
periodically flushes per-`(user_id, chat_id, autonomous)` in-memory turn buffers accumulated
mid-conversation into a `hindsight_finalize_jobs` row once a turn-count or quiet-time threshold is
hit (`HINDSIGHT_BURST_MAX_TURNS` / `HINDSIGHT_BURST_QUIET_SECONDS`), instead of waiting for session
close; `close_async()` drains any remaining buffers first.

The third key component keeps autonomous turns out of the same extraction job as live ones. A turn
is autonomous only when `agent_cron` dispatches `on_assistant_response(autonomous=True)`, which it
does only under `HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED` (default `false` — cron turns otherwise
never reach the memory hooks at all). The flag rides the job's `autonomous` column through to
`_extract_hindsight_memory_items`, which appends `AUTONOMOUS_EXTRACTION_ADDENDUM` to the extractor
prompt. `session_close` jobs are always `autonomous=False`: that history interleaves live and cron
turns, so flagging it wholesale would drop real user facts.

## Chat Modes

- Chat modes are defined in `bot/chat_modes.yml` and loaded through `ChatModesRegistry`.
- `OpenAIHelper` constructs the registry and validates mode tool references during init
  (`bot/openai_helper.py:305-306`).
- Missing tool references in `chat_modes.yml` are logged by `validate_tools()`
  (`bot/chat_modes_registry.py:82`).
- During request preparation, the active mode can restrict allowed plugins via its `tools`
  field; absent mode tooling defaults to `['All']` (`bot/openai_helper.py:1315`,
  `bot/openai_helper.py:1346-1347`).
- When editing chat modes, keep plugin names aligned with loaded plugin module names, not
  human-readable descriptions.

## Tool And Context Footprint

Every registered tool spec lands in every prompt reachable by its allow-list; there is no
per-turn trimming beyond `chat_modes.yml`'s `tools:` list and hook self-gating (see below). The
bar for adding a new tool is high because of this. Ranked cheapest to most expensive:

1. **Bot commands.** `PluginManager.get_plugin_commands()` (`bot/plugin_manager.py:865`) and
   `build_bot_commands()` (`bot/plugin_manager.py:892`), registered in `post_init()` in
   `bot/telegram_bot.py`, never enter the model's `tools` array. Free.
2. **Skills.** `SkillsPlugin.get_spec()` (`bot/plugins/skills.py:375`) always returns the same
   fixed set of tool specs regardless of how many skills are installed. A new skill adds only
   one truncated `id: description` line: to the `auto_mode_priority` prompt fragment
   (`bot/plugins/skills.py:164` — `contribute_prompt_fragment`) and to the `skills_agent` mode
   catalog (`bot/plugins/skills.py:208` — `on_before_chat_request`). The cheapest way to add a
   capability the model does not need to be able to call by name on every turn.
3. **Chat modes** (`bot/chat_modes.yml`) are free by themselves: a mode is a system prompt plus
   a `tools:` allow-list, read at `bot/openai_helper.py:1346-1347`, defaulting to `['All']`
   (`bot/openai_helper.py:1315`). Narrowing `tools:` is the main lever for cutting per-call
   payload; most modes in tree list a short explicit set of plugins (commonly 7-10) instead of
   `All`.
4. **New plugin/tool** — one full JSON schema on every request where `allowed_plugins` includes
   it. With an empty `PLUGINS` env var, any `.py` dropped into `bot/plugins/` loads
   automatically and becomes available to every mode with `tools: [All]` without touching
   `chat_modes.yml`.
5. **Hooks** — the most expensive rung. `_active_plugin_instances()`
   (`bot/plugin_manager.py:993`) filters ONLY by the per-user disabled-plugin set, NOT by the
   active mode's `tools:` allow-list. An `on_before_chat_request` mutator therefore runs on
   every request in every mode unless it checks itself. In tree, both
   `agent_tools.on_before_chat_request` (`bot/plugins/agent_tools.py:346`) and
   `skills.on_before_chat_request` (`bot/plugins/skills.py:208`) perform that self-check. A new
   hook without one runs unconditionally.
6. **MCP servers** — the least controllable rung: `register_mcp_server` is itself a
   model-callable tool (`bot/plugins/mcp_server.py:183`), and every tool of a connected remote
   server becomes a full spec at runtime (`bot/plugins/mcp_server.py:275-283`) with no code
   review.

Practical rule: prefer a skill over a new tool when the capability is rarely-needed procedural
knowledge rather than something the model must call by name on every turn. When adding a hook,
decide explicitly whether it should be scoped to a chat mode, and if so, self-gate it — the
framework will not do that for you.

## Deterministic Routing In Agent Plugins

Code decides what happens after a tool call, not the model; the model only reports state. This
is already the pattern in `agent_tools` — new agent-style plugins should follow the same shape,
not treat this as a refactor mandate:

- `_manage_plan_tasks` (`bot/plugins/agent_tools.py:2547`): the model only writes a task's
  status; the code detects the transition and decides the consequence —
  `status=blocked` schedules a re-plan, `status=completed` schedules a verify step
  (`bot/plugins/agent_tools.py:2599-2602` for `action=add`, `:2657-2660` for `action=update`),
  applied via `_apply_plan_runtime_effects` (`bot/plugins/agent_tools.py:2080`).
- `_record_tool_outcome` (`bot/plugins/agent_tools.py:2103`) counts consecutive tool failures
  on the same task and schedules a re-plan itself once a threshold is reached.
- `_reentry_tool_choice(...)` (`bot/openai_tool_handler.py:868`) is a pure function of the
  round counter that picks `"auto"`/`"none"`; once `functions_max_consecutive_calls` is
  exhausted the code forcibly narrows the tool set to the delivery tool
  (`bot/openai_tool_handler.py:930-931`).

`describe_plan_lifecycle()` (`bot/plugins/agent_tools.py:38`) is the single source of truth for
the task-plan lifecycle: the full status set, its terminal/open subsets, the cross-task
invariants actually enforced by `_validate_plan_tasks` (`bot/plugins/agent_tools.py:2349`), and
the status → side-effect mapping planned by the code above. It reads `TASK_STATUSES`/
`CLOSED_STATUSES` rather than holding its own copy, so it cannot drift from what the code
validates. It is not part of `get_spec()` and costs no prompt tokens — it exists for tests and
documentation.

The task-status set is currently duplicated by hand: the `TASK_STATUSES` constant
(`bot/plugins/agent_tools.py:26`) and the `enum` literal in `manage_plan_tasks`'s JSON schema
(`bot/plugins/agent_tools.py:509`) both list the same five values, with nothing enforcing they
stay in sync. `tests/test_agent_tools_plan_lifecycle_describe.py` cross-checks
`describe_plan_lifecycle()` against both `TASK_STATUSES` and the tool spec's `enum` as a
regression guard. Keep the two definitions in sync when changing task statuses, or derive the
`enum` from `TASK_STATUSES` instead.

## Chat Token Pricing

- Cost is computed and stored at write time, not derived at read time. `bot/pricing.py`
  resolves one completion's cost and reports how it was priced via `price_source`:
  `model_split` (per-direction rates), `model_blended` (the two rates averaged, used when only
  a total is known), `legacy_fallback` (flat `TOKEN_PRICE`, for models absent from the table).
- The per-model table comes solely from the `MODEL_TOKEN_PRICES` env var;
  `DEFAULT_MODEL_TOKEN_PRICES` (`bot/pricing.py:25`) is deliberately empty because the models
  in tree are gateway aliases whose real prices are installation-specific.
- The non-streaming path accumulates a per-round-trip split through `usage_accumulator` and
  collapses it with `aggregate_usage_split` (`bot/chat_response_utils.py:72`), which returns
  `None` on any length mismatch — an incomplete split is reported as unknown, never guessed.
- Streaming yields a real split only when `STREAM_INCLUDE_USAGE=true` makes the API append a
  final usage chunk. It is off by default because `stream_options` is not universally accepted
  by OpenAI-compatible gateways and an unsupported parameter fails the whole request. Even
  then the split is recorded only for turns with no tool call: after one, the stream being read
  is a re-entry stream whose usage covers just the last round trip.
- When adding a new pricing path, keep the same rule: report unknown rather than a split that
  silently omits round trips.

## Model Utility Calls

`bot/model_utilities.py`'s `ModelUtilities` (`bot/model_utilities.py:16`) is a thin, stateless
wrapper around `helper.chat_completion(**kwargs)` — not the `AIProvider` interface — for cheap
one-off model calls: `one_shot`, `classify_json`, `generate_title`, `summarize_window`. It only
touches `helper.chat_completion`/`helper.config`, so it also runs against minimal test doubles.
All four apply `asyncio.wait_for(timeout_seconds)` and degrade to `None` on error, except
`summarize_window`, which re-raises so `OpenAIHelper._summarize_and_trim`
(`bot/openai_helper.py:3769`) can catch it and fall back to a deterministic trim.

## Conversation History Compaction

When history needs to shrink, `OpenAIHelper._summarize_and_trim()`
(`bot/openai_helper.py:3769`) tries an LLM summary via `ModelUtilities.summarize_window`; if it
returns `False` (throttled, unresolvable cut, or the summary call itself failing/timing out),
`_fallback_trim_with_summary()` (`bot/openai_helper.py:3856`) head-preserve-trims the window
instead, replacing the cut portion with a deterministic (no model call) excerpt from
`_deterministic_summary_text()` (`bot/openai_helper.py:3718`) — a bounded head+tail rendering —
so history is compacted, never silently dropped.

## Terminal Command Policy

`bot/command_policy.py` backs the terminal plugin's guard: `evaluate_command()`
(`bot/command_policy.py:434`) normalizes a command string (unquoting, `$(...)`/backtick
expansion, heredoc stripping) and matches it against `CommandRule` patterns — built-in
`DEFAULT_RULES` plus any layered from `TERMINAL_COMMAND_POLICY` (JSON, via
`load_policy_from_env()` at `bot/command_policy.py:410`) — to a `CommandDecision` of
`allow`/`deny`/`require_approval`; `TERMINAL_APPROVAL_MODE` governs how the terminal plugin
acts on `require_approval`. It is a heuristic over command text, not a sandbox boundary:
bypassable by obfuscation or by writing a script to a file and then executing it
(`bot/command_policy.py:4-7`).

## Telegram Handler Rules

- Plugin commands are normalized through `PluginManager.get_plugin_commands()` and registered
  in `post_init()` as command handlers or callback handlers (`bot/plugin_manager.py:865`,
  `bot/telegram_bot.py:5337-5366`).
- Plugin command names must not include spaces; a leading `/` is stripped during normalization
  (`bot/plugin_manager.py:927`).
- Plugin message handlers can provide a ready handler object or a `filters.X` string/object.
  Invalid filters are logged and skipped (`bot/telegram_bot.py:5238-5243`).
- Do not reintroduce `eval` for handler filters.

## Database Rules

- `Database` is a singleton with thread-local SQLite connections and an operation `RLock`
  (`bot/database.py:73`, `bot/database.py:80`).
- New SQLite connections enable foreign keys, WAL by default, and `busy_timeout`
  (`bot/database.py:121-128`).
- Async DB access (the `Database.*_async` methods and the `DbHandle` facade) routes through a
  single dedicated worker thread — `Database._run_in_db_thread` over a `max_workers=1`
  `ThreadPoolExecutor` — instead of bare `asyncio.to_thread`. This bounds thread-local
  connections to one worker; the executor is torn down by `Database.shutdown()` (called from
  `_reset_singleton` and `__del__`). The operation `RLock` is unchanged.
- `conversation_context.context` is JSON shaped as `{"messages": [...]}`; do not migrate or
  seed it as a bare list (`bot/database.py:476`).
- `conversation_context` carries a monotonic `version` column (schema `TARGET_SCHEMA_VERSION`
  = 2); each `save_conversation_context()` UPDATE bumps it by one. Writes are serialized by
  the per-operation write-lock (`transaction()` = `BEGIN IMMEDIATE`), so `version` is a
  revision counter, not a rejecting CAS gate. Migration adds the column idempotently across
  fresh-install, v1→v2, and legacy (no-`session_id`) paths.
- `save_conversation_context()` persists `message_count` as the number of user-role messages
  (`bot/database.py:787`).
- Session creation prunes old sessions inline via `_oldest_session_ids_for_limit()`;
  there is no separate oldest-session deletion API.
- Keep long-running OpenAI calls outside active DB transactions. Async session-name
  generation is handled by `OpenAIHelper._ensure_session_name_with_llm()` after the DB write
  (`bot/openai_helper.py:718`); `Database.ensure_session_name_async()` at
  `bot/database.py:902` provides a short fallback but no longer calls the LLM directly.

## Testing And Verification

- Pytest is configured in `pytest.ini` with `asyncio_mode = auto`.
- Main top-level tests live under `tests/`; MCP-specific tests live under `bot/tests/`.
- Prefer targeted tests for touched behavior:
  - plugin registry/specs/commands: `tests/test_plugin_manager.py`,
    `tests/test_plugin_commands.py`, `tests/test_plugin_arg_validation.py`
  - tool-call routing: `tests/test_openai_helper_tool_calls.py`
  - chat mode validation: `tests/test_chat_modes_registry.py`
  - SQLite sessions/context: `tests/test_database.py`
  - MCP plugin behavior: `bot/tests/test_mcp_server.py`
  - hook framework: `tests/test_plugin_hooks.py`, `tests/test_db_handle.py`
  - core/plugin boundary: `tests/test_no_hardcoded_plugin_refs.py`
- `tests/test_exemplar_*.py` is a layer distinct from per-function unit tests: each exercises
  one real, multi-step code path (burst-buffer-to-finalize-job flow, tool-call-interruption
  repair, summarize-then-fallback compaction, terminal command guard) and asserts on the
  *structure* of the result (counts, markers, invariants) rather than exact model/text output,
  so it stays deterministic without pinning wording. Still plain `pytest`-collected, no
  network — unlike `evals/`.
- For narrow documentation-only edits, inspect the rendered Markdown or run no tests and state
  that no runtime tests were needed.
- `evals/` is a third category and is never part of "run the tests". Everything under `tests/`
  and `bot/tests/` is deterministic and fully mocked; `evals/judge` sends real requests to a
  real model and has a second model score the reply against a rubric, so it is
  non-deterministic, needs network, and costs money. It is guarded three ways —
  `testpaths = tests bot/tests` in `pytest.ini` (a bare `pytest` never collects it), a `skipif`
  on `RUN_LLM_JUDGE_EVALS=1` *and* `OPENAI_API_KEY`, and the `llm_judge` marker. Do not run it
  as part of routine verification, and do not wire it into CI or pre-commit. Keep judged
  quality checks there and deterministic assertions in `tests/`; never mix the two, or flakes
  break CI and thresholds get quietly lowered to stay green. See `evals/README.md`.

## Documentation Rules

- Keep `AGENTS.md` as active project instructions, not a refactor diary.
- Put historical plans, migration notes, or large task logs in a separate dated document when
  they are needed.
- If README/runtime docs disagree with code, verify the code first and either update the docs
  or call out the mismatch.

<!-- CODEBASE_MAPPER_GRAPH:START -->
## Codebase Mapper Graph
- Use `/.cli-proxy/.codebase_map/INDEX.md` as the entrypoint for project instructions.
- Load only relevant files under `/.cli-proxy/.codebase_map/nodes/*.md`.
- If code changes affect an area, update `Last reviewed` in the relevant node.
- If update fails, run targeted repair (`update-node`/`repair`).
- Graph root: `/srv/git_projects/chatgpt-telegram-bot/.cli-proxy/.codebase_map`
<!-- CODEBASE_MAPPER_GRAPH:END -->
