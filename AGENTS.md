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
  either is missing (`bot/__main__.py:184-188`).
- Startup creates `PluginManager`, `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`
  (`bot/__main__.py:375-394`).
- After the env-var check but before `PluginManager`/`Database`/`OpenAIHelper` are created, `main()` acquires a single-instance file lock
  (`bot/instance_lock.py`), defaulting to a path next to `DB_PATH` and overridable via
  `INSTANCE_LOCK_PATH`; a second process against the same lock file logs ERROR and exits
  non-zero (`bot/__main__.py:190-197`).
- Telegram polling is owned by `ChatGPTTelegramBot.run()` in `bot/telegram_bot.py`; the current
  builder enables concurrent updates, local Telegram Bot API mode, and
  `http://localhost:8081/bot` as base URL (`bot/telegram_bot.py:6326-6339`).
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
  (`bot/plugins/plugin.py:111-126`).
- Stable plugin identity is `plugin_id`; tool namespace is `function_prefix`, defaulting to
  `plugin_id` (`bot/plugins/plugin.py:12`, `bot/plugins/plugin.py:20`).
- `PluginManager` loads plugin modules from `bot/plugins/*.py`, excluding the files listed in
  `NON_PLUGIN_MODULES` (`bot/plugin_manager.py:34`) — `__init__.py`, the base class plus the
  framework modules `background.py`, `db_handle.py`, `hooks.py`; empty/unset `PLUGINS` loads all
  plugins, and non-empty `PLUGINS` acts as a comma-separated allow-list
  (`bot/plugin_manager.py:65`, `bot/plugin_manager.py:271`).
- The loader runs a plugin through `exec_module` without registering it in `sys.modules`, so a
  plugin module must not combine `from __future__ import annotations` with `@dataclass`:
  `dataclasses` resolves the resulting string annotations through
  `sys.modules[cls.__module__].__dict__` and the plugin silently fails to load with
  `'NoneType' object has no attribute '__dict__'`. Guarded by
  `tests/test_plugin_manager.py::test_plugin_modules_do_not_combine_future_annotations_with_dataclass`.
- Function specs must be unique after namespacing. Unqualified spec names are normalized to
  `<function_prefix>.<name>` (`bot/plugin_manager.py:813-823`).
- Duplicate function names are invalid. With `PLUGIN_STRICT_VALIDATION=true`, duplicates raise;
  otherwise they are logged and skipped (`bot/plugin_manager.py:372-379`).
- Tool arguments are JSON-decoded and validated against the function spec before plugin
  execution (`bot/plugin_manager.py:532`, `bot/plugin_manager.py:562`, `bot/validation.py:33`).
- Tool calls may arrive in batches and are executed with `asyncio.gather`; `chat_id` and
  `user_id` are injected into arguments before execution (`bot/openai_tool_handler.py:219`,
  `bot/openai_tool_handler.py:1438-1444`).
- A plugin response marked as a direct result short-circuits model re-entry
  (`bot/openai_tool_handler.py:1566-1572`, `bot/openai_tool_handler.py:1690-1693`).
- Every model reaches this gateway as an OpenAI-compatible alias, so `_format_specs_for_model()`
  (`bot/plugin_manager.py:388`) unconditionally wraps every spec as
  `{"type": "function", "function": {...}}` — there is no Google-specific branch or
  `function_declarations` envelope in the current code (`bot/model_constants.py` keeps only
  individual model-name constants after T16 removed the model-family tuples).
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
   - `agent_tools.on_before_chat_request` (`bot/plugins/agent_tools.py:347`) — injects the
     planning-prefix system message that reminds the model to call `manage_plan_tasks`
     before non-trivial work.
   - `hindsight_memory.on_before_chat_request` (`bot/plugins/hindsight_memory.py:2447`) —
     injects a recalled long-term-memory system message when auto-recall is enabled.
4. **Collectors** (`collect_fragments` / `collect_objects`): named slots, called
   **sequentially**. Active slots in tree: `auto_mode_priority` (auto-mode prompt prefix,
   `bot/openai_helper.py:4217-4219`), `stats_block` (`/stats` extra blocks,
   `bot/telegram_bot.py:1417-1421`), `settings_menu_buttons` (extra settings-menu button rows,
   `bot/telegram_bot.py:1720-1724` — only consumer of `collect_objects`). Each plugin's
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

Examples in tree: `bot/plugins/hindsight_memory.py:1191-1209` (`hindsight_finalize_jobs` DDL,
`register_schema()` def at `:1189`), `bot/plugins/agent_tools.py` (`agent_plan_contracts` /
`agent_plan_tasks`). The `kind` column on `hindsight_finalize_jobs` distinguishes
`session_close` from `burst` jobs (see Background tasks); long-term-memory consolidation
(`_consolidate_dream_document`, `bot/plugins/hindsight_memory.py:1914`) merges a new summary
into an existing document via a bounded ADD/DELETE action protocol
(`parse_consolidation_actions` / `apply_consolidation_actions`,
`bot/plugins/hindsight_memory.py:237`, `:263`), not a schema change. Plugins that own
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

Hindsight also registers a `burst_sweep` task (`bot/plugins/hindsight_memory.py:833` for
`get_background_tasks()`, task entry at `:843-853`) that
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
  (`bot/openai_helper.py:290-291`).
- Missing tool references in `chat_modes.yml` are logged by `validate_tools()`
  (`bot/chat_modes_registry.py:99`).
- During request preparation, the active mode can restrict allowed plugins via its `tools`
  field; absent mode tooling defaults to `['All']` (`bot/openai_helper.py:1153`,
  `bot/openai_helper.py:1184-1185`).
- When editing chat modes, keep plugin names aligned with loaded plugin module names, not
  human-readable descriptions.

## Tool And Context Footprint

Every registered tool spec lands in every prompt reachable by its allow-list; there is no
per-turn trimming beyond `chat_modes.yml`'s `tools:` list and hook self-gating (see below). The
bar for adding a new tool is high because of this. Ranked cheapest to most expensive:

1. **Bot commands.** `PluginManager.get_plugin_commands()` (`bot/plugin_manager.py:885`) and
   `build_bot_commands()` (`bot/plugin_manager.py:912`), registered in `post_init()` in
   `bot/telegram_bot.py`, never enter the model's `tools` array. Free.
2. **Skills.** `SkillsPlugin.get_spec()` (`bot/plugins/skills.py:388`) always returns the same
   fixed set of tool specs regardless of how many skills are installed. A new skill adds only
   one truncated `id: description` line: to the `auto_mode_priority` prompt fragment
   (`bot/plugins/skills.py:175` — `contribute_prompt_fragment`) and to the `skills_agent` mode
   catalog (`bot/plugins/skills.py:219` — `on_before_chat_request`). The cheapest way to add a
   capability the model does not need to be able to call by name on every turn.
3. **Chat modes** (`bot/chat_modes.yml`) are free by themselves: a mode is a system prompt plus
   a `tools:` allow-list, read at `bot/openai_helper.py:1184-1185`, defaulting to `['All']`
   (`bot/openai_helper.py:1153`). Narrowing `tools:` is the main lever for cutting per-call
   payload; most modes in tree list a short explicit set of plugins (commonly 7-10) instead of
   `All`.
4. **New plugin/tool** — one full JSON schema on every request where `allowed_plugins` includes
   it. With an empty `PLUGINS` env var, any `.py` dropped into `bot/plugins/` loads
   automatically and becomes available to every mode with `tools: [All]` without touching
   `chat_modes.yml`.
5. **Hooks** — the most expensive rung. `_active_plugin_instances()`
   (`bot/plugin_manager.py:1013`) filters ONLY by the per-user disabled-plugin set, NOT by the
   active mode's `tools:` allow-list. An `on_before_chat_request` mutator therefore runs on
   every request in every mode unless it checks itself. In tree, both
   `agent_tools.on_before_chat_request` (`bot/plugins/agent_tools.py:347`) and
   `skills.on_before_chat_request` (`bot/plugins/skills.py:219`) perform that self-check. A new
   hook without one runs unconditionally.
6. **MCP servers** — the least controllable rung: `register_mcp_server` is itself a
   model-callable tool (`bot/plugins/mcp_server.py:225` for `get_spec()`, tool name at `:235`),
   and every tool of a connected remote server becomes a full spec at runtime
   (`bot/plugins/mcp_server.py:328-335`) with no code review.

Practical rule: prefer a skill over a new tool when the capability is rarely-needed procedural
knowledge rather than something the model must call by name on every turn. When adding a hook,
decide explicitly whether it should be scoped to a chat mode, and if so, self-gate it — the
framework will not do that for you.

## Deterministic Routing In Agent Plugins

Code decides what happens after a tool call, not the model; the model only reports state. This
is already the pattern in `agent_tools` — new agent-style plugins should follow the same shape,
not treat this as a refactor mandate:

- `_manage_plan_tasks` (`bot/plugins/agent_tools.py:2554`): the model only writes a task's
  status; the code detects the transition and decides the consequence —
  `status=blocked` schedules a re-plan, `status=completed` schedules a verify step
  (`bot/plugins/agent_tools.py:2606-2609` for `action=add`, `:2664-2667` for `action=update`),
  applied via `_apply_plan_runtime_effects` (`bot/plugins/agent_tools.py:2087`).
- `_record_tool_outcome` (`bot/plugins/agent_tools.py:2110`) counts consecutive tool failures
  on the same task and schedules a re-plan itself once a threshold is reached.
- `_reentry_tool_choice(...)` (`bot/openai_tool_handler.py:947`) is a pure function of the
  round counter that picks `"auto"`/`"none"`; once `functions_max_consecutive_calls` is
  exhausted the code forcibly narrows the tool set to the delivery tool
  (`bot/openai_tool_handler.py:1712-1713`). `_reentry_tool_choice`
  (`bot/openai_tool_handler.py:947`) forces `"none"` once `times >= max_consecutive_calls +
  DELIVERY_GRACE_ROUNDS`, so the mandatory-delivery path (`final_delivery_required`) is bounded
  the same way as the ordinary tool-call path.

`describe_plan_lifecycle()` (`bot/plugins/agent_tools.py:41`) is the single source of truth for
the task-plan lifecycle: the full status set, its terminal/open subsets, the cross-task
invariants actually enforced by `_validate_plan_tasks` (`bot/plugins/agent_tools.py:2356`), and
the status → side-effect mapping planned by the code above. It reads `TASK_STATUSES`/
`CLOSED_STATUSES` rather than holding its own copy, so it cannot drift from what the code
validates. It is not part of `get_spec()` and costs no prompt tokens — it exists for tests and
documentation.

The task-status set is currently duplicated by hand: the `TASK_STATUSES` constant
(`bot/plugins/agent_tools.py:29`) and the `enum` literal in `manage_plan_tasks`'s JSON schema
(`bot/plugins/agent_tools.py:510`) both list the same five values, with nothing enforcing they
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
  collapses it with `aggregate_usage_split` (`bot/chat_response_utils.py:74`), which returns
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
(`bot/openai_helper.py:3727`) can catch it and fall back to a deterministic trim.

## Conversation History Compaction

When history needs to shrink, `OpenAIHelper._summarize_and_trim()`
(`bot/openai_helper.py:3727`) tries an LLM summary via `ModelUtilities.summarize_window`; if it
returns `False` (throttled, unresolvable cut, or the summary call itself failing/timing out),
`_fallback_trim_with_summary()` (`bot/openai_helper.py:3809`) head-preserve-trims the window
instead, replacing the cut portion with a deterministic (no model call) excerpt from
`_deterministic_summary_text()` (`bot/openai_helper.py:3676`) — a bounded head+tail rendering —
so history is compacted, never silently dropped.

## Terminal Command Policy

`bot/command_policy.py` backs the terminal plugin's guard: `evaluate_command()`
(`bot/command_policy.py:467`) normalizes a command string (unquoting, `$(...)`/backtick
expansion, heredoc stripping) and matches it against `CommandRule` patterns — built-in
`DEFAULT_RULES` plus any layered from `TERMINAL_COMMAND_POLICY` (JSON, via
`load_policy_from_env()` at `bot/command_policy.py:443`) — to a `CommandDecision` of
`allow`/`deny`/`require_approval`; `TERMINAL_APPROVAL_MODE` governs how the terminal plugin
acts on `require_approval`. It is a heuristic over command text, not a sandbox boundary:
bypassable by obfuscation or by writing a script to a file and then executing it
(`bot/command_policy.py:4-7`).

## Telegram Handler Rules

- Plugin commands are normalized through `PluginManager.get_plugin_commands()` and registered
  in `post_init()` as command handlers or callback handlers (`bot/plugin_manager.py:885`,
  `bot/telegram_bot.py:5390-5409`).
- Plugin command names must not include spaces; a leading `/` is stripped during normalization
  (`bot/plugin_manager.py:948-950`).
- Plugin message handlers can provide a ready handler object or a `filters.X` string/object.
  Invalid filters are logged and skipped (`bot/telegram_bot.py:5281-5285`).
- Do not reintroduce `eval` for handler filters.

## Database Rules

- `Database` is a singleton with thread-local SQLite connections and an operation `RLock`
  (`bot/database.py:251` for `__new__`, `bot/database.py:262` for `_op_lock`; class starts at
  `:207`, thread-local storage at `:265`).
- New SQLite connections enable foreign keys, WAL by default, and `busy_timeout`
  (`bot/database.py:339` foreign keys, `:340-344` journal mode/WAL, `:345-346` busy_timeout).
- Async DB access (the `Database.*_async` methods and the `DbHandle` facade) routes through a
  single dedicated worker thread — `Database._run_in_db_thread` over a `max_workers=1`
  `ThreadPoolExecutor` — instead of bare `asyncio.to_thread`. This bounds thread-local
  connections to one worker; the executor is torn down by `Database.shutdown()` (called from
  `_reset_singleton` and `__del__`). The operation `RLock` is unchanged.
- `conversation_context.context` is JSON shaped as `{"messages": [...]}`; do not migrate or
  seed it as a bare list (`bot/database.py:730`).
- `conversation_context` carries a monotonic `version` column (schema `TARGET_SCHEMA_VERSION`
  = 2); each `save_conversation_context()` UPDATE bumps it by one. Writes are serialized by
  the per-operation write-lock (`transaction()` = `BEGIN IMMEDIATE`), so `version` is a
  revision counter, not a rejecting CAS gate. Migration adds the column idempotently across
  fresh-install, v1→v2, and legacy (no-`session_id`) paths.
- `save_conversation_context()` persists `message_count` as the number of user-role messages
  (`bot/database.py:1078`, def at `:1064`).
- Session creation prunes old sessions inline via `_oldest_session_ids_for_limit()`;
  there is no separate oldest-session deletion API.
- Keep long-running OpenAI calls outside active DB transactions. Async session-name
  generation is handled by `OpenAIHelper._ensure_session_name_with_llm()` after the DB write
  (`bot/openai_helper.py:713`); `Database.ensure_session_name_async()` at
  `bot/database.py:1196` provides a short fallback but no longer calls the LLM directly.

## Helper Session API

`bot/telegram_bot.py` and plugins must not touch `OpenAIHelper`'s private per-chat state
(`conversations`, `loaded_conversation_sessions`, `_chat_states`, `_clear_chat_state`,
`_with_chat_state`). Use the public API instead (`bot/openai_helper.py:3082-3158`):

- `history_snapshot(chat_id)` — the warm history cache for that chat, or `None` when the cache
  is cold (the caller then reads the history from the DB and hands it to `load_session`). The
  returned list is a shallow copy, but the message dicts inside it are shared with the cache —
  write through `load_session`, never by mutating a message in place.
- `load_session(chat_id, session_id, messages)` — refills the cache from DB messages and
  records which session is loaded; image payloads are stripped exactly as everywhere else that
  populates the cache. Returns the stripped list.
- `async replace_system_message(chat_id, content, *, mode_key=None, ...)` — inserts/replaces
  the leading system message and persists the result. Requires `load_session()` to have run in
  the same turn (the session id is read from the loaded-session map, not passed in).
- `evict(chat_id)` — drops all per-chat state (public wrapper for `_clear_chat_state`).
- `chat_state_scope(state_key)` — context manager that temporarily overrides the effective chat
  key, for parallel processing of deferred messages in a new session.

`tests/test_no_private_helper_access.py` is an AST guard (same shape as
`tests/test_no_hardcoded_plugin_refs.py`) that fails when new private-field access appears
outside `bot/openai_helper.py`.

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
  - helper session API boundary: `tests/test_no_private_helper_access.py`
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
