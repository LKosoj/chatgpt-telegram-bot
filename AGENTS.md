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

## Project Shape

- Runtime is a Python Telegram bot. The process entrypoint is `bot/__main__.py`.
- Required runtime env vars are `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`; startup exits when
  either is missing (`bot/__main__.py:22`).
- Startup creates `PluginManager`, `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`
  (`bot/__main__.py:121`).
- Telegram polling is owned by `ChatGPTTelegramBot.run()` in `bot/telegram_bot.py`; the current
  builder enables concurrent updates, local Telegram Bot API mode, and
  `http://localhost:8081/bot` as base URL (`bot/telegram_bot.py:2878`).
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
  (`bot/plugins/plugin.py:43`).
- Stable plugin identity is `plugin_id`; tool namespace is `function_prefix`, defaulting to
  `plugin_id` (`bot/plugins/plugin.py:11`, `bot/plugins/plugin.py:18`).
- `PluginManager` loads plugin modules from `bot/plugins/*.py`, excluding `__init__.py` and
  `plugin.py`; empty/unset `PLUGINS` loads all plugins, and non-empty `PLUGINS` acts as a
  comma-separated allow-list (`bot/plugin_manager.py:140`, `bot/plugin_manager.py:152`).
- Function specs must be unique after namespacing. Unqualified spec names are normalized to
  `<function_prefix>.<name>` (`bot/plugin_manager.py:279`).
- Duplicate function names are invalid. With `PLUGIN_STRICT_VALIDATION=true`, duplicates raise;
  otherwise they are logged and skipped (`bot/plugin_manager.py:150`).
- Tool arguments are JSON-decoded and validated against the function spec before plugin
  execution (`bot/plugin_manager.py:171`, `bot/validation.py:28`).
- Tool calls may arrive in batches and are executed with `asyncio.gather`; `chat_id` and
  `user_id` are injected into arguments before execution (`bot/openai_tool_handler.py:111`,
  `bot/openai_tool_handler.py:120`).
- A plugin response marked as a direct result short-circuits model re-entry
  (`bot/openai_tool_handler.py:130`, `bot/openai_tool_handler.py:144`).
- Google model tool specs use `{"function_declarations": specs}` while other models receive
  OpenAI-style `{"type": "function", "function": spec}` entries
  (`bot/plugin_manager.py:159`).
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
4. **Collectors** (`collect_fragments` / `collect_objects`): named slots, called
   **sequentially**. Active slots in tree: `auto_mode_priority` (auto-mode prompt prefix,
   `bot/openai_helper.py:2379`), `stats_block` (`/stats` extra blocks,
   `bot/telegram_bot.py:842`), `settings_menu_buttons` (extra settings-menu button rows,
   `bot/telegram_bot.py:1125` — only consumer of `collect_objects`). Each plugin's
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

Examples in tree: `bot/plugins/hindsight_memory.py:366-386` (`hindsight_finalize_jobs`),
`bot/plugins/agent_tools.py` (`agent_plan_contracts` / `agent_plan_tasks`). Plugins that own
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

## Chat Modes

- Chat modes are defined in `bot/chat_modes.yml` and loaded through `ChatModesRegistry`.
- `OpenAIHelper` constructs the registry and validates mode tool references during init
  (`bot/openai_helper.py:137`).
- Missing tool references in `chat_modes.yml` are logged by `validate_tools()`
  (`bot/chat_modes_registry.py:62`).
- During request preparation, the active mode can restrict allowed plugins via its `tools`
  field; absent mode tooling defaults to `['All']` (`bot/openai_helper.py:520`).
- When editing chat modes, keep plugin names aligned with loaded plugin module names, not
  human-readable descriptions.

## Telegram Handler Rules

- Plugin commands are normalized through `PluginManager.get_plugin_commands()` and registered
  in `post_init()` as command handlers or callback handlers (`bot/plugin_manager.py:395`,
  `bot/telegram_bot.py:2003`).
- Plugin command names must not include spaces; a leading `/` is stripped during normalization
  (`bot/plugin_manager.py:450`).
- Plugin message handlers can provide a ready handler object or a `filters.X` string/object.
  Invalid filters are logged and skipped (`bot/telegram_bot.py:2035`,
  `bot/telegram_bot.py:2911`).
- Do not reintroduce `eval` for handler filters.

## Database Rules

- `Database` is a singleton with thread-local SQLite connections and an operation `RLock`
  (`bot/database.py:18`, `bot/database.py:31`).
- New SQLite connections enable foreign keys, WAL by default, and `busy_timeout`
  (`bot/database.py:46`).
- `conversation_context.context` is JSON shaped as `{"messages": [...]}`; do not migrate or
  seed it as a bare list (`bot/database.py:570`).
- `save_conversation_context()` persists `message_count` as the number of user-role messages
  (`bot/database.py:198`, `bot/database.py:211`).
- Session creation enforces `max_sessions` through `delete_oldest_session()`
  (`bot/database.py:528`, `bot/database.py:547`).
- Keep long-running OpenAI calls outside active DB transactions. Async session-name
  generation saves DB state first, then calls `openai_helper.generate_session_name()`
  outside the DB write (`bot/database.py:494`, `bot/database.py:549`).

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
- For narrow documentation-only edits, inspect the rendered Markdown or run no tests and state
  that no runtime tests were needed.

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
