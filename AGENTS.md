# Refactor Plan Log

Date: 2026-02-05
Scope: Full plugin system refactor, tests included. Breaking changes allowed.

## Plan (strict order)
1. Baseline audit: inventory plugin IDs/functions/commands/handlers, identify collisions and mismatches; document current plugin registry behavior and model/tool formats.
2. Define new plugin contract and registry rules: explicit plugin_id, function namespace, unique function names, lifecycle hooks, storage locations; update docs.
3. Refactor PluginManager + plugin interface to enforce IDs, validate config/chat modes, resolve tools format per model, support multiple tool calls; remove singleton; add safe handler registration.
4. Refactor plugins to conform (rename functions, update specs, update chat_modes.yml), fix ConversationAnalytics usage, fix ddg/deepl translate collision, normalize plugin names in code.
5. Testing: add unit tests for plugin registry, collisions, chat_modes validation, tool call routing, multi-tool handling; update existing tests; run test suite.
6. Stabilize and document: update README/SESSIONS if needed, add migration notes for breaking changes.

## Status
- Step 1: completed
- Step 2: completed
- Step 3: completed
- Step 4: completed
- Step 5: completed
- Step 6: completed

## Audit Findings (Step 1)
- Duplicate function names in plugins: `translate` in `ddg_translate.py` and `deepl.py`.
- `chat_modes.yml` references missing tools: `deepl_translate`, `github_analyses` (typo), and omits many existing plugins.
- `ConversationAnalytics` referenced by name in `telegram_bot.py`, but plugin key is `conversation_analytics`.
- Plugin registry uses a singleton and clears/reloads on init; path defaults differ between `__new__` and `__init__`.
- Tool format differs for Google models; lists are inconsistent between `plugin_manager.py` and `openai_helper.py`.
- `post_init` uses `eval` for message handler filters.

## Status Updates
- Step 1 completed.
- Step 2 in progress.

## Contract Summary (Step 2)
- Plugins have `plugin_id` (default: file name) and `function_prefix` (default: plugin_id).
- Function specs are namespaced as `<function_prefix>.<name>` unless already namespaced.
- Plugins can implement `initialize(openai, bot, storage_root)` and `close()`.
- Storage root defaults to `bot/config` or `PLUGIN_STORAGE_ROOT`.
- Tool calls are executed in parallel and results aggregated; any `direct_result` short-circuits.
- Safe handler registration: remove `eval`, allow filter objects or `filters.X` strings only.

## Status Updates
- Step 2 completed.
- Step 3 in progress.
- Step 3 completed.
- Step 4 in progress.
- Step 4 completed.
- Step 5 completed (pytest: 2 passed, 2 skipped).
- Step 6 in progress.
- Step 6 completed.

# DB Refactor Plan Log
Date: 2026-02-05
Scope: Fix DB layer issues with tests.

## Plan (strict order)
1. Audit DB access patterns and concurrency usage in codebase; list critical paths and expected invariants.
2. Design DB fixes: connection handling, WAL/busy_timeout, foreign_keys, session counting logic, message_count correctness, session creation flow; document decisions.
3. Implement DB fixes and migrations (if needed) with minimal behavior changes; add safety guards.
4. Add unit tests for DB invariants: message_count, session creation/deletion limits, concurrency safety (smoke), and migration path.
5. Run tests and update AGENTS.md with DB refactor status and notes.

## Status
- Step 1: in progress
- Step 2: pending
- Step 3: pending
- Step 4: pending
- Step 5: pending

## DB Audit Findings (Step 1)
- SQLite connection is per-thread but used concurrently by async tasks; no async lock, potential re-entrancy issues.
- `save_conversation_context` calculates `message_count` but inserts `1` on new rows.
- `create_session` ignores `max_sessions` argument; `delete_oldest_session` uses env var string.
- Long-running call to `openai_helper.ask_sync` happens inside DB transaction.
- Foreign keys not enabled; no WAL/busy_timeout settings.
- Migration uses `context = '[]'` which mismatches `{messages: []}` expected format.

## DB Status Updates
- Step 1 completed.
- Step 2 in progress.

## DB Design Decisions (Step 2)
- Use `threading.RLock` to serialize DB operations in async contexts.
- Enable `PRAGMA foreign_keys=ON`, `journal_mode=WAL`, and `busy_timeout` for stability.
- Fix `message_count` persistence on insert.
- Respect `max_sessions` parameter and enforce numeric conversion.
- Move any long-running operations out of DB transaction boundaries.
- Migration uses `{"messages": []}` context default.

## DB Status Updates
- Step 2 completed.
- Step 3 in progress.

## DB Status Updates
- Step 3 completed.
- Step 4 completed.
- Step 5 completed (pytest: 6 passed, 2 skipped).

# Code Health Refactor Plan Log
Date: 2026-02-05
Scope: Validation, caching, strict typing; OpenAIHelper modularization; chat_modes tool startup errors. Public method names preserved.

## Plan (strict order)
1. Audit OpenAIHelper responsibilities and identify extraction targets (config validation, tool handling, history, chat mode loading/cache).
2. Design module boundaries and interfaces; keep OpenAIHelper public methods as thin facades.
3. Implement: add chat_modes cache/validator with strict logging for missing tools; move tool-call logic and history management into helper modules; add typed dataclasses for configs.
4. Add validation utilities (JSON schema for plugin args, config typing), and update plugin manager to use them; add startup log errors for invalid tools per chat mode.
5. Add tests for chat_modes validation, tool error logging, and OpenAIHelper facade behavior; run test suite.
6. Update AGENTS.md with plan/status and any migration notes.

## Status
- Step 1: in progress
- Step 2: pending
- Step 3: pending
- Step 4: pending
- Step 5: pending
- Step 6: pending

# Code Health Refactor Status Updates
- Step 1 completed.
- Step 2 completed.
- Step 3 completed.
- Step 4 completed.
- Step 5 completed (pytest: 8 passed, 2 skipped).
- Step 6 completed.
