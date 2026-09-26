# T05 Review — Отправка файлов 1.5 (`bot/artifact_paths.py` + 3 точки проверки)

## Раунд 1

Reviewed files (per task ownership): `bot/artifact_paths.py` (new, untracked, read fully — 131
lines), `bot/plugins/agent_tools.py` (delivery parts: `_deliver_to_user`,
`_normalize_delivery_artifacts`, removed `_allowed_artifact_roots`), `bot/utils.py`
(`handle_direct_result`, `cleanup_intermediate_files`, new `_artifact_scope_for_update`),
`bot/agent_delivery.py` (`_send_direct_payload`), `tests/test_artifact_paths.py` (new, read
fully — 352 lines), and the delivery-related diffs in `tests/test_agent_tools_plugin.py`,
`tests/test_plugin_direct_results.py`, `tests/test_agent_delivery.py`. T02's unrelated
`bot/utils.py` hunks (`.strip()` on `allowed_user_ids`, the group-access
`allow_group_members_via_authorized_user` gate) were identified via diff and ignored per
instructions.

Verification performed:
- Full diff read against `HEAD` for all owned tracked files; full read of both new untracked
  files.
- Cross-checked every default-resolution helper in `artifact_paths.py`
  (`_default_storage_root`/`_default_db_path`/`_default_skills_dir`/
  `_default_skills_workdir_root`) against the live code it mirrors: `bot/plugin_manager.py:66-68`
  (storage_root), `bot/database.py:220-224` (`Database.__new__`'s `db_path` resolution —
  confirmed the documented gap that `cls._configured['db_path']` is checked before `os.getenv`,
  so a test calling `Database.configure(db_path=...)` without setting the env var would diverge;
  this is explicitly disclosed and scoped out in `T05-plan.md` §0.5, not a T05 defect),
  `bot/plugins/skills.py:118-129` (`skills_dir`/`workdir_root`).
- Verified the five documented legitimate-flow producers by reading their actual write code:
  `bot/plugins/auto_tts.py:36`, `bot/plugins/show_me_diagrams.py:200`,
  `bot/llm_gateway_client.py:234` (`stable_diffusion.py`'s path), `bot/plugins/webshot.py:54-57`,
  `bot/plugins/terminal.py:18` (`DEFAULT_CWD`) — all match the plan's claims about where they
  write (bare `tempfile`-top-level, `tempfile.gettempdir()`+uuid, `tempfile.mkdtemp(prefix=...)`,
  cwd-relative `uploads/webshot`, `/tmp`) and all fall into an allow rule in `is_deliverable`.
  Verified `codeinterpreter.py` uses `runtime_output_dir()`/`runtime_plots_dir()`
  (`bot/runtime_paths.py`), matching `artifact_paths.py`'s import of the same helpers.
  Verified skills script execution's workdir shape (`bot/plugins/skills.py`'s
  `_ensure_skill_workdir`: `workdir_root / skill_id / safe_scope`) matches the
  `parts[1] == safe_scope` check in `is_deliverable`'s skills-workdir branch.
- Grepped the whole tree for other storage_root-rooted per-plugin files
  (`agent_cron_jobs.json`, `agent_pending_questions.json`, `agent_background_jobs.json`,
  `language_progress.json`, `mcp_servers.json`, `reminders.json`, `tasks.json`,
  `anythingllm_workspaces.json`) — confirmed all sit directly at `storage_root` root (not
  nested), so the new "bare json/jsonl in storage root" deny rule actually protects real
  multi-user plugin state files, not a hypothetical.
- Grepped for other callers of the removed `_allowed_artifact_roots` / the renamed
  `_normalize_delivery_artifacts` — none outside `agent_tools.py`, no dead/orphaned references.
- Ran the task's test scope:
  `~/.venvs/ctb/bin/python -m pytest tests/test_artifact_paths.py tests/test_agent_tools_plugin.py tests/test_plugin_direct_results.py tests/test_agent_delivery.py -q`
  → **138 passed, 0 failed**.
- `ruff check` on `bot/artifact_paths.py bot/plugins/agent_tools.py bot/utils.py
  bot/agent_delivery.py tests/test_artifact_paths.py tests/test_agent_tools_plugin.py
  tests/test_plugin_direct_results.py tests/test_agent_delivery.py` → clean.
- `mypy` on the four owned source files vs `/tmp/impl/mypy_before.keep`, diffed by
  `(file, line, message)` and again by `(file, message)` ignoring line-number drift from added
  lines: every apparent "new" line is an unchanged pre-existing error shifted down by the new
  code's line count, **except** 3 genuinely new occurrences of
  `bot/utils.py: error: "MaybeInaccessibleMessage" has no attribute "reply_text" [attr-defined]`
  — one per new rejection branch (photo/gif/file). Confirmed exactly matches the pre-existing
  error class already present 14 times elsewhere in the same file for the same reason
  (`message: MaybeInaccessibleMessage | None` narrowing), not a new category of problem.

---

### Security checklist (per review brief)

- **DB/.env/plugin-storage-json/skills-source exfiltration** — blocked. Verified with real
  per-file tests (`test_is_deliverable_rejects_db_path_and_wal_shm_journal`,
  `..._rejects_env_file`, `..._rejects_usage_logs`, `..._rejects_bare_json_in_storage_root_root`,
  `..._rejects_skills_source_dir`) and independently confirmed the 8 real per-plugin storage
  JSON files above all match the "bare file at storage_root root" deny shape.
- **Symlink / `..` traversal** — `is_deliverable`/`is_protected_path` both call
  `os.path.realpath(os.path.expanduser(path))` before any comparison, so a symlink pointing at
  the DB is deny-listed by its resolved target, not its literal argument path (verified by
  `test_is_deliverable_rejects_db_path_via_symlink`, passing). Deny is also checked *before* the
  temp-fallback allow, so a DB path that happens to live inside `tempfile.gettempdir()` is still
  denied (`test_is_deliverable_deny_wins_over_tempdir_fallback`, passing).
- **Scope correctness (per chat/user)** — the two namespaced subtrees
  (`storage_root/artifacts/<safe_scope>`, `storage_root/skill_workdir/<skill_id>/<safe_scope>`)
  are scope-checked, verified allow/deny for matching/mismatched scope in both directions. All
  three integration points compute the scope the same way: `_deliver_to_user` via
  `compute_scope_key(chat_id, user_id)`, `_artifact_scope_for_update` via the same function with
  the same priority, `_send_direct_payload` via `compute_scope_key(chat_id)` — since
  `compute_scope_key` always prefers `chat_id` when present (`bot/utils.py:1038-1058`), all three
  agree on `"chat:{id}"` for the common case; no scope-mismatch between the three call sites.
- **All 8 legitimate flows deliverable** — verified against the plan's table by reading each
  producer's actual write path (see verification list above); all pass through an allow rule.
  `haiper_image_to_video.py` remains out of scope (uses `reply_video` directly, bypasses all
  three integration points) — correctly not touched, matches plan §0.8/§7.4.
- **Rejection UX** — every rejection path sends
  `"Artifact path is unavailable: {basename}"` via `reply_text`/`send_text_chunks` and returns
  without raising; server-side `logging.warning`/`logger.warning` logs the full path + reason.
  No behavior change for already-passing paths (existing failure-path tests for missing/corrupt
  files at `tmp_path` locations — which are under `tempfile.gettempdir()` in this environment —
  still pass unchanged, confirmed by the 138/138 run).
- **Cleanup never deletes protected files** — `cleanup_intermediate_files` now short-circuits via
  `is_protected_path(value)` before `os.remove`, and does so both for the direct call in
  `handle_direct_result` and for the "final"-kind recursive per-artifact cleanup path (same
  function, recurses into itself). Verified with
  `test_cleanup_intermediate_files_skips_protected_path` (passing) and by reading the
  `agent_delivery.py:237-239` cleanup call after a successful send (only reached for paths that
  already passed `is_deliverable`, so no double-jeopardy false rejection).

---

### WARNING

None that qualify — see NIT items below for the two findings that are real but already
explicitly disclosed/accepted in `T05-plan.md` and out of this task's fix scope.

### NIT

**N1 — +3 pre-existing-class mypy `attr-defined` notes on `bot/utils.py`, developer-disclosed**

- New `message.reply_text(**common_args, text=..., parse_mode=None)` calls in the three
  rejection branches (`bot/utils.py:78-82`, `:110-114`, `:127-131` in the diff) each trip
  `"MaybeInaccessibleMessage" has no attribute "reply_text"`, the same pre-existing error class
  already present 14× elsewhere in the file for the same `message: MaybeInaccessibleMessage |
  None` narrowing gap. Confirmed exact count (+3, no other new pairs) via a `(file, message)`
  diff against `/tmp/impl/mypy_before.keep`. Matches the developer's own disclosure; per this
  review's instructions, accepted — mypy cleanup is T13's job, not T05's.

**N2 — Wide "inside storage_root" allow rule still has no owner-scope check outside the two
namespaced subtrees (pre-existing behavior, not a T05 regression, already disclosed)**

- `is_deliverable`'s fallback `if _is_relative_to(resolved, effective_storage_root): return True,
  None` (`bot/artifact_paths.py:116-117`) allows any file directly under `storage_root` from any
  scope, as long as it isn't a bare root-level `.json`/`.jsonl` file. Several real plugins write
  nested, non-scope-namespaced directories straight under `storage_root`
  (`conversation_analytics/`, `language_data/`, `document_metadata/`, `temp_pdfs/`,
  `pdf_cache/` — confirmed via source read of the corresponding plugins), which would be
  deliverable across scopes if a model knew/guessed an exact path there.
- This is unchanged from `HEAD`: the removed `_allowed_artifact_roots` already granted the
  *entire* `storage_root` unconditionally with zero scope check
  (`bot/plugins/agent_tools.py` diff, old code at `roots.append(Path(storage_root))`), so T05 is
  a net narrowing (adds scope checks for 2 of N subtrees), not a new hole. `T05-plan.md` §7.7
  explicitly documents this exact gap as accepted and deferred. Flagging only for visibility
  since "plugin storage of other users" was an explicit item in this round's review brief —
  no action expected within T05.

---

### Summary

- ERROR: 0
- WARNING: 0
- NIT: 2 (N1 — 3 pre-existing-class mypy notes, developer-disclosed, deferred to T13; N2 — no
  owner-scope check for files directly under `storage_root` outside the two namespaced
  subtrees, pre-existing/unchanged from `HEAD`, already accepted in `T05-plan.md` §7.7)

Implementation matches `T05-plan.md` closely: API shape, deny-then-allow ordering, all three
integration points, `cleanup_intermediate_files` guard, and every test listed in plan §5 are
present with matching names/assertions. 138/138 targeted tests pass, ruff clean, no new mypy
error categories beyond the developer-disclosed one.
