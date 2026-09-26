# T04 Review — agent_cron & reminders → SQLite

## Раунд 1

Reviewed files (per task ownership, extended per T04-plan.md §0):
`bot/plugins/agent_cron.py`, `bot/plugins/reminders.py`, `tests/test_agent_cron_plugin.py`,
`tests/test_agent_cron_storage.py` (new), `tests/test_reminders_fixes.py`, and the
reminders-related tests in `tests/test_background_tasks.py` and
`tests/test_concurrent_tool_state.py`.

Verification performed: full diff read against `HEAD` for all owned files; `get_spec()` output
byte-diffed against `HEAD` for both plugins (no drift); targeted test run
(`tests/test_agent_cron_plugin.py tests/test_agent_cron_storage.py tests/test_reminders_fixes.py
tests/test_background_tasks.py tests/test_concurrent_tool_state.py`) — 58 passed, 0 failed; ruff
clean on all owned files; mypy run on both plugin files and compared against
`/tmp/impl/mypy_before.keep` grouped by `(file, error_code)`.

---

### ERROR

**E1 — Deleting a cron job mid-run no longer stops its completion/failure message or hook dispatch**

- File: `bot/plugins/agent_cron.py:323-383` (`_run_job`)
- The only existence check left is the claim at the top (`agent_cron.py:325-334`: claim via
  `_claim_job_by_id_sync`/`fetch_one`, then `if not job: return`). Once past that point there is
  no re-check before the side effects: `_finish_job()` (`:358`/`:375`),
  `_maybe_dispatch_autonomous_response_hook(...)` (`:359`), and
  `send_agent_response(...)`/`send_text_chunks(...)` (`:360-368`, `:376-383`) all run
  unconditionally.
- `HEAD:bot/plugins/agent_cron.py:223-226` (success branch) and `:245-248` (failure branch) had:
  ```python
  live = (self.jobs.get(scope) or {}).get(job_id)
  if live is None:
      return
  job = live
  ```
  placed right after the `await helper.get_chat_response(...)` call (success) and right after the
  `except Exception as exc:` catch (failure). This check is fully removed in the new code with no
  replacement.
- Failure scenario: a user runs `/cron remove <job_id>` while that job is mid-execution (a
  potentially long agent turn with tool-call rounds). The `DELETE FROM agent_cron_jobs ...` in
  `_handle_job_action` (`:229-233`) succeeds immediately. When `_run_job` resumes from its
  `await`, `_finish_job()`'s `UPDATE ... WHERE id = ?` silently affects 0 rows (harmless), but the
  bot still posts "Cron job `xxx` completed."/"failed: ..." into the chat and still dispatches the
  autonomous-response memory hook for a job the user just deleted. This is a genuine,
  user-visible behavior regression from `HEAD`, not called out anywhere in `T04-plan.md` as an
  intentional change, and it conflicts with the "preserve user-visible behavior" requirement in
  `00-master-plan.md`.
- Suggested fix: before dispatching the hook/response (after the `await`), re-check the row still
  exists (e.g. `SELECT 1 FROM agent_cron_jobs WHERE id = ?`), or have `_finish_job` report its
  `cursor.rowcount` and short-circuit the caller when it's 0.

---

### WARNING

**W1 — Manual `/cron run` racing the 60s checker can leave a job claimed but never executed until its lease expires**

- Files: `bot/plugins/agent_cron.py:235-238` (`_handle_job_action`, "run" branch) and
  `:250-263` (`_check_due_jobs`)
- `_handle_job_action`'s manual-run branch creates the task and registers it in
  `self._running_tasks[job_id]` synchronously (`:235-236`), *before* the task body's first
  `await`, then awaits `message.reply_text(...)` (a real network call).
- `_check_due_jobs` claims **all** due jobs unconditionally first
  (`await self.db_handle.run_sync(self._claim_due_jobs_sync, ...)`, `:254-256` — this marks
  `status='running'`, sets a fresh `locked_at` for every matched id), and only *afterward* checks
  `if job_id in self._running_tasks: continue` (`:259-260`) to decide whether to spawn a task.
- If the 60-second checker tick lands inside the window between the manual task's registration in
  `_running_tasks` and its first actual DB claim call, the following happens: (1) the automatic
  side claims the row in the DB (status='running', fresh lease) but skips spawning a task because
  `job_id` is already in `_running_tasks`; (2) the manual task's own claim
  (`_claim_job_by_id_sync`, `:304-321`) then sees `status='running'` with an unexpired lease and
  fails (`cursor.rowcount == 0` → returns `None`), so `_run_job` no-ops via `if not job: return`
  (`:333-334`). Nobody actually runs the job — it stays claimed/locked in the DB until
  `AGENT_CRON_JOB_LEASE_SECONDS` (1800s) elapses and a later tick reclaims it. The user sees
  "queued for manual run" but gets no result for up to 30 minutes.
- This is self-healing (not a permanent stuck state, no duplicate execution, no crash) and
  requires a narrow timing window, but it is a real behavior gap versus the original
  single-threaded code, where the decision to spawn and the "already in-flight" check were the
  same synchronous condition with no DB side effect happening unless a task was actually about to
  be created. Not covered by any existing or new test.
- Suggested fix: exclude ids already present in `self._running_tasks` from the SQL claim filter
  in `_claim_due_jobs_sync` (pass them as a parameterized exclusion list, or filter the
  candidate id list in Python before issuing the claiming `UPDATE`), so the automatic path never
  claims a row it isn't going to execute.

**W2 — New mypy note on `bot/plugins/reminders.py` (baseline 0 → current 1)**

- File: `bot/plugins/reminders.py:26` (`self.db_handle: Any = None` in `__init__`)
- Comparing `/tmp/impl/mypy_before.keep` against a fresh `mypy bot/plugins/agent_cron.py
  bot/plugins/reminders.py` run, grouped by `(file, error_code)`: every pair is unchanged except
  `(bot/plugins/reminders.py, annotation-unchecked)`, which went from 0 to 1. The new note is
  mypy's default "body of an untyped function is not checked" note, triggered because
  `self.reminders_file = ...` (line 25, `__init__`'s existing style) leaves `__init__` untyped and
  the new annotated assignment on line 26 is what mypy now has something to comment on.
  `bot/plugins/agent_cron.py`'s equivalent `self.db_handle: Any = None` doesn't add a *new*
  pair because that file already had 3 baseline `annotation-unchecked` notes from other untyped
  defs.
- It's a "note", not an "error" — low severity, no real type-safety regression — but per the
  review instructions ("new errors = WARNING at least") it's flagged for completeness. No fix
  required unless the project wants zero new mypy output on touched files.

---

### NIT

**N1 — `fire_at_utc` write precision vs. claim-comparison precision**

- File: `bot/plugins/reminders.py:404` (`fire_at_utc = (reminder_time - offset).isoformat()`,
  pre-existing line, unchanged by this diff) vs. the new claim comparison's `now_utc_iso`, which
  is truncated to `timespec="seconds"`.
- `fire_at_utc` is stored with full microsecond precision while `now_utc_iso` used in
  `_claim_due_reminders_sync`'s `fire_at_utc <= ?` comparison is second-truncated. This can in
  theory delay firing a reminder by up to ~1 second at an exact second boundary due to
  lexicographic string comparison. Given the 60-second poll tick, this is practically negligible.
  Not asserting a fix is needed — noted for completeness only.

---

### Summary

- ERROR: 1
- WARNING: 2
- NIT: 1

Everything else checked (claim atomicity via `BEGIN IMMEDIATE`, lease-based reclaim math,
JSON-import idempotency/rename-on-success-only, `get_spec()` byte-identity, async DB calls routed
through `db_handle`/`run_sync` off the event loop, `_advance_job` left untouched,
`initialize(..., db=None, plugin_config=None)` signature matching the established
`hindsight_memory.py`/`agent_tools.py` convention) matched the plan and showed no further issues.

---

## Раунд 2

Reviewed files (same scope as round 1, per `T04-plan.md` §0 extension): `bot/plugins/agent_cron.py`,
`bot/plugins/reminders.py`, `tests/test_agent_cron_plugin.py`, `tests/test_agent_cron_storage.py`,
`tests/test_reminders_fixes.py`, `tests/test_background_tasks.py`, `tests/test_concurrent_tool_state.py`.

Verification performed: re-read the full current diff against `HEAD` for all owned files; traced
each round-1 finding to the exact lines that changed and to its regression test; re-ran
`get_spec()` byte-diff against `HEAD` for both plugins (still identical); full targeted test run
(62 passed, up from 58 in round 1 — 4 new tests); ruff clean on all seven owned files; mypy on both
plugin files, grouped by `(file, error_code)` and diffed against `/tmp/impl/mypy_before.keep`.

### Round-1 findings — verification

- **E1 (deleted-mid-run job still posts a message/dispatches the hook) — FIXED.** The
  `live = await self.db_handle.fetch_one(...); if live is None: return` re-check is back in both
  the success and failure branches of `_run_job` (`bot/plugins/agent_cron.py:356`, `:377`), same
  place it lived at `HEAD` before it was dropped. Covered by two new regression tests that delete
  the row from inside a fake `helper.get_chat_response()` (simulating a `/cron remove` landing
  mid-run) and assert no message/hook fire:
  `tests/test_agent_cron_plugin.py::test_run_job_deleted_during_run_skips_completion_message_and_hook`
  and `::test_run_job_deleted_during_run_skips_failure_message`.
- **W1 (manual run vs. automatic checker race can strand a claimed-but-never-run job) — FIXED.**
  `_claim_due_jobs_sync` now takes `exclude_ids` and filters the candidate id list *before* issuing
  the claiming `UPDATE` (`bot/plugins/agent_cron.py:227`); `_check_due_jobs` passes
  `frozenset(self._running_tasks.keys())` (`:265-267`). Registration of a manual run into
  `_running_tasks` still happens synchronously right after `create_task`, before any `await`
  (`:235-236`), so the snapshot the checker sees is always current by the time it can run on the
  same single-threaded event loop. Covered by
  `tests/test_agent_cron_storage.py::test_claim_due_jobs_excludes_ids_already_running` (unit, calls
  `_claim_due_jobs_sync` directly with `exclude_ids`) and
  `::test_check_due_jobs_does_not_claim_job_already_in_running_tasks` (integration, through
  `_check_due_jobs`).
- **W2 (new mypy `annotation-unchecked` note on `reminders.py`) — FIXED.** `reminders.py:26` is now
  the unannotated `self.db_handle = None` (vs. `agent_cron.py`'s `self.db_handle: Any = None`,
  which doesn't add a new pair there since that file already had baseline notes). Re-ran mypy on
  both files and grouped by `(file, error_code)`: every pair matches `mypy_before.keep` exactly,
  including `('bot/plugins/reminders.py', 'annotation-unchecked')` now absent from both baseline
  and current output. No regression test needed for a type-annotation-only change; verified by
  direct mypy diff.
- **N1 (`fire_at_utc` write precision vs. second-truncated claim comparison) — FIXED.**
  `now_utc_iso` in `check_reminders` no longer passes `timespec="seconds"`
  (`bot/plugins/reminders.py:331`), with an inline comment explaining why (full-precision write vs.
  truncated comparison could delay firing by up to ~1s at a boundary). This was flagged as a NIT,
  not requiring a fix, but the fix is correct and low-risk.

### WARNING

**W3 (new) — `check_reminders`'s post-send persistence step can itself fail, and failure is then
treated as "send failed", risking a duplicate Telegram message on a later tick**

- File: `bot/plugins/reminders.py:339-364` (`check_reminders`).
- The loop does `await self.send_reminder(...)` (an external Telegram call) then
  `await self.db_handle.execute("DELETE FROM reminders WHERE id = ?", ...)` (`:344-345`) inside the
  same `try`. If the `DELETE` itself raises — e.g. `DatabaseLockTimeoutError` from
  `Database.get_connection()`'s `_op_lock.acquire(timeout=...)` under contention, or a transient
  `sqlite3.OperationalError` — the `except Exception as exc:` block (`:346`) has no way to know the
  send already succeeded. It increments `send_attempts` and leaves the row `status='pending'`,
  `locked_at=NULL` for retry (`:359-364`), so the next tick sends the same reminder again: a real
  duplicate delivery, not just a retried-but-idempotent operation. The same risk applies to the
  give-up-branch `DELETE` at `:353` (a failure there propagates out of the `for` loop entirely,
  leaving any later, still-claimed reminders in that batch `status='processing'` until their lease
  expires at `REMINDER_LEASE_SECONDS`, i.e. delayed rather than lost).
- This is not a clean regression from `HEAD`: the old code had a structurally similar risk (an
  in-memory `del self.reminders[...]` that essentially cannot fail, but the single end-of-tick
  `save_reminders()` file write could fail-and-be-swallowed internally, leaving the on-disk file
  stale and able to reproduce the same reminder on the next process restart). The new code makes
  the failure surface per-reminder and more plausible (a real DB call inside the hot path instead
  of a single file write for the whole tick), and the master plan's invariant #9 ("напоминание
  отправляется ровно один раз... успешная отправка = удаление записи под локом") is written as if
  the delete-after-send were guaranteed, which it is not under this failure mode.
- No test exercises "send succeeds, then the persistence call raises" — `test_reminder_sent_exactly_once_under_claim`
  only checks the atomic-claim path (two full, successful `check_reminders()` calls), not a
  mid-flight DB failure after a successful send.
- Suggested fix (not required immediately given the narrow window and pre-existing risk shape, but
  worth a decision): either accept this as a documented at-least-once edge case (update the
  invariant wording), or make the delete idempotent from the caller's perspective, e.g. only treat
  `send_reminder` failures (not post-send persistence failures) as retryable by narrowing the
  `try` block around just the `send_reminder` call and handling the follow-up `DELETE`/`UPDATE`
  failures separately (log-only, don't re-flag as a send failure).

### NIT

**N2 (new) — Crash between the JSON-import commit and the `.migrated` rename leaves the legacy
file orphaned forever**

- Files: `bot/plugins/agent_cron.py:455-459`, `bot/plugins/reminders.py:230-234` (both
  `_import_json_*_sync`, the `count > 0: return` early-exit before the file-exists check).
- If the process is killed after the `executemany` INSERT commits but before the subsequent
  `os.replace(self.jobs_file, self.jobs_file + ".migrated")` (`agent_cron.py:499`) /
  `os.replace(self.reminders_file, ...)` (`reminders.py:279`) runs, the table is already non-empty
  on every future `initialize()`, so the `count > 0: return` guard short-circuits before ever
  reaching the rename again. No data loss or duplicate import (the guard is what prevents
  re-import), but the original `.json` file is left on disk indefinitely with no automatic
  cleanup — purely cosmetic/operational, not covered by a test, not required to fix given the
  narrow crash window and harmless outcome.

### Summary

- ERROR: 0
- WARNING: 1 (W3 — post-send DELETE failure in `check_reminders` can cause a duplicate reminder
  send on retry)
- NIT: 1 (N2 — crash between import-commit and file-rename orphans the legacy JSON file)

All four round-1 findings (E1, W1, W2, N1) verified fixed with accompanying regression tests
(E1, W1) or direct verification (W2 mypy diff, N1 code read). Targeted tests: 62 passed, 0 failed.
Ruff: clean. Mypy: 0 new `(file, error_code)` pairs vs. baseline for both plugin files. `get_spec()`
byte-identical to `HEAD` for both plugins.

---

## Раунд 3

Reviewed files (same scope, per `T04-plan.md` §0 extension): `bot/plugins/agent_cron.py`,
`bot/plugins/reminders.py`, `tests/test_agent_cron_plugin.py`, `tests/test_agent_cron_storage.py`,
`tests/test_reminders_fixes.py`, `tests/test_background_tasks.py`, `tests/test_concurrent_tool_state.py`.

Verification performed: re-read the full current diff against `HEAD` for all owned files, focused
on the W3 and N2 fixes; re-ran the full targeted test suite (63 passed — one extra test vs. round 2:
`test_delete_failure_after_successful_send_does_not_resend`); ruff clean on all seven owned files;
mypy on both plugin files grouped by `(file, error_code)` and diffed against
`/tmp/impl/mypy_before.keep` — all six pairs match exactly (no new errors); confirmed `get_spec()`
bodies untouched in the diff for both plugins; grepped the whole tree (excluding generated
`ai_docs_site/`) for leftover references to the removed `self.jobs`/`self.reminders`/
`save_reminders`/`load_reminders`/`_load_jobs`/`_save_jobs` — none found outside the two owned test
files (where the only matches are `plugin.reminders_file`, a still-live attribute, and an unrelated
substring hit).

One test run mid-review threw `AttributeError: 'Database' object has no attribute
'_migrate_conversation_context_backfill_mode_key'` from `bot/database.py` (outside T04 ownership).
Confirmed transient — an immediate re-run of the same single test passed, and the full targeted
suite then ran clean (63 passed). Root cause is a concurrent edit to `bot/database.py` by another
in-flight task in this shared working tree (that file has uncommitted changes not present at the
start of this review session); not caused by any T04-owned file. Not reported as a finding per the
"report, don't fix" rule for out-of-ownership failures — noted here only so it isn't mistaken for
flakiness in T04's own code.

### Round-2 findings — verification

- **W3 (post-send DELETE failure treated as a send failure, risking a duplicate send) — FIXED.**
  `check_reminders` (`bot/plugins/reminders.py:328-397`) now wraps only the `send_reminder(...)`
  call in the retryable `try/except` (`:361-384`); a failure there is the only thing that increments
  `send_attempts` / retries the send. The follow-up `DELETE` after a successful send is in a
  separate `try/except` (`:390-397`): on failure it does **not** touch `send_attempts` and instead
  sets `status = 'sent'` so a later tick retries only the deletion. The claim query
  (`_claim_due_reminders_sync`, `:299-326`) adds `WHERE status != 'sent'` (`:308`) — verified this
  exclusion is load-bearing, not decorative: without it, a `'sent'` row would be immediately
  reclaimable on the very next tick regardless of any lease, because the lease gate
  (`status != 'processing' OR locked_at IS NULL OR locked_at <= ?`, `:309`) short-circuits to true
  for any status other than `'processing'` — so omitting the `'sent'` exclusion would have caused an
  actual resend, not just a display glitch. A dedicated top-of-tick cleanup
  (`:335-345`) retries the `DELETE` for any row left in `status = 'sent'` from a prior tick, wrapped
  per-row in its own `try/except` so one failing cleanup doesn't block the rest of the tick or the
  claim/send loop that follows. Covered end-to-end by the new
  `test_delete_failure_after_successful_send_does_not_resend`
  (`tests/test_reminders_fixes.py:521-558`), which injects a one-shot `DELETE` failure right after a
  successful send via a patched `db_handle.execute`, and asserts: the send is not retried
  (`send_attempts == 0` after the failure), the row survives so it can be cleaned up, a *second*
  `check_reminders()` call does not resend (`helper.sent` still length 1), and the row is gone by
  the end. This is a solid regression test for the exact failure mode W3 described.
- **N2 (crash between import-commit and rename orphans the legacy JSON file) — FIXED for both
  plugins.** `_import_json_jobs_sync` (`bot/plugins/agent_cron.py:455-462`) and
  `_import_json_reminders_sync` (`bot/plugins/reminders.py:230-240`) both now do, inside the
  `count > 0: return` early-exit: `if os.path.exists(self.<file>): os.replace(self.<file>, self.<file>
  + ".migrated")` before returning — i.e. the table-non-empty guard still prevents re-import, but
  now also finishes the rename it may have missed on a prior crash, and is idempotent on every
  subsequent `initialize()` (file no longer exists → the `os.path.exists` check is false → no-op).
  Covered by `test_import_skips_when_table_not_empty` in both `tests/test_agent_cron_storage.py:118-132`
  and `tests/test_reminders_fixes.py:414-434`, both updated in this round to assert the `.migrated`
  file exists and the JSON's own row was *not* imported (simulating the crash window by inserting a
  row directly, then running `initialize()` with the legacy file still present).

### WARNING

**W4 (new) — `list_reminders` and the `/reminders` inline menu still display `status='sent'` rows as
if they were still pending**

- Files: `bot/plugins/reminders.py:467-479` (`execute`, `function_name == "list_reminders"`),
  `:142-192` (`handle_prompt_constructor`, the `/list_reminders` command's inline-keyboard view), and
  `:570-573` (the post-delete-refresh listing inside `handle_reminder_callback`).
- All three call sites run `SELECT * FROM reminders WHERE owner_id = ? ORDER BY created_at ASC` with
  no `status` filter. A row can sit in `status = 'sent'` for up to one 60-second tick — from the
  moment the post-send `DELETE` fails (the exact scenario W3 fixes) until the top-of-tick cleanup in
  `check_reminders` (`:338-345`) successfully deletes it on the next run, or indefinitely if that
  retry keeps failing (e.g. the same underlying disk/lock issue persists).
- During that window, a reminder that has **already been delivered** to the user is shown by both
  `list_reminders` (the tool the model calls) and the `/list_reminders` inline-keyboard menu with its
  original time/message, indistinguishable from a still-active one — this directly contradicts the
  tool's own spec text, `"List the active (not yet fired) reminders..."`
  (`bot/plugins/reminders.py:69-72`, unchanged, not touched by this diff). A user (or the model, via
  `list_reminders`) asking "what reminders do I have" in that window sees a reminder as pending that
  has, in fact, already fired.
- Not a data-loss or duplicate-send risk (clicking "delete" on a `'sent'` row in the inline menu just
  performs the same cleanup the background retry would have done anyway — harmless), and the window
  is narrow and self-healing in the common case. But it is a real, untested gap in the `'sent'`-status
  handling the round-3 fix introduced: no test in this diff exercises what `list_reminders` /
  `handle_prompt_constructor` return when a row is in `status = 'sent'`.
- Suggested fix: add `AND status != 'sent'` (or `WHERE ... AND status = 'pending'` if `'processing'`
  rows — mid-send — should also stay hidden, matching "not yet fired") to the three `SELECT` queries
  listed above.

### Summary

- ERROR: 0
- WARNING: 1 (W4 — `list_reminders`/`/reminders` menu display already-sent (`status='sent'`)
  reminders as if still pending, for up to one tick after a post-send `DELETE` failure)
- NIT: 0

Both round-2 findings (W3, N2) verified fixed with accompanying regression tests. Targeted tests: 63
passed, 0 failed (one transient failure mid-review traced to a concurrent out-of-ownership edit in
`bot/database.py`, not reproducible on rerun — see verification notes above). Ruff: clean. Mypy: 0
new `(file, error_code)` pairs vs. baseline for both plugin files. `get_spec()` untouched for both
plugins. No leftover references to removed storage internals found anywhere else in the tree.

---

## Раунд 4

Scope: verify the W4 fix (all reminder-listing SELECTs exclude `status='sent'`) plus a quick
re-scan of the whole cumulative T04 diff for `bot/plugins/agent_cron.py` and
`bot/plugins/reminders.py`.

Verification performed: `git diff HEAD` read in full for both owned plugin files (470 and 598
diff lines respectively); grepped `bot/plugins/reminders.py` for every `FROM reminders` SELECT to
confirm all three listing queries carry the filter and that the two by-id lookups (view/delete,
which intentionally must still find a `'sent'` row so it can be cleaned up) do not; ran the
targeted test suite; ruff on all seven owned/relevant files; mypy on both plugin files grouped by
`(file, error_code)` and diffed against `/tmp/impl/mypy_before.keep`; byte-diffed `get_spec()`
bodies against `HEAD` for both plugins; grepped both files for any remaining reference to the
removed `self.reminders`/`self.jobs`/`save_reminders`/`load_reminders`/`_load_jobs`/`_save_jobs`.

### W4 — verification

**FIXED.** All three reminder-listing `SELECT`s now read
`SELECT * FROM reminders WHERE owner_id = ? AND status != 'sent' ORDER BY created_at ASC`:
- `handle_prompt_constructor` (`bot/plugins/reminders.py:147`, the `/reminders` inline-keyboard
  view),
- `execute`'s `list_reminders` branch (`bot/plugins/reminders.py:469`, the tool the model calls),
- the post-delete list refresh inside `handle_reminder_callback` (`bot/plugins/reminders.py:572`).

The two by-id lookups that intentionally must still see a `'sent'` row —
`handle_reminder_callback`'s "view" (`:535`) and "delete" (`:562`) branches, and `execute`'s
`delete_reminder` (`:492`) — correctly have no status filter, since a user must still be able to
open/clean up a reminder stuck in `'sent'` from a failed post-send `DELETE` (the scenario W3/the
top-of-tick cleanup at `:338-345` handle). This is the right scope for the fix: exclude `'sent'`
only from "what's still pending" listings, not from "can I still touch this specific row".

Covered by the new `tests/test_reminders_fixes.py::test_list_reminders_excludes_sent_status`
(`:582-594`): inserts one `'pending'` and one `'sent'` reminder for the same owner, calls
`execute("list_reminders", ...)`, and asserts the pending reminder's id appears in the rendered
text while the sent one's does not. This is a real behavioral assertion (not a mock/structure
check) and would catch a regression of the filter on the tool-facing path. It does not cover the
other two call sites (`handle_prompt_constructor`, the callback post-delete refresh) — those were
verified by direct code read instead (identical query text, same fix), which is adequate given
all three are one-line, mechanically identical changes.

### Full-diff re-scan — no new issues found

Re-read both plugin files' cumulative diffs end to end (not just the W4 hunk): claim/lease logic
in `_claim_due_jobs_sync`/`_claim_job_by_id_sync`/`_claim_due_reminders_sync`, the mid-run
existence re-checks in `_run_job` (E1 fix, still present at `bot/plugins/agent_cron.py:286` and
`:304`), the `exclude_ids` plumbing (W1 fix, `:180-182`, `:209-241`), the send/delete split and
`'sent'`-marker fallback in `check_reminders` (W3 fix, `bot/plugins/reminders.py:352-397`), and
both `_import_json_*_sync` crash-recovery renames (N2 fix, `agent_cron.py:342-352`,
`reminders.py:160-170`) — all match the round 1-3 descriptions with no further drift or new
defects. Nothing outside the W4 hunk changed since round 3 in either plugin file.

Test run: `tests/test_agent_cron_plugin.py tests/test_agent_cron_storage.py
tests/test_reminders_fixes.py tests/test_background_tasks.py tests/test_concurrent_tool_state.py`
— 64 passed (up from 63 in round 3, the one new test), 0 failed. Ruff: clean on all seven files.
Mypy on both plugin files, grouped by `(file, error_code)`, diffed against
`/tmp/impl/mypy_before.keep`: all pairs match exactly
(`agent_cron.py`: `annotation-unchecked`=3, `arg-type`=1, `misc`=2, `union-attr`=1;
`reminders.py`: `assignment`=1, `union-attr`=17) — no new errors. `get_spec()` bodies
byte-identical to `HEAD` for both plugins. No leftover references to removed storage internals
(`self.reminders`, `self.jobs`, `save_reminders`, `load_reminders`, `_load_jobs`, `_save_jobs`)
in either file.

### Summary

- ERROR: 0
- WARNING: 0
- NIT: 0

W4 verified fixed with a targeted regression test on the tool-facing path and direct-read
verification on the other two identical call sites. Full re-scan of the cumulative diff found no
new issues. T04 is clean as of round 4.
