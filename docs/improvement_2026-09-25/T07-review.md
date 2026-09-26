# T07 Review — Блокировка одного экземпляра

## Раунд 1

Reviewed files (per task ownership, `T07-plan.md` §0): `bot/instance_lock.py` (new, read in
full), `bot/__main__.py` (only the lock-insertion hunk: the new import and the
`acquire_instance_lock`/`InstanceLockError` block after the required-env check — the
`ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` hunks in the same file belong to another task and
were not reviewed), `tests/test_instance_lock.py` (new, read in full), `AGENTS.md`,
`README.md`, `README.ru.md`, `.env.example`, `.gitignore`.

Verification performed: full diff read against `HEAD` for `bot/__main__.py`,
`.env.example`, `.gitignore`, `README.md`, `README.ru.md`, `AGENTS.md`; `bot/instance_lock.py`
and `tests/test_instance_lock.py` read in full as new/untracked files; line-number check that
the lock block (`bot/__main__.py:206-213`) runs before `PluginManager(` (`:391`),
`Database.configure`/`Database(` (`:394`/`:400`), `OpenAIHelper(` (`:407`),
`ChatGPTTelegramBot(` (`:410`); cross-checked `default_lock_path`'s fallback against
`Database.__new__`'s real fallback (`bot/database.py:250-256`, both resolve to
`os.path.dirname(os.path.abspath(__file__))` of their own module, i.e. the real `bot/`
package directory when `DB_PATH` is unset); targeted test run
(`tests/test_instance_lock.py tests/test_telegram_builder_config.py`) — 54 passed; full
`tests/` — 1775 passed (baseline 1650, no new red); ruff clean on
`bot/instance_lock.py bot/__main__.py tests/test_instance_lock.py`; mypy on
`bot/instance_lock.py bot/__main__.py`, filtered to lines reported in those two files and
compared against `/tmp/impl/mypy_before.keep` — 0 errors in both files, before and after (the
375 errors mypy prints come entirely from `bot/telegram_bot.py`, pulled in transitively, not
our ownership); reproduced the actual repo state after a test run to check for stray lock
files (see E3 below).

---

### ERROR

**E1 — Required `AGENTS.md` documentation line (plan §4) was never added**

- File: `AGENTS.md` ("Project Shape" section).
- `git diff HEAD -- AGENTS.md` is empty, and a full-text search of the current file for
  `instance_lock`, `INSTANCE_LOCK`, `single-instance`, `single instance`, `flock` returns zero
  matches anywhere in the file.
- `T07-plan.md` §0 lists `AGENTS.md` (раздел Project Shape — одна фраза) as an owned file, and
  §4 specifies the exact sentence to insert right after "Startup creates `PluginManager`,
  `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`". This is a required deliverable of the
  task, not optional polish, and it is simply missing.
- Fix: add the sentence from `T07-plan.md` §4, with the line reference updated to the actual
  post-edit location of the lock block (`bot/__main__.py:206-213`), e.g.: "`main()` acquires a
  single-instance file lock (`bot/instance_lock.py`) before any of the above are created; a
  second process against the same lock file logs ERROR and exits non-zero
  (`bot/__main__.py:206-213`)."

**E2 — `.env.example` edited despite the plan explicitly forbidding it**

- File: `.env.example` (new hunk near the `DB_PATH`/Docker section).
- `T07-plan.md` §7 item 3, verbatim: "**`.env.example` не во владении T07** ... Не добавлять
  `INSTANCE_LOCK_PATH` в `.env.example` — файл не во владении T07." Neither `T07-plan.md` §0
  nor the master plan's T07 section lists `.env.example` as an owned file.
- The diff nonetheless adds:
  ```
  +# Only one bot process may run against the same TELEGRAM_BOT_TOKEN/database at a time;
  +# a second process fails fast (see README). Default lock file path lives next to
  +# DB_PATH (<dir(DB_PATH)>/bot.instance.lock); override to relocate it.
  +# INSTANCE_LOCK_PATH=/app/data/bot.instance.lock
  ```
  The content itself is accurate (`/app/data/bot.instance.lock` matches the Docker default
  `DB_PATH=/app/data/user_data.db`), so this is not a correctness bug, but it is a direct
  violation of the task's own plan and of the common ownership rule ("Fix ONLY files listed in
  your task's ownership"). `.env.example` is being edited concurrently by another task in the
  same diff (the unrelated `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` hunk right above this
  one), which is exactly the kind of shared-file collision the ownership rule exists to avoid.
- Fix: revert just these four lines from `.env.example` (reverse string replacement, leaving
  the rest of the file's diff untouched), or get explicit coordinator sign-off and update
  `T07-plan.md` §0/§7 to reflect the ownership change.

**E3 — A real lock file leaks into the source tree during test runs (confirmed by reproduction)**

- Files: `bot/instance_lock.py` (`default_lock_path`, root cause) and
  `tests/test_telegram_builder_config.py` (not T07-owned, trigger site).
- `default_lock_path(None)` resolves to `os.path.dirname(os.path.abspath(__file__))` inside
  `bot/instance_lock.py`, i.e. the real `bot/` package directory, when neither `DB_PATH` nor
  `INSTANCE_LOCK_PATH` is set. This intentionally mirrors `Database.__new__`'s existing
  fallback (`bot/database.py:250-256`), so it is not a novel anti-pattern in isolation.
- However, `tests/test_telegram_builder_config.py::_run_main_with_fake_dependencies` (owned by
  another task) fakes `PluginManager`/`Database`/`OpenAIHelper`/`ChatGPTTelegramBot` precisely
  so calling `bot_main.main()` twice in-process has no real side effects, and it does **not**
  set `DB_PATH` or `INSTANCE_LOCK_PATH`. The new `acquire_instance_lock(...)` call in `main()`
  is not faked by that test and is not mockable without editing it (out of T07's ownership), so
  it runs for real: it opens and `flock`s the actual file `bot/bot.instance.lock` in the
  working tree.
- Reproduced directly in this review: after running `pytest tests -q --no-header
  -p no:cacheprovider` in this session, `bot/bot.instance.lock` exists on disk (0 bytes,
  created during the run). It does not show up in `git status --short` only because of the
  `*.instance.lock` line newly added to `.gitignore` (see W1) — i.e. the leak is masked, not
  fixed. The OS-level lock on that real file is held open for the rest of the pytest process
  (the module-level `_held_handles` cache in `instance_lock.py` is never reset outside
  `tests/test_instance_lock.py`'s own autouse fixture).
- This was called out explicitly as a review focus item ("no leaked lock files in repo from
  tests") and is not satisfied.
- Suggested fix (crosses ownership, needs coordinator arbitration since the trigger site is a
  file T07 cannot edit): either (a) the owner of `test_telegram_builder_config.py` adds
  `monkeypatch.setenv("INSTANCE_LOCK_PATH", str(tmp_path / "bot.instance.lock"))` to
  `_run_main_with_fake_dependencies`, or (b) T07 reconsiders defaulting `default_lock_path`
  to a path inside the package's own source directory at all (e.g. requiring `DB_PATH` to be
  set, or falling back to a path outside the repo tree) even though that would diverge from
  `Database`'s existing convention. Whichever direction is chosen, note it explicitly in
  `T07-plan.md` since the current text treats this as already resolved (§0 "Критичное
  ограничение" only discusses the *same-process* re-entrancy problem, not this real-disk-write
  side effect).

---

### WARNING

**W1 — `.gitignore` edited despite not being an owned file**

- File: `.gitignore` (new line `*.instance.lock`).
- Neither the master plan's T07 section nor `T07-plan.md` §0 lists `.gitignore` as owned by
  T07 (it is owned by T01). `T07-plan.md` §7 item 1 explicitly identifies this exact gap
  (default lock file not covered by any existing pattern) but defers the decision to the
  coordinator ("либо координатор добавляет её сам, либо это одна строка для следующей
  волны/финального ревью") rather than authorizing T07 to add it unilaterally.
- Lower risk than E2: it's a single additive line that doesn't conflict with T01's other
  additions in the same diff (`.mypy_cache/`, `.coverage`), and the plan itself flags the line
  as eventually necessary. Still a formal ownership-boundary violation, and it's the change
  that hides E3 from `git status`.
- Fix: same as E2 — revert the line, or get explicit coordinator sign-off and update the plan.

---

### NIT

(none)

---

### Summary

- ERROR: 3 (E1 — `AGENTS.md` doc line missing; E2 — `.env.example` edited against the plan's
  explicit instruction; E3 — real `bot/bot.instance.lock` leaks into the source tree during
  `tests/test_telegram_builder_config.py`, confirmed by reproduction)
- WARNING: 1 (W1 — `.gitignore` edited outside T07's ownership)
- NIT: 0

Everything else checked matched the plan and showed no issues: fd-lifetime (module-level
`_held_handles` dict holds a strong reference, no GC risk), lock-acquired-before-components
ordering (verified by line numbers), `default_lock_path`'s DB_PATH-relative logic (matches
`Database`'s own fallback and the plan's test expectations), directory creation
(`os.makedirs(..., exist_ok=True)`), error message content and the `INSTANCE_LOCK_PATH`
pointer, the Windows `fcntl is None` soft-degradation path, per-process cache correctness
(single `threading.Lock()` around the whole read-check-open-flock-store sequence, keyed by
`os.path.abspath`, no race), the subprocess test
(`test_second_process_subprocess_denied`, genuinely spawns a separate Python process and
exercises real cross-process `flock` contention), and the pre-approved use of
`log_exception_shape(exc)` instead of `str(exc)` in the error-log call
(`bot/__main__.py:212`) — it only prepends the exception class name and does not strip or add
anything that would hide the "Another bot instance..." message the test asserts on.

---

## Раунд 2

Scope per coordinator ruling: E2 (`.env.example`) and W1 (`.gitignore`) from Раунд 1 are
explicitly authorized by the coordinator and are recorded below as ACCEPTED, not findings.
Focus of this round is re-verifying E1 (`AGENTS.md` doc line) and E3 (stray
`bot/bot.instance.lock` leak during tests), plus a regression check that nothing else broke.

Verification performed: `git diff HEAD` read in full for `AGENTS.md`, `tests/conftest.py`,
`.env.example`, `.gitignore`; `bot/__main__.py` re-read at lines 190-213 and 360-410 to
re-confirm the lock block's exact line span and its position relative to the required-env
check and component creation; `bot/instance_lock.py` re-read in full (unchanged since Раунд 1);
`tests/test_instance_lock.py` re-read in full (unchanged, 8 tests matching plan §5); ran
`tests/test_instance_lock.py tests/test_telegram_builder_config.py` (54 passed) and then the
full `tests/` suite (1775 passed, same baseline as Раунд 1, no new red) with a filesystem sweep
(`find . -iname "*.instance.lock"`) and `git status --short | grep -i lock` after each run;
`ruff check` on `bot/instance_lock.py bot/__main__.py tests/test_instance_lock.py
tests/conftest.py` (clean); `mypy` on the same three non-test-data files plus `tests/conftest.py`
against `~/.venvs/ctb/bin/python` (0 errors reported for any of them, consistent with Раунд 1's
baseline comparison); repo-wide text search confirmed `INSTANCE_LOCK_PATH` appears only in
`tests/conftest.py` and `tests/test_instance_lock.py` (no other test hardcodes or fights the new
autouse fixture).

---

### ACCEPTED (coordinator ruling, not findings)

**E2 — `.env.example` `INSTANCE_LOCK_PATH` block.** Present, unchanged since Раунд 1
(4 lines after the `DB_OP_LOCK_TIMEOUT_SECONDS` comment). Coordinator explicitly authorized this
edit; content still accurate (`/app/data/bot.instance.lock` matches the Docker
`DB_PATH=/app/data/user_data.db` default). No further action.

**W1 — `.gitignore` `*.instance.lock` line.** Present, unchanged since Раунд 1. Coordinator
explicitly authorized this edit. No further action.

---

### ERROR

(none)

---

### WARNING

**E1 (re-verified) — line citation is accurate, but "before any of the above" over-claims
ordering relative to the required-env-vars check**

- File: `AGENTS.md` (Project Shape section, new bullet).
- The missing-line finding from Раунд 1 is fixed: the sentence is now present, immediately after
  the `PluginManager`/`Database`/`OpenAIHelper`/`ChatGPTTelegramBot` bullet, and its citation
  `bot/__main__.py:206-213` is byte-exact — that span is precisely `lock_path = ...` through the
  `exit(1)` of the `except InstanceLockError` block (confirmed by reading `bot/__main__.py:204-214`
  directly). `defaulting to a path next to DB_PATH and overridable via INSTANCE_LOCK_PATH` and
  `logs ERROR and exits non-zero` both match the code at those lines.
- However, the sentence reads: "Before any of the above, `main()` acquires a single-instance
  file lock...". `T07-plan.md` §4's suggested wording was "before any of the above **are
  created**" — anaphoric to the object list of the immediately preceding bullet ("creates
  `PluginManager`, `Database`, `OpenAIHelper`, then `ChatGPTTelegramBot`"), which is exactly what
  the plan's own §2 discusses: the lock is deliberately placed *after* the required-env-vars
  check (lines 199-204) and *before* component creation (`PluginManager(` at line ~391), a
  distinction the plan calls out on purpose ("ошибка конфигурации... более базовая, чем занятый
  лок, пусть репортится первой").
- Dropping "are created" broadens "the above" to plausibly include the required-env-vars bullet
  two lines up too ("Required runtime env vars are `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`;
  startup exits when either is missing"). Read that way, the sentence claims the lock is acquired
  before that check as well, which is false: `bot/__main__.py:194-204` (the missing-env check)
  runs strictly before `bot/__main__.py:206` (the lock). Confirmed by direct code read, not
  inference.
- Impact is low (both orderings still guarantee the lock precedes any component construction,
  which is the property AGENTS.md callers actually rely on per the "Do not make runtime claims
  from memory... cite file:line" rule), but the sentence as literally written is checkable and,
  under the natural "all preceding bullets" reading, wrong about one specific fact (the lock vs.
  the env-var check ordering).
- Suggested fix: restore the plan's original anaphora, e.g. "Before `PluginManager`, `Database`,
  `OpenAIHelper`, and `ChatGPTTelegramBot` are created, `main()` acquires a single-instance file
  lock (`bot/instance_lock.py`)...", which removes the ambiguity without changing the rest of the
  sentence.

---

### NIT

(none)

---

### E3 (re-verified) — PASS, no finding

- Files: `tests/conftest.py` (new autouse fixture `_instance_lock_path_in_tmp`), `bot/__main__.py`
  (unchanged lock block), `bot/instance_lock.py` (unchanged).
- The new fixture (`tests/conftest.py`, added right after `_close_pytest_asyncio_baseline_loop`)
  is `autouse=True`, takes `tmp_path`/`monkeypatch`, and does
  `monkeypatch.setenv("INSTANCE_LOCK_PATH", str(tmp_path / "bot.instance.lock"))` for every test
  in `tests/` — not just the two files named in this round's scope. Confirmed this is the only
  other place `INSTANCE_LOCK_PATH` appears in `tests/` besides `tests/test_instance_lock.py`
  itself (repo-wide search), so no test fights or duplicates the redirect.
- Reproduced live: ran `tests/test_instance_lock.py tests/test_telegram_builder_config.py`
  (54 passed) then swept the tree with `find . -iname "*.instance.lock"` and
  `git status --short | grep -i lock` — no lock file anywhere, only the two pre-existing
  untracked source files (`bot/instance_lock.py`, `tests/test_instance_lock.py`) show as `??`.
  Repeated the same sweep after the full `tests/` run (1775 passed) — same clean result. This is
  a genuine fix, not just a `.gitignore` mask as in Раунд 1: the file is never created in the
  source tree at all now, because `bot_main.main()`'s
  `os.environ.get('INSTANCE_LOCK_PATH') or default_lock_path(...)` (`bot/__main__.py:206-208`)
  picks up the fixture's tmp-path override before falling back to the in-repo default.
- Checked the "doesn't mask default-path coverage" requirement directly: `default_lock_path()`
  (`bot/instance_lock.py:36-45`) is a pure function of its `db_path` parameter — it never reads
  `os.environ` itself. `test_default_lock_path_next_to_db` (`tests/test_instance_lock.py:89-91`)
  calls it directly with explicit arguments (`"/x/data/user_data.db"` and `None`), so the
  autouse fixture's env-var override has no effect on that test's assertions; the `None`-path
  branch (asserting `.../bot/bot.instance.lock`) is exercised exactly as before. No masking.
- `test_main_exits_nonzero_when_lock_held` (`tests/test_instance_lock.py:103-151`) sets its own
  `INSTANCE_LOCK_PATH` via `monkeypatch.setenv` (line 140), which shadows the fixture's value for
  that test and is correctly torn down in LIFO order by `monkeypatch` — no interaction bug.

---

### Summary (Раунд 2)

- ERROR: 0
- WARNING: 1 (E1 re-verified — line citation and code-relevant facts are accurate, but "before
  any of the above" ambiguously over-claims ordering relative to the required-env-vars check;
  low impact, easy rewording fix)
- NIT: 0
- ACCEPTED (not findings, per coordinator ruling): E2 (`.env.example`), W1 (`.gitignore`) — both
  present and unchanged since Раунд 1.
- E3 re-verified as fully fixed (not just masked): reproduced zero stray `*.instance.lock` files
  in the source tree after both the targeted test pair and the full suite; confirmed the fixture
  does not mask `test_default_lock_path_next_to_db`'s coverage of the `None`-path branch.
- Full `tests/` suite: 1775 passed (same as Раунд 1 baseline, no new red). `ruff check` clean on
  `bot/instance_lock.py`, `bot/__main__.py`, `tests/test_instance_lock.py`, `tests/conftest.py`.
  `mypy` on the same four files: 0 errors, consistent with Раунд 1.
