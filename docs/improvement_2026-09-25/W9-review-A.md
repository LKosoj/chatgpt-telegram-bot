# W9 Review — GROUP A (security area)

Files owned: `bot/net_safety.py`, `bot/artifact_paths.py`, `bot/instance_lock.py`,
`bot/agent_delivery.py`, `bot/plugins/plugin.py`, `bot/plugins/mcp_server.py`,
`bot/plugins/webshot.py`, `bot/plugins/codeinterpreter.py`, `bot/plugins/text_summarizer.py`,
`bot/plugins/github_analysis.py`, `bot/plugins/haiper_image_to_video.py`,
`bot/plugins/terminal.py`, `bot/__main__.py`, `bot/README_MCP.md`, taint/untrusted-wrapping
parts of `bot/openai_tool_handler.py` and `bot/openai_helper.py`
(`__add_function_call_to_history`), delivery/access parts of `bot/utils.py`
(`handle_direct_result`, `cleanup_intermediate_files`, `_charge_user_and_guest*`, group-access
gate), `bot/plugins/agent_tools.py` (`_normalize_delivery_artifacts`, `_deliver_to_user`,
subagent loop wrapping), network helpers in `bot/plugins/skills.py`, plus their tests.

## Раунд 1

### Summary

- ERROR: 0
- WARNING: 2
- NIT: 0

All owned modules (`net_safety.py`, `artifact_paths.py`, `instance_lock.py`) were read in full
and cross-checked against `T02/T03/T05/T07/T08-plan.md`; every ERROR/WARNING raised in the prior
`Txx-review.md` rounds for files in this group is fixed in the current tree, and the few
explicitly-accepted items (T05 mypy NIT deferred to T13, T05 wide-storage_root scope-check gap
documented as pre-existing, T07 `.env.example`/`.gitignore` edits, T07 E1 wording fix) are not
re-raised here. `get_spec()` output was diffed line-by-line across all 9 owned plugin files —
only 3 benign `-> [Dict]` → `-> List[Dict]` type-annotation changes were found, no content
changes (not a forbidden tool-spec change).

Full diff for `bot/plugins/text_summarizer.py`, `github_analysis.py`,
`haiper_image_to_video.py`, `webshot.py` was read line-by-line in this round (previously only
checked via the T03-review claim that rounds 1→2 were unchanged): matches `T03-plan.md`
exactly for the SSRF migration in all four files. `haiper_image_to_video.py`'s diff also
contains a menu-text/keyboard dedup refactor (`_build_settings_summary_text` /
`_build_style_effect_keyboard`) and mypy-driven `assert`/`cast` additions unrelated to T03/T05;
spot-checked these are behavior-preserving (exact code motion, no logic change) and confirmed
via `tests/test_t12d_haiper_menu_text.py` (10 passed) — not flagged, not a regression.

Targeted tests: 295 passed (`tests/test_net_safety.py tests/test_artifact_paths.py
tests/test_instance_lock.py bot/tests/test_mcp_server.py tests/test_webshot_plugin.py
tests/test_codeinterpreter_plugin.py tests/test_text_summarizer_plugin.py
tests/test_github_analysis_plugin.py tests/test_agent_delivery.py
tests/test_plugin_direct_results.py tests/test_callback_authorization.py
tests/test_usage_budget.py tests/test_skills_plugin.py`); 308 passed
(`tests/test_openai_helper_tool_calls.py tests/test_agent_tools_plugin.py
tests/test_no_hardcoded_plugin_refs.py tests/test_no_private_helper_access.py
tests/test_plugin_manager.py`); plus 10 passed for
`tests/test_haiper_image_to_video_async_db.py tests/test_t12d_haiper_menu_text.py` (not in the
explicit ownership test list but cover an owned file, run for due diligence). `ruff check` on
all 18 owned files: all clean.

Spot-verified test bodies (not just names) for the highest-risk new tests:
`test_is_deliverable_rejects_db_path_via_symlink` in `tests/test_artifact_paths.py` genuinely
creates a symlink to `DB_PATH` and asserts `is_deliverable` denies the symlink path — confirms
`is_protected_path`/`is_deliverable` resolve symlinks before comparison, not a superficial
string-match test. `test_execute_rejects_oversized_response` in `tests/test_webshot_plugin.py`
and `test_process_animate_command_downloads_video_via_safe_get` /
`test_extract_text_from_url_uses_safe_get` genuinely assert the `net_safety.safe_get` call
shape (`max_bytes`, `timeout`) and the size-limit rejection path, not just that *a* call
happened.

### WARNING — AGENTS.md `bot/__main__.py` line refs are stale (3 citations)

- **File:** `AGENTS.md:26`, `AGENTS.md:28`, `AGENTS.md:32`
- **Why:** Three `bot/__main__.py` line-range citations no longer point at the described code:
  - `AGENTS.md:26`: "required runtime env vars ... startup exits when either is missing
    (`bot/__main__.py:199-203`)" — the actual check+exit is now at
    `bot/__main__.py:184-188`. Lines 199-203 are now inside `# Setup configurations`
    (`model_choices = parse_model_list_env(...)` etc.), unrelated content.
  - `AGENTS.md:28`: "Startup creates `PluginManager`, `Database`, `OpenAIHelper`, then
    `ChatGPTTelegramBot` (`bot/__main__.py:366-385`)" — actual current lines are
    `PluginManager(` at 375, `Database.configure(` at 378, `db = Database()` at 384,
    `OpenAIHelper(` at 391, `ChatGPTTelegramBot(` at 394 (i.e. ~375-394, not 366-385).
  - `AGENTS.md:32` (the T07-introduced instance-lock bullet): "... a second process against
    the same lock file logs ERROR and exits non-zero (`bot/__main__.py:206-213`)" — the
    actual `try: acquire_instance_lock(lock_path) / except InstanceLockError ...` block is
    at `bot/__main__.py:190-197`.

  All three point into the same file (owned by this review group) and appear to have drifted
  from cumulative line-count changes across the whole change set, not from a single task in
  isolation — but the `:32` one documents this group's own T07 feature, so it is squarely this
  group's responsibility.
- **Failure scenario:** A developer or agent trusts the AGENTS.md citation, opens
  `bot/__main__.py:199-203` or `:366-385` or `:206-213` expecting to see the described logic,
  and instead reads unrelated code — wasting time, or worse, editing the wrong block under a
  false assumption of where the described behavior lives.
- **Suggested fix:** Update the three ranges to `184-188`, `375-394`, `190-197` respectively.

### WARNING — AGENTS.md `bot/plugins/plugin.py:18` citation now points to the wrong method

- **File:** `AGENTS.md:55`
- **Why:** "Stable plugin identity is `plugin_id`; tool namespace is `function_prefix`,
  defaulting to `plugin_id` (`bot/plugins/plugin.py:11`, `bot/plugins/plugin.py:18`)." At HEAD
  `08bc457`, line 11 was `plugin_id: str | None = None` and line 18 was
  `def get_function_prefix(self) -> str:` (the function implementing "function_prefix
  defaulting to plugin_id") — both citations were exactly correct there. This group's diff to
  `bot/plugins/plugin.py` adds a new `import re` at the top (line 1) and a new
  `returns_untrusted_content: bool = False` class attribute (line 14, for T08's untrusted-output
  wrapping), which together shift everything from the old `get_plugin_id`/`get_function_prefix`
  block down by 2. Current line 18 is now `return self.plugin_id or self.__class__.__name__`
  — the body of `get_plugin_id()` (plugin_id defaulting to class name), not
  `get_function_prefix()` (function_prefix defaulting to plugin_id, now at line 20). The `:11`
  citation also drifted by one line (now a blank line; `plugin_id` itself is at line 12) but at
  least still lands adjacent to the right attribute — the `:18` one now cites a different
  method's logic than the sentence describes.
- **Failure scenario:** Same class of risk as above: a reader following `:18` to verify how
  `function_prefix` defaults to `plugin_id` lands inside `get_plugin_id()` instead and may
  misread or miscite the wrong method when explaining/relying on this behavior.
- **Suggested fix:** Update to `bot/plugins/plugin.py:12` and `bot/plugins/plugin.py:20`.

## Раунд 2

### Round-1 WARNINGs verified fixed

Opened every cited line on disk directly (not trusting the round-1 text):

- `AGENTS.md:26` → `bot/__main__.py:184-188` is exactly the `required_values`/`missing_values`
  check + `exit(1)` block. Matches.
- `AGENTS.md:28` → `bot/__main__.py:375-394` runs from `PluginManager(` through
  `ChatGPTTelegramBot(...)` (`telegram_bot.run()` immediately follows at `:395`). Matches.
- `AGENTS.md:32` → `bot/__main__.py:190-197` is exactly the `acquire_instance_lock`/
  `InstanceLockError`/`exit(1)` block. Matches.
- `AGENTS.md:55` → `bot/plugins/plugin.py:12` is `plugin_id: str | None = None`,
  `bot/plugins/plugin.py:20` is `def get_function_prefix(self) -> str:`. Matches.

All four round-1 WARNINGs are fixed. Not re-raised.

### Fresh pass — scope and method

Re-read `bot/net_safety.py` in full (433 lines) against `T02-plan.md`/`T03-plan.md`/
`T03-review.md` (both rounds), re-read `bot/artifact_paths.py` and `bot/instance_lock.py` in
full (new untracked files, no `git diff HEAD` output for them — confirmed via
`git status --short`), and re-read `git diff HEAD` for every other owned file (`agent_delivery.py`,
`plugin.py`, `mcp_server.py`, `webshot.py`, `codeinterpreter.py`, `text_summarizer.py`,
`github_analysis.py`, `haiper_image_to_video.py`, `terminal.py`, `__main__.py`,
`README_MCP.md`, the SSRF/network hunks of `skills.py`, and the taint-wrapping hunks of
`openai_tool_handler.py`/`openai_helper.py`/`agent_tools.py`). Re-ran
`tests/test_openai_helper_tool_calls.py tests/test_agent_tools_plugin.py
tests/test_no_hardcoded_plugin_refs.py tests/test_no_private_helper_access.py
tests/test_plugin_manager.py` (308 passed, matches round 1 — code unchanged since round 1).
Re-diffed `get_spec()` in the 7 plugin files with public HTTP calls
(`mcp_server.py`, `webshot.py`, `codeinterpreter.py`, `text_summarizer.py`,
`github_analysis.py`, `haiper_image_to_video.py`, `terminal.py`): only the
`-> [Dict]` → `-> List[Dict]` annotation changes, no content changes — confirmed independently,
not a forbidden tool-spec change.

Checked and confirmed **not** findings (documented, plan-approved, or structurally
non-exploitable — not re-raised):

- **DNS rebinding in the async path stays open by design.** `bot/net_safety.safe_request`/
  `safe_get` (httpx-based) validate each URL via `validate_public_url_async` right before the
  request but do not pin the resolved IP for the actual TCP connect — httpx/httpcore resolve
  the hostname again internally. This is the exact thing round 2 was asked to check ("is the IP
  pinned or re-resolved?") — verified: for the async path it is **re-resolved**, a real TOCTOU
  window. But this is explicitly documented as an accepted, deliberate limitation, both in the
  module docstring (`bot/net_safety.py:17-26`) and in `T03-plan.md` §0 ("DNS rebinding в
  async-пути остаётся неполностью закрытым — осознанное и задокументированное ограничение, не
  баг"), with the concrete reason (closing it needs a custom `httpx.AsyncBaseTransport`/
  `httpcore.AsyncNetworkBackend`) and no test claims otherwise. The **sync** path
  (`safe_urlopen`, used by `skills.py` install flow) does pin correctly: `_build_conn` calls
  `resolve_public_ip` (which itself re-filters for `is_global`, not just replaying the earlier
  `validate_public_url` result) immediately before `socket.create_connection`, so the actual
  security boundary is enforced at connect-time, not just at the earlier check — verified by
  reading `bot/net_safety.py:298-350`.
- **Redirects and IPv6 forms** — re-verified by reading the code (not just trusting T03-review):
  both `safe_urlopen`'s `_Validating(HTTPRedirectHandler)` and `safe_request`'s manual redirect
  loop re-validate every redirect target through the same `validate_public_url`/`_async`, with a
  hard `max_redirects` cap in both. IPv6 literals (`http://[::1]/`) resolve correctly because
  `_check_url_shape` uses `urllib.parse.urlparse(...).hostname`, which strips brackets before
  `ipaddress.ip_address()`/`getaddrinfo()` ever see it; NAT64/6to4/Teredo embedding is unwrapped
  by `_is_global_ip` (T03-review round 1→2 fix, already re-verified in round 1 of this review).
- **Symlink races between `is_deliverable`/`is_protected_path` check and actual
  `open()`/`os.remove()`.** All three integration points
  (`agent_delivery.py:208-231`, `utils.py:1317-1376`, `agent_tools.py:2937-2942`) resolve via
  `os.path.realpath` for the *check*, then reopen by path afterward — a generic check-then-open
  TOCTOU gap exists in principle (swap the filesystem entry between check and open). Judged
  acceptable, not raised: (a) `cleanup_intermediate_files`'s `os.remove(value)` on a swapped
  symlink deletes only the symlink itself, never the target (POSIX `unlink` doesn't follow
  symlinks) — the delete path is safe by construction; (b) the delivery/open path would require
  an attacker who can race writes into the bot's own artifact directories with sub-request
  timing — a threat model requiring capabilities (arbitrary concurrent filesystem writes timed
  against a specific request) beyond what `is_deliverable`'s stated threat model
  (`T05-plan.md` — DB/`.env`/usage_logs/skills-source exfiltration, cross-scope artifact
  access) defends against. `is_deliverable`'s `request_started_at` recency-allow branch
  (`artifact_paths.py:120-125`) looked concerning in isolation (any file freshly touched
  anywhere is allowed) but is confirmed dead in production: none of the three call sites pass
  `request_started_at` (verified by reading all three call sites directly), matching
  `T05-plan.md` §0's explicit note that this parameter is unreachable and exists only for
  forward-compatibility with a future `RequestContext` timestamp field.

### WARNING — taint-logging (`DANGEROUS_TOOL_NAMES`) has no coverage inside the subagent tool loop

- **File:** `bot/plugins/agent_tools.py:3444-3458` (wrap point), `:3623-3675`
  (`_call_subagent_tool`), `:149-163` (`SUBAGENT_BLOCKED_FUNCTIONS`); contrast with
  `bot/openai_tool_handler.py:290-338` (`DANGEROUS_TOOL_NAMES`/`_tainted_plugin_ids`) and
  `:1445-1452` (the only call site of the check).
- **Why:** T08 added two independent things: (1) wrapping untrusted plugin output in
  `<untrusted_tool_output>` envelopes before it re-enters the model (content-level defense,
  read by the model itself), and (2) a `logger.warning(...)` at
  `bot/openai_tool_handler.py:1448-1452` that fires when a tool in `DANGEROUS_TOOL_NAMES`
  (`terminal.terminal`, `codeinterpreter.deep_analysis`, `skills.run_skill_script`,
  `mcp_server.register_mcp_server`, `agent_cron.create_cron_job`, plus 3 more) is called after
  an untrusted-content plugin already ran in the same request — an operator-visible signal for
  detecting a possible successful prompt injection (per plan, call still proceeds; this is
  telemetry, not a block). Part (1) is duplicated for subagents at
  `agent_tools.py:3451-3453` inside `_run_subagent_completion_loop`, matching the plan's
  explicit scope ("`bot/plugins/agent_tools.py` (только цикл субагента,
  `_run_subagent_completion_loop`)", `T08-plan.md`). Part (2) is **not** duplicated anywhere in
  `agent_tools.py` — confirmed by grepping the whole file and `bot/plugin_manager.py` for
  `DANGEROUS`/`tainted`: zero hits outside `openai_tool_handler.py`. Subagent tool calls go
  through `_call_subagent_tool` (`:3623-3675`), which calls
  `helper.plugin_manager.call_function(...)` directly (`:3670-3675`) — it never reaches
  `openai_tool_handler.handle_function_call`, the sole place the check lives. `grep` confirms
  `handle_function_call` is only invoked from `openai_helper.py`/`chat_run.py` (the main chat
  loop), never from `agent_tools.py`.
  `SUBAGENT_BLOCKED_FUNCTIONS` (`:149-163`) blocks 4 of the 9 `DANGEROUS_TOOL_NAMES` entries
  for subagents (`agent_tools.deliver_to_user`, `skills.install_skill`, `skills.create_skill`,
  `skills.run_skill_agent`), but leaves the other 5 — `terminal.terminal`,
  `codeinterpreter.deep_analysis`, `skills.run_skill_script`, `mcp_server.register_mcp_server`,
  and `agent_cron.create_cron_job` — reachable by subagents whenever the parent's
  allowed-plugins list includes them; this is not a theoretical unreachable combination.
- **Failure scenario:** A subagent (`agent_tools.run_subagents`) fetches attacker-controlled
  content via an untrusted-content plugin (e.g. a web-search/fetch plugin with
  `returns_untrusted_content=True`), then — following an injected instruction inside that
  content — calls `terminal.terminal` or `skills.run_skill_script` within the same subagent
  loop. The model-facing envelope still marks the earlier content as untrusted (part 1 still
  works), but the operator-facing "Dangerous tool ... called ... after untrusted content"
  warning log — the signal an operator or alerting pipeline would use to notice this exact
  pattern — never fires, because this code path never reaches
  `bot/openai_tool_handler.py:1445-1452`. The main chat loop has this coverage; the subagent
  loop silently does not.
- **Also unverified:** no test exercises this combination — grepped
  `tests/test_agent_tools_plugin.py` and `tests/test_openai_helper_tool_calls.py` for
  `dangerous`/`tainted` near `subagent`, zero hits. `T08-review.md` (both rounds) covers the
  same-batch edge case and idempotency for the main path in depth but never examines whether
  the subagent path reaches the check at all.
- **Suggested fix:** either call `_tainted_plugin_ids`-equivalent logic (or a shared helper) from
  `_call_subagent_tool`/`_run_subagent_completion_loop` before dispatching a
  `DANGEROUS_TOOL_NAMES` tool, mirroring the check already in
  `bot/openai_tool_handler.py:1445-1452`, or explicitly document (plan/AGENTS.md) that
  subagent-originated dangerous-tool calls are a known blind spot for this telemetry.

### Summary

- ERROR: 0
- WARNING: 1 (new)
- NIT: 0

All four round-1 WARNINGs confirmed fixed. One new WARNING found on fresh pass (subagent
taint-logging coverage gap). DNS-rebinding/redirect/IPv6 SSRF checks and symlink-race path
checks were re-verified by reading code directly and found either solid or already
documented/accepted — not re-raised.

## Раунд 3

### Round-2 WARNING verified fixed

Read the fix directly on disk rather than trusting the round-2 text, and cross-checked every
sub-claim from the round-3 task brief:

- **Present and wired up.** `bot/plugins/agent_tools.py:19` imports
  `DANGEROUS_TOOL_NAMES, _tainted_plugin_ids` straight from `bot.openai_tool_handler` (no
  duplicated constant/copy). `_run_subagent_completion_loop` (`bot/plugins/agent_tools.py:3356`)
  now carries a `run_tools_used: tuple[str, ...] = ()` local (`:3380`, comment `:3376-3379`
  explicitly notes it exists so concurrent subagents gathered in `_run_subagents`
  (`:3019`, `asyncio.gather`) never share taint state) and, inside the per-round tool-call loop
  (`:3402-3412`), logs `logging.warning("Dangerous tool %s called (subagent) chat_id=%s
  user_id=%s after untrusted content from plugins=%s", ...)` when `call_name in
  DANGEROUS_TOOL_NAMES` and `_tainted_plugin_ids(helper, run_tools_used)` is non-empty.
  `run_tools_used` itself is only appended to at `:3475-3478`, after the current round's tool
  responses are already computed — same "only rounds 1..N-1, never the current batch" semantics
  `_tainted_plugin_ids`'s own docstring describes (`bot/openai_tool_handler.py:316-321`).
- **Per-run state, not shared.** `run_tools_used` is a plain local variable inside an `async def`
  invoked once per subagent (via `_execute_single_subagent` → `_run_subagent_completion_loop`,
  `:3299`); each concurrently-gathered subagent coroutine gets its own closure — verified by
  reading the function signature and call site, not just trusting the comment.
- **Canonical vs model-mangled names — verified as a real, not theoretical, concern, and
  confirmed correctly handled.** `bot/plugin_manager.py:36`
  (`MODEL_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")`, no dot) plus
  `to_model_function_name` (`:405-432`) prove every dotted tool name (e.g. `terminal.terminal`)
  really is rewritten to an underscore form (`terminal_terminal`) before it reaches the model —
  confirmed independently via the unrelated pre-existing test
  `tests/test_agent_tools_plugin.py:455`
  (`pm.to_canonical_function_name("agent_tools_manage_plan_tasks") ==
  "agent_tools.manage_plan_tasks"` on a real `PluginManager`). The subagent loop's
  `call_name = call.get("name") or ""` (`agent_tools.py:3403`) is fed by `_extract_tool_calls`
  (`:3591-3620`), which converts the raw model-returned `model_name` to canonical form via
  `plugin_manager.to_canonical_function_name(model_name)` (`:3608-3613`) before storing it as
  `"name"` — the same canonicalization `_call_subagent_tool` re-applies defensively before
  dispatch (`:3656-3659`). Both `_extract_tool_calls` and `_call_subagent_tool` are **unchanged
  since HEAD** (`git diff HEAD -- bot/plugins/agent_tools.py` shows no hunk touching them) — this
  canonicalization plumbing pre-dates T08/T09 and the round-2 fix correctly plugs into it, using
  the already-canonical `call["name"]` exactly like `openai_tool_handler.handle_function_call`
  does with its own `tool_name = call["name"]` (`bot/openai_tool_handler.py:1408`, taint check at
  `:1445`). Not a bug.
  - **Caveat (not re-raised as a finding):** both the new subagent tests and the pre-existing
    main-loop tests (`test_dangerous_tool_after_untrusted_plugin_logs_warning`,
    `tests/test_openai_helper_tool_calls.py:2305`) construct their fake plugin managers
    (`DangerousAfterUntrustedPluginManager` at `tests/test_agent_tools_plugin.py:389`,
    `DummyPluginManager` at `tests/test_openai_helper_tool_calls.py:231`) without a
    `to_canonical_function_name` method, so `hasattr(...)` is `False` and `_extract_tool_calls`
    falls back to using `model_name` verbatim — the tests pass an already-canonical dotted name
    (`"terminal.terminal"`) as if it were what the model sent, so neither test suite actually
    exercises a mangled-name (`terminal_terminal`) round-trip through the `DANGEROUS_TOOL_NAMES`
    check. Not flagged as a new finding: it's the identical test-construction pattern the
    main-loop tests already use (accepted in T08-review, not re-raised there), production
    correctness was verified by reading the canonicalization code directly rather than relying on
    the tests, and singling out the subagent copy of an already-accepted systemic test-design
    choice would be inconsistent.
- **PII-safe fields.** The log call's format args are `call_name` (a tool name, e.g.
  `"terminal.terminal"`), `kwargs.get("chat_id")`, `kwargs.get("user_id")` (numeric identifiers,
  same shape logged pervasively elsewhere in this codebase and in the main-loop's own equivalent
  warning at `openai_tool_handler.py:1448-1452`), and `sorted(tainted)` (plugin id strings, e.g.
  `["skills"]`). No message content, tool arguments, or user text is interpolated. Matches the
  main-loop warning's field shape exactly.
- **Tool still executes.** Read `:3404-3443` end to end: the `DANGEROUS_TOOL_NAMES`/taint block
  only logs — it does not set `tool_responses[index]`, does not `continue`, and does not remove
  the call from consideration. The call falls through to the fingerprint/repeat check and (absent
  a stuck-loop condition) is appended to `pending_calls` (`:3443`) and dispatched via
  `_call_subagent_tool` in the `asyncio.gather` at `:3445-3455`. Confirmed by test assertion, not
  just by reading: `test_run_subagents_logs_dangerous_tool_warning_after_untrusted_content`
  (`tests/test_agent_tools_plugin.py:1468`) asserts
  `helper.plugin_manager.calls == [("skills.list_skills", ...), ("terminal.terminal", ...)]` —
  the dangerous tool actually reached `plugin_manager.call_function`, not just the log line.
- **Tests are meaningful, not superficial.** Both new tests
  (`test_run_subagents_logs_dangerous_tool_warning_after_untrusted_content:1468` and
  `test_run_subagents_no_dangerous_tool_warning_without_untrusted_content:1507`) assert on the
  actual dispatched-call list (`helper.plugin_manager.calls`) *and* on `caplog.text` presence vs.
  absence, using two different plugin-manager fixtures where only one marks `skills` as
  `returns_untrusted_content=True` — they distinguish "warning fires" from "warning doesn't
  fire" via a real behavioral difference (an untrusted-content call happened first or didn't),
  not a mocked-out log call. Field values are checked too (`"chat_id=10"`, `"user_id=42"`,
  `"plugins=['skills']"` at `:1502-1504`), not just substring presence of the word "Dangerous".

The round-2 WARNING is fixed as designed. No re-raise.

### Fresh pass — subagent hunks since round 2

Re-read every subagent-touching hunk in `git diff HEAD -- bot/plugins/agent_tools.py` (`_run_subagents`,
`_run_subagent_completion_loop`, `_call_subagent_tool`, `_subagent_tools`, `_extract_tool_calls`) end
to end, not just the taint-warning hunk covered above.

### NIT — dangerous-tool warning can log an attempt that the stuck-loop guard then blocks from dispatch

- **File:** `bot/plugins/agent_tools.py:3402-3443`.
- **Why:** The `if call_name in DANGEROUS_TOOL_NAMES:` taint-check/log block (`:3404-3412`) runs
  unconditionally for every call in the batch, before the fingerprint/repeat-loop guard
  (`:3413-3442`) that can reject a call as `stuck_loop: true` (identical call repeated
  `SUBAGENT_CONSECUTIVE_REPEAT_LIMIT`/`SUBAGENT_TOTAL_REPEAT_LIMIT` times) and synthesize an
  error response for it without ever appending it to `pending_calls` — i.e. without ever
  reaching `_call_subagent_tool`/`plugin_manager.call_function`. In the rare case where a
  dangerous tool call is itself the one being repeat-blocked, the operator-facing "Dangerous
  tool ... called (subagent) ..." warning still fires even though that specific call never
  dispatches. The main chat loop has no equivalent repeat-loop guard at its taint-check point
  (`openai_tool_handler.py:1445-1452` runs during "preparing", with no stuck-loop short-circuit
  in that path), so this ordering nuance is specific to the subagent completion loop.
- **Failure scenario:** Purely a telemetry-accuracy nit, not a security gap — the warning already
  errs in the safe direction (occasionally over-logs an attempt rather than under-logging one).
  An operator reading logs could see a "Dangerous tool called" line for a call that was actually
  never executed, and might momentarily misjudge that a tool ran when it didn't; cross-checking
  against `plugin_manager.calls`/downstream tool-result messages would clarify it did not.
- **Suggested fix (optional, low priority):** move the `DANGEROUS_TOOL_NAMES` check after the
  `repeated`/`continue` branch (i.e. only warn for calls actually added to `pending_calls`), or
  leave as-is given the harmless-direction bias — not worth a WARNING.

Also checked and **not** re-reviewed (already covered in rounds 1-2 and unrelated to the
round-2 taint fix): the `_normalize_delivery_artifacts` hunk that swaps the old bespoke
`_allowed_artifact_roots` check for `is_deliverable(resolved, scope=scope,
storage_root=storage_root)` (diff around `agent_tools.py:2798-2943`) delegates to
`artifact_paths.is_deliverable`, which rounds 1 and 2 already read in full and tested
(symlink-resolution, `request_started_at` dead-parameter check, etc.) — the call-site swap
itself introduces no new logic to review. Non-subagent hunks in the same file diff
(`_dynamic_insert_index`, `_task_public_view` extraction, `_clear_ask_user_markup` dedup,
`chat_id = message.chat_id` in the `/agent` command handler, `_PLAN_RULE_TEXT` wording, mypy
`cast`/type-annotation additions) are out of this round's scope (not subagent-loop, not
security-relevant) and were not reviewed here.

Targeted tests: `tests/test_agent_tools_plugin.py tests/test_pii_safe_logging.py
tests/test_no_hardcoded_plugin_refs.py` — 97 passed. `ruff check bot/plugins/agent_tools.py` —
clean.

### Summary

- ERROR: 0
- WARNING: 0 (round-2's 1 WARNING confirmed fixed, not re-raised)
- NIT: 1 (new — dangerous-tool warning can fire for a stuck-loop-rejected call in the subagent
  loop; optional fix, harmless-direction over-logging)
