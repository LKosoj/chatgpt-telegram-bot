# T09. Промпты в коде — implementation plan

Owner: developer (Sonnet) implementing T09, wave 4. File ownership per master plan
(`docs/improvement_2026-09-25/00-master-plan.md:311-338`):
`bot/openai_helper.py`, `bot/plugins/skills.py`, `bot/plugins/agent_tools.py`,
`bot/plugins/hindsight_memory.py`, `bot/chat_modes.yml` (only the `skills_agent`
"Function … returned" rule), and the test files listed under "Тесты" below.
**Never touch `get_spec()`, tool names/params, or `tools:` lists.**

## 0. Blocking dependency — verify before starting

`bot/openai_helper.py:3322-3326` (`__add_function_call_to_history`) still contains the
legacy fallback:

```python
if model_to_use in self.get_model_choices():
    function_result = f"Function {function_name} returned: {content}"
    self.conversations[state_key].append({"role": "assistant", "content": function_result})
```

This is the path T08 is supposed to remove (wrap in `<untrusted_tool_output>`, flip to
`role: user`). **As of this plan being written, T08 has not run yet** — this file is not
in T09's ownership, so do not touch it, but **step 5 below (deleting the skills_agent
"Function … returned" rule) must not be applied until T08 has actually landed the change**.
Before doing step 5, re-check `bot/openai_helper.py:3322-3326`: if the literal string
`f"Function {function_name} returned: {content}"` is still there, skip step 5 and note
it as still-blocked in your report instead of silently applying it.

## 1. Router prompt — `OpenAIHelper._build_auto_chat_mode_prompt` (`bot/openai_helper.py:4105-4135`)

Current text (verified verbatim):

```python
        return f"""Определи режим работы для сообщения и верни только ключ режима.

{priority_block}Остальные правила выбора:
1. Если задача простая и может быть решена одним коротким ответом или одним очевидным инструментом, выбирай наиболее простой подходящий режим.
2. Если задача сложная, открытая или ожидаемо требует больше двух шагов, выбирай skills_agent, если такой режим есть в списке доступных режимов.
3. Сложная задача - это задача, где нужно построить план и выполнить больше двух связанных шагов: последовательно использовать инструменты, обработать файлы или артефакты, запустить локальные scripts, проверить результат, исправить ошибки или уточнить требования у пользователя.
4. Не выбирай skills_agent по отдельным словам. Выбирай его по структуре задачи: больше двух шагов, неопределенный маршрут, необходимость orchestration или проверки промежуточных результатов.
5. Если сложность не нужна, не выбирай skills_agent.
6. Если ни один режим не подходит, верни assistant.

Сообщение: ^{query}^
Доступные режимы: ^{self.get_all_modes()}^"""
```

Findings vs. master-plan step 1:
- "Точка", "ПЕРЕБИВАЕТ", `writing_assistant` — **not present anywhere in the current
  string** (verified: `grep`-equivalent scan of the file found zero hits). Nothing to
  remove; do not add a no-op edit.
- Order — currently `query` comes **before** `Доступные режимы`, and neither is a
  "постоянные правила → список режимов → запрос последним" order. Needs the swap below.
- Rule 4 already reads as a soft heuristic ("не по отдельным словам… по структуре
  задачи"), not the rigid "matched one term → skip the rest" framing the master plan
  describes wanting weakened. It is already close to the target, but does not use the
  literal words the plan asks for ("сильный сигнал"). Apply the minimal reword below so
  the master-plan bullet is demonstrably addressed without regressing the existing
  (already-good) semantics.

**Edit — reorder the tail (swap `Сообщение` and `Доступные режимы`, query last):**

```python
Доступные режимы: ^{self.get_all_modes()}^
Сообщение: ^{query}^"""
```

**Edit — reword rule 4 only (minimal, keep same meaning, add "сильный сигнал" wording):**

```python
4. Совпадение одного отдельного слова с skills_agent — не сильный сигнал для его выбора. Выбирай его по структуре задачи: больше двух шагов, неопределенный маршрут, необходимость orchestration или проверки промежуточных результатов.
```

Rules 1, 2, 3, 5, 6 and the `priority_block` handling stay untouched.

Router call site / answer parser (context only, not edited — confirms the model's raw
text answer is the whole contract, so reordering the prompt body is safe):
`bot/openai_helper.py:1214-1216` calls `_build_auto_chat_mode_prompt`;
`bot/openai_helper.py:1235-1239` reads `_required_choice_message_text(...)`, does
`.strip().lower()`, and looks up `chat_modes_registry.get_mode_by_key(mode_key)` — a
plain key match, no regex, no position sensitivity to the prompt text.

## 2. `ask()` prompt (`bot/openai_helper.py:793,795`)

Current (verified verbatim):

```python
793:            add_prompt1 = f" Текущая дата и время: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
794:            if assistant_prompt is None:
795:                assistant_prompt = "Ты помошник, который отвечает на вопросы пользователя. Ты должен использовать все свои знания и навыки для того, чтобы помочь пользователю. " + add_prompt1
```

Edits:
- Line 793: change the strftime format from the compact `%Y%m%d%H%M%S`
  (`20260925143000`) to a human-readable one: `%Y-%m-%d %H:%M:%S UTC`
  (`2026-09-25 14:30:00 UTC`). No other project convention for a full prompt-facing
  datetime string was found in this file to match instead (the only other `strftime`
  calls, `bot/openai_helper.py:3958/3964/3968`, format a short `dd.mm` session-name
  fallback and are unrelated) — this is a free choice, keep it readable and keep UTC
  explicit since the value is `datetime.timezone.utc`.
- Line 795: fix the typo `помошник` → `помощник`. No other wording on this line changes.

## 3. `_PLAN_RULE_TEXT` (`bot/plugins/agent_tools.py:184-192`)

Current (verified verbatim):

```python
_PLAN_RULE_TEXT = (
    "Если для выполнения запроса понадобятся 3+ tool-вызова или последовательная координация "
    "шагов (несколько разных тулов, проверка промежуточных результатов, исправление ошибок) — "
    "ПЕРВЫМ ходом вызови agent_tools.manage_plan_tasks с goal/success_criteria/verification и "
    "зафиксируй план. Для тривиальных запросов (быстрый ответ из знаний, один очевидный тул, "
    "перевод/время/факт) план НЕ создавай. "
    "Перед каждым нетривиальным tool-вызовом одной строкой укажи намерение: что вызываешь и зачем. "
    "После результата тула одной строкой оцени, продвинул ли он к цели и что делать дальше."
)
```

Master-plan step 2 asks for: threshold "больше двух шагов" instead of "3+ tool-вызова";
drop the "state intent / assess in one line" sentences; keep the `manage_plan_tasks`
marker string; the verify-trigger must not require a plain-text phrase that conflicts
with skills_agent's rule (`chat_modes.yml:1709`: "После вызова tools не пишите
промежуточный plain-text статус... В следующем ходе выберите ровно одно действие").

**New text:**

```python
_PLAN_RULE_TEXT = (
    "Если для выполнения запроса понадобится больше двух шагов или последовательная "
    "координация (несколько разных тулов, проверка промежуточных результатов, "
    "исправление ошибок) — ПЕРВЫМ ходом вызови agent_tools.manage_plan_tasks с "
    "goal/success_criteria/verification и зафиксируй план. Для тривиальных запросов "
    "(быстрый ответ из знаний, один очевидный тул, перевод/время/факт) план НЕ создавай."
)
```

`_PLAN_RULE_MARKER = "[plan-rule-v1] "` (line 182) and the literal substring
`"manage_plan_tasks"` inside the rule text — **do not change**. Both are load-bearing:
- `_PLAN_RULE_MARKER` is the idempotency prefix checked at `agent_tools.py:397`.
- `"manage_plan_tasks" in content` is checked at `agent_tools.py:412` to detect when a
  mode's own prompt already covers the rule (`mode_prompt_covers_rule`).
- `bot/openai_helper.py:186` (`INFORMATION_ONLY_TOOLS`) and `:1826`
  (`name.endswith("manage_plan_tasks")`, in `_build_force_planner_tool_choice`) are a
  **separate** gate mechanism, keyed off the canonical/model tool name, not off this
  prompt string — unaffected by wording changes as long as the literal function name
  string stays `manage_plan_tasks`.

Verify-trigger text (`_verify_message_body`, `agent_tools.py:2156-2163`) currently says
"одной фразой подтверди" (state confirmation in one phrase) — this is the plain-text
status phrasing that conflicts with skills_agent's "no plain-text status between tool
calls" rule. Reword to require the confirmation to happen **through the tool call
itself** rather than as prose:

```python
@staticmethod
def _verify_message_body(task_id: str) -> str:
    return (
        f"{_VERIFY_TRIGGER_MARKER}task {task_id} помечена completed. "
        "Перед следующим действием проверь через manage_plan_tasks, что результат "
        "закрыл success_criteria/verification из контракта плана. Если нет — верни "
        "задачу в работу через manage_plan_tasks(action=update)."
    )
```

`_REPLAN_TRIGGER_MARKER`/`_VERIFY_TRIGGER_MARKER` values (lines 178, 180) and
`_replan_message_body` (lines 2142-2153) — leave text as-is; it already only asks for a
tool call (`manage_plan_tasks(action=update)`), no plain-text status.

## 4. Insertion positions (master-plan step 4)

Bucket split, per master plan wording — stable/start:
"промпт режима, язык, статичный каталог skills, базовая память"; dynamic/end:
"чекпоинт плана, триггеры re-plan/verify, динамические воспоминания hindsight, список
активных skills/скриптов".

**Common helper (duplicate verbatim in each of the three files — matches the existing
local pattern where each plugin already independently computes its own `insert_at`
loop instead of sharing one; a cross-file shared helper would have to live in
`bot/plugins/hooks.py`, which is outside T09's file ownership, so do not add it there):**

```python
def _dynamic_insert_index(messages: List[Dict[str, Any]]) -> int:
    """Index for a per-turn dynamic system message: right before the trailing user
    message, or at the very end when something else (assistant/tool messages from an
    in-progress tool round) already follows the last user turn."""
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        return len(messages) - 1
    return len(messages)
```

Why this is correct for "how to find the last user message during tool rounds":
`_apply_before_chat_request_mutators` (`bot/openai_helper.py:2691-2805`) is called once
per round-trip, always right before the next `chat_completion` call
(`bot/openai_tool_handler.py:974`, `:1053`, `:1674`). By the time it runs, any tool
results from the previous round are already appended to `self.conversations[state_key]`.
So the tail of `messages` is `role == "user"` only on a genuinely fresh turn (nothing
after the user's message yet); during an in-progress tool round the tail is `role ==
"assistant"` (just-made tool_calls) or `role == "tool"` (results already appended) —
either way `messages[-1]["role"] != "user"`, so the helper appends at the end instead of
splicing before a stale, no-longer-last user message.

### 4a. `bot/plugins/agent_tools.py` — `on_before_chat_request` (lines 349-482)

Keep unchanged: the leading-system-cluster scan (lines 447-453) and the plan-rule
injection at `insert_at` (lines 455-460) — the plan rule is constant text, it belongs to
the stable/start bucket like a mode-prompt addendum.

Change: working checkpoint, re-plan trigger, and verify trigger currently insert at
`insert_at + injected` (near the start, right after the plan rule). Move all three to
`_dynamic_insert_index`, computed once after the plan-rule insert, then incremented per
insert to preserve their existing relative order (checkpoint → re-plan → verify):

```python
        new_messages = list(messages)
        insert_at = 0
        for msg in new_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                insert_at += 1
            else:
                break
        if inject_plan_rule:
            new_messages.insert(insert_at, {
                "role": "system",
                "content": _PLAN_RULE_MARKER + _PLAN_RULE_TEXT,
            })

        dyn_idx = _dynamic_insert_index(new_messages)
        if checkpoint:
            new_messages.insert(dyn_idx, {
                "role": "system",
                "content": _WORKING_CHECKPOINT_MARKER + self._format_working_checkpoint(checkpoint),
            })
            dyn_idx += 1
        if pending:
            reason = str(pending.get("reason") or "errors")
            task_id = str(pending.get("task_id") or "")
            body = self._replan_message_body(reason, task_id)
            new_messages.insert(dyn_idx, {"role": "system", "content": body})
            dyn_idx += 1
        if pending_verify:
            task_id = str(pending_verify.get("task_id") or "")
            new_messages.insert(dyn_idx, {
                "role": "system",
                "content": self._verify_message_body(task_id),
            })
        return new_messages
```

(Drop the `injected` counter — it's no longer needed since the plan rule no longer
shares an index with the other three.)

### 4b. `bot/plugins/hindsight_memory.py` — `on_before_chat_request` (lines 2435-2507)

Keep unchanged: baseline recall (`HINDSIGHT_CONTEXT_PROMPT`, lines 2478-2491) stays
inserted at `insert_at` — it is the "базовая память" that belongs in the stable/start
bucket (also note it gets persisted into `self.conversations` by the caller,
`bot/openai_helper.py:2779-2789`, so its start-of-history position is durable, not just
per-request).

Change: dynamic recall (`HINDSIGHT_DYNAMIC_CONTEXT_PROMPT`, lines 2493-2505) currently
inserts at `insert_at` too (right after the baseline message, still near the start).
Move it to `_dynamic_insert_index`, computed on the current `new_messages` (after any
baseline insert):

```python
        if needs_dynamic_recall and should_retrieve:
            memory = await self._recall_memory_text(
                user_id,
                query,
                max_tokens=int(self.config.get('hindsight_dynamic_recall_max_tokens', 1024)),
            )
            if memory:
                new_messages.insert(_dynamic_insert_index(new_messages), {
                    "role": "system",
                    "content": HINDSIGHT_DYNAMIC_CONTEXT_PROMPT.format(memory=memory),
                })
                changed = True
                logger.info("Hindsight recalled dynamic memory for bank %s", self.bank_id_for(user_id))
```

(The `insert_at` variable is still needed for the baseline branch; only the dynamic
branch's insert target changes.)

### 4c. `bot/plugins/skills.py` — split the catalog, then position each half

`_build_session_skills_catalog` (lines 272-316) currently returns one string combining
a static "available skills" section and a dynamic "active skills in this session"
section, inserted as a single message at `insert_at`. Split into two functions,
preserving each section's exact existing text and the `"\n\n".join(...)` spacing
convention:

```python
    def _build_static_skills_catalog(self, disabled_skills: set[str]) -> str:
        catalog_lines: List[str] = []
        for skill_id, info in self.available_skills.items():
            if skill_id in disabled_skills:
                continue
            desc = (info.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 240:
                desc = desc[:240].rstrip() + "..."
            catalog_lines.append(f"- {skill_id}: {desc}" if desc else f"- {skill_id}")
        if not catalog_lines:
            return ""
        return "\n\n".join([
            "Доступные локальные skills (id: description). Активируйте через "
            "skills.activate_skill, читайте через skills.get_skill:",
            "\n".join(catalog_lines),
        ])

    def _build_active_skills_catalog(self, active_scope_state: Dict[str, Any]) -> str:
        active_lines: List[str] = []
        for skill_id in sorted(active_scope_state.keys()):
            info = self.available_skills.get(skill_id)
            if not info:
                continue
            scripts = info.get("scripts") or []
            if not scripts:
                continue
            skill_root = Path(info.get("path") or "")
            scripts_dir = skill_root / "scripts"
            active_lines.append(f"{skill_id}:")
            for script_name in scripts:
                abs_path = (scripts_dir / script_name).as_posix()
                active_lines.append(f"  - {script_name}  →  {abs_path}")
        if not active_lines:
            return ""
        return "\n\n".join([
            "Активные skills в этой сессии и их scripts. Запускайте через "
            "skills.run_skill_script(skill_name=..., script_name=...); абсолютные "
            "пути даны как fallback на случай, если требуется terminal.terminal:",
            "\n".join(active_lines),
        ])
```

Remove `_build_session_skills_catalog` (fully replaced by the two functions above — no
other caller references it, only `on_before_chat_request`).

`on_before_chat_request` (lines 209-252), from the `chat_id = ...` line onward:

```python
        chat_id = getattr(payload, "chat_id", None)
        user_id = getattr(payload, "user_id", None)
        disabled_skills = self._disabled_skills_for_user(getattr(self, "openai", None), user_id)
        scope = compute_scope_key(chat_id, user_id) if chat_id is not None else None
        active_scope_state = self.active_skills.get(scope, {}) if scope is not None else {}

        static_text = self._build_static_skills_catalog(disabled_skills)
        active_text = self._build_active_skills_catalog(active_scope_state)
        if not static_text and not active_text:
            return None

        new_messages = list(messages)
        insert_at = 0
        for msg in new_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                insert_at += 1
            else:
                break
        if static_text:
            new_messages.insert(insert_at, {"role": "system", "content": static_text})
        if active_text:
            new_messages.insert(_dynamic_insert_index(new_messages), {
                "role": "system",
                "content": active_text,
            })
        return new_messages
```

Everything above this block (the `if not messages`, `first_system`, `_is_skills_agent_mode`,
`if not self.available_skills` guards, lines 221-232) stays unchanged.

## 5. `bot/chat_modes.yml` — skills_agent "Function … returned" rule

**Conditional on T08 being actually landed — see section 0.** Once confirmed, delete
line 1712:

```
    6. Никогда не выводите сырые результаты tools вида "Function ... returned: ..."; используйте их только как внутренние данные для следующего действия или нормального ответа.
```

It is the last item in the "Формат ответа" list (items 1-6, `chat_modes.yml:1706-1712`),
so deleting it requires no renumbering of the remaining items (1-5 stay as-is). Do not
touch `prompt_markers` (`chat_modes.yml:1717-1719`, `"локальные skills"` / `"skills."`)
or any other line in this mode.

## Tests

### Update (position/text now differs)

- `tests/test_openai_helper_tool_calls.py:1485-1505`
  (`test_auto_chat_mode_prompt_routes_by_complexity_not_keywords`): assert the new tail
  order — `prompt.index("Доступные режимы") < prompt.index("Сообщение")` (or equivalent),
  and update the rule-4 substring check from `"Не выбирай skills_agent по отдельным
  словам"` to the new wording (`"сильный сигнал"` or the exact new sentence). Keep the
  existing "больше двух шагов" assertion (unaffected).
- `tests/test_skills_prompt_fragment.py:96-139`
  (`test_build_auto_chat_mode_prompt_includes_fragments_from_collector`,
  `test_build_auto_chat_mode_prompt_works_without_fragments`): both assert
  `"Остальные правила выбора" in prompt` — unaffected by the reorder, no change expected,
  but re-run to confirm.
- `tests/test_agent_tools_plan_rule_mutator.py`: no test currently pins checkpoint/
  re-plan/verify position relative to a longer history (all use 2-message
  system+user fixtures, where "start" and "before last user" coincide) — re-run as a
  regression check; expect all to still pass unchanged. If `_PLAN_RULE_TEXT` wording
  assertions exist elsewhere (only `_PLAN_RULE_TEXT in new[1]["content"]` at line 53,
  which compares against the imported constant, not a literal string) — unaffected by
  rewording since it imports the constant.
- `tests/test_hindsight_mutator.py`: same — no test pins dynamic-recall position
  relative to history; existing 2-message fixtures keep passing unchanged. Re-run as
  regression.
- `tests/test_skills_plugin.py:2464-2482`
  (`test_on_before_chat_request_lists_scripts_for_active_skills`): **must change.**
  With the split, `new_messages` becomes `[sys, static, active, user]` instead of
  `[sys, combined, user]`. Update:
  ```python
  content = new_messages[2]["content"]  # was new_messages[1]
  assert "Активные skills в этой сессии" in content
  assert "demo:" in content
  assert (tmp_path / "skills" / "demo" / "scripts" / "echo.py").as_posix() in content
  assert "Активные skills в этой сессии" not in new_messages[1]["content"]  # static-only
  ```
  For the `other_payload` branch (no active skills in that scope), `other_messages` will
  only contain the static message (`len == len(messages) + 1`), so
  `other_messages[1]["content"]` remains correct as-is.
- `tests/test_skills_plugin.py:2431-2448`
  (`test_on_before_chat_request_injects_catalog_in_skills_agent`): no active skills in
  this scope → only `static_text` is emitted → `new_messages[1]` is still the injected
  message, `len == len(messages) + 1` still holds. Re-run to confirm, no edit expected.
- `tests/test_skills_plugin.py:2494-2504`
  (`test_on_before_chat_request_does_not_mutate_input`): same as above, no active skill
  activated in this test → unaffected, re-run to confirm.

### New tests (position-under-history / tool-round correctness — one representative
pair per plugin is enough; do not over-generalize into a shared test helper across
files you don't own)

For each of `agent_tools.py`, `hindsight_memory.py`, `skills.py`, add two cases to the
matching test file:

1. **Dynamic content lands right before the trailing user message when earlier history
   exists.** Build `messages = [system, user("first"), assistant("first reply"),
   user("second")]`, trigger the dynamic condition (checkpoint/pending trigger for
   agent_tools; `hindsight_dynamic_recall=True` with a baseline marker already present
   for hindsight; an active skill for skills), call `on_before_chat_request`, and assert
   the injected dynamic message is at `new_messages[-2]` and `new_messages[-1]` is still
   the `user("second")` message unchanged.
2. **Dynamic content appends at the end when a tool round is in progress.** Build
   `messages = [system, user("q"), {"role": "assistant", "content": None, "tool_calls":
   [...]}, {"role": "tool", "tool_call_id": "1", "content": "result"}]`, trigger the same
   dynamic condition, call `on_before_chat_request`, and assert the injected message is
   `new_messages[-1]` and the assistant/tool pair stays adjacent and unmodified
   (`new_messages[-3]` is the assistant tool_calls message, `new_messages[-2]` is the
   tool result, both byte-identical to the input).

### New prefix-stability test (master-plan acceptance criterion)

Add to `tests/test_openai_helper_tool_calls.py` (reuses the existing `DummyPluginManager`
/ `_make_helper` harness already used by `tests/test_skills_prompt_fragment.py:98`, so no
new fixtures needed):

```python
@pytest.mark.asyncio
async def test_before_chat_request_prefix_stable_across_requests_with_dynamic_change():
    """The stable prefix (mode prompt, language instruction, static skills catalog,
    baseline memory) must be byte-identical across two consecutive requests even when
    only a dynamic part (e.g. hindsight dynamic recall, a pending re-plan trigger)
    differs between them."""
    pm = DummyPluginManager({})
    helper = _make_helper(pm)
    helper.conversations[helper._chat_state_key(1)] = [
        {"role": "system", "content": "mode prompt", "mode_key": "assistant"},
        {"role": "user", "content": "hello"},
    ]

    first = await helper._apply_before_chat_request_mutators(
        chat_id=1, user_id=1, session_id=None, request_id=None, persist=False,
    )
    # Simulate a dynamic-only change: e.g. append a new user turn (or, for a plugin
    # under test, flip whatever internal state produces its dynamic message) so the
    # only difference between the two calls is a per-turn dynamic value, not the
    # stable prefix.
    helper.conversations[helper._chat_state_key(1)].append(
        {"role": "assistant", "content": "hi"}
    )
    helper.conversations[helper._chat_state_key(1)].append(
        {"role": "user", "content": "second question"}
    )
    second = await helper._apply_before_chat_request_mutators(
        chat_id=1, user_id=1, session_id=None, request_id=None, persist=False,
    )

    stable_len = min(len(first), len(second))
    # Walk from the start: every leading message that is NOT a per-turn dynamic
    # marker must match byte-for-byte between the two calls.
    for a, b in zip(first[:stable_len], second[:stable_len]):
        if a == b:
            continue
        # First divergence marks the start of the dynamic tail — stop comparing.
        break
    else:
        return
    # At minimum, the mode prompt (index 0) and language instruction (index 1) must
    # be identical regardless of where the dynamic tail starts.
    assert first[0] == second[0]
    assert first[1] == second[1]
```

Note: the exact assertion shape depends on which plugins are registered in
`DummyPluginManager({})` for this test (likely none, since it's an empty plugin map) —
adjust to actually exercise a real dynamic mutator (e.g. instantiate a real
`AgentToolsPlugin`/`HindsightMemoryPlugin` with a pending trigger/dynamic-recall state
and register it on `pm`, the way `tests/test_skills_prompt_fragment.py` does for
`SkillsPlugin`) so the test is not vacuous. Pick whichever plugin is simplest to wire up
with the existing fakes in this file.

## Acceptance commands

```
~/.venvs/ctb/bin/python -m pytest \
  tests/test_agent_tools_plan_rule_mutator.py \
  tests/test_hindsight_mutator.py \
  tests/test_skills_plugin.py \
  tests/test_skills_prompt_fragment.py \
  tests/test_openai_helper_tool_calls.py \
  -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m ruff check bot/openai_helper.py bot/plugins/agent_tools.py \
  bot/plugins/hindsight_memory.py bot/plugins/skills.py

python3 -m mypy bot/openai_helper.py bot/plugins/agent_tools.py \
  bot/plugins/hindsight_memory.py bot/plugins/skills.py \
  --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports
```

Also run the full suite once before declaring done (`~/.venvs/ctb/bin/python -m pytest
tests bot/tests -q --no-header -p no:cacheprovider`), since other in-flight tasks share
the tree and a full run is the only way to catch cross-file breakage from this change
(e.g. anything constructing a skills-catalog message and asserting its exact index).

## Risks

1. **T08 not yet landed** (section 0). Applying step 5 early would delete a still-true
   warning about a code path that still exists. Re-verify `bot/openai_helper.py:3322-3326`
   immediately before doing step 5; if still present, skip step 5 and say so in the
   report.
2. **Rule-4 rewording is a judgment call.** The current text already reads as a soft
   heuristic; the reword in section 1 is the minimal change that makes the master-plan's
   literal "сильный сигнал" wording traceable in a diff. If a reviewer considers the
   original wording already sufficient, this edit is low-risk to revert (one line).
3. **`_dynamic_insert_index` is duplicated 3×.** Deliberate — a shared version would
   belong in `bot/plugins/hooks.py`, which T09 does not own. Flag for a future cleanup
   task rather than reaching outside ownership.
4. **Multiple simultaneous dynamic inserts in agent_tools** (checkpoint + re-plan +
   verify all pending in the same turn) — order preserved as checkpoint → re-plan →
   verify by incrementing `dyn_idx` after each insert (matches current relative order,
   only the anchor position changes).
5. **Skills catalog split changes message count** whenever a session has active skills
   (`len(new_messages)` becomes `+2` instead of `+1`) — any other code reading
   `self.conversations` by fixed offset (not found in this investigation, but not
   exhaustively ruled out outside owned files) could be affected; the ownership rule
   means such breakage should be reported, not fixed, if discovered outside T09's files.
