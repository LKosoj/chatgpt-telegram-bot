---
name: find-skills
description: Helps users discover, install, and use bot skills when they ask questions like "how do I do X", "find a skill for X", "install a skill for X", "is there a skill that can...", or express interest in extending capabilities. Use this skill to route through the bot's `skills` plugin install flow.
---

# Find Skills

This skill helps discover, install, and use skills in the Telegram bot runtime.
The bot's source of truth for local skills is the `skills` plugin. New skills
are discovered through `skills.find_installable_skills` and installed through
`skills.install_skill`, which wraps the real `npx skills` CLI and syncs the
result into the bot's `SKILLS_DIR`.

## When to Use This Skill

Use this skill when the user:

- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Asks "can you do X" where X is a specialized capability
- Expresses interest in extending agent capabilities
- Asks to install, add, or enable a new skill
- Wants to search for tools, templates, or workflows
- Mentions they wish they had help with a specific domain (design, testing, deployment, etc.)

## Bot Rules

- Discover local skills with `skills.list_skills`, optionally with `refresh=true`.
- Inspect a candidate with `skills.get_skill`.
- Activate the chosen skill with `skills.activate_skill` before following its workflow.
- Search installable external skills with `skills.find_installable_skills`.
- Install a new skill with `skills.install_skill` only after explicit user confirmation for the exact package id.
- After installing, call `skills.list_skills` with `refresh=true`, then inspect and activate the installed skill if the user wants the task handled.
- Track work through `skills.update_skill_progress` when the skill has a multi-step flow.
- If a skill script must run, use `skills.run_skill_script` or `terminal.terminal`; never route active skill scripts through `codeinterpreter.deep_analysis`.
- If skill use fails because the skill instruction is unclear, start a background reflection subagent with `agent_tools.run_subagents` and `background=true`; have it call `skills.record_skill_reflection`.
- Do not claim that a skill is installed until it appears in `skills.list_skills` after refresh.
- Do not call `terminal.terminal` for normal skill search/install. Use the native `skills.find_installable_skills` and `skills.install_skill` tools first. Use terminal only as an operator fallback if the native install tool is unavailable or returns a clear infrastructure error.
- If `skills.install_skill` says installs are disabled or restricted, explain that the operator must enable `SKILLS_ALLOW_INSTALLS=true` and configure `SKILLS_INSTALL_ADMIN_USER_IDS`; do not pretend installation succeeded.
- Final user-facing answers in `skills_agent` mode must go through `agent_tools.deliver_to_user`.

## How to Help Users Find Skills

### Step 1: Understand What They Need

When a user asks for help with something, identify:

1. The domain (e.g., React, testing, design, deployment)
2. The specific task (e.g., writing tests, creating animations, reviewing PRs)
3. Whether this is a common enough task that a skill likely exists

### Step 2: List Local Skills

Call:

```json
{"refresh": true}
```

with `skills.list_skills`.

Compare the user's task against each returned skill's `id`, `name`,
`description`, and `scripts`.

### Step 3: Inspect Candidate Skills

For plausible matches, call `skills.get_skill` with the skill id/name. Read the
workflow and confirm it actually covers the user's task.

Do not infer capabilities from the skill name alone. If the body does not
support the task, say that the skill is not a fit.

### Step 4: Use or Present Local Options

If one skill clearly fits and the user asked for the task to be done, activate
it and proceed:

```json
{"skill_name": "<skill-id>", "initial_context": "<short task context>"}
```

with `skills.activate_skill`.

If multiple skills fit, present a short choice list:

```
I found two local skills that may fit:

- `frontend-design`: UI implementation and polish.
- `webapp-testing`: Playwright checks for local web apps.

Which one should I use?
```

Keep the list short. Prefer the best 1-3 matches.

### Step 5: Search Installable Skills

Use this step when no local skill fits, or when the user explicitly asks to find
or install a new skill.

Call `skills.find_installable_skills` with a short query:

```json
{"query": "playwright testing"}
```

Use the returned `results[].package` values exactly. These are package ids such
as:

```text
anthropics/skills@webapp-testing
```

Present at most the best 1-3 candidates and ask for explicit confirmation before
installation:

```
I found these installable skills:

- `anthropics/skills@webapp-testing`: browser testing workflows.
- `wshobson/agents@python-testing-patterns`: Python testing patterns.

Install `anthropics/skills@webapp-testing` into this bot?
```

Use `agent_tools.ask_telegram_user` for this confirmation in bot mode.

### Step 6: Install and Refresh

After the user confirms the exact package, call:

```json
{"package": "anthropics/skills@webapp-testing", "confirmed": true}
```

with `skills.install_skill`.

Then call:

```json
{"refresh": true}
```

with `skills.list_skills`.

Only after the installed skill appears in that refreshed list may you say it is
installed. Then inspect it with `skills.get_skill` and activate it if the user
wants to use it immediately.

If installation fails because installs are disabled or the caller is not
allowed, explain the concrete gate from the tool error. Do not retry through
terminal unless the user/operator explicitly asks for a manual fallback.

## Common Skill Categories

When searching, consider these common categories:

| Category        | Example Queries                          |
| --------------- | ---------------------------------------- |
| Web Development | react, nextjs, typescript, css, tailwind |
| Testing         | testing, jest, playwright, e2e           |
| DevOps          | deploy, docker, kubernetes, ci-cd        |
| Documentation   | docs, readme, changelog, api-docs        |
| Code Quality    | review, lint, refactor, best-practices   |
| Design          | ui, ux, design-system, accessibility     |
| Productivity    | workflow, automation, git                |

## Tips for Effective Searches

1. Use specific task words from the user's request.
2. Try synonyms mentally when matching descriptions: "deploy" vs. "deployment", "review" vs. "audit".
3. Prefer local skill metadata and `SKILL.md` content over memory or external assumptions.
4. If the user asks for a skill by exact id/name, inspect that skill directly.

## When No Local Skills Are Found

If no relevant skills exist:

1. Search installable skills with `skills.find_installable_skills`.
2. If installable candidates exist, ask whether to install one.
3. If the user declines or installation is disabled, offer to help with the task directly using general capabilities and available tools.

Example:

```
I checked the local skills and did not find one that covers "xyz".
I found `owner/repo@xyz-skill` as an installable candidate. Install it into
this bot?
```
