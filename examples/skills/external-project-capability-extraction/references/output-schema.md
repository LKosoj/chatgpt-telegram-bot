# Output Schema

Use this structure for project capability extraction.

## Capability Inventory

For each capability:

- Name
- Summary
- Trigger phrases
- Evidence
- Workflow
- Required references
- Optional scripts or agents
- Validation checks
- Known limitations
- Security boundaries
- Confidence

## Skill Draft

If a reusable skill is requested, draft:

```markdown
---
name: <skill-id>
description: <when to use>
allow_scripts: false
---

# <Skill Name>

## When To Use

## Do Not Use

## Required Inputs

## Workflow

## Output Format

## Verification
```

Keep copied source text short. Prefer summaries and evidence pointers.
