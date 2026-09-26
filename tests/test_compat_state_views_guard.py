"""T11 review Round 1 WARNING follow-up: pins the current count of accesses to the
ten compat-view properties (bot/openai_helper.py:2932-3020 -- conversations,
loaded_conversation_sessions, last_updated, _chat_request_models,
_chat_request_usage_split, _chat_request_extra_tokens, _gate_fired,
_last_summary_at, last_image_file_ids, _per_chat_locks; each a _FieldView over
ChatStateRegistry, see bot/conversation_state.py).

The master plan (00-master-plan.md step 6) asked that, if these dict-shaped
compat properties are kept, "code core code must not use them" -- i.e. new core
code should go through self._get_chat_states() / the registry, or (from outside
OpenAIHelper) the public session API (history_snapshot/load_session/
replace_system_message/evict/chat_state_scope), not the old dict-like names.
Nothing enforced that before this test.

Two files are intentionally exempt from "zero": bot/chat_run.py (carved out of
OpenAIHelper, still reads/writes per-chat state the old dict-like way via
`helper.<name>[state_key]`) and tests (not scanned here). bot/openai_helper.py
itself keeps ~126 `self.<name>` accesses in method bodies that were deliberately
left unchanged by T11 (T11-plan.md SS7 step 2: "method bodies need zero changes"),
and bot/openai_tool_handler.py keeps one read-only `helper.conversations.get(...)`
logging call (see tests/test_no_private_helper_access.py's module docstring).

This test does not require any of that to reach zero. It pins each file's current
count exactly (no slack) so any *new* access is a deliberate, reviewed addition --
not a silent regression. Lower a count in ALLOWED when a site is migrated to the
registry/public API; raise one only with a deliberate review.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# The ten compat-view properties defined in bot/openai_helper.py:2932-3020.
COMPAT_VIEW_NAMES = {
    "conversations",
    "loaded_conversation_sessions",
    "last_updated",
    "_chat_request_models",
    "_chat_request_usage_split",
    "_chat_request_extra_tokens",
    "_gate_fired",
    "_last_summary_at",
    "last_image_file_ids",
    "_per_chat_locks",
}

# (file_relpath, attr_name) -> exact current count of `self.<attr>` accesses
# (bot/openai_helper.py) / `helper.<attr>` accesses (bot/openai_tool_handler.py,
# bot/chat_run.py). Any (file, attr) pair not listed here is pinned at 0.
ALLOWED: Dict[Tuple[str, str], int] = {
    ("bot/openai_helper.py", "conversations"): 67,
    ("bot/openai_helper.py", "loaded_conversation_sessions"): 21,
    ("bot/openai_helper.py", "last_updated"): 9,
    ("bot/openai_helper.py", "_chat_request_models"): 4,
    ("bot/openai_helper.py", "_chat_request_usage_split"): 3,
    ("bot/openai_helper.py", "_chat_request_extra_tokens"): 10,
    ("bot/openai_helper.py", "_gate_fired"): 3,
    ("bot/openai_helper.py", "_last_summary_at"): 3,
    ("bot/openai_helper.py", "last_image_file_ids"): 3,

    ("bot/openai_tool_handler.py", "conversations"): 1,

    ("bot/chat_run.py", "_chat_request_models"): 5,
    ("bot/chat_run.py", "_chat_request_usage_split"): 4,
    ("bot/chat_run.py", "_chat_request_extra_tokens"): 2,
    ("bot/chat_run.py", "_gate_fired"): 1,
}

# (file_relpath, receiver expression to match, e.g. "self" or "helper")
FILES = [
    ("bot/openai_helper.py", "self"),
    ("bot/openai_tool_handler.py", "helper"),
    ("bot/chat_run.py", "helper"),
]


def _count_accesses(path: Path, receiver: str) -> Counter:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    counts: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in COMPAT_VIEW_NAMES:
            try:
                base = ast.unparse(node.value)
            except Exception:
                continue
            if base == receiver:
                counts[node.attr] += 1
    return counts


def _check(rel_path: str, receiver: str) -> list[str]:
    counts = _count_accesses(REPO_ROOT / rel_path, receiver)
    failures = []
    for attr, actual in counts.items():
        allowed = ALLOWED.get((rel_path, attr), 0)
        if actual > allowed:
            failures.append(
                f"  {rel_path}: '{receiver}.{attr}' accessed {actual} times, pinned at {allowed}"
            )
    return failures


def test_openai_helper_compat_view_access_count_is_pinned() -> None:
    failures = _check("bot/openai_helper.py", "self")
    assert not failures, (
        "New self.<compat-view> access(es) beyond the pinned count:\n" + "\n".join(failures)
        + "\n\nUse self._get_chat_states() / the registry instead, or bump the pinned "
        "count in ALLOWED with a deliberate review."
    )


def test_openai_tool_handler_compat_view_access_count_is_pinned() -> None:
    failures = _check("bot/openai_tool_handler.py", "helper")
    assert not failures, (
        "New helper.<compat-view> access(es) beyond the pinned count:\n" + "\n".join(failures)
        + "\n\nUse the registry/public session API instead, or bump the pinned count "
        "in ALLOWED with a deliberate review."
    )


def test_chat_run_compat_view_access_count_is_pinned() -> None:
    failures = _check("bot/chat_run.py", "helper")
    assert not failures, (
        "New helper.<compat-view> access(es) beyond the pinned count:\n" + "\n".join(failures)
        + "\n\nUse the registry/public session API instead, or bump the pinned count "
        "in ALLOWED with a deliberate review."
    )
