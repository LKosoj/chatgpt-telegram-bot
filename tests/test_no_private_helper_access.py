"""Linter test: catch new private-attribute access on the OpenAIHelper instance from
bot/telegram_bot.py and bot/plugins/*.py.

T19 replaced direct dict mutation (self.openai.conversations[...]) and
getattr(self.openai, "_private_method", None) probing with a small public API
(history_snapshot/load_session/replace_system_message/evict/chat_state_scope). This
test scans for regressions: any future `self.openai._foo` / `helper._foo` / the
explicit shared-state dict names, direct or via getattr(). If a new access is
legitimate, add it to ALLOWED with a reason -- do not silently raise the threshold.

Scanned: bot/telegram_bot.py, bot/plugins/*.py and bot/skill_script_routing.py --
the three places that consume an OpenAIHelper instance from outside.

Deliberately NOT scanned: bot/openai_tool_handler.py. It is not a consumer of the
helper but a piece of the helper's own request machinery that was split into its own
module, and it reaches into ~16 helper internals on purpose
(`_add_function_call_to_history`, `_apply_before_chat_request_mutators`,
`_without_chat_lock`, ...). Guarding it would freeze OpenAIHelper's internals rather
than a boundary. Two of those accesses do mutate the shared history cache from outside
the class (`helper.conversations.setdefault` at bot/openai_tool_handler.py:255,
`helper.conversations.get` at :1658); converting them to the public API was outside
T19's scope and is a known remaining gap.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEGRAM_BOT_FILE = REPO_ROOT / "bot" / "telegram_bot.py"
PLUGINS_DIR = REPO_ROOT / "bot" / "plugins"
SKILL_ROUTING_FILE = REPO_ROOT / "bot" / "skill_script_routing.py"

# Names that are not underscore-prefixed but are still considered private
# per-chat state, mirroring OpenAIHelper._clear_chat_state's docstring.
DENYLISTED_STATE_ATTRS = {"conversations", "loaded_conversation_sessions", "last_updated"}

# (file_relpath, attr_name) -> (expected_count, reason)
ALLOWED: Dict[Tuple[str, str], Tuple[int, str]] = {
    # Pre-existing, read-only mode detection for the skills-agent gate. Both reads are
    # defensive getattr() probes so the routing helper also works against the minimal
    # test doubles in tests/test_skills_agent_gate.py. Lower these to 0 if the helper
    # ever exposes a public "system message of the active chat" accessor.
    ("bot/skill_script_routing.py", "conversations"): (
        1, "read-only system-message lookup via getattr(), bot/skill_script_routing.py:20"),
    ("bot/skill_script_routing.py", "_mode_from_system_message"): (
        1, "read-only mode resolution via getattr(), bot/skill_script_routing.py:31"),
}


def _is_private_attr(name: str) -> bool:
    return name.startswith("_") or name in DENYLISTED_STATE_ATTRS


def _matches_target(node: ast.AST, targets: set[str]) -> bool:
    try:
        return ast.unparse(node) in targets
    except Exception:
        return False


def _find_violations(tree: ast.AST, targets: set[str]) -> list[tuple[int, str]]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _matches_target(node.value, targets):
            if _is_private_attr(node.attr):
                violations.append((node.lineno, node.attr))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _matches_target(node.args[0], targets)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attr = node.args[1].value
            if _is_private_attr(attr):
                violations.append((node.lineno, attr))
    return violations


def _check_file(path: Path, targets: set[str]) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _find_violations(tree, targets)


def _assert_against_allowlist(rel: str, violations: list[tuple[int, str]]) -> list[str]:
    from collections import Counter
    counts = Counter(attr for _lineno, attr in violations)
    failures = []
    for attr, actual in counts.items():
        allowed_count, _reason = ALLOWED.get((rel, attr), (0, ""))
        if actual > allowed_count:
            lines = [lineno for lineno, a in violations if a == attr]
            failures.append(
                f"  {rel}: found {actual} private access(es) to '{attr}' at lines {lines}, "
                f"allowed {allowed_count}"
            )
    return failures


def test_telegram_bot_does_not_touch_openai_privates() -> None:
    violations = _check_file(TELEGRAM_BOT_FILE, {"self.openai"})
    failures = _assert_against_allowlist("bot/telegram_bot.py", violations)
    assert not failures, (
        "bot/telegram_bot.py reaches into OpenAIHelper privates:\n" + "\n".join(failures)
        + "\n\nUse the public session API (history_snapshot/load_session/"
        "replace_system_message/evict/chat_state_scope) instead, or add a "
        "documented ALLOWED entry."
    )


@pytest.mark.parametrize("plugin_path", sorted(PLUGINS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_plugins_do_not_touch_helper_privates(plugin_path: Path) -> None:
    violations = _check_file(plugin_path, {"helper", "self.openai", "self.helper"})
    rel = plugin_path.relative_to(REPO_ROOT).as_posix()
    failures = _assert_against_allowlist(rel, violations)
    assert not failures, (
        f"{rel} reaches into the OpenAIHelper instance's privates:\n" + "\n".join(failures)
        + "\n\nUse a public accessor, or add a documented ALLOWED entry."
    )


def test_skill_script_routing_does_not_add_helper_privates() -> None:
    violations = _check_file(SKILL_ROUTING_FILE, {"helper"})
    failures = _assert_against_allowlist("bot/skill_script_routing.py", violations)
    assert not failures, (
        "bot/skill_script_routing.py reaches into OpenAIHelper privates:\n"
        + "\n".join(failures)
        + "\n\nUse a public accessor, or add a documented ALLOWED entry."
    )
