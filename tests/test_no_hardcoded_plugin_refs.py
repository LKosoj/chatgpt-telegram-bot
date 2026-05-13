"""Linter test: catch new hardcoded plugin-name references in core modules.

The Stage 0-5 migration moved plugin-specific logic out of ``bot/openai_helper.py``,
``bot/telegram_bot.py`` and ``bot/database.py`` into the plugin layer. To keep the
boundary clean going forward, this test counts string-literal references to known
plugin ids in those three files and fails when the count exceeds a documented
allow-list.

Each allowed occurrence has a short note explaining *why* the hardcode is
acceptable (typically: generic UI menu reading, or a documented compromise like
Strategy Z for hindsight_memory). If you add a legitimate new hardcode, bump the
allowed count and add a reason. If a hardcode can be removed, lower the count.
"""

from __future__ import annotations

import re
import ast
import io
import tokenize
from pathlib import Path
from typing import Dict, Tuple

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_FILES = (
    REPO_ROOT / "bot" / "telegram_bot.py",
    REPO_ROOT / "bot" / "openai_helper.py",
    REPO_ROOT / "bot" / "openai_tool_handler.py",
    REPO_ROOT / "bot" / "database.py",
)
PLUGIN_IDS = (
    "conversation_analytics",
    "reminders",
    "skills",
    "hindsight_memory",
    "agent_tools",
)

# Allow-list: (file_relpath, plugin_id) -> (expected_count, reason).
# Counts the number of source lines containing a quoted occurrence of the
# plugin id ("name" or 'name'). Bump or lower deliberately.
ALLOWED: Dict[Tuple[str, str], Tuple[int, str]] = {
    ("bot/telegram_bot.py", "agent_tools"): (
        3,
        "UI: live busy-status plan reader via generic get_plugin('agent_tools') and related log/doc text.",
    ),
    ("bot/telegram_bot.py", "skills"): (
        4,
        "UI: callback action literal 'skills' in the {'skills','skill_page'} set + settings menu reader (has_plugin + get_plugin).",
    ),
    ("bot/openai_tool_handler.py", "agent_tools"): (
        3,
        "Strategy Z delivery contract: skills_agent final delivery is routed through agent_tools.deliver_to_user.",
    ),
    ("bot/openai_helper.py", "hindsight_memory"): (
        1,
        "Strategy Z (4B): helper persists plugin-injected memory marker via plugin.is_hindsight_memory_message; documented compromise.",
    ),
}


def _count_quoted_occurrences(text: str, plugin_id: str) -> int:
    pattern = re.compile(rf"\b{re.escape(plugin_id)}(?:\.|\b)")
    count = 0
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str):
            count += len(pattern.findall(value))
    return count


@pytest.mark.parametrize("plugin_id", PLUGIN_IDS)
def test_no_new_hardcoded_plugin_refs(plugin_id: str) -> None:
    failures = []
    for path in CORE_FILES:
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        actual = _count_quoted_occurrences(text, plugin_id)
        allowed_count, _reason = ALLOWED.get((rel, plugin_id), (0, ""))
        if actual > allowed_count:
            failures.append(
                f"  {rel}: found {actual} hardcoded '{plugin_id}' references, allowed {allowed_count}"
            )
        elif actual < allowed_count:
            failures.append(
                f"  {rel}: found {actual} hardcoded '{plugin_id}' references, "
                f"but allow-list expects {allowed_count} — lower the allow-list entry."
            )
    assert not failures, (
        "Hardcoded plugin-name references diverged from allow-list:\n"
        + "\n".join(failures)
        + "\n\nUpdate ALLOWED in this test if the change is intentional."
    )
