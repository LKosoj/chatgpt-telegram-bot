"""AST guard: `import openai` and `X.client.Y` attribute access must live only in
bot/ai_providers/ (T10, единый интерфейс провайдера). Every other module reaches the
SDK exclusively through OpenAIHelper._provider / bot.ai_provider.* error classes.

Two node shapes are flagged for every .py file under bot/, excluding bot/ai_providers/:
1. `import openai` / `import openai.foo` / `from openai import ...` / `from openai.foo
   import ...`.
2. An Attribute node shaped `X.client.Y` (e.g. `self.client.images.generate`,
   `helper.client.images.generate`) -- an Attribute whose value is itself an Attribute
   with `.attr == "client"`. This does NOT match `self.client = ...` (assignment
   target, no further chained attribute) or passing `self.client` as a bare argument
   (also no chained attribute) -- both of which OpenAIHelper's provider wiring relies
   on (get_client=lambda: self.client).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = REPO_ROOT / "bot"
ALLOWED_DIR = BOT_DIR / "ai_providers"

# (file_relpath, kind) -> reason
ALLOWLIST: dict[tuple[str, str], str] = {
    ("bot/openai_helper.py", "client_attr:close"):
        "OpenAIHelper.close() resource cleanup on shutdown, no request "
        "semantics; not worth an indirection through the provider.",
    # bot/net_safety.py: `http.client.HTTPSConnection`/`HTTPConnection` is the stdlib
    # http.client module, not an OpenAI SDK client instance -- the "X.client.Y" shape
    # matches textually (module attribute chain) but is unrelated to this guard.
    ("bot/net_safety.py", "client_attr:HTTPSConnection"):
        "http.client.HTTPSConnection -- stdlib module, not the OpenAI SDK.",
    ("bot/net_safety.py", "client_attr:HTTPConnection"):
        "http.client.HTTPConnection -- stdlib module, not the OpenAI SDK.",
    # bot/plugins/hindsight_memory.py: self.client there is a HindsightClient
    # (bot/plugins/hindsight_memory.py:594), an unrelated plain-httpx client for the
    # memory service, not the OpenAI SDK. Verified during T10 inventory (T10-plan.md §0).
    ("bot/plugins/hindsight_memory.py", "client_attr:enabled"):
        "self.client is HindsightClient, not the OpenAI SDK.",
    ("bot/plugins/hindsight_memory.py", "client_attr:retain_memories"):
        "self.client is HindsightClient, not the OpenAI SDK.",
    ("bot/plugins/hindsight_memory.py", "client_attr:recall"):
        "self.client is HindsightClient, not the OpenAI SDK.",
    ("bot/plugins/hindsight_memory.py", "client_attr:list_memories"):
        "self.client is HindsightClient, not the OpenAI SDK.",
    ("bot/plugins/hindsight_memory.py", "client_attr:stats"):
        "self.client is HindsightClient, not the OpenAI SDK.",
    ("bot/plugins/hindsight_memory.py", "client_attr:clear_bank"):
        "self.client is HindsightClient, not the OpenAI SDK.",
}


def _violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "openai" or alias.name.startswith("openai.") for alias in node.names):
                out.append((node.lineno, "import_openai"))
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "openai" or node.module.startswith("openai.")):
                out.append((node.lineno, "import_openai"))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "client":
                out.append((node.lineno, f"client_attr:{node.attr}"))
    return out


def test_no_raw_openai_access_outside_ai_providers():
    failures = []
    for path in sorted(BOT_DIR.rglob("*.py")):
        if ALLOWED_DIR in path.parents:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for lineno, kind in _violations(path):
            reason = ALLOWLIST.get((rel, kind))
            if reason is None:
                failures.append(f"{rel}:{lineno}: {kind} (not allow-listed)")
    assert not failures, (
        "Raw openai SDK access found outside bot/ai_providers/ -- go through "
        "OpenAIHelper._provider (bot.ai_provider.AIProvider) instead, or add a "
        "documented ALLOWLIST entry:\n" + "\n".join(failures)
    )
