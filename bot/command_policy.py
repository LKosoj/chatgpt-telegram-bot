"""Best-effort command policy: allow/deny/require_approval rules over a normalized
command string, plus the string normalization used to evaluate them.

Command policy is bypassable. It classifies shell text and catches configured or
common dangerous forms, but obfuscation, encoding, or writing and then executing a
script can evade it. It is a speed bump against mistakes and injection, not a
sandbox boundary.

Known gaps versus a full shell tokenizer (deliberately not ported from the reference
TypeScript implementation, since a full tokenizer is a large surface for a benefit
this module does not claim to provide):

- No piped-shell-consumer tracking, so a literal payload piped into a shell that reads
  stdin (not via ``-c``) is not inlined into the scanned text (e.g.
  ``echo "rm -rf /" | bash`` without ``-c``, since the payload only reaches the shell at
  runtime through stdin).
- No simple-variable-assignment tracking, so ``x=rm; $x -rf /`` is not resolved back to
  ``rm -rf /``.
- No here-string payload extraction (``bash <<< "..."``) or a general literal-producer
  reconstruction for ``echo``/``printf`` arguments feeding into another shell (unless
  that shell invocation itself is in command position via ``-c``/``-ec``, see below).
- Command-position matching for built-in rules (``rm``, ``git push``, SQL clients,
  ``curl | sh``) is regex-based on a normalized/segmented string, not a real argv
  parser: exotic quoting or option-bundling forms can still slip past.

What normalization does handle: unquoting of ``"..."``, ``'...'`` and ``$'...'``
strings (content is always inlined for matching purposes, never erased), ANSI-C
backslash decoding, recursive expansion of ``$(...)``/backtick command substitutions
up to ``MAX_SUBSTITUTION_DEPTH`` levels, removal of heredocs that are written to a file
rather than executed by a shell, and splicing a segment boundary right after an
interpreter's ``-c``/``-ec`` flag (``sh``, ``bash``, ``zsh``, ``dash``, ``ksh``,
``python``, ``python3``, ``perl``, ``ruby``) so the script string that flag takes is
scanned as its own command, e.g. ``bash -c "rm -rf /"`` is scanned as if ``rm -rf /``
were its own segment. Inputs longer than ``MAX_NORMALIZE_LENGTH`` skip normalization
entirely (matched as raw text, with a warning logged) rather than pay its cost.

Built-in rules that care about command position (``rm``, ``git push``, SQL clients,
``curl | sh``) first split the normalized text into segments on ``;``, ``&&``, ``||``,
``|`` and newlines, then require the guarded command name to be the first token of a
segment (allowing a path prefix, variable assignments, and a leading ``sudo``/``env``).
Flags like ``-rf`` or ``--force`` are only honored within that same segment, not
elsewhere in the command. When a segment's command position is an unresolvable dynamic
substitution (``$(...)`` or `` `...` ``), the ``rm`` rule falls back to a conservative
match on the presence of a recursive-delete flag in that segment, since the real
command name cannot be determined statically (e.g. ``$(echo rm) -rf /``).

Rule patterns come from the operator (``TERMINAL_COMMAND_POLICY`` env var or the
built-in defaults below), never from Telegram message content, so patterns are treated
as trusted input: this module does not defend against ReDoS in a configured pattern.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

MAX_SUBSTITUTION_DEPTH = 8
MAX_NORMALIZE_LENGTH = 8192
ENV_VAR = "TERMINAL_COMMAND_POLICY"


@dataclass(frozen=True)
class CommandRule:
    pattern: str
    decision: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class CommandPolicy:
    mode: str
    rules: "tuple[CommandRule, ...]"


@dataclass(frozen=True)
class CommandDecision:
    decision: str
    reason: Optional[str]
    matched: Optional[str]


class CommandPolicyError(ValueError):
    pass


_PIP_BREAK_SYSTEM_PACKAGES_PATTERN = (
    r"(?:^|[;&(]\s*)(?:sudo\s+)?"
    r"(?:(?:\S*/)?pip(?:\d+(?:\.\d+)?)?|(?:\S*/)?python(?:\d+(?:\.\d+)?)?\s+-m\s+pip|uv\s+pip)"
    r"\b[^|;&\n]*\binstall\b[^|;&\n]*--break-system-packages\b"
)
_PIP_BREAK_SYSTEM_PACKAGES_REASON = (
    "Refusing to run pip install with --break-system-packages from terminal. "
    "Install runtime dependencies through the bot environment/setup flow instead of mutating system Python."
)

# Segment-boundary anchor: true start of string, or right after one of the segment
# separator characters (`;`, `&` — covering `&&`, `|` — covering `||`, or newline).
_SEG_START = r"(?:^|[;\n&|])\s*"
# Command position within a segment: optional leading VAR=value assignments, optional
# leading `sudo`/`env` (with its own flags), optional path prefix before the command.
_CMD_PREFIX = _SEG_START + r"(?:[A-Za-z_]\w*=\S*\s+)*(?:(?:sudo|env)\s+(?:-\S+\s+)*)*(?:\S*/)?"
# Rest of the current segment only: excludes the segment separator characters so a
# flag/keyword search below never leaks into a different segment.
_SEG_REST = r"[^;&|\n]*"

_INTERPRETER_NAMES = r"(?:sh|bash|zsh|dash|ksh|python3?|perl|ruby)"
_SQL_CLIENTS = r"(?:psql|mysql|mariadb|sqlite3|clickhouse-client|mongosh|mongo)"

# Splices a segment boundary (newline) right after an interpreter's -c/-ec flag so the
# script string it takes is scanned as its own command position, e.g.
# `bash -c "rm -rf /"` -> `bash -c\nrm -rf /`.
_INTERPRETER_DASH_C_RE = re.compile(
    r"((?:^|[;\n&|])\s*(?:\S+/)?" + _INTERPRETER_NAMES + r"\b(?:\s+-{1,2}\S+)*?\s+-[^-\s]*c)(\s+)",
    re.IGNORECASE,
)

# rm in command position, OR (conservatively) a segment whose command position is an
# unresolvable dynamic substitution ($(...) / `...`) — since the real command name
# cannot be determined statically, a recursive-delete flag there is still flagged
# (e.g. `$(echo rm) -rf /`).
_RM_HEAD_ALT = r"(?:" + _CMD_PREFIX + r"rm\b|" + _SEG_START + r"(?:\$\(|`))"

DEFAULT_RULES: "tuple[CommandRule, ...]" = (
    CommandRule(
        pattern=_RM_HEAD_ALT + _SEG_REST + r"(?:-[a-zA-Z]*r|--recursive)",
        decision="require_approval",
        reason="recursive delete",
    ),
    CommandRule(
        pattern=(
            _CMD_PREFIX + r"git\s+push\b" + _SEG_REST
            + r"(?:--force(?![-\w])|(?:^|\s)-[a-zA-Z]*f\b)"
        ),
        decision="require_approval",
        reason="force push",
    ),
    CommandRule(
        pattern=_CMD_PREFIX + _SQL_CLIENTS + r"\b" + _SEG_REST + r"\b(?:drop|truncate)\s+table\b",
        decision="require_approval",
        reason="destructive SQL",
    ),
    CommandRule(
        pattern=r"\bmkfs\b|:\(\)\s*\{",
        decision="deny",
        reason="destructive / fork bomb",
    ),
    CommandRule(
        pattern=_CMD_PREFIX + r"curl\b" + _SEG_REST + r"\|\s*(?:\S+/)?(?:ba|da|k|z)?sh\b",
        decision="require_approval",
        reason="pipe-to-shell",
    ),
    CommandRule(
        pattern=_PIP_BREAK_SYSTEM_PACKAGES_PATTERN,
        decision="deny",
        reason=_PIP_BREAK_SYSTEM_PACKAGES_REASON,
    ),
)

DEFAULT_POLICY = CommandPolicy(mode="denylist", rules=DEFAULT_RULES)


@lru_cache(maxsize=None)
def _compile_pattern(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.IGNORECASE)


_ANSI_C_ESCAPES = {
    "a": "\x07", "b": "\b", "e": "\x1b", "f": "\f",
    "n": "\n", "r": "\r", "t": "\t", "v": "\v",
}


def _safe_chr(code: int) -> str:
    try:
        return chr(code)
    except (ValueError, OverflowError):
        return ""


def _decode_ansi_c(value: str) -> str:
    value = re.sub(r"\\x([0-9a-fA-F]{1,2})", lambda m: _safe_chr(int(m.group(1), 16)), value)
    value = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: _safe_chr(int(m.group(1), 16)), value)
    value = re.sub(r"\\U([0-9a-fA-F]{8})", lambda m: _safe_chr(int(m.group(1), 16)), value)
    value = re.sub(r"\\([0-7]{1,3})", lambda m: _safe_chr(int(m.group(1), 8)), value)
    value = re.sub(
        r"\\([\\'\"abefnrtv])",
        lambda m: _ANSI_C_ESCAPES.get(m.group(1), m.group(1)),
        value,
    )
    return value


_SHELL_INVOCATION_RE = re.compile(r"(?:^|[|;&]\s*)(?:\S*/)?(?:ba|da|k|z)?sh((?:\s+[^|;&]*)?)")
_SHELL_DASH_C_RE = re.compile(r"(?:^|\s)-[^-\s]*c(?:\s|$)")


def _heredoc_runs_shell(command_line: str) -> bool:
    for match in _SHELL_INVOCATION_RE.finditer(command_line):
        args = match.group(1) or ""
        if not _SHELL_DASH_C_RE.search(args):
            return True
    return False


_HEREDOC_RE = re.compile(
    r"^([^\n]*)<<-?\s*([\"']?)([A-Za-z_]\w*)\2([^\n]*)\n([\s\S]*?)^\s*\3\s*$",
    re.MULTILINE,
)


def _strip_written_heredocs(command: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        pre, post = match.group(1), match.group(4)
        combined = pre + post
        if ">" in combined and not _heredoc_runs_shell(combined):
            return ""
        return match.group(0)

    return _HEREDOC_RE.sub(_replace, command)


_DOUBLE_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_QUOTED_SUB_RE = re.compile(r"\$\([^)]*\)|`[^`]*`")
_ANSI_C_QUOTED_RE = re.compile(r"\$'((?:[^'\\]|\\.)*)'")
_SINGLE_QUOTED_RE = re.compile(r"'[^']*'")
_BACKSLASH_ESCAPE_RE = re.compile(r"\\([\w@%+=:,./-])")


def _double_quote_repl(match: "re.Match[str]") -> str:
    matched = match.group(0)
    subs = _QUOTED_SUB_RE.findall(matched)
    if subs:
        return " ".join(subs)
    # Inline the quoted content for matching purposes rather than erasing it — an
    # unquoted-looking payload like `"rm -rf /"` must still be visible to the rules.
    return matched[1:-1]


def _ansi_c_repl(match: "re.Match[str]") -> str:
    return _decode_ansi_c(match.group(1))


def _single_quote_repl(match: "re.Match[str]") -> str:
    return match.group(0)[1:-1]


def _extract_top_level_substitutions(text: str) -> "list[str]":
    """Single-pass, quote-aware scan for outermost $(...) and `...` command
    substitutions. Content inside single-quoted strings is skipped (the shell does not
    substitute there); content inside double-quoted strings is still scanned, since
    substitution does happen there. Nested substitutions are tracked with an explicit
    stack (each frame gets its own fresh quote context, matching how a nested $(...)
    or `...` starts a new quoting context in the shell) so this is O(len(text)) even
    for many unbalanced/nested markers — a prior recursive-rescan implementation was
    O(n^2) on adversarial input like a long run of unclosed "$(". Returns the raw
    substitution bodies, in order of appearance, for the outermost level only (callers
    recurse into each returned body separately)."""
    results: "list[str]" = []
    stack: "list[tuple[str, int, str]]" = []  # (kind, body_start, saved_quote)
    quote = ""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if quote == "'":
            if c == "'":
                quote = ""
            i += 1
            continue
        if quote == '"':
            if c == '"':
                quote = ""
                i += 1
                continue
            if c == "$" and i + 1 < n and text[i + 1] == "(":
                stack.append(("paren", i + 2, quote))
                quote = ""
                i += 2
                continue
            if c == "`":
                stack.append(("backtick", i + 1, quote))
                quote = ""
                i += 1
                continue
            i += 1
            continue
        # quote == "" here: unquoted context, top-level or inside a substitution frame.
        if c == "'":
            quote = "'"
            i += 1
            continue
        if c == '"':
            quote = '"'
            i += 1
            continue
        if c == "$" and i + 1 < n and text[i + 1] == "(":
            stack.append(("paren", i + 2, quote))
            i += 2
            continue
        if c == "`":
            if stack and stack[-1][0] == "backtick":
                _, start, saved_quote = stack.pop()
                if not stack:
                    results.append(text[start:i])
                quote = saved_quote
                i += 1
                continue
            stack.append(("backtick", i + 1, quote))
            i += 1
            continue
        if c == ")" and stack and stack[-1][0] == "paren":
            _, start, saved_quote = stack.pop()
            if not stack:
                results.append(text[start:i])
            quote = saved_quote
            i += 1
            continue
        i += 1
    return results


def _normalize_at_depth(command: str, depth: int) -> str:
    stripped = _strip_written_heredocs(command)
    base = _DOUBLE_QUOTED_RE.sub(_double_quote_repl, stripped)
    base = _ANSI_C_QUOTED_RE.sub(_ansi_c_repl, base)
    base = _SINGLE_QUOTED_RE.sub(_single_quote_repl, base)
    base = _BACKSLASH_ESCAPE_RE.sub(r"\1", base)
    if depth >= MAX_SUBSTITUTION_DEPTH:
        return base
    substitutions = _extract_top_level_substitutions(stripped)
    if not substitutions:
        return base
    parts = [base] + [_normalize_at_depth(sub, depth + 1) for sub in substitutions]
    return "\n".join(parts)


def normalize_command(text: str) -> str:
    """Return a best-effort unquoted/expanded view of `text` for pattern matching.

    Not a shell parser: it inlines quoted content, decodes ANSI-C escapes, and
    recursively inlines $(...)/`...` substitutions, up to MAX_SUBSTITUTION_DEPTH, then
    splices a segment boundary after an interpreter's -c/-ec flag. See the module
    docstring for what it deliberately does not handle.

    Inputs longer than MAX_NORMALIZE_LENGTH are returned unmodified (a warning is
    logged) rather than normalized — normalization cost scales with input size, and an
    operator-unbounded string (e.g. large pasted output re-fed as a command) should not
    buy a disproportionate amount of regex work.
    """
    if len(text) > MAX_NORMALIZE_LENGTH:
        logger.warning(
            "command_policy: input length %d exceeds MAX_NORMALIZE_LENGTH (%d); "
            "matching raw text without normalization",
            len(text),
            MAX_NORMALIZE_LENGTH,
        )
        return text
    normalized = _normalize_at_depth(text, 0)
    return _INTERPRETER_DASH_C_RE.sub(lambda m: m.group(1) + "\n", normalized)


def parse_command_policy(data: object) -> CommandPolicy:
    if not isinstance(data, dict):
        raise CommandPolicyError("command policy must be an object")

    mode = data.get("mode")
    if mode not in ("denylist", "allowlist"):
        raise CommandPolicyError('mode must be "denylist" or "allowlist"')

    rules_data = data.get("rules")
    if not isinstance(rules_data, list):
        raise CommandPolicyError("rules must be an array")

    rules: "list[CommandRule]" = []
    for i, raw in enumerate(rules_data):
        if not isinstance(raw, dict):
            raise CommandPolicyError(f"rules[{i}] must be an object")

        pattern = raw.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise CommandPolicyError(f"rules[{i}].pattern must be a non-empty string")
        try:
            _compile_pattern(pattern)
        except re.error as exc:
            raise CommandPolicyError(f"rules[{i}].pattern is not a valid regex: {exc}") from exc

        decision = raw.get("decision")
        if decision not in ("allow", "deny", "require_approval"):
            raise CommandPolicyError(f'rules[{i}].decision must be "allow", "deny", or "require_approval"')

        reason = raw.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise CommandPolicyError(f"rules[{i}].reason must be a string")

        rules.append(CommandRule(pattern=pattern, decision=decision, reason=reason))

    return CommandPolicy(mode=mode, rules=tuple(rules))


def load_policy_from_env(env_value: Optional[str] = None) -> CommandPolicy:
    """Load an operator-configured policy from `TERMINAL_COMMAND_POLICY`, layered on
    top of the built-in DEFAULT_RULES floor (floor rules are always checked first).
    Never raises: any parse/validation failure is logged and DEFAULT_POLICY is used.
    """
    raw = os.getenv(ENV_VAR) if env_value is None else env_value
    if raw is None or not raw.strip():
        return DEFAULT_POLICY

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("command_policy: invalid JSON in %s (%s); using default policy", ENV_VAR, exc)
        return DEFAULT_POLICY

    try:
        override = parse_command_policy(data)
    except CommandPolicyError as exc:
        logger.warning("command_policy: invalid %s (%s); using default policy", ENV_VAR, exc)
        return DEFAULT_POLICY

    return CommandPolicy(mode=override.mode, rules=DEFAULT_POLICY.rules + override.rules)


def evaluate_command(command: str, policy: CommandPolicy) -> CommandDecision:
    scannable = normalize_command(command)
    for rule in policy.rules:
        try:
            pattern = _compile_pattern(rule.pattern)
        except re.error:
            logger.warning(
                "command_policy: skipping invalid stored rule pattern %r (%s)",
                rule.pattern,
                rule.decision,
            )
            continue
        match = pattern.search(scannable)
        if match:
            return CommandDecision(decision=rule.decision, reason=rule.reason, matched=match.group(0))

    if policy.mode == "allowlist":
        return CommandDecision(decision="deny", reason="not in allowlist", matched=None)
    return CommandDecision(decision="allow", reason=None, matched=None)
