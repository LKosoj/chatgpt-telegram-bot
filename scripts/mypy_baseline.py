"""Track mypy error counts against a committed baseline (`mypy_baseline.json`).

Usage:
    python3 scripts/mypy_baseline.py update   # regenerate the baseline from current mypy output
    python3 scripts/mypy_baseline.py check    # fail if error counts grew or new (file, code) pairs appeared

The baseline is keyed by (file, error code) -> count, without line numbers, so unrelated
line-shifting edits don't cause false regressions. `check` fails (exit 1) when any pair's
count increases or a brand-new pair appears; it exits 0 (with a suggestion to run `update`)
when counts only decreased.

`MYPY_PYTHON` selects the `--python-executable` passed to mypy (defaults to `sys.executable`).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

MYPY_ERROR_RE = re.compile(
    r'^(?P<file>[^:]+):\d+: error: .*\[(?P<code>[\w-]+)\]\s*$'
)
MYPY_ERROR_NO_CODE_RE = re.compile(r'^(?P<file>[^:]+):\d+: error: .*$')

BASELINE_PATH = Path(__file__).resolve().parent.parent / "mypy_baseline.json"


def parse_mypy_output(output: str) -> dict[tuple[str, str], int]:
    """Count mypy `error:` lines by (file, code); `note:` lines are ignored.
    Errors without a trailing `[code]` (rare) are bucketed under code "no-code"."""
    counts: dict[tuple[str, str], int] = {}
    for line in output.splitlines():
        m = MYPY_ERROR_RE.match(line)
        if m:
            key = (m.group("file"), m.group("code"))
        else:
            m2 = MYPY_ERROR_NO_CODE_RE.match(line)
            if not m2:
                continue
            key = (m2.group("file"), "no-code")
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_baseline(path: Path) -> dict[tuple[str, str], int]:
    """Missing file -> {} (empty baseline, not an error)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[tuple[str, str], int] = {}
    for file, codes in data.items():
        for code, count in codes.items():
            counts[(file, code)] = count
    return counts


def save_baseline(path: Path, counts: dict[tuple[str, str], int]) -> None:
    """Nested JSON {file: {code: count}}, sorted keys, trailing newline."""
    nested: dict[str, dict[str, int]] = {}
    for (file, code), count in counts.items():
        nested.setdefault(file, {})[code] = count
    data = {file: dict(sorted(codes.items())) for file, codes in sorted(nested.items())}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def diff_counts(
    baseline: dict[tuple[str, str], int], current: dict[tuple[str, str], int]
) -> tuple[list[str], list[str]]:
    """Returns (regressions, improvements) as printable "file [code]: N -> M" lines.
    Regression: current > baseline (includes brand-new pairs, baseline=0).
    Improvement: current < baseline (includes pairs that disappeared, current=0)."""
    regressions: list[str] = []
    improvements: list[str] = []
    for file, code in sorted(set(baseline) | set(current)):
        before = baseline.get((file, code), 0)
        after = current.get((file, code), 0)
        if after > before:
            regressions.append(f"{file} [{code}]: {before} -> {after}")
        elif after < before:
            improvements.append(f"{file} [{code}]: {before} -> {after}")
    return regressions, improvements


def run_mypy() -> str:
    """Invoke mypy via subprocess, return combined stdout (ignore returncode: mypy
    exits 1 whenever there is at least one error, which is expected/normal here)."""
    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "pyproject.toml"
    python_executable = os.environ.get("MYPY_PYTHON", sys.executable)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config_file),
         "--python-executable", python_executable],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout


def main(argv: list[str]) -> int:
    """argv[0] must be "update" or "check"; anything else -> usage on stderr, exit 2."""
    if not argv or argv[0] not in ("update", "check"):
        print("usage: python3 scripts/mypy_baseline.py update|check", file=sys.stderr)
        return 2

    command = argv[0]
    current = parse_mypy_output(run_mypy())

    if command == "update":
        save_baseline(BASELINE_PATH, current)
        total = sum(current.values())
        print(f"mypy_baseline.json updated: {total} errors across {len(current)} (file, code) pairs.")
        return 0

    # command == "check"
    baseline = load_baseline(BASELINE_PATH)
    regressions, improvements = diff_counts(baseline, current)
    if regressions:
        print("mypy baseline regressions:")
        for line in regressions:
            print(f"  {line}")
        return 1
    if improvements:
        print("mypy error counts decreased; consider running: python scripts/mypy_baseline.py update")
        for line in improvements:
            print(f"  {line}")
        print("mypy baseline check passed.")
        return 0
    print("mypy baseline check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
