from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List

from .plugin import Plugin

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 260
DEFAULT_CWD = "/tmp"
BOUNDED_OUTPUT_HINT = (
    "Output was truncated. Re-run with a bounded command that selects only the needed data "
    "(for example filters, head/tail, max-depth searches, or exact known artifact paths)."
)


def _output_byte_limit() -> int:
    try:
        return max(1024, int(os.getenv("TERMINAL_OUTPUT_BYTE_LIMIT", str(8 * 1024))))
    except (TypeError, ValueError):
        return 8 * 1024


OUTPUT_BYTE_LIMIT = _output_byte_limit()

PIP_BREAK_SYSTEM_PACKAGES_RE = re.compile(
    r"(?:^|[;&(]\s*)(?:sudo\s+)?"
    r"(?:(?:\S*/)?pip(?:\d+(?:\.\d+)?)?|(?:\S*/)?python(?:\d+(?:\.\d+)?)?\s+-m\s+pip|uv\s+pip)"
    r"\b[^|;&\n]*\binstall\b[^|;&\n]*--break-system-packages\b"
)
INSTALL_COMMAND_RE = re.compile(
    r"(?:^|[;&(]\s*)(?:sudo\s+)?"
    r"(?:"
    r"(?:\S*/)?pip(?:\d+(?:\.\d+)?)?\s+install\b|"
    r"(?:\S*/)?python(?:\d+(?:\.\d+)?)?\s+-m\s+pip\s+install\b|"
    r"uv\s+pip\s+install\b|"
    r"(?:\S*/)?npm\s+(?:install|i|ci)\b|"
    r"(?:\S*/)?apt(?:-get)?\s+(?:install|upgrade|dist-upgrade)\b|"
    r"(?:\S*/)?dnf\s+install\b|"
    r"(?:\S*/)?yum\s+install\b"
    r")"
)
TAIL_COMMAND_RE = re.compile(r"^\s*(?:\S*/)?tail\b")

PIP_BREAK_SYSTEM_PACKAGES_ERROR = (
    "Refusing to run pip install with --break-system-packages from terminal. "
    "Install runtime dependencies through the bot environment/setup flow instead of mutating system Python."
)
INSTALL_PIPE_TAIL_ERROR = (
    "Refusing to pipe an install command directly into tail because it hides the installer exit code. "
    "Use: cmd >log 2>&1; rc=$?; tail -30 log; exit $rc"
)


class TerminalPlugin(Plugin):
    """Run a shell or argv command and return stdout/stderr/return code."""

    plugin_id = "terminal"
    function_prefix = "terminal"

    def get_source_name(self) -> str:
        return "Terminal"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "terminal",
            "description": (
                "Execute a shell command and return stdout, stderr, and the return code. "
                "By default runs via /bin/sh -c (shell=true), so pipes, redirects and chained "
                "commands work as written. "
                "Pass shell=false to bypass the shell and exec a single program directly. "
                "Use for shell workflows: pipes, redirects, file-system operations, invoking "
                "installed CLI tools, running existing scripts. Keep output bounded; prefer "
                "commands that filter, limit, or inspect exact known paths instead of broad listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Command to run. With shell=true it is passed to /bin/sh -c "
                            "verbatim; with shell=false it is split via shlex and exec'd directly."
                        ),
                    },
                    "shell": {
                        "type": "boolean",
                        "description": "If true, run via /bin/sh -c. Set false to exec directly without a shell.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds for the command.",
                    },
                },
                "required": ["command"],
            },
        }]

    async def execute(self, function_name: str, helper: Any, **kwargs: Any) -> Dict:
        if function_name != "terminal":
            return {"error": f"Unknown function: {function_name}"}

        command = kwargs.get("command")
        if not isinstance(command, str) or not command.strip():
            return {"error": "command must be a non-empty string"}

        guard_error = self._guard_command(command)
        if guard_error:
            return {
                "success": False,
                "error": guard_error,
                "guard": "terminal_command_guard",
            }

        shell = bool(kwargs.get("shell", True))
        cwd = kwargs.get("cwd") or DEFAULT_CWD
        try:
            timeout = float(kwargs.get("timeout") or DEFAULT_TIMEOUT_SECONDS)
        except (TypeError, ValueError):
            return {"error": "timeout must be a number"}

        cwd_path = Path(cwd).expanduser()
        if not cwd_path.is_dir():
            return {"error": f"cwd does not exist or is not a directory: {cwd}"}

        if shell:
            display = command
            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(cwd_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                return {"error": f"Failed to spawn shell: {exc}"}
        else:
            try:
                argv = shlex.split(command)
            except ValueError as exc:
                return {"error": f"Failed to parse command: {exc}"}
            if not argv:
                return {"error": "command parsed to empty argv"}
            display = " ".join(shlex.quote(a) for a in argv)
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(cwd_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                return {"error": f"Command not found: {exc}"}
            except Exception as exc:
                return {"error": f"Failed to spawn process: {exc}"}

        logger.info(
            "Terminal exec shell=%s cwd=%s timeout=%s cmd=%s",
            shell,
            cwd_path,
            timeout,
            display,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "cwd": str(cwd_path),
                "shell": shell,
            }

        stdout, stdout_truncated = self._truncate_output(stdout_bytes.decode("utf-8", errors="replace"))
        stderr, stderr_truncated = self._truncate_output(stderr_bytes.decode("utf-8", errors="replace"))
        logger.info(
            "Terminal finished returncode=%s stdout_bytes=%s stderr_bytes=%s",
            process.returncode,
            len(stdout_bytes),
            len(stderr_bytes),
        )

        result = {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": str(cwd_path),
            "shell": shell,
        }
        if stdout_truncated or stderr_truncated:
            result.update({
                "output_truncated": True,
                "output_limit_bytes": OUTPUT_BYTE_LIMIT,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "stdout_bytes": len(stdout_bytes),
                "stderr_bytes": len(stderr_bytes),
                "output_hint": BOUNDED_OUTPUT_HINT,
            })
        return result

    @staticmethod
    def _truncate_output(text: str) -> tuple[str, bool]:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= OUTPUT_BYTE_LIMIT:
            return text, False
        return (
            encoded[:OUTPUT_BYTE_LIMIT].decode("utf-8", errors="ignore")
            + "\n... [truncated]",
            True,
        )

    @staticmethod
    def _truncate(text: str) -> str:
        return TerminalPlugin._truncate_output(text)[0]

    @staticmethod
    def _guard_command(command: str) -> str | None:
        if PIP_BREAK_SYSTEM_PACKAGES_RE.search(command):
            return PIP_BREAK_SYSTEM_PACKAGES_ERROR

        pipe_segments = TerminalPlugin._split_unquoted_pipes(command)
        for index, segment in enumerate(pipe_segments[:-1]):
            if not INSTALL_COMMAND_RE.search(segment):
                continue
            if any(TAIL_COMMAND_RE.search(next_segment) for next_segment in pipe_segments[index + 1:]):
                return INSTALL_PIPE_TAIL_ERROR
        return None

    @staticmethod
    def _split_unquoted_pipes(command: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        quote: str | None = None
        escaped = False

        for char in command:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                current.append(char)
                escaped = True
                continue
            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                current.append(char)
                quote = char
                continue
            if char == "|":
                parts.append("".join(current))
                current = []
                continue
            current.append(char)

        parts.append("".join(current))
        return parts
