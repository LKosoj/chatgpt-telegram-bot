from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
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


def _kill_process_group(pid: int) -> None:
    """Send SIGKILL to the entire process group of pid."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _drain_pipe(stream: asyncio.StreamReader) -> None:
    """Read and discard remaining pipe data until EOF."""
    try:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
    except Exception:
        pass


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Read up to limit+1 bytes from stream; return (data[:limit], over_limit)."""
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = await stream.read(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    data = b"".join(chunks)
    if len(data) > limit:
        return data[:limit], True
    return data, False


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
                    start_new_session=True,
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
                    start_new_session=True,
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

        limit = OUTPUT_BYTE_LIMIT
        stdout_task = asyncio.ensure_future(_read_bounded(process.stdout, limit))
        stderr_task = asyncio.ensure_future(_read_bounded(process.stderr, limit))

        deadline = asyncio.get_event_loop().time() + timeout
        pending = {stdout_task, stderr_task}
        timed_out = False
        over_limit = False
        while pending:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                timed_out = True
                break
            done, pending = await asyncio.wait(pending, timeout=remaining)
            if not done:
                timed_out = True
                break
            # If any reader hit the limit, kill the process and cancel the rest
            for t in done:
                if not t.cancelled() and not t.exception() and t.result()[1]:
                    over_limit = True
                    _kill_process_group(process.pid)
                    for p in pending:
                        p.cancel()
                    pending = set()
                    break

        if timed_out:
            stdout_task.cancel()
            stderr_task.cancel()
            _kill_process_group(process.pid)
            # Settle the cancelled readers so their StreamReader waiters are released
            # before draining; otherwise _drain_pipe's read() would hit a pending waiter.
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            await asyncio.gather(
                _drain_pipe(process.stdout),
                _drain_pipe(process.stderr),
                return_exceptions=True,
            )
            try:
                await process.wait()
            except Exception:
                pass
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "cwd": str(cwd_path),
                "shell": shell,
            }

        # Settle any cancelled/pending tasks before accessing results
        all_tasks = [t for t in (stdout_task, stderr_task) if not t.done()]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        def _task_result(task: asyncio.Task) -> tuple[bytes, bool]:
            if task.cancelled() or task.exception():
                return b"", False
            return task.result()

        stdout_bytes, stdout_over = _task_result(stdout_task)
        stderr_bytes, stderr_over = _task_result(stderr_task)

        if stdout_over or stderr_over:
            _kill_process_group(process.pid)
            await asyncio.gather(
                _drain_pipe(process.stdout),
                _drain_pipe(process.stderr),
                return_exceptions=True,
            )
        await process.wait()

        stdout_truncated = stdout_over
        stderr_truncated = stderr_over

        if stdout_truncated:
            stdout = stdout_bytes.decode("utf-8", errors="replace") + "\n... [truncated]"
        else:
            stdout = stdout_bytes.decode("utf-8", errors="replace")

        if stderr_truncated:
            stderr = stderr_bytes.decode("utf-8", errors="replace") + "\n... [truncated]"
        else:
            stderr = stderr_bytes.decode("utf-8", errors="replace")

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
                "output_limit_bytes": limit,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "stdout_bytes": len(stdout_bytes),
                "stderr_bytes": len(stderr_bytes),
                "output_hint": BOUNDED_OUTPUT_HINT,
            })
        return result

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
