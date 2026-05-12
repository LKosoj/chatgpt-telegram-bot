from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any, Dict, List

from .plugin import Plugin

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 260
DEFAULT_CWD = "/tmp"
OUTPUT_BYTE_LIMIT = 32 * 1024


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
                "commands like 'node /tmp/x.js && ls /tmp/' work as written. "
                "Pass shell=false to bypass the shell and exec a single program directly. "
                "Use for shell workflows: pipes, redirects, file-system operations, invoking "
                "installed CLI tools, running existing scripts. Do NOT use for general Python "
                "data analysis — use deep_analysis instead."
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

        stdout = self._truncate(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = self._truncate(stderr_bytes.decode("utf-8", errors="replace"))
        logger.info(
            "Terminal finished returncode=%s stdout_bytes=%s stderr_bytes=%s",
            process.returncode,
            len(stdout_bytes),
            len(stderr_bytes),
        )

        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": str(cwd_path),
            "shell": shell,
        }

    @staticmethod
    def _truncate(text: str) -> str:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= OUTPUT_BYTE_LIMIT:
            return text
        return encoded[:OUTPUT_BYTE_LIMIT].decode("utf-8", errors="ignore") + "\n... [truncated]"
