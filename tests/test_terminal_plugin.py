import pytest

from bot.plugins import terminal as terminal_module
from bot.plugins.terminal import BOUNDED_OUTPUT_HINT, TerminalPlugin


@pytest.mark.asyncio
async def test_terminal_truncates_large_stdout_with_bounded_output_hint(monkeypatch):
    monkeypatch.setattr(terminal_module, "OUTPUT_BYTE_LIMIT", 32)
    plugin = TerminalPlugin()

    result = await plugin.execute(
        "terminal",
        helper=None,
        command="printf '%s' 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'",
        shell=True,
    )

    assert result["success"] is True
    assert result["output_truncated"] is True
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is False
    assert result["output_limit_bytes"] == 32
    assert result["stdout"].endswith("\n... [truncated]")
    assert result["output_hint"] == BOUNDED_OUTPUT_HINT


@pytest.mark.asyncio
async def test_terminal_blocks_pip_break_system_packages():
    plugin = TerminalPlugin()

    result = await plugin.execute(
        "terminal",
        helper=None,
        command="pip3 install --break-system-packages Pillow",
        shell=True,
    )

    assert result["success"] is False
    assert result["guard"] == "terminal_command_guard"
    assert "--break-system-packages" in result["error"]


@pytest.mark.asyncio
async def test_terminal_blocks_install_command_piped_to_tail():
    plugin = TerminalPlugin()

    result = await plugin.execute(
        "terminal",
        helper=None,
        command="which python3; pip3 install Pillow 2>&1 | tail -3",
        shell=True,
    )

    assert result["success"] is False
    assert result["guard"] == "terminal_command_guard"
    assert "hides the installer exit code" in result["error"]


@pytest.mark.parametrize(
    "command",
    [
        "npm install 2>&1 | tail -20",
        "apt-get install -y libreoffice 2>&1 | tail -30",
        "uv pip install Pillow | tee install.log | tail -5",
    ],
)
def test_terminal_guard_blocks_install_commands_before_tail(command):
    assert TerminalPlugin._guard_command(command) is not None


def test_terminal_guard_allows_install_log_tail_with_preserved_exit_code():
    command = "pip3 install Pillow >install.log 2>&1; rc=$?; tail -30 install.log; exit $rc"

    assert TerminalPlugin._guard_command(command) is None


@pytest.mark.asyncio
async def test_terminal_allows_install_log_tail_with_preserved_exit_code(tmp_path):
    plugin = TerminalPlugin()
    log_path = tmp_path / "install.log"

    result = await plugin.execute(
        "terminal",
        helper=None,
        command=f"printf 'ok\\n' >{log_path}; rc=$?; tail -30 {log_path}; exit $rc",
        shell=True,
    )

    assert result["success"] is True
    assert result["stdout"] == "ok\n"


def test_terminal_spec_tells_model_to_keep_output_bounded():
    spec = TerminalPlugin().get_spec()[0]

    assert "Keep output bounded" in spec["description"]
    assert "exact known paths" in spec["description"]


@pytest.mark.asyncio
async def test_terminal_timeout_kills_child_processes():
    """Timeout must kill the whole process group, including grandchildren."""
    plugin = TerminalPlugin()

    # Parent shell spawns a child sleep; parent sleeps too.  Timeout fires
    # before either finishes.  If only the shell is killed the child survives
    # and process.wait() would block forever — the test itself times out then.
    result = await plugin.execute(
        "terminal",
        helper=None,
        command="sleep 60 & sleep 60",
        shell=True,
        timeout=0.5,
    )

    assert result["success"] is False
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_terminal_large_output_truncated_by_bounded_reader(monkeypatch):
    """Bounded reader stops after OUTPUT_BYTE_LIMIT bytes and sets truncated flags."""
    monkeypatch.setattr(terminal_module, "OUTPUT_BYTE_LIMIT", 64)
    plugin = TerminalPlugin()

    # yes produces an infinite stream of 'y\n'; the bounded reader must stop it.
    result = await plugin.execute(
        "terminal",
        helper=None,
        command="yes",
        shell=True,
        timeout=5.0,
    )

    assert result["output_truncated"] is True
    assert result["stdout_truncated"] is True
    assert result["stdout"].endswith("\n... [truncated]")
    assert result["output_limit_bytes"] == 64
    # stdout_bytes reflects bytes read, not total written — must be <= limit
    assert result["stdout_bytes"] <= 64


@pytest.mark.asyncio
async def test_terminal_output_exactly_at_limit_not_truncated(monkeypatch):
    """Output whose byte length exactly equals OUTPUT_BYTE_LIMIT must not be truncated."""
    limit = 32
    monkeypatch.setattr(terminal_module, "OUTPUT_BYTE_LIMIT", limit)
    plugin = TerminalPlugin()

    # printf produces exactly 32 bytes, no newline
    payload = "A" * limit
    result = await plugin.execute(
        "terminal",
        helper=None,
        command=f"printf '%s' '{payload}'",
        shell=True,
    )

    assert result["success"] is True
    assert result.get("output_truncated") is not True
    assert result.get("stdout_truncated") is not True
    assert result["stdout"] == payload
