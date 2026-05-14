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


def test_terminal_spec_tells_model_to_keep_output_bounded():
    spec = TerminalPlugin().get_spec()[0]

    assert "Keep output bounded" in spec["description"]
    assert "exact known paths" in spec["description"]
