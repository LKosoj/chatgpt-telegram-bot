import importlib.util
import sys
import types

import pytest


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_pandas = types.ModuleType("pandas")
_pandas.read_csv = lambda *_args, **_kwargs: None
_pandas.read_excel = lambda *_args, **_kwargs: None
_pandas.read_json = lambda *_args, **_kwargs: None
_pandas.read_parquet = lambda *_args, **_kwargs: None
_pandas.read_pickle = lambda *_args, **_kwargs: None
_install_module_if_missing("pandas", _pandas)

_numpy = types.ModuleType("numpy")
_install_module_if_missing("numpy", _numpy)

_matplotlib = types.ModuleType("matplotlib")
_matplotlib.__path__ = []
_matplotlib.use = lambda *_args, **_kwargs: None
_pyplot = types.ModuleType("matplotlib.pyplot")
_pyplot.close = lambda *_args, **_kwargs: None
_install_module_if_missing("matplotlib", _matplotlib)
_install_module_if_missing("matplotlib.pyplot", _pyplot)

_plotly = types.ModuleType("plotly")
_plotly.__path__ = []
_plotly_express = types.ModuleType("plotly.express")
_install_module_if_missing("plotly", _plotly)
_install_module_if_missing("plotly.express", _plotly_express)

from bot.plugins.codeinterpreter import CodeInterpreterPlugin

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _plugin(tmp_path):
    plugin = object.__new__(CodeInterpreterPlugin)
    plugin.timeout_seconds = 20
    plugin.python_alias_dir = str(tmp_path / "bin")
    plugin.data = None
    return plugin


@pytest.mark.asyncio
async def test_code_prompt_python_is_executed_without_regeneration(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    monkeypatch.chdir(tmp_path)

    async def fail_generate_code(*_args, **_kwargs):
        raise AssertionError("generate_code should not be called for Python code prompts")

    async def explain_code(_code):
        return "explanation"

    monkeypatch.setattr(plugin, "generate_code", fail_generate_code)
    monkeypatch.setattr(plugin, "explain_code", explain_code)
    monkeypatch.setattr(plugin, "advanced_visualization", lambda *_args, **_kwargs: None)

    result = await plugin.run_code(None, "print('hello from code')", "direct", attempts=1)

    assert "hello from code" in result


@pytest.mark.asyncio
async def test_direct_python_code_returns_subprocess_errors_without_internal_debug(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    monkeypatch.chdir(tmp_path)

    async def fail_debug_code(*_args, **_kwargs):
        raise AssertionError("debug_code should not be called for direct Python code")

    monkeypatch.setattr(plugin, "debug_code", fail_debug_code)
    monkeypatch.setattr(plugin, "advanced_visualization", lambda *_args, **_kwargs: None)

    result = await plugin.run_code(
        None,
        "print('STDERR: TypeError: Cannot create property options on string')",
        "direct_error",
        attempts=3,
    )

    assert "TypeError: Cannot create property options on string" in result


@pytest.mark.asyncio
async def test_execute_code_exposes_python_command_alias(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/usr/sbin")

    result = await plugin._execute_code(
        "import subprocess\n"
        "result = subprocess.run(['python', '-c', 'print(\"alias-ok\")'], capture_output=True, text=True)\n"
        "print(result.stdout.strip())\n"
    )

    assert result["__captured_print__"] == "alias-ok"


@pytest.mark.asyncio
async def test_execute_code_captures_system_exit_instead_of_crashing_bot(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = await plugin._execute_code("import sys\nprint('before exit')\nsys.exit(1)\n")

    assert result["error"] == "Выполняемый код вызвал sys.exit(1)"
    assert result["output"] == "before exit"


@pytest.mark.asyncio
async def test_deep_analysis_direct_file_result_also_includes_text_result(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    monkeypatch.chdir(tmp_path)

    async def fake_run_code(_data_path, _code_prompt, _session_id):
        return "STDOUT text"

    monkeypatch.setattr(plugin, "run_code", fake_run_code)
    monkeypatch.setattr(plugin, "advanced_visualization", lambda *_args, **_kwargs: None)

    result = await plugin.execute("deep_analysis", helper=None, code_prompt="print('x')")

    assert result["result"] == "STDOUT text"
    assert result["direct_result"]["kind"] == "file"
