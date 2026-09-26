import json

from scripts import mypy_baseline


# --- parse_mypy_output ---------------------------------------------------------------

def test_parse_mypy_output_counts_errors_by_file_and_code():
    output = (
        "bot/x.py:10: error: msg one  [arg-type]\n"
        "bot/x.py:20: error: msg two  [arg-type]\n"
        "bot/y.py:5: error: msg three  [return-value]\n"
    )
    counts = mypy_baseline.parse_mypy_output(output)
    assert counts == {
        ("bot/x.py", "arg-type"): 2,
        ("bot/y.py", "return-value"): 1,
    }


def test_parse_mypy_output_ignores_note_lines():
    output = (
        "bot/x.py:10: error: msg  [arg-type]\n"
        "bot/x.py:211: note: By default the bodies of untyped functions are not checked, "
        "consider using --check-untyped-defs  [annotation-unchecked]\n"
    )
    counts = mypy_baseline.parse_mypy_output(output)
    assert counts == {("bot/x.py", "arg-type"): 1}


def test_parse_mypy_output_ignores_summary_line():
    output = (
        "bot/x.py:10: error: msg  [arg-type]\n"
        "Found 2 errors in 1 file (checked 1 source file)\n"
    )
    counts = mypy_baseline.parse_mypy_output(output)
    assert counts == {("bot/x.py", "arg-type"): 1}


def test_parse_mypy_output_handles_missing_code():
    output = "bot/x.py:10: error: msg without a code\n"
    counts = mypy_baseline.parse_mypy_output(output)
    assert counts == {("bot/x.py", "no-code"): 1}


# --- load_baseline / save_baseline ----------------------------------------------------

def test_save_and_load_baseline_roundtrip(tmp_path):
    path = tmp_path / "mypy_baseline.json"
    counts = {("bot/x.py", "arg-type"): 2, ("bot/y.py", "return-value"): 1}
    mypy_baseline.save_baseline(path, counts)
    assert mypy_baseline.load_baseline(path) == counts


def test_load_baseline_missing_file_returns_empty(tmp_path):
    assert mypy_baseline.load_baseline(tmp_path / "nope.json") == {}


# --- diff_counts -----------------------------------------------------------------------

def test_diff_counts_detects_growth():
    baseline = {("bot/x.py", "arg-type"): 1}
    current = {("bot/x.py", "arg-type"): 2}
    regressions, improvements = mypy_baseline.diff_counts(baseline, current)
    assert len(regressions) == 1
    assert "bot/x.py" in regressions[0]
    assert "arg-type" in regressions[0]
    assert improvements == []


def test_diff_counts_detects_new_pair():
    baseline: dict = {}
    current = {("bot/x.py", "arg-type"): 1}
    regressions, improvements = mypy_baseline.diff_counts(baseline, current)
    assert len(regressions) == 1
    assert "bot/x.py" in regressions[0]
    assert improvements == []


def test_diff_counts_detects_decrease():
    baseline = {("bot/x.py", "arg-type"): 3}
    current = {("bot/x.py", "arg-type"): 1}
    regressions, improvements = mypy_baseline.diff_counts(baseline, current)
    assert regressions == []
    assert len(improvements) == 1
    assert "bot/x.py" in improvements[0]


def test_diff_counts_no_change_is_empty_both():
    baseline = {("bot/x.py", "arg-type"): 1}
    current = {("bot/x.py", "arg-type"): 1}
    regressions, improvements = mypy_baseline.diff_counts(baseline, current)
    assert regressions == []
    assert improvements == []


# --- main ------------------------------------------------------------------------------

def test_main_check_exits_1_on_growth(tmp_path, monkeypatch, capsys):
    baseline_path = tmp_path / "mypy_baseline.json"
    mypy_baseline.save_baseline(baseline_path, {("bot/x.py", "arg-type"): 1})
    monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        mypy_baseline, "run_mypy",
        lambda: "bot/x.py:10: error: msg  [arg-type]\nbot/x.py:20: error: msg  [arg-type]\n",
    )
    exit_code = mypy_baseline.main(["check"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "regressions" in captured.out


def test_main_check_exits_1_on_new_pair(tmp_path, monkeypatch, capsys):
    baseline_path = tmp_path / "mypy_baseline.json"
    mypy_baseline.save_baseline(baseline_path, {})
    monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        mypy_baseline, "run_mypy",
        lambda: "bot/x.py:10: error: msg  [arg-type]\n",
    )
    exit_code = mypy_baseline.main(["check"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "regressions" in captured.out


def test_main_check_exits_0_and_suggests_update_on_decrease(tmp_path, monkeypatch, capsys):
    baseline_path = tmp_path / "mypy_baseline.json"
    mypy_baseline.save_baseline(baseline_path, {("bot/x.py", "arg-type"): 3})
    monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        mypy_baseline, "run_mypy",
        lambda: "bot/x.py:10: error: msg  [arg-type]\n",
    )
    exit_code = mypy_baseline.main(["check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "update" in captured.out


def test_main_check_exits_0_when_unchanged(tmp_path, monkeypatch, capsys):
    baseline_path = tmp_path / "mypy_baseline.json"
    mypy_baseline.save_baseline(baseline_path, {("bot/x.py", "arg-type"): 1})
    monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        mypy_baseline, "run_mypy",
        lambda: "bot/x.py:10: error: msg  [arg-type]\n",
    )
    exit_code = mypy_baseline.main(["check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "update" not in captured.out


def test_main_update_writes_expected_json(tmp_path, monkeypatch):
    baseline_path = tmp_path / "mypy_baseline.json"
    monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        mypy_baseline, "run_mypy",
        lambda: "bot/x.py:10: error: msg  [arg-type]\nbot/y.py:5: error: msg  [return-value]\n",
    )
    exit_code = mypy_baseline.main(["update"])
    assert exit_code == 0
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data == {"bot/x.py": {"arg-type": 1}, "bot/y.py": {"return-value": 1}}


def test_main_rejects_unknown_command(capsys):
    exit_code = mypy_baseline.main(["bogus"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "update|check" in (captured.out + captured.err)


# --- regression guard for deleted PluginManager methods --------------------------------

def test_deleted_methods_are_gone():
    from bot.plugin_manager import PluginManager
    assert not hasattr(PluginManager, "is_subagent_function_allowed")
    assert not hasattr(PluginManager, "get_all_plugin_descriptions")
    assert not hasattr(PluginManager, "get_plugin_spec")
