
from bot.runtime_paths import (
    REPO_ROOT,
    runtime_data_dir,
    runtime_output_dir,
    runtime_plots_dir,
)


def test_runtime_path_defaults_are_repo_rooted_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOT_DATA_DIR", raising=False)
    monkeypatch.delenv("BOT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("BOT_PLOTS_DIR", raising=False)

    assert runtime_data_dir() == (REPO_ROOT / "data").resolve()
    assert runtime_output_dir() == (REPO_ROOT / "output").resolve()
    assert runtime_plots_dir() == (REPO_ROOT / "output" / "plots").resolve()


def test_relative_runtime_path_env_values_are_repo_rooted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_DATA_DIR", "runtime-data")
    monkeypatch.setenv("BOT_OUTPUT_DIR", "runtime-output")
    monkeypatch.setenv("BOT_PLOTS_DIR", "runtime-plots")

    assert runtime_data_dir() == (REPO_ROOT / "runtime-data").resolve()
    assert runtime_output_dir() == (REPO_ROOT / "runtime-output").resolve()
    assert runtime_plots_dir() == (REPO_ROOT / "runtime-plots").resolve()


def test_absolute_runtime_path_env_values_are_used(tmp_path, monkeypatch):
    data_dir = tmp_path / "data-root"
    output_dir = tmp_path / "output-root"
    plots_dir = tmp_path / "plots-root"
    monkeypatch.setenv("BOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BOT_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("BOT_PLOTS_DIR", str(plots_dir))

    assert runtime_data_dir() == data_dir.resolve()
    assert runtime_output_dir() == output_dir.resolve()
    assert runtime_plots_dir() == plots_dir.resolve()
