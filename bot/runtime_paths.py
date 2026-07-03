from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parent.parent


def runtime_path(value: str | os.PathLike | None, default: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    return default.resolve()


def _configured_path(env_name: str, default: Path) -> Path:
    return runtime_path(os.getenv(env_name), default)


def runtime_data_dir() -> Path:
    return _configured_path("BOT_DATA_DIR", REPO_ROOT / "data")


def runtime_output_dir() -> Path:
    return _configured_path("BOT_OUTPUT_DIR", REPO_ROOT / "output")


def runtime_plots_dir() -> Path:
    return _configured_path("BOT_PLOTS_DIR", runtime_output_dir() / "plots")


def ensure_runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
