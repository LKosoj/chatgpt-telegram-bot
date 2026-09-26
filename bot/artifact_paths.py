from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .runtime_paths import runtime_output_dir, runtime_plots_dir

_DENY_JSON_SUFFIXES = {".json", ".jsonl"}


def _is_relative_to(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b)


def _safe_scope(scope: str) -> str:
    return scope.replace(":", "_").replace("/", "_") or "global"


def _default_storage_root() -> Path:
    storage_root = os.getenv("PLUGIN_STORAGE_ROOT")
    if storage_root:
        return Path(storage_root).resolve()
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def _default_db_path() -> Path:
    db_path = os.getenv("DB_PATH")
    if db_path:
        return Path(db_path).resolve()
    return (Path(__file__).resolve().parent / "user_data.db").resolve()


def _default_skills_dir(storage_root: Path) -> Path:
    skills_dir = os.getenv("SKILLS_DIR")
    if skills_dir:
        return Path(skills_dir).expanduser().resolve()
    return (storage_root / "skills").resolve()


def _default_skills_workdir_root(storage_root: Path) -> Path:
    workdir_env = os.getenv("SKILLS_WORKDIR")
    if workdir_env:
        return Path(workdir_env).expanduser().resolve()
    return (storage_root / "skill_workdir").resolve()


def artifact_workspace(storage_root: str, scope: str) -> Path:
    """<storage_root>/artifacts/<safe_scope>. safe_scope — та же трансформация, что и
    skills._ensure_skill_workdir (bot/plugins/skills.py):
    scope.replace(":", "_").replace("/", "_") or "global"."""
    return Path(storage_root) / "artifacts" / _safe_scope(scope)


def _deny_reason(resolved: Path, effective_storage_root: Path) -> str | None:
    db = _default_db_path()
    if resolved in {
        db,
        db.with_name(db.name + "-wal"),
        db.with_name(db.name + "-shm"),
        db.with_name(db.name + "-journal"),
    }:
        return "database file"
    if resolved == (Path.cwd() / ".env").resolve():
        return ".env"
    if _is_relative_to(resolved, (Path.cwd() / "usage_logs").resolve()):
        return "usage_logs directory"
    if resolved.parent == effective_storage_root and resolved.suffix.lower() in _DENY_JSON_SUFFIXES:
        return "bare json/jsonl in storage root"
    skills_dir = _default_skills_dir(effective_storage_root)
    if _is_relative_to(resolved, skills_dir):
        return "skills source directory"
    return None


def is_protected_path(path: str, *, storage_root: str | None = None) -> bool:
    """True, если resolved(path) попадает в жёсткий deny-список (БД + -wal/-shm/-journal,
    .env, usage_logs/, голый *.json|*.jsonl прямо в корне storage_root, исходники skills).
    Не требует scope — используется как для is_deliverable, так и для cleanup_intermediate_files
    (защита от удаления после отказа в доставке)."""
    resolved = Path(os.path.realpath(os.path.expanduser(path)))
    effective_storage_root = Path(storage_root).resolve() if storage_root else _default_storage_root()
    return _deny_reason(resolved, effective_storage_root) is not None


def is_deliverable(
    path: str,
    *,
    scope: str,
    request_started_at: float | None = None,
    storage_root: str | None = None,
) -> tuple[bool, str | None]:
    """(allowed, reason). reason всегда None при allowed=True."""
    resolved = Path(os.path.realpath(os.path.expanduser(path)))
    effective_storage_root = Path(storage_root).resolve() if storage_root else _default_storage_root()

    deny_reason = _deny_reason(resolved, effective_storage_root)
    if deny_reason:
        return False, deny_reason

    artifacts_root = artifact_workspace(str(effective_storage_root), scope).parent
    if _is_relative_to(resolved, artifacts_root):
        expected = artifact_workspace(str(effective_storage_root), scope)
        if resolved == expected or _is_relative_to(resolved, expected):
            return True, None
        return False, "artifact belongs to a different delivery scope"

    workdir_root = _default_skills_workdir_root(effective_storage_root)
    if _is_relative_to(resolved, workdir_root):
        parts = resolved.relative_to(workdir_root).parts
        safe = _safe_scope(scope)
        if len(parts) >= 2 and parts[1] == safe:
            return True, None
        return False, "artifact belongs to a different skills workdir scope"

    if _is_relative_to(resolved, effective_storage_root):
        return True, None
    if _is_relative_to(resolved, runtime_output_dir()) or _is_relative_to(resolved, runtime_plots_dir()):
        return True, None
    if request_started_at is not None:
        try:
            if resolved.stat().st_mtime >= request_started_at - 2:
                return True, None
        except OSError:
            return True, None
    if _is_relative_to(resolved, Path(tempfile.gettempdir()).resolve()):
        return True, None
    if _is_relative_to(resolved, (Path.cwd() / "uploads" / "webshot").resolve()):
        return True, None

    return False, "outside allowed delivery locations"
