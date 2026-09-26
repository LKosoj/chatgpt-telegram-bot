import tempfile
import time
from pathlib import Path

import pytest

from bot.artifact_paths import artifact_workspace, is_deliverable, is_protected_path


def test_is_deliverable_allows_inside_storage_root(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    file_path = storage_root / "report.txt"
    file_path.write_text("hello", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_is_deliverable_allows_runtime_output_and_plots_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    plots_dir = tmp_path / "plots"
    output_dir.mkdir()
    plots_dir.mkdir()
    monkeypatch.setenv("BOT_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("BOT_PLOTS_DIR", str(plots_dir))

    output_file = output_dir / "chart.png"
    output_file.write_bytes(b"data")
    plots_file = plots_dir / "plot.png"
    plots_file.write_bytes(b"data")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    allowed_output, reason_output = is_deliverable(
        str(output_file), scope="chat:1", storage_root=str(storage_root)
    )
    allowed_plots, reason_plots = is_deliverable(
        str(plots_file), scope="chat:1", storage_root=str(storage_root)
    )

    assert allowed_output is True
    assert reason_output is None
    assert allowed_plots is True
    assert reason_plots is None


def test_is_deliverable_allows_anywhere_under_tempdir(tmp_path, monkeypatch):
    fake_tempdir = tmp_path / "faketemp"
    fake_tempdir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tempdir))

    nested = fake_tempdir / "sub1" / "sub2"
    nested.mkdir(parents=True)
    file_path = nested / "artifact.bin"
    file_path.write_bytes(b"data")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_is_deliverable_allows_uploads_webshot_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    upload_dir = tmp_path / "uploads" / "webshot"
    upload_dir.mkdir(parents=True)
    file_path = upload_dir / "shot.png"
    file_path.write_bytes(b"data")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_is_deliverable_rejects_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=1", encoding="utf-8")

    allowed, reason = is_deliverable(str(env_file), scope="chat:1")

    assert allowed is False
    assert reason == ".env"


def test_is_deliverable_rejects_usage_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    usage_logs = tmp_path / "usage_logs"
    usage_logs.mkdir()
    file_path = usage_logs / "1.json"
    file_path.write_text("{}", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:1")

    assert allowed is False
    assert reason == "usage_logs directory"


def test_is_deliverable_rejects_bare_json_in_storage_root_root(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    mcp_servers = storage_root / "mcp_servers.json"
    mcp_servers.write_text("{}", encoding="utf-8")
    reminders = storage_root / "reminders.json"
    reminders.write_text("{}", encoding="utf-8")

    allowed_mcp, reason_mcp = is_deliverable(str(mcp_servers), scope="chat:1", storage_root=str(storage_root))
    allowed_reminders, reason_reminders = is_deliverable(
        str(reminders), scope="chat:1", storage_root=str(storage_root)
    )

    assert allowed_mcp is False
    assert reason_mcp == "bare json/jsonl in storage root"
    assert allowed_reminders is False
    assert reason_reminders == "bare json/jsonl in storage root"

    nested_dir = storage_root / "sub"
    nested_dir.mkdir()
    nested_json = nested_dir / "reminders.json"
    nested_json.write_text("{}", encoding="utf-8")

    allowed_nested, reason_nested = is_deliverable(
        str(nested_json), scope="chat:1", storage_root=str(storage_root)
    )

    assert allowed_nested is True
    assert reason_nested is None


def test_is_deliverable_rejects_skills_source_dir(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    file_path = skills_dir / "writing" / "SKILL.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("# skill", encoding="utf-8")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is False
    assert reason == "skills source directory"


def test_is_deliverable_rejects_db_path_and_wal_shm_journal(tmp_path, monkeypatch):
    db_path = tmp_path / "user_data.db"
    db_path.write_bytes(b"sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))

    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = tmp_path / f"user_data.db{suffix}"
        candidate.write_bytes(b"data")
        allowed, reason = is_deliverable(str(candidate), scope="chat:1")
        assert allowed is False
        assert reason == "database file"


def test_is_deliverable_rejects_db_path_via_symlink(tmp_path, monkeypatch):
    real_db = tmp_path / "real.db"
    real_db.write_bytes(b"sqlite")
    monkeypatch.setenv("DB_PATH", str(real_db))
    link_db = tmp_path / "link.db"
    link_db.symlink_to(real_db)

    allowed, reason = is_deliverable(str(link_db), scope="chat:1")

    assert allowed is False
    assert reason == "database file"


def test_is_deliverable_deny_wins_over_tempdir_fallback(tmp_path, monkeypatch):
    fake_tempdir = tmp_path / "faketemp"
    fake_tempdir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tempdir))
    db_path = fake_tempdir / "user_data.db"
    db_path.write_bytes(b"sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))

    allowed, reason = is_deliverable(str(db_path), scope="chat:1")

    assert allowed is False
    assert reason == "database file"


def test_is_deliverable_allows_matching_scope_artifact_workspace(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    workspace = artifact_workspace(str(storage_root), "chat:1")
    workspace.mkdir(parents=True)
    file_path = workspace / "out.txt"
    file_path.write_text("hi", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_is_deliverable_rejects_other_scope_artifact_workspace(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    workspace = artifact_workspace(str(storage_root), "chat:1")
    workspace.mkdir(parents=True)
    file_path = workspace / "out.txt"
    file_path.write_text("hi", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:2", storage_root=str(storage_root))

    assert allowed is False
    assert reason == "artifact belongs to a different delivery scope"


def test_is_deliverable_allows_matching_scope_skills_workdir(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    workdir_root = storage_root / "skill_workdir"
    scope_dir = workdir_root / "writing" / "chat_1"
    scope_dir.mkdir(parents=True)
    file_path = scope_dir / "out.txt"
    file_path.write_text("hi", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_is_deliverable_rejects_other_scope_skills_workdir(tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    workdir_root = storage_root / "skill_workdir"
    scope_dir = workdir_root / "writing" / "chat_1"
    scope_dir.mkdir(parents=True)
    file_path = scope_dir / "out.txt"
    file_path.write_text("hi", encoding="utf-8")

    allowed, reason = is_deliverable(str(file_path), scope="chat:2", storage_root=str(storage_root))

    assert allowed is False
    assert reason == "artifact belongs to a different skills workdir scope"


def test_is_deliverable_request_started_at_allows_recent_file_outside_known_roots(tmp_path, monkeypatch):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    fake_tempdir = tmp_path / "faketemp"
    fake_tempdir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tempdir))
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    started_at = time.time()
    file_path = outside_dir / "recent.bin"
    file_path.write_bytes(b"data")

    allowed, reason = is_deliverable(
        str(file_path),
        scope="chat:1",
        request_started_at=started_at,
        storage_root=str(storage_root),
    )

    assert allowed is True
    assert reason is None


def test_is_deliverable_missing_file_inside_known_root_is_allowed(tmp_path, monkeypatch):
    fake_tempdir = tmp_path / "faketemp"
    fake_tempdir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tempdir))

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    missing_path = fake_tempdir / "missing.bin"

    allowed, reason = is_deliverable(str(missing_path), scope="chat:1", storage_root=str(storage_root))

    assert allowed is True
    assert reason is None


def test_artifact_workspace_path_shape():
    assert artifact_workspace("/data", "chat:10") == Path("/data/artifacts/chat_10")


@pytest.mark.parametrize(
    "make_path",
    [
        pytest.param(lambda tmp_path, monkeypatch: _protected_env_path(tmp_path, monkeypatch), id="env_file"),
        pytest.param(lambda tmp_path, monkeypatch: _protected_usage_logs_path(tmp_path, monkeypatch), id="usage_logs"),
        pytest.param(lambda tmp_path, monkeypatch: _protected_db_path(tmp_path, monkeypatch), id="db_path"),
        pytest.param(lambda tmp_path, monkeypatch: _protected_skills_dir_path(tmp_path, monkeypatch), id="skills_dir"),
    ],
)
def test_is_protected_path_matches_is_deliverable_deny_set(tmp_path, monkeypatch, make_path):
    path, storage_root = make_path(tmp_path, monkeypatch)

    kwargs = {"storage_root": storage_root} if storage_root else {}
    assert is_protected_path(str(path), **kwargs) is True

    allowed_path = tmp_path / "allowed" / "report.txt"
    allowed_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_path.write_text("ok", encoding="utf-8")
    assert is_protected_path(str(allowed_path), **kwargs) is False


def _protected_env_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=1", encoding="utf-8")
    return env_file, None


def _protected_usage_logs_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    usage_logs = tmp_path / "usage_logs"
    usage_logs.mkdir()
    file_path = usage_logs / "1.json"
    file_path.write_text("{}", encoding="utf-8")
    return file_path, None


def _protected_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "user_data.db"
    db_path.write_bytes(b"sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))
    return db_path, None


def _protected_skills_dir_path(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("SKILLS_DIR", str(skills_dir))
    file_path = skills_dir / "writing" / "SKILL.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("# skill", encoding="utf-8")
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    return file_path, str(storage_root)
