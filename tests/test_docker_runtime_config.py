from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_as_non_root_and_declares_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY . ." in dockerfile
    assert "COPY --chown=bot:bot . ." not in dockerfile
    assert re.search(r"^USER\s+bot\s*$", dockerfile, re.MULTILINE)
    assert not re.search(r"^USER\s+root\s*$", dockerfile, re.MULTILINE)
    assert "DB_PATH=/app/data/user_data.db" in dockerfile
    assert "PLUGIN_STORAGE_ROOT=/app/data" in dockerfile
    assert "SESSION_LOG_DIR=/app/log" in dockerfile
    assert "BOT_DATA_DIR=/app/data" in dockerfile
    assert "BOT_OUTPUT_DIR=/app/output" in dockerfile
    assert "BOT_PLOTS_DIR=/app/plots" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/proc/1/cmdline" in dockerfile
    for writable_path in (
        "data",
        "output/plots",
        "plots",
        "usage_logs",
        "log",
        "bot/temp",
        "uploads/webshot",
        "bot/plugins/temp_pdfs",
        "bot/plugins/pdf_cache",
        "bot/plugins/document_metadata",
        "bot/plugins/analytics",
        "bot/plugins/language_data",
    ):
        assert writable_path in dockerfile


def test_compose_uses_env_file_and_explicit_writable_volumes():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["chatgpt-telegram-bot"]

    assert service["env_file"] == [".env"]
    volumes = service["volumes"]
    assert ".:/app" not in volumes
    assert "./:/app" not in volumes

    environment = service["environment"]
    assert "/app/data/user_data.db" in environment["DB_PATH"]
    assert "/app/data" in environment["PLUGIN_STORAGE_ROOT"]
    assert "/app/log" in environment["SESSION_LOG_DIR"]
    assert "/app/data" in environment["BOT_DATA_DIR"]
    assert "/app/output" in environment["BOT_OUTPUT_DIR"]
    assert "/app/plots" in environment["BOT_PLOTS_DIR"]
    assert environment["SKILLS_DIR"] == "/app/data/skills"
    assert environment["SKILLS_WORKDIR"] == "/app/data/skill_workdir"

    expected_mounts = {
        "bot-data": "/app/data",
        "bot-output": "/app/output",
        "bot-plots": "/app/plots",
        "bot-usage-logs": "/app/usage_logs",
        "bot-session-logs": "/app/log",
        "bot-uploads": "/app/uploads",
    }
    actual_mounts = {
        source: target
        for source, target in (volume.split(":", 1) for volume in volumes)
    }
    assert actual_mounts == expected_mounts
    assert set(compose["volumes"]) == set(expected_mounts)
    assert sorted(service["tmpfs"]) == [
        "/app/bot/temp:rw,nosuid,nodev,mode=1777",
        "/tmp:rw,nosuid,nodev,mode=1777",
    ]


def test_dockerignore_excludes_runtime_state_from_image_context():
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {
        "bot/user_data.db",
        "bot/user_data.db-shm",
        "bot/user_data.db-wal",
        "bot/temp",
        "data",
        "log",
        "media",
        "output",
        "plots",
        "uploads",
        "usage_logs",
    } <= ignored
