"""T12d D4: _clone_git_source / _clone_git_source_branch in bot/plugins/skills.py
were byte-identical except for the "-b <branch>" insertion, so they were merged
into a single `_clone_git_source(source, temp_dir, branch=None)`. No test
previously exercised the constructed subprocess command directly (existing
tests only monkeypatch the whole method away), so this pins down the actual
`git clone` command for both the branch and no-branch cases.
"""
from types import SimpleNamespace

import bot.plugins.skills as skills
from bot.plugins.skills import SkillsPlugin


def _make_plugin(tmp_path):
    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(tmp_path / "storage"))
    return plugin


def _fake_which(monkeypatch):
    monkeypatch.setattr(skills.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None)


def _capture_run(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(skills.subprocess, "run", fake_run)
    return captured


def test_clone_git_source_branch_includes_branch_flag(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path)
    _fake_which(monkeypatch)
    captured = _capture_run(monkeypatch)

    path, err = plugin._clone_git_source(
        "https://example.com/repo.git", tmp_path, branch="feature-x"
    )

    assert err is None
    assert path == tmp_path / "git-source"
    assert "-b" in captured["command"]
    branch_index = captured["command"].index("-b")
    assert captured["command"][branch_index + 1] == "feature-x"


def test_clone_git_source_plain_has_no_branch_flag(tmp_path, monkeypatch):
    plugin = _make_plugin(tmp_path)
    _fake_which(monkeypatch)
    captured = _capture_run(monkeypatch)

    path, err = plugin._clone_git_source("https://example.com/repo.git", tmp_path)

    assert err is None
    assert path == tmp_path / "git-source"
    assert "-b" not in captured["command"]
