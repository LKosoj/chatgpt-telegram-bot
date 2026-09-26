"""Coverage for bot/openai_tool_handler.py's _append_artifact_entry (and the
_artifact_path validator it relies on), written before T12-plan.md T12c
C-extra2 removes the module's private duplicate of tool_result.py's
_artifact_path in favor of importing the one canonical implementation. No
direct test existed for this call site before (tests/test_artifact_paths.py
covers the unrelated bot/artifact_paths.py module); this pins the four path
shapes _artifact_path is expected to distinguish, exercised through the
actual manifest-building call site.
"""

from __future__ import annotations

from bot.openai_tool_handler import _append_artifact_entry


def test_append_artifact_entry_accepts_absolute_path():
    manifest: list[dict] = []
    seen: set[str] = set()

    _append_artifact_entry(manifest, seen, {"path": "/tmp/output/report.png", "kind": "image"})

    assert manifest == [{"path": "/tmp/output/report.png", "kind": "image"}]
    assert seen == {"/tmp/output/report.png"}


def test_append_artifact_entry_rejects_path_with_newline():
    manifest: list[dict] = []
    seen: set[str] = set()

    _append_artifact_entry(manifest, seen, {"path": "/tmp/output/report.png\nrm -rf /", "kind": "image"})

    assert manifest == []
    assert seen == set()


def test_append_artifact_entry_rejects_relative_path():
    manifest: list[dict] = []
    seen: set[str] = set()

    _append_artifact_entry(manifest, seen, {"path": "relative/report.png", "kind": "image"})

    assert manifest == []
    assert seen == set()


def test_append_artifact_entry_rejects_url():
    manifest: list[dict] = []
    seen: set[str] = set()

    _append_artifact_entry(manifest, seen, {"path": "https://example.com/report.png", "kind": "image"})

    assert manifest == []
    assert seen == set()


def test_append_artifact_entry_dedupes_seen_path():
    manifest: list[dict] = []
    seen: set[str] = set()

    _append_artifact_entry(manifest, seen, {"path": "/tmp/output/report.png", "kind": "image"})
    _append_artifact_entry(manifest, seen, {"path": "/tmp/output/report.png", "kind": "image"})

    assert len(manifest) == 1
