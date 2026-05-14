from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

TOOL_METADATA_KEY = "x_tool_metadata"
ARTIFACT_PATH_KEYS = ("value", "file_path", "path", "output_path", "artifact_path")


@dataclass(frozen=True)
class ToolResult:
    payload: Any
    content: str
    success: bool
    error: str | None = None
    direct_result: dict | None = None
    artifacts: tuple[dict, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def json_dict(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except Exception:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def tool_result_content(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def direct_result_payload(value) -> dict | None:
    payload = json_dict(value)
    if not payload:
        return None
    direct_result = payload.get("direct_result")
    if not isinstance(direct_result, dict):
        return None
    return direct_result if direct_result.get("kind") else None


def _artifact_path(value) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip()
    if not path or "\n" in path or "://" in path:
        return None
    return path if os.path.isabs(path) else None


def artifact_entries_from_tool_response(tool_name: str, value) -> tuple[dict, ...]:
    payload = json_dict(value)
    if not payload:
        return ()

    direct_result = direct_result_payload(payload)
    source = direct_result if isinstance(direct_result, dict) else payload
    entries: list[dict] = []
    kind = source.get("kind")
    for key in ARTIFACT_PATH_KEYS:
        path = _artifact_path(source.get(key))
        if path:
            entries.append({
                "tool": tool_name,
                "kind": kind,
                "path": path,
                "format": source.get("format"),
                "caption": source.get("caption") or source.get("add_value"),
            })

    artifacts = source.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_kind = artifact.get("kind") or kind
            for key in ARTIFACT_PATH_KEYS:
                path = _artifact_path(artifact.get(key))
                if path:
                    entries.append({
                        "tool": tool_name,
                        "kind": artifact_kind,
                        "path": path,
                        "format": artifact.get("format"),
                        "caption": artifact.get("caption") or artifact.get("add_value"),
                    })

    return tuple(entries)


def tool_response_succeeded(value) -> bool:
    payload = json_dict(value)
    if isinstance(payload, dict):
        if payload.get("error"):
            return False
        if payload.get("success") is False:
            return False
    return True


def tool_response_error(value) -> str | None:
    payload = json_dict(value)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error:
        return str(error)
    if payload.get("success") is False:
        return str(payload.get("message") or payload.get("result") or "Tool returned success=false")
    return None


def normalize_tool_result(value, *, tool_name: str = "", metadata: dict[str, Any] | None = None) -> ToolResult:
    payload = json_dict(value)
    content = tool_result_content(value)
    direct_result = direct_result_payload(value)
    artifacts = artifact_entries_from_tool_response(tool_name, value)
    success = tool_response_succeeded(value)
    return ToolResult(
        payload=payload if payload is not None else value,
        content=content,
        success=success,
        error=None if success else tool_response_error(value),
        direct_result=direct_result,
        artifacts=artifacts,
        metadata=dict(metadata or {}),
    )
