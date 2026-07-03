from __future__ import annotations

import logging
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ChatMode:
    key: str
    data: Dict


class ChatModesRegistry:
    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime: Optional[float] = None
        self._data: Dict[str, Dict] = {}

    def _load_if_needed(self) -> None:
        if not self.path.exists():
            logger.error(f"chat_modes.yml not found at {self.path}")
            self._data = {}
            self._mtime = None
            return

        mtime = self.path.stat().st_mtime_ns
        if self._mtime is None or mtime != self._mtime:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    raise ValueError("chat_modes.yml root must be a mapping")
            except Exception as exc:
                logger.error("Failed to load chat_modes.yml from %s: %s", self.path, exc)
                if self._mtime is None:
                    self._data = {}
                return
            self._data = data
            self._mtime = mtime

    def all_modes(self) -> Dict[str, Dict]:
        self._load_if_needed()
        return copy.deepcopy(self._data)

    def get_mode_by_key(self, key: str) -> Optional[Dict]:
        self._load_if_needed()
        return self._data.get(key)

    def get_mode_by_system_prompt(self, system_content: str | None) -> Optional[Dict]:
        if not system_content:
            return None
        self._load_if_needed()
        for mode_data in self._data.values():
            if not isinstance(mode_data, dict):
                continue
            if mode_data.get("prompt_start", "").strip() == system_content.strip():
                return mode_data
        normalized_content = system_content.lower()
        for mode_data in self._data.values():
            if not isinstance(mode_data, dict):
                continue
            markers = mode_data.get("prompt_markers") or []
            if markers and all(str(marker).lower() in normalized_content for marker in markers):
                return mode_data
        return None

    def get_all_modes_list(self) -> List[str]:
        self._load_if_needed()
        modes = []
        for mode_key, mode_data in self._data.items():
            if isinstance(mode_data, dict) and "welcome_message" in mode_data:
                modes.append(f"name: {mode_key}, welcome_message: {mode_data['welcome_message']}")
        return modes

    def validate_tools(self, plugin_manager) -> None:
        self._load_if_needed()
        for mode_key, mode_data in self._data.items():
            if not isinstance(mode_data, dict):
                logger.error(f"chat_modes.yml: mode '{mode_key}' must be a mapping")
                continue
            tools = mode_data.get("tools", [])
            if not tools:
                continue
            missing = [t for t in tools if t not in ("All", "None") and not plugin_manager.has_plugin(t)]
            if missing:
                logger.error(f"chat_modes.yml: mode '{mode_key}' references missing tools: {missing}")
