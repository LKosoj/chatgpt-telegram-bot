from __future__ import annotations

from typing import Any

USER_LANGUAGE_SETTING = "language"
USER_TTS_MODEL_SETTING = "tts_model"
USER_TTS_VOICE_SETTING = "tts_voice"
USER_DISABLED_PLUGINS_SETTING = "disabled_plugins"
USER_DISABLED_SKILLS_SETTING = "disabled_skills"


def ensure_settings_dict(settings: Any) -> dict[str, Any]:
    return settings if isinstance(settings, dict) else {}


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({
        item
        for item in (str(item).strip() for item in value)
        if item
    })


def get_user_settings(db: Any, user_id: int | None) -> dict[str, Any]:
    if user_id is None:
        return {}
    getter = getattr(db, "get_user_settings", None)
    if not callable(getter):
        return {}
    return ensure_settings_dict(getter(user_id) or {})


def set_disabled_value(settings: dict[str, Any], key: str, value: str, disabled: bool) -> None:
    values = set(normalize_string_list(settings.get(key)))
    if disabled:
        values.add(value)
    else:
        values.discard(value)
    settings[key] = sorted(values)
