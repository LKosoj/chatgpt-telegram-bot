from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def validate_openai_config(config: Dict[str, Any]) -> None:
    required = {
        "api_key": str,
        "model": str,
        "enable_functions": bool,
        "max_history_size": int,
        "max_conversation_age_minutes": int,
        "temperature": (int, float),
        "presence_penalty": (int, float),
        "frequency_penalty": (int, float),
    }
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"OpenAI config missing required keys: {missing}")
    for key, expected in required.items():
        if not isinstance(config.get(key), expected):
            raise ValueError(f"OpenAI config '{key}' has invalid type: {type(config.get(key))}")


def validate_function_args(spec: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not spec:
        return errors
    params = spec.get("parameters") or {}
    required = params.get("required") or []
    properties = params.get("properties") or {}

    for key in required:
        if key not in args:
            errors.append(f"Missing required arg '{key}'")

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, prop in properties.items():
        if key not in args:
            continue
        expected_type = type_map.get(prop.get("type"))
        if expected_type and not isinstance(args[key], expected_type):
            errors.append(f"Invalid type for '{key}': expected {prop.get('type')}")
    return errors
