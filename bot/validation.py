from __future__ import annotations

import logging
from typing import Any, Dict, List

from jsonschema import Draft7Validator, SchemaError

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
            raise ValueError(
                f"OpenAI config '{key}' has invalid type: "
                f"{type(config.get(key))}"
            )


def validate_function_args(
    spec: Dict[str, Any],
    args: Dict[str, Any],
) -> List[str]:
    if not spec:
        return []
    params = spec.get("parameters") or {}

    try:
        Draft7Validator.check_schema(params)
        validator = Draft7Validator(params)
        validation_errors = sorted(
            validator.iter_errors(args),
            key=lambda error: list(error.path),
        )
    except SchemaError as e:
        return [f"Invalid schema: {e.message}"]

    return [_format_validation_error(error) for error in validation_errors]


def _format_validation_error(error) -> str:
    if error.validator == "required":
        missing = _required_property_name(error.message)
        path = _join_path(error.path)
        arg_name = missing if not path else f"{path}.{missing}"
        return f"Missing required arg '{arg_name}'"

    path = _join_path(error.path) or "arguments"

    if error.validator == "type":
        expected = error.schema.get("type")
        return f"Invalid type for '{path}': expected {expected}"

    if error.validator == "enum":
        allowed = ", ".join(repr(value) for value in error.validator_value)
        return f"Invalid value for '{path}': expected one of [{allowed}]"

    if error.validator == "additionalProperties":
        return f"Unexpected arg for '{path}': {error.message}"

    return f"Invalid value for '{path}': {error.message}"


def _join_path(path) -> str:
    parts = []
    for item in path:
        if isinstance(item, int) and parts:
            parts[-1] = f"{parts[-1]}[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)


def _required_property_name(message: str) -> str:
    parts = message.split("'")
    if len(parts) >= 2:
        return parts[1]
    return message
