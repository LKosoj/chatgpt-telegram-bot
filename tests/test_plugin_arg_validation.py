import asyncio
import json
import textwrap
from pathlib import Path

from bot.plugin_manager import PluginManager
from bot.validation import validate_function_args


def _schema(properties, required=None):
    return {
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        }
    }


def test_jsonschema_rejects_bool_for_integer():
    spec = _schema({"count": {"type": "integer"}})

    errors = validate_function_args(spec, {"count": True})

    assert errors == ["Invalid type for 'count': expected integer"]


def test_jsonschema_rejects_bool_for_number():
    spec = _schema({"ratio": {"type": "number"}})

    errors = validate_function_args(spec, {"ratio": True})

    assert errors == ["Invalid type for 'ratio': expected number"]


def test_jsonschema_rejects_invalid_enum():
    spec = _schema({"mode": {"type": "string", "enum": ["fast", "safe"]}})

    errors = validate_function_args(spec, {"mode": "unsafe"})

    assert errors == [
        "Invalid value for 'mode': expected one of ['fast', 'safe']"
    ]


def test_jsonschema_reports_missing_required():
    spec = _schema({"text": {"type": "string"}}, required=["text"])

    errors = validate_function_args(spec, {})

    assert "Missing required arg 'text'" in errors


def test_jsonschema_rejects_invalid_nested_object_property():
    spec = _schema(
        {
            "filters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
            }
        }
    )

    errors = validate_function_args(spec, {"filters": {"limit": "10"}})

    assert errors == ["Invalid type for 'filters.limit': expected integer"]


def test_jsonschema_rejects_invalid_array_item_type():
    spec = _schema(
        {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            }
        }
    )

    errors = validate_function_args(spec, {"tags": ["valid", 3]})

    assert errors == ["Invalid type for 'tags[1]': expected string"]


def _write_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class DemoPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Demo"

    def get_spec(self):
        return [{
            "name": "do",
            "description": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"}
                },
                "required": ["text"]
            }
        }]

    async def execute(self, function_name, helper, **kwargs):
        return {"result": "ok"}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def _write_strict_plugin(path: Path):
    code = """
from bot.plugins.plugin import Plugin

class StrictPlugin(Plugin):
    def get_source_name(self) -> str:
        return "Strict"

    def get_spec(self):
        return [{
            "name": "do",
            "description": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }]

    async def execute(self, function_name, helper, **kwargs):
        return {"text": kwargs["text"], "chat_id": kwargs["chat_id"], "user_id": kwargs["user_id"]}
"""
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def test_plugin_arg_validation(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir / "demo.py")

    pm = PluginManager(
        config={"plugins": []},
        plugins_directory=str(plugin_dir),
    )
    bad_args = json.dumps({"count": 1})
    result = asyncio.run(pm.call_function("demo.do", None, bad_args))
    assert "Missing required arg" in result


def test_plugin_arg_validation_ignores_framework_args_for_strict_schema(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_strict_plugin(plugin_dir / "strict.py")

    pm = PluginManager(
        config={"plugins": []},
        plugins_directory=str(plugin_dir),
    )
    result = asyncio.run(pm.call_function(
        "strict.do",
        None,
        json.dumps({"text": "ok", "chat_id": 10, "user_id": 42}),
    ))

    assert json.loads(result) == {"text": "ok", "chat_id": 10, "user_id": 42}
