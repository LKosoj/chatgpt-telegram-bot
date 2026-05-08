from __future__ import annotations

import re

SKILL_SCRIPT_PATH_RE = re.compile(
    r"(?:^|[\s'\"`=(:])(?:[A-Za-z]:)?[^\s'\"`]*skills[/\\][^\s'\"`/\\]+[/\\]scripts[/\\][^\s'\"`]+",
    re.IGNORECASE,
)
SCRIPT_FILE_CREATION_RE = re.compile(
    r"\b(?:write|create|save)\b[\s\S]{0,240}\b(?:javascript|js|python|shell|bash|node)\b"
    r"[\s\S]{0,240}(?:/tmp/|\.js\b|\.py\b|\.sh\b)"
    r"|\bfs\.writeFileSync\b",
    re.IGNORECASE,
)


def _system_message(helper, chat_id) -> dict | None:
    messages = getattr(helper, "conversations", {}).get(chat_id, [])
    return next((msg for msg in messages if msg.get("role") == "system"), None)


def _is_skills_agent_mode(helper, chat_id) -> bool:
    system_message = _system_message(helper, chat_id)
    if not system_message:
        return False
    if system_message.get("mode_key") == "skills_agent":
        return True

    mode_from_system_message = getattr(helper, "_mode_from_system_message", None)
    registry = getattr(helper, "chat_modes_registry", None)
    get_mode_by_key = getattr(registry, "get_mode_by_key", None)
    if not callable(mode_from_system_message) or not callable(get_mode_by_key):
        return False

    current_mode = mode_from_system_message(system_message)
    skills_mode = get_mode_by_key("skills_agent")
    return bool(current_mode and skills_mode and current_mode == skills_mode)


def _active_skill_scripts(helper, tool_args: dict) -> list[dict]:
    plugin_manager = getattr(helper, "plugin_manager", None)
    get_plugin = getattr(plugin_manager, "get_plugin", None)
    if not callable(get_plugin):
        return []

    skills_plugin = get_plugin("skills")
    if not skills_plugin:
        return []

    scope_key = getattr(skills_plugin, "_scope_key", None)
    if not callable(scope_key):
        return []

    scope = scope_key(tool_args)
    active_skills = getattr(skills_plugin, "active_skills", {}).get(scope, {})
    available_skills = getattr(skills_plugin, "available_skills", {})
    scripts = []
    for skill_id in sorted(active_skills):
        for script_name in available_skills.get(skill_id, {}).get("scripts", []):
            scripts.append({
                "skill_name": str(skill_id),
                "script_name": str(script_name),
            })
    return scripts


def _skill_script_routing_payload(error: str, active_scripts: list[dict] | None = None) -> dict:
    payload = {
        "error": error,
        "required_tool": "skills.run_skill_script",
    }
    if active_scripts:
        payload["call_instruction"] = (
            "Call skills.run_skill_script with one of the available skill_name/script_name pairs."
        )
        payload["available_skill_scripts"] = active_scripts
        if len(active_scripts) == 1:
            payload["suggested_tool_call"] = {
                "tool_name": "skills.run_skill_script",
                "arguments": active_scripts[0],
            }
    return payload


def _refers_to_active_script(text: str, active_scripts: list[dict]) -> bool:
    if not text:
        return False
    for entry in active_scripts:
        script_name = str(entry.get("script_name") or "").strip()
        if not script_name:
            continue
        if script_name in text:
            return True
        basename = script_name.rsplit("/", 1)[-1]
        if basename and basename != script_name and basename in text:
            return True
    return False


def _skill_script_routing_error(helper, chat_id, tool_name: str, tool_args: dict) -> dict | None:
    if tool_name != "codeinterpreter.deep_analysis":
        return None

    text = "\n".join(
        str(tool_args.get(field) or "")
        for field in ("code_prompt", "data_path")
    )

    active_scripts = _active_skill_scripts(helper, tool_args)
    if active_scripts and _refers_to_active_script(text, active_scripts):
        return _skill_script_routing_payload(
            "Routing denied: this script belongs to an active skill and must be run "
            "through skills.run_skill_script, not codeinterpreter.deep_analysis.",
            active_scripts,
        )

    if not _is_skills_agent_mode(helper, chat_id):
        return None

    if SKILL_SCRIPT_PATH_RE.search(text):
        return _skill_script_routing_payload(
            "Routing denied in skills_agent: skill scripts must be executed through "
            "skills.run_skill_script, not codeinterpreter.deep_analysis."
        )

    if SCRIPT_FILE_CREATION_RE.search(text):
        return _skill_script_routing_payload(
            "Routing denied in skills_agent: ad-hoc script files must not be created or "
            "executed through codeinterpreter.deep_analysis. Use active skill scripts via "
            "skills.run_skill_script, then deliver artifacts through agent_tools.deliver_to_user."
        )

    return None
