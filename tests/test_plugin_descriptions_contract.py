"""Contract test for plugin ``get_spec()`` tool descriptions (Wave 1).

Each tool exposed by an audited plugin must:

1. Have a top-level ``description`` long enough or two-sentence enough to carry
   both "what it does" and "when to call it".
2. Have concrete per-parameter descriptions (not the placeholder phrasings the
   model already knows to ignore).
3. Not reference tool names belonging to *other* plugins (intra-plugin refs are
   fine). Plugins are wired up independently, so cross-plugin references can
   point at tools the model never sees.
4. Not start with template phrases like ``Execute a ...`` that look auto-
   generated.

Plugins still pending audit (Wave 2/3) and a few intentionally exempt sources
(remote-sourced ``vkusvill``, dynamic ``mcp_server`` server-prefixed tools) are
listed in :data:`PENDING_AUDIT_PLUGINS` / :data:`SKIP_PLUGINS`. Move a plugin
out of the allow-list once its descriptions have been rewritten.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pytest

from bot.plugin_manager import PluginManager


# Plugins whose descriptions have been audited and rewritten in Wave 1 — they
# must pass every check in this file.
WAVE1_AUDITED_PLUGINS: frozenset[str] = frozenset({
    "agent_tools",
    "skills",
    "hindsight_memory",
    "reminders",
    "ddg_web_search",
    "google_web_search",
    "jina_web_search",
    "mcp_server",
    "pravo_gov_ru_api",
    "text_document_qa",
})

# Plugins not yet audited. Their tools are loaded but skipped by this contract.
# Bump or shrink this set deliberately as later waves land.
PENDING_AUDIT_PLUGINS: frozenset[str] = frozenset({
    # Wave 2/3 — pending audit
    "agent_cron",
    "ask_your_pdf",
    "auto_tts",
    "chief",
    "codeinterpreter",
    "conversation_analytics",
    "crypto",
    "ddg_image_search",
    "ddg_translate",
    "deepl",
    "dice",
    "github_analysis",
    "haiper_image_to_video",
    "iplocation",
    "language_learning",
    "movie_info",
    "prompt_perfect",
    "reaction",
    "show_me_diagrams",
    "spotify",
    "stable_diffusion",
    "task_management",
    "terminal",
    "text_summarizer",
    "weather",
    "web_research",
    "webshot",
    "website_content",
    "wolfram_alpha",
    "youtube_audio_extractor",
    "youtube_transcript",
})

# Plugins whose specs come from a remote/dynamic source and cannot be audited
# from a static description. ``mcp_server``'s *own* three admin tools are
# audited; the per-server prefixed tools it injects dynamically are filtered
# out by the prefix check below.
SKIP_PLUGINS: frozenset[str] = frozenset({
    "vkusvill",  # remote-sourced MCP adapter
})


# Placeholder phrasings that don't communicate anything specific about a
# parameter. A description equal to (or starting with) one of these — modulo a
# trailing period — should fail the concreteness check.
BLACKLIST_PARAM_PHRASES: tuple[str, ...] = (
    "input string",
    "options",
    "the user query",
    "user query",
    "value",
    "data",
    "text",
    "string",
    "query",
    "argument",
    "arguments",
    "param",
    "parameter",
)

# Template phrases that betray an auto-generated top-level description.
BLACKLIST_TEMPLATE_PHRASES: tuple[str, ...] = (
    "execute a ",
    "the user query",
    "input string",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    # No plugin allow-list ⇒ load every discovered plugin module.
    return PluginManager(config={"plugins": []})


def _instantiate(plugin_manager: PluginManager, plugin_name: str):
    plugin_class = plugin_manager.plugins[plugin_name]
    instance = plugin_class()
    return instance


def _collect_specs(plugin_manager: PluginManager) -> Dict[str, List[Dict]]:
    """Return ``{plugin_name: [spec, ...]}`` for plugins we want to audit.

    Plugins in :data:`SKIP_PLUGINS` and dynamic-tool entries from ``mcp_server``
    (their ``description`` is server-prefixed with ``[server_name]``) are
    filtered out so the contract operates on author-written descriptions only.

    Some plugins refuse to instantiate without external credentials (e.g.
    ``chief`` requires Edamam keys). For Wave 2/3 plugins that is fine — we
    skip them silently. For Wave 1 audited plugins it is a hard error, since
    we expect them to be statically instantiable from this test process.
    """
    by_plugin: Dict[str, List[Dict]] = {}
    for plugin_name in plugin_manager.plugins:
        if plugin_name in SKIP_PLUGINS:
            continue
        try:
            instance = _instantiate(plugin_manager, plugin_name)
            raw_specs = instance.get_spec() or []
        except Exception as exc:
            if plugin_name in WAVE1_AUDITED_PLUGINS:
                pytest.fail(
                    f"Failed to instantiate Wave 1 plugin {plugin_name!r}: {exc!r}"
                )
            # Wave 2/3 plugin without runtime config — skip for now.
            continue
        cleaned: List[Dict] = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                continue
            desc = str(spec.get("description") or "")
            # mcp_server injects per-server tools with description "[server] ..."
            if desc.lstrip().startswith("["):
                continue
            cleaned.append(spec)
        if cleaned:
            by_plugin[plugin_name] = cleaned
    return by_plugin


def _audited_specs(by_plugin: Dict[str, List[Dict]]) -> Iterable[Tuple[str, Dict]]:
    for plugin_name in WAVE1_AUDITED_PLUGINS:
        for spec in by_plugin.get(plugin_name, ()):
            yield plugin_name, spec


def _iter_param_descriptions(properties: Dict) -> Iterable[Tuple[str, str]]:
    """Yield ``(path, description)`` for every leaf property carrying one.

    Walks nested ``object``/``array`` schemas. ``path`` is dotted (e.g.
    ``tasks.items.content``) so failures point at the right place.
    """
    for prop_name, prop_schema in (properties or {}).items():
        if not isinstance(prop_schema, dict):
            continue
        desc = prop_schema.get("description")
        if isinstance(desc, str):
            yield prop_name, desc
        # Recurse into nested objects.
        nested_props = prop_schema.get("properties")
        if isinstance(nested_props, dict):
            for sub_path, sub_desc in _iter_param_descriptions(nested_props):
                yield f"{prop_name}.{sub_path}", sub_desc
        # Recurse into array item schemas.
        items = prop_schema.get("items")
        if isinstance(items, dict):
            item_props = items.get("properties")
            if isinstance(item_props, dict):
                for sub_path, sub_desc in _iter_param_descriptions(item_props):
                    yield f"{prop_name}.items.{sub_path}", sub_desc


def _is_two_sentences(text: str) -> bool:
    """True if ``text`` contains at least one mid-string sentence break."""
    candidate = text.strip()
    # Strip a trailing terminator before counting — a single sentence with a
    # final period should not pass.
    if candidate.endswith("."):
        candidate = candidate[:-1]
    return ". " in candidate


def _normalized_param_desc(desc: str) -> str:
    return desc.strip().rstrip(".").lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_descriptions_have_what_and_when(plugin_manager: PluginManager) -> None:
    """Top-level description must be substantial enough to carry WHAT + WHEN."""
    by_plugin = _collect_specs(plugin_manager)
    failures: List[str] = []
    for plugin_name, spec in _audited_specs(by_plugin):
        desc = str(spec.get("description") or "").strip()
        name = spec.get("name", "<unnamed>")
        if len(desc) < 30:
            failures.append(
                f"{plugin_name}:{name} — description too short ({len(desc)} chars): {desc!r}"
            )
            continue
        # Either >=2 sentences OR long enough that one sentence covers both
        # what and when.
        if not (_is_two_sentences(desc) or len(desc) > 80):
            failures.append(
                f"{plugin_name}:{name} — description is one short sentence ({len(desc)} chars), "
                f"add a 'when to call' sentence: {desc!r}"
            )
    assert not failures, "\n".join(failures)


def test_param_descriptions_concrete(plugin_manager: PluginManager) -> None:
    """Per-parameter descriptions must be present and concrete."""
    by_plugin = _collect_specs(plugin_manager)
    failures: List[str] = []
    for plugin_name, spec in _audited_specs(by_plugin):
        params = spec.get("parameters") or {}
        properties = params.get("properties") or {}
        name = spec.get("name", "<unnamed>")
        for path, desc in _iter_param_descriptions(properties):
            stripped = (desc or "").strip()
            if not stripped:
                failures.append(
                    f"{plugin_name}:{name} — parameter {path!r} has empty description"
                )
                continue
            normalized = _normalized_param_desc(stripped)
            if normalized in BLACKLIST_PARAM_PHRASES:
                failures.append(
                    f"{plugin_name}:{name} — parameter {path!r} uses placeholder "
                    f"description {stripped!r}; replace with something concrete"
                )
    assert not failures, "\n".join(failures)


def test_no_cross_plugin_refs_in_descriptions(plugin_manager: PluginManager) -> None:
    """Audited tool descriptions must not reference tools from other plugins."""
    by_plugin = _collect_specs(plugin_manager)

    # Map plugin -> set of own tool names; and tool_name -> owning plugin.
    plugin_tools: Dict[str, set[str]] = {}
    owner: Dict[str, str] = {}
    for plugin_name, specs in by_plugin.items():
        names = {
            str(spec.get("name") or "").strip()
            for spec in specs
            if isinstance(spec.get("name"), str) and spec.get("name").strip()
        }
        plugin_tools[plugin_name] = names
        for tool_name in names:
            # First registration wins (matches PluginManager dedup order).
            owner.setdefault(tool_name, plugin_name)

    failures: List[str] = []
    for plugin_name, spec in _audited_specs(by_plugin):
        desc = str(spec.get("description") or "")
        own_tools = plugin_tools.get(plugin_name, set())
        name = spec.get("name", "<unnamed>")
        for tool_name, tool_owner in owner.items():
            if tool_owner == plugin_name:
                continue
            if tool_name in own_tools:
                continue
            # Word-boundary check: avoid matching substrings inside longer
            # words. We rely on the tool naming convention (snake_case
            # identifiers) so a simple split on non-identifier chars works.
            tokens = _identifier_tokens(desc)
            if tool_name in tokens:
                failures.append(
                    f"{plugin_name}:{name} description references "
                    f"{tool_name!r} which belongs to plugin {tool_owner!r}"
                )
    assert not failures, "\n".join(failures)


def test_no_template_phrases(plugin_manager: PluginManager) -> None:
    """Top-level descriptions must not start with auto-generated boilerplate."""
    by_plugin = _collect_specs(plugin_manager)
    failures: List[str] = []
    for plugin_name, spec in _audited_specs(by_plugin):
        desc = str(spec.get("description") or "").strip().lower()
        name = spec.get("name", "<unnamed>")
        for phrase in BLACKLIST_TEMPLATE_PHRASES:
            if desc.startswith(phrase) or f" {phrase}" in f" {desc}":
                failures.append(
                    f"{plugin_name}:{name} — description contains template "
                    f"phrase {phrase!r}: {spec.get('description')!r}"
                )
                break
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _identifier_tokens(text: str) -> set[str]:
    """Split ``text`` into identifier-like tokens for word-boundary matching.

    A "token" here is a maximal run of ``[A-Za-z0-9_]`` characters, which is
    exactly the alphabet of our tool names. We avoid a regex compile in a tight
    loop by walking the string directly.
    """
    tokens: set[str] = set()
    buf: List[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            buf.append(ch)
            continue
        if buf:
            tokens.add("".join(buf))
            buf.clear()
    if buf:
        tokens.add("".join(buf))
    return tokens
