from __future__ import annotations

from typing import Any, Dict, List

from .plugin import Plugin
from ..hindsight_client import format_recall_results


class HindsightMemoryPlugin(Plugin):
    plugin_id = "hindsight_memory"
    function_prefix = "hindsight_memory"

    def get_source_name(self) -> str:
        return "Hindsight Memory"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "recall",
                "description": "Search the current Telegram user's Hindsight long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Memory search query."},
                        "budget": {
                            "type": "string",
                            "enum": ["low", "mid", "high"],
                            "description": "Recall budget. Use mid by default.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum recall payload size in tokens.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "list_memories",
                "description": "List recently saved memories for the current Telegram user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum memories to return."},
                        "offset": {"type": "integer", "description": "Pagination offset."},
                        "query": {"type": "string", "description": "Optional text filter."},
                        "memory_type": {
                            "type": "string",
                            "description": "Optional memory type filter, such as world or experience.",
                        },
                    },
                },
            },
            {
                "name": "stats",
                "description": "Get Hindsight memory statistics for the current Telegram user.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    async def execute(self, function_name: str, helper: Any, **kwargs) -> Dict:
        if not getattr(helper, "is_hindsight_enabled", lambda: False)():
            return {"error": "Hindsight memory is not configured."}

        user_id = kwargs.get("user_id")
        if not user_id:
            return {"error": "Telegram user_id is required for Hindsight memory."}

        bank_id = helper.get_hindsight_bank_id(user_id)
        client = helper.hindsight_client

        if function_name == "recall":
            data = await client.recall(
                bank_id,
                str(kwargs["query"]),
                budget=str(kwargs.get("budget") or helper.config.get("hindsight_recall_budget", "mid")),
                max_tokens=int(kwargs.get("max_tokens") or helper.config.get("hindsight_recall_max_tokens", 4096)),
                memory_types=helper._hindsight_memory_types(),
            )
            return {
                "bank_id": bank_id,
                "summary": format_recall_results(data),
                "results": data.get("results", []),
            }

        if function_name == "list_memories":
            return await client.list_memories(
                bank_id,
                limit=int(kwargs.get("limit") or 20),
                offset=int(kwargs.get("offset") or 0),
                query=kwargs.get("query"),
                memory_type=kwargs.get("memory_type"),
            )

        if function_name == "stats":
            return await client.stats(bank_id)

        return {"error": f"Unknown Hindsight memory function: {function_name}"}
