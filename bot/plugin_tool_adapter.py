from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ai_events import AIToolCall
from .tool_result import ToolResult, normalize_tool_result


@dataclass(frozen=True, slots=True)
class PluginToolAdapter:
    helper: Any
    model_to_use: str
    allowed_plugins: list[str] | None = None

    @property
    def plugin_manager(self):
        return self.helper.plugin_manager

    def get_tools(self):
        return self.plugin_manager.get_functions_specs(
            self.helper,
            self.model_to_use,
            self.allowed_plugins,
        )

    def to_model_name(self, function_name: str) -> str:
        return self.plugin_manager.to_model_function_name(function_name)

    def to_canonical_name(self, function_name: str) -> str:
        return self.plugin_manager.to_canonical_function_name(function_name)

    def is_allowed(self, function_name: str) -> bool:
        return self.plugin_manager.is_function_allowed(
            self.to_canonical_name(function_name),
            self.allowed_plugins,
        )

    def to_canonical_tool_call(self, tool_call: AIToolCall) -> AIToolCall:
        model_name = tool_call.model_name or tool_call.name
        return AIToolCall(
            id=tool_call.id,
            name=self.to_canonical_name(tool_call.name),
            model_name=model_name,
            arguments=tool_call.arguments,
        )

    async def call(self, tool_call: AIToolCall, *, request_context=None) -> Any:
        canonical_tool_call = self.to_canonical_tool_call(tool_call)
        return await self.plugin_manager.call_function(
            canonical_tool_call.name,
            self.helper,
            canonical_tool_call.arguments,
            request_context=request_context,
        )

    @staticmethod
    def normalize_response(tool_response: Any, *, tool_name: str = "") -> ToolResult:
        return normalize_tool_result(tool_response, tool_name=tool_name)
