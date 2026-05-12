from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            result = self.fn(**tool_input)
            return str(result), False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def as_anthropic(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self.tools.values()]

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        if name not in self.tools:
            return f"Unknown tool: {name}", True
        return self.tools[name].execute(tool_input)
