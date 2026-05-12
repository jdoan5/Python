from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from agent_core.llm import DEFAULT_MODEL, get_client
from agent_core.tools import ToolRegistry

console = Console()


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        system_prompt: str,
        tools: ToolRegistry,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 15,
        max_tokens: int = 8000,
        verbose: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.client = get_client()

    def run(self, user_message: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        tool_calls: list[dict[str, Any]] = []
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        for iteration in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                tools=self.tools.as_anthropic(),
                messages=messages,
            )

            for key in total_usage:
                total_usage[key] += getattr(response.usage, key, 0) or 0

            if self.verbose:
                self._log_iteration(iteration, response)

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return AgentResult(
                    final_text=final_text,
                    iterations=iteration + 1,
                    tool_calls=tool_calls,
                    usage=total_usage,
                )

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    output, is_error = self.tools.execute(block.name, block.input)
                    tool_calls.append(
                        {"name": block.name, "input": block.input, "output": output, "error": is_error}
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                            "is_error": is_error,
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Agent did not converge in {self.max_iterations} iterations")

    def _log_iteration(self, i: int, response: Any) -> None:
        for block in response.content:
            if block.type == "text" and block.text.strip():
                console.print(f"[dim cyan]iter {i}[/dim cyan] {block.text}")
            elif block.type == "tool_use":
                console.print(
                    f"[dim cyan]iter {i}[/dim cyan] [yellow]→ {block.name}[/yellow]({block.input})"
                )
