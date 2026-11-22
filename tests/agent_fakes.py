from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from agent.model import ModelResponse


def tool_step(
    name: str,
    arguments: dict[str, Any] | str,
    *,
    input_tokens: int = 32,
    output_tokens: int = 12,
    estimated_cost_usd: float = 0.00001,
) -> ModelResponse:
    raw_arguments = (
        arguments if isinstance(arguments, str) else json.dumps(arguments)
    )
    return ModelResponse(
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": raw_arguments},
                }
            ],
        },
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source="synthetic",
        estimated_cost_usd=estimated_cost_usd,
    )


def final_step(
    text: str,
    *,
    input_tokens: int = 32,
    output_tokens: int = 12,
    estimated_cost_usd: float = 0.00001,
) -> ModelResponse:
    return ModelResponse(
        assistant_message={"role": "assistant", "content": text},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source="synthetic",
        estimated_cost_usd=estimated_cost_usd,
    )


class ScriptedModel:
    is_live = False

    def __init__(self, steps: list[ModelResponse]):
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []
        self._position = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str,
        max_completion_tokens: int,
        allow_tools: bool,
        timeout_seconds: float,
    ) -> ModelResponse:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "allow_tools": allow_tools,
            "timeout_seconds": timeout_seconds,
        })
        if self._position >= len(self.steps):
            raise RuntimeError("The scripted model ran out of responses.")
        response = self.steps[self._position]
        self._position += 1
        message = deepcopy(response.assistant_message)
        for index, tool_call in enumerate(message.get("tool_calls", [])):
            tool_call["id"] = f"call-{self._position}-{index}"
        return ModelResponse(
            assistant_message=message,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            usage_source=response.usage_source,
            estimated_cost_usd=response.estimated_cost_usd,
        )
