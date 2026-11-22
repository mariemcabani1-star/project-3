"""Model provider used by the LangGraph agent node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    assistant_message: dict[str, Any]
    input_tokens: int
    output_tokens: int
    usage_source: str
    estimated_cost_usd: float | None = None


class ModelClient(Protocol):
    is_live: bool

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str,
        max_completion_tokens: int,
        allow_tools: bool,
        timeout_seconds: float,
    ) -> ModelResponse: ...


class OpenAIChatModel:
    is_live = True

    def __init__(self) -> None:
        self._client = None

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
        try:
            from openai import APITimeoutError, OpenAI, OpenAIError
        except ImportError as error:
            raise RuntimeError(
                "The OpenAI SDK is required for live mode. Install requirements.txt."
            ) from error

        try:
            if self._client is None:
                self._client = OpenAI(max_retries=0)

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if allow_tools else "none",
                parallel_tool_calls=False,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout_seconds,
            )
        except APITimeoutError as error:
            raise TimeoutError("The OpenAI request timed out.") from error
        except OpenAIError as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error

        if not response.choices:
            raise RuntimeError("The model returned no response choices.")
        if response.usage is None:
            raise RuntimeError("The model response did not include token usage.")

        message = response.choices[0].message
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

        return ModelResponse(
            assistant_message=assistant_message,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            usage_source="openai",
        )
