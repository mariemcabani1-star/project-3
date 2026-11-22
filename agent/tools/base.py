"""Tool abstraction: name, strict schema, a `writes` flag, and clear errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class ToolError(Exception):
    """An error the MODEL can understand and act on (never a stack trace).

    code    : stable machine-readable class (maps to errors.md taxonomy)
    message : what went wrong, in plain text
    hint    : what the agent should try next
    """

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema, additionalProperties=false
    fn: Callable[[dict], dict]
    writes: bool = False      # True -> the loop MUST pass the human checkpoint first

    def openai_spec(self) -> dict:
        """Function-calling spec in the OpenAI chat-completions format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
