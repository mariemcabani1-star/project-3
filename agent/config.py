"""Runtime bounds and model-cost configuration."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

LIVE_MODEL = "gpt-4o-mini"
MAX_REQUEST_COST_USD = 0.10


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value greater than zero.")
    return value


def _optional_rate(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    rate = float(raw)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return rate


@dataclass(frozen=True)
class AgentSettings:
    model: str = LIVE_MODEL
    max_model_turns: int = 8
    max_tool_calls: int = 15
    max_runtime_seconds: float = 60.0
    max_cost_usd: float = MAX_REQUEST_COST_USD
    max_completion_tokens: int = 512
    input_usd_per_1m_tokens: float | None = None
    output_usd_per_1m_tokens: float | None = None

    def __post_init__(self) -> None:
        if self.model != LIVE_MODEL:
            raise ValueError(f"Live mode is fixed to {LIVE_MODEL}.")
        if self.max_model_turns < 1 or self.max_tool_calls < 1:
            raise ValueError("Model-turn and tool-call limits must be positive.")
        if self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive.")
        for name, value in (
            ("max_runtime_seconds", self.max_runtime_seconds),
            ("max_cost_usd", self.max_cost_usd),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite value greater than zero.")
        if self.max_cost_usd > MAX_REQUEST_COST_USD:
            raise ValueError(
                f"Per-request cost cannot exceed ${MAX_REQUEST_COST_USD:.2f}."
            )
        for name, value in (
            ("input_usd_per_1m_tokens", self.input_usd_per_1m_tokens),
            ("output_usd_per_1m_tokens", self.output_usd_per_1m_tokens),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative value.")

    @classmethod
    def from_env(cls) -> AgentSettings:
        return cls(
            max_model_turns=_positive_int("MAX_MODEL_TURNS", 8),
            max_tool_calls=_positive_int("MAX_TOOL_CALLS", 15),
            max_runtime_seconds=_positive_float("MAX_RUNTIME_SECONDS", 60.0),
            max_completion_tokens=_positive_int("MAX_COMPLETION_TOKENS", 512),
            input_usd_per_1m_tokens=_optional_rate("OPENAI_INPUT_USD_PER_1M"),
            output_usd_per_1m_tokens=_optional_rate("OPENAI_OUTPUT_USD_PER_1M"),
        )

    def cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        if (
            self.input_usd_per_1m_tokens is None
            or self.output_usd_per_1m_tokens is None
        ):
            raise ValueError(
                "Set OPENAI_INPUT_USD_PER_1M and OPENAI_OUTPUT_USD_PER_1M "
                "to enable live cost accounting."
            )
        return (
            input_tokens * self.input_usd_per_1m_tokens
            + output_tokens * self.output_usd_per_1m_tokens
        ) / 1_000_000
