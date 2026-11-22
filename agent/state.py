
from typing import TypedDict, Any


class CalendarState(TypedDict, total=False):
    user_request: str
    request_id: str
    messages: list[dict[str, Any]]
    pending_tool_call: dict[str, Any] | None
    approved_tool_call: dict[str, Any] | None
    approval_decisions: list[str]
    approval: str
    result: str
    error: dict[str, Any]
    status: str
    stop_reason: str
    model_turns: int
    tool_calls: int
    successful_reads: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    started_at: float
    trajectory: list[dict[str, Any]]
