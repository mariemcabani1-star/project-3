"""Tool registry + safe executor.

`execute()` NEVER raises: every outcome is a JSON string the model can read.
Every call (input, output, duration) is appended to a JSONL log.

Writes require a one-use approval grant bound to the exact tool arguments.
The LangGraph creation node issues that grant only after its approval route.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import Tool, ToolError
from .calendar_tools import CREATE_EVENT, FIND_FREE_SLOTS, GET_CALENDAR
from .validation import validate

ALL_TOOLS: list[Tool] = [GET_CALENDAR, FIND_FREE_SLOTS, CREATE_EVENT]
TOOLS: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
_APPROVAL_ISSUER = object()


class _WriteApproval:
    __slots__ = ("tool_name", "arguments", "_used")

    def __init__(self, tool_name: str, arguments: str, issuer: object):
        if issuer is not _APPROVAL_ISSUER:
            raise PermissionError("Write approval grants cannot be constructed directly.")
        self.tool_name = tool_name
        self.arguments = arguments
        self._used = False

    def consume(self, tool_name: str, args: dict) -> None:
        if self._used:
            raise ToolError("approval_reused", "This write approval has already been used.")
        self._used = True
        if self.tool_name != tool_name or self.arguments != _canonical(args):
            raise ToolError(
                "approval_mismatch",
                "The write arguments do not match the operation approved by the user.",
            )


def log_path() -> Path:
    return Path(os.environ.get("TOOL_LOG_PATH", Path(__file__).resolve().parents[2] / "logs" / "tool_calls.jsonl"))


def openai_specs() -> list[dict]:
    specs = deepcopy([t.openai_spec() for t in ALL_TOOLS])
    for spec in specs:
        _strip_model_schema_hints(spec["function"]["parameters"])
        if spec["function"]["name"] == "create_event":
            parameters = spec["function"]["parameters"]
            parameters["properties"].pop("idempotency_key")
            parameters["required"] = [
                field for field in parameters["required"]
                if field != "idempotency_key"
            ]
    return specs


def _strip_model_schema_hints(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("x-format-hint", None)
        for nested in value.values():
            _strip_model_schema_hints(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_model_schema_hints(nested)


def needs_approval(name: str) -> bool:
    return name in TOOLS and TOOLS[name].writes


def describe_call(name: str, args: dict) -> str:
    """Plain-text preview of a pending write, to show the human at the checkpoint."""
    if name == "create_event":
        who = ", ".join(args.get("attendees", [])) or "nobody"
        return (f"Create event '{args.get('title')}' from {args.get('start')} to "
                f"{args.get('end')}, invite: {who}")
    return f"Call {name} with {json.dumps(args)}"


def _parse(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise ToolError("bad_arguments", f"Arguments are not valid JSON ({e.msg}).",
                        "Send a JSON object with the documented argument names.")
    if not isinstance(val, dict):
        raise ToolError("bad_arguments", "Arguments must be a JSON object.", "")
    return val


def _canonical(args: dict) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _issue_write_approval(name: str, raw_args) -> _WriteApproval:
    """Create an internal one-use grant for an operation after graph approval."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError("unknown_tool", f"No tool named '{name}'.",
                        f"Available tools: {sorted(TOOLS)}.")
    if not tool.writes:
        raise ToolError("approval_not_required", f"Tool '{name}' is not a write tool.")

    args = _parse(raw_args)
    problems = validate(args, tool.parameters)
    if problems:
        raise ToolError("bad_arguments", "; ".join(problems),
                        "Fix the arguments and request approval again.")

    return _WriteApproval(name, _canonical(args), _APPROVAL_ISSUER)


def execute(name: str, raw_args) -> str:
    """Execute reads directly; writes return approval_required without a grant."""
    return _execute(name, raw_args)


def execute_approved(name: str, raw_args, approval: _WriteApproval) -> str:
    """Execute a write with its matching, one-use approval grant."""
    return _execute(name, raw_args, approval)


def _execute(name: str, raw_args, approval: _WriteApproval | None = None) -> str:
    """Run a tool call and return a JSON result without leaking stack traces."""
    t0 = time.perf_counter()
    args: object = raw_args
    try:
        tool = TOOLS.get(name)
        if tool is None:
            raise ToolError("unknown_tool", f"No tool named '{name}'.",
                            f"Available tools: {sorted(TOOLS)}.")
        args = _parse(raw_args)
        problems = validate(args, tool.parameters)
        if problems:
            raise ToolError("bad_arguments", "; ".join(problems),
                            "Fix the arguments and call the tool again.")
        if tool.writes:
            if approval is None:
                raise ToolError(
                    "approval_required",
                    f"Write tool '{name}' cannot run without explicit human approval.",
                    "Route the operation through the LangGraph approval checkpoint.",
                )
            if not isinstance(approval, _WriteApproval):
                raise ToolError("approval_required", "A valid write approval grant is required.")
            approval.consume(name, args)
        elif approval is not None:
            raise ToolError("approval_not_required", f"Tool '{name}' is not a write tool.")
        result = {"ok": True, **tool.fn(args)}
    except ToolError as e:
        result = {"ok": False, "error": {"code": e.code, "message": e.message, "hint": e.hint}}
    except Exception:  # never leak a stack trace to the model
        _log(name, args, {"ok": False, "internal_trace": traceback.format_exc()}, t0)
        return json.dumps({"ok": False, "error": {
            "code": "internal_error",
            "message": "The tool failed unexpectedly.",
            "hint": "Retry once; if it fails again, tell the user the calendar is unavailable."}})
    out = json.dumps(result)
    _log(name, args, result, t0)
    return out


def _log(name, args, result, t0, phase: str = "execute") -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": name,
        "phase": phase,
        "input": args,
        "output": result,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def precheck(name: str, raw_args) -> str | None:
    """Validate a call WITHOUT running it. Returns an error JSON string, or None if valid.

    The loop uses this so a human is never asked to approve a write that would be
    rejected anyway.
    """
    t0 = time.perf_counter()
    args: object = raw_args
    try:
        tool = TOOLS.get(name)
        if tool is None:
            raise ToolError("unknown_tool", f"No tool named '{name}'.",
                            f"Available tools: {sorted(TOOLS)}.")
        args = _parse(raw_args)
        problems = validate(args, tool.parameters)
        if problems:
            raise ToolError("bad_arguments", "; ".join(problems),
                            "Fix the arguments and call the tool again.")
    except ToolError as e:
        result = {
            "ok": False,
            "error": {"code": e.code, "message": e.message, "hint": e.hint},
        }
        _log(name, args, result, t0, phase="precheck")
        return json.dumps(result)
    _log(
        name,
        args,
        {"ok": True, "validated_only": True},
        t0,
        phase="precheck",
    )
    return None


def parse_args(raw_args) -> dict:
    """Parse already-prechecked arguments into a dict."""
    return _parse(raw_args)
