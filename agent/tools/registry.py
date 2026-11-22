"""Tool registry + safe executor.

`execute()` NEVER raises: every outcome is a JSON string the model can read.
Every call (input, output, duration) is appended to a JSONL log.

The registry does not enforce the human checkpoint itself; it exposes
`needs_approval()` and `describe_call()` so the agent loop can gate writes.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .base import Tool, ToolError
from .calendar_tools import CREATE_EVENT, FIND_FREE_SLOTS, GET_CALENDAR
from .validation import validate

ALL_TOOLS: list[Tool] = [GET_CALENDAR, FIND_FREE_SLOTS, CREATE_EVENT]
TOOLS: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def log_path() -> Path:
    return Path(os.environ.get("TOOL_LOG_PATH", Path(__file__).resolve().parents[2] / "logs" / "tool_calls.jsonl"))


def openai_specs() -> list[dict]:
    return [t.openai_spec() for t in ALL_TOOLS]


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


def execute(name: str, raw_args) -> str:
    """Run a tool call; returns a JSON string: {"ok": true, ...} or {"ok": false, "error": {...}}."""
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


def _log(name, args, result, t0) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "tool": name, "input": args,
           "output": result, "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def precheck(name: str, raw_args) -> str | None:
    """Validate a call WITHOUT running it. Returns an error JSON string, or None if valid.

    The loop uses this so a human is never asked to approve a write that would be
    rejected anyway.
    """
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
        return json.dumps({"ok": False, "error": {"code": e.code, "message": e.message, "hint": e.hint}})
    return None


def parse_args(raw_args) -> dict:
    """Parse already-prechecked arguments into a dict."""
    return _parse(raw_args)
