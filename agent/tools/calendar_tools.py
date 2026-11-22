"""Calendar tools: get_calendar, find_free_slots (read) and create_event (write).

All times are naive local times, ISO format 'YYYY-MM-DDTHH:MM' (24h). Everything
operates on the sandbox store, never a real calendar.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta

from sandbox import store

from .base import Tool, ToolError

DT_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"
DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
TIME_PATTERN = r"\d{2}:\d{2}"
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
KEY_PATTERN = r"[A-Za-z0-9_-]{8,64}"

MAX_RANGE_DAYS = 14
SLOT_STEP_MIN = 30


# ---------- helpers ----------
def _dt(s: str, field: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M")
    except ValueError:
        raise ToolError("bad_arguments", f"'{field}' is not a real date/time: {s!r}",
                        "Use 'YYYY-MM-DDTHH:MM', e.g. 2026-12-03T14:30.")


def _d(s: str, field: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ToolError("bad_arguments", f"'{field}' is not a real date: {s!r}",
                        "Use 'YYYY-MM-DD', e.g. 2026-12-03.")


def _t(s: str, field: str):
    try:
        return datetime.strptime(s, "%H:%M").time()
    except ValueError:
        raise ToolError("bad_arguments", f"'{field}' is not a real time: {s!r}",
                        "Use 'HH:MM' 24h, e.g. 09:00.")


def _range(start_date: str, end_date: str) -> tuple[date, date]:
    a, b = _d(start_date, "start_date"), _d(end_date, "end_date")
    if b < a:
        raise ToolError("bad_arguments", "'end_date' is before 'start_date'.",
                        "Swap the two dates.")
    if (b - a).days + 1 > MAX_RANGE_DAYS:
        raise ToolError("range_too_large", f"Range is longer than {MAX_RANGE_DAYS} days.",
                        f"Query at most {MAX_RANGE_DAYS} days at a time.")
    return a, b


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def _public(ev: dict) -> dict:
    public_event = {
        k: ev[k] for k in ("id", "title", "start", "end", "attendees")
    }
    if "description" in ev:
        public_event["description"] = ev["description"]
    return public_event


# ---------- read tool 1 ----------
def get_calendar(args: dict) -> dict:
    a, b = _range(args["start_date"], args["end_date"])
    lo = datetime.combine(a, datetime.min.time())
    hi = datetime.combine(b + timedelta(days=1), datetime.min.time())
    events = [
        e for e in store.load()["events"]
        if _overlaps(_dt(e["start"], "start"), _dt(e["end"], "end"), lo, hi)
    ]
    events.sort(key=lambda e: e["start"])
    out = {"events": [_public(e) for e in events], "count": len(events)}
    if not events:
        out["note"] = "No events in this range; the whole period is free."
    return out


# ---------- read tool 2 ----------
def find_free_slots(args: dict) -> dict:
    a, b = _range(args["start_date"], args["end_date"])
    duration = timedelta(minutes=args["duration_minutes"])
    day_start = _t(args.get("day_start", "09:00"), "day_start")
    day_end = _t(args.get("day_end", "17:00"), "day_end")
    if day_end <= day_start:
        raise ToolError("bad_arguments", "'day_end' must be after 'day_start'.",
                        "Example: day_start=09:00, day_end=17:00.")
    max_results = args.get("max_results", 5)

    events = [(_dt(e["start"], "start"), _dt(e["end"], "end")) for e in store.load()["events"]]
    slots: list[dict] = []
    day = a
    while day <= b and len(slots) < max_results:
        if day.weekday() < 5:  # weekdays only
            cur = datetime.combine(day, day_start)
            limit = datetime.combine(day, day_end)
            while cur + duration <= limit and len(slots) < max_results:
                end = cur + duration
                if not any(_overlaps(cur, end, s, e) for s, e in events):
                    slots.append({"start": cur.strftime("%Y-%m-%dT%H:%M"),
                                  "end": end.strftime("%Y-%m-%dT%H:%M")})
                cur += timedelta(minutes=SLOT_STEP_MIN)
        day += timedelta(days=1)

    out = {"slots": slots, "count": len(slots)}
    if not slots:
        out["note"] = ("No free slot of that length in this range (weekends are skipped). "
                       "Try a shorter duration, wider hours, or a later date range.")
    return out


# ---------- write tool ----------
def _fingerprint(args: dict) -> str:
    core = {k: args[k] for k in ("title", "start", "end")}
    core["attendees"] = sorted(a.lower() for a in args.get("attendees", []))
    if args.get("description"):
        core["description"] = args["description"]
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def create_event(args: dict) -> dict:
    """WRITE. Creates an event in the sandbox and 'sends' invites to the sandbox outbox.

    Idempotent: the same idempotency_key + same arguments returns the original event
    instead of creating a second one. Same key with different arguments is an error.
    """
    start, end = _dt(args["start"], "start"), _dt(args["end"], "end")
    if end <= start:
        raise ToolError("bad_arguments", "'end' must be after 'start'.",
                        "Check the times, or call find_free_slots for a valid slot.")
    if start.date() != end.date():
        raise ToolError("bad_arguments", "Events must start and end on the same day.",
                        "Split it into one event per day.")
    attendees = [a.strip().lower() for a in args.get("attendees", [])]
    args = {**args, "attendees": attendees}

    state = store.load()
    key, fp = args["idempotency_key"], _fingerprint(args)

    # 1) idempotency: replay of a call that already happened
    seen = state["idempotency"].get(key)
    if seen:
        if seen["fingerprint"] != fp:
            raise ToolError("idempotency_conflict",
                            "This idempotency_key was already used for a different event.",
                            "Use a new unique idempotency_key for a new event.")
        ev = next(e for e in state["events"] if e["id"] == seen["event_id"])
        return {"created": False, "already_existed": True, "event": _public(ev),
                "note": "Same request was already executed; nothing new was created."}

    # 2) conflict check
    for e in state["events"]:
        if _overlaps(start, end, _dt(e["start"], "start"), _dt(e["end"], "end")):
            raise ToolError("conflict",
                            f"Overlaps existing event '{e['title']}' ({e['start']} to {e['end']}).",
                            "Call find_free_slots to pick a free time, then retry.")

    # 3) write (sandbox only)
    event_id = f"evt_{state['next_id']:04d}"
    state["next_id"] += 1
    ev = {"id": event_id, "title": args["title"], "start": args["start"],
          "end": args["end"], "attendees": attendees,
          "description": args.get("description", "")}
    state["events"].append(ev)
    for who in attendees:
        state["outbox"].append({"to": who, "event_id": event_id, "type": "invite"})
    state["idempotency"][key] = {"fingerprint": fp, "event_id": event_id}
    store.save(state)
    return {"created": True, "event": _public(ev), "invites_sent": len(attendees)}


# ---------- tool definitions ----------
GET_CALENDAR = Tool(
    name="get_calendar",
    description=("Read-only. List calendar events between two dates (inclusive). "
                 f"Max range {MAX_RANGE_DAYS} days. Use it to see what is already booked."),
    parameters={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "pattern": DATE_PATTERN,
                           "x-format-hint": "YYYY-MM-DD", "description": "First day, YYYY-MM-DD."},
            "end_date": {"type": "string", "pattern": DATE_PATTERN,
                         "x-format-hint": "YYYY-MM-DD", "description": "Last day, YYYY-MM-DD."},
        },
        "required": ["start_date", "end_date"],
        "additionalProperties": False,
    },
    fn=get_calendar,
)

FIND_FREE_SLOTS = Tool(
    name="find_free_slots",
    description=("Read-only. Find free time slots of a given length on weekdays within working hours. "
                 "Slots start on 30-minute boundaries. Use this before create_event."),
    parameters={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "pattern": DATE_PATTERN,
                           "x-format-hint": "YYYY-MM-DD", "description": "First day to search."},
            "end_date": {"type": "string", "pattern": DATE_PATTERN,
                         "x-format-hint": "YYYY-MM-DD", "description": "Last day to search."},
            "duration_minutes": {"type": "integer", "minimum": 15, "maximum": 480,
                                 "description": "Meeting length in minutes (15-480)."},
            "day_start": {"type": "string", "pattern": TIME_PATTERN,
                          "x-format-hint": "HH:MM", "description": "Earliest start, default 09:00."},
            "day_end": {"type": "string", "pattern": TIME_PATTERN,
                        "x-format-hint": "HH:MM", "description": "Latest end, default 17:00."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20,
                            "description": "How many slots to return (default 5)."},
        },
        "required": ["start_date", "end_date", "duration_minutes"],
        "additionalProperties": False,
    },
    fn=find_free_slots,
)

CREATE_EVENT = Tool(
    name="create_event",
    description=("WRITE. Create a calendar event and send invites to attendees. Requires human approval. "
                 "Fails if the time overlaps another event. Generate one unique idempotency_key per "
                 "intended event and reuse the SAME key if you retry."),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 120},
            "start": {"type": "string", "pattern": DT_PATTERN,
                      "x-format-hint": "YYYY-MM-DDTHH:MM", "description": "Start, e.g. 2026-12-03T14:00."},
            "end": {"type": "string", "pattern": DT_PATTERN,
                    "x-format-hint": "YYYY-MM-DDTHH:MM", "description": "End, same day as start."},
            "attendees": {"type": "array", "maxItems": 20,
                          "items": {"type": "string", "pattern": EMAIL_PATTERN,
                                    "x-format-hint": "a valid email address"}},
            "description": {"type": "string"},
            "idempotency_key": {"type": "string", "pattern": KEY_PATTERN,
                                "x-format-hint": "8-64 chars of letters, digits, _ or -"},
        },
        "required": ["title", "start", "end", "idempotency_key"],
        "additionalProperties": False,
    },
    fn=create_event,
    writes=True,
)
