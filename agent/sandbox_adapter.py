
import hashlib
import json
from zoneinfo import ZoneInfo

from schemas.event_schema import EventPayload


TUNIS_TIMEZONE = ZoneInfo("Africa/Tunis")


def prepare_sandbox_event(event: dict, request_id: str) -> dict:
    """
    Convert the project's EventPayload into the sandbox tool format.

    The sandbox uses naive Tunis-local datetime strings:
    YYYY-MM-DDTHH:MM
    """

    payload = EventPayload.model_validate(event)

    if not request_id.strip():
        raise ValueError("A request ID is required for sandbox idempotency.")

    if len(payload.title) > 120:
        raise ValueError("Sandbox event titles cannot exceed 120 characters.")

    if len(payload.attendees) > 20:
        raise ValueError("The sandbox supports at most 20 attendees.")

    start_time = payload.start_time
    end_time = payload.end_time

    # Require timezone-aware datetimes.
    if (
        start_time.tzinfo is None
        or start_time.utcoffset() is None
        or end_time.tzinfo is None
        or end_time.utcoffset() is None
    ):
        raise ValueError("Both event times must include a timezone.")

    # Convert both times to Tunis local time.
    start = start_time.astimezone(TUNIS_TIMEZONE)
    end = end_time.astimezone(TUNIS_TIMEZONE)

    if end <= start:
        raise ValueError("Event end time must be after start time.")

    # The provided sandbox only supports same-day events.
    if start.date() != end.date():
        raise ValueError(
            "The sandbox currently supports events within one local day."
        )

    start_str = start.strftime("%Y-%m-%dT%H:%M")
    end_str = end.strftime("%Y-%m-%dT%H:%M")

    # A stable key prevents duplicate creation if the same operation
    # is resumed or retried within the same request.
    identity = {
        "request_id": request_id,
        "title": payload.title,
        "start": start_str,
        "end": end_str,
        "attendees": sorted(a.lower() for a in payload.attendees),
    }
    if payload.description:
        identity["description"] = payload.description

    idempotency_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]

    return {
        "title": payload.title,
        "start": start_str,
        "end": end_str,
        "attendees": payload.attendees,
        "description": payload.description,
        "idempotency_key": idempotency_key,
    }


def prepare_model_create_event(arguments: dict, request_id: str) -> dict:
    """Add the request-scoped idempotency key to model-supplied tool arguments."""
    if not request_id.strip():
        raise ValueError("A request ID is required for sandbox idempotency.")
    if not isinstance(arguments, dict):
        raise ValueError("Event arguments must be a JSON object.")
    if "idempotency_key" in arguments:
        raise ValueError("The idempotency key is system-managed and cannot be supplied by the model.")

    attendees = arguments.get("attendees", [])
    if not isinstance(attendees, list) or any(
        not isinstance(attendee, str) for attendee in attendees
    ):
        raise ValueError("Attendees must be a list of email addresses.")
    description = arguments.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Description must be text.")

    operation = {
        **arguments,
        "attendees": attendees,
        "description": description,
    }
    identity = {
        "request_id": request_id,
        "title": operation.get("title"),
        "start": operation.get("start"),
        "end": operation.get("end"),
        "attendees": sorted(a.lower() for a in operation["attendees"]),
    }
    if operation["description"]:
        identity["description"] = operation["description"]

    operation["idempotency_key"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]
    return operation
