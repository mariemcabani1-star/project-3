
from typing import TypedDict, Any


class CalendarState(TypedDict, total=False):
    event: dict[str, Any]

    # Unique identifier for the current booking request.
    request_id: str

    # Human approval status.
    approval: str

    # Exact sandbox operation presented to and approved by the user.
    approved_operation: dict[str, Any] | None

    # Operation result.
    result: dict[str, Any]

    error: str
