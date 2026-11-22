
import json

from langgraph.types import interrupt

from schemas.event_schema import EventPayload
from agent.sandbox_adapter import prepare_sandbox_event
from agent.tools.registry import _issue_write_approval, execute_approved


def approval_node(state):
    """
    Present the structured event proposal and wait for human approval.
    """

    # Validate the proposed event and prepare the exact sandbox operation.
    operation = prepare_sandbox_event(
        state["event"],
        state["request_id"],
    )

    proposal = {
        "title": operation["title"],
        "start_time": operation["start"],
        "end_time": operation["end"],
        "attendees": operation["attendees"],
        "description": operation["description"],
    }

    response = interrupt({
        "type": "approval",
        "message": "Review this event before creating it.",
        "proposal": proposal,
        "choices": ["yes", "no", "edit"],
    })

    # Support either a simple string or a structured response.
    if isinstance(response, dict):
        decision = response.get("decision", "no")
    else:
        decision = response

    decision = str(decision).strip().lower()

    if decision not in ("yes", "no", "edit"):
        decision = "no"

    return {
        "approval": decision,

        # Preserve the exact operation that was presented for approval.
        "approved_operation": operation if decision == "yes" else None,
    }


def edit_event_node(state):
    """
    Collect a replacement event and send it through approval again.
    """

    edited_event = interrupt({
        "type": "edit",
        "message": "Provide the corrected event details.",
        "current_event": state["event"],
    })

    if isinstance(edited_event, dict) and "event" in edited_event:
        edited_event = edited_event["event"]

    validated = EventPayload.model_validate(edited_event)

    return {
        "event": validated.model_dump(mode="json"),
        "approval": "pending",
        "approved_operation": None,
    }


def route_after_approval(state):
    decision = state.get("approval", "no")

    if decision == "yes":
        return "create"

    if decision == "edit":
        return "edit"

    return "cancel"


def create_event_node(state):
    """
    Execute the approved operation in the isolated JSON sandbox.
    """

    if state.get("approval") != "yes":
        raise PermissionError(
            "Event creation blocked: explicit approval is required."
        )

    operation = state.get("approved_operation")

    if not operation:
        raise PermissionError(
            "Event creation blocked: no approved operation was recorded."
        )

    approval = _issue_write_approval("create_event", operation)
    raw_result = execute_approved("create_event", operation, approval)
    result = json.loads(raw_result)

    if not result.get("ok"):
        error = result.get("error", {})
        raise RuntimeError(
            f"Sandbox event creation failed: "
            f"{error.get('code', 'unknown_error')} - "
            f"{error.get('message', 'Unknown error')}"
        )

    return {
        "result": result,
        "approval": "completed",
    }


def cancel_event_node(state):
    """
    End the operation without creating an event.
    """

    return {
        "approval": "rejected",
        "approved_operation": None,
        "result": {
            "status": "cancelled",
            "message": "No event was created in the sandbox.",
        },
    }
