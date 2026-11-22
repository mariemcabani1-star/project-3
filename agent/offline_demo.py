"""Deterministic model client for the interactive, no-API demo."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from agent.model import ModelResponse


class OfflineDemoModel:
    is_live = False

    def __init__(
        self,
        event: dict[str, Any],
        on_trace: Callable[[str], None] | None = None,
    ) -> None:
        self._event = dict(event)
        self._on_trace = on_trace
        self._turn = 0

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
        del tools, model, max_completion_tokens, allow_tools, timeout_seconds
        self._turn += 1
        last_tool = next(
            (message for message in reversed(messages) if message.get("role") == "tool"),
            None,
        )

        if last_tool is None:
            event_date = self._event["start"][:10]
            self._trace(f"Planner: reading the sandbox calendar for {event_date}.")
            return self._tool_response(
                "get_calendar",
                {"start_date": event_date, "end_date": event_date},
            )

        observation = json.loads(last_tool["content"])
        if last_tool.get("name") == "get_calendar":
            if not observation.get("ok"):
                return self._final_response(
                    "I could not read the sandbox calendar, so I did not propose a write."
                )
            conflict = self._find_conflict(observation.get("events", []))
            if conflict:
                self._trace(
                    f"Observation: the requested time overlaps '{conflict['title']}'."
                )
                return self._final_response(
                    f"That time conflicts with '{conflict['title']}'. "
                    "Nothing was created; choose another time and try again."
                )
            self._trace(
                f"Observation: read {observation.get('count', 0)} event(s); "
                "the requested time is available."
            )
            self._trace("Planner: proposing create_event; human approval is required.")
            return self._tool_response("create_event", self._event)

        if last_tool.get("name") == "create_event":
            error = observation.get("error", {})
            if not observation.get("ok") and error.get("code") == "rejected_by_user":
                text = "You rejected the proposal, so no event was created."
            elif not observation.get("ok"):
                text = (
                    "The sandbox did not create the event: "
                    f"{error.get('message', 'unknown tool error')}"
                )
            else:
                event = observation.get("event", {})
                attendees = event.get("attendees", [])
                if attendees:
                    self._trace(
                        "Observation: event is in the sandbox. Its invitations have "
                        "not been queued; each requires separate approval."
                    )
                    self._trace(
                        "Planner: proposing send_invite; this only queues locally."
                    )
                    return self._tool_response(
                        "send_invite",
                        {
                            "event_id": event["id"],
                            "attendees": attendees,
                        },
                    )
                if observation.get("created"):
                    text = f"Created '{event.get('title', self._event['title'])}' in the sandbox."
                else:
                    text = "That event already exists in the sandbox; no duplicate was created."
            self._trace(f"Observation: {text}")
            return self._final_response(text)

        if last_tool.get("name") == "send_invite":
            error = observation.get("error", {})
            if not observation.get("ok") and error.get("code") == "rejected_by_user":
                text = "The event remains in the sandbox, but you rejected the invitation queue request."
            elif not observation.get("ok"):
                text = (
                    "The sandbox did not queue the invitations: "
                    f"{error.get('message', 'unknown tool error')}"
                )
            else:
                text = (
                    f"Queued {observation.get('queued', 0)} invitation(s) in the sandbox "
                    f"outbox; {observation.get('already_queued', 0)} were already queued. "
                    "No email was sent."
                )
            self._trace(f"Observation: {text}")
            return self._final_response(text)

        return self._final_response(
            "The offline demo received an unexpected tool result and stopped safely."
        )

    def _find_conflict(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        start = datetime.strptime(self._event["start"], "%Y-%m-%dT%H:%M")
        end = datetime.strptime(self._event["end"], "%Y-%m-%dT%H:%M")
        for existing in events:
            existing_start = datetime.strptime(existing["start"], "%Y-%m-%dT%H:%M")
            existing_end = datetime.strptime(existing["end"], "%Y-%m-%dT%H:%M")
            if start < existing_end and existing_start < end:
                return existing
        return None

    def _tool_response(self, name: str, arguments: dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"offline-call-{self._turn}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            input_tokens=0,
            output_tokens=0,
            usage_source="offline-scripted",
            estimated_cost_usd=0.0,
        )

    def _final_response(self, text: str) -> ModelResponse:
        return ModelResponse(
            assistant_message={"role": "assistant", "content": text},
            input_tokens=0,
            output_tokens=0,
            usage_source="offline-scripted",
            estimated_cost_usd=0.0,
        )

    def _trace(self, message: str) -> None:
        if self._on_trace is not None:
            self._on_trace(message)
