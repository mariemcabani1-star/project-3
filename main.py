import argparse
import time
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langsmith import traceable

from agent.config import AgentSettings
from agent.graph import build_graph
from agent.model import ModelClient, OpenAIChatModel
from agent.nodes import SYSTEM_PROMPT
from agent.offline_demo import OfflineDemoModel
from agent.tools.calendar_tools import CREATE_EVENT
from agent.tools.validation import validate


def collect_user_request() -> str:
    request = input("What would you like to add to the calendar? ").strip()
    if not request:
        raise ValueError("Please enter a calendar request.")
    return request


def collect_offline_event() -> tuple[str, dict[str, Any]]:
    request = collect_user_request()
    print(
        "\nOffline demo uses a deterministic scripted planner, not a live language "
        "model. Enter event details so it can demonstrate the LangGraph workflow."
    )
    title = input("Event title: ").strip()
    date = input("Date (YYYY-MM-DD): ").strip()
    start_time = input("Start time (HH:MM): ").strip()
    end_time = input("End time (HH:MM): ").strip()
    attendee_text = input("Attendees, comma-separated (optional): ").strip()
    description = input("Description (optional): ").strip()

    start = f"{date}T{start_time}"
    end = f"{date}T{end_time}"
    try:
        start_value = datetime.strptime(start, "%Y-%m-%dT%H:%M")
        end_value = datetime.strptime(end, "%Y-%m-%dT%H:%M")
    except ValueError as error:
        raise ValueError(
            "Enter a real date as YYYY-MM-DD and times as 24-hour HH:MM."
        ) from error
    if end_value <= start_value:
        raise ValueError("The event end time must be after its start time.")

    event = {
        "title": title,
        "start": start,
        "end": end,
        "attendees": [
            item.strip() for item in attendee_text.split(",") if item.strip()
        ],
        "description": description,
        "idempotency_key": "offline-demo-key",
    }
    problems = validate(event, CREATE_EVENT.parameters)
    if problems:
        raise ValueError("; ".join(problems))
    event.pop("idempotency_key")
    return request, event


def display_proposal(proposal: dict) -> None:
    print("\nProposed sandbox calendar operation")
    if proposal.get("operation"):
        print(f"Action: {proposal['operation']}")
    if proposal.get("event_id"):
        print(f"Event ID: {proposal['event_id']}")
    print(f"Title: {proposal.get('title', '(missing)')}")
    print(f"Start: {proposal.get('start_time', '(missing)')}")
    print(f"End: {proposal.get('end_time', '(missing)')}")
    attendees = proposal.get("attendees", [])
    print(f"Attendees: {', '.join(attendees) if attendees else 'None'}")
    if proposal.get("description"):
        print(f"Description: {proposal['description']}")


def collect_edited_tool_args(proposal: dict) -> dict:
    if proposal.get("event_id"):
        current_attendees = ", ".join(proposal.get("attendees", []))
        attendees = input(
            "Recipients to queue, comma-separated "
            f"[{current_attendees}]: "
        ).strip()
        return {
            "event_id": proposal["event_id"],
            "attendees": (
                [item.strip() for item in attendees.split(",") if item.strip()]
                if attendees
                else proposal.get("attendees", [])
            ),
        }

    print("\nEnter the corrected event details. Press Enter to keep a value.")
    title = input(f"Title [{proposal.get('title', '')}]: ").strip()
    start = input(f"Start [{proposal.get('start', '')}]: ").strip()
    end = input(f"End [{proposal.get('end', '')}]: ").strip()
    current_attendees = ", ".join(proposal.get("attendees", []))
    attendees = input(f"Attendees, comma-separated [{current_attendees}]: ").strip()
    description = input(
        f"Description [{proposal.get('description', '')}]: "
    ).strip()
    return {
        "title": title or proposal.get("title", ""),
        "start": start or proposal.get("start", ""),
        "end": end or proposal.get("end", ""),
        "attendees": (
            [item.strip() for item in attendees.split(",") if item.strip()]
            if attendees
            else proposal.get("attendees", [])
        ),
        "description": description or proposal.get("description", ""),
    }


def _run_workflow(
    user_request: str,
    *,
    model_client: ModelClient,
    settings: AgentSettings | None = None,
) -> dict:
    runtime_settings = settings or AgentSettings.from_env()
    graph = build_graph(model_client=model_client, settings=runtime_settings)
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": (
            runtime_settings.max_model_turns
            + 5 * runtime_settings.max_tool_calls
            + 10
        ),
    }
    initial_state = {
        "request_id": thread_id,
        "user_request": user_request,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ],
        "pending_tool_call": None,
        "approved_tool_call": None,
        "approval": "pending",
        "approval_decisions": [],
        "successful_reads": 0,
        "model_turns": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "started_at": time.monotonic(),
        "trajectory": [{"type": "user_request", "text": user_request}],
        "status": "running",
    }

    result = graph.invoke(initial_state, config=config)
    while result.get("__interrupt__"):
        interrupt_value = result["__interrupt__"][0].value
        interrupt_type = interrupt_value.get("type")
        if interrupt_type == "approval":
            display_proposal(interrupt_value["proposal"])
            print("1) Yes  2) No  3) Edit")
            choice = input("Choose an option (1/2/3): ").strip()
            decision = {"1": "yes", "2": "no", "3": "edit"}.get(choice, "no")
            resume_value = {"decision": decision}
        elif interrupt_type == "edit":
            resume_value = collect_edited_tool_args(interrupt_value["proposal"])
        else:
            raise RuntimeError(f"Unknown graph interrupt type: {interrupt_type!r}")
        result = graph.invoke(Command(resume=resume_value), config=config)

    print(f"\nResult: {result.get('result', 'No result was produced.')}")
    if getattr(model_client, "is_live", False):
        print(
            "Usage: "
            f"{result.get('model_turns', 0)} model turns, "
            f"{result.get('tool_calls', 0)} tool calls, "
            f"{result.get('input_tokens', 0)} input tokens, "
            f"{result.get('output_tokens', 0)} output tokens, "
            f"estimated ${result.get('estimated_cost_usd', 0.0):.6f}"
        )
    else:
        print(
            "Offline demo: "
            f"{result.get('model_turns', 0)} scripted turns, "
            f"{result.get('tool_calls', 0)} sandbox tool calls, "
            "0 OpenAI calls, 0 billed tokens, $0.00 API cost."
        )
    return result


_traced_run_workflow = traceable(name="langgraph-calendar-agent")(_run_workflow)


def run_workflow(
    user_request: str,
    *,
    model_client: ModelClient,
    settings: AgentSettings | None = None,
) -> dict:
    runner = (
        _traced_run_workflow
        if getattr(model_client, "is_live", False)
        else _run_workflow
    )
    return runner(user_request, model_client=model_client, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sandbox calendar agent.")
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        required=True,
        help="offline demonstrates the graph without an API; live calls OpenAI.",
    )
    args = parser.parse_args()
    try:
        if args.mode == "offline":
            user_request, event = collect_offline_event()
            result = run_workflow(
                user_request,
                model_client=OfflineDemoModel(event, on_trace=print),
                settings=AgentSettings(),
            )
        else:
            load_dotenv(override=True)
            result = run_workflow(
                collect_user_request(),
                model_client=OpenAIChatModel(),
            )
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
