
from calendar_tools.google_calendar import (
    authenticate_google_calendar
)
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

# Load configuration before importing LangSmith or the graph.
load_dotenv()

import uuid

from langsmith import traceable
from langgraph.types import Command

from agent.graph import build_graph
from schemas.event_schema import EventPayload
def main():

    service = authenticate_google_calendar()

    # Use events.list instead of calendars.get
    events = service.events().list(
        calendarId="primary",
        maxResults=5,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    print("\nGoogle Calendar connection successful!")

    print("Recent events:")

    for event in events.get("items", []):

        print(
            event.get("summary", "Untitled event")
        )


if __name__ == "__main__":
    main()
def display_event(event):
    print("\n========== EVENT PROPOSAL ==========")
    print("Title:", event["title"])
    print("Start:", event["start_time"])
    print("End:", event["end_time"])
    print("Attendees:", event.get("attendees", []))
    print("Description:", event.get("description", ""))
    print("====================================")

def collect_initial_event():
    print("\n========== CREATE EVENT ==========")

    title = input("Event title: ").strip()
    start = input(
        "Start time (example: 2026-11-16T14:00:00+01:00): "
    ).strip()
    end = input(
        "End time (example: 2026-11-16T14:30:00+01:00): "
    ).strip()

    description = input("Description (optional): ").strip()

    attendee_input = input(
        "Attendee emails separated by commas (or press Enter): "
    ).strip()

    attendees = [
        email.strip()
        for email in attendee_input.split(",")
        if email.strip()
    ]

    event = EventPayload(
        title=title,
        start_time=start,
        end_time=end,
        attendees=attendees,
        description=description
    )

    return event.model_dump(mode="json")

def collect_edited_event(current_event):
    print("\n========== EDIT EVENT ==========")
    print("Enter the complete corrected event details.")

    title = input(
        f"New title [{current_event['title']}]: "
    ).strip() or current_event["title"]

    start = input(
        f"New start [{current_event['start_time']}]: "
    ).strip() or current_event["start_time"]

    end = input(
        f"New end [{current_event['end_time']}]: "
    ).strip() or current_event["end_time"]

    description = input(
        "New description (press Enter to keep current): "
    ).strip() or current_event.get("description", "")

    # Keep the original attendee list unless a separate,
    # explicitly approved attendee-edit feature is implemented.
    edited = EventPayload(
        title=title,
        start_time=start,
        end_time=end,
        attendees=current_event.get("attendees", []),
        description=description
    )

    return edited.model_dump(mode="json")

@traceable(name="Calendar Agent Workflow")
def run_workflow(event):

    graph = build_graph()

    # Each run gets a unique conversation/checkpoint ID.
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    result = graph.invoke(
        {"event": event, "approval": "pending"},
        config=config
    )

    while result.get("__interrupt__"):

        interruption = result["__interrupt__"][0]
        request = interruption.value

        if request["type"] == "approval":

            display_event(request["event"])

            decision = input(
                "\nApprove this event? (yes/no/edit): "
            ).strip().lower()

            if decision not in ("yes", "no", "edit"):
                print("Invalid choice. Treating it as No.")
                decision = "no"

            result = graph.invoke(
                Command(resume=decision),
                config=config
            )

        elif request["type"] == "edit":

            edited_event = collect_edited_event(
                request["current_event"]
            )

            result = graph.invoke(
                Command(resume=edited_event),
                config=config
            )

    print("\n========== WORKFLOW RESULT ==========")
    print(result.get("result", "No result returned."))

if __name__ == "__main__":

    try:
        event = collect_initial_event()
        run_workflow(event)

    except Exception as error:
        print("\nERROR:", error)