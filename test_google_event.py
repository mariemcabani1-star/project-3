
from datetime import datetime

from schemas.event_schema import EventPayload
from calendar_tools.google_calendar import create_google_event


def main():

    event = EventPayload(
        title="Calendar Agent Integration Test",
        start_time=datetime.fromisoformat(
            "2026-11-16T14:00:00+01:00"
        ),
        end_time=datetime.fromisoformat(
            "2026-11-16T14:30:00+01:00"
        ),
        attendees=[],
        description="Testing create_event with Google Calendar API"
    )

    print("Creating Google Calendar event...")

    result = create_google_event(event)

    print("\nSUCCESS!")
    print("Event ID:", result["event_id"])
    print("Title:", result["title"])
    print("Calendar URL:", result["html_link"])


if __name__ == "__main__":
    main()
