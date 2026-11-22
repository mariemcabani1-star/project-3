
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from zoneinfo import ZoneInfo
from googleapiclient.errors import HttpError

from schemas.event_schema import EventPayload

BASE_DIR = Path(__file__).resolve().parent.parent

CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events.owned"
]


def authenticate_google_calendar():

    credentials = None

    # Load previously authorized credentials
    if TOKEN_FILE.exists():

        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES
        )

    # Refresh or request authorization
    if not credentials or not credentials.valid:

        if (
            credentials
            and credentials.expired
            and credentials.refresh_token
        ):
            credentials.refresh(Request())

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                SCOPES
            )

            credentials = flow.run_local_server(
                port=0,
                access_type="offline",
                prompt="consent"
            )

        # Save token for future authentication
        TOKEN_FILE.write_text(credentials.to_json())

    service = build(
        "calendar",
        "v3",
        credentials=credentials
    )

    return service
def create_google_event(payload: EventPayload):

    # Validate event
    payload = EventPayload.model_validate(
        payload.model_dump()
    )

    if (
        payload.start_time.tzinfo is None
        or payload.end_time.tzinfo is None
    ):
        raise ValueError("Timezone is required")

    if payload.end_time <= payload.start_time:
        raise ValueError("Invalid event duration")

    service = authenticate_google_calendar()

    timezone_name = "Africa/Tunis"

    # Convert event times to Tunis local time
    start = payload.start_time.astimezone(
        ZoneInfo(timezone_name)
    )

    end = payload.end_time.astimezone(
        ZoneInfo(timezone_name)
    )

    event_body = {
        "summary": payload.title,

        "description": payload.description,

        "start": {
            "dateTime": start.isoformat(),
            "timeZone": timezone_name
        },

        "end": {
            "dateTime": end.isoformat(),
            "timeZone": timezone_name
        }
    }

    try:

        created_event = service.events().insert(
            calendarId="primary",
            body=event_body
        ).execute()

        return {
            "status": "created",
            "event_id": created_event["id"],
            "title": created_event["summary"],
            "html_link": created_event.get("htmlLink"),
            "start_time": created_event["start"],
            "end_time": created_event["end"]
        }

    except HttpError as error:

        raise RuntimeError(
            f"Google Calendar API error: {error}"
        ) from error
