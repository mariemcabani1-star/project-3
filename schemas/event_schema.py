
import re
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


class EventPayload(BaseModel):

    title: str = Field(
        min_length=1
    )

    start_time: datetime

    end_time: datetime

    attendees: List[str] = Field(default_factory=list)

    description: str = ""

    @field_validator("title")
    @classmethod
    def validate_title(cls, value):
        if not value.strip():
            raise ValueError("Event title cannot be empty")

        return value.strip()

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_timezone(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Event date and time must include a timezone."
            )

        return value

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, value, info):
        start = info.data.get("start_time")

        if start and value <= start:
            raise ValueError(
                "End time must be after start time."
            )

        return value

    @field_validator("attendees")
    @classmethod
    def validate_attendees(cls, emails):
        pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

        for email in emails:
            if not re.match(pattern, email):
                raise ValueError(
                    f"Invalid attendee email: {email}"
                )

        return emails
