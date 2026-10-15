"""The seeded world and the approved lines, as the tests see them.

These are the tester's copies, written out as literals rather than imported from
the app, so a test compares the app against an independent statement of what
it should hold. A drift check in tests/frontdesk keeps the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DEFAULT_AT = "2026-10-14T10:00:00-04:00"  # Wednesday, 10 AM clinic time: staffed hours
AFTER_HOURS_AT = "2026-10-14T19:00:00-04:00"  # Wednesday, 7 PM: staff have gone home
WEEKEND_AT = "2026-10-17T10:00:00-04:00"  # Saturday, 10 AM: no staff

CLINIC = "Brookside Family Clinic"
PROVIDERS = ("Dr. Okafor", "Dr. Lindqvist", "Dr. Patel")
SUPPORTED_SSML_TAGS = frozenset({"speak", "break", "say-as", "prosody"})


@dataclass(frozen=True)
class Appointment:
    provider: str  # "Dr. Lindqvist"
    start: str  # ISO 8601 with offset
    spoken: str  # how Front Desk says it: "Thursday, October 22nd at 2 PM"


@dataclass(frozen=True)
class Patient:
    id: str
    first_name: str
    last_name: str
    dob: date
    phone: str  # "215-555-0142"
    next_appointment: Appointment | None  # as of DEFAULT_AT

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


MARIA_ALVAREZ = Patient(
    id="P-1001",
    first_name="Maria",
    last_name="Alvarez",
    dob=date(1980, 3, 3),
    phone="215-555-0142",
    next_appointment=Appointment(
        "Dr. Lindqvist", "2026-10-22T14:00:00-04:00", "Thursday, October 22nd at 2 PM"
    ),
)
JAMES_WHITAKER = Patient(
    id="P-1002",
    first_name="James",
    last_name="Whitaker",
    dob=date(1975, 11, 9),
    phone="215-555-0177",
    next_appointment=Appointment(
        "Dr. Patel", "2026-10-20T09:30:00-04:00", "Tuesday, October 20th at 9:30 AM"
    ),
)
PRIYA_RAMAN = Patient(
    id="P-1003",
    first_name="Priya",
    last_name="Raman",
    dob=date(1992, 7, 21),
    phone="215-555-0199",
    next_appointment=Appointment(
        "Dr. Okafor", "2026-10-27T09:00:00-04:00", "Tuesday, October 27th at 9 AM"
    ),
)
THOMAS_BRENNAN = Patient(
    id="P-1004",
    first_name="Thomas",
    last_name="Brennan",
    dob=date(1968, 1, 15),
    phone="215-555-0123",
    next_appointment=None,
)


def patients() -> list[Patient]:
    """Every seeded patient."""
    return [MARIA_ALVAREZ, JAMES_WHITAKER, PRIYA_RAMAN, THOMAS_BRENNAN]


# Approved lines, spoken word for word.

DISCLOSURE = "This call may be recorded and transcribed to help us serve you."
EMERGENCY_SCRIPT = "If this is a medical emergency, please hang up and call 911 now."
NURSE_LINE_OFFER = (
    "I can't give medical advice, but I can send a message to our nurse line and a nurse "
    "will call you back. Would you like me to do that?"
)
TRANSFER_LINE = "I'm transferring you to a member of our staff now. Please hold."
TRANSFER_OFFER = "Would you like me to transfer you to a member of our staff?"
CALLBACK_OFFER = (
    "Our staff are available Monday to Friday, 8 AM to 5 PM, so I can't transfer you "
    "right now. I can have someone call you back."
)
CALLBACK_ASK_NUMBER = "What number should they call?"
CALLBACK_REQUESTED = (
    "Thank you. Someone from the clinic will call you back on the next business day."
)
VERIFY_LOCKED = "I'm sorry, I wasn't able to verify your identity."
LOW_CONFIDENCE_REPROMPT = "Sorry, I didn't catch that. Could you say it again?"
LOW_CONFIDENCE_TROUBLE = "I'm having trouble hearing you."
OPT_OUT_ACK = (
    "Understood. I've stopped recording, and I won't keep a transcript of the rest of this call."
)
MODEL_TROUBLE = "I'm sorry, I'm having trouble on my end right now."
