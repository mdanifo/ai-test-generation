"""Spoken forms and SSML rendering (FERN-11).

A reply is built from sentences made of pieces. A plain string is text the app
wrote; ``Caller`` is text that came from the caller; ``Phone`` is a phone
number, read digit by digit. ``render`` turns them into the plain text and the
SSML the speech engine reads. The engine supports only the ``speak``,
``break``, ``say-as`` and ``prosody`` tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from xml.sax.saxutils import escape

from frontdesk import bugs
from frontdesk.world import CLINIC_TZ


@dataclass(frozen=True)
class Caller:
    """Words the caller said. Always escaped before it goes into SSML."""

    text: str


@dataclass(frozen=True)
class Phone:
    """A phone number: shown as 215-555-0142, spoken digit by digit."""

    digits: str


Piece = str | Caller | Phone


@dataclass(frozen=True)
class Sentence:
    pieces: tuple[Piece, ...]
    rate: str | None = None  # prosody rate, for lines that must be heard clearly


@dataclass(frozen=True)
class Pause:
    ms: int = 300


Unit = Sentence | Pause


@dataclass(frozen=True)
class Reply:
    text: str
    ssml: str


def sentence(*pieces: Piece, rate: str | None = None) -> Sentence:
    return Sentence(tuple(pieces), rate)


def format_phone(digits: str) -> str:
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"
    return digits


def _piece_text(piece: Piece) -> str:
    if isinstance(piece, Caller):
        return piece.text
    if isinstance(piece, Phone):
        return format_phone(piece.digits)
    return piece


def _piece_ssml(piece: Piece) -> str:
    if isinstance(piece, Caller):
        if bugs.active("ssml-unescaped"):
            return piece.text
        return escape(piece.text)
    if isinstance(piece, Phone):
        return f'<say-as interpret-as="telephone">{format_phone(piece.digits)}</say-as>'
    return escape(piece)


def render(units: list[Unit]) -> Reply:
    texts: list[str] = []
    marks: list[str] = []
    for unit in units:
        if isinstance(unit, Pause):
            marks.append(f'<break time="{unit.ms}ms"/>')
            continue
        texts.append("".join(_piece_text(p) for p in unit.pieces))
        body = "".join(_piece_ssml(p) for p in unit.pieces)
        marks.append(f'<prosody rate="{unit.rate}">{body}</prosody>' if unit.rate else body)
    return Reply(text=" ".join(texts), ssml="<speak>" + " ".join(marks) + "</speak>")


def lines(*texts: str) -> list[Unit]:
    """Plain sentences, the common case."""
    return [sentence(t) for t in texts]


def ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def spoken_date(day: date) -> str:
    """Tuesday, October 20th"""
    return f"{day:%A}, {day:%B} {ordinal(day.day)}"


def spoken_time(moment: datetime) -> str:
    """9:30 AM, 2 PM, 12 PM"""
    local = moment.astimezone(CLINIC_TZ)
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    if local.minute:
        return f"{hour}:{local.minute:02d} {meridiem}"
    return f"{hour} {meridiem}"


def spoken_slot(moment: datetime) -> str:
    """Tuesday, October 20th at 9:30 AM"""
    local = moment.astimezone(CLINIC_TZ)
    return f"{spoken_date(local.date())} at {spoken_time(local)}"


def spoken_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
