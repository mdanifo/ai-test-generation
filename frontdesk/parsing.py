"""Reading dates, times and phone numbers out of recognized speech.

The scripted model uses the ``find_*`` functions to pick spans out of a turn;
the verification and schedule tools use the ``parse_*`` functions to turn a
span into a value.
"""

from __future__ import annotations

import re
from datetime import date, time, timedelta

from frontdesk import bugs

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ALIASES = {name[:3]: number for name, number in MONTHS.items()} | {"sept": 9}
MONTH_LOOKUP = MONTHS | _MONTH_ALIASES

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "twenty-fourth": 24,
    "twenty-fifth": 25,
    "twenty-sixth": 26,
    "twenty-seventh": 27,
    "twenty-eighth": 28,
    "twenty-ninth": 29,
    "thirtieth": 30,
    "thirty-first": 31,
}

_MONTH_RE = r"(?:" + "|".join(sorted(MONTH_LOOKUP, key=len, reverse=True)) + r")\.?"
_ORD_WORD_RE = "|".join(
    sorted((w.replace("-", "[- ]") for w in _ORDINAL_WORDS), key=len, reverse=True)
)
_DAY_RE = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?|{_ORD_WORD_RE})"
_YEAR_RE = r"\d{4}"

_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMERIC = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})\b")
_NUMERIC_NO_YEAR = re.compile(r"\b\d{1,2}/\d{1,2}\b(?![/.-]\d)")
_MONTH_FIRST = re.compile(
    rf"\b{_MONTH_RE}\s+(?:the\s+)?{_DAY_RE}\b(?:,?\s+{_YEAR_RE}\b)?", re.IGNORECASE
)
_DAY_FIRST = re.compile(
    rf"\b(?:the\s+)?{_DAY_RE}\s+(?:of\s+)?{_MONTH_RE}(?:,?\s+{_YEAR_RE}\b)?", re.IGNORECASE
)


def find_date_spans(text: str) -> list[str]:
    """Every date-like span in ``text``, in order of appearance, without overlaps."""
    found: list[tuple[int, int]] = []
    for pattern in (_ISO, _NUMERIC, _MONTH_FIRST, _DAY_FIRST, _NUMERIC_NO_YEAR):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < e and s < end for s, e in found):
                continue
            found.append((start, end))
    return [text[s:e].strip(" ,") for s, e in sorted(found)]


def _day_number(token: str) -> int | None:
    token = token.lower().strip().replace(" ", "-")
    if token in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[token]
    digits = re.match(r"(\d{1,2})(?:st|nd|rd|th)?$", token)
    return int(digits.group(1)) if digits else None


def _full_year(value: str, ref_year: int) -> int:
    year = int(value)
    if len(value) == 2:
        year += 1900 if year > ref_year % 100 else 2000
    return year


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_calendar_date(span: str, ref_year: int) -> tuple[date | None, bool]:
    """Parse one date span. Returns (date, had_year). US order for numeric forms."""
    text = span.strip().strip(",").lower()
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        return _safe_date(int(iso[1]), int(iso[2]), int(iso[3])), True
    numeric = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2}|\d{4}))?", text)
    if numeric:
        year = _full_year(numeric[3], ref_year) if numeric[3] else ref_year
        return _safe_date(year, int(numeric[1]), int(numeric[2])), bool(numeric[3])
    month_first = re.fullmatch(
        rf"({_MONTH_RE})\s+(?:the\s+)?({_DAY_RE})(?:,?\s+({_YEAR_RE}))?", text
    )
    day_first = re.fullmatch(
        rf"(?:the\s+)?({_DAY_RE})\s+(?:of\s+)?({_MONTH_RE})(?:,?\s+({_YEAR_RE}))?", text
    )
    if month_first:
        month_word, day_word, year_word = month_first[1], month_first[2], month_first[3]
    elif day_first:
        day_word, month_word, year_word = day_first[1], day_first[2], day_first[3]
    else:
        return None, False
    month = MONTH_LOOKUP.get(month_word.rstrip("."))
    day_number = _day_number(day_word)
    if month is None or day_number is None:
        return None, False
    year = int(year_word) if year_word else ref_year
    return _safe_date(year, month, day_number), bool(year_word)


def parse_dob(span: str, ref_year: int) -> date | None:
    """A date of birth in any form callers use: "March 3rd, 1980", "3/3/80", "1980-03-03"."""
    if bugs.active("dob-format-strict"):
        try:
            return date.fromisoformat(span.strip())
        except ValueError:
            return None
    parsed, had_year = parse_calendar_date(span, ref_year)
    return parsed if had_year else None


def parse_day(text: str, today: date) -> date | None:
    """The day a booking request names, relative to the call's date.

    "today", "tomorrow", "Tuesday", "next Tuesday" and "this Tuesday" (all three
    mean the first Tuesday after today), "October 20th", "10/20", "2026-10-20".
    """
    lowered = text.lower()
    if re.search(r"\btoday\b", lowered):
        return today
    if re.search(r"\btomorrow\b", lowered):
        return today + timedelta(days=1)
    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}s?\b", lowered):
            ahead = (index - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
    for span in find_date_spans(text):
        parsed, had_year = parse_calendar_date(span, today.year)
        if parsed is None:
            continue
        if not had_year and parsed < today:
            parsed = _safe_date(today.year + 1, parsed.month, parsed.day)
        return parsed
    return None


_TIME_COLON = re.compile(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?![\w:])", re.IGNORECASE)
_TIME_MERIDIEM = re.compile(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?!\w)", re.IGNORECASE)
_TIME_AT = re.compile(r"\bat\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th|/|-|\.\d|:))", re.IGNORECASE)


def _to_time(hour: int, minute: int, meridiem: str | None) -> time | None:
    if minute > 59 or hour > 23:
        return None
    marker = (meridiem or "").lower().replace(".", "")
    if marker == "pm" and hour < 12:
        hour += 12
    elif marker == "am" and hour == 12:
        hour = 0
    elif not marker and 1 <= hour <= 7:
        hour += 12  # clinic hours: "at 2" means 2 PM
    return time(hour, minute)


def parse_time(text: str) -> time | None:
    """A clock time: "9:30", "9:30 am", "2 pm", "at 2" (2 PM), "noon", "14:00"."""
    if re.search(r"\bnoon\b", text, re.IGNORECASE):
        return time(12, 0)
    match = _TIME_COLON.search(text)
    if match:
        return _to_time(int(match[1]), int(match[2]), match[3])
    match = _TIME_MERIDIEM.search(text)
    if match:
        return _to_time(int(match[1]), 0, match[2])
    match = _TIME_AT.search(text)
    if match:
        return _to_time(int(match[1]), 0, None)
    return None


_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def find_phone_digits(text: str) -> str:
    """Digits of a phone number said as digits or as words ("two one five...")."""
    digits = []
    for token in re.findall(r"[a-z]+|\d+", text.lower()):
        if token.isdigit():
            digits.append(token)
        elif token in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[token])
    return "".join(digits)
