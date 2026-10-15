"""The language model behind Front Desk, and its deterministic stand-in.

In the product the dialog manager calls Claude for two jobs: understanding a
caller turn (intent and the spans that matter, such as a name or a date) and
phrasing the reply. ``Model`` is that boundary. ``ScriptedModel`` fills it with
rules and templates, so the app behaves the same on every run and its tests
are deterministic. Testing the real model's behavior is what an eval set is
for (FERN-14); this repository tests the system around the model.

Approved scripts (``frontdesk.scripts``) never come from the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, time
from typing import Protocol

from frontdesk import parsing
from frontdesk.speech import Caller, Unit, lines, sentence
from frontdesk.world import PROVIDERS, Provider


class ModelUnavailable(RuntimeError):
    """The model could not be reached for this turn."""


@dataclass(frozen=True)
class Context:
    """What the dialog manager tells the model about the call so far."""

    today: date
    awaiting: str | None = None  # "identity", "booking", or None
    offer_size: int = 0
    verified_name: str | None = None


@dataclass(frozen=True)
class Interpretation:
    intent: str  # see ScriptedModel.interpret
    name: str | None = None
    dob_text: str | None = None
    provider: Provider | None = None
    day: date | None = None
    part: str | None = None  # "morning" or "afternoon"
    time: time | None = None
    pick: int | None = None  # 0-based index into the offer; -1 is the last


class Model(Protocol):
    def interpret(self, text: str, context: Context) -> Interpretation: ...

    def write_reply(self, kind: str, facts: Mapping[str, object]) -> list[Unit]: ...


_I = re.IGNORECASE
_CANCEL = re.compile(
    r"\b(never ?mind|forget (it|that)|cancel (that|it)|changed my mind|scratch that|"
    r"don'?t book|do not book)\b",
    _I,
)
_GOODBYE = re.compile(r"\b(good ?bye|bye|that'?s all|that is all|that'?s everything)\b", _I)
_YES = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|correct|that'?s right|that is right|right|"
    r"that'?s correct|that is correct|please do|sounds good|go ahead|absolutely|that works|"
    r"perfect|book it)\b",
    _I,
)
_NO = re.compile(r"^\W*(no|nope|nah|not really)\b", _I)
_BOOK = re.compile(
    r"\b(book|schedule|make an appointment|set up an appointment|new appointment|"
    r"get an appointment|come in|see (dr|doctor)|appointment with|openings?|available|"
    r"availability)\b",
    _I,
)
_ASK_APPOINTMENT = re.compile(r"\b(appointments?|when am i|when do i)\b", _I)
_PICK = re.compile(r"\b(first|second|third|last|1st|2nd|3rd|earliest)\b(?!\s+name)", _I)
_PICK_INDEX = {
    "first": 0,
    "1st": 0,
    "earliest": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "last": -1,
}
_MORNING = re.compile(r"\bmorning\b", _I)
_AFTERNOON = re.compile(r"\bafternoon\b", _I)
_BIRTH_WORDS = re.compile(r"\b(born|birth|birthday|dob|d\.o\.b\.)\b", _I)
_RELATIONS = re.compile(
    r"\b(my|our) (wife|husband|son|daughter|mother|mom|father|dad|partner|child|kid|"
    r"brother|sister|grandmother|grandfather|friend|neighbou?r)\b|\bsomeone else\b|"
    r"\banother patient\b|\b(his|her|their) appointment",
    _I,
)
_FULL_NAME = r"([A-Z][a-z'\-]+\s+[A-Z][a-z'\-]+)"
_POSSESSIVE = re.compile(_FULL_NAME + r"'s\b")
_ABOUT_NAME = re.compile(r"\b(?:about|for)\s+" + _FULL_NAME)
_ANY_NAME = re.compile(r"(?=\b" + _FULL_NAME + r"\b)")
_SENTENCE_WORDS = frozenset(
    {"is", "can", "could", "when", "what", "where", "who", "how", "does", "do", "did", "will"}
    | {"would", "hi", "hello", "hey", "thanks", "thank", "yes", "no", "ok", "okay", "or"}
    | {"but", "so", "also", "actually", "front", "desk", "family", "am", "pm", "that", "tell"}
    | {"book", "schedule", "look", "check", "find", "give", "let", "what's", "when's", "has"}
)

_NAME_INTRO = re.compile(r"\b(?:my name is|my name's|the name is|name is)\s+(?P<rest>.+)", _I)
_NAME_INTRO_CAPS = re.compile(r"\b(?:this is|i'm|i am|it's|it is)\s+(?P<rest>.+)", _I)
_NOT_NAME_WORDS = frozenset(
    {
        "and",
        "born",
        "date",
        "my",
        "dob",
        "birthday",
        "i",
        "calling",
        "here",
        "from",
        "with",
        "at",
        "on",
        "about",
        "the",
        "a",
        "an",
        "phone",
        "number",
        "to",
        "for",
        "speaking",
        "please",
        "next",
        "this",
        "is",
        "dr",
        "doctor",
        "morning",
        "afternoon",
        "today",
        "tomorrow",
        "brookside",
        "clinic",
    }
    | set(parsing.MONTHS)
    | set(parsing.WEEKDAYS)
)


def _leading_name(rest: str, require_caps: bool) -> str | None:
    words: list[str] = []
    for token in rest.split():
        word = token.rstrip(",.;:!?")
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word):
            break
        if word.lower() in _NOT_NAME_WORDS or (require_caps and not word[0].isupper()):
            break
        words.append(word)
        if word != token or len(words) == 4:
            break  # trailing punctuation, or a fourth word, ends a name
    if require_caps and len(words) < 2:
        return None
    return " ".join(words) or None


def find_name(text: str, expecting_identity: bool, has_dob: bool) -> str | None:
    """The name a caller gives for themselves, or None."""
    match = _NAME_INTRO.search(text)
    if match:
        return _leading_name(match["rest"], require_caps=False)
    match = _NAME_INTRO_CAPS.search(text)
    if match:
        name = _leading_name(match["rest"], require_caps=True)
        if name:
            return name
    if expecting_identity or has_dob:
        return _leading_name(text, require_caps=True)
    return None


def _not_a_person(name: str) -> bool:
    first = name.split()[0].lower()
    return first in _NOT_NAME_WORDS


def _looks_like_a_name(name: str) -> bool:
    return not any(w.lower() in _NOT_NAME_WORDS | _SENTENCE_WORDS for w in name.split())


def mentions_other_patient(text: str, verified_name: str | None) -> bool:
    """A relative, someone else, or a full name that is not the verified caller's."""
    if _RELATIONS.search(text):
        return True
    own = (verified_name or "").lower()
    for pattern in (_POSSESSIVE, _ABOUT_NAME):
        for match in pattern.finditer(text):
            name = match.group(1)
            if not _not_a_person(name) and name.lower() != own:
                return True
    if verified_name:  # once verified, any other full name is someone else
        for match in _ANY_NAME.finditer(text):
            name = match.group(1)
            if _looks_like_a_name(name) and name.lower() != own:
                return True
    return False


def find_provider(text: str) -> Provider | None:
    lowered = text.lower()
    for provider in PROVIDERS:
        if re.search(rf"\b{provider.id}\b", lowered):
            return provider
    if re.search(r"\blindquist\b", lowered):
        return PROVIDERS[1]
    return None


TEMPLATES: dict[str, str] = {
    "verify_ask": (
        "Before I can help with that, I need to confirm who you are. "
        "Please tell me your full name and date of birth."
    ),
    "verify_need_dob": "Thanks. And what is your date of birth?",
    "verify_need_name": "Thanks. And what is your full name?",
    "verify_retry": (
        "I'm sorry, I couldn't match that to a patient record. "
        "Please tell me your full name and date of birth again."
    ),
    "verified": "Thanks, {first_name}, you're verified.",
    "how_help": "How can I help you today?",
    "anything_else": "Is there anything else I can help with?",
    "next_appointment": "Your next appointment is with {provider} on {when}.",
    "no_appointments": "I don't see any upcoming appointments for you. I can book one if you like.",
    "other_patient": (
        "I'm sorry, I can only share information about the patient I've verified on this call."
    ),
    "other_patient_unverified": (
        "I'm sorry, I can only share appointment details with the patient, "
        "once I've confirmed who they are."
    ),
    "ask_provider": "Which provider would you like to see: Dr. Okafor, Dr. Lindqvist or Dr. Patel?",
    "ask_day": "What day would you like to come in?",
    "offer_one": "{provider} has {slot} open. Would you like me to book it?",
    "offer_many": "{provider} has these openings: {slots}. Which would you like?",
    "offer_nearest": (
        "I don't have anything open with {provider} then. "
        "The nearest openings are {slots}. Which would you like?"
    ),
    "slot_taken": (
        "I'm sorry, {slot} was just taken by someone else. "
        "The next openings with {provider} are {slots}. Which would you like?"
    ),
    "no_openings": "I'm sorry, {provider} has no openings on the schedule right now.",
    "which_one": "Which one would you like: {slots}?",
    "booked": "You're booked with {provider} on {when}.",
    "already_booked": "You're already booked with {provider} on {when}.",
    "booking_cancelled": "No problem. I haven't booked anything.",
    "nothing_pending": "Okay. How else can I help?",
    "goodbye": "Thank you for calling Brookside Family Clinic. Goodbye.",
}


class ScriptedModel:
    """Rule-based understanding and templated replies. Deterministic."""

    def __init__(self, available: bool = True) -> None:
        self.available = available

    def _check(self) -> None:
        if not self.available:
            raise ModelUnavailable("the model is unavailable for this call")

    def interpret(self, text: str, context: Context) -> Interpretation:
        """Intents: cancel, goodbye, other_patient, pick, confirm, decline, book,
        ask_appointment, identify, unknown."""
        self._check()
        dob_text = self._find_dob(text, context)
        rest = text.replace(dob_text, " ") if dob_text else text
        name = find_name(rest, context.awaiting == "identity", dob_text is not None)
        provider = find_provider(rest)
        day = parsing.parse_day(rest, context.today)
        part = (
            "morning" if _MORNING.search(rest) else "afternoon" if _AFTERNOON.search(rest) else None
        )
        at = parsing.parse_time(rest)
        pick_match = _PICK.search(rest)
        pick = _PICK_INDEX[pick_match[1].lower()] if pick_match else None
        found = {
            "name": name,
            "dob_text": dob_text,
            "provider": provider,
            "day": day,
            "part": part,
            "time": at,
            "pick": pick,
        }
        booking_detail = any(v is not None for v in (provider, day, part, at))

        if _CANCEL.search(text):
            intent = "cancel"
        elif _GOODBYE.search(text):
            intent = "goodbye"
        elif mentions_other_patient(text, context.verified_name):
            intent = "other_patient"
        elif context.offer_size and (pick is not None or at is not None):
            intent = "pick"
        elif _YES.search(text):
            intent = "confirm"
        elif _NO.search(text):
            intent = "decline"
        elif (
            _BOOK.search(rest)
            or (provider and (day or part or at))
            or (context.awaiting == "booking" and booking_detail)
        ):
            intent = "book"
        elif _ASK_APPOINTMENT.search(rest):
            intent = "ask_appointment"
        elif name or dob_text:
            intent = "identify"
        else:
            intent = "unknown"
        return Interpretation(intent=intent, **found)

    @staticmethod
    def _find_dob(text: str, context: Context) -> str | None:
        birth_words = bool(_BIRTH_WORDS.search(text))
        for span in parsing.find_date_spans(text):
            parsed, had_year = parsing.parse_calendar_date(span, context.today.year)
            if not had_year:
                continue
            past_year = parsed is not None and parsed.year < context.today.year
            if birth_words or context.awaiting == "identity" or past_year:
                return span
        return None

    def write_reply(self, kind: str, facts: Mapping[str, object]) -> list[Unit]:
        self._check()
        if kind == "not_understood":
            heard = str(facts["heard"])
            return [
                sentence("I'm sorry, I didn't follow \"", Caller(heard), '".'),
                sentence("I can check or book appointments, or connect you with our staff."),
            ]
        return lines(TEMPLATES[kind].format(**facts))
