"""Rules that run on a caller turn before the model is called.

They are plain pattern checks on purpose: they must work when the model is
unavailable, and a safety rule that depends on the model's mood is not a rule.
"""

from __future__ import annotations

import re

_I = re.IGNORECASE

# FERN-4: emergency phrases. The rule name goes into the call record.
EMERGENCY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "chest pain",
        re.compile(
            r"\bchest (pains?|hurts?|pressure|tightness|is tight)\b|\bpain in (my|his|her) chest\b|"
            r"\bheart attack\b",
            _I,
        ),
    ),
    (
        "trouble breathing",
        re.compile(
            r"\b(trouble|difficulty|a hard time) breathing\b|\bcan'?t breathe\b|"
            r"\bcannot breathe\b|\bshort(ness)? of breath\b|\bstruggling to breathe\b|"
            r"\bnot breathing\b",
            _I,
        ),
    ),
    (
        "stroke",
        re.compile(
            r"\bstroke\b|\bface (is )?drooping\b|\bdrooping face\b|\bslurr(ed|ing)\b|"
            r"\bnumb(ness)? on one side\b|\bweakness on one side\b|"
            r"\bcan'?t (lift|raise|move) (my|his|her) arm\b",
            _I,
        ),
    ),
    ("emergency", re.compile(r"\bemergency\b", _I)),
)
_NOT_AN_EMERGENCY = re.compile(
    r"\b(not an?|isn'?t an?|is not an?|non-?)\s*emergency\b|\bno emergency\b", _I
)


def emergency_rule(text: str) -> str | None:
    """The name of the first emergency rule ``text`` matches, or None."""
    for name, pattern in EMERGENCY_RULES:
        checked = _NOT_AN_EMERGENCY.sub(" ", text) if name == "emergency" else text
        if pattern.search(checked):
            return name
    return None


# FERN-4: questions a clinician has to answer.
_MEDICATION = re.compile(
    r"\b(medications?|medicines?|meds|doses?|dosage|prescriptions?|pills?|tablets?|mg|"
    r"milligrams?|insulin|inhaler)\b",
    _I,
)
_MEDICATION_CHANGE = re.compile(
    r"\b(change|changing|stop|stopping|start|starting|increase|decrease|double|skip|switch|"
    r"lower|raise|take|taking|cut|reduce|more|less|half|should i|can i|is it ok|safe)\b",
    _I,
)
_SYMPTOM = re.compile(
    r"\b(rash|fever|headaches?|cough|pain|aches?|dizzy|dizziness|nausea|nauseous|swelling|"
    r"swollen|lump|bump|bleeding|itch|itchy|itching|sore|fatigue|tingling|spots?|"
    r"symptoms?|blood pressure|blood sugar)\b",
    _I,
)
_MEANING = re.compile(
    r"\bmeans?\b|\bwhat (is|'s) causing\b|\bwhat causes\b|\bwhat could (it|this|that) be\b|"
    r"\bis (it|this|that) (normal|serious|bad|an infection|contagious)\b|"
    r"\bshould i (be )?worried\b|\bwhy (do|does|am|is)\b|\bwhat'?s wrong\b|\bwhat is wrong\b",
    _I,
)
# Words that ask for a judgment, counted only when the caller is asking a question.
_JUDGEMENT = re.compile(
    r"\b(normal|serious|dangerous|concerning|worried|worry|causing|causes|cause)\b", _I
)
_QUESTION = re.compile(r"\?|^\W*(is|are|could|should|what|why|does|do|can|how)\b", _I)


def medical_topic(text: str) -> str | None:
    """Return "medication" or "symptom" for a clinical question, else None."""
    if _MEDICATION.search(text) and _MEDICATION_CHANGE.search(text):
        return "medication"
    if _SYMPTOM.search(text) and (
        _MEANING.search(text) or (_JUDGEMENT.search(text) and _QUESTION.search(text))
    ):
        return "symptom"
    return None


# FERN-13: the caller declines recording.
_OPT_OUT = re.compile(
    r"\b(don'?t|do not|didn'?t|did not)\s+(want|wish|consent|agree)\b[^.?!]{0,30}?"
    r"\b(recorded|recording)\b|\b(don'?t|do not|not)\s+record\b|\bstop (the )?recording\b|"
    r"\bno recording\b|\bopt out of (the )?recording\b|\bturn off (the )?recording\b|"
    r"\brather not be recorded\b|\bobject to (being )?record(ed|ing)\b",
    _I,
)


def wants_opt_out(text: str) -> bool:
    return bool(_OPT_OUT.search(text))


# FERN-8: the caller asks for a person.
_PERSON = re.compile(
    r"\b(agent|representative|real person|live person|operator|human being|a human|"
    r"(talk|speak) (to|with) (a person|someone|somebody|a human|staff))\b",
    _I,
)


def wants_person(text: str) -> bool:
    return bool(_PERSON.search(text))


# Answers to a yes-or-no offer.
_YES = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|correct|that'?s right|that is right|right|"
    r"that'?s correct|that is correct|please do|sounds good|go ahead|absolutely)\b",
    _I,
)
_NO = re.compile(r"^\W*(no|nope|nah|not really|that'?s wrong|wrong)\b", _I)


def is_yes(text: str) -> bool:
    return bool(_YES.search(text))


def is_no(text: str) -> bool:
    return bool(_NO.search(text))
