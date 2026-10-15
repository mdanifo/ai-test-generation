"""The harness's copy of the world matches the app, and its helpers behave as documented."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

import harness
from frontdesk import bugs, scripts
from frontdesk.speech import spoken_slot
from frontdesk.world import PATIENTS, PROVIDERS, SEED_APPOINTMENTS
from harness import patients, say_as, spoken_text, ssml_tags, well_formed

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "HARNESS.md"


def test_harness_patients_match_the_seeded_world():
    now = datetime.fromisoformat(harness.DEFAULT_AT)
    for mine, theirs in zip(patients(), PATIENTS, strict=True):
        upcoming = sorted(
            (a for a in SEED_APPOINTMENTS if a.patient_id == theirs.id and a.start > now),
            key=lambda a: a.start,
        )
        assert (mine.id, mine.full_name, mine.dob) == (theirs.id, theirs.full_name, theirs.dob)
        assert mine.phone.replace("-", "") == theirs.phone
        if not upcoming:
            assert mine.next_appointment is None
            continue
        first = upcoming[0]
        assert mine.next_appointment.start == first.start.isoformat()
        assert mine.next_appointment.spoken == spoken_slot(first.start)
        assert mine.next_appointment.provider == next(
            p.name for p in PROVIDERS if p.id == first.provider_id
        )


def test_harness_providers_match_the_seeded_world():
    assert tuple(p.name for p in PROVIDERS) == harness.PROVIDERS


@pytest.mark.parametrize(
    ("mine", "theirs"),
    [
        ("DISCLOSURE", "DISCLOSURE"),
        ("EMERGENCY_SCRIPT", "EMERGENCY"),
        ("NURSE_LINE_OFFER", "NURSE_LINE_OFFER"),
        ("TRANSFER_LINE", "TRANSFER"),
        ("TRANSFER_OFFER", "TRANSFER_OFFER"),
        ("CALLBACK_OFFER", "CALLBACK_OFFER"),
        ("CALLBACK_ASK_NUMBER", "CALLBACK_ASK_NUMBER"),
        ("CALLBACK_REQUESTED", "CALLBACK_REQUESTED"),
        ("VERIFY_LOCKED", "VERIFY_LOCKED"),
        ("LOW_CONFIDENCE_REPROMPT", "LOW_CONFIDENCE_REPROMPT"),
        ("LOW_CONFIDENCE_TROUBLE", "LOW_CONFIDENCE_TROUBLE"),
        ("OPT_OUT_ACK", "OPT_OUT_ACK"),
        ("MODEL_TROUBLE", "MODEL_TROUBLE"),
    ],
)
def test_harness_lines_match_the_approved_scripts(mine, theirs):
    assert getattr(harness, mine) == getattr(scripts, theirs)


def test_the_card_texts_of_the_two_fixed_scripts():
    assert harness.DISCLOSURE == "This call may be recorded and transcribed to help us serve you."
    assert harness.EMERGENCY_SCRIPT == (
        "If this is a medical emergency, please hang up and call 911 now."
    )


def test_ssml_helpers_on_good_markup():
    ssml = (
        '<speak>Call <say-as interpret-as="telephone">215-555-0142</say-as> '
        '<break time="300ms"/> now &amp; then.</speak>'
    )

    assert well_formed(ssml) is True
    assert ssml_tags(ssml) == {"speak", "say-as", "break"}
    assert spoken_text(ssml) == "Call 215-555-0142 now & then."
    assert say_as(ssml, "telephone") == ["215-555-0142"]
    assert say_as(ssml, "date") == []


@pytest.mark.parametrize("bad", ["<speak>Salt & pepper</speak>", "<speak>open", "<p>not ssml</p>"])
def test_well_formed_is_false_for_bad_markup(bad):
    assert well_formed(bad) is False


def test_ssml_tags_raises_on_markup_that_does_not_parse():
    with pytest.raises(ValueError, match="not well-formed SSML"):
        ssml_tags("<speak>Salt & pepper</speak>")


def test_the_guide_documents_every_public_harness_name():
    guide = GUIDE.read_text(encoding="utf-8")
    missing = [name for name in harness.__all__ if f"`{name}" not in guide]
    fixtures = ["call", "start_call", "api", "frontdesk_url"]
    missing += [f for f in fixtures if f"`{f}`" not in guide]
    assert missing == []


def test_the_guide_never_mentions_the_seeded_bugs():
    guide = GUIDE.read_text(encoding="utf-8").lower()

    assert "seeded" not in guide
    assert re.search(r"\bbugs?\b", guide) is None
    assert not any(name in guide for name in bugs.CATALOG)
    assert "SEEDED_BUG".lower() not in guide
