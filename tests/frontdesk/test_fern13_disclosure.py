"""FERN-13: disclose recording and honor an opt-out."""

from __future__ import annotations

import pytest

from harness import (
    AFTER_HOURS_AT,
    DISCLOSURE,
    MARIA_ALVAREZ,
    OPT_OUT_ACK,
    TRANSFER_LINE,
    spoken_text,
)


@pytest.mark.criterion("FERN-13/AC-1")
@pytest.mark.parametrize(
    "start",
    [
        {},
        {"model_available": False},
        {"at": AFTER_HOURS_AT},
    ],
)
def test_every_call_opens_with_the_disclosure_word_for_word(start_call, start):
    call = start_call(**start)

    assert call.greeting.text.startswith(DISCLOSURE)
    assert spoken_text(call.greeting.ssml).startswith(DISCLOSURE)
    first_turn = call.record()["turns"][0]
    assert first_turn["speaker"] == "agent"
    assert first_turn["text"].startswith(DISCLOSURE)


@pytest.mark.criterion("FERN-13/AC-2")
@pytest.mark.catches("disclosure-skipped-on-barge-in")
def test_a_barge_in_during_the_greeting_still_gets_the_full_disclosure(call):
    reply = call.say("Hi, when is my appointment?", barge_in=True)

    assert reply.text.startswith(DISCLOSURE)
    assert spoken_text(reply.ssml).startswith(DISCLOSURE)
    assert "full name and date of birth" in reply.text


@pytest.mark.criterion("FERN-13/AC-2")
@pytest.mark.catches("disclosure-skipped-on-barge-in")
def test_a_barge_in_asking_for_a_person_hears_the_disclosure_before_the_transfer(call):
    reply = call.say("agent", barge_in=True)

    assert reply.text == f"{DISCLOSURE} {TRANSFER_LINE}"
    assert reply.transferred is True


@pytest.mark.criterion("FERN-13/AC-2")
def test_the_disclosure_is_repeated_once_not_on_every_turn(call):
    call.say("Hi, when is my appointment?", barge_in=True)
    reply = call.verify_as(MARIA_ALVAREZ)

    assert DISCLOSURE not in reply.text


@pytest.mark.criterion("FERN-13/AC-3")
def test_opting_out_stops_transcript_storage_for_the_rest_of_the_call(call):
    call.say("When is my next appointment?")
    reply = call.say("I don't want to be recorded.")
    call.verify_as(MARIA_ALVAREZ)

    assert reply.text.startswith(OPT_OUT_ACK)
    record = call.record()
    assert record["transcriptStored"] is False
    assert record["recording"] is False
    stored = [t["text"] for t in record["turns"]]
    assert "When is my next appointment?" in stored
    assert "I don't want to be recorded." not in stored
    assert not any(MARIA_ALVAREZ.last_name in text for text in stored)


@pytest.mark.criterion("FERN-13/AC-3")
def test_after_opting_out_the_call_carries_on(call):
    call.say("Please don't record this call.")
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book Dr. Lindqvist on October 20th at 9:30.")
    call.say("Yes.")

    assert len(call.appointments()) == 1
    assert call.record()["optedOut"] is True


@pytest.mark.criterion("FERN-13/AC-3")
def test_after_opting_out_the_caller_can_still_be_transferred(call):
    call.say("Stop recording.")
    reply = call.say("representative")

    assert reply.transferred is True


@pytest.mark.criterion("FERN-13/AC-3")
@pytest.mark.parametrize(
    "said",
    [
        "I do not want this call recorded.",
        "I'd rather not be recorded.",
        "Please don't record me.",
        "Turn off the recording.",
    ],
)
def test_ways_of_declining_recording(call, said):
    call.say(said)

    assert call.record()["optedOut"] is True


@pytest.mark.criterion("FERN-13/AC-4")
def test_the_record_shows_no_opt_out_by_default(call):
    call.say("When is my next appointment?")

    record = call.record()
    assert record["optedOut"] is False
    assert record["transcriptStored"] is True


@pytest.mark.criterion("FERN-13/AC-4")
def test_the_record_shows_the_opt_out(call):
    call.say("I do not consent to being recorded.")

    record = call.record()
    assert record["optedOut"] is True
    assert [e["type"] for e in record["events"]] == ["recording_stopped"]
