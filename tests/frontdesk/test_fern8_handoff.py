"""FERN-8: hand off to a person on request or when the agent is stuck.

Staffed hours are Monday to Friday, 8:00 to 17:00 clinic time. The default
call clock (Wednesday 10 AM) is inside them.
"""

from __future__ import annotations

import json

import pytest

from harness import (
    AFTER_HOURS_AT,
    CALLBACK_ASK_NUMBER,
    CALLBACK_OFFER,
    CALLBACK_REQUESTED,
    LOW_CONFIDENCE_REPROMPT,
    LOW_CONFIDENCE_TROUBLE,
    MARIA_ALVAREZ,
    TRANSFER_LINE,
    TRANSFER_OFFER,
    WEEKEND_AT,
    say_as,
)


@pytest.mark.criterion("FERN-8/AC-1")
@pytest.mark.catches("transfer-asks-why")
@pytest.mark.parametrize("said", ["agent", "Representative, please.", "I want a real person."])
def test_asking_for_a_person_transfers_in_that_turn(call, said):
    reply = call.say(said)

    assert reply.text == TRANSFER_LINE
    assert reply.transferred is True
    assert [e["reason"] for e in call.events("transfer")] == ["caller_request"]


@pytest.mark.criterion("FERN-8/AC-1")
@pytest.mark.catches("transfer-asks-why")
def test_pressing_zero_transfers_in_that_turn(call):
    reply = call.press("0")

    assert reply.text == TRANSFER_LINE
    assert reply.transferred is True
    assert reply.ended is True


@pytest.mark.criterion("FERN-8/AC-1")
@pytest.mark.catches("transfer-asks-why")
def test_a_transfer_request_mid_booking_does_not_ask_why(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.say("Actually, let me talk to someone.")

    assert "what this is about" not in reply.text
    assert reply.transferred is True


@pytest.mark.criterion("FERN-8/AC-2")
def test_two_low_confidence_turns_in_a_row_offer_a_transfer(call):
    first = call.say("mm the uh", confidence=0.41)
    second = call.say("tuh thing ah", confidence=0.52)

    assert first.text == LOW_CONFIDENCE_REPROMPT
    assert second.text == f"{LOW_CONFIDENCE_TROUBLE} {TRANSFER_OFFER}"
    assert [e["reason"] for e in call.events("transfer_offered")] == ["low_confidence"]


@pytest.mark.criterion("FERN-8/AC-2")
@pytest.mark.catches("low-confidence-not-consecutive")
def test_low_confidence_turns_that_are_not_consecutive_do_not_offer_a_transfer(call):
    call.say("mm the uh", confidence=0.41)
    call.say("When is my next appointment?", confidence=0.93)
    reply = call.say("tuh thing ah", confidence=0.52)

    assert reply.text == LOW_CONFIDENCE_REPROMPT
    assert call.events("transfer_offered") == []


@pytest.mark.criterion("FERN-8/AC-2")
def test_confidence_of_exactly_point_six_is_not_low(call):
    call.say("mm the uh", confidence=0.6)
    call.say("tuh thing ah", confidence=0.6)

    assert call.events("transfer_offered") == []


@pytest.mark.criterion("FERN-8/AC-2")
def test_accepting_the_low_confidence_offer_transfers(call):
    call.say("mm the uh", confidence=0.41)
    call.say("tuh thing ah", confidence=0.52)
    reply = call.say("Yes.")

    assert reply.transferred is True
    summary = call.record()["handoffSummary"]
    assert summary is not None
    assert summary["reason"] == "low_confidence"


@pytest.mark.criterion("FERN-8/AC-3")
def test_staff_summary_has_reason_verification_and_request_and_no_personal_data(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("I'd like to book next Tuesday morning with Dr. Okafor.")
    call.say("Can I talk to a representative?")

    summary = call.record()["handoffSummary"]
    assert summary is not None
    assert summary["reason"] == "caller_request"
    assert summary["verified"] is True
    assert summary["request"] == "to book an appointment with Dr. Okafor"
    stored = json.dumps(summary)
    assert MARIA_ALVAREZ.first_name not in stored
    assert MARIA_ALVAREZ.last_name not in stored
    assert "1980" not in stored
    assert "0142" not in stored
    assert MARIA_ALVAREZ.id not in stored


@pytest.mark.criterion("FERN-8/AC-3")
def test_summary_for_an_unverified_caller_says_so(call):
    call.say("agent")

    summary = call.record()["handoffSummary"]
    assert summary is not None
    assert summary["verified"] is False
    assert summary["request"] == "to speak to a person"
    assert "Verified: no." in summary["text"]


@pytest.mark.criterion("FERN-8/AC-4")
@pytest.mark.catches("transfer-asks-why")
def test_after_hours_offers_a_callback_and_reads_the_number_back(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    offer = call.say("agent")
    readback = call.say("two one five, five five five, zero one four two")
    done = call.say("Yes, that's right.")

    assert offer.text == f"{CALLBACK_OFFER} {CALLBACK_ASK_NUMBER}"
    assert offer.transferred is False
    assert readback.text == "I have 215-555-0142. Is that right?"
    assert say_as(readback.ssml, "telephone") == ["215-555-0142"]
    assert done.text.startswith(CALLBACK_REQUESTED)
    assert [e["number"] for e in call.events("callback_requested")] == ["2155550142"]


@pytest.mark.criterion("FERN-8/AC-4")
def test_weekend_callback_number_can_be_keyed_in(start_call):
    call = start_call(at=WEEKEND_AT)
    call.press("0")
    readback = call.press("2155550177")

    assert readback.text == "I have 215-555-0177. Is that right?"
    assert call.events("callback_requested") == []
    call.say("Yes.")
    assert len(call.events("callback_requested")) == 1
    assert call.record()["transferred"] is False


@pytest.mark.criterion("FERN-8/AC-4")
def test_a_wrong_number_can_be_corrected_before_the_callback_is_requested(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    call.say("representative")
    call.say("215 555 0100")
    again = call.say("No, that's wrong.")
    call.say("215 555 0142")
    call.say("Yes.")

    assert again.text == CALLBACK_ASK_NUMBER
    assert [e["number"] for e in call.events("callback_requested")] == ["2155550142"]


@pytest.mark.criterion("FERN-8/AC-4")
def test_a_number_given_with_the_request_is_read_back_at_once(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    reply = call.say("I need to talk to someone, call me back at 215-555-0142.")

    assert reply.text == f"{CALLBACK_OFFER} I have 215-555-0142. Is that right?"
    assert call.events("callback_requested") == []
    call.say("That's correct.")
    assert [e["number"] for e in call.events("callback_requested")] == ["2155550142"]
