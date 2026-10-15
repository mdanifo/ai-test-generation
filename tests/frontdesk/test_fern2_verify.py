"""FERN-2: verify the caller before sharing any appointment details."""

from __future__ import annotations

import re

import pytest

from harness import (
    JAMES_WHITAKER,
    MARIA_ALVAREZ,
    PRIYA_RAMAN,
    PROVIDERS,
    THOMAS_BRENNAN,
    TRANSFER_LINE,
    TRANSFER_OFFER,
    VERIFY_LOCKED,
)

TIME_OF_DAY = re.compile(r"\b\d{1,2}(:\d{2})? (AM|PM)\b")


@pytest.mark.criterion("FERN-2/AC-1")
def test_unverified_caller_is_asked_for_name_and_dob_and_hears_no_details(call):
    reply = call.say("When is my next appointment?")

    assert "Please tell me your full name and date of birth." in reply.text
    assert not any(provider in reply.text for provider in PROVIDERS)
    assert TIME_OF_DAY.search(reply.text) is None
    assert call.record()["verified"] is False


@pytest.mark.criterion("FERN-2/AC-1")
def test_a_name_alone_does_not_unlock_the_appointment(call):
    call.say("When is my next appointment?")
    reply = call.say("I'm Maria Alvarez.")

    assert reply.text == "Thanks. And what is your date of birth?"
    assert MARIA_ALVAREZ.next_appointment.spoken not in reply.text
    assert call.record()["verified"] is False


@pytest.mark.criterion("FERN-2/AC-2")
@pytest.mark.catches("other-patient-leak")
def test_matching_name_and_dob_verifies_and_gives_that_patients_next_appointment(call):
    call.say("When is my next appointment?")
    reply = call.verify_as(MARIA_ALVAREZ)

    assert reply.text == (
        "Thanks, Maria, you're verified. Your next appointment is with Dr. Lindqvist "
        "on Thursday, October 22nd at 2 PM."
    )
    record = call.record()
    assert record["verified"] is True
    assert record["patientId"] == MARIA_ALVAREZ.id


@pytest.mark.criterion("FERN-2/AC-2")
@pytest.mark.catches("other-patient-leak")
def test_verified_caller_asking_later_hears_their_own_appointment(call):
    call.verify_as(PRIYA_RAMAN)
    reply = call.say("When is my next appointment?")

    assert (
        reply.text == "Your next appointment is with Dr. Okafor on Tuesday, October 27th at 9 AM."
    )
    assert JAMES_WHITAKER.next_appointment.spoken not in reply.text


@pytest.mark.criterion("FERN-2/AC-2")
@pytest.mark.catches("other-patient-leak")
def test_verified_caller_with_no_upcoming_appointment_hears_none(call):
    call.verify_as(THOMAS_BRENNAN)
    reply = call.say("Do I have an appointment coming up?")

    assert reply.text.startswith("I don't see any upcoming appointments for you.")
    assert not any(provider in reply.text for provider in PROVIDERS)


@pytest.mark.criterion("FERN-2/AC-3")
@pytest.mark.catches("dob-format-strict")
@pytest.mark.parametrize(
    "said",
    [
        "This is Maria Alvarez, born March 3rd 1980.",
        "My name is Maria Alvarez and my date of birth is March 3rd, 1980.",
        "Maria Alvarez, 3/3/80.",
        "It's Maria Alvarez, date of birth 03/03/1980.",
        "My name is Maria Alvarez, I was born on the third of March 1980.",
    ],
)
def test_date_of_birth_in_another_form_still_verifies(call, said):
    reply = call.say(said)

    assert reply.text.startswith("Thanks, Maria, you're verified.")
    assert call.record()["patientId"] == MARIA_ALVAREZ.id


@pytest.mark.criterion("FERN-2/AC-3")
def test_date_of_birth_in_the_record_form_verifies(call):
    call.say("My name is Maria Alvarez, date of birth 1980-03-03.")

    assert call.record()["verified"] is True


@pytest.mark.criterion("FERN-2/AC-4")
@pytest.mark.catches("verify-attempts-off-by-one")
def test_three_failed_attempts_stop_asking_and_offer_a_transfer(call):
    call.say("My name is Maria Alvarez, born January 1st 1990.")
    call.say("My name is Maria Alvarez, born January 2nd 1990.")
    reply = call.say("My name is Maria Alvarez, born January 3rd 1990.")

    assert reply.text == f"{VERIFY_LOCKED} {TRANSFER_OFFER}"
    assert "date of birth" not in reply.text
    record = call.record()
    assert "verification_failed" in record["flags"]
    assert record["verified"] is False


@pytest.mark.criterion("FERN-2/AC-4")
@pytest.mark.catches("verify-attempts-off-by-one")
def test_after_three_failures_a_correct_identity_is_not_accepted(call):
    call.say("My name is Maria Alvarez, born January 1st 1990.")
    call.say("My name is Maria Alvarez, born January 2nd 1990.")
    call.say("My name is Maria Alvarez, born January 3rd 1990.")
    reply = call.verify_as(MARIA_ALVAREZ)

    assert reply.text == f"{VERIFY_LOCKED} {TRANSFER_OFFER}"
    assert call.record()["verified"] is False


@pytest.mark.criterion("FERN-2/AC-4")
@pytest.mark.catches("verify-attempts-off-by-one")
def test_accepting_the_offer_after_three_failures_transfers(call):
    call.say("My name is Maria Alvarez, born January 1st 1990.")
    call.say("My name is Maria Alvarez, born January 2nd 1990.")
    call.say("My name is Maria Alvarez, born January 3rd 1990.")
    reply = call.say("Yes, please.")

    assert reply.text == TRANSFER_LINE
    assert reply.transferred is True
    summary = call.record()["handoffSummary"]
    assert summary is not None
    assert summary["reason"] == "verification_failed"


@pytest.mark.criterion("FERN-2/AC-4")
def test_two_failed_attempts_ask_again_without_flagging(call):
    call.say("My name is Maria Alvarez, born January 1st 1990.")
    reply = call.say("My name is Maria Alvarez, born January 2nd 1990.")

    assert "Please tell me your full name and date of birth again." in reply.text
    assert "verification_failed" not in call.record()["flags"]


@pytest.mark.criterion("FERN-2/AC-5")
def test_verified_caller_asking_about_another_patient_by_name_hears_nothing(call):
    call.verify_as(MARIA_ALVAREZ)
    reply = call.say("When is James Whitaker's appointment?")

    assert reply.text == (
        "I'm sorry, I can only share information about the patient I've verified on this call."
    )
    assert JAMES_WHITAKER.next_appointment.provider not in reply.text
    assert TIME_OF_DAY.search(reply.text) is None


@pytest.mark.criterion("FERN-2/AC-5")
def test_verified_caller_asking_about_a_relative_hears_nothing(call):
    call.verify_as(JAMES_WHITAKER)
    reply = call.say("Can you tell me when my wife's appointment is?")

    assert "only share information about the patient I've verified" in reply.text
    assert TIME_OF_DAY.search(reply.text) is None
    assert not any(provider in reply.text for provider in PROVIDERS)


@pytest.mark.criterion("FERN-2/AC-5")
def test_a_bare_full_name_of_another_patient_is_declined(call):
    call.verify_as(MARIA_ALVAREZ)
    reply = call.say("Is James Whitaker booked this week?")

    assert "only share information about the patient I've verified" in reply.text
    assert JAMES_WHITAKER.next_appointment.spoken not in reply.text
