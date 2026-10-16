"""Flows the harness guide describes that no card's criteria pin down.

Declining an offer, an unused key, a short callback number, speech the agent does not
follow, a booking given one piece at a time, the model being down. A generated test can
lean on any of these, so what the guide says about them has to hold.
"""

from __future__ import annotations

from harness import (
    AFTER_HOURS_AT,
    MARIA_ALVAREZ,
    MODEL_TROUBLE,
    NURSE_LINE_OFFER,
    TRANSFER_LINE,
    TRANSFER_OFFER,
)

OKAY_HOW_CAN_I_HELP = "Okay. How can I help?"


def test_no_to_a_transfer_offer_keeps_the_call_going(call):
    call.say("mm the uh", confidence=0.4)
    call.say("tuh thing ah", confidence=0.4)
    reply = call.say("No, thanks.")

    assert reply.text == OKAY_HOW_CAN_I_HELP
    assert reply.transferred is False
    assert call.events("transfer") == []


def test_no_to_the_nurse_line_offer_sends_no_message(call):
    offer = call.say("Should I double my blood pressure medication dose?")
    reply = call.say("No.")

    assert offer.text == NURSE_LINE_OFFER
    assert reply.text == OKAY_HOW_CAN_I_HELP
    assert call.events("nurse_message_requested") == []


def test_an_unused_key_is_explained(call):
    reply = call.press("5")

    assert reply.text == "Sorry, that key isn't used here. You can press 0 to reach our staff."
    assert reply.transferred is False


def test_a_short_callback_number_is_asked_for_again(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    call.say("agent")
    reply = call.say("215 555")

    assert reply.text == (
        "Sorry, I didn't get a full ten-digit number. What number should they call?"
    )
    assert call.events("callback_requested") == []


def test_a_leading_one_is_dropped_from_a_callback_number(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    call.say("agent")
    reply = call.say("1 215 555 0142")

    assert reply.text == "I have 215-555-0142. Is that right?"


def test_speech_the_agent_does_not_follow_is_quoted_back(call):
    reply = call.say("Purple monkey dishwasher")

    assert reply.text == (
        "I'm sorry, I didn't follow \"Purple monkey dishwasher\". "
        "I can check or book appointments, or connect you with our staff."
    )
    assert call.appointments() == []


def test_a_booking_asked_for_one_piece_at_a_time(call):
    call.verify_as(MARIA_ALVAREZ)
    provider = call.say("I'd like to book an appointment.")
    day = call.say("Dr. Patel, please.")
    call.say("Monday afternoon.")

    assert provider.text == (
        "Which provider would you like to see: Dr. Okafor, Dr. Lindqvist or Dr. Patel?"
    )
    assert day.text == "What day would you like to come in?"
    assert [(o["provider"], o["slots"]) for o in call.events("slots_offered")] == [
        (
            "Dr. Patel",
            [
                "2026-10-19T12:00:00-04:00",
                "2026-10-19T12:30:00-04:00",
                "2026-10-19T13:00:00-04:00",
            ],
        )
    ]


def test_a_booking_asked_for_before_verifying_resumes_after_it(call):
    asked = call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.verify_as(MARIA_ALVAREZ)

    assert asked.text.startswith("Before I can help with that, I need to confirm who you are.")
    assert reply.text == (
        "Thanks, Maria, you're verified. Dr. Okafor has these openings: Tuesday, October 20th "
        "at 11 AM and Tuesday, October 20th at 11:30 AM. Which would you like?"
    )


def test_a_date_of_birth_given_before_the_name_gets_a_request_for_the_name(call):
    asked = call.say("I was born on March 3rd 1980.")
    reply = call.say("Maria Alvarez.")

    assert asked.text == "Thanks. And what is your full name?"
    assert reply.text.startswith("Thanks, Maria, you're verified.")
    assert call.record()["patientId"] == MARIA_ALVAREZ.id


def test_picking_a_slot_that_was_not_offered_asks_which_one(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.say("The third one.")

    assert reply.text == (
        "Which one would you like: Tuesday, October 20th at 11 AM and "
        "Tuesday, October 20th at 11:30 AM?"
    )
    assert call.appointments() == []


def test_an_unverified_caller_asking_about_someone_else_hears_nothing(call):
    reply = call.say("When is my wife's appointment?")

    assert reply.text == (
        "I'm sorry, I can only share appointment details with the patient, "
        "once I've confirmed who they are."
    )
    assert call.record()["verified"] is False


def test_with_the_model_down_a_yes_to_the_offer_transfers(start_call):
    call = start_call(model_available=False)
    offer = call.say("When is my next appointment?")
    reply = call.say("Yes.")

    assert offer.text == f"{MODEL_TROUBLE} {TRANSFER_OFFER}"
    assert reply.text == TRANSFER_LINE
    assert [e["reason"] for e in call.events("transfer")] == ["model_unavailable"]
