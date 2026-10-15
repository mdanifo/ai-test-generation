"""FERN-3: book an appointment from a spoken request.

The call clock is Wednesday 2026-10-14 10 AM, so "next Tuesday" is 2026-10-20
and "Monday" is 2026-10-19. Dr. Okafor's Tuesday is booked from 8 to 10:30 by
other patients; Dr. Patel's Monday morning is fully booked.
"""

from __future__ import annotations

import pytest

from harness import MARIA_ALVAREZ

OKAFOR_TUESDAY_MORNING = ["2026-10-20T11:00:00-04:00", "2026-10-20T11:30:00-04:00"]


@pytest.mark.criterion("FERN-3/AC-1")
@pytest.mark.catches("morning-includes-noon")
def test_next_tuesday_morning_offers_only_open_okafor_slots_before_noon(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("I'd like next Tuesday morning with Dr. Okafor.")

    offers = call.events("slots_offered")
    assert [(o["provider"], o["slots"]) for o in offers] == [("Dr. Okafor", OKAFOR_TUESDAY_MORNING)]


@pytest.mark.criterion("FERN-3/AC-1")
@pytest.mark.catches("morning-includes-noon")
def test_the_morning_offer_is_spoken_without_the_noon_slot(call):
    call.verify_as(MARIA_ALVAREZ)
    reply = call.say("Can I book next Tuesday morning with Dr. Okafor?")

    assert reply.text == (
        "Dr. Okafor has these openings: Tuesday, October 20th at 11 AM and "
        "Tuesday, October 20th at 11:30 AM. Which would you like?"
    )


@pytest.mark.criterion("FERN-3/AC-2")
def test_accepting_a_slot_creates_one_appointment_and_reads_it_back(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.say("The 11:30 one, please.")

    assert reply.text.startswith(
        "You're booked with Dr. Okafor on Tuesday, October 20th at 11:30 AM."
    )
    appointments = call.appointments()
    assert [(a["provider"], a["start"], a["patientId"]) for a in appointments] == [
        ("Dr. Okafor", "2026-10-20T11:30:00-04:00", MARIA_ALVAREZ.id)
    ]


@pytest.mark.criterion("FERN-3/AC-2")
@pytest.mark.catches("double-booking")
def test_confirming_twice_still_creates_exactly_one_appointment(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    call.say("The first one.")
    call.say("Yes, please book it.")

    assert len(call.appointments()) == 1
    assert len(call.events("appointment_booked")) == 1


@pytest.mark.criterion("FERN-3/AC-2")
@pytest.mark.catches("double-booking")
def test_a_repeated_yes_is_told_it_is_already_booked(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book Dr. Lindqvist on October 20th at 9:30.")
    call.say("Yes.")
    reply = call.say("Yes.")

    assert (
        reply.text
        == "You're already booked with Dr. Lindqvist on Tuesday, October 20th at 9:30 AM."
    )
    assert len(call.appointments()) == 1


@pytest.mark.criterion("FERN-3/AC-3")
@pytest.mark.catches("morning-includes-noon")
def test_no_matching_slot_offers_the_three_nearest_and_books_nothing(call):
    call.verify_as(MARIA_ALVAREZ)
    reply = call.say("Can I see Dr. Patel on Monday morning?")

    assert [o["slots"] for o in call.events("slots_offered")] == [
        [
            "2026-10-19T12:00:00-04:00",
            "2026-10-19T12:30:00-04:00",
            "2026-10-19T13:00:00-04:00",
        ]
    ]
    assert reply.text.startswith("I don't have anything open with Dr. Patel then.")
    assert call.appointments() == []


@pytest.mark.criterion("FERN-3/AC-3")
def test_picking_one_of_the_nearest_slots_books_it(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Can I see Dr. Patel on Monday morning?")
    call.say("Yes.")
    assert call.appointments() == []

    call.say("The second one.")

    assert [a["start"] for a in call.appointments()] == ["2026-10-19T12:30:00-04:00"]


@pytest.mark.criterion("FERN-3/AC-4")
def test_never_mind_before_confirming_books_nothing(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.say("Actually, never mind.")

    assert reply.text == "No problem. I haven't booked anything."
    assert call.appointments() == []


@pytest.mark.criterion("FERN-3/AC-4")
def test_a_yes_after_never_mind_still_books_nothing(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book Dr. Lindqvist on October 20th at 9:30.")
    call.say("Actually, never mind.")
    call.say("Yes.")

    assert call.appointments() == []
    assert call.events("appointment_booked") == []


@pytest.mark.criterion("FERN-3/AC-5")
def test_slot_taken_between_offer_and_confirmation_books_nothing_and_offers_next(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    call.take_slot("Dr. Okafor", "2026-10-20T11:00:00-04:00")
    reply = call.say("The first one.")

    assert reply.text.startswith(
        "I'm sorry, Tuesday, October 20th at 11 AM was just taken by someone else."
    )
    assert call.appointments() == []
    assert call.events("slots_offered")[-1]["slots"] == [
        "2026-10-20T11:30:00-04:00",
        "2026-10-20T12:00:00-04:00",
        "2026-10-20T12:30:00-04:00",
    ]


@pytest.mark.criterion("FERN-3/AC-5")
def test_after_a_taken_slot_the_caller_can_book_a_new_one(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book Dr. Lindqvist on October 20th at 9:30.")
    call.take_slot("Dr. Lindqvist", "2026-10-20T09:30:00-04:00")
    call.say("Yes.")
    call.say("The first one.")

    assert [a["start"] for a in call.appointments()] == ["2026-10-20T10:00:00-04:00"]
