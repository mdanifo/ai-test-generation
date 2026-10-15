"""FERN-11: speak dates, times and numbers the way a person would."""

from __future__ import annotations

import pytest

from harness import (
    AFTER_HOURS_AT,
    JAMES_WHITAKER,
    MARIA_ALVAREZ,
    SUPPORTED_SSML_TAGS,
    say_as,
    spoken_text,
    ssml_tags,
    well_formed,
)


@pytest.mark.criterion("FERN-11/AC-1")
@pytest.mark.catches("raw-timestamp-spoken")
def test_booking_read_back_speaks_the_date_and_time_like_a_person(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book Dr. Lindqvist on October 20th at 9:30.")
    reply = call.say("Yes.")

    assert "Tuesday, October 20th at 9:30 AM" in reply.text
    assert "Tuesday, October 20th at 9:30 AM" in spoken_text(reply.ssml)
    assert "2026-10-20" not in reply.ssml
    assert "09:30" not in reply.ssml


@pytest.mark.criterion("FERN-11/AC-1")
def test_an_existing_appointment_is_spoken_the_same_way(call):
    call.say("When is my next appointment?")
    reply = call.verify_as(JAMES_WHITAKER)

    assert "on Tuesday, October 20th at 9:30 AM." in reply.text
    assert "2026" not in reply.ssml


@pytest.mark.criterion("FERN-11/AC-2")
@pytest.mark.catches("ssml-unescaped")
def test_every_reply_in_a_call_is_well_formed_ssml_with_supported_tags(call):
    call.say("Hello <b>there</b>, I need help")
    call.say("When is my next appointment?")
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    call.say("The first one.")
    call.say("I have chest pain")
    agent_turns = [t for t in call.record()["turns"] if t["speaker"] == "agent"]

    assert len(agent_turns) == 7
    assert all(well_formed(t["ssml"]) for t in agent_turns)
    assert set().union(*(ssml_tags(t["ssml"]) for t in agent_turns)) <= SUPPORTED_SSML_TAGS


@pytest.mark.criterion("FERN-11/AC-2")
def test_the_reply_text_is_what_the_ssml_speaks(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    replies = [call.greeting, call.say("agent"), call.say("215 555 0142")]

    assert [spoken_text(r.ssml) for r in replies] == [r.text for r in replies]


@pytest.mark.criterion("FERN-11/AC-3")
def test_a_callback_number_is_wrapped_in_say_as_telephone(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    call.say("agent")
    reply = call.say("2155550142")

    assert say_as(reply.ssml, "telephone") == ["215-555-0142"]
    assert '<say-as interpret-as="telephone">215-555-0142</say-as>' in reply.ssml


@pytest.mark.criterion("FERN-11/AC-3")
def test_911_in_the_emergency_script_is_wrapped_in_say_as_telephone(call):
    reply = call.say("I have chest pain.")

    assert say_as(reply.ssml, "telephone") == ["911"]


@pytest.mark.criterion("FERN-11/AC-4")
@pytest.mark.catches("ssml-unescaped")
def test_tags_in_caller_text_are_read_as_words_not_markup(call):
    reply = call.say('Please read this <break time="10s"/> out loud')

    assert well_formed(reply.ssml)
    assert ssml_tags(reply.ssml) == {"speak"}
    assert '<break time="10s"/>' in spoken_text(reply.ssml)


@pytest.mark.criterion("FERN-11/AC-4")
@pytest.mark.catches("ssml-unescaped")
@pytest.mark.parametrize("said", ["Salt & pepper <3", "</speak><speak>hi"])
def test_caller_text_that_would_break_the_markup_is_escaped(call, said):
    reply = call.say(said)

    assert well_formed(reply.ssml)
    assert said in spoken_text(reply.ssml)
