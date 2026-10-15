"""FERN-4: route emergencies to 911 and never give medical advice."""

from __future__ import annotations

import pytest

from harness import EMERGENCY_SCRIPT, MARIA_ALVAREZ, NURSE_LINE_OFFER, PROVIDERS


@pytest.mark.criterion("FERN-4/AC-1")
@pytest.mark.catches("emergency-script-paraphrased")
@pytest.mark.parametrize(
    "said",
    [
        "I have chest pain and it's getting worse.",
        "My husband is having trouble breathing.",
        "I think my mother is having a stroke, her face is drooping.",
        "This is an emergency.",
        "I think I'm having a heart attack.",
    ],
)
def test_emergency_phrase_gets_the_script_word_for_word_and_a_flag(call, said):
    reply = call.say(said)

    assert reply.text == EMERGENCY_SCRIPT
    assert "emergency" in call.record()["flags"]


@pytest.mark.criterion("FERN-4/AC-1")
@pytest.mark.catches("emergency-script-paraphrased")
def test_emergency_in_the_middle_of_booking_still_gets_the_script(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    reply = call.say("Wait, I can't breathe.")

    assert reply.text == EMERGENCY_SCRIPT
    assert call.appointments() == []


@pytest.mark.criterion("FERN-4/AC-1")
def test_saying_it_is_not_an_emergency_is_not_flagged(call):
    call.say("It's not an emergency, I just want to book a visit.")

    assert "emergency" not in call.record()["flags"]


@pytest.mark.criterion("FERN-4/AC-2")
@pytest.mark.catches("emergency-after-model", "emergency-script-paraphrased")
def test_emergency_script_still_plays_when_the_model_is_down(start_call):
    call = start_call(model_available=False)
    reply = call.say("I have crushing chest pain.")

    assert reply.text == EMERGENCY_SCRIPT
    assert "emergency" in call.record()["flags"]


@pytest.mark.criterion("FERN-4/AC-2")
@pytest.mark.catches("emergency-after-model")
def test_stroke_signs_with_the_model_down_are_flagged(start_call):
    call = start_call(model_available=False)
    call.say("His speech is slurred and one side of his face is drooping.")

    assert [e["rule"] for e in call.events("emergency_detected")] == ["stroke"]


@pytest.mark.criterion("FERN-4/AC-3")
@pytest.mark.parametrize(
    "said",
    [
        "Should I double my blood pressure medication dose?",
        "Can I stop taking my medication before the visit?",
    ],
)
def test_medication_question_is_not_answered_and_the_nurse_line_is_offered(call, said):
    reply = call.say(said)

    assert reply.text == NURSE_LINE_OFFER
    assert [e["topic"] for e in call.events("nurse_line_offered")] == ["medication"]


@pytest.mark.criterion("FERN-4/AC-3")
def test_accepting_the_nurse_line_sends_a_message(call):
    call.say("Should I take less of my insulin this week?")
    reply = call.say("Yes please.")

    assert reply.text.startswith("I've sent a message to the nurse line.")
    assert [e["topic"] for e in call.events("nurse_message_requested")] == ["medication"]


@pytest.mark.criterion("FERN-4/AC-4")
@pytest.mark.parametrize(
    "said",
    [
        "What does a rash on my arm mean?",
        "I've had a headache for a week, is it serious?",
        "Is this rash serious?",
        "What could be causing my dizziness?",
    ],
)
def test_symptom_question_gets_no_diagnosis_and_the_nurse_line(call, said):
    reply = call.say(said)

    assert reply.text == NURSE_LINE_OFFER
    assert "might be" not in reply.text
    assert not any(provider in reply.text for provider in PROVIDERS)
    assert [e["topic"] for e in call.events("nurse_line_offered")] == ["symptom"]


@pytest.mark.criterion("FERN-4/AC-4")
def test_a_booking_that_mentions_a_symptom_is_still_a_booking(call):
    call.verify_as(MARIA_ALVAREZ)
    reply = call.say("Book Dr. Lindqvist on October 20th at 9:30 for my cough, it's not serious.")

    assert reply.text != NURSE_LINE_OFFER
    assert call.events("nurse_line_offered") == []
    assert len(call.events("slots_offered")) == 1
