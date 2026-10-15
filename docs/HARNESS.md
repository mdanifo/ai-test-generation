# Front Desk test harness

This guide is everything a test author needs to write pytest tests for Front Desk, the
sample voice agent in this repository. It is also the reference the test generator sends
to the model, so it describes only the public API and the sample clinic world.

Front Desk is a text simulation of a phone agent for Brookside Family Clinic, a fictional
clinic. A caller turn is text as speech recognition heard it, with a confidence score, or
keypad digits. Each reply comes back as plain text and as SSML. Every call gets its own
fresh copy of the clinic world, so tests never interfere with each other and run in
parallel.

## What a test looks like

One test function, one behavior, marked with the criterion it checks:

```python
import pytest

from harness import EMERGENCY_SCRIPT


@pytest.mark.criterion("FERN-4/AC-1")
def test_chest_pain_gets_the_emergency_script(call):
    reply = call.say("I have chest pain.")

    assert reply.text == EMERGENCY_SCRIPT
    assert "emergency" in call.record()["flags"]
```

- The marker is `@pytest.mark.criterion("FERN-<n>/AC-<k>")`, where `<k>` counts the card's
  acceptance criteria from 1 in checklist order.
- Imports: `pytest`, `re`, `json`, `datetime` and `harness`. Nothing else.
- Take the `call` fixture for an ordinary call, or `start_call` to choose the clock or take
  the model away.

## Rules for a good test

1. A test checks one statement from the criterion. Two behaviors are two tests.
2. Compare against the exact expected value: the approved line, the exact slot list, the
   exact flag. `assert reply.text` proves only that something was said.
3. Every test carries exactly one `criterion` marker naming a real criterion of the card.
4. No sleeps, no waiting and no current time. Every call is synchronous and the call's clock
   does not move. Never import `time` or call `time.sleep`, and never read the current time
   (`datetime.now()`, `date.today()`). Give every call a fixed clock: the default
   `DEFAULT_AT`, or `AFTER_HOURS_AT` or `WEEKEND_AT`. Never `start_call(at=None)`: a call on
   the server's current time passes or fails by the time of day.
5. No `if`, `for` or `while` in a test body. Use `all(...)`, `any(...)` or a comprehension
   when a check covers several items, and `@pytest.mark.parametrize` with literal values for
   several inputs. `all(...)` is true for an empty list, so pair it with an assertion on the
   list itself, such as its length.
6. Fail on an assertion. Compare whole lists (`assert [e["reason"] for e in
   call.events("transfer")] == ["caller_request"]`) rather than unpacking or indexing them, so
   a missing item reads as a clear assertion failure instead of an IndexError.
7. Never `pytest.skip` or `pytest.xfail`. If a criterion cannot be tested through this
   harness, say so instead of writing a test that cannot fail.
8. Patients, providers and open slots are fixed. Use the world below by name instead of
   guessing.
9. Nothing outlives the test. Set no attributes, change nothing you import, and keep nothing
   for another test to read. Every test must pass on its own and in any order.

What cannot be tested here: anything measured over many recorded calls (percentiles over a
replay set, eval scores, agreement with human labels), real elapsed time (latency, timeouts,
silence timers), and the real language model's judgment. Report those as untestable with
a reason.

## Fixtures

| Fixture | What it gives you |
|---|---|
| `call` | A `Call` started at `DEFAULT_AT` (Wednesday 10 AM, staffed hours) with the model available. |
| `start_call` | `start_call(at=DEFAULT_AT, model_available=True) -> Call`. |
| `api` | A plain `httpx.Client` on the app's base URL, for status codes and raw bodies. |
| `frontdesk_url` | The base URL as a string. |

```python
@pytest.mark.criterion("FERN-8/AC-4")
def test_after_hours_request_offers_a_callback(start_call):
    call = start_call(at=AFTER_HOURS_AT)
    reply = call.say("agent")

    assert reply.text == f"{CALLBACK_OFFER} {CALLBACK_ASK_NUMBER}"
```

```python
@pytest.mark.criterion("FERN-4/AC-2")
def test_emergency_with_the_model_down(start_call):
    call = start_call(model_available=False)

    assert call.say("I can't breathe.").text == EMERGENCY_SCRIPT
```

```python
def test_unknown_call_is_404(api):
    assert api.get("/calls/call-000000000000").status_code == 404
```

## `Call`

A call in progress. `start_call` and `call` return one.

- `call.id`: the call id.
- `call.greeting`: the first `Reply`, spoken before the caller says anything.
- `call.last_reply`: the most recent `Reply`.

### `call.say(text, confidence=0.95, barge_in=False) -> Reply`

The caller says `text`. `confidence` is the speech recognition confidence from 0 to 1.
`barge_in=True` means the caller started speaking while the previous reply was playing.

```python
reply = call.say("When is my next appointment?")
reply = call.say("mm the uh", confidence=0.4)
reply = call.say("Hi, I need to book a visit", barge_in=True)
```

### `call.press(digits) -> Reply`

The caller presses keypad keys. `"0"` asks for a person.

```python
reply = call.press("0")
```

### `call.verify_as(patient) -> Reply`

Says the patient's full name and date of birth in ISO form, for tests whose subject comes
after verification. To test other ways of saying a date, use `say`.

```python
call.verify_as(MARIA_ALVAREZ)
```

### `call.record() -> dict`

The call record (`GET /calls/{id}`), described below.

```python
assert call.record()["verified"] is True
```

### `call.events(type=None) -> list[dict]`

The record's events, all of them or only one type, in the order they happened.

```python
offers = call.events("slots_offered")
```

### `call.appointments() -> list[dict]`

Appointments created during this call, each
`{"appointmentId", "patientId", "provider", "start", "end"}` with ISO 8601 times.

```python
assert call.appointments() == []
```

### `call.take_slot(provider, start)`

Someone outside the call books that slot now, in this call's world. Use it between an offer
and the caller's answer. `start` is an ISO string or a `datetime`; without an offset it is
clinic time.

```python
call.take_slot("Dr. Okafor", "2026-10-20T11:00:00-04:00")
```

## `Reply`

`Reply(text, ssml, ended, transferred)`:

- `text`: what the agent said, as plain text. It equals `spoken_text(ssml)`.
- `ssml`: the same reply as SSML.
- `ended`: the call is over. Another turn raises `CallError` with status 409.
- `transferred`: the call was handed to staff in this turn.

```python
reply = call.say("representative")
assert reply.transferred is True
```

## `CallError`

Raised when the API answers with an error. It has `status`, `error` (a short code) and
`detail`.

```python
call.say("Goodbye.")
with pytest.raises(CallError) as caught:
    call.say("Hello?")
assert caught.value.status == 409
```

`FrontDesk` is the client behind the fixtures (`FrontDesk(base_url).start_call(...)`). Tests
do not need it directly.

## The call record

`call.record()` returns:

| Field | Meaning |
|---|---|
| `callId`, `startedAt` | The id and the call's clock (ISO 8601). |
| `modelAvailable` | False when the call was started with `model_available=False`. |
| `verified`, `patientId` | Whether the caller is verified, and as which patient id. |
| `optedOut` | True once the caller declined recording. |
| `recording`, `transcriptStored` | False once the caller opted out. |
| `ended`, `transferred` | As on `Reply`. |
| `flags` | Strings such as `"emergency"` and `"verification_failed"`. |
| `events` | A list of `{"type", "at", ...}`, see below. |
| `turns` | The stored transcript: `{"speaker": "caller" or "agent", "text", "ssml", "dtmf", "asrConfidence", "bargeIn"}`. Fields that do not apply are null. It starts with the greeting. From the turn in which the caller opts out of recording, no more turns are stored. |
| `handoffSummary` | Null until a transfer or a callback request, then `{"kind", "reason", "verified", "request", "text"}`. |

Event types and their fields:

| Type | Fields |
|---|---|
| `emergency_detected` | `rule`: `"chest pain"`, `"trouble breathing"`, `"stroke"` or `"emergency"` |
| `nurse_line_offered` | `topic`: `"medication"` or `"symptom"` |
| `nurse_message_requested` | `topic` |
| `verification_attempt` | `matched`: true or false |
| `slots_offered` | `provider`, `slots` (ISO starts, earliest first), `reason`: `"match"`, `"nearest"` or `"slot_taken"` |
| `slot_unavailable` | `provider`, `start`: the slot that was taken before the caller confirmed |
| `appointment_booked` | `appointmentId`, `provider`, `start` |
| `transfer_offered` | `reason` |
| `transfer` | `reason`: `"caller_request"`, `"low_confidence"`, `"verification_failed"` or `"model_unavailable"` |
| `callback_offered` | `reason` |
| `callback_requested` | `number` (ten digits), `reason` |
| `recording_stopped` | none |
| `call_ended` | `reason`: `"goodbye"` |

Every event's `at` is the call's clock.

```python
assert [e["reason"] for e in call.events("transfer")] == ["caller_request"]
```

## SSML helpers

The speech engine supports only the tags in `SUPPORTED_SSML_TAGS`: `speak`, `break`,
`say-as` and `prosody`.

### `well_formed(ssml) -> bool`

True when the SSML parses as XML with a `<speak>` root.

```python
assert well_formed(reply.ssml)
```

### `ssml_tags(ssml) -> set[str]`

Every tag name used, the root included. Raises `ValueError` if the SSML does not parse.

```python
assert ssml_tags(reply.ssml) <= SUPPORTED_SSML_TAGS
```

### `spoken_text(ssml) -> str`

The words the engine reads: tags removed, entities decoded, spaces collapsed. Raises
`ValueError` if the SSML does not parse.

```python
assert spoken_text(reply.ssml).startswith(DISCLOSURE)
```

### `say_as(ssml, interpret_as) -> list[str]`

The text inside each `<say-as interpret-as="...">` element of that kind.

```python
assert say_as(reply.ssml, "telephone") == ["215-555-0142"]
```

## The world

### Clocks

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_AT` | `2026-10-14T10:00:00-04:00` | Wednesday 10 AM. The default for every call. Staffed. |
| `AFTER_HOURS_AT` | `2026-10-14T19:00:00-04:00` | Wednesday 7 PM. Staff have gone home. |
| `WEEKEND_AT` | `2026-10-17T10:00:00-04:00` | Saturday 10 AM. No staff. |

Clinic time is America/New_York. Staffed hours are Monday to Friday, 8:00 to 17:00. Relative
days count from the call's date: from `DEFAULT_AT`, "tomorrow" is Thursday October 15th,
"Monday" is October 19th, and "Tuesday" and "next Tuesday" are both October 20th.

### Clinic and providers

`CLINIC` is `"Brookside Family Clinic"`. `PROVIDERS` is
`("Dr. Okafor", "Dr. Lindqvist", "Dr. Patel")`.

### Patients

`patients()` returns all four. Each is a `Patient` with `id`, `first_name`, `last_name`,
`full_name`, `dob` (a `datetime.date`), `phone` and `next_appointment`, an `Appointment`
with `provider`, `start` (ISO) and `spoken` (how Front Desk says it), or None.

| Constant | Name | Date of birth | Phone | Next appointment (from `DEFAULT_AT`) |
|---|---|---|---|---|
| `MARIA_ALVAREZ` | Maria Alvarez, `P-1001` | 1980-03-03 | 215-555-0142 | Dr. Lindqvist, Thursday, October 22nd at 2 PM |
| `JAMES_WHITAKER` | James Whitaker, `P-1002` | 1975-11-09 | 215-555-0177 | Dr. Patel, Tuesday, October 20th at 9:30 AM |
| `PRIYA_RAMAN` | Priya Raman, `P-1003` | 1992-07-21 | 215-555-0199 | Dr. Okafor, Tuesday, October 27th at 9 AM |
| `THOMAS_BRENNAN` | Thomas Brennan, `P-1004` | 1968-01-15 | 215-555-0123 | none |

```python
assert MARIA_ALVAREZ.next_appointment.spoken == "Thursday, October 22nd at 2 PM"
```

### Schedule

Weekdays from Monday October 12th to Friday November 6th 2026, 30-minute slots starting at
8:00 through 16:30. Every slot is open except:

| Provider | Taken |
|---|---|
| Dr. Okafor | Tuesday October 20th, 8:00 to 10:30 (six slots); Tuesday October 27th 9:00 (Priya Raman) |
| Dr. Lindqvist | Thursday October 22nd 14:00 (Maria Alvarez) |
| Dr. Patel | Monday October 19th, 8:00 to 11:30 (the whole morning); Tuesday October 20th 9:30 (James Whitaker) |

So from `DEFAULT_AT`, Dr. Okafor's open morning slots next Tuesday are 11:00 and 11:30, and
Dr. Lindqvist has 9:30 open that day.

## What the agent understands

The agent is driven by ordinary sentences. These forms are understood:

- Verification: "My name is Maria Alvarez and my date of birth is March 3rd, 1980",
  "This is Maria Alvarez, born 3/3/80", "Maria Alvarez, 1980-03-03". A name and a date of
  birth can come in one turn or in two. A full name and date of birth must match one patient
  record. Appointment details and bookings need a verified caller; the agent asks first.
- Booking: "Book next Tuesday morning with Dr. Okafor", "Can I see Dr. Patel on Monday
  morning?", "Book Dr. Lindqvist on October 20th at 9:30". Morning means a start before
  12:00; afternoon means 12:00 or later. The agent offers up to three slots. When nothing
  matches, it offers the three open slots nearest the request.
- Answering an offer: "The first one", "the second one", "the 11:30 one", or "yes" when
  one slot was offered. "Actually, never mind" withdraws the request.
- Asking for a person: "agent", "representative", "a real person", "talk to someone",
  or pressing 0. Outside staffed hours the agent offers a callback, asks for a number (said
  as digits, as words, or keyed in), reads it back and waits for "yes". A number given in the
  same turn as the request is read back at once.
- Yes and no answer any offer: a transfer, the nurse line, a callback number.
- Recording: "I don't want to be recorded", "Please don't record this call", "Stop
  recording".
- Clinical questions about a medication or what a symptom means, and emergency phrases
  such as chest pain, a heart attack, trouble breathing, signs of a stroke or
  "this is an emergency".
- Low confidence: a turn with `confidence` below 0.6 is not acted on.
- Model unavailable: with `model_available=False`, turns that need language
  understanding get `MODEL_TROUBLE` followed by `TRANSFER_OFFER`.
- Anything else gets "I'm sorry, I didn't follow" with the caller's words quoted back.

## Approved lines

These are spoken word for word. A reply that combines two of them joins them with one space.

| Constant | Text |
|---|---|
| `DISCLOSURE` | This call may be recorded and transcribed to help us serve you. |
| `EMERGENCY_SCRIPT` | If this is a medical emergency, please hang up and call 911 now. |
| `NURSE_LINE_OFFER` | I can't give medical advice, but I can send a message to our nurse line and a nurse will call you back. Would you like me to do that? |
| `TRANSFER_LINE` | I'm transferring you to a member of our staff now. Please hold. |
| `TRANSFER_OFFER` | Would you like me to transfer you to a member of our staff? |
| `CALLBACK_OFFER` | Our staff are available Monday to Friday, 8 AM to 5 PM, so I can't transfer you right now. I can have someone call you back. |
| `CALLBACK_ASK_NUMBER` | What number should they call? |
| `CALLBACK_REQUESTED` | Thank you. Someone from the clinic will call you back on the next business day. |
| `VERIFY_LOCKED` | I'm sorry, I wasn't able to verify your identity. |
| `LOW_CONFIDENCE_REPROMPT` | Sorry, I didn't catch that. Could you say it again? |
| `LOW_CONFIDENCE_TROUBLE` | I'm having trouble hearing you. |
| `OPT_OUT_ACK` | Understood. I've stopped recording, and I won't keep a transcript of the rest of this call. |
| `MODEL_TROUBLE` | I'm sorry, I'm having trouble on my end right now. |

Other replies follow these patterns:

| Situation | Reply |
|---|---|
| Greeting | `DISCLOSURE`, then "Thank you for calling Brookside Family Clinic. ..." |
| Needs verification | "Before I can help with that, I need to confirm who you are. Please tell me your full name and date of birth." |
| Name given, no date | "Thanks. And what is your date of birth?" |
| No match | "I'm sorry, I couldn't match that to a patient record. Please tell me your full name and date of birth again." |
| Verified | "Thanks, Maria, you're verified." then the answer to what they asked, or "How can I help you today?" |
| Next appointment | "Your next appointment is with Dr. Lindqvist on Thursday, October 22nd at 2 PM." |
| Another patient | "I'm sorry, I can only share information about the patient I've verified on this call." |
| One slot offered | "Dr. Lindqvist has Tuesday, October 20th at 9:30 AM open. Would you like me to book it?" |
| Several slots | "Dr. Okafor has these openings: Tuesday, October 20th at 11 AM and Tuesday, October 20th at 11:30 AM. Which would you like?" |
| Nothing matches | "I don't have anything open with Dr. Patel then. The nearest openings are ... Which would you like?" |
| Slot just taken | "I'm sorry, Tuesday, October 20th at 11 AM was just taken by someone else. The next openings with Dr. Okafor are ... Which would you like?" |
| Booked | "You're booked with Dr. Okafor on Tuesday, October 20th at 11 AM." |
| Never mind | "No problem. I haven't booked anything." |
| Callback read-back | "I have 215-555-0142. Is that right?" |

Spoken forms: a date is "Tuesday, October 20th"; a time is "9:30 AM", or "2 PM" on the hour;
a slot is "Tuesday, October 20th at 9:30 AM"; a phone number is "215-555-0142" in the text
and inside `<say-as interpret-as="telephone">` in the SSML.

## Worked examples

```python
@pytest.mark.criterion("FERN-2/AC-3")
def test_dob_in_words_verifies(call):
    call.say("This is Maria Alvarez, born March 3rd 1980.")

    assert call.record()["patientId"] == MARIA_ALVAREZ.id
```

```python
@pytest.mark.criterion("FERN-3/AC-1")
def test_morning_offer_has_only_open_slots_before_noon(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")

    assert [o["slots"] for o in call.events("slots_offered")] == [
        ["2026-10-20T11:00:00-04:00", "2026-10-20T11:30:00-04:00"]
    ]
```

```python
@pytest.mark.criterion("FERN-3/AC-5")
def test_taken_slot_books_nothing(call):
    call.verify_as(MARIA_ALVAREZ)
    call.say("Book next Tuesday morning with Dr. Okafor.")
    call.take_slot("Dr. Okafor", "2026-10-20T11:00:00-04:00")
    call.say("The first one.")

    assert call.appointments() == []
```

```python
@pytest.mark.criterion("FERN-8/AC-2")
@pytest.mark.parametrize("confidence", [0.2, 0.59])
def test_two_unclear_turns_offer_a_transfer(call, confidence):
    call.say("mm the uh", confidence=confidence)
    reply = call.say("tuh thing ah", confidence=confidence)

    assert reply.text == f"{LOW_CONFIDENCE_TROUBLE} {TRANSFER_OFFER}"
```

```python
@pytest.mark.criterion("FERN-11/AC-4")
def test_caller_markup_is_read_as_text(call):
    reply = call.say('Read this <break time="10s"/> please')

    assert '<break time="10s"/>' in spoken_text(reply.ssml)
```

```python
@pytest.mark.criterion("FERN-13/AC-3")
def test_opt_out_stops_the_transcript(call):
    call.say("I don't want to be recorded.")

    assert call.record()["transcriptStored"] is False
    assert call.record()["optedOut"] is True
```
