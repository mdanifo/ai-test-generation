"""The dialog manager: one ``Call`` per phone call.

Order of work on every caller turn:

1. the emergency rule (FERN-4), before anything else and before the model;
2. the recording opt-out (FERN-13);
3. keypad digits, speech recognition confidence and requests for a person (FERN-8);
4. answers to an open offer (transfer, nurse line, callback number);
5. the medical-advice guard (FERN-4);
6. the model: understanding the turn, then the verification and schedule tools.

The call's clock is the ``at`` it was started with. It does not move during the
call, so every event carries that time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from frontdesk import bugs, parsing, rules, scripts
from frontdesk.model import Context, Interpretation, Model, ModelUnavailable
from frontdesk.speech import (
    Pause,
    Phone,
    Reply,
    Unit,
    lines,
    render,
    sentence,
    spoken_list,
    spoken_slot,
)
from frontdesk.world import CLINIC_TZ, SLOT, Appointment, Patient, World, is_staffed

LOW_CONFIDENCE = 0.6
MAX_VERIFY_ATTEMPTS = 3
MAX_OFFERED = 3
NOON = time(12, 0)
OPEN = time(8, 0)
CLOSE = time(17, 0)

REASON_TEXT = {
    "caller_request": "the caller asked for a person",
    "low_confidence": "speech recognition could not understand the caller",
    "verification_failed": "the caller could not be verified",
    "model_unavailable": "the assistant was unavailable",
}
REQUEST_TEXT = {
    "appointment_info": "their next appointment",
    "booking": "to book an appointment",
    "other_patient": "information about another patient",
    "medication": "advice about a medication",
    "symptom": "what a symptom means",
    "person": "to speak to a person",
}


class CallEnded(Exception):
    """The call has ended; no more turns."""


@dataclass
class Response:
    units: list[Unit]
    exact: bool = False  # an approved script that must be spoken alone, word for word


@dataclass
class BookingRequest:
    provider_id: str | None = None
    day: date | None = None
    part: str | None = None
    time: time | None = None


@dataclass
class Offer:
    provider_id: str
    slots: list[datetime]
    chosen: datetime | None = None


@dataclass
class TurnResult:
    reply: Reply
    ended: bool
    transferred: bool


@dataclass
class Call:
    call_id: str
    started_at: datetime
    model: Model
    model_available: bool
    world: World = field(default_factory=World.fresh)

    patient: Patient | None = None
    verify_failures: int = 0
    verify_locked: bool = False
    partial_name: str | None = None
    partial_dob: str | None = None
    after_verify: str | None = None

    booking: BookingRequest | None = None
    offer: Offer | None = None
    last_booking: Appointment | None = None
    created: list[Appointment] = field(default_factory=list)

    awaiting: str | None = None
    offer_reason: str | None = None
    nurse_topic: str | None = None
    callback_reason: str | None = None
    callback_number: str | None = None
    asked_why: bool = False

    caller_turns: int = 0
    low_turns: int = 0
    disclosure_owed: bool = False
    opted_out: bool = False
    ended: bool = False
    transferred: bool = False

    flags: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    handoff_summary: dict[str, Any] | None = None

    @property
    def verified(self) -> bool:
        return self.patient is not None

    @property
    def today(self) -> date:
        return self.started_at.astimezone(CLINIC_TZ).date()

    def greet(self) -> Reply:
        reply = render([sentence(scripts.DISCLOSURE), Pause(300), sentence(scripts.GREETING)])
        self._store_agent(reply)
        return reply

    def turn(
        self, text: str | None, dtmf: str | None, confidence: float, barge_in: bool
    ) -> TurnResult:
        if self.ended:
            raise CallEnded(self.call_id)
        self.caller_turns += 1
        # A barge-in on the first turn means the caller spoke over the greeting,
        # so the disclosure may not have been heard in full.
        if (
            self.caller_turns == 1
            and barge_in
            and not bugs.active("disclosure-skipped-on-barge-in")
        ):
            self.disclosure_owed = True
        caller = {
            "speaker": "caller",
            "text": text or "",
            "dtmf": dtmf,
            "asrConfidence": confidence if text else None,
            "bargeIn": barge_in,
        }
        response = self._respond((text or "").strip(), dtmf, confidence)
        units = response.units
        if self.disclosure_owed and not response.exact:
            units = [sentence(scripts.DISCLOSURE), Pause(300), *units]
            self.disclosure_owed = False
        reply = render(units)
        if not self.opted_out:
            self.turns.append(caller)
            self._store_agent(reply)
        return TurnResult(reply=reply, ended=self.ended, transferred=self.transferred)

    def take_slot(self, provider_id: str, start: datetime) -> None:
        self.world.take(provider_id, start)

    def record(self) -> dict[str, Any]:
        return {
            "callId": self.call_id,
            "startedAt": self.started_at.isoformat(),
            "modelAvailable": self.model_available,
            "verified": self.verified,
            "patientId": self.patient.id if self.patient else None,
            "optedOut": self.opted_out,
            "recording": not self.opted_out,
            "transcriptStored": not self.opted_out,
            "ended": self.ended,
            "transferred": self.transferred,
            "flags": list(self.flags),
            "events": [dict(e) for e in self.events],
            "turns": [dict(t) for t in self.turns],
            "handoffSummary": dict(self.handoff_summary) if self.handoff_summary else None,
        }

    def appointment_records(self) -> list[dict[str, Any]]:
        return [self._appointment_record(a) for a in self.created]

    def _store_agent(self, reply: Reply) -> None:
        self.turns.append({"speaker": "agent", "text": reply.text, "ssml": reply.ssml})

    def _event(self, kind: str, **data: Any) -> None:
        self.events.append({"type": kind, "at": self.started_at.isoformat(), **data})

    def _note(self, request: str) -> None:
        if request not in self.requests:
            self.requests.append(request)

    def _say(self, kind: str, **facts: object) -> list[Unit]:
        return self.model.write_reply(kind, facts)

    def _provider_name(self, provider_id: str) -> str:
        return self.world.provider(provider_id).name

    def _appointment_record(self, a: Appointment) -> dict[str, Any]:
        return {
            "appointmentId": a.id,
            "patientId": a.patient_id,
            "provider": self._provider_name(a.provider_id),
            "start": a.start.isoformat(),
            "end": a.end.isoformat(),
        }

    def _context(self) -> Context:
        return Context(
            today=self.today,
            awaiting=self.awaiting if self.awaiting in ("identity", "booking") else None,
            offer_size=len(self.offer.slots) if self.offer else 0,
            verified_name=self.patient.full_name if self.patient else None,
        )

    def _respond(self, text: str, dtmf: str | None, confidence: float) -> Response:
        low = bool(text) and not dtmf and confidence < LOW_CONFIDENCE
        if low:
            self.low_turns += 1
        elif not bugs.active("low-confidence-not-consecutive"):
            self.low_turns = 0

        emergency_first = not bugs.active("emergency-after-model")
        if text and emergency_first and (found := self._emergency(text)):
            return found
        if text and rules.wants_opt_out(text):
            return self._opt_out()
        if dtmf:
            return self._keys(dtmf)
        if low:
            return self._low_confidence()
        if rules.wants_person(text):
            self._note("person")
            return self._ask_for_person(parsing.find_phone_digits(text))
        if self.awaiting in ("callback_number", "callback_confirm"):
            answered = self._callback_answer(text)
            if answered:
                return answered
        if self.awaiting in ("transfer_offer", "nurse_offer"):
            answered = self._offer_answer(text)
            if answered:
                return answered
        topic = rules.medical_topic(text)
        if topic:
            return self._nurse_offer(topic)
        try:
            meaning = self.model.interpret(text, self._context())
            if not emergency_first and (found := self._emergency(text)):
                return found
            return Response(self._handle(meaning, text))
        except ModelUnavailable:
            return self._offer_transfer(
                "model_unavailable", [scripts.MODEL_TROUBLE, scripts.TRANSFER_OFFER]
            )

    # Rules that run before the model (FERN-4, FERN-13).

    def _emergency(self, text: str) -> Response | None:
        rule = rules.emergency_rule(text)
        if rule is None:
            return None
        if "emergency" not in self.flags:
            self.flags.append("emergency")
        self._event("emergency_detected", rule=rule)
        script = scripts.EMERGENCY
        if bugs.active("emergency-script-paraphrased"):
            script = "If this is an emergency, please call 911."
        before, after = script.split("911", 1)
        return Response([sentence(before, Phone("911"), after, rate="90%")], exact=True)

    def _opt_out(self) -> Response:
        self.opted_out = True
        self._event("recording_stopped")
        return Response(lines(scripts.OPT_OUT_ACK, scripts.HOW_CAN_I_HELP))

    def _nurse_offer(self, topic: str) -> Response:
        self._note(topic)
        self.awaiting = "nurse_offer"
        self.nurse_topic = topic
        self._event("nurse_line_offered", topic=topic)
        return Response(lines(scripts.NURSE_LINE_OFFER))

    # FERN-8: handoff to a person.

    def _keys(self, dtmf: str) -> Response:
        digits = dtmf.strip("*#")
        if self.awaiting in ("callback_number", "callback_confirm") and len(digits) > 1:
            return self._callback_number(digits)
        if digits == "0":
            self._note("person")
            return self._ask_for_person()
        return Response(lines(scripts.KEY_UNUSED))

    def _low_confidence(self) -> Response:
        if self.low_turns >= 2:
            return self._offer_transfer(
                "low_confidence", [scripts.LOW_CONFIDENCE_TROUBLE, scripts.TRANSFER_OFFER]
            )
        return Response(lines(scripts.LOW_CONFIDENCE_REPROMPT))

    def _ask_for_person(self, digits: str = "") -> Response:
        if bugs.active("transfer-asks-why") and not self.asked_why:
            self.asked_why = True
            return Response(lines("Can I ask what this is about?"))
        return self._transfer("caller_request", digits)

    def _offer_transfer(self, reason: str, texts: list[str]) -> Response:
        self.awaiting = "transfer_offer"
        self.offer_reason = reason
        self._event("transfer_offered", reason=reason)
        return Response(lines(*texts))

    def _transfer(self, reason: str, digits: str = "") -> Response:
        """Transfer now in staffed hours; otherwise offer a callback.

        ``digits`` is a number the caller gave in the same turn, read back at once.
        """
        self.awaiting = None
        if not is_staffed(self.started_at):
            self.callback_reason = reason
            self.awaiting = "callback_number"
            self._event("callback_offered", reason=reason)
            if len(digits) >= 10:
                readback = self._callback_number(digits)
                return Response([*lines(scripts.CALLBACK_OFFER), *readback.units])
            return Response(lines(scripts.CALLBACK_OFFER, scripts.CALLBACK_ASK_NUMBER))
        self.transferred = True
        self.ended = True
        self.handoff_summary = self._summary("transfer", reason)
        self._event("transfer", reason=reason)
        return Response(lines(scripts.TRANSFER))

    def _summary(self, kind: str, reason: str) -> dict[str, Any]:
        """What staff see. Reasons and requests only: no name, date of birth or number."""
        asked = [self._request_text(r) for r in self.requests if r != "person"]
        if not asked:
            asked = [REQUEST_TEXT["person"]] if "person" in self.requests else []
        request = "; ".join(asked) or "nothing stated"
        heading = "Transfer" if kind == "transfer" else "Callback request"
        verified = "yes" if self.verified else "no"
        return {
            "kind": kind,
            "reason": reason,
            "verified": self.verified,
            "request": request,
            "text": f"{heading}. Reason: {REASON_TEXT[reason]}. Verified: {verified}. "
            f"Asked for: {request}.",
        }

    def _request_text(self, request: str) -> str:
        text = REQUEST_TEXT[request]
        if request == "booking":
            provider = self.booking.provider_id if self.booking else None
            if provider is None and self.last_booking:
                provider = self.last_booking.provider_id
            if provider:
                text += f" with {self._provider_name(provider)}"
        return text

    def _callback_number(self, digits: str) -> Response:
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            self.awaiting = "callback_number"
            return Response(lines(scripts.CALLBACK_NUMBER_AGAIN))
        self.callback_number = digits
        self.awaiting = "callback_confirm"
        return Response(
            [
                sentence(scripts.CALLBACK_CONFIRM_BEFORE + " ", Phone(digits), "."),
                sentence(scripts.CALLBACK_CONFIRM_AFTER),
            ]
        )

    def _callback_answer(self, text: str) -> Response | None:
        digits = parsing.find_phone_digits(text)
        if self.awaiting == "callback_confirm" and rules.is_yes(text) and len(digits) < 7:
            self.awaiting = None
            reason = self.callback_reason or "caller_request"
            self._event("callback_requested", number=self.callback_number, reason=reason)
            self.handoff_summary = self._summary("callback", reason)
            return Response(lines(scripts.CALLBACK_REQUESTED, scripts.ANYTHING_ELSE))
        if digits:
            return self._callback_number(digits)
        if self.awaiting == "callback_confirm" and rules.is_no(text):
            self.awaiting = "callback_number"
            return Response(lines(scripts.CALLBACK_ASK_NUMBER))
        self.awaiting = None
        return None

    def _offer_answer(self, text: str) -> Response | None:
        offer = self.awaiting
        if rules.is_yes(text):
            self.awaiting = None
            if offer == "transfer_offer":
                return self._transfer(self.offer_reason or "caller_request")
            self._event("nurse_message_requested", topic=self.nurse_topic)
            return Response(lines(scripts.NURSE_MESSAGE_SENT, scripts.ANYTHING_ELSE))
        self.awaiting = None
        if rules.is_no(text):
            return Response(lines(scripts.OKAY, scripts.HOW_CAN_I_HELP))
        return None

    def _handle(self, meaning: Interpretation, text: str) -> list[Unit]:
        intent = meaning.intent
        if intent == "cancel":
            had_something = self.offer is not None or self.booking is not None
            self.offer = None
            self.booking = None
            self.awaiting = None
            return self._say("booking_cancelled" if had_something else "nothing_pending")
        if intent == "goodbye":
            self.ended = True
            self._event("call_ended", reason="goodbye")
            return self._say("goodbye")
        if intent == "other_patient":
            self._note("other_patient")
            return self._say("other_patient" if self.verified else "other_patient_unverified")
        if not self.verified and (meaning.name or meaning.dob_text):
            return self._identify(meaning)
        if intent == "book":
            return self._book(meaning)
        if intent == "pick" and self.offer:
            return self._pick(meaning)
        if intent == "confirm":
            return self._confirm()
        if intent == "decline":
            if self.offer:
                self.offer = None
                self.booking = None
                return self._say("booking_cancelled")
            return self._say("nothing_pending")
        if intent == "ask_appointment":
            return self._appointment_info()
        if intent == "identify":
            return self._say("how_help")
        return self._say("not_understood", heard=text)

    # FERN-2: verification.

    def _locked(self) -> list[Unit]:
        texts = [scripts.VERIFY_LOCKED, scripts.TRANSFER_OFFER]
        return self._offer_transfer("verification_failed", texts).units

    def _need_identity(self, then: str) -> list[Unit]:
        if self.verify_locked:
            return self._locked()
        self.after_verify = then
        self.awaiting = "identity"
        return self._say("verify_ask")

    def _identify(self, meaning: Interpretation) -> list[Unit]:
        if self.verify_locked:
            return self._locked()
        if meaning.intent in ("ask_appointment", "book"):
            self.after_verify = meaning.intent
            if meaning.intent == "book":
                self._remember_booking(meaning)
        self.partial_name = meaning.name or self.partial_name
        self.partial_dob = meaning.dob_text or self.partial_dob
        if self.partial_dob is None:
            self.awaiting = "identity"
            return self._say("verify_need_dob")
        if self.partial_name is None:
            self.awaiting = "identity"
            return self._say("verify_need_name")

        dob = parsing.parse_dob(self.partial_dob, self.today.year)
        matches = self.world.find_patients(self.partial_name, dob) if dob else []
        self.partial_name = None
        self.partial_dob = None
        if len(matches) == 1:
            return self._verified(matches[0])

        self.verify_failures += 1
        self._event("verification_attempt", matched=False)
        if bugs.active("verify-attempts-off-by-one"):
            stop = self.verify_failures > MAX_VERIFY_ATTEMPTS
        else:
            stop = self.verify_failures >= MAX_VERIFY_ATTEMPTS
        if stop:
            self.verify_locked = True
            self.flags.append("verification_failed")
            return self._locked()
        self.awaiting = "identity"
        return self._say("verify_retry")

    def _verified(self, patient: Patient) -> list[Unit]:
        self.patient = patient
        self.awaiting = None
        self._event("verification_attempt", matched=True)
        units = self._say("verified", first_name=patient.first_name)
        follow, self.after_verify = self.after_verify, None
        if follow == "ask_appointment":
            return units + self._appointment_info()
        if follow == "book":
            return units + self._search()
        return units + self._say("how_help")

    def _appointment_info(self) -> list[Unit]:
        self._note("appointment_info")
        if not self.patient:
            return self._need_identity("ask_appointment")
        patient_id = None if bugs.active("other-patient-leak") else self.patient.id
        upcoming = self.world.upcoming(patient_id, self.started_at)
        if not upcoming:
            return self._say("no_appointments")
        nxt = upcoming[0]
        return self._say(
            "next_appointment",
            provider=self._provider_name(nxt.provider_id),
            when=spoken_slot(nxt.start),
        )

    # FERN-3: booking.

    def _remember_booking(self, meaning: Interpretation) -> None:
        self._note("booking")
        request = self.booking or BookingRequest()
        if meaning.provider:
            request.provider_id = meaning.provider.id
        if meaning.day:
            request.day = meaning.day
        if meaning.part:
            request.part = meaning.part
            request.time = None
        if meaning.time:
            request.time = meaning.time
            request.part = None
        self.booking = request

    def _book(self, meaning: Interpretation) -> list[Unit]:
        self._remember_booking(meaning)
        if not self.patient:
            return self._need_identity("book")
        return self._search()

    def _window(self, request: BookingRequest) -> tuple[datetime, datetime]:
        assert request.day is not None
        if request.time:
            start = datetime.combine(request.day, request.time, tzinfo=CLINIC_TZ)
            return start, start + SLOT
        first, last = OPEN, CLOSE
        if request.part == "morning":
            last = NOON
        elif request.part == "afternoon":
            first = NOON
        return (
            datetime.combine(request.day, first, tzinfo=CLINIC_TZ),
            datetime.combine(request.day, last, tzinfo=CLINIC_TZ),
        )

    @staticmethod
    def _matches(slot: datetime, request: BookingRequest) -> bool:
        local = slot.astimezone(CLINIC_TZ)
        if local.date() != request.day:
            return False
        if request.time:
            return local.time() == request.time
        if request.part == "morning":
            if bugs.active("morning-includes-noon"):
                return local.time() <= NOON
            return local.time() < NOON
        if request.part == "afternoon":
            return local.time() >= NOON
        return True

    def _search(self) -> list[Unit]:
        request = self.booking or BookingRequest()
        if request.provider_id is None:
            self.awaiting = "booking"
            return self._say("ask_provider")
        if request.day is None:
            self.awaiting = "booking"
            return self._say("ask_day")
        self.awaiting = None
        open_slots = self.world.open_slots(request.provider_id, after=self.started_at)
        matching = [s for s in open_slots if self._matches(s, request)][:MAX_OFFERED]
        if matching:
            return self._offer(request.provider_id, matching, "match")
        start, end = self._window(request)

        def distance(slot: datetime) -> timedelta:
            if slot >= end:
                return slot - end
            if slot + SLOT <= start:
                return start - (slot + SLOT)
            return timedelta(0)

        nearest = sorted(sorted(open_slots, key=lambda s: (distance(s), s))[:MAX_OFFERED])
        return self._offer(request.provider_id, nearest, "nearest")

    def _offer(self, provider_id: str, slots: list[datetime], reason: str) -> list[Unit]:
        provider = self._provider_name(provider_id)
        if not slots:
            self.offer = None
            return self._say("no_openings", provider=provider)
        self.offer = Offer(provider_id, slots)
        self._event(
            "slots_offered",
            provider=provider,
            slots=[s.isoformat() for s in slots],
            reason=reason,
        )
        spoken = spoken_list([spoken_slot(s) for s in slots])
        if reason == "nearest":
            return self._say("offer_nearest", provider=provider, slots=spoken)
        if len(slots) == 1:
            return self._say("offer_one", provider=provider, slot=spoken)
        return self._say("offer_many", provider=provider, slots=spoken)

    def _pick(self, meaning: Interpretation) -> list[Unit]:
        assert self.offer is not None
        slots = self.offer.slots
        if meaning.time is not None:
            chosen = [s for s in slots if s.astimezone(CLINIC_TZ).time() == meaning.time]
            if meaning.day:
                chosen = [s for s in chosen if s.astimezone(CLINIC_TZ).date() == meaning.day]
            if not chosen:
                return self._book(meaning)  # a time we did not offer is a new request
            return self._confirm_slot(chosen[0])
        assert meaning.pick is not None
        if meaning.pick >= len(slots):
            return self._say("which_one", slots=spoken_list([spoken_slot(s) for s in slots]))
        return self._confirm_slot(slots[meaning.pick])

    def _confirm(self) -> list[Unit]:
        if self.offer:
            if self.offer.chosen:
                return self._confirm_slot(self.offer.chosen)
            if len(self.offer.slots) == 1:
                return self._confirm_slot(self.offer.slots[0])
            return self._say(
                "which_one", slots=spoken_list([spoken_slot(s) for s in self.offer.slots])
            )
        if self.last_booking:
            return self._say(
                "already_booked",
                provider=self._provider_name(self.last_booking.provider_id),
                when=spoken_slot(self.last_booking.start),
            )
        return self._say("nothing_pending")

    def _confirm_slot(self, slot: datetime) -> list[Unit]:
        assert self.offer is not None and self.patient is not None
        offer = self.offer
        offer.chosen = slot
        provider = self._provider_name(offer.provider_id)
        if self.world.taken_by_someone_else(offer.provider_id, slot, self.patient.id):
            self._event("slot_unavailable", provider=provider, start=slot.isoformat())
            following = self.world.open_slots(offer.provider_id, after=slot)[:MAX_OFFERED]
            self.offer = Offer(offer.provider_id, following)
            self._event(
                "slots_offered",
                provider=provider,
                slots=[s.isoformat() for s in following],
                reason="slot_taken",
            )
            return self._say(
                "slot_taken",
                slot=spoken_slot(slot),
                provider=provider,
                slots=spoken_list([spoken_slot(s) for s in following]),
            )
        appointment = self.world.book(self.patient.id, offer.provider_id, slot)
        self.created.append(appointment)
        self.last_booking = appointment
        self._event(
            "appointment_booked",
            appointmentId=appointment.id,
            provider=provider,
            start=slot.isoformat(),
        )
        if not bugs.active("double-booking"):
            self.offer = None  # the offer is used up
        self.booking = None
        when = slot.isoformat() if bugs.active("raw-timestamp-spoken") else spoken_slot(slot)
        return self._say("booked", provider=provider, when=when) + lines(scripts.ANYTHING_ELSE)
