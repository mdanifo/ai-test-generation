"""Front Desk's HTTP API: start a call, send caller turns, read the call record.

Run it with ``uvicorn frontdesk.main:app``. Set ``FRONTDESK_TEST_HOOKS=1`` to
enable the test-only routes under ``/calls/{id}/test/``.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from frontdesk import bugs
from frontdesk.dialog import Call, CallEnded
from frontdesk.model import Model, ScriptedModel
from frontdesk.world import CLINIC_TZ, World

log = logging.getLogger("frontdesk")

TEST_HOOKS_ENV = "FRONTDESK_TEST_HOOKS"
MAX_CALLS_KEPT = 10_000


class ErrorOut(BaseModel):
    error: str = Field(description="A short machine-readable code, such as call_not_found.")
    detail: str = Field(description="What went wrong, for a person to read.")


class Health(BaseModel):
    status: Literal["ok"]


class StartCallIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: AwareDatetime | None = Field(
        default=None,
        description="The call's clock: an ISO 8601 datetime with an offset. Default: now. "
        "The clock does not move during the call.",
    )
    modelAvailable: bool = Field(
        default=True,
        description="False simulates a call where the language model cannot be reached.",
    )


class Reply(BaseModel):
    text: str = Field(description="What the agent says, as plain text.")
    ssml: str = Field(
        description="The same reply as SSML for speech synthesis. Uses only the speak, "
        "break, say-as and prosody tags."
    )


class CallStarted(BaseModel):
    callId: str
    reply: Reply = Field(description="The greeting, which starts with the recording disclosure.")


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(
        default=None,
        max_length=2000,
        description="What the caller said, as recognized by speech recognition.",
    )
    dtmf: str | None = Field(
        default=None,
        pattern=r"^[0-9*#]{1,20}$",
        description="Keypad digits the caller pressed.",
    )
    asrConfidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Speech recognition confidence for text, from 0 to 1.",
    )
    bargeIn: bool = Field(
        default=False,
        description="True when the caller started speaking while the previous reply was playing.",
    )

    @model_validator(mode="after")
    def _text_or_keys(self) -> TurnIn:
        if not (self.text and self.text.strip()) and not self.dtmf:
            raise ValueError("a turn needs text, dtmf or both")
        return self


class TurnOut(BaseModel):
    reply: Reply
    ended: bool = Field(description="True once the call is over. Later turns get 409.")
    transferred: bool = Field(description="True when this call was transferred to staff.")


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(
        description="One of: emergency_detected, nurse_line_offered, nurse_message_requested, "
        "verification_attempt, slots_offered, slot_unavailable, appointment_booked, "
        "transfer_offered, transfer, callback_offered, callback_requested, "
        "recording_stopped, call_ended. Other fields depend on the type."
    )
    at: str = Field(description="The call's clock when it happened (ISO 8601).")


class Turn(BaseModel):
    speaker: Literal["caller", "agent"]
    text: str
    ssml: str | None = Field(default=None, description="Agent turns only.")
    dtmf: str | None = Field(default=None, description="Caller turns only.")
    asrConfidence: float | None = Field(default=None, description="Caller turns with text.")
    bargeIn: bool | None = Field(default=None, description="Caller turns only.")


class HandoffSummary(BaseModel):
    kind: Literal["transfer", "callback"]
    reason: str = Field(
        description="caller_request, low_confidence, verification_failed or model_unavailable."
    )
    verified: bool
    request: str = Field(description="What the caller asked for, in plain words.")
    text: str = Field(description="The one-line summary staff read.")


class CallRecord(BaseModel):
    callId: str
    startedAt: str
    modelAvailable: bool
    verified: bool
    patientId: str | None
    optedOut: bool = Field(description="True once the caller declined recording.")
    recording: bool
    transcriptStored: bool = Field(description="False once the caller opted out.")
    ended: bool
    transferred: bool
    flags: list[str] = Field(description="For example emergency or verification_failed.")
    events: list[Event]
    turns: list[Turn] = Field(description="The stored transcript, starting with the greeting.")
    handoffSummary: HandoffSummary | None


class AppointmentOut(BaseModel):
    appointmentId: str
    patientId: str
    provider: str = Field(description='As spoken, for example "Dr. Okafor".')
    start: str
    end: str


class TakeSlotIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(description='"Dr. Okafor", "Okafor" or "okafor".')
    start: datetime = Field(description="Slot start. Without an offset it is read as clinic time.")


class TakeSlotOut(BaseModel):
    provider: str
    start: str
    taken: bool


class ApiError(Exception):
    def __init__(self, status: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


class CallStore:
    """Calls in memory. Each call holds its own copy of the world."""

    def __init__(self, model_factory: Callable[[bool], Model]) -> None:
        self.lock = threading.Lock()
        self._calls: OrderedDict[str, Call] = OrderedDict()
        self._model_factory = model_factory

    def start(self, at: datetime | None, model_available: bool) -> Call:
        call = Call(
            call_id=f"call-{uuid.uuid4().hex[:12]}",
            started_at=(at or datetime.now(CLINIC_TZ)),
            model=self._model_factory(model_available),
            model_available=model_available,
            world=World.fresh(),
        )
        self._calls[call.call_id] = call
        while len(self._calls) > MAX_CALLS_KEPT:
            self._calls.popitem(last=False)
        return call

    def get(self, call_id: str) -> Call:
        call = self._calls.get(call_id)
        if call is None:
            raise ApiError(404, "call_not_found", f"No call with id {call_id!r}.")
        return call


def hooks_enabled() -> bool:
    return os.environ.get(TEST_HOOKS_ENV, "") == "1"


def create_app(model_factory: Callable[[bool], Model] = ScriptedModel) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        seeded = bugs.current()  # an unknown SEEDED_BUG stops startup here
        if seeded:
            log.warning("SEEDED_BUG=%s is active", seeded)
        yield

    app = FastAPI(
        title="Front Desk (sample)",
        version="1.0.0",
        description=(
            "A text simulation of Front Desk, a fictional AI voice agent that answers the "
            "phone for Brookside Family Clinic. Voice is represented by turn fields (text as "
            "recognized, a recognition confidence, keypad digits, a barge-in flag); speech "
            "output is SSML. Every call gets its own copy of the sample clinic world."
        ),
        lifespan=lifespan,
    )
    store = CallStore(model_factory)
    app.state.store = store
    errors: dict[int | str, dict[str, Any]] = {
        404: {"model": ErrorOut, "description": "Unknown call."},
        422: {"model": ErrorOut, "description": "Invalid request body."},
    }

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.error, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for item in exc.errors():
            where = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            problems.append(f"{where or 'body'}: {item.get('msg', 'invalid')}")
        return _error(422, "invalid_request", "; ".join(problems))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(Exception)
    async def _crash(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(500, "internal_error", type(exc).__name__)

    @app.get("/health", response_model=Health)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/calls", response_model=CallStarted, status_code=201, responses=errors)
    def start_call(body: StartCallIn | None = None) -> dict[str, Any]:
        """Start a call. The reply is the greeting."""
        body = body or StartCallIn()
        with store.lock:
            call = store.start(body.at, body.modelAvailable)
            greeting = call.greet()
        return {"callId": call.call_id, "reply": {"text": greeting.text, "ssml": greeting.ssml}}

    @app.post(
        "/calls/{call_id}/turns",
        response_model=TurnOut,
        responses={
            **errors,
            409: {"model": ErrorOut, "description": "The call has ended."},
        },
    )
    def caller_turn(call_id: str, body: TurnIn) -> dict[str, Any]:
        """Send one caller turn and get the agent's reply."""
        with store.lock:
            call = store.get(call_id)
            try:
                result = call.turn(body.text, body.dtmf, body.asrConfidence, body.bargeIn)
            except CallEnded:
                raise ApiError(409, "call_ended", f"Call {call_id!r} has ended.") from None
        return {
            "reply": {"text": result.reply.text, "ssml": result.reply.ssml},
            "ended": result.ended,
            "transferred": result.transferred,
        }

    @app.get("/calls/{call_id}", response_model=CallRecord, responses=errors)
    def call_record(call_id: str) -> dict[str, Any]:
        """The call record: state, flags, events, the stored transcript and the handoff summary."""
        with store.lock:
            return store.get(call_id).record()

    @app.get("/calls/{call_id}/appointments", response_model=list[AppointmentOut], responses=errors)
    def call_appointments(call_id: str) -> list[dict[str, Any]]:
        """Appointments created during this call."""
        with store.lock:
            return store.get(call_id).appointment_records()

    @app.post(
        "/calls/{call_id}/test/take-slot",
        response_model=TakeSlotOut,
        responses=errors,
    )
    def take_slot(call_id: str, body: TakeSlotIn) -> dict[str, Any]:
        """Test hook: someone else books this slot in this call's world.

        Enabled only when FRONTDESK_TEST_HOOKS=1; otherwise 404.
        """
        if not hooks_enabled():
            raise ApiError(404, "test_hooks_disabled", f"Set {TEST_HOOKS_ENV}=1 to enable.")
        with store.lock:
            call = store.get(call_id)
            provider = World.find_provider(body.provider)
            if provider is None:
                raise ApiError(422, "unknown_provider", f"No provider {body.provider!r}.")
            start = body.start
            if start.tzinfo is None:
                start = start.replace(tzinfo=CLINIC_TZ)
            start = start.astimezone(CLINIC_TZ)
            if not call.world.is_slot(start):
                raise ApiError(422, "not_a_slot", f"{start.isoformat()} is not a schedule slot.")
            call.take_slot(provider.id, start)
        return {"provider": provider.name, "start": start.isoformat(), "taken": True}

    @app.get("/test/build", include_in_schema=False)
    def build() -> dict[str, Any]:
        """Which build is running, for the harness's consistency check. Test hooks only."""
        if not hooks_enabled():
            raise ApiError(404, "test_hooks_disabled", f"Set {TEST_HOOKS_ENV}=1 to enable.")
        return {"seededBug": bugs.current()}

    return app


app = create_app()
