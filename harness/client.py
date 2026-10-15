"""A small client for the Front Desk API, shaped for short tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from harness.world import DEFAULT_AT, Patient


class CallError(Exception):
    """The API answered with an error status. ``status``, ``error`` and ``detail`` say which."""

    def __init__(self, status: int, error: str, detail: str) -> None:
        super().__init__(f"{status} {error}: {detail}")
        self.status = status
        self.error = error
        self.detail = detail


@dataclass(frozen=True)
class Reply:
    """What the agent said in one turn, plus whether the call ended or was transferred."""

    text: str
    ssml: str
    ended: bool = False
    transferred: bool = False


def _iso(value: str | datetime) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


def _check(response: httpx.Response) -> Any:
    if response.is_success:
        return response.json()
    try:
        body = response.json()
        error, detail = str(body.get("error", "")), str(body.get("detail", ""))
    except ValueError:
        error, detail = "http_error", response.text
    raise CallError(response.status_code, error, detail)


class Call:
    """One phone call in progress."""

    def __init__(self, http: httpx.Client, call_id: str, greeting: Reply) -> None:
        self._http = http
        self.id = call_id
        self.greeting = greeting
        self.last_reply = greeting

    def _turn(self, body: dict[str, Any]) -> Reply:
        data = _check(self._http.post(f"/calls/{self.id}/turns", json=body))
        reply = Reply(
            text=data["reply"]["text"],
            ssml=data["reply"]["ssml"],
            ended=data["ended"],
            transferred=data["transferred"],
        )
        self.last_reply = reply
        return reply

    def say(self, text: str, confidence: float = 0.95, barge_in: bool = False) -> Reply:
        """The caller says ``text`` (as speech recognition heard it)."""
        return self._turn({"text": text, "asrConfidence": confidence, "bargeIn": barge_in})

    def press(self, digits: str) -> Reply:
        """The caller presses keypad digits, such as "0"."""
        return self._turn({"dtmf": digits})

    def verify_as(self, patient: Patient) -> Reply:
        """Say the patient's full name and date of birth (in ISO form, 1980-03-03)."""
        return self.say(f"My name is {patient.full_name}, date of birth {patient.dob.isoformat()}.")

    def record(self) -> dict[str, Any]:
        """The call record, as GET /calls/{id} returns it."""
        return _check(self._http.get(f"/calls/{self.id}"))

    def events(self, type: str | None = None) -> list[dict[str, Any]]:
        """The call record's events, optionally only those of one type."""
        return [e for e in self.record()["events"] if type is None or e["type"] == type]

    def appointments(self) -> list[dict[str, Any]]:
        """Appointments created during this call."""
        return _check(self._http.get(f"/calls/{self.id}/appointments"))

    def take_slot(self, provider: str, start: str | datetime) -> None:
        """Someone else books this slot now, in this call's world."""
        _check(
            self._http.post(
                f"/calls/{self.id}/test/take-slot",
                json={"provider": provider, "start": _iso(start)},
            )
        )


class FrontDesk:
    """The API at ``base_url``."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def start_call(
        self, at: str | datetime | None = DEFAULT_AT, model_available: bool = True
    ) -> Call:
        """Start a call. ``at`` is the call's clock; None means the server's current time."""
        body: dict[str, Any] = {"modelAvailable": model_available}
        if at is not None:
            body["at"] = _iso(at)
        data = _check(self.http.post("/calls", json=body))
        greeting = Reply(text=data["reply"]["text"], ssml=data["reply"]["ssml"])
        return Call(self.http, data["callId"], greeting)

    def close(self) -> None:
        self.http.close()
