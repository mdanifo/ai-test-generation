"""The HTTP contract around the dialog: status codes, error bodies, isolation, startup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from frontdesk import bugs
from frontdesk.main import app
from harness import DEFAULT_AT, MARIA_ALVAREZ, CallError

REPO = Path(__file__).resolve().parents[2]


def test_health(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_call_is_404_with_an_error_body(api):
    response = api.get("/calls/call-000000000000")

    assert response.status_code == 404
    assert response.json()["error"] == "call_not_found"


def test_a_turn_on_an_ended_call_is_409(call):
    call.say("That's all, goodbye.")

    with pytest.raises(CallError) as caught:
        call.say("Hello?")
    assert caught.value.status == 409
    assert caught.value.error == "call_ended"


def test_a_turn_without_text_or_keys_is_422(api, call):
    response = api.post(f"/calls/{call.id}/turns", json={"asrConfidence": 0.9})

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_a_call_clock_without_an_offset_is_422(api):
    response = api.post("/calls", json={"at": "2026-10-14T10:00:00"})

    assert response.status_code == 422


def test_a_confidence_above_one_is_422(api, call):
    response = api.post(f"/calls/{call.id}/turns", json={"text": "hi", "asrConfidence": 1.5})

    assert response.status_code == 422
    assert "asrConfidence" in response.json()["detail"]


def test_start_call_returns_the_call_clock_in_the_record(call):
    assert call.record()["startedAt"] == DEFAULT_AT


def test_calls_do_not_share_the_world(start_call):
    first = start_call()
    second = start_call()
    first.take_slot("Dr. Lindqvist", "2026-10-20T09:30:00-04:00")
    second.verify_as(MARIA_ALVAREZ)
    second.say("Book Dr. Lindqvist on October 20th at 9:30.")
    second.say("Yes.")

    assert [a["start"] for a in second.appointments()] == ["2026-10-20T09:30:00-04:00"]
    assert first.appointments() == []


def test_take_slot_rejects_a_time_that_is_not_a_slot(call):
    with pytest.raises(CallError) as caught:
        call.take_slot("Dr. Okafor", "2026-10-20T09:15:00-04:00")
    assert caught.value.status == 422


def test_take_slot_is_404_without_test_hooks(monkeypatch):
    monkeypatch.delenv("FRONTDESK_TEST_HOOKS", raising=False)
    with TestClient(app) as client:
        call_id = client.post("/calls", json={"at": DEFAULT_AT}).json()["callId"]
        response = client.post(
            f"/calls/{call_id}/test/take-slot",
            json={"provider": "Dr. Okafor", "start": "2026-10-20T11:00:00-04:00"},
        )

    assert response.status_code == 404
    assert response.json()["error"] == "test_hooks_disabled"


def test_the_app_refuses_to_start_with_an_unknown_seeded_bug():
    env = {**os.environ, "SEEDED_BUG": "no-such-bug"}
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "frontdesk.main:app", "--port", "0"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "'no-such-bug' is not a known seeded bug" in result.stdout + result.stderr


def test_an_unknown_bug_name_in_a_branch_is_an_error():
    with pytest.raises(bugs.UnknownBugError):
        bugs.active("not-in-the-catalog")


def test_the_openapi_document_never_mentions_the_seeded_bugs():
    document = json.dumps(app.openapi()).lower()

    assert "seeded" not in document
    assert "bug" not in document
    assert not any(name in document for name in bugs.CATALOG)
    assert "/test/build" not in document


def test_the_suite_runs_without_model_credentials():
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ
