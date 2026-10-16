"""The generators, with no network and no API key.

* A fake client object stands in for ``anthropic.Anthropic`` to check the request
  ClaudeGenerator sends and how each kind of response is handled.
* A real ``anthropic.Anthropic`` client on an ``httpx2.MockTransport`` checks the
  same request as it goes over the wire, and the SDK's own exceptions. Its key is a
  dummy, and the transport never opens a socket.
* Cassettes are recorded into and replayed from a temporary directory.

tests/conftest.py removes ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN before any of
this runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from taskgen.context import system_text
from taskgen.generate import (
    CANNED_NOTE,
    DEFAULT_MODEL,
    FALLBACK_BETA,
    MAX_TOKENS,
    CannedGenerator,
    CassetteGenerator,
    Cassettes,
    ClaudeGenerator,
    FakeGenerator,
    GenerationError,
    GenerationResult,
    GeneratorConfigError,
    MissingCannedResponseError,
    MissingRecordingError,
    ModelResponse,
    RecordingError,
    StaleRecordingError,
    Usage,
    canonical_request,
    cost_line,
    make_generator,
    request_key,
)
from taskgen.prompt import card_message
from taskgen.schema import GeneratedTests, output_format, output_schema
from taskgen.sources import FileSource

REPO = Path(__file__).resolve().parents[2]
SOURCE = FileSource(REPO / "examples" / "fernhill-board.json")
CARD = SOURCE.card("FERN-2")
OTHER_CARD = SOURCE.card("FERN-13")
SYSTEM = system_text()

TEST_CODE = """\
import pytest

from harness import MARIA_ALVAREZ


@pytest.mark.criterion("FERN-2/AC-3")
def test_dob_in_words_verifies(call):
    call.say("This is Maria Alvarez, born March 3rd 1980.")

    assert call.record()["patientId"] == MARIA_ALVAREZ.id
"""
GOOD: dict[str, Any] = {
    "card": "FERN-2",
    "tests": [
        {
            "criterion": "AC-3",
            "name": "test_dob_in_words_verifies",
            "intent": "Fails if a date of birth said in words does not verify the caller.",
            "code": TEST_CODE,
        }
    ],
    "untestable": [],
}
GOOD_TEXT = json.dumps(GOOD)


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    """A TASKGEN_MODEL in the developer's shell must not change what these tests expect."""
    monkeypatch.delenv("TASKGEN_MODEL", raising=False)


def test_the_suite_has_no_model_credentials():
    assert [name for name in os.environ if name.startswith("ANTHROPIC_")] == ["ANTHROPIC_BASE_URL"]
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9"  # nothing listens there


def test_a_default_client_in_the_suite_cannot_reach_the_api(monkeypatch, tmp_path):
    """Even with an on-disk profile the SDK would find, the request goes nowhere."""
    configs = tmp_path / ".config" / "anthropic"
    (configs / "configs").mkdir(parents=True)
    (configs / "credentials").mkdir()
    (configs / "configs" / "default.json").write_text(
        json.dumps({"authentication": {"type": "user_oauth"}}), encoding="utf-8"
    )
    token = configs / "credentials" / "default.json"
    token.write_text(json.dumps({"access_token": "not-a-real-token"}), encoding="utf-8")
    token.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    generator = ClaudeGenerator(system=SYSTEM)

    assert str(generator.client.base_url).rstrip("/") == os.environ["ANTHROPIC_BASE_URL"]


def usage(
    input_tokens: int = 1000,
    output_tokens: int = 2000,
    cache_read: int = 10_000,
    cache_write: int = 0,
    iterations: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        iterations=iterations,
    )


def message(
    text: str = GOOD_TEXT,
    *,
    stop_reason: str = "end_turn",
    model: str = DEFAULT_MODEL,
    content: list[Any] | None = None,
    stop_details: Any = None,
    tokens: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if content is None else content,
        model=model,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=tokens or usage(),
    )


class FakeStream:
    def __init__(self, final: Any, request_id: str) -> None:
        self.final = final
        self.request_id = request_id

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.final


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeStream(outcome, request_id=f"req_test_{len(self.requests)}")


class FakeAnthropic:
    """Has only ``beta.messages.stream``, the one method ClaudeGenerator uses."""

    def __init__(self, *outcomes: Any) -> None:
        self.messages = FakeMessages(list(outcomes))
        self.beta = SimpleNamespace(messages=self.messages)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests


class NoClient:
    """Fails the test if anything tries to reach the API."""

    @property
    def beta(self) -> Any:
        raise AssertionError("the API was called")


def claude(*outcomes: Any, **kwargs: Any) -> tuple[ClaudeGenerator, FakeAnthropic]:
    client = FakeAnthropic(*outcomes)
    return ClaudeGenerator(client, system=SYSTEM, **kwargs), client


def test_the_request_has_exactly_the_documented_shape():
    generator, client = claude(message())

    generator.generate(CARD)

    assert client.requests == [
        {
            "model": "claude-opus-5-5",
            "max_tokens": 32000,
            "system": [
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": card_message(CARD)}],
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": output_schema()},
            },
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
    ]


def test_no_thinking_key_is_sent():
    generator, _ = claude()

    assert "thinking" not in generator.request(CARD)


def test_the_cache_marker_is_on_the_system_block_only():
    request = ClaudeGenerator(NoClient(), system=SYSTEM).request(CARD)

    assert [block.get("cache_control") for block in request["system"]] == [{"type": "ephemeral"}]
    assert isinstance(request["messages"][0]["content"], str)
    assert "cache_control" not in request


def test_every_card_shares_the_cached_prefix_and_differs_after_it():
    generator = ClaudeGenerator(NoClient(), system=SYSTEM)
    first, second = generator.request(CARD), generator.request(OTHER_CARD)

    assert first["system"] == second["system"]
    assert first["messages"] != second["messages"]
    assert card_message(OTHER_CARD) not in json.dumps(first["system"])


def test_the_default_system_block_is_the_built_context():
    assert ClaudeGenerator(NoClient()).system == SYSTEM


def test_the_output_schema_comes_from_the_pydantic_model():
    schema = output_schema()

    assert output_format() == {"type": "json_schema", "schema": schema}
    assert schema == anthropic.transform_schema(GeneratedTests)
    assert schema["required"] == ["card", "tests", "untestable"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ProposedTest"]["required"] == ["criterion", "name", "intent", "code"]


def test_taskgen_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("TASKGEN_MODEL", "claude-sonnet-5-5")

    assert ClaudeGenerator(NoClient(), system=SYSTEM).request(CARD)["model"] == "claude-sonnet-5-5"
    assert ClaudeGenerator(NoClient(), system=SYSTEM, model="m").request(CARD)["model"] == "m"


def test_the_fallback_can_be_turned_off():
    request = ClaudeGenerator(NoClient(), system=SYSTEM, fallbacks=False).request(CARD)

    assert "fallbacks" not in request
    assert "betas" not in request


def test_a_good_response_is_validated_and_its_usage_and_cost_kept():
    generator, _ = claude(message(tokens=usage(1000, 2000, 10_000, 500)))

    result = generator.generate(CARD)

    assert result.status == "ok"
    assert result.output == GeneratedTests.model_validate(GOOD)
    assert result.usage == Usage(1000, 2000, 10_000, 500)
    assert result.request_id == "req_test_1"
    assert result.model == "claude-opus-5-5"
    # 1000 x $4 + 2000 x $20 + 10,000 x $0.20 + 500 x $5, per million tokens
    assert result.cost_usd == pytest.approx(0.0485)


def test_text_is_read_by_block_type_not_position():
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=GOOD_TEXT[:40]),
        SimpleNamespace(type="text", text=GOOD_TEXT[40:]),
    ]
    generator, _ = claude(message(content=content))

    assert generator.generate(CARD).output == GeneratedTests.model_validate(GOOD)


def test_a_refusal_is_a_failed_card_not_a_crash():
    details = SimpleNamespace(category="cyber", explanation="Declined by a classifier.")
    generator, _ = claude(
        message('{"card": "FERN-2", "te', stop_reason="refusal", stop_details=details)
    )

    result = generator.generate(CARD)

    assert (result.status, result.output) == ("refusal", None)
    assert result.error == "The model declined this card (cyber). Declined by a classifier."


def test_a_refusals_partial_text_is_dropped_before_anything_reads_it():
    generator, _ = claude(message('{"card": "FERN-2", "te', stop_reason="refusal"))

    assert generator.call(generator.request(CARD)).text == ""


def test_a_refusal_without_details_still_reports_cleanly():
    generator, _ = claude(message("", stop_reason="refusal", content=[]))

    assert generator.generate(CARD).error == "The model declined this card (no category given)."


def test_an_output_cut_off_at_max_tokens_is_a_failed_card():
    generator, _ = claude(message(GOOD_TEXT[:50], stop_reason="max_tokens"))

    result = generator.generate(CARD)

    assert (result.status, result.output) == ("max_tokens", None)
    assert f"max_tokens ({MAX_TOKENS})" in (result.error or "")


def test_an_unexpected_stop_reason_is_reported():
    generator, _ = claude(message(stop_reason="pause_turn"))

    assert generator.generate(CARD).status == "unexpected_stop"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("not json", "does not match the output schema"),
        (json.dumps({**GOOD, "extra": 1}), "does not match the output schema"),
        (
            json.dumps({**GOOD, "tests": [{**GOOD["tests"][0], "name": "dob_check"}]}),
            "tests.0.name",
        ),
        (json.dumps({**GOOD, "card": "FERN-3"}), "for card FERN-3, not FERN-2"),
        ("", "no text"),
    ],
)
def test_output_that_fails_validation_is_a_failed_card(text, error):
    generator, _ = claude(message(text))

    result = generator.generate(CARD)

    assert result.status == "invalid_output"
    assert error in (result.error or "")


def iteration(kind: str, model: str | None, input_tokens: int, output_tokens: int) -> Any:
    return SimpleNamespace(
        type=kind,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=5000,
        cache_creation_input_tokens=0,
    )


def test_a_response_served_by_the_fallback_model_says_so_and_sums_both_attempts():
    iterations = [
        iteration("message", None, 100, 5),
        iteration("fallback_message", "claude-opus-4-8", 1000, 50),
    ]
    generator, _ = claude(message(model="claude-opus-4-8", tokens=usage(iterations=iterations)))

    result = generator.generate(CARD)

    assert result.status == "ok"
    assert result.served_by_fallback is True
    assert result.usage == Usage(1100, 55, 10_000, 0)
    assert result.cost_usd is None  # no list price for the fallback model here
    assert "cost unknown" in (cost_line([result]) or "")


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def streamed(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    first: list[Any] | None = None,
    final_usage: dict[str, Any] | None = None,
) -> bytes:
    """An SSE body for one text block, split in two deltas, after optional leading blocks."""
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 40,
                "output_tokens": 1,
                "cache_read_input_tokens": 12_000,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    leading = [
        event
        for index, block in enumerate(first or [])
        for event in (
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        )
    ]
    index = len(first or [])
    half = len(text) // 2
    return sse(
        start,
        *leading,
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        *(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": t},
            }
            for t in (text[:half], text[half:])
        ),
        {"type": "content_block_stop", "index": index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": final_usage or {"output_tokens": 900},
        },
        {"type": "message_stop"},
    )


class Wire:
    """A MockTransport handler that records requests and answers with fixed responses."""

    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="test-key-not-real",
            base_url="http://anthropic.test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self)),
        )


def sse_response(body: bytes) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda _: httpx2.Response(
        200,
        headers={"content-type": "text/event-stream", "request-id": "req_wire_1"},
        content=body,
    )


def test_the_sdk_sends_the_request_to_the_beta_endpoint_as_documented():
    wire = Wire(sse_response(streamed(GOOD_TEXT)))
    generator = ClaudeGenerator(wire.client(), system=SYSTEM)

    result = generator.generate(CARD)

    [request] = wire.requests
    body = json.loads(request.content)
    assert (request.method, request.url.path, request.url.params.get("beta")) == (
        "POST",
        "/v1/messages",
        "true",
    )
    assert request.headers["anthropic-beta"] == FALLBACK_BETA
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 32000
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "high", "format": output_format()}
    assert body["system"] == [
        {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": card_message(CARD)}]
    assert "thinking" not in body
    assert "betas" not in body
    assert result.status == "ok"
    assert result.output == GeneratedTests.model_validate(GOOD)
    assert result.request_id == "req_wire_1"
    assert result.usage == Usage(40, 900, 12_000, 0)


def test_the_sdk_path_reports_a_pre_output_fallback():
    fallback = {"type": "fallback", "from": {"model": DEFAULT_MODEL}, "to": {"model": "claude-x"}}
    attempts = [
        {"type": "message", "model": DEFAULT_MODEL, "input_tokens": 40, "output_tokens": 0},
        {"type": "fallback_message", "model": "claude-x", "input_tokens": 40, "output_tokens": 900},
    ]
    iterations = [
        {**attempt, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        for attempt in attempts
    ]
    body = streamed(
        GOOD_TEXT,
        model="claude-x",
        first=[fallback],
        final_usage={"output_tokens": 900, "iterations": iterations},
    )

    result = ClaudeGenerator(Wire(sse_response(body)).client(), system=SYSTEM).generate(CARD)

    assert result.status == "ok"
    assert result.served_by_fallback is True
    assert result.model == "claude-x"
    assert result.usage == Usage(80, 900, 0, 0)


def error_response(status: int) -> Callable[[httpx2.Request], httpx2.Response]:
    body = {"type": "error", "error": {"type": "test_error", "message": f"status {status}"}}
    return lambda _: httpx2.Response(status, json=body)


@pytest.mark.parametrize(
    ("status", "message_part"),
    [
        (400, "rejected the request (400)"),
        (401, "rejected the credentials (401)"),
        (403, "not allowed to use claude-opus-5-5 (403)"),
        (404, "claude-opus-5-5 or the endpoint was not found (404)"),
    ],
)
def test_errors_every_card_would_hit_stop_the_run(status, message_part):
    wire = Wire(error_response(status))

    with pytest.raises(GeneratorConfigError, match=re.escape(message_part)):
        ClaudeGenerator(wire.client(), system=SYSTEM).generate(CARD)


@pytest.mark.parametrize(
    ("status", "message_part"),
    [
        (429, "Rate limited (429)"),
        (500, "API error 500"),
        (529, "API error 529"),
    ],
)
def test_errors_left_after_the_sdk_retries_fail_only_that_card(status, message_part):
    wire = Wire(error_response(status))

    result = ClaudeGenerator(wire.client(), system=SYSTEM).generate(CARD)

    assert (result.status, result.output) == ("api_error", None)
    assert message_part in (result.error or "")


@pytest.mark.parametrize(
    ("error", "message_part"),
    [
        (httpx2.ConnectError("refused"), "Could not reach the API"),
        (httpx2.ReadTimeout("slow"), "timed out"),
    ],
)
def test_network_failures_fail_only_that_card(error, message_part):
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise error

    result = ClaudeGenerator(Wire(fail).client(), system=SYSTEM).generate(CARD)

    assert result.status == "api_error"
    assert message_part in (result.error or "")


def test_no_credentials_stop_the_run_before_any_request(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    wire = Wire(sse_response(streamed(GOOD_TEXT)))
    client = anthropic.Anthropic(
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wire)),
    )

    with pytest.raises(GeneratorConfigError, match="set ANTHROPIC_API_KEY"):
        ClaudeGenerator(client, system=SYSTEM).generate(CARD)
    assert wire.requests == []


def test_a_credentials_profile_that_cannot_be_loaded_stops_the_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("ANTHROPIC_PROFILE", "no-such-profile")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")

    with pytest.raises(GeneratorConfigError, match="credentials could not be loaded") as caught:
        ClaudeGenerator(system=SYSTEM, max_tokens=16).generate(CARD)
    assert isinstance(caught.value.__cause__, anthropic.CredentialsError)
    assert "no-such-profile" in str(caught.value)


def test_the_request_key_is_a_sha256_of_the_canonical_request():
    request = ClaudeGenerator(NoClient(), system=SYSTEM).request(CARD)
    canonical = {
        "model": "claude-opus-5-5",
        "system": [SYSTEM],
        "messages": [{"role": "user", "content": card_message(CARD)}],
        "schema": output_schema(),
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert canonical_request(request) == canonical
    assert request_key(request) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_request_key_ignores_transport_settings_and_follows_content():
    generator = ClaudeGenerator(NoClient(), system=SYSTEM)
    request = generator.request(CARD)
    no_fallback = ClaudeGenerator(NoClient(), system=SYSTEM, fallbacks=False).request(CARD)

    assert request_key(request) == request_key(no_fallback)
    assert request_key(request) != request_key(generator.request(OTHER_CARD))
    assert request_key(request) != request_key(
        ClaudeGenerator(NoClient(), system="x").request(CARD)
    )


def recorder(tmp_path: Path, *outcomes: Any) -> tuple[CassetteGenerator, FakeAnthropic]:
    client = FakeAnthropic(*outcomes)
    generator = CassetteGenerator(
        "record", claude=ClaudeGenerator(client, system=SYSTEM), cassettes=Cassettes(tmp_path)
    )
    return generator, client


def replayer(tmp_path: Path, **kwargs: Any) -> CassetteGenerator:
    return CassetteGenerator(
        "replay",
        claude=ClaudeGenerator(NoClient(), system=SYSTEM, **kwargs),
        cassettes=Cassettes(tmp_path),
    )


def test_record_then_replay_gives_the_same_result_without_the_api(tmp_path):
    recorded = recorder(tmp_path, message(tokens=usage(1000, 2000, 10_000, 500)))[0].generate(CARD)

    replayed = replayer(tmp_path).generate(CARD)

    assert (recorded.source, replayed.source) == ("record", "replay")
    assert replayed.output == recorded.output == GeneratedTests.model_validate(GOOD)
    assert (replayed.usage, replayed.cost_usd, replayed.request_id) == (
        recorded.usage,
        recorded.cost_usd,
        "req_test_1",
    )


def test_a_recording_holds_the_response_text_model_stop_reason_usage_and_request_id(tmp_path):
    generator, client = recorder(tmp_path, message())

    generator.generate(CARD)

    saved = json.loads((tmp_path / "FERN-2.json").read_text(encoding="utf-8"))
    assert saved["card"] == "FERN-2"
    assert saved["key"] == request_key(client.requests[0])
    assert set(saved["parts"]) == {"model", "system", "messages", "schema"}
    assert saved["settings"] == {
        "max_tokens": 32000,
        "effort": "high",
        "fallbacks": "default",
        "betas": [FALLBACK_BETA],
    }
    assert saved["response"]["text"] == GOOD_TEXT
    assert saved["response"]["model"] == "claude-opus-5-5"
    assert saved["response"]["stop_reason"] == "end_turn"
    assert saved["response"]["request_id"] == "req_test_1"
    assert saved["response"]["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 2000,
        "cache_read_input_tokens": 10_000,
        "cache_creation_input_tokens": 0,
    }


def test_a_recorded_refusal_replays_as_a_refusal(tmp_path):
    recorder(tmp_path, message("", stop_reason="refusal", content=[]))[0].generate(CARD)

    assert replayer(tmp_path).generate(CARD).status == "refusal"


def test_replay_names_the_missing_card_and_the_command_to_record_it(tmp_path):
    with pytest.raises(MissingRecordingError) as err:
        replayer(tmp_path).generate(CARD)

    assert str(err.value) == (
        f"No recording for FERN-2 (expected {tmp_path / 'FERN-2.json'}). Replay mode answers "
        "only from recordings. Record one with your own API key: "
        "ANTHROPIC_API_KEY=... taskgen generate --mode record --cards FERN-2"
    )
    assert (err.value.card, err.value.path) == ("FERN-2", tmp_path / "FERN-2.json")


def test_replay_says_which_part_of_the_request_changed(tmp_path, monkeypatch):
    recorder(tmp_path, message())[0].generate(CARD)

    with pytest.raises(StaleRecordingError, match="system changed") as err:
        CassetteGenerator(
            "replay",
            claude=ClaudeGenerator(NoClient(), system=SYSTEM + "\nOne more rule."),
            cassettes=Cassettes(tmp_path),
        ).generate(CARD)
    assert err.value.changed == ["system"]

    monkeypatch.setenv("TASKGEN_MODEL", "claude-sonnet-5-5")
    with pytest.raises(StaleRecordingError) as err:
        replayer(tmp_path).generate(CARD)
    assert err.value.changed == ["model"]


def test_an_unreadable_recording_is_reported_by_path(tmp_path):
    (tmp_path / "FERN-2.json").write_text("{half a recording", encoding="utf-8")

    with pytest.raises(RecordingError, match=r"FERN-2\.json is unreadable"):
        replayer(tmp_path).generate(CARD)


def test_live_mode_calls_the_api_and_keeps_nothing(tmp_path):
    client = FakeAnthropic(message())
    generator = CassetteGenerator(
        "live", claude=ClaudeGenerator(client, system=SYSTEM), cassettes=Cassettes(tmp_path)
    )

    assert generator.generate(CARD).source == "live"
    assert list(tmp_path.iterdir()) == []


def test_record_mode_reports_an_api_error_and_writes_nothing(tmp_path):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://anthropic.test"))
    generator, _ = recorder(tmp_path, error)

    assert generator.generate(CARD).status == "api_error"
    assert list(tmp_path.iterdir()) == []


def test_replay_mode_never_creates_a_client(tmp_path):
    recorder(tmp_path, message())[0].generate(CARD)
    generator = CassetteGenerator(
        "replay", claude=ClaudeGenerator(system=SYSTEM), cassettes=Cassettes(tmp_path)
    )

    assert generator.generate(CARD).status == "ok"
    assert generator.claude._client is None


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        make_generator("sometimes")
    with pytest.raises(ValueError, match="mode must be"):
        CassetteGenerator("canned", claude=ClaudeGenerator(NoClient(), system=SYSTEM))


def test_make_generator_picks_the_generator_for_each_mode(tmp_path):
    assert isinstance(make_generator("canned", canned=tmp_path), CannedGenerator)
    replay = make_generator("replay", recordings=tmp_path, client=NoClient())
    assert isinstance(replay, CassetteGenerator)
    assert replay.mode == "replay"
    assert replay.cassettes.directory == tmp_path


def write_canned(directory: Path, ref: str, data: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{ref}.json").write_text(json.dumps(data), encoding="utf-8")


def test_a_canned_response_is_validated_and_its_notes_dropped(tmp_path):
    tests = [{**GOOD["tests"][0], "_why": "a good test"}]
    write_canned(tmp_path, "FERN-2", {"_note": CANNED_NOTE, **GOOD, "tests": tests})

    result = CannedGenerator(tmp_path).generate(CARD)

    assert (result.status, result.source, result.used_model) == ("ok", "canned", False)
    assert result.output == GeneratedTests.model_validate(GOOD)
    assert cost_line([result]) is None


def test_a_canned_file_without_the_note_is_refused(tmp_path):
    write_canned(tmp_path, "FERN-2", GOOD)

    with pytest.raises(GenerationError, match="must carry"):
        CannedGenerator(tmp_path).generate(CARD)


def test_a_missing_canned_response_names_the_cards_that_have_one(tmp_path):
    write_canned(tmp_path, "FERN-11", {"_note": CANNED_NOTE, **GOOD, "card": "FERN-11"})
    write_canned(tmp_path, "FERN-4", {"_note": CANNED_NOTE, **GOOD, "card": "FERN-4"})

    with pytest.raises(MissingCannedResponseError, match=r"canned: FERN-4, FERN-11"):
        CannedGenerator(tmp_path).generate(CARD)
    assert CannedGenerator(tmp_path).available() == ["FERN-4", "FERN-11"]


def test_every_canned_stand_in_in_the_repo_is_labelled_and_valid():
    canned = CannedGenerator()
    results = [canned.generate(SOURCE.card(ref)) for ref in canned.available()]

    assert [result.status for result in results] == ["ok"] * len(results)


def test_the_fake_generator_answers_from_code_and_records_calls():
    refusal = ModelResponse(text="", model="fake", stop_reason="refusal")
    fake = FakeGenerator(
        {"FERN-2": GOOD, "FERN-13": refusal},
        default=lambda card: {**GOOD, "card": card.ref, "tests": []},
    )

    results = [fake.generate(card) for card in (CARD, OTHER_CARD, SOURCE.card("FERN-4"))]

    assert [result.status for result in results] == ["ok", "refusal", "ok"]
    assert results[2].output == GeneratedTests(card="FERN-4", tests=[], untestable=[])
    assert fake.calls == ["FERN-2", "FERN-13", "FERN-4"]


def test_the_fake_generator_refuses_a_card_it_has_no_answer_for():
    with pytest.raises(KeyError, match="no response for FERN-2"):
        FakeGenerator().generate(CARD)


def test_a_result_survives_a_round_trip_through_json():
    result = claude(message())[0].generate(CARD)

    assert GenerationResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_a_card_that_failed_with_an_api_error_costs_nothing_in_the_cost_line():
    answered = claude(message(tokens=usage(1000, 5000, 0, 12_000)))[0].generate(CARD)
    error = anthropic.InternalServerError(
        "overloaded",
        response=httpx2.Response(500, request=httpx2.Request("POST", "http://anthropic.test")),
        body=None,
    )
    failed = claude(error)[0].generate(OTHER_CARD)

    assert (failed.status, failed.cost_usd, failed.used_model) == ("api_error", None, True)
    assert cost_line([answered, failed]) == (
        "claude-opus-5-5 · 1 card(s), 1 more with no response · 1,000 input, 0 cache read, "
        "12,000 cache write, 5,000 output tokens · about $0.16 (estimated from list prices "
        "at the time of writing, which may change)"
    )


def test_the_cost_line_shows_tokens_cache_reads_and_the_estimate():
    results = [
        claude(message(tokens=usage(1000, 2000, 10_000, 500)))[0].generate(CARD),
        claude(message(json.dumps({**GOOD, "card": "FERN-13"}), tokens=usage(200, 1000, 12_000)))[
            0
        ].generate(OTHER_CARD),
    ]

    assert cost_line(results) == (
        "claude-opus-5-5 · 2 card(s) · 1,200 input, 22,000 cache read, 500 cache write, "
        "3,000 output tokens · about $0.07 (estimated from list prices at the time of "
        "writing, which may change)"
    )


def test_a_canned_file_that_fails_the_schema_is_a_failed_card_not_a_crash(tmp_path):
    misnamed = {**GOOD, "tests": [{**GOOD["tests"][0], "name": "dob_check"}]}
    write_canned(tmp_path, "FERN-2", {"_note": CANNED_NOTE, **misnamed})

    result = CannedGenerator(tmp_path).generate(CARD)

    assert (result.status, result.source, result.output) == ("invalid_output", "canned", None)
    assert "tests.0.name" in (result.error or "")


def test_a_canned_file_that_is_not_json_is_reported_by_path(tmp_path):
    (tmp_path / "FERN-2.json").write_text("{half a file", encoding="utf-8")

    with pytest.raises(GenerationError, match=r"FERN-2\.json is not valid JSON"):
        CannedGenerator(tmp_path).generate(CARD)


def test_a_recording_missing_a_field_is_unreadable_not_a_crash(tmp_path):
    (tmp_path / "FERN-2.json").write_text(
        json.dumps({"card": "FERN-2", "key": "x"}), encoding="utf-8"
    )

    with pytest.raises(
        RecordingError, match=r"FERN-2\.json is unreadable \(KeyError: 'response'\)"
    ):
        replayer(tmp_path).generate(CARD)


def test_cassettes_refuse_a_ref_that_is_not_a_card(tmp_path):
    with pytest.raises(ValueError, match="not a card reference"):
        Cassettes(tmp_path).path("../../etc/passwd")
