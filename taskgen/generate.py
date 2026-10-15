"""Generators: turn one card into proposed tests.

* ``ClaudeGenerator`` calls the Anthropic API through the official SDK.
* ``CassetteGenerator`` wraps it for ``taskgen generate --mode replay|record|live``.
  Replay (the default) never touches the network: it reads ``recordings/<card>.json``.
* ``CannedGenerator`` reads the stand-ins in ``fixtures/canned/`` for ``--mode canned``.
  They were written for the offline demo and are labeled as not model output.
* ``FakeGenerator`` returns responses given in code, for tests.

Every generator returns a ``GenerationResult``. A refusal, an output cut off at
``max_tokens``, a response that fails the schema, or an API error after the SDK's
own retries is a failed result for that card, never a crash. Credentials, a wrong
model and a malformed request fail every card alike, so they raise
``GeneratorConfigError`` and stop the run.

How the model is called (``ClaudeGenerator.request``):

* model ``claude-opus-5-5`` unless ``TASKGEN_MODEL`` names another;
* ``output_config.effort`` set to ``high`` explicitly (this model defaults to
  medium); no ``thinking`` key, so thinking is adaptive, the model's default;
* structured output: ``output_config.format`` carries the JSON schema of
  ``taskgen.schema.GeneratedTests``, and the reply is validated with pydantic;
* server-side refusal fallback: ``fallbacks="default"`` with the beta
  ``server-side-fallback-2026-07-01`` on the beta messages surface;
* streaming, read with ``get_final_message()``, and ``max_tokens`` 32000;
* prompt caching: one system block (instructions plus context) marked
  ``cache_control: {"type": "ephemeral"}``; the card is the user message after it.
  Run cards one after another: parallel requests cannot read each other's cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from taskgen.cards import REF_PATTERN, Card, card_sort_key
from taskgen.context import REPO_ROOT, system_text
from taskgen.prompt import PROMPT_VERSION, card_message
from taskgen.schema import GeneratedTests, output_format

DEFAULT_MODEL = "claude-opus-5-5"
MODEL_ENV = "TASKGEN_MODEL"
EFFORT = "high"
MAX_TOKENS = 32000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACKS = "default"

RECORDINGS_DIR = REPO_ROOT / "recordings"
CANNED_DIR = REPO_ROOT / "fixtures" / "canned"
CANNED_NOTE = (
    "Stand-in for a model response, written for the offline demo and to test the pipeline "
    "without an API key. Not model output."
)
RECORDING_NOTE = "Recorded from the Anthropic API by taskgen generate --mode record."
RECORD_COMMAND = "ANTHROPIC_API_KEY=... taskgen generate --mode record --cards {card}"

MODES = ("replay", "record", "live", "canned")

Status = Literal["ok", "refusal", "max_tokens", "unexpected_stop", "invalid_output", "api_error"]


class GenerationError(Exception):
    """Base class for generator errors."""


class GeneratorConfigError(GenerationError):
    """A problem every card would hit (credentials, model, request shape). Stops the run."""


class APICallError(GenerationError):
    """The API call failed after the SDK's retries. Reported as a failed card."""


class RecordingError(GenerationError):
    """Replay cannot use the recording for a card."""

    def __init__(self, message: str, card: str, path: Path) -> None:
        super().__init__(message)
        self.card = card
        self.path = path


class MissingRecordingError(RecordingError):
    """No recording exists for the card."""


class StaleRecordingError(RecordingError):
    """A recording exists, but for a different request."""

    def __init__(self, message: str, card: str, path: Path, changed: list[str]) -> None:
        super().__init__(message, card, path)
        self.changed = changed


class MissingCannedResponseError(GenerationError):
    """No canned stand-in exists for the card."""


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, Price] = {
    # List prices at the time of writing; they may change. Cache writes (5-minute TTL)
    # cost 1.25 times the input price.
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=4.00 * 1.25),
}
PRICE_NOTE = "estimated from list prices at the time of writing, which may change"


@dataclass(frozen=True)
class Usage:
    """Token counts. ``input_tokens`` is the uncached part of the prompt only."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    def cost_usd(self, model: str) -> float | None:
        price = PRICES.get(model)
        if price is None:
            return None
        return (
            self.input_tokens * price.input
            + self.output_tokens * price.output
            + self.cache_read_input_tokens * price.cache_read
            + self.cache_creation_input_tokens * price.cache_write
        ) / 1_000_000

    @classmethod
    def from_sdk(cls, usage: Any) -> Usage:
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Usage:
        return cls(**{name: int(data.get(name, 0) or 0) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Attempt:
    """One model's attempt at the request. A refusal fallback adds a second attempt."""

    model: str
    usage: Usage

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "usage": self.usage.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Attempt:
        return cls(model=data["model"], usage=Usage.from_dict(data.get("usage") or {}))


def _cost(attempts: Iterable[Attempt]) -> float | None:
    total = 0.0
    for attempt in attempts:
        cost = attempt.usage.cost_usd(attempt.model)
        if cost is None:
            return None
        total += cost
    return total


@dataclass(frozen=True)
class ModelResponse:
    """What came back from the API, reduced to what taskgen uses. Cassettes store this."""

    text: str
    model: str
    stop_reason: str | None
    stop_details: dict[str, Any] | None = None
    request_id: str | None = None
    attempts: tuple[Attempt, ...] = ()
    served_by_fallback: bool = False

    @property
    def usage(self) -> Usage:
        total = Usage()
        for attempt in self.attempts:
            total = total + attempt.usage
        return total

    @property
    def cost_usd(self) -> float | None:
        return _cost(self.attempts) if self.attempts else None

    @classmethod
    def from_message(
        cls, message: Any, *, request_id: str | None, requested_model: str
    ) -> ModelResponse:
        """Read a (beta) Message from the SDK. Text blocks are read by type, never by position.

        A refusal's partial text is dropped here: it is never a usable answer.
        """
        text = (
            ""
            if message.stop_reason == "refusal"
            else "".join(
                block.text
                for block in message.content or []
                if getattr(block, "type", None) == "text"
            )
        )
        details = getattr(message, "stop_details", None)
        stop_details = (
            {
                "category": getattr(details, "category", None),
                "explanation": getattr(details, "explanation", None),
            }
            if details is not None
            else None
        )
        usage = message.usage
        iterations = list(getattr(usage, "iterations", None) or [])
        attempts = tuple(
            Attempt(model=getattr(it, "model", None) or requested_model, usage=Usage.from_sdk(it))
            for it in iterations
            if getattr(it, "type", None) in ("message", "fallback_message")
        ) or (Attempt(model=message.model or requested_model, usage=Usage.from_sdk(usage)),)
        return cls(
            text=text,
            model=message.model or requested_model,
            stop_reason=message.stop_reason,
            stop_details=stop_details,
            request_id=request_id,
            attempts=attempts,
            served_by_fallback=any(
                getattr(it, "type", None) == "fallback_message" for it in iterations
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "stop_details": self.stop_details,
            "request_id": self.request_id,
            "served_by_fallback": self.served_by_fallback,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelResponse:
        return cls(
            text=data.get("text") or "",
            model=data["model"],
            stop_reason=data.get("stop_reason"),
            stop_details=data.get("stop_details"),
            request_id=data.get("request_id"),
            attempts=tuple(Attempt.from_dict(item) for item in data.get("attempts") or []),
            served_by_fallback=bool(data.get("served_by_fallback")),
        )


@dataclass(frozen=True)
class GenerationResult:
    """The outcome for one card. ``output`` is set only when ``status`` is ``"ok"``."""

    card: str
    status: Status
    source: str
    output: GeneratedTests | None = None
    error: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    request_id: str | None = None
    served_by_fallback: bool = False
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    prompt_version: str = PROMPT_VERSION

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def used_model(self) -> bool:
        """True when a model answered, now or in a recording (not canned or fake)."""
        return self.source in ("live", "record", "replay")

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card,
            "status": self.status,
            "source": self.source,
            "output": self.output.model_dump() if self.output is not None else None,
            "error": self.error,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "request_id": self.request_id,
            "served_by_fallback": self.served_by_fallback,
            "usage": self.usage.to_dict(),
            "cost_usd": self.cost_usd,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GenerationResult:
        output = data.get("output")
        return cls(
            card=data["card"],
            status=data["status"],
            source=data["source"],
            output=GeneratedTests.model_validate(output) if output is not None else None,
            error=data.get("error"),
            model=data.get("model"),
            stop_reason=data.get("stop_reason"),
            request_id=data.get("request_id"),
            served_by_fallback=bool(data.get("served_by_fallback")),
            usage=Usage.from_dict(data.get("usage") or {}),
            cost_usd=data.get("cost_usd"),
            prompt_version=data.get("prompt_version", PROMPT_VERSION),
        )


def _schema_problem(exc: ValidationError) -> str:
    errors = exc.errors()
    first = errors[0]
    where = ".".join(str(part) for part in first.get("loc", ())) or "the top level"
    return f"{len(errors)} problem(s); first at {where}: {first.get('msg', 'invalid')}"


def parse_output(text: str, card: Card) -> tuple[GeneratedTests | None, str | None]:
    """Validate the model's JSON for ``card``. Returns ``(output, None)`` or ``(None, why)``."""
    if not text.strip():
        return None, "The response has no text to parse."
    try:
        output = GeneratedTests.model_validate_json(text)
    except ValidationError as exc:
        return None, f"The response does not match the output schema: {_schema_problem(exc)}"
    if output.card != card.ref:
        return None, f"The response is for card {output.card}, not {card.ref}."
    return output, None


def interpret(
    response: ModelResponse, card: Card, *, source: str, max_tokens: int = MAX_TOKENS
) -> GenerationResult:
    """Check ``stop_reason`` first, then validate the text. Never raises for a bad response."""
    common: dict[str, Any] = {
        "card": card.ref,
        "source": source,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "request_id": response.request_id,
        "served_by_fallback": response.served_by_fallback,
        "usage": response.usage,
        "cost_usd": response.cost_usd,
    }
    if response.stop_reason == "refusal":
        details = response.stop_details or {}
        category = details.get("category") or "no category given"
        explanation = f" {details['explanation']}" if details.get("explanation") else ""
        return GenerationResult(
            status="refusal",
            error=f"The model declined this card ({category}).{explanation}".strip(),
            **common,
        )
    if response.stop_reason == "max_tokens":
        return GenerationResult(
            status="max_tokens",
            error=(
                f"The output reached max_tokens ({max_tokens}) before the JSON was complete; "
                "the partial output was discarded."
            ),
            **common,
        )
    if response.stop_reason != "end_turn":
        return GenerationResult(
            status="unexpected_stop",
            error=f"Unexpected stop_reason {response.stop_reason!r}; expected 'end_turn'.",
            **common,
        )
    output, problem = parse_output(response.text, card)
    if output is None:
        return GenerationResult(status="invalid_output", error=problem, **common)
    return GenerationResult(status="ok", output=output, **common)


class Generator(Protocol):
    def generate(self, card: Card) -> GenerationResult: ...


class ClaudeGenerator:
    """Writes tests for a card with Claude, through the Anthropic SDK's beta messages API.

    ``client`` is any object with ``client.beta.messages.stream(**request)``; by default
    ``anthropic.Anthropic()``, created on first use, which reads ``ANTHROPIC_API_KEY``.
    Building a request never needs a client or a key.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        max_tokens: int = MAX_TOKENS,
        effort: str = EFFORT,
        fallbacks: bool = True,
        system: str | None = None,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self.system = system_text() if system is None else system
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def request(self, card: Card) -> dict[str, Any]:
        """The keyword arguments for ``client.beta.messages.stream`` for this card."""
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": card_message(card)}],
            "output_config": {"effort": self.effort, "format": output_format()},
        }
        if self.fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = FALLBACKS
        return request

    def call(self, request: Mapping[str, Any]) -> ModelResponse:
        """Send one request, streaming, and return the final message.

        The SDK retries connection errors, 408, 409, 429 and 5xx itself before any of
        these handlers sees them.
        """
        import anthropic  # imported on first use: the SDK takes seconds to import

        model = request["model"]
        try:
            with self.client.beta.messages.stream(**request) as stream:
                message = stream.get_final_message()
                request_id = getattr(stream, "request_id", None)
        except anthropic.AuthenticationError as exc:
            raise GeneratorConfigError(
                "The API rejected the credentials (401): check ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise GeneratorConfigError(
                f"The API key is not allowed to use {model} (403): {exc.message}"
            ) from exc
        except anthropic.NotFoundError as exc:
            raise GeneratorConfigError(
                f"Model {model} or the endpoint was not found (404): check {MODEL_ENV}."
            ) from exc
        except anthropic.BadRequestError as exc:
            raise GeneratorConfigError(
                f"The API rejected the request (400): {exc.message}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise APICallError(
                f"Rate limited (429) after the SDK's retries: {exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise APICallError(
                f"API error {exc.status_code} after the SDK's retries: {exc.message}"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise APICallError("The request timed out after the SDK's retries.") from exc
        except anthropic.APIConnectionError as exc:
            raise APICallError(f"Could not reach the API: {exc.message}") from exc
        except anthropic.APIError as exc:
            raise APICallError(f"API error: {exc.message}") from exc
        except anthropic.CredentialsError as exc:
            # A profile named by ANTHROPIC_PROFILE that does not exist, an unreadable
            # credentials file, a failed token refresh. Not an APIError, so caught here.
            raise GeneratorConfigError(f"Anthropic credentials could not be loaded: {exc}") from exc
        except anthropic.WorkloadIdentityError as exc:
            raise GeneratorConfigError(
                f"Workload identity federation could not get a token: {exc}"
            ) from exc
        except TypeError as exc:
            if "authentication" not in str(exc):
                raise
            raise GeneratorConfigError(
                "No Anthropic credentials: set ANTHROPIC_API_KEY to record or run live. "
                "Replay and canned modes need no key."
            ) from exc
        return ModelResponse.from_message(
            message,
            request_id=request_id or getattr(message, "_request_id", None),
            requested_model=model,
        )

    def generate(self, card: Card) -> GenerationResult:
        try:
            response = self.call(self.request(card))
        except APICallError as exc:
            return GenerationResult(
                card=card.ref, status="api_error", source="live", error=str(exc), model=self.model
            )
        return interpret(response, card, source="live", max_tokens=self.max_tokens)


def canonical_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of a request that decide the answer: model, system, messages, schema."""
    system = request.get("system", "")
    blocks = [system] if isinstance(system, str) else [block["text"] for block in system]
    return {
        "model": request["model"],
        "system": blocks,
        "messages": request["messages"],
        "schema": request["output_config"]["format"]["schema"],
    }


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_key(request: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical request. A recording replays only for the same key."""
    return _digest(canonical_request(request))


def request_parts(request: Mapping[str, Any]) -> dict[str, str]:
    """A digest per part, so a stale recording can say which part changed."""
    return {part: _digest(value) for part, value in canonical_request(request).items()}


@dataclass(frozen=True)
class Recording:
    """One recorded response, stored as ``recordings/<card>.json``."""

    card: str
    key: str
    parts: dict[str, str]
    response: ModelResponse
    settings: dict[str, Any]
    prompt_version: str = PROMPT_VERSION
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "_note": RECORDING_NOTE,
            "card": self.card,
            "key": self.key,
            "parts": self.parts,
            "settings": self.settings,
            "prompt_version": self.prompt_version,
            "recorded_at": self.recorded_at,
            "response": self.response.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recording:
        return cls(
            card=data["card"],
            key=data["key"],
            parts=dict(data.get("parts") or {}),
            response=ModelResponse.from_dict(data["response"]),
            settings=dict(data.get("settings") or {}),
            prompt_version=data.get("prompt_version", ""),
            recorded_at=data.get("recorded_at", ""),
        )


class Cassettes:
    """The recordings directory: one JSON file per card."""

    def __init__(self, directory: str | Path = RECORDINGS_DIR) -> None:
        self.directory = Path(directory)

    def path(self, ref: str) -> Path:
        if not re.fullmatch(REF_PATTERN, ref):
            raise ValueError(f"{ref!r} is not a card reference like 'FERN-2'")
        return self.directory / f"{ref}.json"

    def load(self, ref: str) -> Recording | None:
        path = self.path(ref)
        if not path.exists():
            return None
        try:
            return Recording.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise RecordingError(
                f"Recording {path} is unreadable ({type(exc).__name__}: {exc}).", ref, path
            ) from None

    def save(self, recording: Recording) -> Path:
        path = self.path(recording.card)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(recording.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8")
        return path


class CassetteGenerator:
    """``--mode replay`` (default), ``record`` or ``live``.

    * replay: answer from ``recordings/<card>.json``, matched by the request hash.
      Needs no key and makes no network call.
    * record: call the API and write the recording (refusals and cut-off outputs
      too, so a replay reproduces them).
    * live: call the API and keep nothing.
    """

    def __init__(
        self,
        mode: str = "replay",
        *,
        claude: ClaudeGenerator | None = None,
        cassettes: Cassettes | None = None,
        record_command: str = RECORD_COMMAND,
    ) -> None:
        if mode not in ("replay", "record", "live"):
            raise ValueError(f"mode must be replay, record or live, not {mode!r}")
        self.mode = mode
        self.claude = claude or ClaudeGenerator()
        self.cassettes = cassettes or Cassettes()
        self.record_command = record_command

    def generate(self, card: Card) -> GenerationResult:
        request = self.claude.request(card)
        if self.mode == "replay":
            return self._replay(card, request)
        try:
            response = self.claude.call(request)
        except APICallError as exc:
            return GenerationResult(
                card=card.ref,
                status="api_error",
                source=self.mode,
                error=str(exc),
                model=self.claude.model,
            )
        if self.mode == "record":
            self.cassettes.save(
                Recording(
                    card=card.ref,
                    key=request_key(request),
                    parts=request_parts(request),
                    response=response,
                    settings=_settings(request),
                    recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                )
            )
        return interpret(response, card, source=self.mode, max_tokens=self.claude.max_tokens)

    def _replay(self, card: Card, request: Mapping[str, Any]) -> GenerationResult:
        path = self.cassettes.path(card.ref)
        command = self.record_command.format(card=card.ref)
        recording = self.cassettes.load(card.ref)
        if recording is None:
            raise MissingRecordingError(
                f"No recording for {card.ref} (expected {path}). Replay mode answers only from "
                f"recordings. Record one with your own API key: {command}",
                card.ref,
                path,
            )
        if recording.key != request_key(request):
            now = request_parts(request)
            changed = [part for part, digest in now.items() if recording.parts.get(part) != digest]
            raise StaleRecordingError(
                f"The recording for {card.ref} ({path}) was made for a different request: "
                f"{', '.join(changed) or 'the request'} changed since it was recorded. "
                f"Record it again: {command}",
                card.ref,
                path,
                changed,
            )
        return interpret(
            recording.response, card, source="replay", max_tokens=self.claude.max_tokens
        )


def _settings(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_tokens": request["max_tokens"],
        "effort": request["output_config"].get("effort"),
        "fallbacks": request.get("fallbacks"),
        "betas": list(request.get("betas") or []),
    }


def _strip_notes(value: Any) -> Any:
    """Drop keys that start with an underscore, at any depth (notes for readers)."""
    if isinstance(value, Mapping):
        return {key: _strip_notes(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_notes(item) for item in value]
    return value


class CannedGenerator:
    """Stand-ins from ``fixtures/canned/<card>.json``, for ``--mode canned``.

    Each file is a response in the output schema plus ``"_note": CANNED_NOTE``, so it
    can never pass for model output. Any other key starting with ``_`` is a note for
    readers and is dropped before validation.
    """

    def __init__(self, directory: str | Path = CANNED_DIR) -> None:
        self.directory = Path(directory)

    def available(self) -> list[str]:
        """Refs that have a canned response, in board order."""
        refs = [path.stem for path in self.directory.glob("*.json")]
        return sorted((ref for ref in refs if re.fullmatch(REF_PATTERN, ref)), key=card_sort_key)

    def generate(self, card: Card) -> GenerationResult:
        path = self.directory / f"{card.ref}.json"
        if not path.exists():
            have = ", ".join(self.available()) or "none"
            raise MissingCannedResponseError(
                f"No canned response for {card.ref} in {self.directory} (canned: {have})."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GenerationError(f"{path} is not valid JSON: {exc}") from None
        if not isinstance(data, dict) or data.get("_note") != CANNED_NOTE:
            raise GenerationError(f'{path} must carry "_note": {json.dumps(CANNED_NOTE)}')
        output, problem = parse_output(json.dumps(_strip_notes(data)), card)
        if output is None:
            return GenerationResult(
                card=card.ref, status="invalid_output", source="canned", error=problem
            )
        return GenerationResult(card=card.ref, status="ok", source="canned", output=output)


Canned = GeneratedTests | Mapping[str, Any] | str | ModelResponse | GenerationResult


class FakeGenerator:
    """Answers from responses given in code. For tests; never calls the API.

    ``responses`` maps a card ref to a ``GeneratedTests``, a dict or JSON string in the
    output schema, a ``ModelResponse`` (to fake a refusal or a cut-off output), a
    finished ``GenerationResult``, or a function of the card returning one of those.
    ``calls`` lists the refs asked for, in order.
    """

    def __init__(
        self,
        responses: Mapping[str, Canned | Callable[[Card], Canned]] | None = None,
        *,
        default: Canned | Callable[[Card], Canned] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list[str] = []

    def generate(self, card: Card) -> GenerationResult:
        self.calls.append(card.ref)
        value = self.responses.get(card.ref, self.default)
        if value is None:
            raise KeyError(f"FakeGenerator has no response for {card.ref}")
        if callable(value):
            value = value(card)
        if isinstance(value, GenerationResult):
            return value
        if isinstance(value, ModelResponse):
            return interpret(value, card, source="fake")
        if isinstance(value, GeneratedTests):
            text = value.model_dump_json()
        elif isinstance(value, str):
            text = value
        else:
            text = json.dumps(value)
        response = ModelResponse(text=text, model="fake", stop_reason="end_turn")
        return interpret(response, card, source="fake")


def make_generator(
    mode: str = "replay",
    *,
    recordings: str | Path = RECORDINGS_DIR,
    canned: str | Path = CANNED_DIR,
    client: Any = None,
    model: str | None = None,
    record_command: str = RECORD_COMMAND,
) -> Generator:
    """The generator for ``taskgen generate --mode <mode>``."""
    if mode == "canned":
        return CannedGenerator(canned)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    return CassetteGenerator(
        mode,
        claude=ClaudeGenerator(client, model=model),
        cassettes=Cassettes(recordings),
        record_command=record_command,
    )


def total_usage(results: Iterable[GenerationResult]) -> Usage:
    total = Usage()
    for result in results:
        total = total + result.usage
    return total


def cost_line(results: Iterable[GenerationResult]) -> str | None:
    """One line for a report: tokens, cache reads and estimated cost. None without a model."""
    used = [result for result in results if result.used_model]
    if not used:
        return None
    # A card whose call failed (an API error after the SDK's retries) carries no usage:
    # no tokens were counted, so it adds nothing to the estimate rather than making it
    # unknown. Only tokens from a model with no list price make the cost unknown.
    answered = [result for result in used if result.usage != Usage()]
    failed = len(used) - len(answered)
    usage = total_usage(answered)
    models = sorted({result.model for result in answered if result.model})
    costs = [result.cost_usd for result in answered]
    if any(cost is None for cost in costs):
        estimate = "cost unknown (no list price for a model used)"
    else:
        estimate = f"about ${sum(c for c in costs if c is not None):.2f} ({PRICE_NOTE})"
    if answered and all(result.source == "replay" for result in answered):
        estimate += "; replayed from recordings, so this run spent nothing"
    cards = f"{len(answered)} card(s)"
    if failed:
        cards += f", {failed} more with no response"
    return (
        f"{', '.join(models) or 'model'} · {cards} · "
        f"{usage.input_tokens:,} input, {usage.cache_read_input_tokens:,} cache read, "
        f"{usage.cache_creation_input_tokens:,} cache write, {usage.output_tokens:,} output "
        f"tokens · {estimate}"
    )
