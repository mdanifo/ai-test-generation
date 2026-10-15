"""Where cards come from: the task board's REST API, or a snapshot file.

``TaskBoardSource`` reads a board with GET requests only:

* ``GET {TASKS_URL}/api/v1/boards/{slug}`` for the lists and their cards
  (``list.role`` is todo, doing, done or hold), then
* ``GET {TASKS_URL}/api/v1/cards/{ref}`` for each card's checklists. The checklist
  named "Acceptance criteria" (any case) holds the criteria, in position order.

Auth is a Cloudflare Access service token in ``CF_ACCESS_CLIENT_ID`` and
``CF_ACCESS_CLIENT_SECRET``, sent as the ``CF-Access-Client-Id`` and
``CF-Access-Client-Secret`` headers. Their values are never logged, printed or
put in an error message.

``FileSource`` reads a snapshot, the JSON that ``TaskBoardSource.snapshot()``
writes (``taskgen snapshot``)::

    {
      "board": {"slug", "name", "key", "kind", "accent"},
      "labels": {"<label>": "<color>", ...},
      "cards": [
        {"ref": "FERN-2", "title": ..., "list": "done", "labels": [...],
         "description": ..., "criteria": [{"text": ..., "done": true}, ...]}
      ]
    }

A criterion may also be a plain string (not done), and a card without ``ref``
takes ``<key>-<n>`` from its position, so a snapshot written without a board stays short.

Both sources return only cards that have at least one criterion. A snapshot keeps
every card, so it can stand in for the whole board.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from taskgen.cards import REF_PATTERN, Card, card_sort_key, number_criteria

URL_ENV = "TASKS_URL"
CLIENT_ID_ENV = "CF_ACCESS_CLIENT_ID"
CLIENT_SECRET_ENV = "CF_ACCESS_CLIENT_SECRET"
CRITERIA_CHECKLIST = "acceptance criteria"
API_PREFIX = "/api/v1"

_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class SourceError(Exception):
    """A board or snapshot could not be read. The message says what to fix."""


class CardSource(Protocol):
    """What the generator and the gate need from a source."""

    def cards(self) -> list[Card]: ...

    def card(self, ref: str) -> Card: ...


def dump_snapshot(snapshot: Mapping[str, Any]) -> str:
    """The snapshot as written to disk: stable, indented, UTF-8, ending in a newline."""
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"


def write_snapshot(source: TaskBoardSource, path: str | Path) -> Path:
    """Write ``source.snapshot()`` to ``path`` and return the path.

    The output has no timestamps or ids, so snapshotting an unchanged board twice
    gives the same bytes.
    """
    target = Path(path)
    data = source.snapshot()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_snapshot(data), encoding="utf-8")
    return target


def cards_from_snapshot(data: Mapping[str, Any], where: str = "snapshot") -> list[Card]:
    """Turn snapshot JSON into cards, keeping only those with criteria, in ref order."""
    if not isinstance(data, Mapping) or not isinstance(data.get("cards"), list):
        raise SourceError(f"{where}: expected an object with a 'cards' list")
    board = data.get("board") or {}
    key = board.get("key") if isinstance(board, Mapping) else None
    cards: list[Card] = []
    seen: set[str] = set()
    for position, raw in enumerate(data["cards"], start=1):
        if not isinstance(raw, Mapping):
            raise SourceError(f"{where}: card {position} is not an object")
        ref = raw.get("ref") or (f"{key}-{position}" if key else None)
        if not ref:
            raise SourceError(f"{where}: card {position} has no 'ref' and the board has no 'key'")
        if ref in seen:
            raise SourceError(f"{where}: {ref} appears twice")
        seen.add(ref)
        items = raw.get("criteria") or []
        if not isinstance(items, list):
            raise SourceError(f"{where}: {ref} 'criteria' is not a list")
        if not items:
            continue
        try:
            criteria = number_criteria(_criterion_item(item, ref, where) for item in items)
            card = Card(
                ref=ref,
                title=raw.get("title") or "",
                description=raw.get("description") or "",
                list_role=raw.get("list") or "",
                labels=tuple(raw.get("labels") or ()),
                criteria=criteria,
            )
        except ValidationError as exc:
            raise SourceError(f"{where}: {ref} is not a valid card: {_first_error(exc)}") from None
        cards.append(card)
    return sorted(cards, key=lambda card: card_sort_key(card.ref))


def _criterion_item(item: Any, ref: str, where: str) -> str | tuple[str, bool]:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and isinstance(item.get("text"), str):
        return item["text"], bool(item.get("done", False))
    raise SourceError(f"{where}: {ref} has a criterion that is neither text nor {{'text', 'done'}}")


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    where = ".".join(str(part) for part in error.get("loc", ()))
    return f"{where}: {error.get('msg', 'invalid')}" if where else str(error.get("msg"))


def _find(cards: list[Card], ref: str, where: str) -> Card:
    for card in cards:
        if card.ref == ref:
            return card
    known = ", ".join(card.ref for card in cards) or "none"
    raise SourceError(f"{where} has no card {ref} with acceptance criteria (cards: {known})")


class FileSource:
    """Cards from a snapshot file written by ``taskgen snapshot``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"FileSource({str(self.path)!r})"

    def snapshot(self) -> dict[str, Any]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SourceError(f"snapshot {self.path} not found") from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceError(f"snapshot {self.path} is not valid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise SourceError(f"snapshot {self.path}: expected a JSON object")
        return data

    def cards(self) -> list[Card]:
        return cards_from_snapshot(self.snapshot(), where=str(self.path))

    def card(self, ref: str) -> Card:
        return _find(self.cards(), ref, str(self.path))


class TaskBoardSource:
    """Cards read live from the task board's REST API, with GET requests only.

    ``post_comment`` is the single exception, and only ``taskgen comment --yes``
    calls it.
    """

    def __init__(
        self,
        board: str,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not _SLUG.fullmatch(board or ""):
            raise SourceError(f"{board!r} is not a board slug like 'fernhill-front-desk'")
        env = os.environ if env is None else env
        base_url = (base_url or env.get(URL_ENV) or "").strip()
        if not base_url:
            raise SourceError(f"{URL_ENV} is not set: give the task board's base URL")
        base_url = base_url.rstrip("/").removesuffix(API_PREFIX)
        client_id = client_id or env.get(CLIENT_ID_ENV) or None
        client_secret = client_secret or env.get(CLIENT_SECRET_ENV) or None
        if bool(client_id) != bool(client_secret):
            raise SourceError(f"set both {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV}, or neither")
        headers = {"Accept": "application/json", "User-Agent": "taskgen/0.1"}
        if client_id and client_secret:
            headers["CF-Access-Client-Id"] = client_id
            headers["CF-Access-Client-Secret"] = client_secret
        self.board = board
        self.base_url = base_url
        self._client = httpx.Client(
            base_url=base_url + API_PREFIX,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return f"TaskBoardSource({self.board!r}, base_url={self.base_url!r})"

    def __enter__(self) -> TaskBoardSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        where = f"{method} {API_PREFIX}{path}"
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise SourceError(f"{where} failed: {type(exc).__name__}: {exc}") from None
        status = response.status_code
        if 300 <= status < 400 or status in (401, 403):
            raise SourceError(
                f"{where} was refused ({status}). Cloudflare Access wants a login: check "
                f"{CLIENT_ID_ENV} and {CLIENT_SECRET_ENV}."
            )
        if status == 404:
            raise SourceError(f"{where} returned 404: not found on {self.base_url}")
        if status not in (200, 201):
            raise SourceError(f"{where} returned {status}")
        if "json" not in response.headers.get("content-type", ""):
            raise SourceError(
                f"{where} did not return JSON ({response.headers.get('content-type') or 'no type'})"
            )
        try:
            return response.json()
        except ValueError:
            raise SourceError(f"{where} returned malformed JSON") from None

    def _board_payload(self) -> dict[str, Any]:
        payload = self._request("GET", f"/boards/{self.board}")
        if not isinstance(payload, dict) or not isinstance(payload.get("lists"), list):
            raise SourceError(f"GET {API_PREFIX}/boards/{self.board}: expected a board with lists")
        return payload

    def _card_detail(self, ref: str) -> dict[str, Any]:
        if not re.fullmatch(REF_PATTERN, ref):
            raise SourceError(f"{ref!r} is not a card reference like 'FERN-2'")
        detail = self._request("GET", f"/cards/{ref}")
        if not isinstance(detail, dict):
            raise SourceError(f"GET {API_PREFIX}/cards/{ref}: expected an object")
        return detail

    def board_refs(self) -> list[str]:
        """The refs of the board's open cards, in number order."""
        return self._refs_in(self._board_payload())

    def snapshot(self) -> dict[str, Any]:
        """The whole board in the FileSource format, every card included."""
        payload = self._board_payload()
        project = payload.get("project") or {}
        board = {
            "slug": payload.get("slug", self.board),
            "name": payload.get("name", ""),
            "key": payload.get("key", ""),
            "kind": payload.get("kind") or project.get("kind"),
            "accent": payload.get("accent"),
        }
        labels = {
            label["name"]: label.get("color")
            for label in sorted(payload.get("labels") or [], key=lambda label: label["name"])
        }
        cards = [_snapshot_card(self._card_detail(ref)) for ref in self._refs_in(payload)]
        return {"board": board, "labels": labels, "cards": cards}

    @staticmethod
    def _refs_in(payload: Mapping[str, Any]) -> list[str]:
        refs = [
            card["ref"]
            for lst in payload.get("lists") or []
            if not lst.get("archived")
            for card in lst.get("cards") or []
            if card.get("ref") and not card.get("archived_at")
        ]
        return sorted(refs, key=card_sort_key)

    def cards(self) -> list[Card]:
        """Cards that have an Acceptance criteria checklist with at least one item."""
        return cards_from_snapshot(self.snapshot(), where=f"board {self.board}")

    def card(self, ref: str) -> Card:
        """One card, read with a single GET."""
        found = cards_from_snapshot(
            {"cards": [_snapshot_card(self._card_detail(ref))]}, where=f"board {self.board}"
        )
        if not found:
            raise SourceError(f"{ref} has no Acceptance criteria checklist with items")
        return found[0]

    def post_comment(self, ref: str, body: str) -> dict[str, Any]:
        """Add a comment to a card: ``POST /api/v1/cards/{ref}/comments`` with ``{"body"}``."""
        if not re.fullmatch(REF_PATTERN, ref):
            raise SourceError(f"{ref!r} is not a card reference like 'FERN-2'")
        if not body.strip():
            raise SourceError("a comment needs a body")
        result = self._request("POST", f"/cards/{ref}/comments", json={"body": body})
        return result if isinstance(result, dict) else {}


def _snapshot_card(detail: Mapping[str, Any]) -> dict[str, Any]:
    checklist = next(
        (
            checklist
            for checklist in sorted(
                detail.get("checklists") or [], key=lambda c: c.get("position", 0)
            )
            if str(checklist.get("name", "")).strip().lower() == CRITERIA_CHECKLIST
        ),
        None,
    )
    items = sorted((checklist or {}).get("items") or [], key=lambda item: item.get("position", 0))
    return {
        "ref": detail.get("ref"),
        "title": detail.get("title", ""),
        "list": (detail.get("list") or {}).get("role", ""),
        "labels": sorted(label["name"] for label in detail.get("labels") or []),
        "description": detail.get("description") or "",
        "criteria": [{"text": item["text"], "done": bool(item.get("done"))} for item in items],
    }
