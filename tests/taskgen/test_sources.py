"""Card sources: the snapshot file, and the board API through a fake server.

The board API is served by httpx.MockTransport from the committed snapshot, in the
shape the real API returns. No test here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from taskgen.cards import Card, Criterion, number_criteria, split_marker
from taskgen.sources import (
    FileSource,
    SourceError,
    TaskBoardSource,
    cards_from_snapshot,
    dump_snapshot,
    write_snapshot,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "fernhill-board.json"
CLIENT_ID = "test-client-id-0000"
CLIENT_SECRET = "test-client-secret-0000"
ENV = {
    "TASKS_URL": "https://tasks.example.test",
    "CF_ACCESS_CLIENT_ID": CLIENT_ID,
    "CF_ACCESS_CLIENT_SECRET": CLIENT_SECRET,
}
DONE_CARDS = ["FERN-2", "FERN-3", "FERN-4", "FERN-8", "FERN-11", "FERN-13"]


def example() -> dict[str, Any]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


LIST_NAMES = {"todo": "To Do", "doing": "In Progress", "hold": "On Hold", "done": "Done"}


def board_api(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The board and card payloads the real API would return for this snapshot.

    Cards sit in their lists out of number order, checklist items come back out of
    position order, every card has a second checklist, and one archived card is on
    the board, so the source has to sort and filter the way it does against the
    real API.
    """
    labels = [
        {"id": f"lbl_{k}", "name": name, "color": color}
        for k, (name, color) in enumerate(reversed(snapshot["labels"].items()))
    ]
    colors = dict(snapshot["labels"])
    details: dict[str, dict[str, Any]] = {}
    lists: dict[str, list[dict[str, Any]]] = {role: [] for role in LIST_NAMES}
    for card in reversed(snapshot["cards"]):
        items = [
            {"id": f"itm_{k}", "text": item["text"], "done": item["done"], "position": k * 1000.0}
            for k, item in enumerate(card["criteria"], start=1)
        ]
        checklists = [
            {"id": "chk_notes", "name": "Notes", "position": 1.0, "items": [{"text": "x"}]},
        ]
        if card["criteria"]:
            checklists.append(
                {
                    "id": "chk_ac",
                    "name": "ACCEPTANCE CRITERIA",
                    "position": 2.0,
                    "items": list(reversed(items)),
                }
            )
        card_labels = [{"name": name, "color": colors[name]} for name in reversed(card["labels"])]
        summary = {
            "ref": card["ref"],
            "title": card["title"],
            "labels": card_labels,
            "archived_at": None,
        }
        lists[card["list"]].append(summary)
        details[card["ref"]] = {
            **summary,
            "description": card["description"],
            "list": {"role": card["list"], "name": LIST_NAMES[card["list"]]},
            "checklists": checklists,
            "comments": [],
        }
    lists["todo"].append({"ref": "FERN-99", "title": "Archived", "archived_at": "2026-10-01"})
    board = {
        "slug": snapshot["board"]["slug"],
        "name": snapshot["board"]["name"],
        "key": snapshot["board"]["key"],
        "accent": snapshot["board"]["accent"],
        "project": {"kind": snapshot["board"]["kind"]},
        "labels": labels,
        "lists": [
            {"role": role, "name": name, "archived": False, "cards": lists[role]}
            for role, name in LIST_NAMES.items()
        ],
    }
    return board, details


class FakeBoard:
    """A MockTransport handler for the board API. Records every request."""

    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.board, self.details = board_api(snapshot or example())
        self.requests: list[httpx.Request] = []
        self.comments: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/boards/fernhill-front-desk":
            return httpx.Response(200, json=self.board)
        if request.method == "GET" and path.startswith("/api/v1/cards/"):
            ref = path.removeprefix("/api/v1/cards/")
            if ref in self.details:
                return httpx.Response(200, json=self.details[ref])
        if request.method == "POST" and path.endswith("/comments"):
            ref = path.split("/")[-2]
            self.comments.append((ref, json.loads(request.content)))
            return httpx.Response(201, json={"id": "cmt_1", "body": "ok"})
        return httpx.Response(404, json={"error": "not_found"})

    def source(self, **kwargs: Any) -> TaskBoardSource:
        return TaskBoardSource(
            "fernhill-front-desk", env=ENV, transport=httpx.MockTransport(self), **kwargs
        )


def responding(response: httpx.Response) -> TaskBoardSource:
    return TaskBoardSource(
        "fernhill-front-desk", env=ENV, transport=httpx.MockTransport(lambda _: response)
    )


def test_the_example_snapshot_is_the_whole_sample_board_in_order():
    data = example()

    assert data["board"]["slug"] == "fernhill-front-desk"
    assert [card["ref"] for card in data["cards"]] == [f"FERN-{n}" for n in range(1, 17)]
    assert data["cards"][0]["criteria"] == []
    assert sum(len(card["criteria"]) for card in data["cards"]) == 63


def test_the_example_snapshot_ticks_criteria_exactly_on_the_done_cards():
    data = example()

    assert [card["ref"] for card in data["cards"] if card["list"] == "done"] == DONE_CARDS
    assert all(
        item["done"] is (card["list"] == "done")
        for card in data["cards"]
        for item in card["criteria"]
    )


def test_the_example_snapshot_is_written_the_way_write_snapshot_writes():
    assert EXAMPLE.read_text(encoding="utf-8") == dump_snapshot(example())


def test_file_source_returns_the_cards_that_have_criteria():
    cards = FileSource(EXAMPLE).cards()

    assert [card.ref for card in cards] == [f"FERN-{n}" for n in range(2, 17)]
    assert [card.ref for card in cards if card.is_done] == DONE_CARDS


def test_file_source_numbers_criteria_in_checklist_order():
    card = FileSource(EXAMPLE).card("FERN-2")

    assert card.title == "Verify the caller before sharing any appointment details"
    assert card.list_role == "done"
    assert card.labels == ("LLM", "Privacy")
    assert card.criterion_ids == ("AC-1", "AC-2", "AC-3", "AC-4", "AC-5")
    assert card.criterion("AC-3").text.startswith("Given the date of birth is said in a different")


def test_file_source_names_the_cards_it_has_when_one_is_missing():
    with pytest.raises(SourceError, match=r"no card FERN-1 .*FERN-2, FERN-3"):
        FileSource(EXAMPLE).card("FERN-1")


def test_a_minimal_snapshot_is_accepted():
    data = {
        "board": {"key": "DEMO"},
        "cards": [
            {"title": "About", "list": "todo", "criteria": []},
            {"title": "Greet", "list": "done", "criteria": ["Says hello.", "Says goodbye."]},
            {"ref": "DEMO-7", "title": "Ticked", "list": "doing", "criteria": [{"text": "Done."}]},
        ],
    }

    cards = cards_from_snapshot(data)

    assert [card.ref for card in cards] == ["DEMO-2", "DEMO-7"]
    assert cards[0].criteria == (
        Criterion(id="AC-1", text="Says hello."),
        Criterion(id="AC-2", text="Says goodbye."),
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"cards": "no"}, "a 'cards' list"),
        ({"cards": [{"title": "x", "criteria": ["a"]}]}, "has no 'ref'"),
        (
            {"cards": [{"ref": "A-1", "criteria": ["a"]}, {"ref": "A-1", "criteria": ["b"]}]},
            "A-1 appears twice",
        ),
        ({"cards": [{"ref": "A-1", "list": "todo", "criteria": [7]}]}, "neither text"),
        ({"cards": [{"ref": "bad ref", "list": "todo", "criteria": ["a"]}]}, "not a valid card"),
    ],
)
def test_a_malformed_snapshot_says_what_is_wrong(data, message):
    with pytest.raises(SourceError, match=message):
        cards_from_snapshot(data)


def test_a_missing_or_broken_snapshot_file_is_a_source_error(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(SourceError, match="not found"):
        FileSource(tmp_path / "missing.json").cards()
    with pytest.raises(SourceError, match="not valid JSON"):
        FileSource(broken).cards()


def test_card_helpers():
    assert split_marker("FERN-2/AC-3") == ("FERN-2", "AC-3")
    assert [c.id for c in number_criteria(["a", ("b", True)])] == ["AC-1", "AC-2"]
    with pytest.raises(ValueError, match="not a criterion marker"):
        split_marker("FERN-2 AC-3")
    card = Card(ref="X-1", title="t", list_role="todo", criteria=number_criteria(["a"]))
    with pytest.raises(KeyError, match="X-1 has no criterion 'AC-2'"):
        card.criterion("AC-2")


def test_the_board_source_snapshot_matches_the_committed_example():
    fake = FakeBoard()

    assert fake.source().snapshot() == example()


def test_snapshot_writes_the_committed_example_byte_for_byte(tmp_path):
    out = write_snapshot(FakeBoard().source(), tmp_path / "board.json")

    assert out.read_bytes() == EXAMPLE.read_bytes()


def test_the_board_source_only_sends_gets_with_the_access_headers():
    fake = FakeBoard()

    cards = fake.source().cards()

    assert cards == FileSource(EXAMPLE).cards()
    assert {request.method for request in fake.requests} == {"GET"}
    assert all(
        request.headers["CF-Access-Client-Id"] == CLIENT_ID
        and request.headers["CF-Access-Client-Secret"] == CLIENT_SECRET
        for request in fake.requests
    )
    assert [request.url.path for request in fake.requests] == [
        "/api/v1/boards/fernhill-front-desk",
        *[f"/api/v1/cards/FERN-{n}" for n in range(1, 17)],
    ]


def test_one_card_is_one_get():
    fake = FakeBoard()

    card = fake.source().card("FERN-8")

    assert card == FileSource(EXAMPLE).card("FERN-8")
    assert [request.url.path for request in fake.requests] == ["/api/v1/cards/FERN-8"]


def test_a_card_without_acceptance_criteria_is_refused_by_name():
    with pytest.raises(SourceError, match="FERN-1 has no Acceptance criteria"):
        FakeBoard().source().card("FERN-1")


def test_a_base_url_ending_in_the_api_prefix_is_accepted():
    fake = FakeBoard()

    fake.source(base_url="https://tasks.example.test/api/v1/").board_refs()

    assert str(fake.requests[0].url) == (
        "https://tasks.example.test/api/v1/boards/fernhill-front-desk"
    )


def test_no_access_headers_are_sent_without_a_token():
    fake = FakeBoard()
    env = {"TASKS_URL": "http://localhost:8000"}

    TaskBoardSource(
        "fernhill-front-desk", env=env, transport=httpx.MockTransport(fake)
    ).board_refs()

    assert "CF-Access-Client-Id" not in fake.requests[0].headers


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"location": "https://example.cloudflareaccess.com/login"}),
        httpx.Response(403, text="Forbidden"),
    ],
)
def test_an_access_refusal_says_which_variables_to_check(response):
    with pytest.raises(SourceError, match="CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET") as err:
        responding(response).snapshot()

    assert CLIENT_SECRET not in str(err.value)
    assert CLIENT_ID not in str(err.value)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, json={"error": "not_found"}), "returned 404"),
        (httpx.Response(500, text="boom"), "returned 500"),
        (httpx.Response(200, html="<html>login</html>"), "did not return JSON"),
        (httpx.Response(200, json=[1, 2]), "expected a board with lists"),
    ],
)
def test_other_failures_are_source_errors(response, message):
    with pytest.raises(SourceError, match=message):
        responding(response).snapshot()


def test_a_network_failure_is_a_source_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    source = TaskBoardSource("fernhill-front-desk", env=ENV, transport=httpx.MockTransport(refuse))

    with pytest.raises(SourceError, match="ConnectError"):
        source.board_refs()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "TASKS_URL is not set"),
        ({"TASKS_URL": "https://x.test", "CF_ACCESS_CLIENT_ID": "id"}, "set both"),
    ],
)
def test_missing_settings_are_named(env, message):
    with pytest.raises(SourceError, match=message):
        TaskBoardSource("fernhill-front-desk", env=env)


def test_slugs_and_refs_are_checked_before_they_reach_a_url():
    fake = FakeBoard()

    with pytest.raises(SourceError, match="not a board slug"):
        TaskBoardSource("../admin", env=ENV)
    with pytest.raises(SourceError, match="not a card reference"):
        fake.source().card("FERN-2/../../boards")
    assert fake.requests == []


def test_the_secret_never_appears_in_the_source_repr():
    source = FakeBoard().source()

    assert CLIENT_SECRET not in repr(source)
    assert CLIENT_ID not in repr(source)


def test_post_comment_posts_one_comment_to_the_fake_server():
    fake = FakeBoard()

    fake.source().post_comment("FERN-2", "3 tests accepted.")

    assert [(r.method, r.url.path) for r in fake.requests] == [
        ("POST", "/api/v1/cards/FERN-2/comments")
    ]
    assert fake.comments == [("FERN-2", {"body": "3 tests accepted."})]


def test_post_comment_refuses_an_empty_body():
    fake = FakeBoard()

    with pytest.raises(SourceError, match="needs a body"):
        fake.source().post_comment("FERN-2", "  ")
    assert fake.requests == []
