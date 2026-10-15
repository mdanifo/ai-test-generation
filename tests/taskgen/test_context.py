"""The cached system block and the user message.

The system block is the instructions, the Front Desk OpenAPI document and the
harness guide. It must be byte-stable (it is the cached prefix of every request)
and it must never reveal the seeded bugs.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from frontdesk import bugs
from frontdesk.main import app, create_app
from taskgen.context import (
    HARNESS_GUIDE,
    ContextLeakError,
    context_block,
    find_leaks,
    openapi_document,
    system_text,
)
from taskgen.prompt import INSTRUCTIONS, card_message
from taskgen.sources import FileSource

REPO = Path(__file__).resolve().parents[2]
CARDS = FileSource(REPO / "examples" / "fernhill-board.json").cards()


def test_the_system_block_is_the_instructions_then_the_two_documents():
    text = system_text()

    assert text.startswith(INSTRUCTIONS)
    assert text[len(INSTRUCTIONS) :] == "\n" + context_block()
    assert context_block().startswith("<openapi>\n")
    assert context_block().endswith("</harness_guide>\n")


def test_the_openapi_document_has_sorted_keys_and_is_the_apps():
    document = openapi_document()

    assert json.loads(document) == app.openapi()
    assert document == json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    assert f"<openapi>\n{document}\n</openapi>" in system_text()


def test_the_harness_guide_is_included_in_full():
    guide = HARNESS_GUIDE.read_text(encoding="utf-8").strip()

    assert f"<harness_guide>\n{guide}\n</harness_guide>" in system_text()


def test_the_system_block_is_the_same_bytes_every_time():
    assert system_text() == system_text()
    assert context_block() == context_block(openapi_document(create_app()))


def test_the_system_block_has_no_date_in_it():
    assert date.today().isoformat() not in system_text()


@pytest.fixture(scope="module")
def clean_context() -> str:
    return context_block(openapi_document(create_app()))


@pytest.mark.parametrize("bug", sorted(bugs.CATALOG))
def test_the_context_is_the_same_whichever_build_is_running(monkeypatch, clean_context, bug):
    monkeypatch.setenv(bugs.ENV_VAR, bug)

    assert context_block(openapi_document(create_app())) == clean_context


def test_no_seeded_bug_name_ever_appears_in_the_context():
    text = system_text()

    assert find_leaks(text) == []
    assert not any(name in text.lower() for name in bugs.CATALOG)
    assert not any(description.lower() in text.lower() for description in bugs.CATALOG.values())
    assert "seeded" not in text.lower()
    assert "bug" not in text.lower()


def test_no_seeded_bug_name_appears_in_any_sample_card_message():
    messages = [card_message(card) for card in CARDS]

    assert [find_leaks(message) for message in messages] == [[] for _ in CARDS]


def test_the_context_carries_no_app_source():
    text = system_text()

    assert "def create_app" not in text
    assert "bugs.active" not in text
    assert "import " not in context_block().split("<harness_guide>")[0]


@pytest.mark.parametrize("planted", ["double-booking", "SEEDED_BUG", "Ssml-Unescaped"])
def test_a_guide_that_mentions_a_seeded_bug_stops_the_run(planted):
    guide = HARNESS_GUIDE.read_text(encoding="utf-8") + f"\nAlso try {planted}.\n"

    with pytest.raises(ContextLeakError, match="mentions the seeded bugs"):
        system_text(guide=guide)


def test_the_instructions_ask_for_one_to_three_tests_and_an_untestable_list():
    assert "one to three tests per testable criterion" in INSTRUCTIONS
    assert "`untestable`" in INSTRUCTIONS
    assert "Never invent a helper, fixture, parameter, field or event" in INSTRUCTIONS
    assert "@pytest.mark.criterion" in INSTRUCTIONS


def test_the_instructions_use_no_long_dashes():
    assert chr(0x2014) not in INSTRUCTIONS  # em dash
    assert chr(0x2013) not in INSTRUCTIONS  # en dash


def test_the_card_message_lists_every_criterion_in_order():
    card = next(card for card in CARDS if card.ref == "FERN-2")
    message = card_message(card)

    assert message.startswith('<card ref="FERN-2">\nTitle: ' + card.title)
    assert "List: Done. The behavior is built" in message
    assert "Labels: LLM, Privacy" in message
    assert [line.split(":")[0] for line in message.splitlines() if line.startswith("AC-")] == [
        "AC-1",
        "AC-2",
        "AC-3",
        "AC-4",
        "AC-5",
    ]
    assert f"AC-3: {card.criterion('AC-3').text}" in message
    assert message.endswith('@pytest.mark.criterion("FERN-2/AC-<k>").')


def test_a_card_that_is_not_done_says_it_may_not_be_built():
    card = next(card for card in CARDS if card.ref == "FERN-5")

    assert "List: In Progress. The behavior may not be built yet" in card_message(card)
