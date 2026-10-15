"""tests/generated/ holds only what ``taskgen accept`` wrote, still passing the static checks.

ruff does not reformat that directory (the files stay exactly as the gate ran them), so
this check is what keeps an edit there from going unnoticed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from taskgen.accept import accept, check_suite
from taskgen.gate import RunResult, gate_cards
from taskgen.generate import CannedGenerator
from taskgen.sources import FileSource

REPO = Path(__file__).resolve().parents[2]
CARDS = {card.ref: card for card in FileSource(REPO / "examples" / "fernhill-board.json").cards()}
ZERO = "test_pressing_zero_transfers_in_that_turn"


def catches_everything(files: Sequence[str], bug: str | None, label: str) -> dict[str, RunResult]:
    if bug is None:
        return dict.fromkeys(files, RunResult(outcome="passed", cases={"t": "passed"}))
    caught = RunResult(outcome="failed", exception="AssertionError", cases={"t": "failed"})
    return dict.fromkeys(files, caught)


@pytest.fixture
def accepted(tmp_path: Path) -> Path:
    """FERN-8's canned tests gated with a fake runner, then accepted into a scratch suite."""
    card = CARDS["FERN-8"]
    output = CannedGenerator().generate(card).output
    gate_cards(
        [(card, output)],
        proposed_dir=tmp_path / "proposed",
        bugs=["transfer-asks-why"],
        runner=catches_everything,
    )
    generated = tmp_path / "generated"
    accept("FERN-8", proposed_dir=tmp_path / "proposed", generated_dir=generated)
    return generated


def test_the_committed_generated_suite_passes_its_checks():
    assert check_suite(REPO / "tests" / "generated", CARDS) == []


def test_what_accept_writes_passes_the_check(accepted):
    assert len(list(accepted.rglob("*.py"))) == 5
    assert check_suite(accepted, CARDS) == []


def test_an_edit_that_breaks_a_rule_is_reported(accepted):
    path = accepted / "FERN-8" / f"{ZERO}.py"
    path.write_text(path.read_text().replace("import pytest\n", "import pytest\nimport time\n"))

    problems = check_suite(accepted, CARDS)

    assert len(problems) == 1
    assert problems[0].startswith(f"FERN-8/{ZERO}.py: line ")
    assert "imports time, which is not allowed" in problems[0]


def test_a_file_without_the_header_is_reported(accepted):
    path = accepted / "FERN-8" / f"{ZERO}.py"
    path.write_text(path.read_text().partition("\n\n")[2])

    assert check_suite(accepted, CARDS) == [
        f"FERN-8/{ZERO}.py: missing the header `taskgen accept` writes"
    ]


def test_a_renamed_file_no_longer_matches_its_function(accepted):
    path = accepted / "FERN-8" / f"{ZERO}.py"
    path.rename(path.with_name("test_renamed.py"))

    problems = check_suite(accepted, CARDS)

    assert len(problems) == 1
    assert problems[0].startswith("FERN-8/test_renamed.py: ")
    assert "[name]" in problems[0]


def test_a_file_outside_a_card_folder_is_reported(accepted):
    (accepted / "test_loose.py").write_text("def test_loose():\n    assert True\n")
    (accepted / "FERN-99").mkdir()
    (accepted / "FERN-99" / "test_x.py").write_text("")

    assert check_suite(accepted, CARDS) == [
        "FERN-99/test_x.py: FERN-99 is not a card on the board",
        "test_loose.py: accepted tests live in generated/<card>/<name>.py",
    ]
