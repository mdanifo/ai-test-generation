"""The ``taskgen`` command, end to end through ``main(argv)``.

No test here calls the model or the real board. Generation uses the canned stand-ins,
a FakeGenerator or a fake SDK client; the board is an httpx.MockTransport. One test
runs the real gate (pytest subprocesses against Front Desk) with a single seeded bug;
the others give the gate a fake runner.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from frontdesk import bugs
from taskgen import cli, gate
from taskgen.gate import RunResult
from taskgen.generate import (
    FakeGenerator,
    GeneratorConfigError,
    ModelResponse,
    make_generator,
)
from taskgen.report import card_summary
from taskgen.sources import FileSource, TaskBoardSource

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "fernhill-board.json"
CANNED = REPO / "fixtures" / "canned"
CANNED_CARDS = ["FERN-2", "FERN-4", "FERN-8", "FERN-11"]


def run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = cli.main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture(autouse=True)
def no_board_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test can reach the real board, even through the real ``board_source``."""
    for name in ("TASKS_URL", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def proposed(tmp_path: Path) -> Path:
    return tmp_path / "proposed"


@pytest.fixture
def no_recordings(tmp_path: Path) -> Path:
    path = tmp_path / "recordings"
    path.mkdir()
    return path


def canned_output(ref: str) -> dict[str, Any]:
    """A canned stand-in without its notes: what a model would have returned."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if not k.startswith("_")}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(json.loads((CANNED / f"{ref}.json").read_text()))


def generation(proposed: Path, ref: str) -> dict[str, Any]:
    return json.loads((proposed / ref / cli.GENERATION_FILE).read_text())


def index(proposed: Path) -> dict[str, Any]:
    return json.loads((proposed / cli.GENERATE_INDEX).read_text())


def board_payloads() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Board and card payloads in the real API's shape, from the example snapshot."""
    snapshot = json.loads(EXAMPLE.read_text())
    details = {}
    lists: dict[str, list[dict[str, Any]]] = {"todo": [], "doing": [], "done": []}
    for card in snapshot["cards"]:
        lists[card["list"]].append({"ref": card["ref"], "archived_at": None})
        items = [
            {"text": item["text"], "done": item["done"], "position": k}
            for k, item in enumerate(card["criteria"])
        ]
        details[card["ref"]] = {
            "ref": card["ref"],
            "title": card["title"],
            "description": card["description"],
            "list": {"role": card["list"]},
            "labels": [{"name": name} for name in card["labels"]],
            "checklists": [{"name": "Acceptance criteria", "position": 0, "items": items}]
            if items
            else [],
        }
    board = {
        **snapshot["board"],
        "labels": [{"name": n, "color": c} for n, c in snapshot["labels"].items()],
        "lists": [{"role": role, "archived": False, "cards": cs} for role, cs in lists.items()],
    }
    return board, details


class FakeBoard:
    """The task board's API on an httpx.MockTransport. Records every request."""

    def __init__(self) -> None:
        self.board, self.details = board_payloads()
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/boards/fernhill-front-desk":
            return httpx.Response(200, json=self.board)
        if request.method == "GET" and path.startswith("/api/v1/cards/"):
            ref = path.rsplit("/", 1)[1]
            if ref in self.details:
                return httpx.Response(200, json=self.details[ref])
        if request.method == "POST" and path.endswith("/comments"):
            return httpx.Response(201, json={"id": "cmt_1", **json.loads(request.content)})
        return httpx.Response(404, json={"detail": "not found"})

    def source(self, board: str) -> TaskBoardSource:
        return TaskBoardSource(
            board,
            env={
                "TASKS_URL": "https://tasks.example.test",
                "CF_ACCESS_CLIENT_ID": "test-id",
                "CF_ACCESS_CLIENT_SECRET": "test-secret",
            },
            transport=httpx.MockTransport(self.handler),
        )

    @property
    def methods(self) -> set[str]:
        return {request.method for request in self.requests}


@pytest.fixture
def fake_board(monkeypatch: pytest.MonkeyPatch) -> FakeBoard:
    board = FakeBoard()
    monkeypatch.setattr(cli, "board_source", board.source)
    return board


@pytest.fixture
def board_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(board: str) -> TaskBoardSource:
        raise AssertionError(f"the command reached for board {board}")

    monkeypatch.setattr(cli, "board_source", refuse)


def fake_runner(catching: str = "transfer-asks-why"):
    """Every runnable test passes clean and fails only under ``catching``."""
    calls: list[tuple[str | None, str]] = []

    def runner(files: Sequence[str], bug: str | None, label: str) -> dict[str, RunResult]:
        calls.append((bug, label))
        if bug == catching:
            result = RunResult(outcome="failed", exception="AssertionError", cases={"t": "failed"})
        else:
            result = RunResult(outcome="passed", cases={"t": "passed"})
        return dict.fromkeys(files, result)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


@pytest.fixture
def fake_gate(monkeypatch: pytest.MonkeyPatch):
    runner = fake_runner()
    monkeypatch.setattr(gate, "gate_results", functools.partial(gate.gate_results, runner=runner))
    return runner


def test_the_console_script_points_at_main():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]

    assert project["scripts"] == {"taskgen": "taskgen.cli:main"}


def test_python_dash_m_taskgen_lists_every_subcommand():
    done = subprocess.run(
        [sys.executable, "-m", "taskgen", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    for name in ("snapshot", "generate", "gate", "report", "accept", "comment"):
        assert f"    {name}" in done.stdout


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2


def test_canned_generation_saves_each_canned_card_and_an_index(capsys, proposed, board_unreachable):
    code, out, _ = run(capsys, "generate", "--mode", "canned", "--proposed", proposed)

    assert code == 0
    assert index(proposed) == {
        "mode": "canned",
        "source": f"snapshot {cli._rel(EXAMPLE)}",
        "cards": CANNED_CARDS,
    }
    assert "canned stand-ins (written for the offline demo, not model output)" in out
    assert "FERN-11  ok · 5 tests, 1 untestable" in out
    saved = generation(proposed, "FERN-8")
    assert saved["card"]["ref"] == "FERN-8"
    assert saved["card"]["list_role"] == "done"
    assert saved["result"]["source"] == "canned"
    assert len(saved["result"]["output"]["tests"]) == 6


def test_cards_limits_generation_to_the_named_cards(capsys, proposed):
    code, _, _ = run(
        capsys, "generate", "--mode", "canned", "--cards", "FERN-8, FERN-2", "--proposed", proposed
    )

    assert code == 0
    assert index(proposed)["cards"] == ["FERN-8", "FERN-2"]
    assert sorted(p.name for p in proposed.iterdir()) == ["FERN-2", "FERN-8", "generate.json"]


def test_a_card_without_criteria_is_refused_by_name(capsys, proposed):
    code, _, err = run(
        capsys, "generate", "--mode", "canned", "--cards", "FERN-1", "--proposed", proposed
    )

    assert code == 1
    assert "has no card FERN-1 with acceptance criteria" in err


def test_a_card_with_no_canned_response_fails_that_card_only(capsys, proposed):
    code, out, err = run(
        capsys, "generate", "--mode", "canned", "--cards", "FERN-3,FERN-8", "--proposed", proposed
    )

    assert code == 1
    assert "FERN-3  no result · No canned response for FERN-3" in out
    assert "FERN-8  ok · 6 tests" in out
    assert index(proposed)["cards"] == ["FERN-8"]
    assert "No tests from: FERN-3." in err


def test_replay_with_no_recordings_says_how_to_record_and_touches_nothing(
    capsys, proposed, no_recordings
):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)

    code, _, err = run(
        capsys,
        "generate",
        "--cards",
        "FERN-2,FERN-8",
        "--recordings",
        no_recordings,
        "--proposed",
        proposed,
    )

    assert code == 1
    assert "holds no recording for these cards" in err
    assert (
        f"ANTHROPIC_API_KEY=... taskgen generate --mode record --recordings {no_recordings} "
        "--cards FERN-2,FERN-8"
    ) in err
    assert index(proposed)["cards"] == ["FERN-8"]  # the canned run's results are untouched


def test_a_missing_recording_for_one_card_names_the_card_and_the_command(
    capsys, proposed, tmp_path, monkeypatch
):
    recordings = record(capsys, proposed, tmp_path, monkeypatch, "FERN-8")

    code, out, _ = run(
        capsys,
        "generate",
        "--cards",
        "FERN-8,FERN-2",
        "--recordings",
        recordings,
        "--proposed",
        proposed,
    )

    assert code == 1
    assert "FERN-8  ok · 6 tests, 0 untestable" in out
    assert "FERN-2  no result · No recording for FERN-2" in out
    assert "taskgen generate --mode record --recordings" in out
    assert not (proposed / "FERN-2" / cli.GENERATION_FILE).exists()


def test_a_config_error_stops_the_run_at_the_first_card(capsys, proposed, monkeypatch):
    asked: list[str] = []

    class Rejected:
        def generate(self, card):
            asked.append(card.ref)
            raise GeneratorConfigError("The API rejected the credentials (401).")

    monkeypatch.setattr(cli, "make_generator", lambda *a, **k: Rejected())

    code, _, err = run(
        capsys, "generate", "--mode", "live", "--cards", "FERN-2,FERN-4", "--proposed", proposed
    )

    assert code == 1
    assert asked == ["FERN-2"]
    assert "The API rejected the credentials (401)." in err
    assert "Generation stopped" in err
    assert index(proposed)["cards"] == []


def test_a_refusal_is_saved_as_a_failed_card_and_the_run_goes_on(capsys, proposed, monkeypatch):
    fake = FakeGenerator(
        {
            "FERN-2": ModelResponse(text="", model="fake", stop_reason="refusal"),
            "FERN-8": canned_output("FERN-8"),
        }
    )
    monkeypatch.setattr(cli, "make_generator", lambda *a, **k: fake)

    code, out, err = run(
        capsys, "generate", "--mode", "live", "--cards", "FERN-2,FERN-8", "--proposed", proposed
    )

    assert code == 1
    assert fake.calls == ["FERN-2", "FERN-8"]
    assert "FERN-2  refusal · The model declined this card" in out
    assert generation(proposed, "FERN-2")["result"]["status"] == "refusal"
    assert index(proposed)["cards"] == ["FERN-2", "FERN-8"]
    assert "No tests from: FERN-2." in err


def test_board_source_reads_cards_with_get_requests_only(capsys, proposed, fake_board):
    code, out, _ = run(
        capsys, "generate", "--source", "board", "--mode", "canned", "--proposed", proposed
    )

    assert code == 0
    assert "from board fernhill-front-desk" in out
    assert index(proposed)["cards"] == CANNED_CARDS
    assert fake_board.methods == {"GET"}
    assert generation(proposed, "FERN-8")["card"] == (
        FileSource(EXAMPLE).card("FERN-8").model_dump(mode="json")
    )


class FakeStream:
    def __init__(self, message: Any) -> None:
        self.message = message
        self.request_id = "req_fake_1"

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


class FakeClient:
    """Stands in for anthropic.Anthropic: ``client.beta.messages.stream(**request)``."""

    def __init__(self, text: str) -> None:
        self.requests: list[dict[str, Any]] = []
        usage = SimpleNamespace(
            input_tokens=900,
            output_tokens=4000,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=12000,
            iterations=None,
        )
        self.message = SimpleNamespace(
            model="claude-opus-5-5",
            stop_reason="end_turn",
            stop_details=None,
            content=[SimpleNamespace(type="text", text=text)],
            usage=usage,
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **request: Any) -> FakeStream:
        self.requests.append(request)
        return FakeStream(self.message)


def record(capsys, proposed: Path, tmp_path: Path, monkeypatch, ref: str) -> Path:
    """Record one card through the CLI with a fake SDK client. Returns the recordings dir."""
    text = json.dumps(canned_output(ref), ensure_ascii=False)
    client = FakeClient(text)
    recordings = tmp_path / "recordings"
    with monkeypatch.context() as patch:
        patch.setattr(cli, "make_generator", functools.partial(make_generator, client=client))
        code, out, _ = run(
            capsys,
            "generate",
            "--mode",
            "record",
            "--cards",
            ref,
            "--recordings",
            recordings,
            "--proposed",
            proposed,
        )
    assert code == 0, out
    assert len(client.requests) == 1
    assert "Cost: claude-opus-5-5 · 1 card(s)" in out
    return recordings


def test_record_writes_a_recording_that_replay_answers_from(
    capsys, proposed, tmp_path, monkeypatch
):
    recordings = record(capsys, proposed, tmp_path, monkeypatch, "FERN-8")
    saved = json.loads((recordings / "FERN-8.json").read_text())
    assert saved["response"]["request_id"] == "req_fake_1"
    assert saved["response"]["usage"]["cache_creation_input_tokens"] == 12000

    code, out, _ = run(
        capsys, "generate", "--cards", "FERN-8", "--recordings", recordings, "--proposed", proposed
    )

    assert code == 0
    assert "recorded model responses (no API call)" in out
    assert "replayed from recordings, so this run spent nothing" in out
    assert generation(proposed, "FERN-8")["result"]["source"] == "replay"


def test_gate_gates_the_last_generate_run_and_writes_the_reports(capsys, proposed, fake_gate):
    run(capsys, "generate", "--mode", "canned", "--proposed", proposed)

    code, out, _ = run(capsys, "gate", "--proposed", proposed)

    assert code == 0
    labels = [label for _, label in fake_gate.calls]  # clean runs go in parallel, in any order
    assert "clean-1" in labels
    # one clean run per test that passed the static checks: 22 proposed, 2 rejected
    assert sorted(label for label in labels if label.startswith("clean-2-")) == sorted(
        f"clean-2-{ref}_{k}"
        for ref, ks in {"FERN-2": 6, "FERN-4": 5, "FERN-8": 6, "FERN-11": 5}.items()
        for k in range(1, ks + 1)
        if (ref, k) not in {("FERN-4", 3), ("FERN-8", 3)}
    )
    assert len(fake_gate.calls) == 1 + 20 + len(bugs.CATALOG)
    assert (proposed / "RUN.md").exists()
    assert sorted(p.parent.name for p in proposed.glob("*/REPORT.md")) == sorted(CANNED_CARDS)
    assert "Total" in out
    assert "20 accepted · 2 rejected" in out  # the fake runner fails everything under one bug


def test_gate_cards_and_bug_narrow_the_run(capsys, proposed, fake_gate):
    run(capsys, "generate", "--mode", "canned", "--proposed", proposed)

    code, out, _ = run(
        capsys, "gate", "--cards", "FERN-8", "--bug", "transfer-asks-why", "--proposed", proposed
    )

    assert code == 0
    clean, bug_runs = fake_gate.calls[:-1], fake_gate.calls[-1:]  # bugs run after both cleans
    assert sorted(clean, key=lambda c: c[1]) == [
        (None, "clean-1"),
        *((None, f"clean-2-FERN-8_{k}") for k in (1, 2, 4, 5, 6)),
    ]
    assert bug_runs == [("transfer-asks-why", "bug-transfer-asks-why")]
    assert json.loads((proposed / "run.json").read_text())["cards"] == ["FERN-8"]
    assert "6 proposed: 5 accepted · 1 rejected" in out


def test_gate_before_generate_says_what_to_run(capsys, proposed):
    code, _, err = run(capsys, "gate", "--proposed", proposed)

    assert code == 1
    assert "run `taskgen generate` first" in err


def test_gate_for_a_card_that_was_never_generated_names_the_command(capsys, proposed):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)

    code, _, err = run(capsys, "gate", "--cards", "FERN-3", "--proposed", proposed)

    assert code == 1
    assert "run `taskgen generate --cards FERN-3` first" in err


def test_an_unknown_bug_is_a_usage_error(proposed):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["gate", "--bug", "not-a-bug", "--proposed", str(proposed)])

    assert exit_info.value.code == 2


def test_gate_runs_the_real_app_under_one_seeded_bug(capsys, proposed):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)

    code, out, err = run(
        capsys, "gate", "--bug", "transfer-asks-why", "--workers", "3", "--proposed", proposed
    )

    assert code == 0, err
    card = gate.load_card("FERN-8", proposed)
    assert {t.name: t.verdict.value for t in card.tests} == {
        "test_asking_for_a_person_transfers_in_that_turn": "accepted",
        "test_pressing_zero_transfers_in_that_turn": "accepted",
        "test_unclear_turns_offer_a_transfer": "rejected",
        "test_low_confidence_turns_apart_do_not_offer_a_transfer": "proves-nothing",
        "test_transfer_summary_has_reason_verification_and_request": "accepted",
        "test_after_hours_request_offers_a_callback_and_reads_the_number_back": "accepted",
    }
    assert "4 accepted · 1 proves-nothing · 1 rejected" in out


def test_report_prints_run_md_and_checks_expected_counts(capsys, proposed, fake_gate):
    run(capsys, "generate", "--mode", "canned", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)
    (proposed / "RUN.md").unlink()

    code, out, _ = run(
        capsys, "report", "--expect", "accepted=20,rejected=2", "--proposed", proposed
    )

    assert code == 0
    assert out.startswith("# Gate run")
    assert (proposed / "RUN.md").exists()
    assert "Verdict counts match: 20 accepted · 2 rejected." in out


def test_report_fails_when_a_count_differs_from_expect(capsys, proposed, fake_gate):
    run(capsys, "generate", "--mode", "canned", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)

    code, _, err = run(
        capsys,
        "report",
        "--quiet",
        "--expect",
        "accepted=14,proves-nothing=5,fails-on-clean=1,rejected=2",
        "--proposed",
        proposed,
    )

    assert code == 1
    assert "accepted: expected 14, got 20" in err
    assert "proves-nothing: expected 5, got 0" in err
    assert "rejected" not in err


@pytest.mark.parametrize("bad", ["accepted", "accepted=x", "passed=3"])
def test_a_malformed_expect_is_a_usage_error(proposed, bad):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["report", "--expect", bad, "--proposed", str(proposed)])

    assert exit_info.value.code == 2


def test_report_before_gate_says_what_to_run(capsys, proposed):
    code, _, err = run(capsys, "report", "--proposed", proposed)

    assert code == 1
    assert "run `taskgen gate` first" in err


@pytest.fixture
def generated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "generated"
    monkeypatch.setattr(cli, "GENERATED_DIR", target)
    return target


def test_accept_copies_accepted_tests_with_a_header(capsys, proposed, fake_gate, generated_dir):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)

    code, out, _ = run(
        capsys,
        "accept",
        "FERN-8",
        "--tests",
        "test_pressing_zero_transfers_in_that_turn",
        "--proposed",
        proposed,
    )

    assert code == 0
    written = generated_dir / "FERN-8" / "test_pressing_zero_transfers_in_that_turn.py"
    assert written.read_text().startswith("# Generated by taskgen and accepted after review")
    assert f"written    {cli._rel(written)}" in out
    code, out, _ = run(capsys, "accept", "FERN-8", "--proposed", proposed)
    assert code == 0
    assert "unchanged" in out


def test_accept_refuses_a_test_that_is_not_accepted(capsys, proposed, fake_gate, generated_dir):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)

    code, _, err = run(
        capsys,
        "accept",
        "FERN-8",
        "--tests",
        "test_pressing_zero_transfers_in_that_turn,test_unclear_turns_offer_a_transfer",
        "--proposed",
        proposed,
    )

    assert code == 1
    assert "test_unclear_turns_offer_a_transfer is rejected" in err
    assert not generated_dir.exists()


def test_comment_without_yes_prints_the_summary_and_posts_nothing(
    capsys, proposed, fake_gate, board_unreachable
):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)

    code, out, _ = run(capsys, "comment", "FERN-8", "--proposed", proposed)

    assert code == 0
    assert out.startswith("taskgen gate · FERN-8")
    assert "Not posted. Add --yes" in out


def test_comment_with_yes_posts_the_summary_and_names_no_seeded_bug(
    capsys, proposed, fake_gate, fake_board
):
    run(capsys, "generate", "--mode", "canned", "--cards", "FERN-8", "--proposed", proposed)
    run(capsys, "gate", "--proposed", proposed)

    code, out, _ = run(capsys, "comment", "FERN-8", "--yes", "--proposed", proposed)

    assert code == 0
    assert [(r.method, r.url.path) for r in fake_board.requests] == [
        ("POST", "/api/v1/cards/FERN-8/comments")
    ]
    body = json.loads(fake_board.requests[0].content)["body"]
    assert body == card_summary(gate.load_card("FERN-8", proposed))
    assert [name for name in bugs.CATALOG if name in body] == []
    assert "Posted as a comment on FERN-8." in out


def test_snapshot_writes_the_board_with_get_requests_only(capsys, tmp_path, fake_board):
    out_path = tmp_path / "board.json"

    code, out, _ = run(capsys, "snapshot", "--board", "fernhill-front-desk", "--out", out_path)

    assert code == 0
    assert fake_board.methods == {"GET"}
    assert "16 cards from fernhill-front-desk, 15 with acceptance criteria" in out
    written = json.loads(out_path.read_text())
    expected = json.loads(EXAMPLE.read_text())
    assert [c["ref"] for c in written["cards"]] == [c["ref"] for c in expected["cards"]]
    assert written["cards"][1]["criteria"] == expected["cards"][1]["criteria"]


def test_snapshot_without_the_board_url_says_which_variable_to_set(capsys, tmp_path):
    code, _, err = run(
        capsys, "snapshot", "--board", "fernhill-front-desk", "--out", tmp_path / "x.json"
    )

    assert code == 1
    assert "TASKS_URL is not set" in err
