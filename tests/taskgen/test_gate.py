"""The gate: verdict logic, JUnit parsing, orchestration, and end-to-end runs on the real app.

The end-to-end tests run the gate as it runs for real: proposals written to disk, pytest
in subprocesses with the harness plugin, Front Desk started for each run, SEEDED_BUG set
for each bug. Each uses a small proposal built for one verdict and, to keep the
suite quick, two seeded bugs instead of twelve. One test gates the canned stand-ins
against the whole catalog and checks every verdict they were written to produce.
"""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from frontdesk import bugs
from taskgen.cards import Card, number_criteria
from taskgen.gate import (
    CANARY_FILE,
    GATE_FILE,
    RUN_FILE,
    RUNS_DIR,
    GateError,
    GateInput,
    PytestRunner,
    RunResult,
    TestOutcome,
    Verdict,
    decide,
    gate_cards,
    gate_env,
    gate_results,
    load_card,
    load_run,
    parse_junit,
)
from taskgen.generate import CannedGenerator, GenerationResult
from taskgen.schema import GeneratedTests, ProposedTest, UntestableCriterion
from taskgen.sources import FileSource

REPO = Path(__file__).resolve().parents[2]
BOARD = REPO / "examples" / "fernhill-board.json"
CANNED = REPO / "fixtures" / "canned"
TWO_BUGS = ("double-booking", "transfer-asks-why")

FERN_8 = Card(
    ref="FERN-8",
    title="Hand off to a person on request or when the agent is stuck",
    list_role="done",
    criteria=number_criteria(
        [
            'Given a caller says "agent", then Front Desk transfers in that turn.',
            "Given two low-confidence turns in a row, then Front Desk offers a transfer.",
        ]
    ),
)
FERN_9 = Card(
    ref="FERN-9",
    title="Keep replies inside the latency budget",
    list_role="todo",
    criteria=number_criteria(["Every turn records its latency in the call record."]),
)

PASSES_AND_CATCHES = """\
from harness import TRANSFER_LINE

assert call.say("agent").text == TRANSFER_LINE
"""
PASSES_WHATEVER = 'assert call.say("agent").text\n'
WRONG_FOR_ONE_INPUT = """\
assert call.say(said).transferred is (said == "agent")
"""


def make(name: str, body: str, *, card: Card = FERN_8, criterion: str = "AC-1") -> ProposedTest:
    """A proposal: imports pulled out of ``body``, then one marked test taking ``call``."""
    lines = body.splitlines()
    imports = [line for line in lines if line.startswith(("from ", "import "))]
    steps = [line for line in lines if line not in imports and line.strip()]
    params = "call"
    decorators = f'@pytest.mark.criterion("{card.ref}/{criterion}")\n'
    if "said" in body:
        decorators += '@pytest.mark.parametrize("said", ["agent", "representative"])\n'
        params = "call, said"
    code = (
        "import pytest\n"
        + "".join(f"{line}\n" for line in imports)
        + "\n\n"
        + decorators
        + f"def {name}({params}):\n"
        + textwrap.indent("\n".join(steps), "    ")
        + "\n"
    )
    return ProposedTest(criterion=criterion, name=name, intent=f"Intent of {name}.", code=code)


def generated(card: Card, *tests: ProposedTest, untestable=()) -> GeneratedTests:
    return GeneratedTests(card=card.ref, tests=list(tests), untestable=list(untestable))


def passed(**cases: str) -> RunResult:
    return RunResult(outcome="passed", cases=cases or {"t": "passed"})


def failed(exception: str | None = "AssertionError", message: str = "assert 1 == 2") -> RunResult:
    return RunResult(outcome="failed", exception=exception, message=message, cases={"t": "failed"})


@pytest.mark.parametrize(
    ("kwargs", "verdict", "caught"),
    [
        (
            {"static_ok": False, "static_reason": "imports time", "gated": False},
            Verdict.REJECTED,
            (),
        ),
        ({"static_ok": True, "gated": False, "list_role": "To Do"}, Verdict.NOT_BUILT_YET, ()),
        ({"static_ok": True, "gated": True, "clean": [passed(), failed()]}, Verdict.FLAKY, ()),
        (
            {"static_ok": True, "gated": True, "clean": [failed(), failed("AssertionError", "x")]},
            Verdict.FAILS_ON_CLEAN,
            (),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"double-booking": passed(), "transfer-asks-why": failed()},
            },
            Verdict.ACCEPTED,
            ("transfer-asks-why",),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"double-booking": passed(), "transfer-asks-why": passed()},
            },
            Verdict.PROVES_NOTHING,
            (),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"double-booking": RunResult(outcome="error", exception="X")},
            },
            Verdict.PROVES_NOTHING,
            (),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"transfer-asks-why": failed("TypeError", "TypeError: 'NoneType'")},
            },
            Verdict.PROVES_NOTHING,
            (),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"double-booking": failed("Failed", "DID NOT RAISE CallError")},
            },
            Verdict.ACCEPTED,
            ("double-booking",),
        ),
        (
            {
                "static_ok": True,
                "gated": True,
                "clean": [passed(), passed()],
                "bug_results": {"double-booking": failed(None, "no exception recorded")},
            },
            Verdict.PROVES_NOTHING,
            (),
        ),
    ],
    ids=[
        "static-problem-beats-everything",
        "not-in-done",
        "clean-runs-disagree",
        "fails-the-same-way-twice",
        "fails-under-one-bug",
        "passes-under-every-bug",
        "setup-error-is-not-a-catch",
        "crash-is-not-a-catch",
        "pytest-raises-not-raising-is-a-catch",
        "unrecorded-exception-is-not-a-catch",
    ],
)
def test_decide_gives_each_verdict(kwargs, verdict, caught):
    got, reason, got_caught = decide(**kwargs)

    assert (got, got_caught) == (verdict, caught)
    assert reason[0].isupper()
    assert kwargs.get("static_reason", "") in reason


def test_decide_reasons_say_what_happened():
    flaky = decide(static_ok=True, gated=True, clean=[passed(), failed()])[1]
    on_clean = decide(static_ok=True, gated=True, clean=[failed(), failed()])[1]
    nothing = decide(
        static_ok=True, gated=True, clean=[passed(), passed()], bug_results={"a": passed()}
    )[1]

    crash = decide(
        static_ok=True,
        gated=True,
        clean=[passed(), passed()],
        bug_results={
            "a": passed(),
            "b": failed("TypeError", "TypeError: x"),
            "c": failed("KeyError"),
        },
    )[1]

    assert flaky == (
        "The two clean runs disagree: with the other proposals it passed, on its own it failed "
        "(AssertionError: assert 1 == 2). Its result depends on other tests or on when it runs, "
        "so it would not hold once accepted."
    )
    assert "Fails on the clean build: failed (AssertionError: assert 1 == 2)." in on_clean
    assert "a person decides which" in on_clean
    assert "under the one seeded bug run" in nothing
    assert crash == (
        "Passes on the clean build twice, and no seeded bug made an assertion fail. Under b, c "
        "it crashed (KeyError, TypeError) instead, which shows a step broke, not that the test "
        "checks anything."
    )


def test_decide_needs_both_clean_runs_for_a_gated_test():
    with pytest.raises(ValueError, match="both clean runs"):
        decide(static_ok=True, gated=True, clean=[passed()])


def test_run_results_compare_outcomes_not_messages():
    first = RunResult(outcome="failed", exception="E", message="at 0x1", cases={"t": "failed"})
    second = RunResult(outcome="failed", exception="E", message="at 0x2", cases={"t": "failed"})

    assert first.same_as(second)
    assert not first.same_as(passed(t="passed"))
    assert RunResult(
        outcome="failed", exception="KeyError", message="KeyError: 'x'"
    ).describe() == ("failed (KeyError: 'x')")


def test_parse_junit_folds_cases_per_file(tmp_path):
    junit = tmp_path / "run.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite>
  <testcase classname="p.FERN-8.test_FERN-8_1" name="test_a[agent]"
            file="p/FERN-8/test_FERN-8_1.py"/>
  <testcase classname="p.FERN-8.test_FERN-8_1" name="test_a[rep]" file="p/FERN-8/test_FERN-8_1.py">
    <failure message="assert 'x' == 'y'&#10;  - y">long text</failure>
  </testcase>
  <testcase classname="" name="p.FERN-8.test_FERN-8_2" file="p/FERN-8/test_FERN-8_2.py">
    <error message="collection failure">Traceback:
E   ImportError: cannot import name 'NOPE' from 'harness'</error>
  </testcase>
  <testcase classname="p.FERN-8.test_FERN-8_3" name="test_c" file="p/FERN-8/test_FERN-8_3.py">
    <skipped message="no reason"/>
  </testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps({"p/FERN-8/test_FERN-8_1.py::test_a[rep]": {"exception": "AssertionError"}}),
        encoding="utf-8",
    )
    files = [f"p/FERN-8/test_FERN-8_{k}.py" for k in (1, 2, 3, 4)]

    found = parse_junit(junit, results, files)

    assert found[files[0]] == RunResult(
        outcome="failed",
        exception="AssertionError",
        message="assert 'x' == 'y'",
        cases={"test_a[agent]": "passed", "test_a[rep]": "failed"},
    )
    assert found[files[1]].outcome == "error"
    assert found[files[1]].exception == "ImportError"
    assert found[files[1]].cases == {"(collection)": "error"}
    assert found[files[2]].outcome == "skipped"
    assert found[files[3]].outcome == "missing"


def test_parse_junit_reports_an_assertion_failure_before_a_crash_in_another_case(tmp_path):
    junit = tmp_path / "run.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite>
  <testcase classname="p" name="test_a[x]" file="p/test_1.py">
    <failure message="TypeError: no"/>
  </testcase>
  <testcase classname="p" name="test_a[y]" file="p/test_1.py">
    <failure message="assert 1 == 2"/>
  </testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "p/test_1.py::test_a[x]": {"exception": "TypeError"},
                "p/test_1.py::test_a[y]": {"exception": "AssertionError"},
            }
        ),
        encoding="utf-8",
    )

    found = parse_junit(junit, results, ["p/test_1.py"])["p/test_1.py"]

    assert (found.exception, found.message, found.caught) == (
        "AssertionError",
        "assert 1 == 2",
        True,
    )


def test_gate_env_drops_credentials_and_earlier_build_choices(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_IDENTITY_TOKEN",
        "FRONTDESK_URL",
        "SEEDED_BUG",
        "PYTEST_XDIST_WORKER",
    ):
        monkeypatch.setenv(name, "dummy")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")

    clean = gate_env(None)
    bug = gate_env("double-booking")

    assert not [name for name in clean if name.startswith("ANTHROPIC_")]
    dropped = {"FRONTDESK_URL", "PYTEST_XDIST_WORKER"}
    assert dropped.isdisjoint(clean)
    assert "SEEDED_BUG" not in clean
    assert bug["SEEDED_BUG"] == "double-booking"
    assert clean["PYTHONPATH"].split(os.pathsep)[:2] == [str(REPO), "/somewhere"]


class FakeRunner:
    """Stands in for pytest: answers from ``outcome(file, bug, label)`` and records every call."""

    def __init__(self, outcome: Callable[[str, str | None, str], RunResult]) -> None:
        self.outcome = outcome
        self.calls: list[tuple[tuple[str, ...], str | None, str]] = []

    def __call__(self, files: Sequence[str], bug: str | None, label: str) -> dict[str, RunResult]:
        self.calls.append((tuple(files), bug, label))
        return {f: self.outcome(f, bug, label) for f in files}


def test_gate_runs_clean_together_and_alone_then_bugs_with_the_same_files(tmp_path):
    good = make("test_good", PASSES_AND_CATCHES)
    wrong = make("test_wrong", 'assert call.say("agent").text == "Hold on."')
    sleepy = make("test_sleepy", 'import time\ntime.sleep(1)\nassert call.say("agent").text')
    later = make("test_later", 'assert call.record()["turns"]', card=FERN_9)

    def outcome(file: str, bug: str | None, label: str) -> RunResult:
        if file.endswith("_2.py") or (bug == "transfer-asks-why" and file.endswith("_1.py")):
            return failed()
        return passed()

    runner = FakeRunner(outcome)
    proposed = tmp_path / "proposed"
    run = gate_cards(
        [(FERN_8, generated(FERN_8, good, wrong, sleepy)), (FERN_9, generated(FERN_9, later))],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        runner=runner,
    )

    first, second = "proposed/FERN-8/test_FERN-8_1.py", "proposed/FERN-8/test_FERN-8_2.py"
    both = (first, second)
    assert sorted(runner.calls, key=lambda c: c[2]) == [
        (both, "double-booking", "bug-double-booking"),
        (both, "transfer-asks-why", "bug-transfer-asks-why"),
        (both, None, "clean-1"),
        ((first,), None, "clean-2-FERN-8_1"),
        ((second,), None, "clean-2-FERN-8_2"),
    ]
    assert [(t.name, t.verdict) for c in run.cards for t in c.tests] == [
        ("test_good", Verdict.ACCEPTED),
        ("test_wrong", Verdict.FAILS_ON_CLEAN),
        ("test_sleepy", Verdict.REJECTED),
        ("test_later", Verdict.NOT_BUILT_YET),
    ]
    assert run.cards[0].tests[0].caught == ("transfer-asks-why",)
    assert (proposed / "FERN-8" / "test_FERN-8_1.py").read_text() == good.code
    assert (proposed / "FERN-8" / "rejected" / "FERN-8_3.py").read_text() == sleepy.code
    assert not (proposed / "FERN-8" / "test_FERN-8_3.py").exists()
    assert (proposed / "FERN-9" / "test_FERN-9_1.py").read_text() == later.code
    assert run.cards[1].seeded_bugs == ()
    assert run.cards[0].seeded_bugs == TWO_BUGS


def test_gate_saves_a_record_that_loads_back(tmp_path):
    proposed = tmp_path / "proposed"
    untestable = UntestableCriterion(criterion="AC-2", reason="Needs real elapsed time.")
    run = gate_cards(
        [
            (
                FERN_8,
                generated(FERN_8, make("test_good", PASSES_AND_CATCHES), untestable=[untestable]),
            )
        ],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        runner=FakeRunner(lambda f, bug, label: failed() if bug else passed()),
    )

    assert (proposed / "FERN-8" / GATE_FILE).exists()
    assert json.loads((proposed / RUN_FILE).read_text()) == {
        "cards": ["FERN-8"],
        "seeded_bugs": list(TWO_BUGS),
    }
    assert load_card("FERN-8", proposed) == run.cards[0]
    assert load_run(proposed) == run
    assert run.cards[0].untestable == (untestable,)
    assert run.cards[0].uncovered() == [FERN_8.criteria[1]]


def test_a_test_that_passes_only_next_to_the_others_is_flaky(tmp_path):
    good = make("test_good", PASSES_AND_CATCHES)
    leaning = make("test_leaning", 'assert call.say("agent").text')

    def outcome(file: str, bug: str | None, label: str) -> RunResult:
        if file.endswith("_2.py") and label.startswith("clean-2"):
            return failed("AttributeError", "AttributeError: no attribute 'shared'")
        return passed()

    run = gate_cards(
        [(FERN_8, generated(FERN_8, good, leaning))],
        proposed_dir=tmp_path / "proposed",
        bugs=TWO_BUGS,
        runner=FakeRunner(outcome),
    )

    test = run.cards[0].tests[1]
    assert test.verdict == Verdict.FLAKY
    assert "on its own it failed (AttributeError" in test.reason
    assert test.bugs == {}


def test_a_crash_under_a_bug_is_recorded_but_not_counted(tmp_path):
    def outcome(file: str, bug: str | None, label: str) -> RunResult:
        if bug == "transfer-asks-why":
            return failed("TypeError", "TypeError: 'NoneType' object is not subscriptable")
        return passed()

    run = gate_cards(
        [(FERN_8, generated(FERN_8, make("test_crashy", PASSES_AND_CATCHES)))],
        proposed_dir=tmp_path / "proposed",
        bugs=TWO_BUGS,
        runner=FakeRunner(outcome),
    )

    test = run.cards[0].tests[0]
    assert (test.verdict, test.caught, test.crashed) == (
        Verdict.PROVES_NOTHING,
        (),
        ("transfer-asks-why",),
    )


def test_a_second_proposal_with_the_same_name_is_rejected(tmp_path):
    first = make("test_transfer", PASSES_AND_CATCHES)
    second = make("test_transfer", 'assert call.press("0").text == "x"')
    runner = FakeRunner(lambda f, bug, label: failed() if bug else passed())

    run = gate_cards(
        [(FERN_8, generated(FERN_8, first, second))],
        proposed_dir=tmp_path / "proposed",
        bugs=TWO_BUGS,
        runner=runner,
    )

    one, two = run.cards[0].tests
    assert (one.verdict, two.verdict) == (Verdict.ACCEPTED, Verdict.REJECTED)
    assert [p.rule for p in two.problems] == ["name"]
    assert "already named test_transfer" in two.reason
    assert {files for files, _, _ in runner.calls} == {("proposed/FERN-8/test_FERN-8_1.py",)}


def test_load_card_names_the_command_when_nothing_was_gated(tmp_path):
    with pytest.raises(FileNotFoundError, match="run `taskgen gate` for FERN-8 first"):
        load_card("FERN-8", tmp_path)


def test_gate_refuses_unknown_bugs_and_mismatched_cards(tmp_path):
    runner = FakeRunner(lambda f, bug, label: passed())

    with pytest.raises(ValueError, match="not seeded bugs: no-such-bug"):
        gate_cards([(FERN_8, generated(FERN_8))], proposed_dir=tmp_path, bugs=["no-such-bug"])
    with pytest.raises(ValueError, match="generated tests are for FERN-9, not FERN-8"):
        gate_cards([(FERN_8, generated(FERN_9))], proposed_dir=tmp_path, runner=runner)
    assert runner.calls == []


def test_gate_runs_every_seeded_bug_by_default(tmp_path):
    runner = FakeRunner(lambda f, bug, label: passed())
    run = gate_cards(
        [(FERN_8, generated(FERN_8, make("test_good", PASSES_AND_CATCHES)))],
        proposed_dir=tmp_path / "proposed",
        runner=runner,
    )

    assert run.seeded_bugs == tuple(bugs.CATALOG)
    assert sorted(bug for _, bug, _ in runner.calls if bug) == sorted(bugs.CATALOG)
    assert run.cards[0].tests[0].verdict == Verdict.PROVES_NOTHING


def test_a_second_gate_replaces_its_own_files_and_keeps_the_rest(tmp_path):
    proposed = tmp_path / "proposed"
    card_dir = proposed / "FERN-8"
    card_dir.mkdir(parents=True)
    (card_dir / "test_FERN-8_5.py").write_text("old proposal\n")
    (card_dir / "generation.json").write_text("{}\n")
    runner = FakeRunner(lambda f, bug, label: passed())

    gate_cards(
        [(FERN_8, generated(FERN_8, make("test_good", PASSES_AND_CATCHES)))],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        runner=runner,
    )

    assert sorted(p.name for p in card_dir.iterdir()) == [
        "gate.json",
        "generation.json",
        "test_FERN-8_1.py",
    ]


def test_gate_results_records_a_card_whose_generation_failed(tmp_path):
    refusal = GenerationResult(
        card="FERN-8",
        status="refusal",
        source="replay",
        error="The model declined this card (no category given).",
        model="claude-opus-5-5",
    )
    runner = FakeRunner(lambda f, bug, label: passed())

    run = gate_results([FERN_8, FERN_9], [refusal], proposed_dir=tmp_path, runner=runner)

    card = run.cards[0]
    assert (card.card, card.tests, card.source) == ("FERN-8", (), "replay")
    assert card.generation_error == "The model declined this card (no category given)."
    assert card.generation is not None and card.generation["output"] is None
    assert runner.calls == []


def test_gate_input_carries_the_generation_record(tmp_path):
    result = GenerationResult(
        card="FERN-8",
        status="ok",
        source="canned",
        output=generated(FERN_8, make("test_good", PASSES_AND_CATCHES)),
    )
    run = gate_cards(
        [GateInput(FERN_8, result.output, result)],
        proposed_dir=tmp_path,
        bugs=TWO_BUGS,
        runner=FakeRunner(lambda f, bug, label: passed()),
    )

    assert run.cards[0].source == "canned"
    assert run.cards[0].generation_error is None


PLUGIN = '''\
"""Loaded into gate runs by one test through PYTEST_PLUGINS, to make outcomes vary."""

import os
from pathlib import Path

import pytest

_SESSION = {}


def pytest_sessionstart(session):
    marker = Path(os.environ["TASKGEN_TEST_STATE"]) / "first-session"
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        _SESSION["first"] = True
    except FileExistsError:
        _SESSION["first"] = False


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    if item.name.startswith(os.environ.get("TASKGEN_TEST_FLIP", "-")) and not _SESSION["first"]:
        raise AssertionError("the second clean run sees a different answer")
    if item.name.startswith(os.environ.get("TASKGEN_TEST_BREAK", "-")):
        raise AssertionError("the app did not answer")
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Load PLUGIN into every pytest process the gate starts from this test."""
    home = tmp_path / "plugin"
    home.mkdir()
    (home / "taskgen_test_plugin.py").write_text(PLUGIN, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(home))
    monkeypatch.setenv("PYTEST_PLUGINS", "taskgen_test_plugin")
    monkeypatch.setenv("TASKGEN_TEST_STATE", str(home))
    return monkeypatch


def gate_one(
    tmp_path: Path, card: Card, test: ProposedTest, **options: object
) -> tuple[Path, TestOutcome]:
    proposed = tmp_path / "proposed"
    run = gate_cards(
        [(card, generated(card, test))],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        workers=2,
        **options,
    )
    return proposed, run.cards[0].tests[0]


def run_files(proposed: Path) -> list[str]:
    return sorted(p.name for p in (proposed / RUNS_DIR).iterdir())


def test_end_to_end_a_test_that_catches_a_bug_is_accepted(tmp_path):
    proposed, test = gate_one(tmp_path, FERN_8, make("test_agent_transfers", PASSES_AND_CATCHES))

    assert test.verdict == Verdict.ACCEPTED
    assert test.caught == ("transfer-asks-why",)
    assert [r.outcome for r in test.clean] == ["passed", "passed"]
    assert test.bugs["double-booking"].outcome == "passed"
    assert test.bugs["transfer-asks-why"].exception == "AssertionError"
    assert run_files(proposed) == sorted(
        [CANARY_FILE]
        + [
            f"{label}.{ext}"
            for label in ("clean-1", "clean-2-FERN-8_1")
            for ext in ("json", "log", "xml")
        ]
        + [f"bug-{bug}.{ext}" for bug in TWO_BUGS for ext in ("json", "log", "xml")]
    )


def test_end_to_end_a_test_that_passes_under_every_bug_proves_nothing(tmp_path):
    _, test = gate_one(tmp_path, FERN_8, make("test_reply_exists", PASSES_WHATEVER))

    assert test.verdict == Verdict.PROVES_NOTHING
    assert test.caught == ()
    assert {bug: r.outcome for bug, r in test.bugs.items()} == {
        "double-booking": "passed",
        "transfer-asks-why": "passed",
    }


def test_end_to_end_a_wrong_expected_value_fails_on_clean_and_skips_the_bugs(tmp_path):
    proposed, test = gate_one(tmp_path, FERN_8, make("test_wrong", WRONG_FOR_ONE_INPUT))

    assert test.verdict == Verdict.FAILS_ON_CLEAN
    assert test.clean[0].cases == {
        "test_wrong[agent]": "passed",
        "test_wrong[representative]": "failed",
    }
    assert test.bugs == {}
    assert not any(name.startswith("bug-") for name in run_files(proposed))


def test_end_to_end_results_that_differ_between_clean_runs_are_flaky(tmp_path, plugin):
    plugin.setenv("TASKGEN_TEST_FLIP", "test_wobbly")
    proposed, test = gate_one(tmp_path, FERN_8, make("test_wobbly_transfer", PASSES_AND_CATCHES))

    assert test.verdict == Verdict.FLAKY
    assert sorted(r.outcome for r in test.clean) == ["failed", "passed"]
    assert test.bugs == {}
    assert not any(name.startswith("bug-") for name in run_files(proposed))


SETS_SHARED = """\
import pytest

call.say("agent")
pytest.taskgen_shared_call = call.id
assert call.record()["transferred"] is False
"""
READS_SHARED = """\
import pytest

record = api.get(f"/calls/{pytest.taskgen_shared_call}").json()
assert record["transferred"] is True
"""


def shared_state_pair() -> list[ProposedTest]:
    writer = make("test_a_sets_shared_state", SETS_SHARED)
    reader = make("test_b_reads_shared_state", READS_SHARED)
    reader = reader.model_copy(update={"code": reader.code.replace("(call)", "(api)")})
    return [writer, reader]


def test_end_to_end_tests_that_share_state_are_rejected_before_they_run(tmp_path):
    proposed = tmp_path / "proposed"
    run = gate_cards(
        [(FERN_8, generated(FERN_8, *shared_state_pair()))],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        workers=2,
    )

    writer, reader = run.cards[0].tests
    assert writer.verdict == Verdict.REJECTED
    assert "shared-state" in [p.rule for p in writer.problems]
    assert "sets pytest.taskgen_shared_call" in " ".join(p.message for p in writer.problems)
    assert reader.verdict == Verdict.REJECTED
    assert "uses pytest.taskgen_shared_call" in reader.reason


def test_end_to_end_a_test_that_needs_another_to_run_first_is_flaky(tmp_path, monkeypatch):
    """Static checks cannot see every way tests share state, so the gate runs each alone."""
    from taskgen import gate as gate_module
    from taskgen.static_checks import StaticResult

    monkeypatch.setattr(gate_module, "check_proposal", lambda test, card: StaticResult())
    proposed = tmp_path / "proposed"
    run = gate_cards(
        [(FERN_8, generated(FERN_8, *shared_state_pair()))],
        proposed_dir=proposed,
        bugs=TWO_BUGS,
        workers=2,
    )

    writer, reader = run.cards[0].tests
    assert writer.verdict == Verdict.FAILS_ON_CLEAN
    assert reader.verdict == Verdict.FLAKY
    assert [r.outcome for r in reader.clean] == ["passed", "failed"]
    assert reader.clean[1].exception == "AttributeError"
    assert reader.bugs == {}


def test_end_to_end_a_test_that_only_crashes_under_a_bug_proves_nothing(tmp_path):
    crashy = make(
        "test_summary_text_has_a_length",
        'call.say("agent")\nassert len(call.record()["handoffSummary"]["text"]) >= 0',
    )
    _, test = gate_one(tmp_path, FERN_8, crashy)

    assert test.verdict == Verdict.PROVES_NOTHING
    assert test.crashed == ("transfer-asks-why",)
    assert test.bugs["transfer-asks-why"].exception == "TypeError"
    assert "crashed (TypeError)" in test.reason


def test_end_to_end_a_static_problem_is_rejected_and_never_runs(tmp_path):
    sleepy = make("test_sleepy", 'import time\ntime.sleep(1)\nassert call.say("agent").text')
    proposed, test = gate_one(tmp_path, FERN_8, sleepy)

    assert test.verdict == Verdict.REJECTED
    assert [p.rule for p in test.problems] == ["import", "sleep"]
    assert test.file == "FERN-8/rejected/FERN-8_1.py"
    assert run_files(proposed) == [CANARY_FILE]


def test_end_to_end_a_card_outside_done_is_not_built_yet_and_never_runs(tmp_path):
    proposed, test = gate_one(
        tmp_path, FERN_9, make("test_turns_record_latency", PASSES_WHATEVER, card=FERN_9)
    )

    assert test.verdict == Verdict.NOT_BUILT_YET
    assert (proposed / "FERN-9" / "test_FERN-9_1.py").exists()
    assert run_files(proposed) == [CANARY_FILE]


def test_end_to_end_a_run_whose_app_does_not_answer_stops_the_gate(tmp_path, plugin):
    plugin.setenv("TASKGEN_TEST_BREAK", "test_gate_canary")

    with pytest.raises(GateError, match="the canary test did not pass"):
        gate_one(tmp_path, FERN_8, make("test_agent_transfers", PASSES_AND_CATCHES))


def test_canned_stand_ins_get_the_verdicts_they_were_written_for(tmp_path):
    canned = CannedGenerator(CANNED)
    source = FileSource(BOARD)
    cards = [source.card(ref) for ref in canned.available()]
    results = [canned.generate(card) for card in cards]
    expected = {}
    for path in CANNED.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        expected |= {(data["card"], t["name"]): t["_expected_verdict"] for t in data["tests"]}

    run = gate_results(cards, results, proposed_dir=tmp_path / "proposed", workers=4)

    got = {(card.card, t.name): t.verdict.value for card in run.cards for t in card.tests}
    assert [card.card for card in run.cards] == ["FERN-2", "FERN-4", "FERN-8", "FERN-11"]
    assert run.seeded_bugs == tuple(bugs.CATALOG)
    assert got == expected
    assert {v.value: n for v, n in run.counts().items()} == {
        "accepted": 14,
        "proves-nothing": 5,
        "fails-on-clean": 1,
        "flaky": 0,
        "rejected": 2,
        "not-built-yet": 0,
    }
    assert sum(len(card.untestable) for card in run.cards) == 1
    assert all(t.caught for card in run.cards for t in card.accepted())


@dataclass
class ScriptRunner(PytestRunner):
    """PytestRunner with ``python -c script`` in place of pytest, for runs that never answer."""

    script: str = ""

    def command(self, files: Sequence[str], label: str) -> list[str]:
        return [self.python, "-c", self.script]


def test_a_pytest_run_that_hangs_stops_the_gate_and_names_the_log(tmp_path):
    runner = ScriptRunner(
        rootdir=tmp_path,
        runs_dir=tmp_path / "runs",
        timeout=1.0,
        script="import time; time.sleep(30)",
    )

    with pytest.raises(
        GateError, match=r"run clean-1 did not finish within 1s; see .*clean-1\.log"
    ):
        runner(["test_x.py"], None, "clean-1")


def test_a_pytest_run_that_exits_without_results_stops_the_gate_with_its_output(tmp_path):
    runner = ScriptRunner(
        rootdir=tmp_path,
        runs_dir=tmp_path / "runs",
        script="import sys; print('ERROR: usage: pytest [options]'); sys.exit(4)",
    )

    with pytest.raises(GateError, match="exited with code 4 and wrote no JUnit file") as caught:
        runner(["test_x.py"], None, "clean-1")
    assert "ERROR: usage: pytest [options]" in str(caught.value)


def test_gate_results_refuses_a_result_for_a_card_it_was_not_given(tmp_path):
    stray = GenerationResult(card="FERN-9", status="ok", source="canned")
    runner = FakeRunner(lambda file, bug, label: passed())

    with pytest.raises(KeyError, match="no card FERN-9 among the cards given to the gate"):
        gate_results([FERN_8], [stray], proposed_dir=tmp_path, runner=runner)
    assert runner.calls == []
