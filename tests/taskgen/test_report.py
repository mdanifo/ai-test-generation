"""Reports: what a reviewer reads in proposed/<card>/REPORT.md and proposed/RUN.md."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from frontdesk import bugs
from taskgen.cards import number_criteria
from taskgen.context import find_leaks
from taskgen.gate import CardGate, GateRun, RunResult, TestOutcome, Verdict, sha256
from taskgen.generate import GenerationResult, Usage
from taskgen.report import card_report, card_summary, run_report, write_reports
from taskgen.schema import UntestableCriterion
from taskgen.static_checks import Problem

CRITERIA = number_criteria(
    [
        "Given an unverified caller asks about an appointment, then Front Desk asks who they are.",
        "Given a match, then Front Desk confirms the caller is verified.",
        'Given "March 3rd 1980" for 1980-03-03, then verification still succeeds.',
        "Given three failed attempts, then Front Desk stops asking.",
        "Given a verified caller asks about a different patient, then Front Desk declines.",
    ]
)
LIVE_USAGE = Usage(
    input_tokens=1200,
    output_tokens=4000,
    cache_read_input_tokens=30000,
    cache_creation_input_tokens=2000,
)


def outcome(
    index: int,
    name: str,
    verdict: Verdict,
    *,
    criterion: str = "AC-1",
    caught: Sequence[str] = (),
    reason: str | None = None,
    problems: Sequence[Problem] = (),
    warnings: Sequence[Problem] = (),
    bugs_run: dict[str, RunResult] | None = None,
) -> TestOutcome:
    code = (
        "import pytest\n\n\n"
        f'@pytest.mark.criterion("FERN-2/{criterion}")\n'
        f"def {name}(call):\n"
        '    assert call.say("When is my next appointment?").text\n'
    )
    folder = "FERN-2/rejected/FERN-2_" if verdict == Verdict.REJECTED else "FERN-2/test_FERN-2_"
    return TestOutcome(
        index=index,
        criterion=criterion,
        name=name,
        intent=f"Fails if {name} stops holding.",
        code=code,
        code_sha256=sha256(code),
        file=f"{folder}{index}.py",
        verdict=verdict,
        reason=reason or f"The reason for {name}.",
        problems=tuple(problems),
        warnings=tuple(warnings),
        bugs=bugs_run or {},
        caught=tuple(caught),
    )


def generation(source: str, status: str = "ok", error: str | None = None) -> dict:
    live = source in ("live", "record", "replay")
    return GenerationResult(
        card="FERN-2",
        status=status,  # type: ignore[arg-type]
        source=source,
        error=error,
        model="claude-opus-5-5" if live else None,
        usage=LIVE_USAGE if live else Usage(),
        cost_usd=LIVE_USAGE.cost_usd("claude-opus-5-5") if live else None,
    ).to_dict()


def fern_2(source: str = "live", **changes: object) -> CardGate:
    fields: dict[str, object] = {
        "card": "FERN-2",
        "title": "Verify the caller before sharing any appointment details",
        "list_role": "done",
        "gated": True,
        "criteria": CRITERIA,
        "tests": (
            outcome(1, "test_reply_exists", Verdict.PROVES_NOTHING),
            outcome(
                2,
                "test_dob_in_words_verifies",
                Verdict.ACCEPTED,
                criterion="AC-3",
                caught=["dob-format-strict"],
                warnings=[Problem(rule="loop", line=8, message="loops over literal values")],
                bugs_run={
                    "dob-format-strict": RunResult(outcome="failed", exception="AssertionError"),
                    "double-booking": RunResult(outcome="passed"),
                },
            ),
            outcome(
                3,
                "test_wrong_time",
                Verdict.FAILS_ON_CLEAN,
                criterion="AC-2",
                reason="Fails on the clean build: failed (AssertionError: assert '9 AM' == "
                "'9:00 AM'). The test or the app is wrong; a person decides which.",
            ),
            outcome(
                4,
                "test_sleepy",
                Verdict.REJECTED,
                criterion="AC-2",
                problems=[
                    Problem(rule="import", line=1, message="imports time, which is not allowed"),
                    Problem(rule="sleep", line=9, message="calls time.sleep()"),
                ],
            ),
            outcome(5, "test_wobbly", Verdict.FLAKY, criterion="AC-3"),
        ),
        "untestable": (UntestableCriterion(criterion="AC-5", reason="Needs a second caller."),),
        "seeded_bugs": tuple(bugs.CATALOG),
        "generation": generation(source),
    }
    fields.update(changes)
    return CardGate.model_validate(fields)


def no_dashes(text: str) -> bool:
    return chr(0x2014) not in text and chr(0x2013) not in text


def test_card_report_opens_with_the_card_its_source_and_how_it_was_gated():
    report = card_report(fern_2())

    assert report.startswith(
        "# FERN-2 · Verify the caller before sharing any appointment details\n"
    )
    assert "Tests from a live model call." in report
    assert "once with every other proposal, once on its own" in report
    assert "judged under each of 12 seeded bugs" in report
    assert "caught only when an assertion fails under it" in report
    assert (
        "Proposed tests: 5 (1 accepted · 1 proves-nothing · 1 fails-on-clean · 1 flaky · "
        "1 rejected)"
    ) in report


def test_card_report_has_a_section_per_criterion_in_order():
    report = card_report(fern_2())
    headings = [line for line in report.splitlines() if line.startswith("## AC-")]

    assert headings == [f"## {c.id} · {c.text}" for c in CRITERIA]


def test_each_test_shows_its_verdict_reason_and_the_bugs_it_caught():
    report = card_report(fern_2())

    assert "**`test_dob_in_words_verifies`** · accepted · caught `dob-format-strict`" in report
    assert "- Intent: Fails if test_dob_in_words_verifies stops holding." in report
    assert "- File: `FERN-2/test_FERN-2_2.py`" in report
    assert "- Warning: line 8: loops over literal values [loop]" in report
    assert "**`test_reply_exists`** · proves-nothing\n" in report
    assert "- Why: The reason for test_reply_exists." in report
    assert "assert '9 AM' == '9:00 AM'" in report
    assert "**`test_wobbly`** · flaky" in report


def test_an_accepted_test_shows_how_it_failed_under_each_bug_it_caught():
    caught = RunResult(
        outcome="failed",
        exception="AssertionError",
        message="assert None == 'P-1001'",
        cases={"t": "failed"},
    )
    test = outcome(
        2,
        "test_dob_in_words_verifies",
        Verdict.ACCEPTED,
        criterion="AC-3",
        caught=["dob-format-strict"],
        bugs_run={"dob-format-strict": caught, "double-booking": RunResult(outcome="passed")},
    )

    report = card_report(fern_2(tests=(test,)))

    assert "- Under `dob-format-strict`: `assert None == 'P-1001'`" in report
    assert "Under `double-booking`" not in report


def test_a_crash_under_a_bug_is_shown_and_said_not_to_count():
    crash = RunResult(
        outcome="failed",
        exception="TypeError",
        message="TypeError: 'NoneType' object is not subscriptable",
        cases={"t": "failed"},
    )
    test = outcome(
        1, "test_summary_exists", Verdict.PROVES_NOTHING, bugs_run={"ssml-unescaped": crash}
    )
    test = test.model_copy(update={"crashed": ("ssml-unescaped",)})

    report = card_report(fern_2(tests=(test,)))

    assert (
        "- Crashed under `ssml-unescaped`, which does not count as a catch: "
        "`failed (TypeError: 'NoneType' object is not subscriptable)`"
    ) in report


def test_a_rejected_test_shows_its_problems_and_its_code():
    report = card_report(fern_2())
    block = report.split("**`test_sleepy`** · rejected", 1)[1].split("**`", 1)[0]

    assert "- Why: Rejected by the static checks, so it never ran." in block
    assert "- Problem: line 1: imports time, which is not allowed [import]" in block
    assert "- Problem: line 9: calls time.sleep() [sleep]" in block
    assert "not collected by pytest" in block
    assert "```python\nimport pytest\n" in block


def test_untestable_criteria_are_listed_with_their_reasons():
    report = card_report(fern_2())

    assert "Listed as untestable: Needs a second caller." in report
    assert "## Untestable criteria\n\n- **AC-5** · Needs a second caller.\n" in report


def test_criteria_with_no_accepted_test_say_why():
    report = card_report(fern_2())
    section = report.split("## Criteria with no accepted test\n\n", 1)[1].split("\n\n", 1)[0]

    assert section.splitlines() == [
        "- **AC-1** · 1 proposed, none accepted (proves-nothing)",
        "- **AC-2** · 2 proposed, none accepted (fails-on-clean, rejected)",
        "- **AC-4** · no test proposed",
        "- **AC-5** · listed as untestable",
    ]


def test_the_cost_line_appears_when_the_tests_came_from_the_model():
    report = card_report(fern_2("live"))
    cost = report.split("## Cost\n\n", 1)[1].split("\n", 1)[0]

    assert cost.startswith("claude-opus-5-5 · 1 card(s) · ")
    assert "1,200 input, 30,000 cache read, 2,000 cache write, 4,000 output tokens" in cost
    assert "about $0.10" in cost  # 1200*4 + 4000*20 + 30000*0.2 + 2000*5 per million
    assert "list prices" in cost


def test_canned_tests_have_no_cost_line_and_say_they_are_not_model_output():
    report = card_report(fern_2("canned"))

    assert (
        "Tests from canned stand-ins in fixtures/canned/ (written for the offline demo, not "
        "model output)."
    ) in report
    assert "No model call: the tests are canned stand-ins, so there is no cost line." in report
    assert "$" not in report


def test_the_next_step_names_the_accept_command_and_explains_weak_verdicts():
    report = card_report(fern_2())
    step = report.split("## Next step\n\n", 1)[1]

    assert "`taskgen accept FERN-2`" in step
    assert "`taskgen accept FERN-2 --tests <name>,<name>`" in step
    assert "it may have found a real bug" in step
    assert "A proves-nothing test may still be correct" in step


def test_a_card_outside_done_says_its_tests_were_not_run():
    card = fern_2(
        list_role="todo",
        gated=False,
        seeded_bugs=(),
        tests=(outcome(1, "test_later", Verdict.NOT_BUILT_YET),),
    )
    report = card_report(card)

    assert "The card is in To Do, so its tests were written for review and not run." in report
    assert "Nothing to accept until the card is in Done" in report


def test_a_card_whose_generation_failed_says_why():
    card = fern_2(
        tests=(),
        untestable=(),
        generation=generation("live", status="refusal", error="The model declined this card."),
    )
    report = card_report(card)

    assert "Generation produced no tests for this card: The model declined this card." in report
    assert report.count("No test was proposed for this criterion.") == 5
    assert "Nothing to accept: no test for this card passed the gate." in report


def two_card_run(source: str = "live") -> GateRun:
    later = fern_2(
        source,
        card="FERN-9",
        title="Keep replies inside the latency budget",
        list_role="todo",
        gated=False,
        seeded_bugs=(),
        untestable=(),
        tests=(outcome(1, "test_later", Verdict.NOT_BUILT_YET),),
    )
    return GateRun(cards=(fern_2(source), later), seeded_bugs=tuple(bugs.CATALOG))


def test_run_report_counts_verdicts_per_card_and_in_total():
    report = run_report(two_card_run())
    table = [line for line in report.splitlines() if line.startswith("|")]

    assert table[0] == (
        "| Card | List | Proposed | accepted | proves-nothing | fails-on-clean | flaky | "
        "rejected | not-built-yet | Untestable |"
    )
    assert table[2] == "| [FERN-2](FERN-2/REPORT.md) | Done | 5 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |"
    assert table[3] == "| [FERN-9](FERN-9/REPORT.md) | To Do | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |"
    assert table[4] == "| **Total** |  | 6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |"
    assert "2 card(s); 1 in Done were gated" in report
    assert "1 not in Done were written for review and not run." in report


def test_run_report_lists_which_accepted_tests_caught_each_seeded_bug():
    report = run_report(two_card_run())

    assert "| `dob-format-strict` | FERN-2 `test_dob_in_words_verifies` |" in report
    assert "| `double-booking` | none |" in report
    assert report.count(" | none |") == len(bugs.CATALOG) - 1


def test_run_report_cost_line_adds_up_the_cards_that_used_the_model():
    live = run_report(two_card_run("live"))
    canned = run_report(two_card_run("canned"))

    assert "claude-opus-5-5 · 2 card(s) · 2,400 input, 60,000 cache read" in live
    assert "No model call in this run, so there is no cost line." in canned


def test_card_summary_names_no_seeded_bug():
    summary = card_summary(fern_2())

    assert summary.splitlines()[0] == "taskgen gate · FERN-2 · tests from a live model call"
    assert "Accepted, waiting for review: test_dob_in_words_verifies (AC-3)" in summary
    assert "No accepted test yet: AC-1, AC-2, AC-4, AC-5" in summary
    assert find_leaks(summary) == []


@pytest.mark.parametrize("source", ["live", "canned"])
def test_reports_use_no_em_or_en_dashes(source):
    run = two_card_run(source)
    texts = [run_report(run), card_summary(run.cards[0]), *(card_report(c) for c in run.cards)]

    assert all(no_dashes(text) for text in texts)


def test_write_reports_writes_a_report_per_card_and_the_run_report(tmp_path):
    run = two_card_run()
    paths = write_reports(run, tmp_path)

    assert paths == [
        tmp_path / "FERN-2" / "REPORT.md",
        tmp_path / "FERN-9" / "REPORT.md",
        tmp_path / "RUN.md",
    ]
    assert (tmp_path / "FERN-2" / "REPORT.md").read_text() == card_report(run.cards[0])
    assert (tmp_path / "RUN.md").read_text() == run_report(run)


def test_a_test_whose_criterion_is_not_on_the_card_is_still_shown():
    stray = outcome(6, "test_for_a_criterion_the_card_lacks", Verdict.REJECTED, criterion="AC-9")
    report = card_report(fern_2(tests=(stray,)))
    before, _, after = report.partition("## Tests listed under no criterion of this card")

    assert "test_for_a_criterion_the_card_lacks" not in before
    assert "**`test_for_a_criterion_the_card_lacks`** · rejected" in after


def test_a_failure_message_with_backticks_is_fenced_so_it_still_reads_as_code():
    caught = RunResult(
        outcome="failed",
        exception="AssertionError",
        message="assert `x` == 'y'",
        cases={"t": "failed"},
    )
    test = outcome(
        2,
        "test_ticks",
        Verdict.ACCEPTED,
        criterion="AC-3",
        caught=["dob-format-strict"],
        bugs_run={"dob-format-strict": caught},
    )

    report = card_report(fern_2(tests=(test,)))

    assert "- Under `dob-format-strict`: `` assert `x` == 'y' ``" in report
