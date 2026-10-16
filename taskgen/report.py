"""Markdown reports from the gate's results.

* ``proposed/<card>/REPORT.md`` (``card_report``): per criterion, each proposed test with
  its verdict and the reason, the seeded bugs each accepted test caught, static-check
  problems, untestable criteria with their reasons, the criteria left with no accepted
  test, and the cost line when the tests came from the model.
* ``proposed/RUN.md`` (``run_report``): verdict counts per card and in total, which
  seeded bugs the accepted tests caught, and the run's cost line.
* ``card_summary``: a few plain lines for a comment on the card. It names no seeded
  bug, since the board is where the generator's input comes from.

The reports are written for the person reviewing the tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from taskgen.gate import (
    DEFAULT_PROPOSED,
    REPORT_FILE,
    VERDICT_ORDER,
    CardGate,
    GateRun,
    TestOutcome,
    Verdict,
)
from taskgen.generate import GenerationResult, cost_line
from taskgen.prompt import LIST_NAMES

RUN_REPORT_FILE = "RUN.md"

SOURCE_LABELS = {
    "canned": (
        "canned stand-ins in fixtures/canned/ (written for the offline demo, not model output)"
    ),
    "replay": "model responses replayed from recordings/ (no API call in this run)",
    "record": "a live model call, recorded to recordings/",
    "live": "a live model call",
    "fake": "a fake generator (for tests only, not model output)",
}


def source_label(source: str | None) -> str:
    if source is None:
        return "an unrecorded source"
    return SOURCE_LABELS.get(source, source)


def list_name(role: str) -> str:
    return LIST_NAMES.get(role, role or "an unknown list")


def _generation(card: CardGate) -> GenerationResult | None:
    if card.generation is None:
        return None
    return GenerationResult.from_dict(card.generation)


def card_cost_line(card: CardGate) -> str | None:
    """Tokens, cache reads and estimated cost for the card's model call, or None."""
    result = _generation(card)
    return cost_line([result]) if result is not None else None


def run_cost_line(run: GateRun) -> str | None:
    results = [r for r in (_generation(card) for card in run.cards) if r is not None]
    return cost_line(results)


def _counts_line(counts: dict[Verdict, int]) -> str:
    return " · ".join(f"{counts[v]} {v.value}" for v in VERDICT_ORDER if counts[v])


def _bug_list(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _inline(text: str) -> str:
    """A failure message as inline code, so quotes and brackets print as they are."""
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}" if fence == "`" else f"{fence} {text} {fence}"


def _test_block(test: TestOutcome) -> list[str]:
    head = f"**`{test.name}`** · {test.verdict.value}"
    if test.caught:
        head += f" · caught {_bug_list(test.caught)}"
    why = test.reason
    if test.verdict == Verdict.REJECTED:
        why = "Rejected by the static checks, so it never ran."
    lines = [head, "", f"- Intent: {test.intent}", f"- Why: {why}"]
    if test.verdict == Verdict.REJECTED:
        lines.append(f"- Kept for reading at `{test.file}` (not collected by pytest).")
    else:
        lines.append(f"- File: `{test.file}`")
    if test.verdict == Verdict.ACCEPTED:
        for name in test.caught:
            result = test.bugs.get(name)
            message = result.message if result else None
            if result and message and result.exception:
                message = message.removeprefix(f"{result.exception}: ")
            if message:
                lines.append(f"- Under `{name}`: {_inline(message)}")
    if test.verdict in (Verdict.ACCEPTED, Verdict.PROVES_NOTHING):
        for name in test.crashed:
            result = test.bugs.get(name)
            detail = result.describe() if result else "failed"
            lines.append(
                f"- Crashed under `{name}`, which does not count as a catch: {_inline(detail)}"
            )
        errors = [name for name, result in test.bugs.items() if result.outcome == "error"]
        if errors:
            lines.append(f"- Did not run to completion under {_bug_list(errors)}.")
    for problem in test.problems:
        lines.append(f"- Problem: {problem}")
    for warning in test.warnings:
        lines.append(f"- Warning: {warning}")
    if test.verdict == Verdict.REJECTED:
        lines += ["", "```python", test.code.rstrip("\n"), "```"]
    lines.append("")
    return lines


def card_report(card: CardGate) -> str:
    """The Markdown for ``proposed/<card>/REPORT.md``."""
    counts = card.counts()
    lines = [f"# {card.card} · {card.title}", ""]
    lines.append(f"Tests from {source_label(card.source)}.")
    if card.gated:
        lines.append(
            "The card is in Done, so each test that passed the static checks ran against the "
            "clean build twice (once with every other proposal, once on its own), and each "
            f"that passed both times was judged under each of {len(card.seeded_bugs)} seeded "
            "bugs. A bug counts as caught only when an assertion fails under it."
        )
    else:
        lines.append(
            f"The card is in {list_name(card.list_role)}, so its tests were written for review "
            "and not run. Gate them again once the card is in Done."
        )
    lines.append("")
    if card.generation_error:
        lines += [f"Generation produced no tests for this card: {card.generation_error}", ""]
    lines += [
        f"Proposed tests: {len(card.tests)}" + (f" ({_counts_line(counts)})" if card.tests else ""),
        "",
    ]

    untestable = {item.criterion: item.reason for item in card.untestable}
    for criterion in card.criteria:
        lines += [f"## {criterion.id} · {criterion.text}", ""]
        tests = [t for t in card.tests if t.criterion == criterion.id]
        if criterion.id in untestable:
            lines += [f"Listed as untestable: {untestable[criterion.id]}", ""]
        if not tests and criterion.id not in untestable:
            lines += ["No test was proposed for this criterion.", ""]
        for test in tests:
            lines += _test_block(test)
    ids = {criterion.id for criterion in card.criteria}
    strays = [t for t in card.tests if t.criterion not in ids]
    if strays:
        lines += ["## Tests listed under no criterion of this card", ""]
        for test in strays:
            lines += _test_block(test)

    lines += ["## Untestable criteria", ""]
    if card.untestable:
        lines += [f"- **{item.criterion}** · {item.reason}" for item in card.untestable]
    else:
        lines.append("None listed.")
    lines.append("")

    lines += ["## Criteria with no accepted test", ""]
    uncovered = card.uncovered()
    if uncovered:
        for criterion in uncovered:
            proposed = [t for t in card.tests if t.criterion == criterion.id]
            if criterion.id in untestable and not proposed:
                why = "listed as untestable"
            elif proposed:
                verdicts = ", ".join(sorted({t.verdict.value for t in proposed}))
                why = f"{len(proposed)} proposed, none accepted ({verdicts})"
            else:
                why = "no test proposed"
            lines.append(f"- **{criterion.id}** · {why}")
    else:
        lines.append("None: every criterion has at least one accepted test.")
    lines.append("")

    lines += ["## Cost", "", card_cost_line(card) or _no_cost(card.source), ""]

    lines += ["## Next step", ""]
    accepted = card.accepted()
    if accepted:
        names = ", ".join(f"`{t.name}`" for t in accepted)
        lines.append(
            f"Read the accepted tests ({names}), then copy the ones you want into the suite "
            f"with `taskgen accept {card.card}` (all accepted tests) or "
            f"`taskgen accept {card.card} --tests <name>,<name>`."
        )
    elif card.gated:
        lines.append("Nothing to accept: no test for this card passed the gate.")
    else:
        lines.append("Nothing to accept until the card is in Done and its tests pass the gate.")
    if counts[Verdict.FAILS_ON_CLEAN]:
        lines.append(
            "A fails-on-clean test means the test or the app is wrong. Read the failure before "
            "throwing the test away: it may have found a real bug."
        )
    if counts[Verdict.PROVES_NOTHING]:
        lines.append(
            "A proves-nothing test may still be correct: no seeded bug breaks what it checks. "
            "Make its assertions exact, or add a seeded bug for that behavior, before relying "
            "on it."
        )
    lines.append("")
    return "\n".join(lines)


def _no_cost(source: str | None) -> str:
    if source == "canned":
        return "No model call: the tests are canned stand-ins, so there is no cost line."
    return "No model call was recorded for this card."


def run_report(run: GateRun) -> str:
    """The Markdown for ``proposed/RUN.md``."""
    lines = ["# Gate run", ""]
    sources = sorted({source_label(card.source) for card in run.cards})
    if len(sources) == 1:
        lines.append(f"Tests from {sources[0]}.")
    else:
        lines.append("Tests from: " + "; ".join(sources) + ".")
    gated = [card for card in run.cards if card.gated]
    not_run = len(run.cards) - len(gated)
    lines.append(
        f"{len(run.cards)} card(s); {len(gated)} in Done were gated against the clean build "
        f"twice and against {len(run.seeded_bugs)} seeded bugs."
        + (f" {not_run} not in Done were written for review and not run." if not_run else "")
    )
    lines.append("")

    header = ["Card", "List", "Proposed", *(v.value for v in VERDICT_ORDER), "Untestable"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    total = run.counts()
    for card in run.cards:
        counts = card.counts()
        cells = [
            f"[{card.card}]({card.card}/{REPORT_FILE})",
            list_name(card.list_role),
            str(len(card.tests)),
            *(str(counts[v]) for v in VERDICT_ORDER),
            str(len(card.untestable)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    totals = [
        "**Total**",
        "",
        str(sum(len(card.tests) for card in run.cards)),
        *(str(total[v]) for v in VERDICT_ORDER),
        str(sum(len(card.untestable) for card in run.cards)),
    ]
    lines += ["| " + " | ".join(totals) + " |", ""]

    lines += ["## Criteria with no accepted test", ""]
    any_uncovered = False
    for card in run.cards:
        uncovered = card.uncovered()
        if uncovered:
            any_uncovered = True
            lines.append(f"- **{card.card}**: {', '.join(c.id for c in uncovered)}")
    if not any_uncovered:
        lines.append("None.")
    lines.append("")

    lines += ["## Seeded bugs caught", ""]
    if run.seeded_bugs and gated:
        lines += ["| Seeded bug | Caught by accepted tests |", "|---|---|"]
        for bug in run.seeded_bugs:
            catchers = [
                f"{card.card} `{test.name}`"
                for card in run.cards
                for test in card.accepted()
                if bug in test.caught
            ]
            lines.append(f"| `{bug}` | {', '.join(catchers) or 'none'} |")
        lines += [
            "",
            "A bug no accepted test catches is a gap in this run's tests, or a bug in an area "
            "this run did not cover.",
        ]
    else:
        lines.append("No card was gated, so no seeded bug was run.")
    lines.append("")

    cost = run_cost_line(run)
    lines += ["## Cost", "", cost or "No model call in this run, so there is no cost line.", ""]
    return "\n".join(lines)


def card_summary(card: CardGate) -> str:
    """A short plain-text summary for a comment on the card. Names no seeded bug."""
    counts = card.counts()
    lines = [
        f"taskgen gate · {card.card} · tests from {source_label(card.source)}",
        f"{len(card.tests)} proposed: " + (_counts_line(counts).replace(" · ", ", ") or "none"),
    ]
    accepted = card.accepted()
    if accepted:
        lines.append(
            "Accepted, waiting for review: "
            + ", ".join(f"{t.name} ({t.criterion})" for t in accepted)
        )
    uncovered = card.uncovered()
    lines.append(
        "No accepted test yet: " + (", ".join(c.id for c in uncovered) if uncovered else "none")
    )
    if card.untestable:
        lines.append(
            "Untestable: " + "; ".join(f"{u.criterion}: {u.reason}" for u in card.untestable)
        )
    if not card.gated:
        lines.append(f"Not run: the card is in {list_name(card.list_role)}.")
    return "\n".join(lines) + "\n"


def write_reports(run: GateRun, proposed_dir: Path = DEFAULT_PROPOSED) -> list[Path]:
    """Write every card's REPORT.md and the run's RUN.md. Returns the paths written."""
    proposed_dir = Path(proposed_dir)
    paths = []
    for card in run.cards:
        path = proposed_dir / card.card / REPORT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card_report(card), encoding="utf-8")
        paths.append(path)
    path = proposed_dir / RUN_REPORT_FILE
    path.write_text(run_report(run), encoding="utf-8")
    return [*paths, path]
