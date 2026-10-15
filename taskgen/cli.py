"""The ``taskgen`` command: snapshot, generate, gate, report, accept, comment.

    taskgen snapshot --board fernhill-front-desk --out examples/fernhill-board.json
    taskgen generate --mode canned --source file
    taskgen gate
    taskgen report
    taskgen accept FERN-8 --tests test_pressing_zero_transfers_in_that_turn
    taskgen comment FERN-8          # prints the comment; add --yes to post it

``generate`` asks a generator for each card, one card after another (so every card
after the first reads the cached prompt prefix), and saves each result as
``proposed/<card>/generation.json`` with the card it was made for. It also writes
``proposed/generate.json``, the list of cards that run produced. ``gate`` reads those
files, so it gates exactly what was generated, and writes ``gate.json`` and the
Markdown reports. ``report`` rebuilds the reports from ``gate.json``. ``accept`` copies
reviewed, accepted tests into ``tests/generated/``. ``comment`` posts a card's summary
to the board, and only with ``--yes``.

Exit status: 0 when the command did what it was asked, 1 when it could not (a card
generated nothing, the gate could not run, a check failed), 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from frontdesk import bugs as seeded
from taskgen import accept as accepting
from taskgen import gate, report
from taskgen.cards import Card
from taskgen.context import ContextLeakError
from taskgen.gate import DEFAULT_PROPOSED, GateError, GateRun, Verdict
from taskgen.generate import (
    CANNED_DIR,
    MODES,
    RECORDINGS_DIR,
    CannedGenerator,
    GenerationError,
    GenerationResult,
    GeneratorConfigError,
    cost_line,
    make_generator,
)
from taskgen.sources import CardSource, FileSource, SourceError, TaskBoardSource, write_snapshot

REPO = gate.REPO
DEFAULT_SNAPSHOT = REPO / "examples" / "fernhill-board.json"
DEFAULT_BOARD = "fernhill-front-desk"
GENERATED_DIR = accepting.GENERATED_DIR
GENERATION_FILE = "generation.json"
GENERATE_INDEX = "generate.json"


class CommandError(Exception):
    """A command that cannot go on. The message says what to do; the exit status is 1."""


def board_source(board: str) -> TaskBoardSource:
    """The live board, configured from TASKS_URL and the CF_ACCESS_* variables."""
    return TaskBoardSource(board)


def _split(value: str | None) -> list[str]:
    """``"FERN-2, FERN-4"`` -> ``["FERN-2", "FERN-4"]``, without duplicates."""
    if not value:
        return []
    return list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def _positive(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number of 1 or more")
    return int(value)


def _rel(path: Path) -> str:
    """A path relative to the working directory when it is inside it."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def _err(message: str) -> None:
    sys.stdout.flush()  # keep the order of stdout and stderr lines when both are piped
    print(message, file=sys.stderr)


def _counts_text(counts: dict[Verdict, int]) -> str:
    return " · ".join(f"{counts[v]} {v.value}" for v in gate.VERDICT_ORDER if counts[v]) or "none"


def _print_counts(run: GateRun) -> None:
    width = max([len(card.card) for card in run.cards] + [5])
    for card in run.cards:
        where = report.list_name(card.list_role)
        print(
            f"{card.card:<{width}}  {where:<11}  {len(card.tests)} proposed: "
            f"{_counts_text(card.counts())}"
            + (f" · {len(card.untestable)} untestable" if card.untestable else "")
        )
    total = sum(len(card.tests) for card in run.cards)
    untestable = sum(len(card.untestable) for card in run.cards)
    print(
        f"{'Total':<{width}}  {'':<11}  {total} proposed: {_counts_text(run.counts())}"
        + (f" · {untestable} untestable" if untestable else "")
    )


def save_generation(card: Card, result: GenerationResult, proposed_dir: Path) -> Path:
    """Write ``proposed/<card>/generation.json``: the card and the generator's result."""
    path = Path(proposed_dir) / card.ref / GENERATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"card": card.model_dump(mode="json"), "result": result.to_dict()}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_generation(ref: str, proposed_dir: Path) -> tuple[Card, GenerationResult]:
    path = Path(proposed_dir) / ref / GENERATION_FILE
    if not path.exists():
        raise CommandError(f"{_rel(path)} not found: run `taskgen generate --cards {ref}` first")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Card.model_validate(data["card"]), GenerationResult.from_dict(data["result"])
    except (ValueError, KeyError, TypeError) as exc:
        raise CommandError(
            f"{_rel(path)} is unreadable ({type(exc).__name__}); run `taskgen generate "
            f"--cards {ref}` again"
        ) from None


def _write_index(proposed_dir: Path, mode: str, source: str, refs: Sequence[str]) -> Path:
    path = Path(proposed_dir) / GENERATE_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {"mode": mode, "source": source, "cards": list(refs)}
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return path


def _index_cards(proposed_dir: Path) -> list[str]:
    path = Path(proposed_dir) / GENERATE_INDEX
    if not path.exists():
        raise CommandError(f"{_rel(path)} not found: run `taskgen generate` first")
    return list(json.loads(path.read_text(encoding="utf-8")).get("cards") or [])


def cmd_snapshot(args: argparse.Namespace) -> int:
    with board_source(args.board) as source:
        path = write_snapshot(source, args.out)
    written = FileSource(path)
    print(
        f"wrote {_rel(path)}: {len(written.snapshot()['cards'])} cards from {args.board}, "
        f"{len(written.cards())} with acceptance criteria"
    )
    return 0


def _record_base(args: argparse.Namespace) -> str:
    """The record command for the same source this run used, without ``--cards``."""
    parts = ["ANTHROPIC_API_KEY=... taskgen generate --mode record"]
    if args.source == "board":
        parts.append(f"--source board --board {args.board}")
    elif Path(args.snapshot).resolve() != DEFAULT_SNAPSHOT.resolve():
        parts.append(f"--snapshot {args.snapshot}")
    if Path(args.recordings).resolve() != RECORDINGS_DIR.resolve():
        parts.append(f"--recordings {args.recordings}")
    return " ".join(parts)


def _record_command(args: argparse.Namespace) -> str:
    """The template a replay error for one card fills in: ``{card}`` is the ref."""
    return _record_base(args).replace("{", "{{").replace("}", "}}") + " --cards {card}"


def _select_cards(args: argparse.Namespace, source: CardSource) -> list[Card]:
    refs = _split(args.cards)
    if refs:
        return [source.card(ref) for ref in refs]
    cards: list[Card] = source.cards()
    if args.mode == "canned":
        have = set(CannedGenerator(args.canned).available())
        cards = [card for card in cards if card.ref in have]
    return cards


MODE_LABELS = {
    "canned": "canned stand-ins (written for the offline demo, not model output)",
    "replay": "recorded model responses (no API call)",
    "record": "the Anthropic API, recording each response",
    "live": "the Anthropic API",
}


def _result_line(result: GenerationResult) -> str:
    if result.ok and result.output is not None:
        tests = len(result.output.tests)
        untestable = len(result.output.untestable)
        line = f"ok · {tests} tests, {untestable} untestable"
        if result.served_by_fallback:
            line += f" · served by the fallback model {result.model}"
        return line
    return f"{result.status} · {result.error}"


def cmd_generate(args: argparse.Namespace) -> int:
    proposed = Path(args.proposed)
    if args.source == "board":
        source: CardSource = board_source(args.board)
        source_name = f"board {args.board}"
    else:
        source = FileSource(args.snapshot)
        source_name = f"snapshot {_rel(Path(args.snapshot))}"
    try:
        cards = _select_cards(args, source)
    finally:
        if isinstance(source, TaskBoardSource):
            source.close()
    if not cards:
        raise CommandError(f"no cards to generate for from {source_name}")
    if args.mode == "replay" and not any(
        (Path(args.recordings) / f"{card.ref}.json").exists() for card in cards
    ):
        record = _record_base(args)
        if args.cards:
            record += f" --cards {','.join(_split(args.cards))}"
        raise CommandError(
            f"{_rel(Path(args.recordings))}/ holds no recording for these cards, and replay mode "
            f"(the default) answers only from recordings. Record them with your own key: {record}. "
            "Or run offline with --mode canned."
        )
    generator = make_generator(
        args.mode,
        recordings=args.recordings,
        canned=args.canned,
        record_command=_record_command(args),
    )
    print(
        f"Generating tests for {len(cards)} card(s) from {source_name} "
        f"with {MODE_LABELS[args.mode]}."
    )
    width = max(len(card.ref) for card in cards)
    saved: list[str] = []
    results: list[GenerationResult] = []
    failed: list[str] = []
    for card in cards:  # one after another: later cards read the prompt cache
        stale = proposed / card.ref / GENERATION_FILE
        try:
            result = generator.generate(card)
        except GeneratorConfigError as exc:
            stale.unlink(missing_ok=True)
            _write_index(proposed, args.mode, source_name, saved)
            _err(f"{card.ref:<{width}}  stopped: {exc}")
            _err("Generation stopped: this would fail for every card.")
            return 1
        except GenerationError as exc:
            stale.unlink(missing_ok=True)
            failed.append(card.ref)
            print(f"{card.ref:<{width}}  no result · {exc}")
            continue
        save_generation(card, result, proposed)
        saved.append(card.ref)
        results.append(result)
        if not result.ok:
            failed.append(card.ref)
        print(f"{card.ref:<{width}}  {_result_line(result)}")
    index = _write_index(proposed, args.mode, source_name, saved)
    cost = cost_line(results)
    if cost:
        print(f"Cost: {cost}")
    print(f"Saved {len(saved)} result(s); {_rel(index)} lists them for `taskgen gate`.")
    if failed:
        _err(f"No tests from: {', '.join(failed)}.")
        return 1
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    proposed = Path(args.proposed)
    refs = _split(args.cards) or _index_cards(proposed)
    if not refs:
        raise CommandError("the last `taskgen generate` run saved no results; nothing to gate")
    pairs = [load_generation(ref, proposed) for ref in refs]
    cards = [card for card, _ in pairs]
    results = [result for _, result in pairs]
    bugs = args.bug or None
    count = len(bugs) if bugs else len(seeded.CATALOG)
    print(
        f"Gating {len(cards)} card(s): the clean build twice (together, then each test on "
        f"its own), then {count} seeded bug(s), "
        f"{args.workers} pytest process(es) at a time."
    )
    try:
        run = gate.gate_results(
            cards, results, proposed_dir=proposed, bugs=bugs, workers=args.workers
        )
    except GateError as exc:
        _err(f"The gate could not run: {exc}")
        return 1
    report.write_reports(run, proposed)
    _print_counts(run)
    print(
        f"Reports: {_rel(proposed / report.RUN_REPORT_FILE)} and "
        f"{_rel(proposed)}/<card>/{gate.REPORT_FILE}"
    )
    return 0


def _parse_expect(value: str) -> dict[Verdict, int]:
    """``"accepted=14,rejected=2"`` -> a count for every verdict; unnamed ones are 0."""
    names = ", ".join(v.value for v in gate.VERDICT_ORDER)
    expected = dict.fromkeys(gate.VERDICT_ORDER, 0)
    for part in _split(value):
        name, _, number = part.partition("=")
        try:
            verdict = Verdict(name.strip())
        except ValueError:
            message = f"{name.strip()!r} is not a verdict ({names})"
            raise argparse.ArgumentTypeError(message) from None
        if not number.strip().isdigit():
            raise argparse.ArgumentTypeError(f"{part!r} needs a count: {verdict.value}=<n>")
        expected[verdict] = int(number)
    return expected


def cmd_report(args: argparse.Namespace) -> int:
    proposed = Path(args.proposed)
    try:
        run = gate.load_run(proposed)
    except FileNotFoundError as exc:
        raise CommandError(str(exc)) from None
    report.write_reports(run, proposed)
    if args.quiet:
        _print_counts(run)
    else:
        print((proposed / report.RUN_REPORT_FILE).read_text(encoding="utf-8"), end="")
    if args.expect is None:
        return 0
    got = run.counts()
    wrong = [
        f"{v.value}: expected {args.expect[v]}, got {got[v]}"
        for v in gate.VERDICT_ORDER
        if got[v] != args.expect[v]
    ]
    if wrong:
        _err("Verdict counts differ from --expect:\n  " + "\n  ".join(wrong))
        return 1
    print(f"Verdict counts match: {_counts_text(got)}.")
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    try:
        done = accepting.accept(
            args.card,
            _split(args.tests) or None,
            proposed_dir=Path(args.proposed),
            generated_dir=GENERATED_DIR,
        )
    except accepting.AcceptError as exc:
        _err(str(exc))
        return 1
    for item in done:
        print(f"{item.status:<9}  {_rel(item.path)}")
    folder = _rel(Path(GENERATED_DIR) / args.card)
    print(f"Next: read them, run `pytest {folder}`, and commit them with the change.")
    return 0


def cmd_comment(args: argparse.Namespace) -> int:
    try:
        card = gate.load_card(args.card, Path(args.proposed))
    except FileNotFoundError as exc:
        raise CommandError(str(exc)) from None
    body = report.card_summary(card)
    print(body, end="")
    if not args.yes:
        print(f"\nNot posted. Add --yes to post this as a comment on {args.card}.")
        return 0
    with board_source(args.board) as source:
        source.post_comment(args.card, body)
    print(f"\nPosted as a comment on {args.card}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskgen",
        description=(
            "Write pytest tests from task-card acceptance criteria with Claude, and keep only "
            "the ones that fail against a seeded bug."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    proposed = argparse.ArgumentParser(add_help=False)
    proposed.add_argument(
        "--proposed",
        type=Path,
        default=DEFAULT_PROPOSED,
        metavar="DIR",
        help="where generation results, proposed tests and reports go (default: proposed/)",
    )

    p = commands.add_parser("snapshot", help="write a board snapshot from the REST API (GET only)")
    p.add_argument("--board", required=True, help="the board slug, e.g. fernhill-front-desk")
    p.add_argument("--out", type=Path, required=True, metavar="PATH", help="the file to write")
    p.set_defaults(handler=cmd_snapshot)

    p = commands.add_parser("generate", parents=[proposed], help="propose tests for each card")
    p.add_argument(
        "--source",
        choices=("file", "board"),
        default="file",
        help="read cards from a snapshot file (default) or from the board's API",
    )
    p.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        metavar="PATH",
        help="the snapshot for --source file (default: examples/fernhill-board.json)",
    )
    p.add_argument(
        "--board",
        default=DEFAULT_BOARD,
        help=f"the board slug for --source board (default: {DEFAULT_BOARD})",
    )
    p.add_argument(
        "--cards",
        metavar="REF,REF",
        help="only these cards (default: every card with criteria; for canned, the canned ones)",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="replay",
        help=(
            "replay recordings (default, no key), record or live (call the API, need "
            "ANTHROPIC_API_KEY), or canned (stand-ins for the offline demo)"
        ),
    )
    p.add_argument(
        "--recordings",
        type=Path,
        default=RECORDINGS_DIR,
        metavar="DIR",
        help="the recordings directory (default: recordings/)",
    )
    p.add_argument(
        "--canned",
        type=Path,
        default=CANNED_DIR,
        metavar="DIR",
        help="the canned stand-ins directory (default: fixtures/canned/)",
    )
    p.set_defaults(handler=cmd_generate)

    p = commands.add_parser(
        "gate", parents=[proposed], help="run proposed tests on the clean build and seeded bugs"
    )
    p.add_argument(
        "--cards", metavar="REF,REF", help="only these cards (default: the last generate run's)"
    )
    p.add_argument(
        "--bug",
        action="append",
        choices=list(seeded.CATALOG),
        metavar="NAME",
        help="run only this seeded bug (repeatable; default: all of them)",
    )
    p.add_argument(
        "--workers", type=_positive, default=4, help="pytest processes at once (default 4)"
    )
    p.set_defaults(handler=cmd_gate)

    p = commands.add_parser(
        "report", parents=[proposed], help="rebuild the reports from gate.json and print RUN.md"
    )
    p.add_argument(
        "--expect",
        type=_parse_expect,
        metavar="VERDICT=N,...",
        help="exit 1 unless the verdict counts are exactly these (unnamed verdicts: 0)",
    )
    p.add_argument("--quiet", action="store_true", help="print the counts, not RUN.md")
    p.set_defaults(handler=cmd_report)

    p = commands.add_parser(
        "accept", parents=[proposed], help="copy accepted tests into tests/generated/<card>/"
    )
    p.add_argument("card", help="the card ref, e.g. FERN-8")
    p.add_argument(
        "--tests", metavar="NAME,NAME", help="only these tests (default: every accepted test)"
    )
    p.set_defaults(handler=cmd_accept)

    p = commands.add_parser(
        "comment", parents=[proposed], help="show a card's summary; post it to the card with --yes"
    )
    p.add_argument("card", help="the card ref, e.g. FERN-8")
    p.add_argument(
        "--board", default=DEFAULT_BOARD, help=f"the board slug (default: {DEFAULT_BOARD})"
    )
    p.add_argument(
        "--yes", action="store_true", help="post the comment (the only write to the board)"
    )
    p.set_defaults(handler=cmd_comment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CommandError, SourceError, ContextLeakError) as exc:
        _err(f"taskgen {args.command}: {exc}")
        return 1
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
