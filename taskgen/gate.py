"""The gate: a generated test counts only once it has been shown to fail against a broken build.

For each card, ``gate_cards``:

1. Runs the static checks (``taskgen.static_checks``), and rejects a proposal that
   reuses a name another proposal for the card already has. A proposal with a problem
   is ``rejected`` and never runs. It is kept, for the report, as
   ``proposed/<card>/rejected/<card>_<k>.py``, a name pytest does not collect.
2. Writes every other proposal to ``proposed/<card>/test_<card>_<k>.py``, where ``k``
   counts the card's proposals from 1 in the order the generator returned them.
3. A card outside the Done list is not built yet: its tests are written for review,
   marked ``not-built-yet`` and not run.
4. Runs the Done cards' tests against the clean build twice: once all together in one
   pytest process, and once each test in a process of its own. A test whose two results
   differ is ``flaky``: it depends on another proposal or on run order. One that does not
   pass both times is ``fails-on-clean``, and a person decides whether the test or the
   app is wrong.
5. Runs the same set of tests as the first clean run, all together, once per seeded bug
   in ``frontdesk.bugs.CATALOG``, with ``SEEDED_BUG=<name>``. Only tests that passed both
   clean runs get a verdict here. A test that fails an assertion (``AssertionError``, or
   pytest's ``Failed`` from ``pytest.raises``) under at least one bug is ``accepted``. Any
   other exception under a bug is a crash: reported, but a broken step is not a catch. A
   test that no bug makes fail on an assertion is ``proves-nothing``.

Each run is ``python -m pytest`` in a subprocess with the harness plugin (loaded by
the repository's pyproject.toml), writing a JUnit XML file, the plugin's results file
and a log to ``proposed/runs/``. Each run starts its own Front Desk server, and carries
a canary test that only checks the app answered: if the canary does not pass, the run
says nothing about the proposals and the gate stops with ``GateError``.

The gate reads ``frontdesk.bugs`` to run the builds, and nothing it produces goes back
to a generator (``taskgen.context`` keeps the catalog out of the prompt). Which bugs each
test caught is for the reviewer, in ``proposed/<card>/gate.json`` and the reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from frontdesk import bugs as seeded
from taskgen.cards import Card, Criterion, card_sort_key
from taskgen.schema import GeneratedTests, ProposedTest, UntestableCriterion
from taskgen.static_checks import Problem, StaticResult, check_proposal

if TYPE_CHECKING:
    from taskgen.generate import GenerationResult

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROPOSED = REPO / "proposed"
GATE_FILE = "gate.json"
RUN_FILE = "run.json"
REPORT_FILE = "REPORT.md"
RUNS_DIR = "runs"
REJECTED_DIR = "rejected"
CANARY_FILE = "canary.py"
CLEAN_TOGETHER = "clean-1"
CLEAN_ALONE = "clean-2"

# How a test fails when an assertion in it does not hold: a plain assert, or pytest's
# Failed, which pytest.raises raises when the expected error does not come.
CATCHING_EXCEPTIONS = frozenset({"AssertionError", "Failed"})

CANARY_CODE = '''\
"""Written by the taskgen gate. It checks only that the app started and answered a call,
so a run whose canary passes is a run whose results mean something."""


def test_gate_canary(call):
    assert call.greeting.text
'''

# Removed from every gate run's environment. FRONTDESK_URL would point the run at a
# server running some other build; the xdist variables would make the harness plugin
# think it is a worker and skip writing its results file; ANTHROPIC_* covers every way
# the SDK finds credentials (keys, tokens, profiles, workload identity).
_DROPPED_ENV = frozenset({"SEEDED_BUG", "FRONTDESK_URL", "PYTEST_CURRENT_TEST"})
_DROPPED_ENV_PREFIXES = ("PYTEST_XDIST", "ANTHROPIC_")


class GateError(RuntimeError):
    """A gate run that produced no usable result: pytest crashed, hung, or the app did not
    start under that build."""


class Verdict(StrEnum):
    ACCEPTED = "accepted"
    PROVES_NOTHING = "proves-nothing"
    FAILS_ON_CLEAN = "fails-on-clean"
    FLAKY = "flaky"
    REJECTED = "rejected"
    NOT_BUILT_YET = "not-built-yet"


VERDICT_ORDER: tuple[Verdict, ...] = tuple(Verdict)

Outcome = Literal["passed", "failed", "error", "skipped", "missing"]


class RunResult(BaseModel):
    """What one proposal did in one pytest run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    exception: str | None = None
    message: str | None = None
    cases: dict[str, str] = Field(
        default_factory=dict,
        description="Each collected test id (a parametrized test has several) and its outcome.",
    )

    def same_as(self, other: RunResult) -> bool:
        """Equal outcomes, case by case. Messages are ignored: they carry object addresses."""
        return self.outcome == other.outcome and self.cases == other.cases

    @property
    def caught(self) -> bool:
        """The test failed on an assertion: it noticed something."""
        return self.outcome == "failed" and self.exception in CATCHING_EXCEPTIONS

    @property
    def crashed(self) -> bool:
        """The test failed in its body on some other exception, such as a TypeError."""
        return self.outcome == "failed" and self.exception not in CATCHING_EXCEPTIONS

    def describe(self) -> str:
        if self.outcome == "passed":
            return "passed"
        detail = self.exception or ""
        message = self.message or ""
        if detail and message.startswith(f"{detail}:"):
            message = message[len(detail) + 1 :].strip()
        if message:
            detail = f"{detail}: {message}" if detail else message
        return f"{self.outcome} ({detail})" if detail else self.outcome


class TestOutcome(BaseModel):
    """One proposed test and what the gate decided about it."""

    __test__ = False  # not a pytest test class, whatever its name

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    criterion: str
    name: str
    intent: str
    code: str
    code_sha256: str
    file: str = Field(description="Path relative to the proposed directory.")
    verdict: Verdict
    reason: str
    problems: tuple[Problem, ...] = ()
    warnings: tuple[Problem, ...] = ()
    clean: tuple[RunResult, ...] = ()
    bugs: dict[str, RunResult] = Field(default_factory=dict)
    caught: tuple[str, ...] = ()
    crashed: tuple[str, ...] = Field(
        default=(),
        description="Seeded bugs under which the test failed on an exception other than an "
        "assertion. Reported, never counted as a catch.",
    )


class CardGate(BaseModel):
    """Everything the gate found for one card. Saved as ``proposed/<card>/gate.json``.

    ``generation`` is the generator's record for the card (``GenerationResult.to_dict()``
    without its output): where the tests came from, the model, usage and cost, or the
    reason generation produced nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    card: str
    title: str
    list_role: str
    gated: bool
    criteria: tuple[Criterion, ...]
    tests: tuple[TestOutcome, ...] = ()
    untestable: tuple[UntestableCriterion, ...] = ()
    seeded_bugs: tuple[str, ...] = ()
    generation: dict[str, Any] | None = None

    @property
    def source(self) -> str | None:
        return (self.generation or {}).get("source")

    @property
    def generation_error(self) -> str | None:
        generation = self.generation or {}
        if generation.get("status", "ok") == "ok":
            return None
        return generation.get("error") or str(generation.get("status"))

    def criterion_text(self, criterion_id: str) -> str:
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion.text
        return ""

    def counts(self) -> dict[Verdict, int]:
        return {v: sum(1 for t in self.tests if t.verdict == v) for v in VERDICT_ORDER}

    def accepted(self) -> list[TestOutcome]:
        return [t for t in self.tests if t.verdict == Verdict.ACCEPTED]

    def uncovered(self) -> list[Criterion]:
        """Criteria left with no accepted test."""
        covered = {t.criterion for t in self.accepted()}
        return [c for c in self.criteria if c.id not in covered]

    def test(self, name: str) -> TestOutcome:
        for test in self.tests:
            if test.name == name:
                return test
        known = ", ".join(t.name for t in self.tests) or "none"
        raise KeyError(f"{self.card} has no proposed test {name!r} (it has {known})")


class GateRun(BaseModel):
    """The gate's result for every card in one invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cards: tuple[CardGate, ...]
    seeded_bugs: tuple[str, ...]

    def counts(self) -> dict[Verdict, int]:
        return {v: sum(c.counts()[v] for c in self.cards) for v in VERDICT_ORDER}

    def card(self, ref: str) -> CardGate:
        for card in self.cards:
            if card.card == ref:
                return card
        raise KeyError(f"this gate run has no card {ref}")

    def write(self, proposed_dir: Path = DEFAULT_PROPOSED) -> list[Path]:
        """Save each card's ``gate.json`` and the run's ``run.json``. Returns the paths."""
        proposed_dir = Path(proposed_dir)
        paths = [write_card(card, proposed_dir) for card in self.cards]
        index = {"cards": [c.card for c in self.cards], "seeded_bugs": list(self.seeded_bugs)}
        path = proposed_dir / RUN_FILE
        path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return [*paths, path]


def write_card(card: CardGate, proposed_dir: Path = DEFAULT_PROPOSED) -> Path:
    path = Path(proposed_dir) / card.card / GATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_card(ref: str, proposed_dir: Path = DEFAULT_PROPOSED) -> CardGate:
    """Read ``proposed/<ref>/gate.json``. Raises FileNotFoundError naming the command to run."""
    path = Path(proposed_dir) / ref / GATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `taskgen gate` for {ref} first")
    return CardGate.model_validate_json(path.read_text(encoding="utf-8"))


def load_run(proposed_dir: Path = DEFAULT_PROPOSED) -> GateRun:
    """The last gate run under ``proposed_dir`` (from ``run.json``), in board order."""
    proposed_dir = Path(proposed_dir)
    index_path = proposed_dir / RUN_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found: run `taskgen gate` first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cards = sorted((load_card(ref, proposed_dir) for ref in index["cards"]), key=_board_order)
    return GateRun(cards=tuple(cards), seeded_bugs=tuple(index["seeded_bugs"]))


def _board_order(card: CardGate) -> tuple[str, int]:
    return card_sort_key(card.card)


def decide(
    *,
    static_ok: bool,
    static_reason: str = "",
    gated: bool,
    list_role: str = "",
    clean: Sequence[RunResult] = (),
    bug_results: Mapping[str, RunResult] | None = None,
) -> tuple[Verdict, str, tuple[str, ...]]:
    """The verdict for one test, its reason in a sentence, and the bugs it caught.

    ``clean`` is the clean run with every proposal together, then the clean run with the
    test on its own. A bug counts as caught when the test fails on an assertion under
    that bug (``RunResult.caught``). A crash under a bug (a TypeError, a KeyError) is
    not a catch: the step that broke may have nothing to do with what the test asserts.
    Nor is a setup error (outcome "error"): it says the build would not run the test,
    not that the test noticed anything.
    """
    if not static_ok:
        return (
            Verdict.REJECTED,
            f"Rejected by the static checks, so it never ran: {static_reason}",
            (),
        )
    if not gated:
        where = f"in {list_role}" if list_role else "not in Done"
        return (
            Verdict.NOT_BUILT_YET,
            f"The card is {where}, so the behavior may not be built. Written for review; not run.",
            (),
        )
    if len(clean) < 2:
        raise ValueError("decide() needs both clean runs for a gated test")
    first, second = clean[0], clean[1]
    if not first.same_as(second):
        return (
            Verdict.FLAKY,
            f"The two clean runs disagree: with the other proposals it {first.describe()}, on "
            f"its own it {second.describe()}. Its result depends on other tests or on when "
            "it runs, so it would not hold once accepted.",
            (),
        )
    if first.outcome != "passed":
        return (
            Verdict.FAILS_ON_CLEAN,
            f"Fails on the clean build: {first.describe()}. The test or the app is wrong; "
            "a person decides which.",
            (),
        )
    results = dict(bug_results or {})
    caught = tuple(name for name, result in results.items() if result.caught)
    if caught:
        return (
            Verdict.ACCEPTED,
            f"Passes on the clean build twice and fails under {len(caught)} of "
            f"{len(results)} seeded bugs.",
            caught,
        )
    crashed = {name: result for name, result in results.items() if result.crashed}
    if crashed:
        kinds = ", ".join(
            sorted({r.exception or "an unrecorded exception" for r in crashed.values()})
        )
        return (
            Verdict.PROVES_NOTHING,
            f"Passes on the clean build twice, and no seeded bug made an assertion fail. Under "
            f"{', '.join(crashed)} it crashed ({kinds}) instead, which shows a step broke, not "
            "that the test checks anything.",
            (),
        )
    under = "the one seeded bug run" if len(results) == 1 else f"all {len(results)} seeded bugs"
    return (
        Verdict.PROVES_NOTHING,
        f"Passes on the clean build and under {under}, so it would not notice any of those "
        "mistakes.",
        (),
    )


Runner = Callable[[Sequence[str], str | None, str], dict[str, RunResult]]
"""``runner(files, bug, label) -> {file: RunResult}``. ``files`` are relative to the rootdir."""


def gate_env(bug: str | None, repo: Path = REPO) -> dict[str, str]:
    """The environment for one gate run: this process's, minus model credentials and any
    earlier build choice, with ``SEEDED_BUG`` set for a bug run."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _DROPPED_ENV and not key.startswith(_DROPPED_ENV_PREFIXES)
    }
    if bug is not None:
        env["SEEDED_BUG"] = bug
    paths = [str(repo), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@dataclass
class PytestRunner:
    """Runs proposals in a ``python -m pytest`` subprocess, one process per build.

    ``rootdir`` is the directory that holds the proposed directory; test ids and JUnit
    file names are relative to it. The repository's pyproject.toml supplies the harness
    plugin and the import mode. Every run also carries the canary test.
    """

    rootdir: Path
    runs_dir: Path
    repo: Path = REPO
    timeout: float = 600.0
    python: str = sys.executable

    def __post_init__(self) -> None:
        # Written once, before any run starts: runs go in parallel and must not race on it.
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / CANARY_FILE).write_text(CANARY_CODE, encoding="utf-8")

    @property
    def canary(self) -> str:
        return (self.runs_dir / CANARY_FILE).relative_to(self.rootdir).as_posix()

    def command(self, files: Sequence[str], label: str) -> list[str]:
        return [
            self.python,
            "-m",
            "pytest",
            "-c",
            str(self.repo / "pyproject.toml"),
            "--rootdir",
            str(self.rootdir),
            self.canary,
            *files,
            "-q",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            "-o",
            "junit_family=xunit1",
            # Full values in assertion messages, so a reason reads "assert 'what the app
            # said' == 'what the test expected'" instead of two clipped strings.
            "-o",
            "verbosity_assertions=2",
            f"--junitxml={self.runs_dir / f'{label}.xml'}",
            f"--write-results={self.runs_dir / f'{label}.json'}",
        ]

    def __call__(self, files: Sequence[str], bug: str | None, label: str) -> dict[str, RunResult]:
        junit = self.runs_dir / f"{label}.xml"
        results = self.runs_dir / f"{label}.json"
        log = self.runs_dir / f"{label}.log"
        for stale in (junit, results):
            stale.unlink(missing_ok=True)
        try:
            with log.open("wb") as out:
                done = subprocess.run(
                    self.command(files, label),
                    cwd=self.rootdir,
                    env=gate_env(bug, self.repo),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise GateError(
                f"run {label} did not finish within {self.timeout:.0f}s; see {log}"
            ) from exc
        if done.returncode not in (0, 1) or not junit.exists():
            wrote = "" if junit.exists() else " and wrote no JUnit file"
            raise GateError(
                f"run {label}: pytest exited with code {done.returncode}{wrote}. "
                f"Last lines of {log}:\n{_tail(log)}"
            )
        found = parse_junit(junit, results, [self.canary, *files])
        canary = found.pop(self.canary)
        if canary.outcome != "passed":
            raise GateError(
                f"run {label}: the canary test did not pass ({canary.describe()}), so the app "
                f"did not answer under this build. Last lines of {log}:\n{_tail(log)}"
            )
        return found


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


_E_LINE = re.compile(r"^E\s+(\S.*)$", re.MULTILINE)


def _first_line(text: str | None, limit: int = 400) -> str | None:
    if not text:
        return None
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 3] + "..."


Case = tuple[str, str, str | None, str | None]  # name, outcome, exception, message


def parse_junit(junit: Path, results: Path, files: Iterable[str]) -> dict[str, RunResult]:
    """Fold one run's JUnit XML (and the harness results file) into one result per file.

    A file whose tests all pass has outcome "passed". Otherwise the worst case decides:
    any failure in a test body is "failed", then setup or collection errors are "error",
    then skips. A file with no test case at all is "missing".
    """
    exceptions: dict[str, Any] = {}
    if results.exists():
        exceptions = json.loads(results.read_text(encoding="utf-8"))
    cases: dict[str, list[Case]] = {f: [] for f in files}
    for case in ET.parse(junit).getroot().iter("testcase"):
        file = case.get("file") or ""
        if file not in cases:
            continue
        name = case.get("name") or ""
        failure, error, skipped = (case.find(tag) for tag in ("failure", "error", "skipped"))
        exception: str | None = (exceptions.get(f"{file}::{name}") or {}).get("exception")
        message: str | None = None
        if failure is not None:
            outcome = "failed"
            message = _first_line(failure.get("message"))
        elif error is not None:
            outcome = "error"
            found = _E_LINE.search(error.text or "")
            message = _first_line(found.group(1) if found else error.get("message"))
            if exception is None and found:
                exception = found.group(1).split(":", 1)[0].strip() or None
            if not name or name.replace(".", "/") + ".py" == file:
                name = "(collection)"
        elif skipped is not None:
            outcome = "skipped"
            message = _first_line(skipped.get("message"))
        else:
            outcome = "passed"
        cases[file].append((name, outcome, exception, message))
    return {file: _fold(found) for file, found in cases.items()}


def _fold(found: list[Case]) -> RunResult:
    if not found:
        return RunResult(outcome="missing", message="pytest reported no test for this file")
    cases = {name: result for name, result, _, _ in found}
    for worst in ("failed", "error", "skipped"):
        hits = [f for f in found if f[1] == worst]
        # A parametrized test that failed an assertion in any case is reported by that
        # case, so a crash in an earlier case cannot hide a real catch.
        hits.sort(key=lambda f: f[2] not in CATCHING_EXCEPTIONS)
        if hits:
            _, outcome, exception, message = hits[0]
            return RunResult.model_validate(
                {"outcome": outcome, "exception": exception, "message": message, "cases": cases}
            )
    return RunResult(outcome="passed", cases=cases)


@dataclass(frozen=True)
class GateInput:
    """One card and what the generator returned for it.

    ``generated`` is None when generation produced nothing for the card (a refusal, an
    output cut off at max_tokens, invalid JSON); ``generation`` then says why.
    """

    card: Card
    generated: GeneratedTests | None
    generation: GenerationResult | None = None


@dataclass
class _Proposal:
    card: Card
    index: int
    test: ProposedTest
    path: Path
    rel: str  # relative to the rootdir, for pytest
    static_ok: bool
    static_reason: str
    problems: tuple[Problem, ...]
    warnings: tuple[Problem, ...]


def _alone_label(proposal: _Proposal) -> str:
    """The label of the clean run that runs this proposal on its own."""
    return f"{CLEAN_ALONE}-{proposal.card.ref}_{proposal.index}"


def _duplicate_name(test: ProposedTest, card: Card) -> Problem:
    return Problem(
        rule="name",
        message=(
            f"another proposal for {card.ref} is already named {test.name}; every test needs "
            f"its own name, because it is kept as tests/generated/{card.ref}/<name>.py"
        ),
    )


def _bug_names(bugs: Sequence[str] | None) -> tuple[str, ...]:
    if bugs is None:
        return tuple(seeded.CATALOG)
    unknown = [b for b in bugs if b not in seeded.CATALOG]
    if unknown:
        raise ValueError(f"not seeded bugs: {', '.join(unknown)}")
    return tuple(dict.fromkeys(bugs))


def _clear(card_dir: Path, ref: str) -> None:
    """Remove what an earlier gate run wrote for this card, and nothing else."""
    for old in card_dir.glob(f"test_{ref}_*.py"):
        old.unlink()
    shutil.rmtree(card_dir / REJECTED_DIR, ignore_errors=True)
    for name in (GATE_FILE, REPORT_FILE):
        (card_dir / name).unlink(missing_ok=True)


def proposal_path(proposed_dir: Path, ref: str, index: int, *, rejected: bool = False) -> Path:
    if rejected:
        return Path(proposed_dir) / ref / REJECTED_DIR / f"{ref}_{index}.py"
    return Path(proposed_dir) / ref / f"test_{ref}_{index}.py"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_text(code: str) -> str:
    """The proposal's code exactly as the gate writes it: with one trailing newline."""
    return code if code.endswith("\n") else code + "\n"


def _as_input(item: GateInput | tuple[Card, GeneratedTests]) -> GateInput:
    if isinstance(item, GateInput):
        return item
    card, generated = item
    return GateInput(card=card, generated=generated)


def _generation_record(result: GenerationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    record = result.to_dict()
    record["output"] = None  # the tests are in the gate record already
    return record


def gate_cards(
    items: Sequence[GateInput | tuple[Card, GeneratedTests]],
    *,
    proposed_dir: Path = DEFAULT_PROPOSED,
    bugs: Sequence[str] | None = None,
    workers: int = 4,
    runner: Runner | None = None,
    repo: Path = REPO,
    timeout: float = 600.0,
) -> GateRun:
    """Gate every card's proposals, and save each card's ``gate.json`` and the run's
    ``run.json`` under ``proposed_dir``.

    ``items`` pairs each card with the generator's output for it, as ``GateInput`` or a
    ``(card, generated)`` tuple. ``bugs`` limits the seeded bugs to run (default: the
    whole catalog; the gate never needs to know which card a bug belongs to).
    ``workers`` is how many pytest processes run at once.
    """
    inputs = [_as_input(item) for item in items]
    proposed_dir = Path(proposed_dir).resolve()
    rootdir = proposed_dir.parent
    bug_names = _bug_names(bugs)
    if runner is None:
        shutil.rmtree(proposed_dir / RUNS_DIR, ignore_errors=True)  # last run's raw files
        runner = PytestRunner(
            rootdir=rootdir, runs_dir=proposed_dir / RUNS_DIR, repo=repo, timeout=timeout
        )
    run = runner

    proposals: list[_Proposal] = []
    for item in inputs:
        card, generated = item.card, item.generated
        if generated is not None and generated.card != card.ref:
            raise ValueError(f"generated tests are for {generated.card}, not {card.ref}")
        card_dir = proposed_dir / card.ref
        card_dir.mkdir(parents=True, exist_ok=True)
        _clear(card_dir, card.ref)
        names: set[str] = set()
        for index, test in enumerate(generated.tests if generated else [], start=1):
            static = check_proposal(test, card)
            if test.name in names:
                static = StaticResult(
                    problems=(*static.problems, _duplicate_name(test, card)),
                    warnings=static.warnings,
                    function=static.function,
                    marker=static.marker,
                )
            names.add(test.name)
            path = proposal_path(proposed_dir, card.ref, index, rejected=not static.ok)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source_text(test.code), encoding="utf-8")
            proposals.append(
                _Proposal(
                    card=card,
                    index=index,
                    test=test,
                    path=path,
                    rel=path.relative_to(rootdir).as_posix(),
                    static_ok=static.ok,
                    static_reason=static.reason,
                    problems=static.problems,
                    warnings=static.warnings,
                )
            )

    gated = [p for p in proposals if p.static_ok and p.card.is_done]
    runnable = [p.rel for p in gated]
    clean: dict[str, list[RunResult]] = {rel: [] for rel in runnable}
    if runnable:
        # Every test together, then each on its own: a test that only passes next to
        # another proposal comes out flaky.
        jobs = [(CLEAN_TOGETHER, runnable), *((_alone_label(p), [p.rel]) for p in gated)]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            runs = list(pool.map(lambda job: run(job[1], None, job[0]), jobs))
        together, alone = runs[0], runs[1:]
        for rel in runnable:
            clean[rel].append(together.get(rel, RunResult(outcome="missing")))
        for (_, files), result in zip(jobs[1:], alone, strict=True):
            clean[files[0]].append(result.get(files[0], RunResult(outcome="missing")))

    survivors = [
        rel
        for rel in runnable
        if clean[rel][0].outcome == "passed" and clean[rel][0].same_as(clean[rel][1])
    ]
    under: dict[str, dict[str, RunResult]] = {rel: {} for rel in survivors}
    if survivors and bug_names:
        # Bug runs take the same file list as the first clean run, so the neighbours match.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            by_bug = list(pool.map(lambda bug: run(runnable, bug, f"bug-{bug}"), bug_names))
        for bug, result in zip(bug_names, by_bug, strict=True):
            for rel in survivors:
                under[rel][bug] = result.get(rel, RunResult(outcome="missing"))

    cards: list[CardGate] = []
    for item in inputs:
        card = item.card
        outcomes: list[TestOutcome] = []
        for p in (p for p in proposals if p.card.ref == card.ref):
            verdict, reason, caught = decide(
                static_ok=p.static_ok,
                static_reason=p.static_reason,
                gated=card.is_done,
                list_role=card.list_role,
                clean=clean.get(p.rel, ()),
                bug_results=under.get(p.rel, {}),
            )
            outcomes.append(
                TestOutcome(
                    index=p.index,
                    criterion=p.test.criterion,
                    name=p.test.name,
                    intent=p.test.intent,
                    code=p.test.code,
                    code_sha256=sha256(source_text(p.test.code)),
                    file=p.path.relative_to(proposed_dir).as_posix(),
                    verdict=verdict,
                    reason=reason,
                    problems=p.problems,
                    warnings=p.warnings,
                    clean=tuple(clean.get(p.rel, ())),
                    bugs=under.get(p.rel, {}),
                    caught=caught,
                    crashed=tuple(
                        name for name, result in under.get(p.rel, {}).items() if result.crashed
                    ),
                )
            )
        cards.append(
            CardGate(
                card=card.ref,
                title=card.title,
                list_role=card.list_role,
                gated=card.is_done,
                criteria=card.criteria,
                tests=tuple(outcomes),
                untestable=tuple(item.generated.untestable) if item.generated else (),
                seeded_bugs=bug_names if card.is_done else (),
                generation=_generation_record(item.generation),
            )
        )
    result = GateRun(cards=tuple(cards), seeded_bugs=bug_names)
    result.write(proposed_dir)
    return result


def gate_results(
    cards: Iterable[Card], results: Iterable[GenerationResult], **options: Any
) -> GateRun:
    """``gate_cards`` from the generator's results: each result is paired with its card.

    A result that is not ok (a refusal, a cut-off output, invalid JSON) still gets a
    ``gate.json`` and a report that says why it has no tests.
    """
    by_ref = {card.ref: card for card in cards}
    inputs = []
    for result in results:
        if result.card not in by_ref:
            raise KeyError(f"no card {result.card} among the cards given to the gate")
        inputs.append(GateInput(by_ref[result.card], result.output, result))
    return gate_cards(inputs, **options)
