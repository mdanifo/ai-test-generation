"""Prove the Front Desk reference suite can fail.

1. Every seeded bug has at least one test marked ``@pytest.mark.catches(bug)``,
   and every criterion on a Done card has at least one test marked with it.
2. The whole suite passes against the clean app.
3. For each bug, every test that claims it fails under ``SEEDED_BUG=<bug>``,
   and fails on an assertion (AssertionError), not on a crash or a setup error.

    python scripts/check_bugs_are_real.py
    python scripts/check_bugs_are_real.py --bug double-booking --bug ssml-unescaped

Exit status 0 when every check holds, 1 otherwise. Raw results go to
reports/bugs/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from frontdesk import bugs  # noqa: E402

SUITE = "tests/frontdesk"
OUT = REPO / "reports" / "bugs"
RIGHT_REASON = "AssertionError"


def _pytest(args: list[str], bug: str | None) -> int:
    env = {k: v for k, v in os.environ.items() if k not in ("SEEDED_BUG", "FRONTDESK_URL")}
    if bug:
        env["SEEDED_BUG"] = bug
    command = [sys.executable, "-m", "pytest", SUITE, "-q", "-p", "no:cacheprovider", *args]
    done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if done.returncode not in (0, 1):  # 1 is "some tests failed", which a bug run expects
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-4000:])
    return done.returncode


def _done_criteria(board: Path) -> list[str]:
    data = json.loads(board.read_text(encoding="utf-8"))
    key = data.get("board", {}).get("key", "FERN")
    wanted = []
    for number, card in enumerate(data["cards"], start=1):
        ref = card.get("ref") or f"{key}-{number}"
        if card.get("list") != "done":
            continue
        criteria = card.get("criteria") or []
        wanted += [f"{ref}/AC-{k}" for k in range(1, len(criteria) + 1)]
    return wanted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--board", type=Path, default=REPO / "examples" / "fernhill-board.json")
    parser.add_argument("--bug", action="append", choices=list(bugs.CATALOG), default=None)
    parser.add_argument("--workers", default="4", help="pytest-xdist workers for the clean run")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    markers_path = OUT / "markers.json"
    if _pytest(["--collect-only", f"--write-markers={markers_path}"], None) != 0:
        print("collection failed")
        return 1
    markers = json.loads(markers_path.read_text())
    catches: dict[str, list[str]] = markers["catches"]
    criteria: dict[str, list[str]] = markers["criteria"]

    for name, tests in catches.items():
        if not tests:
            problems.append(f"{name}: no test is marked catches({name!r})")
    if not args.board.exists():
        problems.append(f"board snapshot {args.board} not found (pass --board)")
    else:
        for criterion in _done_criteria(args.board):
            if not criteria.get(criterion):
                problems.append(f"{criterion}: no test in {SUITE} checks this Done criterion")

    clean_path = OUT / "clean.json"
    _pytest(["-n", args.workers, f"--write-results={clean_path}"], None)
    clean = json.loads(clean_path.read_text())
    not_passing = sorted(n for n, r in clean.items() if r["outcome"] != "passed")
    problems += [f"clean build: {n} did not pass" for n in not_passing]
    print(f"clean build: {len(clean) - len(not_passing)} of {len(clean)} tests passed")

    print(f"\n{'seeded bug':34} {'claims':>6} {'failed':>6}  verdict")
    for name in args.bug or list(bugs.CATALOG):
        path = OUT / f"{name}.json"
        _pytest([f"--only-catching={name}", f"--write-results={path}"], name)
        results = json.loads(path.read_text()) if path.exists() else {}
        claimed = catches.get(name, [])
        right = [
            n
            for n in claimed
            if results.get(n, {}).get("outcome") == "failed"
            and results[n].get("exception") == RIGHT_REASON
        ]
        for nodeid in claimed:
            result = results.get(nodeid, {"outcome": "not run", "exception": None})
            if nodeid in right:
                continue
            reason = result["exception"] or result["outcome"]
            problems.append(f"{name}: {nodeid} did not fail on an assertion ({reason})")
        verdict = "caught" if claimed and len(right) == len(claimed) else "NOT PROVEN"
        print(f"{name:34} {len(claimed):>6} {len(right):>6}  {verdict}")

    if problems:
        print("\nProblems:")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print("\nEvery seeded bug is caught by every test that claims it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
