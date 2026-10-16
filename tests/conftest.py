"""Rules shared by the suites under tests/.

* Every ANTHROPIC_* variable is removed from the environment when this file loads
  (keys, tokens, profiles, workload identity), and ANTHROPIC_BASE_URL is then set
  to a closed local port. The SDK can also find credentials in an on-disk profile
  (~/.config/anthropic); a client that did would send its request to 127.0.0.1:9
  and fail there. No test, and no subprocess a test starts, can reach the API, and
  every model response in the suites is fake or canned. TASKGEN_MODEL goes too,
  so results do not depend on the developer's shell.
* An unknown SEEDED_BUG stops the run before any test executes (the harness plugin checks).
* ``@pytest.mark.catches("<bug>", ...)`` names the seeded bugs a test in the
  reference suite is meant to catch. An unknown name stops the run.
* ``--write-markers=PATH`` writes ``{"catches": {bug: [nodeid]}, "criteria":
  {criterion: [nodeid]}}`` once collection finishes. Works with --collect-only.
* ``--only-catching=BUG`` runs only the tests marked ``catches(BUG)``.
* ``scripts/check_bugs_are_real.py`` reads the markers file, and the
  ``--write-results`` file that the harness plugin writes.

The Front Desk fixtures (``call``, ``start_call``, ``api``...) come from the
harness plugin, loaded for every run by pyproject.toml.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from frontdesk import bugs

NO_API_URL = "http://127.0.0.1:9"  # the discard port: nothing listens there

for _name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
    del os.environ[_name]
os.environ.pop("TASKGEN_MODEL", None)
os.environ["ANTHROPIC_BASE_URL"] = NO_API_URL


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--write-markers",
        metavar="PATH",
        default=None,
        help="write the catches and criterion markers of the selected tests as JSON",
    )
    parser.addoption(
        "--only-catching",
        metavar="BUG",
        default=None,
        help="run only the tests marked @pytest.mark.catches(BUG)",
    )


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [f"seeded bug: {bugs.current() or 'none'}"]


def _marker_args(item: pytest.Item, name: str) -> list[str]:
    found: list[str] = []
    for mark in item.iter_markers(name):
        for arg in mark.args:
            if arg not in found:
                found.append(arg)
    return found


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    problems: list[str] = []
    for item in items:
        for mark in item.iter_markers("catches"):
            if not mark.args:
                problems.append(f"{item.nodeid}: @pytest.mark.catches needs a bug name")
            for name in mark.args:
                if not isinstance(name, str) or name not in bugs.CATALOG:
                    problems.append(f"{item.nodeid}: {name!r} is not in frontdesk.bugs.CATALOG")
    only = config.getoption("--only-catching")
    if only is not None and only not in bugs.CATALOG:
        problems.append(f"--only-catching={only!r} is not in frontdesk.bugs.CATALOG")
    if problems:
        raise pytest.UsageError(
            "Bad @pytest.mark.catches:\n  "
            + "\n  ".join(problems)
            + f"\nKnown bugs: {', '.join(bugs.CATALOG)}"
        )
    if only is not None:
        keep = [item for item in items if only in _marker_args(item, "catches")]
        dropped = [item for item in items if only not in _marker_args(item, "catches")]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = keep


def pytest_collection_finish(session: pytest.Session) -> None:
    target = session.config.getoption("--write-markers")
    if not target or os.environ.get("PYTEST_XDIST_WORKER"):
        return
    catches: dict[str, list[str]] = {name: [] for name in bugs.CATALOG}
    criteria: dict[str, list[str]] = {}
    for item in session.items:
        for name in _marker_args(item, "catches"):
            catches[name].append(item.nodeid)
        for criterion in _marker_args(item, "criterion"):
            criteria.setdefault(criterion, []).append(item.nodeid)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"catches": catches, "criteria": dict(sorted(criteria.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
