"""The pytest plugin every Front Desk test runs on.

pyproject.toml loads it for every run in this repository (``-p harness.conftest``
in addopts). Elsewhere, pass ``-p harness.conftest``.

Fixtures
    frontdesk_url   Base URL of the app (session scope, one server per xdist worker).
                    Uses FRONTDESK_URL when set; otherwise starts
                    ``uvicorn frontdesk.main:app`` on a free port with
                    FRONTDESK_TEST_HOOKS=1 and this process's environment.
    frontdesk       A harness.FrontDesk client for that URL (session scope).
    start_call      ``start_call(at=DEFAULT_AT, model_available=True) -> Call``.
    call            A call started with the defaults.
    api             A plain httpx.Client on the base URL, for status codes and raw bodies.

Markers
    ``@pytest.mark.criterion("FERN-<n>/AC-<k>")`` names the acceptance criterion
    a test checks. A malformed one stops the run before any test executes.

Options
    ``--write-results=PATH`` writes ``{nodeid: {"outcome", "exception"}}`` as JSON
    when the run ends: outcome is passed, failed (in the test body), error (in a
    fixture) or skipped, and exception is the exception's type name, such as
    AssertionError. Works with pytest-xdist.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.client import Call, FrontDesk
from harness.world import DEFAULT_AT

HEALTH_TIMEOUT_S = 20.0
CRITERION_ID = re.compile(r"^[A-Z][A-Z0-9]*-\d+/AC-\d+$")
_RESULTS: dict[str, dict[str, Any]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--write-results",
        metavar="PATH",
        default=None,
        help="write {nodeid: {outcome, exception}} as JSON when the run ends",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "criterion(id): the acceptance criterion a test checks, written FERN-<n>/AC-<k>",
    )
    # The app refuses to start on an unknown build name; say so before any test runs.
    from frontdesk import bugs

    try:
        bugs.current()
    except bugs.UnknownBugError as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    problems = []
    for item in items:
        for mark in item.iter_markers("criterion"):
            args = mark.args
            if len(args) != 1 or not isinstance(args[0], str) or not CRITERION_ID.match(args[0]):
                problems.append(f"{item.nodeid}: criterion{args!r} is not 'FERN-<n>/AC-<k>'")
    if problems:
        raise pytest.UsageError("Bad @pytest.mark.criterion:\n  " + "\n  ".join(problems))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if call.excinfo is not None:
        report.user_properties.append(("exception", call.excinfo.typename))
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    exception = dict(report.user_properties).get("exception")
    entry = _RESULTS.setdefault(report.nodeid, {"outcome": "passed", "exception": None})
    if report.when == "call" and report.failed:
        entry.update(outcome="failed", exception=exception)
    elif report.failed:
        entry.update(outcome="error", exception=exception)
    elif report.skipped and entry["outcome"] == "passed":
        entry.update(outcome="skipped")


def pytest_sessionfinish(session: pytest.Session) -> None:
    target = session.config.getoption("--write-results")
    if not target or os.environ.get("PYTEST_XDIST_WORKER"):
        return  # with xdist, the controller sees every report and writes once
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(_RESULTS.items())), indent=2) + "\n", encoding="utf-8")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no server log)"


def _expected_build() -> str | None:
    return os.environ.get("SEEDED_BUG", "").strip() or None


def _check_build(url: str) -> None:
    """Refuse to run against a server whose build differs from this process's."""
    try:
        response = httpx.get(f"{url}/test/build", timeout=5.0)
    except httpx.HTTPError as exc:
        pytest.fail(f"{url}/test/build did not answer: {exc}", pytrace=False)
    if response.status_code == 404:
        pytest.fail(
            f"The app at {url} has test hooks off. Start it with FRONTDESK_TEST_HOOKS=1.",
            pytrace=False,
        )
    running = response.json().get("seededBug")
    if running != _expected_build():
        pytest.fail(
            f"The app at {url} runs build {running!r} but this run expects "
            f"{_expected_build()!r}. Results would be meaningless.",
            pytrace=False,
        )


def _wait_until_healthy(proc: subprocess.Popen[bytes], url: str, log: Path) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while True:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            pytest.fail(
                f"The app did not answer {url}/health within {HEALTH_TIMEOUT_S:.0f}s.\n"
                f"Server log ({log}):\n{_tail(log)}",
                pytrace=False,
            )
        try:
            code = proc.wait(timeout=0.1)  # doubles as the pause between polls
        except subprocess.TimeoutExpired:
            continue
        pytest.fail(
            f"The app exited with code {code} before it was healthy.\n"
            f"Server log ({log}):\n{_tail(log)}",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def frontdesk_url(pytestconfig: pytest.Config) -> Iterator[str]:
    """Base URL of the app under test, without a trailing slash."""
    external = os.environ.get("FRONTDESK_URL", "").strip()
    if external:
        url = external.rstrip("/")
        _check_build(url)
        yield url
        return

    root = Path(pytestconfig.rootpath)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log_path = reports / f"server-{worker}-{os.getpid()}.log"
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "FRONTDESK_TEST_HOOKS": "1"}
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "frontdesk.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_until_healthy(proc, url, log_path)
        _check_build(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture(scope="session")
def frontdesk(frontdesk_url: str) -> Iterator[FrontDesk]:
    """A client for the app under test."""
    client = FrontDesk(frontdesk_url)
    yield client
    client.close()


@pytest.fixture
def start_call(frontdesk: FrontDesk) -> Callable[..., Call]:
    """``start_call(at=DEFAULT_AT, model_available=True) -> Call``. Each call is independent."""

    def start(at: str | datetime | None = DEFAULT_AT, model_available: bool = True) -> Call:
        return frontdesk.start_call(at=at, model_available=model_available)

    return start


@pytest.fixture
def call(start_call: Callable[..., Call]) -> Call:
    """A call at DEFAULT_AT (Wednesday 10 AM, staffed hours) with the model available."""
    return start_call()


@pytest.fixture
def api(frontdesk_url: str) -> Iterator[httpx.Client]:
    """A plain HTTP client on the app's base URL."""
    with httpx.Client(base_url=frontdesk_url, timeout=10.0) as client:
        yield client
