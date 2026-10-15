"""Static checks on generated tests: every rule rejects with a reason a reviewer can act on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgen.cards import Card, number_criteria
from taskgen.context import find_leaks
from taskgen.schema import GeneratedTests, ProposedTest
from taskgen.static_checks import ALLOWED_IMPORTS, Problem, check_code, check_proposal

REPO = Path(__file__).resolve().parents[2]
CANNED = REPO / "fixtures" / "canned"

CARD = Card(
    ref="FERN-8",
    title="Hand off to a person on request or when the agent is stuck",
    list_role="done",
    criteria=number_criteria(["Asking for a person transfers.", "Two unclear turns offer one."]),
)

GOOD = """\
import pytest

from harness import TRANSFER_LINE


@pytest.mark.criterion("FERN-8/AC-1")
@pytest.mark.parametrize("said", ["agent", "representative"])
def test_asking_for_a_person_transfers(call, said):
    reply = call.say(said)

    assert reply.text == TRANSFER_LINE
    assert [e["reason"] for e in call.events("transfer") if e] == ["caller_request"]
"""


def proposal(*lines: str, imports: str = "", decorators: str = "") -> str:
    """A proposal with the given body lines, marker and imports filled in."""
    body = "\n".join(f"    {line}" for line in lines)
    return (
        f"import pytest\n{imports}\n\n"
        f'{decorators}@pytest.mark.criterion("FERN-8/AC-1")\n'
        f"def test_it(call):\n{body}\n"
    )


def rules(code: str, **declared: str) -> list[str]:
    return [p.rule for p in check_code(code, CARD, **declared).problems]


def messages(code: str, **declared: str) -> str:
    return "\n".join(p.message for p in check_code(code, CARD, **declared).problems)


def test_a_good_test_passes_and_reports_its_function_and_marker():
    result = check_code(GOOD, CARD, name="test_asking_for_a_person_transfers", criterion="AC-1")

    assert result.ok
    assert result.problems == ()
    assert result.warnings == ()
    assert result.function == "test_asking_for_a_person_transfers"
    assert result.marker == "FERN-8/AC-1"
    assert result.reason == "passes the static checks"


def test_comprehensions_all_any_and_pytest_raises_are_allowed():
    code = proposal(
        "replies = [call.say(s) for s in ('agent',) if s]",
        "assert all(r.transferred for r in replies)",
        "assert any(r.ended for r in replies)",
        "with pytest.raises(CallError):",
        "    call.say('Hello?')",
        imports="from harness import CallError",
    )

    assert check_code(code, CARD).ok


def test_the_allowlist_is_the_documented_one():
    assert ALLOWED_IMPORTS == ("pytest", "re", "json", "datetime", "harness")
    code = proposal(
        "started = call.record()['startedAt']",
        "assert re.match('2026', started) and json.loads('1') == 1",
        "assert date.fromisoformat(started[:10]) == date(2026, 10, 14)",
        imports="import json\nimport re\nfrom datetime import date\n",
    )

    assert check_code(code, CARD).ok


def test_all_over_app_values_is_allowed_with_another_assertion_on_them():
    events = proposal(
        "events = call.events('transfer')",
        "assert len(events) == 1",
        "assert all(e['reason'] == 'caller_request' for e in events)",
    )
    same_expression = proposal(
        "assert call.events('transfer') != []",
        "assert all(e['at'] for e in call.events('transfer'))",
    )
    literal_values = proposal(
        "assert all(call.say(said).text for said in ('agent', 'representative'))",
        "assert not any(name in call.say('hi').text for name in PROVIDERS)",
        imports="from harness import PROVIDERS",
    )

    assert check_code(events, CARD).ok
    assert check_code(same_expression, CARD).ok
    assert check_code(literal_values, CARD).ok


def test_a_loop_over_literal_values_is_only_a_warning():
    code = proposal("for said in ['agent', 'representative']:", "    assert call.say(said).text")
    result = check_code(code, CARD)

    assert result.ok
    assert [w.rule for w in result.warnings] == ["loop"]
    assert "parametrize" in result.warnings[0].message


def test_code_that_does_not_parse_is_rejected_with_the_line():
    result = check_code("import pytest\n\ndef test_x(call:\n    pass\n", CARD)

    assert [p.rule for p in result.problems] == ["syntax"]
    assert result.problems[0].message.startswith("does not parse")
    assert result.problems[0].line is not None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import pytest\n\nX = 1\n" + GOOD.split("\n", 1)[1], "top-level"),
        (GOOD + "\n\ndef test_second(call):\n    assert call.say('agent').text\n", "one-test"),
        ("import pytest\n", "one-test"),
        (GOOD + "\n\ndef helper():\n    return 1\n", "one-test"),
        (GOOD + "\n\nclass TestMore:\n    pass\n", "one-test"),
    ],
    ids=["top-level-assign", "two-tests", "no-test", "helper-function", "class"],
)
def test_a_proposal_is_imports_plus_exactly_one_test(code, expected):
    assert expected in rules(code)


def test_async_tests_are_rejected():
    code = GOOD.replace("def test_asking", "async def test_asking")

    assert "is async; the harness is synchronous, use def" in messages(code)


def test_the_function_name_must_match_the_declared_name():
    assert rules(GOOD, name="test_something_else") == ["name"]
    assert "test_something_else" in messages(GOOD, name="test_something_else")


def test_only_criterion_and_parametrize_decorators_are_allowed():
    code = proposal("assert call.say('agent').text", decorators='@pytest.mark.usefixtures("api")\n')

    assert rules(code) == ["decorator"]
    assert "@pytest.mark.usefixtures" in messages(code)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("", "has no criterion marker"),
        ('@pytest.mark.criterion("FERN-8/AC-1")\n@pytest.mark.criterion("FERN-8/AC-2")\n', "has 2"),
        (
            '@pytest.mark.criterion("FERN-2/AC-1")\n',
            "names FERN-2, but this proposal is for FERN-8",
        ),
        (
            '@pytest.mark.criterion("FERN-8/AC-9")\n',
            "FERN-8 has no AC-9; its criteria are AC-1, AC-2",
        ),
        ('@pytest.mark.criterion("FERN-8 AC-1")\n', "is not written <card>/AC-<k>"),
        ("@pytest.mark.criterion(CRITERION)\n", "takes one literal string"),
    ],
    ids=["missing", "two", "other-card", "unknown-criterion", "malformed", "not-literal"],
)
def test_the_criterion_marker_must_name_one_real_criterion_of_the_card(marker, expected):
    code = f"import pytest\n\n\n{marker}def test_it(call):\n    assert call.say('agent').text\n"

    assert "criterion" in rules(code)
    assert expected in messages(code)


def test_the_marker_must_agree_with_the_criterion_the_proposal_is_listed_under():
    assert rules(GOOD, criterion="AC-2") == ["criterion"]
    assert "marker says AC-1, but the proposal is listed under AC-2" in messages(
        GOOD, criterion="AC-2"
    )


def test_parametrize_needs_literal_argument_names_that_the_test_takes():
    unnamed = GOOD.replace('parametrize("said",', "parametrize(NAMES,")
    unused = GOOD.replace("(call, said)", "(call)")

    assert "decorator" in rules(unnamed)
    assert "parametrizes 'said' but the test does not take it" in messages(unused)


@pytest.mark.parametrize(
    ("statement", "reason"),
    [
        ("import time", "never needs to wait"),
        ("import random", "impossible to reproduce"),
        ("import os", "environment, files or processes"),
        ("import subprocess", "must not start processes"),
        ("import socket", "only through the harness"),
        ("import httpx", "only through the harness"),
        ("from urllib.request import urlopen", "only through the harness"),
        ("from pathlib import Path", "reads and writes no files"),
        ("from frontdesk import bugs", "it is the app under test"),
        ("import yaml", "not on the import allowlist"),
    ],
)
def test_imports_outside_the_allowlist_are_rejected_with_a_reason(statement, reason):
    code = proposal("assert call.say('agent').text", imports=statement)
    found = messages(code)

    assert "import" in rules(code)
    assert reason in found
    assert "Allowed imports: pytest, re, json, datetime, harness." in found


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("from harness.client import Call", "import from harness itself"),
        ("from harness import NOT_A_HELPER", "has no such name"),
        ("from harness import *", "import each name it uses"),
        ("from . import harness", "relative import"),
    ],
)
def test_harness_imports_use_the_documented_names(statement, expected):
    code = proposal("assert call.say('agent').text", imports=statement)

    assert expected in messages(code)


@pytest.mark.parametrize(
    ("line", "rule", "expected"),
    [
        ("time.sleep(1)", "sleep", "calls time.sleep()"),
        ("random.choice(['agent'])", "banned", "impossible to reproduce"),
        ("subprocess.run(['ls'])", "banned", "must not start processes"),
        ("socket.create_connection(('localhost', 80))", "network", "only through the harness"),
        ("os.system('ls')", "banned", "shell commands"),
        ("open('out.txt', 'w')", "file", "opens a file"),
        ("tmp.write_text('x')", "file", "calls .write_text()"),
        ("json.dump({}, out)", "file", "json.dump writes to a file"),
        ("eval('1')", "dynamic", "uses eval()"),
        ("getattr(call, 'say')", "dynamic", "uses getattr()"),
        ("hasattr(pytest, 'shared')", "dynamic", "uses hasattr()"),
        ("pytest.shared_call", "pytest", "uses pytest.shared_call"),
        ("pytest.fail('no')", "pytest", "uses pytest.fail"),
        ("call._http.get('/health')", "private", "reaches ._http, a private name"),
        ("api.get('/test/build')", "test-hook", "names a test-only route"),
        ("api.post(f'/calls/{call.id}/test/take-slot', json={})", "test-hook", "call.take_slot"),
        ("call.__dict__", "dynamic", "reaches .__dict__"),
        ("api.get('http://127.0.0.1:9/health')", "network", "absolute URL"),
        ("pytest.skip('not today')", "skip", "calls pytest.skip()"),
        ("pytest.xfail('known')", "skip", "calls pytest.xfail()"),
    ],
)
def test_banned_calls_are_rejected_with_a_reason(line, rule, expected):
    code = proposal(line, "assert call.say('agent').text")

    assert rule in rules(code)
    assert expected in messages(code)


@pytest.mark.parametrize(
    ("lines", "imports", "expected"),
    [
        (["call.say('agent')", "pytest.shared_call = call.id"], "", "sets pytest.shared_call"),
        (["del pytest.mark"], "", "deletes pytest.mark"),
        (["call.last_reply = None"], "", "sets call.last_reply"),
        (["harness.DEFAULT_AT = '2026-01-01T00:00:00Z'"], "import harness", "sets harness."),
        (["PROVIDERS[0] = 'Dr. Who'"], "from harness import PROVIDERS", "belongs to PROVIDERS"),
        (["global SHARED", "SHARED = call.id"], "", "uses global SHARED"),
    ],
    ids=["module-attribute", "delete", "fixture-attribute", "harness-module", "item", "global"],
)
def test_state_that_outlives_the_test_is_rejected(lines, imports, expected):
    code = proposal(*lines, "assert call.say('agent').text", imports=imports)

    assert "shared-state" in rules(code)
    assert expected in messages(code)


def test_items_of_a_dict_the_test_built_itself_may_be_set():
    code = proposal("seen = {}", "seen['reply'] = call.say('agent').text", "assert seen['reply']")

    assert check_code(code, CARD).ok


@pytest.mark.parametrize(
    ("lines", "imports", "expected"),
    [
        (["call = start_call(at=None)"], "", "at=None"),
        (["call = start_call(None)"], "", "at=None"),
        (["call = frontdesk.start_call(at=None)"], "", "at=None"),
        (
            ["call = start_call(at=datetime.datetime.now(datetime.timezone.utc))"],
            "import datetime",
            "calls datetime.datetime.now()",
        ),
        (["call = start_call(at=datetime.now())"], "from datetime import datetime", ".now()"),
        (["call = start_call(at=dt.utcnow())"], "from datetime import datetime as dt", "utcnow"),
        (["day = date.today()", "call = start_call()"], "from datetime import date", "today()"),
        (["api.post('/calls', json={})", "call = start_call()"], "", 'without a literal "at"'),
        (["api.post('/calls')", "call = start_call()"], "", 'without a literal "at"'),
        (["api.post('/calls', json={'at': None})", "call = start_call()"], "", "literal"),
    ],
    ids=[
        "at-none",
        "at-none-positional",
        "frontdesk-client",
        "datetime-module",
        "datetime-class",
        "aliased",
        "date-today",
        "post-without-at",
        "post-without-body",
        "post-at-none",
    ],
)
def test_a_test_that_depends_on_the_wall_clock_is_rejected(lines, imports, expected):
    body = "\n".join(f"    {line}" for line in [*lines, "assert call.say('agent').text"])
    code = (
        f"import pytest\n{imports}\n\n"
        '@pytest.mark.criterion("FERN-8/AC-1")\n'
        f"def test_it(start_call, frontdesk, api):\n{body}\n"
    )

    assert "clock" in rules(code)
    assert expected in messages(code)


def test_a_fixed_clock_is_fine():
    code = (
        "import pytest\n\nfrom harness import AFTER_HOURS_AT\n\n\n"
        '@pytest.mark.criterion("FERN-8/AC-1")\n'
        "def test_it(start_call, api):\n"
        "    call = start_call(at=AFTER_HOURS_AT)\n"
        "    other = api.post('/calls', json={'at': AFTER_HOURS_AT}).json()\n"
        "    assert call.say('agent').text != other['reply']['text']\n"
    )

    assert check_code(code, CARD).ok


def test_skip_and_xfail_markers_are_rejected():
    code = proposal("assert call.say('agent').text", decorators="@pytest.mark.xfail\n")

    assert rules(code) == ["skip"]
    assert "report an untestable criterion instead" in messages(code)


def test_unknown_parameters_are_rejected_as_unknown_fixtures():
    code = GOOD.replace("(call, said)", "(call, said, tmp_path, monkeypatch)")

    assert rules(code) == ["fixture", "fixture"]
    assert "'tmp_path', which is neither a harness fixture" in messages(code)


def test_names_used_but_never_imported_are_rejected():
    code = proposal("assert call.say('agent').text == TRANSFER_LINE")

    assert rules(code) == ["undefined"]
    assert "uses TRANSFER_LINE, which is not imported" in messages(code)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["call.say('agent')"], "has no assert that checks anything"),
        (["call.say('agent')", "assert True"], "asserts True, which is always true"),
        (["reply = call.say('agent')", "assert (reply.text, 'message')"], "always true"),
        (["call.say('agent')", "assert 1 == 1"], "uses no fixture and no value from the app"),
        (
            ["call.say('agent')", "assert TRANSFER_LINE.startswith('I')"],
            "uses no fixture and no value from the app",
        ),
        (
            [
                "summary = call.record()['handoffSummary']",
                "assert summary['text'] == summary['text']",
            ],
            "compares summary['text'] with itself",
        ),
        (
            ["words = spoken_text(call.say('hi').ssml)", "assert words == words"],
            "compares words with itself",
        ),
    ],
    ids=[
        "no-assert",
        "assert-true",
        "assert-tuple",
        "constants",
        "harness-constant",
        "same-subscript",
        "same-name",
    ],
)
def test_a_test_needs_an_assert_that_can_fail(lines, expected):
    code = proposal(*lines, imports="from harness import TRANSFER_LINE, spoken_text")

    assert "assert" in rules(code)
    assert expected in messages(code)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["reply = call.say('agent')", "if reply.transferred:", "    assert reply.ended"], "an if"),
        (["while not call.say('agent').ended:", "    assert call.record()"], "while"),
        (
            ["try:", "    assert call.say('agent').ended", "except AssertionError:", "    pass"],
            "try",
        ),
        (
            ["reply = call.say('agent')", "assert reply.ended if reply.text else True"],
            "conditional",
        ),
        (
            ["match call.say('agent').text:", "    case _:", "        assert call.record()"],
            "match",
        ),
        (
            ["for event in call.events('transfer'):", "    assert event['reason']"],
            "runs zero times",
        ),
    ],
    ids=["if", "while", "try", "conditional-expression", "match", "loop-over-app-data"],
)
def test_control_flow_in_the_test_body_is_rejected(lines, expected):
    code = proposal(*lines)

    assert "control-flow" in rules(code)
    assert expected in messages(code)


def test_the_same_call_on_both_sides_is_not_a_tautology():
    """Two calls can return different values, so comparing them checks something."""
    code = proposal("assert call.say('agent').text == call.say('agent').text")

    assert check_code(code, CARD).ok


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (
            ["call.say('hi')", "assert all(e['reason'] == 'x' for e in call.events('transfer'))"],
            "all() over call.events('transfer')",
        ),
        (
            ["events = call.events('transfer')", "assert all(e['reason'] for e in events)"],
            "all() over events",
        ),
        (
            ["turns = call.record()['turns']", "assert not any(t['bargeIn'] for t in turns)"],
            "not any() over turns",
        ),
        (["flags = call.record()['flags']", "assert all(flags)"], "all() over flags"),
    ],
    ids=["call-result", "local", "not-any", "plain-argument"],
)
def test_all_over_app_values_with_nothing_else_asserted_is_vacuous(lines, expected):
    code = proposal(*lines)

    assert "vacuous" in rules(code)
    assert expected in messages(code)
    assert "empty list is True" in messages(code)


def test_problems_come_back_in_line_order_with_readable_text():
    code = "import time\nimport random\n\nimport pytest\n\n\ndef test_x(call):\n    time.sleep(1)\n"
    result = check_code(code, CARD)
    lines = [p.line or 0 for p in result.problems]

    assert lines == sorted(lines)
    assert all(p.message for p in result.problems)
    assert str(result.problems[0]) == f"line 1: {result.problems[0].message} [import]"
    assert result.reason.endswith(f"(and {len(result.problems) - 1} more)")


def test_a_problem_without_a_line_prints_without_one():
    assert str(Problem(rule="one-test", message="needs exactly one test")) == (
        "needs exactly one test [one-test]"
    )


def test_check_proposal_reads_the_schema_fields():
    test = ProposedTest(
        criterion="AC-2",
        name="test_asking_for_a_person_transfers",
        intent="Fails if asking for a person does not transfer.",
        code=GOOD,
    )

    assert [p.rule for p in check_proposal(test, CARD).problems] == ["criterion"]


def _canned() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(CANNED.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        found += [(data["card"], entry) for entry in data["tests"]]
    return found


def _canned_card(ref: str) -> Card:
    from taskgen.sources import FileSource

    return FileSource(REPO / "examples" / "fernhill-board.json").card(ref)


@pytest.mark.parametrize(
    ("ref", "entry"),
    _canned(),
    ids=[entry["name"] for _, entry in _canned()],
)
def test_canned_tests_fail_the_static_checks_only_where_meant_to(ref, entry):
    test = ProposedTest.model_validate({k: v for k, v in entry.items() if not k.startswith("_")})
    result = check_proposal(test, _canned_card(ref))

    assert result.ok is (entry["_expected_verdict"] != "rejected")


def test_static_check_messages_never_name_a_seeded_bug():
    """Static results are the gate output most likely to be shown to a model one day."""
    texts = []
    for ref, entry in _canned():
        test = ProposedTest.model_validate(
            {k: v for k, v in entry.items() if not k.startswith("_")}
        )
        result = check_proposal(test, _canned_card(ref))
        texts += [str(p) for p in (*result.problems, *result.warnings)]

    assert texts
    assert find_leaks("\n".join(texts)) == []


def test_canned_files_validate_against_the_output_schema():
    for path in sorted(CANNED.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        clean["tests"] = [
            {k: v for k, v in t.items() if not k.startswith("_")} for t in clean["tests"]
        ]

        assert GeneratedTests.model_validate(clean).card == path.stem
