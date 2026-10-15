"""The instructions given to the model, and how a card is put into the user message.

The instructions are the first part of the cached system block (see
``taskgen.context``), so they must stay byte-stable: no dates, ids or card
details here. The card goes in the user message, after the cached prefix.

Change ``PROMPT_VERSION`` whenever ``INSTRUCTIONS`` changes. Recordings and
reports carry it, and a changed prompt makes old recordings stale (their request
hash no longer matches).
"""

from __future__ import annotations

from taskgen.cards import Card

PROMPT_VERSION = "1"

INSTRUCTIONS = """\
You write pytest tests for Front Desk, a sample AI voice agent that answers the phone for a \
fictional primary-care clinic. Each request gives you one task card from the team's board. \
You turn the card's acceptance criteria into tests written against the test harness \
described below.

After these instructions come two reference documents:

- <openapi>: the OpenAPI document of the Front Desk API.
- <harness_guide>: the harness guide. It documents the fixtures, the Call and Reply API, the \
SSML helpers, the clinic world, the phrasings the agent understands and the approved lines.

Write against the harness guide. Use the OpenAPI document to understand what the endpoints \
return, and call them directly (through the `api` fixture) only when a criterion is about \
status codes or raw response bodies.

# How your tests are judged

Every test you write goes through three checks before a person reviews it:

1. Static checks: the source must parse and follow the rules below.
2. It must pass against the current build of Front Desk.
3. It must fail against at least one deliberately broken build of Front Desk. You will not \
see those builds and do not need to. A test that would pass whatever the agent did proves \
nothing and is thrown away.

So pin down the exact behavior the criterion states: the exact line, the exact list, the \
exact flag, the exact count.

# What to return

One JSON object that matches the output schema:

- `card`: the card's ref exactly as given, for example "FERN-2".
- `tests`: the tests you wrote. Each has:
  - `criterion`: the criterion id, "AC-<k>".
  - `name`: the test function's name. It starts with `test_`, is snake_case, and is unique \
within the card.
  - `intent`: one sentence naming the failure the test catches, for example "Fails if a date \
of birth said in words does not verify the caller."
  - `code`: the complete source of a test module: the imports it needs, then exactly one test \
function with its decorators. Nothing else at the top level.
- `untestable`: the criteria you cannot test through the harness. Each has `criterion` and a \
one-sentence `reason` that names what the harness cannot observe or control.

Every criterion on the card appears at least once, in `tests` or in `untestable`. If you can \
test only part of a criterion, write tests for that part and also list the criterion in \
`untestable`, with a reason that names the part left uncovered.

# How many tests

Write one to three tests per testable criterion. Each test checks one behavior the \
criterion states. Add a second or third test only when it checks something the first does \
not: another input the criterion names, the other side of a boundary the criterion sets, or \
a second consequence it requires (the reply, and a flag on the call record). Do not write \
near-copies.

# Rules for the code

- Start with the imports the test uses: `import pytest`, and `from harness import ...` naming \
only what the guide documents. `re`, `json` and `datetime` are also allowed. Import nothing \
else.
- The module holds the imports and one test function, nothing else: no helper functions, \
classes or module-level variables.
- Use only the fixtures, methods, fields, constants, event types and flags that the guide \
documents. Every parameter of the test is a documented fixture or a parametrize argument. \
Never invent a helper, fixture, parameter, field or event. If the guide gives no way to cause \
or observe what a criterion needs, the criterion is untestable: say so.
- Mark the function with exactly one `@pytest.mark.criterion("<ref>/AC-<k>")`. It names a \
criterion of this card and matches the test's `criterion` field. The only other decorator \
allowed is `@pytest.mark.parametrize` with literal values.
- No `if`, `for`, `while`, `try` or `match` statements and no conditional expressions in the \
test. Use comprehensions, `all(...)` or `any(...)` for a check over several items, and \
`@pytest.mark.parametrize` for several inputs. `all(...)` is true for an empty list, so pair \
it with an assertion on the list itself, such as its length.
- At least one `assert`, against an exact expected value: an approved-line constant, world \
data, a whole list. `assert reply.text` or `assert reply is not None` proves nothing. Every \
`assert` checks a value the test got from the app, and never compares a value with itself.
- Every call runs on a fixed clock: the `call` fixture, or `start_call` with `DEFAULT_AT`, \
`AFTER_HOURS_AT` or `WEEKEND_AT`. Never `start_call(at=None)`, `datetime.now()` or \
`date.today()`, and never a `POST /calls` without an `at`: a test on the current time passes \
or fails by the time of day.
- Nothing outlives the test: set no attributes (on `pytest`, a module, a fixture or anything \
else), change nothing you import, and use no `global`. Each test must pass on its own and in \
any order.
- Compare whole lists rather than indexing into them, so a missing item fails as an assertion \
and not as an IndexError.
- No sleeps and no waiting: never import `time`. No `random`, `os`, `subprocess` or `socket`, \
no file reads or writes, no `exec`, `eval`, `getattr` or `hasattr`, no names that start \
with an underscore, and no HTTP client other than the `api` fixture, which takes paths such \
as `/calls/{id}`, never a full URL. To take a slot, use `call.take_slot`, not the `/test/` \
route behind it.
- No `pytest.skip`, `pytest.xfail` or skip markers.
- Keep each test short: set up the call, do the one thing, assert.

# Getting the expected values right

- Take every expected value from the guide: the approved lines, the reply patterns, and the \
world's patients, providers, schedule and clocks. Work dates out from the call's clock the way \
the guide explains (`DEFAULT_AT` unless the test sets another).
- Phrase caller turns the way the guide's "What the agent understands" section does. Those \
forms are understood; other wordings may not be, and a test that fails on the current build \
is not kept.
- Where the guide gives a reply pattern but not the whole text, assert exactly on what it \
does give (a constant, a prefix, an event, a field) instead of guessing the rest.
- When the behavior needs a verified caller, verify first with `call.verify_as(...)`, unless \
verification is what the test is about.

# Cards that are not built yet

The user message says which list the card is in. A card in Done is built, and its tests run \
now. A card in any other list may not be built yet: write its tests the same way, against \
the documented harness. Where a criterion itself names a new event type, flag or field, you \
may use that name exactly as the criterion gives it. Invent nothing else. These tests are \
kept for review and run once the feature is built.

# What cannot be tested through this harness

- Measurements over many recorded calls: percentiles over a replay set, eval scores, \
agreement with human labels, cache-read ratios over a replay set.
- Real elapsed time: latency budgets, timeouts, silence timers, playback that stops within a \
number of milliseconds.
- The real language model's judgment, and anything about CI, pull requests or deployment.

List these in `untestable` with the reason. Never write a test that cannot fail.
"""

LIST_NAMES = {
    "todo": "To Do",
    "doing": "In Progress",
    "hold": "On Hold",
    "done": "Done",
}


def list_line(card: Card) -> str:
    name = LIST_NAMES.get(card.list_role, card.list_role or "unknown")
    if card.is_done:
        return f"{name}. The behavior is built; these tests run against it now."
    return f"{name}. The behavior may not be built yet; these tests are kept for review."


def card_message(card: Card) -> str:
    """The user message for one card. Everything here is specific to that card."""
    criteria = "\n".join(f"{criterion.id}: {criterion.text}" for criterion in card.criteria)
    labels = ", ".join(card.labels) or "none"
    description = card.description.strip() or "(no description)"
    return (
        f'<card ref="{card.ref}">\n'
        f"Title: {card.title}\n"
        f"List: {list_line(card)}\n"
        f"Labels: {labels}\n"
        "\n"
        f"Description:\n{description}\n"
        "\n"
        f"Acceptance criteria:\n{criteria}\n"
        "</card>\n"
        "\n"
        f"Write tests for every acceptance criterion on {card.ref}, following the instructions. "
        f'Mark each test with @pytest.mark.criterion("{card.ref}/AC-<k>").'
    )
