"""The seeded-bug catalog.

Each entry is a realistic mistake behind one marked branch, ``bugs.active("<name>")``,
at the spot where a developer would make it. Set ``SEEDED_BUG=<name>`` to turn one on;
unset or empty means the correct app.

The gate runs proposed tests once per entry and counts a test only when it fails under
at least one. The catalog stays out of the OpenAPI document, docs/HARNESS.md and the
prompt.

An unknown name is an error rather than a quiet "no bug", so a typo cannot pass a clean
build off as a bug run.
"""

from __future__ import annotations

import os

ENV_VAR = "SEEDED_BUG"

CATALOG: dict[str, str] = {
    "dob-format-strict": (
        "FERN-2: verification only accepts the ISO date form, so 'March 3rd 1980' fails."
    ),
    "verify-attempts-off-by-one": (
        "FERN-2: a fourth verification attempt is allowed before stopping."
    ),
    "other-patient-leak": "FERN-2: a verified caller hears another patient's appointment.",
    "morning-includes-noon": "FERN-3: the 12:00 slot is offered as a morning slot.",
    "double-booking": "FERN-3: confirming twice creates two appointments.",
    "emergency-after-model": (
        "FERN-4: the emergency check runs after the model, so no script when the model is down."
    ),
    "emergency-script-paraphrased": "FERN-4: the emergency script is reworded.",
    "transfer-asks-why": "FERN-8: asking for an agent gets 'Can I ask what this is about?' first.",
    "low-confidence-not-consecutive": (
        "FERN-8: two low-confidence turns that are not consecutive still offer a transfer."
    ),
    "ssml-unescaped": "FERN-11: caller text is inserted into SSML unescaped.",
    "raw-timestamp-spoken": "FERN-11: the booking read-back speaks the ISO timestamp.",
    "disclosure-skipped-on-barge-in": (
        "FERN-13: a barge-in during the greeting skips the recording disclosure."
    ),
}


class UnknownBugError(ValueError):
    """Raised for a bug name that is not in CATALOG."""


def current() -> str | None:
    """Return the active seeded bug from the environment, or None.

    Read at call time, so a process that sets SEEDED_BUG after import sees it.
    Raises UnknownBugError when the value is not in CATALOG.
    """
    name = os.environ.get(ENV_VAR, "").strip()
    if not name:
        return None
    if name not in CATALOG:
        known = ", ".join(CATALOG)
        raise UnknownBugError(f"{ENV_VAR}={name!r} is not a known seeded bug. Known: {known}")
    return name


def active(name: str) -> bool:
    """Return True when ``name`` is the active seeded bug.

    Raises UnknownBugError when ``name`` is not in CATALOG (a typo in a branch)
    or when SEEDED_BUG holds an unknown value.
    """
    if name not in CATALOG:
        raise UnknownBugError(f"{name!r} is not in bugs.CATALOG")
    return current() == name
