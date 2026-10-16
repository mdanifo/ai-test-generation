"""The cached system block: the instructions, then the context the model writes against.

The context is the OpenAPI document of ``frontdesk.main.app`` (compact JSON, sorted
keys, so the bytes match from run to run) and ``docs/HARNESS.md`` in full.
``system_text()`` checks the whole block against the seeded-bug catalog before
returning it, so an edit to the guide or the API that names a bug stops the run
instead of reaching the model.

Keep dates, ids and card details out of here. The block is the cached prefix of
every request, and any change to it costs a cache write on the next card.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taskgen.prompt import INSTRUCTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_GUIDE = REPO_ROOT / "docs" / "HARNESS.md"


class ContextLeakError(RuntimeError):
    """The context names something the model must never see."""


def openapi_document(app: Any = None) -> str:
    """The Front Desk OpenAPI document as compact JSON with sorted keys."""
    if app is None:
        from frontdesk.main import app
    return json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def harness_guide(path: str | Path = HARNESS_GUIDE) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def context_block(openapi: str | None = None, guide: str | None = None) -> str:
    """The two reference documents, each in its own tag."""
    openapi = openapi_document() if openapi is None else openapi
    guide = harness_guide() if guide is None else guide
    return f"<openapi>\n{openapi}\n</openapi>\n\n<harness_guide>\n{guide}\n</harness_guide>\n"


def forbidden_terms() -> list[str]:
    """Strings that would reveal the seeded bugs: every name and description in the catalog."""
    from frontdesk import bugs

    terms = ["SEEDED_BUG", "frontdesk.bugs", "frontdesk/bugs", "bugs.CATALOG"]
    for name, description in bugs.CATALOG.items():
        terms += [name, description]
    return terms


def find_leaks(text: str) -> list[str]:
    """The forbidden terms that appear in ``text``, ignoring case."""
    lowered = text.lower()
    return [term for term in forbidden_terms() if term.lower() in lowered]


def check_no_leak(text: str, where: str = "the context") -> str:
    """Return ``text`` unchanged, or raise ContextLeakError naming what leaked."""
    leaks = find_leaks(text)
    if leaks:
        raise ContextLeakError(
            f"{where} mentions the seeded bugs ({', '.join(map(repr, leaks))}); "
            "remove them before calling the model"
        )
    return text


def system_text(
    instructions: str = INSTRUCTIONS,
    openapi: str | None = None,
    guide: str | None = None,
) -> str:
    """The full system block: the instructions, then the context. Checked for leaks."""
    text = instructions.rstrip("\n") + "\n\n" + context_block(openapi, guide)
    return check_no_leak(text, "the system block")
