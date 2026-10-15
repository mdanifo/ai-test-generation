"""Task cards and their acceptance criteria, as the generator sees them.

A card comes from the task board (``taskgen.sources.TaskBoardSource``) or from a
snapshot file (``taskgen.sources.FileSource``). Both produce the same ``Card``.

Criteria are numbered from 1 in checklist order, so the third item on a card's
"Acceptance criteria" checklist is ``AC-3``. A test names the criterion it checks
with ``@pytest.mark.criterion("<ref>/AC-<k>")``, for example ``"FERN-2/AC-3"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

REF_PATTERN = r"^[A-Z][A-Z0-9]*-\d+$"
CRITERION_ID_PATTERN = r"^AC-\d+$"
MARKER_PATTERN = r"^[A-Z][A-Z0-9]*-\d+/AC-\d+$"

DONE = "done"


class Criterion(BaseModel):
    """One acceptance criterion: an item on the card's "Acceptance criteria" checklist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=CRITERION_ID_PATTERN, description='"AC-1" for the first item.')
    text: str = Field(min_length=1)
    done: bool = False


class Card(BaseModel):
    """A task card that carries acceptance criteria."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: str = Field(pattern=REF_PATTERN, description='The card reference, e.g. "FERN-2".')
    title: str
    description: str = ""
    list_role: str = Field(description="The role of the card's list: todo, doing, done or hold.")
    labels: tuple[str, ...] = ()
    criteria: tuple[Criterion, ...] = Field(min_length=1)

    @property
    def is_done(self) -> bool:
        """True when the card is in the Done list: the behavior is built and can be gated."""
        return self.list_role == DONE

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        return tuple(criterion.id for criterion in self.criteria)

    def criterion(self, criterion_id: str) -> Criterion:
        """Return the criterion with this id. Raises KeyError naming the card's ids."""
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        known = ", ".join(self.criterion_ids)
        raise KeyError(f"{self.ref} has no criterion {criterion_id!r} (it has {known})")


def number_criteria(items: Iterable[str | tuple[str, bool]]) -> tuple[Criterion, ...]:
    """Number checklist items from 1: ``["a", "b"]`` becomes AC-1 and AC-2.

    Each item is its text, or ``(text, done)``.
    """
    criteria = []
    for k, item in enumerate(items, start=1):
        text, done = (item, False) if isinstance(item, str) else item
        criteria.append(Criterion(id=f"AC-{k}", text=text, done=done))
    return tuple(criteria)


def split_marker(marker: str) -> tuple[str, str]:
    """Split ``"FERN-2/AC-3"`` into ``("FERN-2", "AC-3")``. Raises ValueError otherwise."""
    if not re.fullmatch(MARKER_PATTERN, marker):
        raise ValueError(f"{marker!r} is not a criterion marker like 'FERN-2/AC-3'")
    ref, criterion_id = marker.split("/")
    return ref, criterion_id


def card_sort_key(ref: str) -> tuple[str, int]:
    """Sort refs by board key, then by number, so FERN-2 comes before FERN-10."""
    key, _, number = ref.rpartition("-")
    return key, int(number)
