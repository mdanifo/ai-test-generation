"""The output the model returns for one card, as pydantic models.

``output_schema()`` is the JSON schema sent as ``output_config.format``. The API
holds the model to its shape; pydantic then validates the text on the way back,
including the patterns the API treats only as hints.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from taskgen.cards import CRITERION_ID_PATTERN, REF_PATTERN

TEST_NAME_PATTERN = r"^test_[a-z0-9_]+$"


class ProposedTest(BaseModel):
    """One test function the model wrote for one criterion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: str = Field(pattern=CRITERION_ID_PATTERN, description='The criterion id, "AC-<k>".')
    name: str = Field(
        pattern=TEST_NAME_PATTERN,
        description="The test function's name: test_ then snake_case, unique within the card.",
    )
    intent: str = Field(
        min_length=1, description="One sentence naming the failure this test catches."
    )
    code: str = Field(
        min_length=1,
        description=(
            "The complete source of a test module: its imports, then exactly one test "
            "function with its decorators."
        ),
    )


class UntestableCriterion(BaseModel):
    """A criterion that cannot be tested through the harness, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: str = Field(pattern=CRITERION_ID_PATTERN, description='The criterion id, "AC-<k>".')
    reason: str = Field(
        min_length=1,
        description="One sentence naming what the harness cannot observe or control.",
    )


class GeneratedTests(BaseModel):
    """Everything the model returns for one card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card: str = Field(pattern=REF_PATTERN, description='The card ref, e.g. "FERN-2".')
    tests: list[ProposedTest]
    untestable: list[UntestableCriterion]


@cache
def _schema_text() -> str:
    import anthropic  # imported here: the SDK takes seconds to import, and only generation needs it

    return json.dumps(anthropic.transform_schema(GeneratedTests), sort_keys=True)


def output_schema() -> dict[str, Any]:
    """The JSON schema for ``output_config.format``: a fresh copy each call.

    Built with the SDK's ``transform_schema``, the same conversion ``messages.parse``
    uses, so unsupported keywords (such as ``pattern``) become description hints.
    """
    return json.loads(_schema_text())


def output_format() -> dict[str, Any]:
    """The ``output_config.format`` value."""
    return {"type": "json_schema", "schema": output_schema()}
