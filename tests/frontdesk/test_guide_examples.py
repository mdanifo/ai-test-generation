"""Every example test in docs/HARNESS.md runs and passes against the app.

The guide is what the model reads, so an example that no longer passes would
teach it something false.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import harness

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "HARNESS.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.DOTALL)
EXAMPLES = [block for block in BLOCKS if "def test_" in block]


def _name(block: str) -> str:
    return re.search(r"def (test_\w+)", block).group(1)


def test_the_guide_has_examples():
    assert len(EXAMPLES) >= 8


@pytest.mark.parametrize("source", EXAMPLES, ids=[_name(b) for b in EXAMPLES])
def test_guide_example_passes(source, start_call, api):
    namespace = {"pytest": pytest, **{name: getattr(harness, name) for name in harness.__all__}}
    exec(compile(source, str(GUIDE), "exec"), namespace)
    example = namespace[_name(source)]

    params = [{}]
    for mark in getattr(example, "pytestmark", []):
        if mark.name == "parametrize":
            argname, values = mark.args
            params = [{argname: value} for value in values]
    for extra in params:
        fixtures = {"call": start_call(), "start_call": start_call, "api": api}
        wanted = inspect.signature(example).parameters
        example(**{name: (extra | fixtures)[name] for name in wanted})
