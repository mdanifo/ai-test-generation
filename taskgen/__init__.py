"""taskgen: Claude writes pytest tests from task-card acceptance criteria, and a gate keeps
only the tests that pass on the current build and fail against a deliberately broken one.

Modules:

* ``taskgen.cards``: the ``Card`` and ``Criterion`` models.
* ``taskgen.sources``: ``TaskBoardSource`` (the board's REST API, read with GET; the one
  write is ``taskgen comment --yes``) and ``FileSource`` (a snapshot file).
* ``taskgen.context``: the cached context block (OpenAPI document and harness guide).
* ``taskgen.prompt``: the instructions given to the model.
* ``taskgen.schema``: the output models the model's JSON is validated against.
* ``taskgen.generate``: ``ClaudeGenerator``, ``FakeGenerator``, ``CannedGenerator`` and
  record/replay cassettes.
* ``taskgen.static_checks``: the AST checks a proposed test must pass before it runs.
* ``taskgen.gate``: runs proposed tests against the clean build twice (together, then
  each on its own) and once per seeded bug, and gives each a verdict.
* ``taskgen.report``: ``proposed/<card>/REPORT.md``, ``proposed/RUN.md`` and the card
  comment summary.
* ``taskgen.accept``: copies reviewed, accepted tests into ``tests/generated/``.
* ``taskgen.cli``: the ``taskgen`` command.
"""

__version__ = "0.1.0"
