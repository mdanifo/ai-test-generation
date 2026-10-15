"""Static checks on a generated test, before anything runs.

A proposal is a small module: imports, then one test function with its markers.
``check_code`` parses it and applies every rule below. Each failure is a
``Problem`` with a rule id, the line in the proposal and a sentence a reviewer
can act on. A proposal with any problem is ``rejected`` by the gate and never runs.

Rules (rule id in brackets):

* [syntax] it parses as Python.
* [top-level] the module holds only imports and the test function.
* [one-test] exactly one ``test_*`` function, and no helper functions or classes.
* [name] the function's name is the name the proposal declares.
* [decorator] the only decorators are ``pytest.mark.criterion`` and ``pytest.mark.parametrize``.
* [criterion] exactly one ``@pytest.mark.criterion("<card>/AC-<k>")`` naming a real
  criterion of this card, and the one the proposal is listed under.
* [import] imports come only from pytest, re, json, datetime and harness, and a name
  imported from harness exists there.
* [sleep] no ``time.sleep``.
* [clock] nothing that depends on when the test runs: no ``datetime.now()``,
  ``utcnow()`` or ``today()``, no ``start_call(at=None)``, and no ``POST /calls`` through
  ``api`` without a literal ``at``. The app's default clock is the server's current time,
  so such a test passes or fails by the time of day.
* [banned] no ``random``, ``subprocess``, ``socket`` or ``os.system`` calls.
* [file] no file reads or writes.
* [network] no network client other than the harness, and no absolute URL.
* [test-hook] no path containing ``/test/``: those routes serve the harness itself, and
  ``call.take_slot`` is the one test hook a test needs.
* [dynamic] no ``exec``, ``eval``, ``getattr``, ``hasattr`` or dunder access that hides
  what the test does.
* [private] no attribute whose name starts with an underscore: a test uses the public
  harness API only.
* [pytest] from ``pytest`` a test uses only ``mark``, ``raises``, ``approx`` and ``param``.
  Any other name on the module is one a test could only have put there itself.
* [shared-state] no attribute assignment or deletion, no item assignment on an imported
  name or a fixture, and no ``global`` or ``nonlocal``. A value set on a module or a
  fixture outlives the test, so another test could pass only because this one ran.
* [fixture] every parameter is a harness fixture or a parametrize argument.
* [undefined] every name the test reads is imported, a parameter or assigned in the test.
* [assert] at least one ``assert``, and every ``assert`` checks something about the app:
  it is not a constant, it uses a fixture or a value the test got from the app, and it
  compares no expression with itself (``assert x == x``).
* [vacuous] ``all(...)`` (or ``not any(...)``) over values the app returns needs another
  assertion on those values, such as their length: both are true for an empty list, so
  on their own they pass when the app returns nothing.
* [skip] no ``pytest.skip``, ``pytest.xfail`` or skip/xfail markers.
* [control-flow] no ``if``, ``while``, ``try``, ``match`` or conditional expression in the
  test body. A ``for`` over values the app returns is rejected too: when the app returns
  nothing, the loop runs zero times and its asserts pass without checking anything.

Warnings (the test still runs):

* [loop] a ``for`` over literal values. ``@pytest.mark.parametrize`` with the same values
  reports each case on its own.

Comprehensions, ``all()`` and ``any()`` are allowed; docs/HARNESS.md recommends them.

The gate does not rely on these rules alone to keep tests independent: it also runs
every test on its own and calls it flaky when that result differs.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable, Iterator
from functools import cache

from pydantic import BaseModel, ConfigDict

from taskgen.cards import Card, split_marker
from taskgen.schema import ProposedTest

ALLOWED_IMPORTS: tuple[str, ...] = ("pytest", "re", "json", "datetime", "harness")
HARNESS_FIXTURES: tuple[str, ...] = ("call", "start_call", "api", "frontdesk", "frontdesk_url")
ALLOWED_MARKS: tuple[str, ...] = ("criterion", "parametrize")

_ALLOWED_LIST = ", ".join(ALLOWED_IMPORTS)
_TIME_REASON = (
    "harness calls are synchronous and each call has a fixed clock, so a test never needs "
    "to wait or read the time"
)
_CLOCK_REASON = (
    "the result would depend on when the test runs; give the call a fixed clock (the "
    "default DEFAULT_AT, or AFTER_HOURS_AT or WEEKEND_AT)"
)
_CLOCK_METHODS = frozenset({"now", "utcnow", "today"})
_NETWORK_REASON = "a test talks to the app only through the harness (call, start_call or api)"
_FILE_REASON = "a test reads and writes no files; it observes the app only through the harness"

# Why a module is off the allowlist, for the ones a test is most likely to reach for.
_IMPORT_REASONS: dict[str, str] = {
    "time": _TIME_REASON,
    "random": "random input makes a failure impossible to reproduce",
    "secrets": "random input makes a failure impossible to reproduce",
    "uuid": "random ids make a failure impossible to reproduce",
    "subprocess": "a test must not start processes",
    "multiprocessing": "a test must not start processes",
    "os": "a test must not touch the environment, files or processes",
    "sys": "a test must not touch the interpreter or the environment",
    "socket": _NETWORK_REASON,
    "httpx": _NETWORK_REASON,
    "requests": _NETWORK_REASON,
    "urllib": _NETWORK_REASON,
    "http": _NETWORK_REASON,
    "aiohttp": _NETWORK_REASON,
    "websockets": _NETWORK_REASON,
    "pathlib": _FILE_REASON,
    "shutil": _FILE_REASON,
    "tempfile": _FILE_REASON,
    "io": _FILE_REASON,
    "threading": "threads make results depend on timing",
    "asyncio": "the harness is synchronous; concurrency makes results depend on timing",
    "concurrent": "threads make results depend on timing",
    "unittest": "mocks replace the behavior under test; use the real app through the harness",
    "frontdesk": (
        "it is the app under test; a test observes the app only through its API, never its source"
    ),
    "importlib": "dynamic imports hide what a test depends on",
    "builtins": "dynamic access hides what a test does",
}

_BANNED_ROOTS: dict[str, str] = {
    "random": "random input makes a failure impossible to reproduce",
    "subprocess": "a test must not start processes",
    "socket": _NETWORK_REASON,
}
_BANNED_CALLS: dict[str, str] = {
    "os.system": "a test must not run shell commands",
    "os.popen": "a test must not run shell commands",
    "os.remove": _FILE_REASON,
    "os.unlink": _FILE_REASON,
    "os.makedirs": _FILE_REASON,
    "os.mkdir": _FILE_REASON,
    "json.dump": "json.dump writes to a file; compare values instead",
    "json.load": "json.load reads a file; a test observes the app only through the harness",
}
_FILE_METHODS = frozenset(
    {"write_text", "write_bytes", "mkdir", "touch", "unlink", "rmdir", "symlink_to", "chmod"}
)
_DYNAMIC_BUILTINS = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "hasattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "exit",
        "quit",
    }
)
_SKIP_CALLS = frozenset({"pytest.skip", "pytest.xfail", "pytest.importorskip", "pytest.exit"})
_PYTEST_NAMES = frozenset({"mark", "raises", "approx", "param"})
_SKIP_MARKS = frozenset({"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail"})
_URL_PREFIXES = ("http://", "https://", "ws://", "wss://", "ftp://")
_TEST_ROUTE = "/test/"
# Comparisons that are always true when both sides are the same expression.
_REFLEXIVE_OPS = (ast.Eq, ast.Is, ast.LtE, ast.GtE)


class Problem(BaseModel):
    """One reason a proposal fails a static check (or, as a warning, a style note)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    line: int | None = None
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        return f"{where}{self.message} [{self.rule}]"


class StaticResult(BaseModel):
    """The outcome of the static checks on one proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problems: tuple[Problem, ...] = ()
    warnings: tuple[Problem, ...] = ()
    function: str | None = None
    marker: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def reason(self) -> str:
        """One readable line: the first problem, and how many more there are."""
        if not self.problems:
            return "passes the static checks"
        first = str(self.problems[0])
        more = len(self.problems) - 1
        return first if not more else f"{first} (and {more} more)"


@cache
def _harness_names() -> frozenset[str]:
    import harness

    return frozenset(harness.__all__)


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a chain of names and attributes, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _root_name(node: ast.AST) -> str | None:
    """The name an attribute or subscript chain starts from: ``a`` in ``a.b[0].c``."""
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _key(node: ast.AST) -> str:
    """A structural key for an expression, ignoring positions and load/store context."""
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Each name the module's imports bind, mapped to the module it came from."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[(alias.asname or alias.name).split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = node.module.split(".")[0]
    return names


class _Checker:
    def __init__(self, card: Card, name: str | None, criterion: str | None) -> None:
        self.card = card
        self.declared_name = name
        self.declared_criterion = criterion
        self.problems: list[Problem] = []
        self.warnings: list[Problem] = []
        self.function: str | None = None
        self.marker: str | None = None
        self.imported: dict[str, str] = {}

    def problem(self, rule: str, node: ast.AST | None, message: str) -> None:
        line = getattr(node, "lineno", None) if node is not None else None
        found = Problem(rule=rule, line=line, message=message)
        if found not in self.problems:
            self.problems.append(found)

    def warn(self, rule: str, node: ast.AST, message: str) -> None:
        found = Problem(rule=rule, line=getattr(node, "lineno", None), message=message)
        if found not in self.warnings:
            self.warnings.append(found)

    def run(self, code: str) -> StaticResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.problems.append(
                Problem(rule="syntax", line=exc.lineno, message=f"does not parse: {exc.msg}")
            )
            return self.result()

        functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for index, stmt in enumerate(tree.body):
            if isinstance(stmt, ast.Import | ast.ImportFrom):
                continue
            if (
                index == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue  # a module docstring
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append(stmt)
                continue
            if isinstance(stmt, ast.ClassDef):
                self.problem(
                    "one-test",
                    stmt,
                    f"defines class {stmt.name}; a proposal is one test function, not a class",
                )
                continue
            kind = type(stmt).__name__.lower()
            self.problem(
                "top-level",
                stmt,
                f"has a top-level {kind} statement; a proposal is imports plus one test "
                "function, so put values inside the test or in a parametrize list",
            )

        tests = [f for f in functions if f.name.startswith("test")]
        for helper in (f for f in functions if not f.name.startswith("test")):
            self.problem(
                "one-test",
                helper,
                f"defines helper function {helper.name}(); a proposal is one test function, "
                "so write the steps in the test itself",
            )
        if len(tests) != 1:
            names = ", ".join(f.name for f in tests) or "none"
            self.problem(
                "one-test",
                tests[1] if len(tests) > 1 else None,
                f"needs exactly one test_* function, found {len(tests)} ({names})",
            )

        self.imported = _imported_names(tree)
        self.check_imports(tree)
        self.check_names(tree)
        self.check_state(tree)
        if tests:
            self.check_test(tests[0])
            self.check_defined(tree, tests[0])
        return self.result()

    def result(self) -> StaticResult:
        def by_line(problem: Problem) -> int:
            return problem.line or 0

        return StaticResult(
            problems=tuple(sorted(self.problems, key=by_line)),
            warnings=tuple(sorted(self.warnings, key=by_line)),
            function=self.function,
            marker=self.marker,
        )

    def check_imports(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.check_module(node, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    self.problem("import", node, "uses a relative import; import from harness")
                    continue
                module = node.module or ""
                if module == "__future__":
                    continue
                if not self.check_module(node, module):
                    continue
                if module == "harness":
                    self.check_harness_names(node)

    def check_module(self, node: ast.AST, module: str) -> bool:
        root = module.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            reason = _IMPORT_REASONS.get(root, "it is not on the import allowlist")
            self.problem(
                "import",
                node,
                f"imports {module}, which is not allowed: {reason}. "
                f"Allowed imports: {_ALLOWED_LIST}.",
            )
            return False
        if root == "harness" and module != "harness":
            self.problem(
                "import",
                node,
                f"imports {module}; import from harness itself (from harness import ...), "
                "which is the documented API",
            )
            return False
        return True

    def check_harness_names(self, node: ast.ImportFrom) -> None:
        known = _harness_names()
        for alias in node.names:
            if alias.name == "*":
                self.problem("import", node, "uses from harness import *; import each name it uses")
            elif alias.name not in known:
                self.problem(
                    "import",
                    node,
                    f"imports {alias.name} from harness, which has no such name "
                    "(docs/HARNESS.md lists every helper)",
                )

    def check_names(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.check_call(node)
            elif isinstance(node, ast.Name):
                if node.id in _DYNAMIC_BUILTINS:
                    self.problem(
                        "dynamic",
                        node,
                        f"uses {node.id}(), which hides what the test does; call the harness "
                        "directly",
                    )
                elif node.id == "open":
                    self.problem("file", node, f"opens a file: {_FILE_REASON}")
                elif node.id.startswith("__") and node.id.endswith("__"):
                    self.problem(
                        "dynamic",
                        node,
                        f"uses {node.id}; a test has no reason to reach interpreter internals",
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    self.problem(
                        "dynamic",
                        node,
                        f"reaches .{node.attr}; a test has no reason to reach interpreter "
                        "internals",
                    )
                elif node.attr.startswith("_"):
                    self.problem(
                        "private",
                        node,
                        f"reaches .{node.attr}, a private name; a test uses only the public "
                        "harness API in docs/HARNESS.md",
                    )
                elif (
                    isinstance(node.value, ast.Name)
                    and self.imported.get(node.value.id) == "pytest"
                    and node.attr not in _PYTEST_NAMES
                    and f"pytest.{node.attr}" not in _SKIP_CALLS
                ):
                    self.problem(
                        "pytest",
                        node,
                        f"uses pytest.{node.attr}; a test uses pytest only for pytest.mark, "
                        "pytest.raises, pytest.approx and pytest.param",
                    )
                dotted = _dotted(node)
                if dotted in _SKIP_MARKS:
                    self.problem(
                        "skip",
                        node,
                        f"uses {dotted}. A test that can be skipped can pass without checking "
                        "anything; report an untestable criterion instead",
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.strip().lower().startswith(_URL_PREFIXES):
                    self.problem(
                        "network",
                        node,
                        f"contains the absolute URL {node.value.strip()[:60]!r}: {_NETWORK_REASON}",
                    )
                elif _TEST_ROUTE in node.value:
                    self.problem(
                        "test-hook",
                        node,
                        f"names a test-only route ({node.value.strip()[:60]!r}); those serve "
                        "the harness itself, and call.take_slot is the one test hook a test "
                        "needs",
                    )

    def check_state(self, tree: ast.Module) -> None:
        """Nothing a test does may outlive it: no attributes set, no module state changed."""
        fixtures = set(HARNESS_FIXTURES)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global | ast.Nonlocal):
                kind = "global" if isinstance(node, ast.Global) else "nonlocal"
                self.problem(
                    "shared-state",
                    node,
                    f"uses {kind} {', '.join(node.names)}; a value kept outside the test "
                    "outlives it and can make another test pass or fail",
                )
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store | ast.Del):
                verb = "deletes" if isinstance(node.ctx, ast.Del) else "sets"
                self.problem(
                    "shared-state",
                    node,
                    f"{verb} {ast.unparse(node)}; a test sets no attributes. On a module or "
                    "a fixture the change outlives the test, so another test could pass only "
                    "because this one ran",
                )
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del):
                root = _root_name(node)
                if root in self.imported or root in fixtures:
                    self.problem(
                        "shared-state",
                        node,
                        f"changes {ast.unparse(node)}, which belongs to {root}; the change "
                        "outlives the test, so another test could pass only because this "
                        "one ran",
                    )

    def check_clock(self, node: ast.Call, dotted: str) -> bool:
        """Report a call that ties the test to the wall clock. True when it did."""
        parts = dotted.split(".")
        if parts[-1] in _CLOCK_METHODS and self.imported.get(parts[0]) == "datetime":
            self.problem("clock", node, f"calls {dotted}(): {_CLOCK_REASON}")
            return True
        if parts[-1] == "start_call":
            at = next((k.value for k in node.keywords if k.arg == "at"), None)
            if at is None and node.args:
                at = node.args[0]
            if isinstance(at, ast.Constant) and at.value is None:
                self.problem(
                    "clock",
                    node,
                    f"starts a call with at=None, the server's current time: {_CLOCK_REASON}",
                )
                return True
        if parts[-1] == "post" and len(parts) > 1 and node.args:
            path = node.args[0]
            if (
                isinstance(path, ast.Constant)
                and isinstance(path.value, str)
                and path.value.rstrip("/") == "/calls"
            ):
                body = next((k.value for k in node.keywords if k.arg == "json"), None)
                at = None
                if isinstance(body, ast.Dict):
                    at = next(
                        (
                            value
                            for key, value in zip(body.keys, body.values, strict=True)
                            if isinstance(key, ast.Constant) and key.value == "at"
                        ),
                        None,
                    )
                if at is None or (isinstance(at, ast.Constant) and at.value is None):
                    self.problem(
                        "clock",
                        node,
                        'posts to /calls without a literal "at" in its json body, so the call '
                        f"runs on the server's current time: {_CLOCK_REASON}",
                    )
                    return True
        return False

    def check_call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        if dotted is None:
            return
        if self.check_clock(node, dotted):
            return
        if dotted in ("time.sleep", "sleep"):
            self.problem(
                "sleep",
                node,
                f"calls {dotted}(): harness calls are synchronous and the call's clock does "
                "not move, so a sleep only slows the suite and hides a timing assumption",
            )
            return
        if dotted in _SKIP_CALLS:
            self.problem(
                "skip",
                node,
                f"calls {dotted}(). A test that can skip or expect failure can pass without "
                "checking anything; report an untestable criterion instead",
            )
            return
        if dotted in _BANNED_CALLS:
            rule = "file" if dotted.startswith(("json.", "os.m", "os.r", "os.u")) else "banned"
            self.problem(rule, node, f"calls {dotted}(): {_BANNED_CALLS[dotted]}")
            return
        root = dotted.split(".")[0]
        if root in _BANNED_ROOTS:
            rule = "network" if root == "socket" else "banned"
            self.problem(rule, node, f"calls {dotted}(): {_BANNED_ROOTS[root]}")
            return
        method = dotted.rsplit(".", 1)[-1]
        if "." in dotted and method in _FILE_METHODS:
            self.problem("file", node, f"calls .{method}(): {_FILE_REASON}")

    def check_defined(self, tree: ast.Module, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Every name the test reads is imported, a parameter, assigned in it, or a builtin."""
        known = set(dir(builtins))
        for node in tree.body:
            if isinstance(node, ast.Import):
                known |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                known |= {a.asname or a.name for a in node.names}
        args = func.args
        known |= {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                known.add(node.id)
            elif isinstance(node, ast.arg):
                known.add(node.arg)  # lambda parameters
            elif isinstance(node, ast.Import | ast.ImportFrom):
                known |= {(a.asname or a.name).split(".")[0] for a in node.names}
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in known
            ):
                self.problem(
                    "undefined",
                    node,
                    f"uses {node.id}, which is not imported, a fixture or assigned in the "
                    "test; import it (from harness import ...) or take it as a fixture",
                )

    def check_test(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function = func.name
        if isinstance(func, ast.AsyncFunctionDef):
            self.problem("one-test", func, "is async; the harness is synchronous, use def")
        if self.declared_name and func.name != self.declared_name:
            self.problem(
                "name",
                func,
                f"defines {func.name}() but the proposal is named {self.declared_name}",
            )
        argnames = self.check_decorators(func)
        self.check_signature(func, argnames)
        self.check_body(func)

    def check_decorators(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        argnames: set[str] = set()
        criteria: list[ast.Call] = []
        for decorator in func.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            dotted = _dotted(target) or ""
            mark = dotted.removeprefix("pytest.mark.") if dotted.startswith("pytest.mark.") else ""
            if mark not in ALLOWED_MARKS:
                shown = dotted or ast.unparse(decorator)
                if dotted not in _SKIP_MARKS:  # check_names reports those as skips
                    self.problem(
                        "decorator",
                        decorator,
                        f"is decorated with @{shown}; only @pytest.mark.criterion and "
                        "@pytest.mark.parametrize are allowed",
                    )
                continue
            if not isinstance(decorator, ast.Call):
                self.problem("decorator", decorator, f"uses @{dotted} without arguments")
                continue
            if mark == "criterion":
                criteria.append(decorator)
            else:
                argnames |= self.parametrize_names(decorator)
        self.check_criterion(func, criteria)
        return argnames

    def parametrize_names(self, decorator: ast.Call) -> set[str]:
        node: ast.AST | None = decorator.args[0] if decorator.args else None
        for keyword in decorator.keywords:
            if keyword.arg == "argnames":
                node = keyword.value
        names: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [n.strip() for n in node.value.split(",") if n.strip()]
        elif isinstance(node, ast.List | ast.Tuple) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts
        ):
            names = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if not names:
            self.problem(
                "decorator",
                decorator,
                "parametrize needs its argument names as a literal string, e.g. "
                '@pytest.mark.parametrize("said", [...])',
            )
        return set(names)

    def check_criterion(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef, marks: list[ast.Call]
    ) -> None:
        example = f'@pytest.mark.criterion("{self.card.ref}/AC-1")'
        if not marks:
            self.problem("criterion", func, f"has no criterion marker; add one such as {example}")
            return
        if len(marks) > 1:
            self.problem(
                "criterion",
                marks[1],
                f"has {len(marks)} criterion markers; a test checks exactly one criterion",
            )
            return
        mark = marks[0]
        arg = mark.args[0] if len(mark.args) == 1 and not mark.keywords else None
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            self.problem(
                "criterion",
                mark,
                f"criterion marker takes one literal string, such as {example}",
            )
            return
        try:
            ref, criterion_id = split_marker(arg.value)
        except ValueError:
            self.problem(
                "criterion",
                mark,
                f"criterion marker {arg.value!r} is not written <card>/AC-<k>, such as {example}",
            )
            return
        self.marker = arg.value
        if ref != self.card.ref:
            self.problem(
                "criterion",
                mark,
                f"criterion marker names {ref}, but this proposal is for {self.card.ref}",
            )
            return
        if criterion_id not in self.card.criterion_ids:
            known = ", ".join(self.card.criterion_ids)
            self.problem(
                "criterion",
                mark,
                f"{self.card.ref} has no {criterion_id}; its criteria are {known}",
            )
            return
        if self.declared_criterion and criterion_id != self.declared_criterion:
            self.problem(
                "criterion",
                mark,
                f"criterion marker says {criterion_id}, but the proposal is listed under "
                f"{self.declared_criterion}",
            )

    def check_signature(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef, argnames: set[str]
    ) -> None:
        args = func.args
        if args.vararg or args.kwarg:
            self.problem("fixture", func, "takes *args or **kwargs; name each fixture it uses")
        allowed = set(HARNESS_FIXTURES) | argnames
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        for param in params:
            if param.arg not in allowed:
                fixtures = ", ".join(HARNESS_FIXTURES)
                self.problem(
                    "fixture",
                    param,
                    f"takes parameter {param.arg!r}, which is neither a harness fixture "
                    f"({fixtures}) nor a parametrize argument",
                )
        for name in sorted(argnames - {p.arg for p in params}):
            self.problem(
                "fixture",
                func,
                f"parametrizes {name!r} but the test does not take it as a parameter",
            )

    def app_names(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        """Fixtures the test takes and names it assigns: what can hold a value from the app."""
        args = func.args
        params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        names = params & set(HARNESS_FIXTURES)
        for node in _walk_body(func.body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

    def check_assert(self, node: ast.Assert, app: set[str]) -> bool:
        """Problems with one assert. True when it can check something about the app."""
        if _always_true(node.test):
            self.problem(
                "assert",
                node,
                f"asserts {ast.unparse(node.test)}, which is always true and checks nothing",
            )
            return False
        if not any(isinstance(n, ast.Name) and n.id in app for n in ast.walk(node.test)):
            self.problem(
                "assert",
                node,
                f"asserts {ast.unparse(node.test)}, which uses no fixture and no value from "
                "the app, so it checks nothing about Front Desk",
            )
            return False
        for compare in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
            sides = [compare.left, *compare.comparators]
            for op, left, right in zip(compare.ops, sides, sides[1:], strict=False):
                if (
                    isinstance(op, _REFLEXIVE_OPS)
                    and _key(left) == _key(right)
                    and not _has_call(left)
                ):
                    self.problem(
                        "assert",
                        node,
                        f"compares {ast.unparse(left)} with itself, which is always true and "
                        "checks nothing",
                    )
                    return False
        return True

    def check_vacuous(self, asserts: list[ast.Assert], app: set[str]) -> None:
        """``all()`` over the app's values needs another assertion on those values."""
        mentions: dict[str, int] = {}
        for node in asserts:
            for sub in ast.walk(node.test):
                mentions[_key(sub)] = mentions.get(_key(sub), 0) + 1
        for node in asserts:
            for sub in ast.walk(node.test):
                negated = isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not)
                found = sub.operand if negated and isinstance(sub, ast.UnaryOp) else sub
                if not (
                    isinstance(found, ast.Call)
                    and isinstance(found.func, ast.Name)
                    and len(found.args) == 1
                    and (found.func.id == "all" or (negated and found.func.id == "any"))
                ):
                    continue
                arg = found.args[0]
                values = (
                    arg.generators[0].iter
                    if isinstance(arg, ast.GeneratorExp | ast.ListComp | ast.SetComp)
                    else arg
                )
                from_app = any(isinstance(n, ast.Name) and n.id in app for n in ast.walk(values))
                if from_app and mentions.get(_key(values), 0) < 2:
                    what = "not any()" if negated else "all()"
                    self.problem(
                        "vacuous",
                        node,
                        f"asserts {what} over {ast.unparse(values)}, which comes from the app: "
                        f"{what} of an empty list is True, so on its own this passes when the "
                        "app returns nothing. Also assert the values themselves, such as their "
                        "length or the whole list",
                    )

    def check_body(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        asserts = 0
        app = self.app_names(func)
        found_asserts: list[ast.Assert] = []
        for node in _walk_body(func.body):
            if isinstance(node, ast.Assert):
                found_asserts.append(node)
                if self.check_assert(node, app):
                    asserts += 1
            elif isinstance(node, ast.If):
                self.control_flow(node, "an if statement")
            elif isinstance(node, ast.While):
                self.control_flow(node, "a while loop")
            elif isinstance(node, ast.Try | ast.TryStar):
                self.problem(
                    "control-flow",
                    node,
                    "has a try statement, which can catch the very failure the test exists "
                    "to report; use pytest.raises for an expected error",
                )
            elif isinstance(node, ast.Match):
                self.control_flow(node, "a match statement")
            elif isinstance(node, ast.IfExp):
                self.control_flow(node, "a conditional expression")
            elif isinstance(node, ast.For | ast.AsyncFor):
                self.check_loop(node)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                self.problem(
                    "one-test",
                    node,
                    f"defines {node.name} inside the test; write the steps in the test itself",
                )
        self.check_vacuous(found_asserts, app)
        if not asserts:
            self.problem(
                "assert",
                func,
                "has no assert that checks anything; a test without one passes whenever "
                "nothing raises",
            )

    def control_flow(self, node: ast.AST, what: str) -> None:
        self.problem(
            "control-flow",
            node,
            f"has {what} in the test body; a test takes one path every run. Use all(), any() "
            "or a comprehension, or @pytest.mark.parametrize for several inputs",
        )

    def check_loop(self, node: ast.For | ast.AsyncFor) -> None:
        if isinstance(node.iter, ast.List | ast.Tuple | ast.Set) and _is_literal(node.iter):
            self.warn(
                "loop",
                node,
                "loops over literal values; @pytest.mark.parametrize with the same values "
                "reports each case on its own",
            )
            return
        self.problem(
            "control-flow",
            node,
            f"loops over {ast.unparse(node.iter)}, which comes from the app: when it is empty "
            "the loop runs zero times and its asserts pass without checking anything. Compare "
            "the whole list, or use all() or any()",
        )


def _walk_body(body: Iterable[ast.stmt]) -> Iterator[ast.AST]:
    for stmt in body:
        yield from ast.walk(stmt)


def _always_true(test: ast.expr) -> bool:
    if isinstance(test, ast.Constant):
        return bool(test.value)
    return isinstance(test, ast.Tuple) and bool(test.elts)


def check_code(
    code: str, card: Card, *, name: str | None = None, criterion: str | None = None
) -> StaticResult:
    """Check one proposal's source against the rules in this module's docstring.

    ``card`` is the card the proposal is for. ``name`` and ``criterion`` ("AC-3") are
    what the proposal declares; when given, the function name and marker must agree.
    """
    return _Checker(card, name, criterion).run(code)


def check_proposal(test: ProposedTest, card: Card) -> StaticResult:
    """``check_code`` for one test from the model's output."""
    return check_code(test.code, card, name=test.name, criterion=test.criterion)
