"""Static guard: the application core must contain zero domain-specific knowledge.

The engine derives every table, column, filter and join from the live database
catalog at runtime. That property is easy to state and easy to break by accident
-- someone adds a ``if "sharma" in q`` special case and genericity silently dies.

This module walks the AST of every source file under ``app/`` and fails if any
string literal encodes knowledge of a concrete business domain.

Run standalone::

    pytest tests/test_no_hardcoding.py -v
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent / "app"

HTTP_STATUS_CODES = frozenset(
    {200, 201, 204, 400, 401, 403, 404, 409, 422, 429, 500, 502, 503, 504}
)

# ---------------------------------------------------------------------------
# Banned vocabulary
# ---------------------------------------------------------------------------
# Table names from the demonstration database in artifacts/ and from the synthetic
# fixtures in conftest.py. An exact (stripped, lowercased) match on a string
# literal is treated as hardcoded schema knowledge.

BANNED_TABLES: frozenset[str] = frozenset(
    {
        # demonstration database
        "police_stations",
        "officers",
        "cases",
        "case_assignments",
        "parties",
        "case_parties",
        "evidence_records",
        "case_activity_logs",
        # fixtures
        "products",
        "customers",
        "orders",
        "order_lines",
        "clinicians",
        "patients",
        "encounters",
        "authors",
        "titles",
        "loans",
    }
)

# Column names from the same sources. Matched token-wise so that `case_number`,
# `case number` and `CASE-NUMBER` are all caught.

BANNED_COLUMNS: tuple[str, ...] = (
    # demonstration database
    "badge_number",
    "fir_number",
    "case_number",
    "station_code",
    "jurisdiction_zone",
    "estimated_fraud_amount",
    "chain_of_custody_officer_id",
    "role_in_case",
    "legal_sections",
    "custody_status",
    "national_id_number",
    "residential_address",
    "logged_by_officer_id",
    "evidence_number",
    "activity_type",
    "assignment_notes",
    "tagged_date",
    "storage_location",
    "district",
    "full_name",
    "is_current",
    # fixtures
    "order_reference",
    "order_state",
    "grand_total",
    "display_name",
    "unit_price",
    "stock_on_hand",
    "line_total",
    "registration_code",
    "practitioner_name",
    "encounter_code",
    "visit_status",
    "billed_amount",
    "enrolment_state",
    "date_of_birth",
    "accession_code",
    "circulation_status",
    "author_name",
    "birth_year",
)

# Literal values and canned strings that only make sense for one specific domain.

BANNED_VALUES: tuple[str, ...] = (
    "sharma",
    "rajesh",
    "ind-8842",
    "case-2025-0211",
    "fir-2025-891",
    "evd-2025-0045",
    "ps-crime-01",
    "under investigation",
    "charge sheet",
    "financial fraud",
    "cyber crime",
    "corruption",
    "inspector",
    "sub-inspector",
    "constable",
    "investigating officer",
    "₹",
)

# Modules that must not be imported by the generic core.

BANNED_IMPORTS: tuple[str, ...] = (
    "app.usecases",
    "app.usecases.investigation",
)

# Lexical matching of any kind is prohibited: no stopword lists, no keyword
# tables, no stemmers, no regex-driven text search. Schema understanding comes
# from pg_catalog, retrieval comes from embeddings.
BANNED_MODULES: tuple[str, ...] = ("re", "regex", "nltk", "spacy", "gensim")
BANNED_ATTRIBUTES: frozenset[str] = frozenset(
    {"compile", "match", "search", "findall", "finditer", "sub", "split", "fullmatch"}
)

_WORD_LITERAL = re.compile(r"^[a-z]{2,}$")

PERMITTED_NUMBERS = frozenset({-1, 0, 1, 2, 1000}) | HTTP_STATUS_CODES

QUESTION_NAMES = frozenset({"question", "query", "user_input", "prompt"})

QUESTION_PREDICATES = frozenset({"startswith", "endswith", "count", "find", "index", "split"})


def _looks_like_a_word(value: str) -> bool:
    """True for a bare lowercase word such as 'the' or 'stop'."""
    return bool(_WORD_LITERAL.match(value))


def _collection_elements(node: ast.AST) -> list[ast.AST] | None:
    """Returns the literal elements of a list/tuple/set/frozenset, else None."""
    target = node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in ("frozenset", "set", "list", "tuple") or not node.args:
            return None
        target = node.args[0]
    elif isinstance(node, ast.Call):
        return None

    if isinstance(target, (ast.List, ast.Tuple, ast.Set)):
        return list(target.elts)
    return None


def _word_lists(tree: ast.Module, path: Path) -> set[str]:
    """Finds collections whose entries are all bare lowercase words.

    This is the shape of a stopword list or a keyword table. The rule is
    deliberately narrow -- two or more entries, all purely alphabetic and
    lowercase -- so SQL type names, function names and column names, which
    contain underscores or capitals, are not caught by accident.
    """
    findings: set[str] = set()
    for node in ast.walk(tree):
        elements = _collection_elements(node)
        if not elements or len(elements) < 2:
            continue
        values = [
            e.value for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if len(values) != len(elements):
            continue
        if all(_looks_like_a_word(v) for v in values):
            findings.add(f"word list {sorted(values)[:6]}")
    return findings


def _lexical_usage(tree: ast.Module) -> set[str]:
    """Flags imports of regular-expression or lexical-matching libraries."""
    findings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BANNED_MODULES:
                    findings.add(f"lexical module import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_MODULES:
                findings.add(f"lexical module import '{root}'")
    return findings


def _regex_usage(tree: ast.Module) -> set[str]:
    """Flags attribute calls into a regex module."""
    findings: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in BANNED_MODULES
            and node.attr in BANNED_ATTRIBUTES
        ):
            findings.add(f"regex call '{node.value.id}.{node.attr}'")
    return findings


def _token_pattern(phrase: str) -> re.Pattern[str]:
    """Builds a boundary-aware regex tolerant of _ - and whitespace separators.

    Phrases with no alphanumeric content (e.g. currency symbols) degrade to a
    plain substring match -- an empty token body would otherwise compile to a
    pattern that matches every string.
    """
    parts = [p for p in re.split(r"[^a-z0-9]+", phrase.lower()) if p]
    if not parts:
        return re.compile(re.escape(phrase))
    body = r"[\s_\-]*".join(re.escape(p) for p in parts)
    return re.compile(rf"\b{body}\b")


_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (phrase, _token_pattern(phrase)) for phrase in BANNED_COLUMNS + BANNED_VALUES
)

_TABLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)) for name in sorted(BANNED_TABLES)
)

# A literal only gets table-name scanning applied when it reads like SQL.
# This keeps the guard from flagging ordinary prose that happens to use a
# common word such as "cases" or "parties".
_SQL_HINT = re.compile(
    r"\b(select|insert\s+into|update|delete\s+from|from|join|truncate|drop|alter)\b",
    re.IGNORECASE,
)

# Documentation is allowed to discuss the prohibition itself.
_DOC_NODES: set[int] = set()


def _iter_source_files() -> list[Path]:
    if not APP_ROOT.exists():  # pragma: no cover - defensive
        return []
    return sorted(p for p in APP_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Collects ids of Constant nodes that are docstrings, so they can be skipped."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    ids.add(id(first.value))
    return ids


def _literal_strings(tree: ast.Module) -> list[str]:
    skip = _docstring_node_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


def _violations(path: Path, tree: ast.Module) -> list[str]:
    findings: set[str] = set()

    for literal in _literal_strings(tree):
        stripped = literal.strip().lower()
        if stripped in BANNED_TABLES:
            findings.add(f"hardcoded table name {stripped!r}")

        # Table names embedded in SQL text, e.g. "SELECT * FROM officers".
        if _SQL_HINT.search(literal):
            for name, pattern in _TABLE_PATTERNS:
                if pattern.search(literal):
                    findings.add(f"hardcoded table name {name!r} in SQL")

        for phrase, pattern in _TOKEN_PATTERNS:
            if pattern.search(literal.lower()):
                findings.add(f"hardcoded domain term {phrase!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BANNED_IMPORTS:
                    findings.add(f"banned import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module in BANNED_IMPORTS:
            findings.add(f"banned import from {node.module!r}")

    rel = _display_path(path)
    return [f"{rel}: {f}" for f in sorted(findings)]


def _display_path(path: Path) -> str:
    """Renders a path relative to the project when possible, else by name."""
    try:
        return path.relative_to(APP_ROOT.parent).as_posix()
    except ValueError:
        return path.name


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:  # pragma: no cover - defensive
        return None


def _all_violations() -> list[str]:
    findings: list[str] = []
    for path in _iter_source_files():
        tree = _parse(path)
        if tree is None:
            continue
        findings.extend(_violations(path, tree))
    return findings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_application_sources_exist() -> None:
    """Sanity check -- the guard is only meaningful if it actually walks files."""
    files = _iter_source_files()
    assert files, f"no Python sources discovered under {APP_ROOT}"


def test_no_hardcoded_domain_vocabulary_in_app() -> None:
    """No string literal anywhere in app/ may name a concrete table, column or value."""
    findings = _all_violations()
    assert not findings, "Hardcoded domain knowledge detected:\n  " + "\n  ".join(findings)


def test_no_word_lists_in_app() -> None:
    """No collection of bare words may exist. That shape is a stopword list."""
    findings: set[str] = set()
    for path in _iter_source_files():
        tree = _parse(path)
        if tree is None:
            continue
        findings.update(f"{path.name}: {f}" for f in _word_lists(tree, path))
    assert not findings, "Word lists detected:\n  " + "\n  ".join(sorted(findings))


def test_no_regex_or_lexical_matching_in_app() -> None:
    """Lexical matching is prohibited; retrieval is embedding-based only."""
    findings: set[str] = set()
    for path in _iter_source_files():
        tree = _parse(path)
        if tree is None:
            continue
        findings.update(f"{path.name}: {f}" for f in _lexical_usage(tree))
        findings.update(f"{path.name}: {f}" for f in _regex_usage(tree))
    assert not findings, "Lexical matching detected:\n  " + "\n  ".join(sorted(findings))


def test_no_magic_numbers_in_app() -> None:
    """Tunable numbers belong in .env, not inline in the runtime.

    The allowlist holds only positions and unit conversions, which carry no
    policy: indexing offsets, a rounding pair, and milliseconds per second.
    """
    findings: set[str] = set()
    for path in _iter_source_files():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(node.value, bool):
                continue
            if isinstance(node.value, str):
                if node.value.isdigit():
                    findings.add(f"{path.name}:{node.lineno} numeric string {node.value!r}")
                continue
            if not isinstance(node.value, int | float):
                continue
            if node.value in PERMITTED_NUMBERS:
                continue
            findings.add(f"{path.name}:{node.lineno} numeric literal {node.value!r}")
    assert not findings, "Magic numbers detected:\n  " + "\n  ".join(sorted(findings))


def test_no_scenario_branching_on_the_question() -> None:
    """The runtime may not inspect what the user asked for.

    Any comparison of the incoming question against a literal -- equality,
    containment, a prefix test, a lowercase fold -- would be catering to one
    phrasing. Everything must be resolved from the catalog instead.
    """
    findings: set[str] = set()
    for path in _iter_source_files():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and _compares_question_to_a_literal(node):
                findings.add(f"{path.name}:{node.lineno} compares question text")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in QUESTION_PREDICATES
                and _touches_question_text(node.func.value)
            ):
                findings.add(f"{path.name}:{node.lineno} applies {node.func.attr}")
            if isinstance(node, ast.Subscript) and _is_question_itself(node.value):
                findings.add(f"{path.name}:{node.lineno} indexes question text")
    assert not findings, "Scenario branching detected:\n  " + "\n  ".join(sorted(findings))


def test_every_setting_is_declared_without_a_default() -> None:
    """.env is the only source of configuration, so no field may carry one."""
    from pydantic.fields import PydanticUndefined

    from app.config import Settings

    offenders = [
        name
        for name, field in Settings.model_fields.items()
        if field.default is not PydanticUndefined or field.default_factory is not None
    ]
    assert not offenders, f"settings with a Python default: {sorted(offenders)}"


def test_runtime_constants_are_referenced_not_repeated() -> None:
    """The SQL operator set must have exactly one home in app/.

    It is used both to validate a plan and to describe the contract to the
    model. Written twice, the two copies drift and the model is told to use
    operators the engine will then reject.
    """
    sources = _iter_source_files()
    holders = [
        path.name
        for path in sources
        if "OPERATORS = " in path.read_text(encoding="utf-8")
        or "OPERATORS: " in path.read_text(encoding="utf-8")
    ]
    assert holders == ["constants.py"], f"operator set declared in {holders}"


def test_no_module_constant_is_left_unused() -> None:
    """A constant defined and never read is either dead or a duplicate.

    Duplicates are worse than dead code: they drift, and the engine ends up
    validating against one copy while the model was told about another.
    """
    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in _iter_source_files()}
    findings: set[str] = set()

    for path, tree in trees.items():
        for node in tree.body:
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                uses = sum(
                    1
                    for other in trees.values()
                    for inner in ast.walk(other)
                    if isinstance(inner, ast.Name) and inner.id == target.id
                )
                if uses <= 1:
                    findings.add(f"{path.name}:{target.id}")

    assert not findings, "Unused module constants:\n  " + "\n  ".join(sorted(findings))


def _compares_question_to_a_literal(node: ast.Compare) -> bool:
    """Flag a question compared against something baked into the code.

    Bounding the length against a configured setting is not branching on what
    was asked, so an operand that is a literal is what makes this a problem.
    """
    operands = [node.left, *node.comparators]
    if not any(isinstance(operand, ast.Constant) for operand in operands):
        return False
    return any(_touches_question_text(operand) for operand in operands)


def _touches_question_text(node: ast.AST) -> bool:
    """True when the expression is, or is derived from, the incoming question.

    Only for use where the whole subtree is the thing being inspected. Handing
    the question to a function -- ``embedder.aencode([question])`` -- is how
    the runtime is supposed to use it, and that is not branching.
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in QUESTION_NAMES:
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in QUESTION_NAMES:
            return True
    return False


def _is_question_itself(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in QUESTION_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in QUESTION_NAMES
    return False


def test_guard_would_catch_a_reintroduced_hardcode(tmp_path: Path) -> None:
    """Proves the guard is not vacuous by feeding it a deliberate violation."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def run(query):\n"
        "    if 'sharma' in query:\n"
        "        return 'SELECT * FROM officers'\n"
        "    return None\n",
        encoding="utf-8",
    )
    tree = ast.parse(sample.read_text(encoding="utf-8"))
    findings = _violations(sample, tree)

    assert any("'sharma'" in f for f in findings), findings
    assert any("'officers'" in f for f in findings), findings


@pytest.mark.parametrize(
    "source",
    [
        "def run(query):\n    if query == 'total revenue':\n        pass\n",
        "def run(question):\n    if 'refund' in question:\n        pass\n",
        "def run(question):\n    if question.startswith('show me'):\n        pass\n",
        "def run(question):\n    first = question.split()[0]\n",
        "def run(question):\n    n = question.count('and')\n",
        "def run(question):\n    head = question[0]\n",
    ],
)
def test_scenario_guard_is_not_vacuous(source: str) -> None:
    """Each shape of keyword catering must be rejected."""
    tree = ast.parse(source)
    findings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and _compares_question_to_a_literal(node):
            findings.add("compare")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in QUESTION_PREDICATES
            and _touches_question_text(node.func.value)
        ):
            findings.add("predicate")
        if isinstance(node, ast.Subscript) and _is_question_itself(node.value):
            findings.add("subscript")
    assert findings, f"guard missed: {source!r}"


def test_length_bounds_against_settings_are_not_flagged() -> None:
    """Bounded input length is policy from .env, not catering to a phrase."""
    source = (
        "from app.config import settings\n"
        "def run(question):\n"
        "    query = (question or '').strip()\n"
        "    if len(query) < settings.min_query_length:\n"
        "        raise ValueError('short')\n"
        "    return query\n"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Compare) and _compares_question_to_a_literal(node)), source


def test_passing_the_question_to_a_function_is_not_flagged() -> None:
    """Embedding the question is the intended use of it."""
    source = "async def run(question):\n    return await embedder.aencode([question])\n"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            assert not _is_question_itself(node.value), source


@pytest.mark.parametrize(
    "source",
    [
        "TIMEOUT = 30\n",
        "LIMIT = 25\n",
        "def run():\n    return [1] * 25\n",
        "def run():\n    return round(1.0, 6)\n",
        "def run():\n    return '12'\n",
    ],
)
def test_magic_number_guard_is_not_vacuous(source: str) -> None:
    tree = ast.parse(source)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.isdigit():
                findings.append(node.value)
            continue
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int | float):
            continue
        if node.value in PERMITTED_NUMBERS:
            continue
        findings.append(node.value)
    assert findings, f"guard missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "def run():\n    return 500\n",
        "def run():\n    return ms * 1000\n",
        "def run():\n    return value[2]\n",
        "def run():\n    return [0, 1, -1]\n",
        "def run():\n    return 'LENGTH-12'\n",
    ],
)
def test_magic_number_guard_permits_the_allowlist(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int | float):
            continue
        assert node.value in PERMITTED_NUMBERS, source


@pytest.mark.parametrize(
    "module_path",
    [
        "app/render.py",
        "app/guard.py",
        "app/graph.py",
        "app/service.py",
    ],
)
def test_core_modules_do_not_import_a_domain(module_path: str) -> None:
    path = APP_ROOT.parent / module_path
    if not path.exists():
        pytest.skip(f"{module_path} not present")
    tree = _parse(path)
    assert tree is not None, f"could not parse {module_path}"

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offenders = imported.intersection(BANNED_IMPORTS)
    assert not offenders, f"{module_path} imports banned module(s): {sorted(offenders)}"
