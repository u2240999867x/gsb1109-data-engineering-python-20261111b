"""
Index slimming preview for a declared workload.

Given a table and a declared workload - the query patterns and uniqueness
constraints the application actually relies on - this module works out the
minimal set of existing indexes that must be retained to keep serving every
declared fetch order and to keep enforcing every declared uniqueness rule.

The workload model is deliberately limited:

* Each query declares an equality prefix (columns pinned to parameters or
  constants), at most one range column, an ``ORDER BY`` list with per-column
  direction, and optional per-column collations.
* Each uniqueness rule declares a set of columns and optional partial-index
  style predicate constants.
* Partial index predicates are only understood when they are a conjunction
  of ``column = constant`` equalities. Anything else (ranges, expressions,
  ``OR``, ...) marks the index as ``unknown`` - it is conservatively kept
  but never used to justify dropping another index.

The preview reports three groups:

* ``keep`` - indexes in the minimal retained set. Each entry explains what
  would be lost if it were dropped anyway: the query orderings that no
  other retained index provides (``lost_orders``) and example duplicate
  rows that could then be written (``writable_rows``).
* ``drop`` - indexes safe to remove, each with the ``successor`` retained
  index that takes over its declared duties.
* ``unknown`` - indexes that could not be analyzed (unsupported predicate
  or expression columns). They are kept, but cannot act as successors.

Requirements no existing index can satisfy are listed under ``uncovered``.
"""

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

# Exact set-cover is attempted up to this many candidate indexes,
# beyond that a greedy fallback is used.
_EXACT_COVER_LIMIT = 16

_MISSING = object()


def _quote(identifier: str) -> str:
    return '"{}"'.format(identifier.replace('"', '""'))


@dataclass
class KeyColumn:
    name: str
    desc: bool
    coll: str


@dataclass
class IndexDetails:
    name: str
    unique: bool
    origin: str
    partial: bool
    columns: list[KeyColumn]
    predicate: dict[str, Any] = field(default_factory=dict)
    predicate_sql: str | None = None
    unknown_reason: str | None = None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class DeclaredQuery:
    name: str
    equality: dict[str, Any]  # column -> constant, or None for a parameter
    range: str | None
    order_by: list[tuple[str, str]]
    collate: dict[str, str]


@dataclass
class DeclaredUnique:
    name: str
    columns: list[str]
    where: dict[str, Any]


# ---------------------------------------------------------------------------
# Predicate parsing - only "col = constant AND ..." conjunctions are understood
# ---------------------------------------------------------------------------

_IDENTIFIER = r"(?:\"(?:[^\"]|\"\")+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)"
_COLUMN_REF_RE = re.compile(
    r"^\s*(?:" + _IDENTIFIER + r"\s*\.\s*)?(?P<col>" + _IDENTIFIER + r")\s*$"
)
_STRING_RE = re.compile(r"^'((?:[^']|'')*)'$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _unquote_identifier(token: str) -> str:
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('""', '"')
    if token.startswith("`") and token.endswith("`"):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        return token[1:-1]
    return token


def _parse_literal(token: str) -> Any:
    token = token.strip()
    match = _STRING_RE.match(token)
    if match:
        return match.group(1).replace("''", "'")
    if _NUMBER_RE.match(token):
        return float(token) if any(c in token for c in ".eE") else int(token)
    if token.lower() == "true":
        return 1
    if token.lower() == "false":
        return 0
    return _MISSING


def _split_top_level(text: str, keyword: str) -> list[str] | None:
    "Split text on a top-level keyword, respecting quotes and parentheses."
    parts = []
    depth = 0
    quote = None
    start = 0
    i = 0
    keyword = keyword.lower()
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                if quote == "'" and text[i + 1 : i + 2] == "'":
                    i += 2
                    continue
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == "[":
            quote = "]"
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
        elif (
            depth == 0
            and text[i : i + len(keyword)].lower() == keyword
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
            and (
                i + len(keyword) >= len(text)
                or not (
                    text[i + len(keyword)].isalnum() or text[i + len(keyword)] == "_"
                )
            )
        ):
            parts.append(text[start:i])
            i += len(keyword)
            start = i
            continue
        i += 1
    if quote is not None or depth != 0:
        return None
    parts.append(text[start:])
    return parts


def _extract_where(sql: str) -> str | None:
    "Return the top-level WHERE clause text of a CREATE INDEX statement."
    parts = _split_top_level(sql, "where")
    if parts is None or len(parts) < 2:
        return None
    # Everything after the first top-level WHERE
    where_text = sql[len(parts[0]) + len("where") :]
    return where_text.strip().rstrip(";").strip()


def _strip_wrapping_parens(term: str) -> str:
    "Remove redundant parentheses wrapping a whole term, e.g. ``((a = 1))``."
    term = term.strip()
    while term.startswith("("):
        depth = 0
        quote = None
        wraps = False
        i = 0
        while i < len(term):
            ch = term[i]
            if quote is not None:
                if ch == quote:
                    if quote == "'" and term[i + 1 : i + 2] == "'":
                        i += 2
                        continue
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "[":
                quote = "]"
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    wraps = i == len(term) - 1
                    break
            i += 1
        if not wraps:
            return term
        term = term[1:-1].strip()
    return term


def parse_predicate(sql: str) -> dict[str, Any] | None:
    """
    Parse the WHERE clause of a partial index CREATE statement.

    Returns a ``{column: constant}`` dictionary if the predicate is a
    conjunction of ``column = constant`` equalities, otherwise ``None``.
    """
    where = _extract_where(sql)
    if where is None or not where:
        return None
    terms = _split_top_level(where, "and")
    if terms is None:
        return None
    predicate: dict[str, Any] = {}
    for term in terms:
        term = _strip_wrapping_parens(term)
        if not term:
            return None
        sides = _split_on_equals(term)
        if sides is None or len(sides) != 2:
            return None
        left, right = sides
        left_col = _COLUMN_REF_RE.match(left)
        right_col = _COLUMN_REF_RE.match(right)
        left_lit = _parse_literal(left)
        right_lit = _parse_literal(right)
        if left_col is not None and right_lit is not _MISSING:
            column, value = _unquote_identifier(left_col.group("col")), right_lit
        elif right_col is not None and left_lit is not _MISSING:
            column, value = _unquote_identifier(right_col.group("col")), left_lit
        else:
            return None
        if column in predicate and predicate[column] != value:
            return None
        predicate[column] = value
    return predicate


def _split_on_equals(text: str) -> list[str] | None:
    "Split on the first top-level '=' that is not part of <=, >=, !=, ==."
    depth = 0
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                if quote == "'" and text[i + 1 : i + 2] == "'":
                    i += 2
                    continue
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == "[":
            quote = "]"
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev in "<>!" or nxt == "=":
                return None
            return [text[:i], text[i + 1 :]]
        i += 1
    return None


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def _collect_indexes(table) -> list[IndexDetails]:
    db = table.db
    indexes = []
    for row in db.execute_returning_dicts(
        f"PRAGMA index_list({_quote(table.name)})"
    ):
        name = row["name"]
        key_columns = []
        expression = False
        for info in db.execute_returning_dicts(f"PRAGMA index_xinfo({_quote(name)})"):
            if not info.get("key", 1):
                continue
            if info.get("name") is None:
                expression = True
                continue
            key_columns.append(
                KeyColumn(
                    name=info["name"],
                    desc=bool(info.get("desc", 0)),
                    coll=(info.get("coll") or "BINARY").upper(),
                )
            )
        sql_row = db.execute(
            "select sql from sqlite_master where type = 'index' and name = ?",
            [name],
        ).fetchone()
        sql = sql_row[0] if sql_row else None
        partial = bool(row.get("partial", 0))
        predicate: dict[str, Any] = {}
        predicate_sql = None
        unknown_reason = None
        if expression:
            unknown_reason = "index uses expressions, which are not supported"
        elif partial:
            predicate_sql = _extract_where(sql) if sql else None
            parsed = parse_predicate(sql) if sql else None
            if parsed is None:
                unknown_reason = (
                    "unsupported partial index predicate: "
                    + (predicate_sql or "unknown")
                )
            else:
                predicate = parsed
        indexes.append(
            IndexDetails(
                name=name,
                unique=bool(row.get("unique", 0)),
                origin=row.get("origin", "c"),
                partial=partial,
                columns=key_columns,
                predicate=predicate,
                predicate_sql=predicate_sql,
                unknown_reason=unknown_reason,
            )
        )
    # An INTEGER PRIMARY KEY column is an alias for the rowid: SQLite
    # creates no entry in PRAGMA index_list for it, but it still serves
    # equality, range and ordered access on that column
    pks = table.pks
    pk_type = (
        next(
            (column.type for column in table.columns if column.name == pks[0]),
            "",
        )
        if len(pks) == 1
        else ""
    )
    if len(pks) == 1 and str(pk_type).upper() == "INTEGER":
        indexes.append(
            IndexDetails(
                name="rowid",
                unique=True,
                origin="pk",
                partial=False,
                columns=[KeyColumn(pks[0], False, "BINARY")],
            )
        )
    return indexes


# ---------------------------------------------------------------------------
# Workload normalization
# ---------------------------------------------------------------------------


def _normalize_queries(queries) -> list[DeclaredQuery]:
    normalized = []
    for i, query in enumerate(queries or []):
        equality = query.get("equality", [])
        if isinstance(equality, dict):
            equality_dict = dict(equality)
        else:
            equality_dict = {column: None for column in equality}
        order_by = []
        for item in query.get("order_by") or []:
            if isinstance(item, str):
                order_by.append((item, "asc"))
            elif isinstance(item, dict):
                order_by.append((item["column"], item.get("direction", "asc")))
            else:
                column, *rest = item
                order_by.append((column, rest[0] if rest else "asc"))
        order_by = [(column, direction.lower()) for column, direction in order_by]
        for _, direction in order_by:
            if direction not in ("asc", "desc"):
                raise ValueError(f"Invalid sort direction: {direction!r}")
        collate = {
            column: collation.upper()
            for column, collation in (query.get("collate") or {}).items()
        }
        normalized.append(
            DeclaredQuery(
                name=query.get("name") or f"query_{i + 1}",
                equality=equality_dict,
                range=query.get("range"),
                order_by=order_by,
                collate=collate,
            )
        )
    return normalized


def _normalize_uniques(uniques) -> list[DeclaredUnique]:
    normalized = []
    for i, unique in enumerate(uniques or []):
        normalized.append(
            DeclaredUnique(
                name=unique.get("name") or f"unique_{i + 1}",
                columns=list(unique["columns"]),
                where=dict(unique.get("where") or {}),
            )
        )
    return normalized


# ---------------------------------------------------------------------------
# Coverage rules
# ---------------------------------------------------------------------------


def _collation_matches(required: str | None, index_coll: str) -> bool:
    return (required or "BINARY").upper() == index_coll.upper()


def _index_covers_query(index: IndexDetails, query: DeclaredQuery) -> bool:
    if index.unknown_reason is not None:
        return False
    # A partial index only serves queries that pin every predicate
    # column to the same constant
    for column, constant in index.predicate.items():
        if query.equality.get(column, _MISSING) != constant:
            return False
    available = set(query.equality) | set(index.predicate)
    columns = index.columns
    pos = 0
    while pos < len(columns) and columns[pos].name in available:
        if not _collation_matches(
            query.collate.get(columns[pos].name), columns[pos].coll
        ):
            return False
        pos += 1
    range_pos = None
    if query.range is not None:
        if pos >= len(columns) or columns[pos].name != query.range:
            return False
        if not _collation_matches(query.collate.get(query.range), columns[pos].coll):
            return False
        range_pos = pos
        pos += 1
    # ORDER BY terms pinned to constants by the equality prefix are no-ops
    needed = [(c, d) for c, d in query.order_by if c not in available]
    if needed:
        start = range_pos if range_pos is not None else pos
        segment = columns[start : start + len(needed)]
        if len(segment) < len(needed):
            return False
        for (column, _), key_column in zip(needed, segment):
            if key_column.name != column:
                return False
            if not _collation_matches(query.collate.get(column), key_column.coll):
                return False
        # An index scan provides the required order if every column
        # direction matches, or if every direction is reversed (the
        # index can be scanned backwards)
        forward = all(
            key_column.desc == (direction == "desc")
            for (_, direction), key_column in zip(needed, segment)
        )
        backward = all(
            key_column.desc != (direction == "desc")
            for (_, direction), key_column in zip(needed, segment)
        )
        if not (forward or backward):
            return False
    return True


def _index_covers_unique(index: IndexDetails, unique: DeclaredUnique) -> bool:
    if index.unknown_reason is not None or not index.unique:
        return False
    if set(index.column_names) != set(unique.columns):
        return False
    if not index.partial:
        return True
    # A partial unique index only enforces the rule inside its predicate
    return bool(unique.where) and index.predicate == unique.where


# ---------------------------------------------------------------------------
# Successor rules
# ---------------------------------------------------------------------------


def _predicate_compatible(candidate: IndexDetails, dropped: IndexDetails) -> bool:
    "candidate indexes at least the rows that dropped does."
    if not dropped.partial:
        return not candidate.partial
    if not candidate.partial:
        return True
    return candidate.predicate == dropped.predicate


def _key_tuple(index: IndexDetails) -> list[tuple[str, bool, str]]:
    return [(c.name, c.desc, c.coll) for c in index.columns]


def _is_successor(candidate: IndexDetails, dropped: IndexDetails) -> bool:
    if candidate.unknown_reason is not None:
        return False
    if not _predicate_compatible(candidate, dropped):
        return False
    if dropped.unique:
        # Only an equivalent unique index can take over a constraint
        return candidate.unique and set(candidate.column_names) == set(
            dropped.column_names
        )
    # A non-unique index is redundant if its key columns are a left
    # prefix of the candidate's key columns
    dropped_keys = _key_tuple(dropped)
    candidate_keys = _key_tuple(candidate)
    return candidate_keys[: len(dropped_keys)] == dropped_keys


# ---------------------------------------------------------------------------
# Minimal set cover
# ---------------------------------------------------------------------------


def _minimal_cover(
    candidate_names: list[str],
    remaining: set[int],
    covered_by: dict[str, set[int]],
    key_counts: dict[str, int],
) -> list[str]:
    if not remaining:
        return []
    names = sorted(candidate_names)
    if len(names) <= _EXACT_COVER_LIMIT:
        best = None
        for size in range(1, len(names) + 1):
            for combo in combinations(names, size):
                union: set[int] = set()
                for name in combo:
                    union |= covered_by[name]
                if union >= remaining:
                    score = (sum(key_counts[n] for n in combo), combo)
                    if best is None or score < best[0]:
                        best = (score, list(combo))
            if best is not None:
                return sorted(best[1])
        return []
    # Greedy fallback for very large index counts
    chosen = []
    left = set(remaining)
    while left:
        best_name = min(
            names,
            key=lambda n: (-len(covered_by[n] & left), key_counts[n], n),
        )
        if not covered_by[best_name] & left:
            break
        chosen.append(best_name)
        left -= covered_by[best_name]
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------


def _describe_query(query: DeclaredQuery) -> str:
    parts = []
    for column, value in query.equality.items():
        parts.append(f"{column} = ?" if value is None else f"{column} = {value!r}")
    if query.range is not None:
        parts.append(f"{query.range} BETWEEN ? AND ?")
    description = " AND ".join(parts) if parts else "(full table scan)"
    if query.order_by:
        description += " ORDER BY " + ", ".join(
            f"{column} {direction.upper()}" for column, direction in query.order_by
        )
    return description


def _writable_rows(unique: DeclaredUnique) -> list[dict[str, Any]]:
    row: dict[str, Any] = {column: f"<{column}>" for column in unique.columns}
    row.update(unique.where)
    return [row, dict(row)]


def _column_label(column: KeyColumn) -> str:
    label = column.name
    if column.desc:
        label += " DESC"
    if column.coll != "BINARY":
        label += f" COLLATE {column.coll}"
    return label


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def index_slimming_preview(table, queries=None, uniques=None) -> dict:
    """
    Preview which indexes on ``table`` can be dropped given a declared
    workload, and which must be kept. See module docstring for details.

    :param table: A ``sqlite_utils.db.Table`` instance
    :param queries: List of declared query patterns, each a dictionary with
      optional ``name``, ``equality`` (list of columns or ``{column:
      constant}`` mapping, ``None`` marks a bound parameter), ``range`` (a
      single range column), ``order_by`` (list of columns, ``(column,
      direction)`` pairs or ``{"column", "direction"}`` dictionaries) and
      ``collate`` (``{column: collation}`` mapping)
    :param uniques: List of declared uniqueness rules, each a dictionary
      with ``columns`` and optional ``name`` and ``where``
      (``{column: constant}`` predicate for partial rules)
    """
    declared_queries = _normalize_queries(queries)
    declared_uniques = _normalize_uniques(uniques)
    indexes = _collect_indexes(table)

    requirements: list[tuple[str, Any]] = [("query", q) for q in declared_queries] + [
        ("unique", u) for u in declared_uniques
    ]

    known = [idx for idx in indexes if idx.unknown_reason is None]
    unknown = [idx for idx in indexes if idx.unknown_reason is not None]

    def covers(index: IndexDetails, requirement) -> bool:
        kind, declared = requirement
        if kind == "query":
            return _index_covers_query(index, declared)
        return _index_covers_unique(index, declared)

    covered_by: dict[str, set[int]] = {
        idx.name: {i for i, req in enumerate(requirements) if covers(idx, req)}
        for idx in known
    }
    universe = set(range(len(requirements)))
    covered_universe = set().union(*covered_by.values()) if covered_by else set()
    uncovered = sorted(universe - covered_universe)

    # Automatic indexes (primary key / UNIQUE constraint) cannot be dropped,
    # so they are always retained and cover their requirements for free
    automatic = [idx for idx in known if idx.origin in ("pk", "u")]
    droppable = [idx for idx in known if idx.origin not in ("pk", "u")]
    remaining = (
        covered_universe - set().union(*(covered_by[idx.name] for idx in automatic))
        if automatic
        else covered_universe
    )

    retained_names = {idx.name for idx in automatic}
    retained_names.update(
        _minimal_cover(
            [idx.name for idx in droppable],
            remaining,
            covered_by,
            {idx.name: len(idx.columns) for idx in known},
        )
    )

    by_name = {idx.name: idx for idx in indexes}
    retained = [by_name[name] for name in sorted(retained_names)]
    dropped = [idx for idx in droppable if idx.name not in retained_names]

    # A requirement is "solely provided" by the one retained index that
    # covers it - dropping that index would lose it
    sole_provider: dict[int, str] = {}
    for requirement_id in covered_universe:
        providers = [
            idx.name
            for idx in retained
            if requirement_id in covered_by.get(idx.name, set())
        ]
        if len(providers) == 1:
            sole_provider[requirement_id] = providers[0]

    keep = []
    for idx in retained:
        lost_orders = []
        writable_rows = []
        for requirement_id, provider in sole_provider.items():
            if provider != idx.name:
                continue
            kind, declared = requirements[requirement_id]
            if kind == "query":
                lost_orders.append(_describe_query(declared))
            else:
                writable_rows.extend(_writable_rows(declared))
        entry = {
            "index": idx.name,
            "unique": idx.unique,
            "origin": idx.origin,
            "columns": [_column_label(c) for c in idx.columns],
            "covers": sorted(
                requirements[i][1].name for i in covered_by.get(idx.name, set())
            ),
            "if_dropped": {
                "lost_orders": lost_orders,
                "writable_rows": writable_rows,
            },
        }
        if idx.partial:
            entry["predicate"] = idx.predicate
        if idx.origin in ("pk", "u"):
            entry["reason"] = (
                f"automatic index (origin {idx.origin!r}) cannot be dropped"
            )
        else:
            entry["reason"] = "required by the declared workload"
        keep.append(entry)

    drop = []
    for idx in dropped:
        successors = [
            candidate
            for candidate in retained
            if _is_successor(candidate, idx)
        ]
        if not successors:
            # Fall back to any retained index covering the same requirements
            own = covered_by.get(idx.name, set())
            if own:
                successors = [
                    candidate
                    for candidate in retained
                    if covered_by.get(candidate.name, set()) >= own
                ]
        successor = None
        if successors:
            successor = min(successors, key=lambda c: (len(c.columns), c.name)).name
        if successor is not None:
            reason = f"all declared uses are covered by {successor!r}"
        elif covered_by.get(idx.name):
            reason = "declared uses are split across several retained indexes"
        else:
            reason = "not used by the declared workload"
        entry = {
            "index": idx.name,
            "successor": successor,
            "reason": reason,
        }
        if idx.unique:
            entry["unique_columns"] = idx.column_names
            entry["warning"] = (
                "dropping this unique index also removes a uniqueness rule "
                "that is not part of the declared workload"
            )
        drop.append(entry)

    unknown_entries = [
        {
            "index": idx.name,
            "reason": idx.unknown_reason,
            "predicate": idx.predicate_sql,
            "kept": True,
        }
        for idx in unknown
    ]

    uncovered_entries = []
    for requirement_id in uncovered:
        kind, declared = requirements[requirement_id]
        entry = {"type": kind, "name": declared.name}
        if kind == "query":
            entry["detail"] = (
                "no existing index provides: " + _describe_query(declared)
            )
        else:
            entry["detail"] = (
                "no existing unique index enforces uniqueness on: "
                + ", ".join(declared.columns)
            )
        uncovered_entries.append(entry)

    return {
        "table": table.name,
        "keep": keep,
        "drop": drop,
        "unknown": unknown_entries,
        "uncovered": uncovered_entries,
    }
