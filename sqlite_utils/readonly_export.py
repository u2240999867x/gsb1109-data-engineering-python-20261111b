"""
Read-only, auditable dataset exports.

This module implements the machinery behind ``sqlite-utils export-readonly``.
Reports are executed through a SQLite `authorizer callback`_ that:

* rejects every write, DDL, transaction or attach statement before it can run
* records the source tables and columns each statement actually reads,
  including columns reached through views (the authorizer reports the nesting
  view in its ``source`` argument, so views nested two levels deep are
  resolved all the way down to the underlying table)
* keeps tables that live in attached databases apart, even when they share a
  name, because every read event carries its schema (``main`` / attach alias)
* only permits functions the caller explicitly registered as pure functions

Nothing is published until every statement in the batch has been validated and
every attached source has proven it belongs to the version the caller asked
for, so a failed export never leaves behind a partial result set.

.. _authorizer callback: https://www.sqlite.org/c3ref/set_authorizer.html
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections import defaultdict
from typing import Any, Generator, Sequence

from .db import Database, _iter_complete_sql_statements, quote_identifier
from .utils import sqlite3

# Authorizer action codes that mutate state. SQLITE_TRANSACTION covers
# BEGIN/COMMIT/ROLLBACK; SQLITE_SAVEPOINT covers SAVEPOINT/RELEASE.
_WRITE_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
    }
)

# Tables maintained by SQLite itself. Reads of these are needed when SQLite
# expands a view definition but they are not caller-owned source data.
_INTERNAL_TABLES = frozenset(
    {
        "sqlite_master",
        "sqlite_schema",
        "sqlite_temp_master",
        "sqlite_temp_schema",
    }
)

# Introspection pragmas that take an object name (table/index) as their
# argument but never mutate state. Any other pragma carrying an argument is an
# assignment such as ``PRAGMA user_version = 5`` and is refused, so unknown
# pragmas fail closed.
_READ_PRAGMAS = frozenset(
    {
        "collation_list",
        "compile_options",
        "database_list",
        "foreign_key_check",
        "foreign_key_list",
        "function_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "module_list",
        "pragma_list",
        "table_info",
        "table_list",
        "table_xinfo",
    }
)

# Authorizer action codes share numeric values with unrelated result codes
# (SQLITE_UPDATE == SQLITE_AUTH == 23, SQLITE_INSERT == SQLITE_TOOBIG == 18),
# so action names need an explicit mapping rather than a reverse lookup.
_ACTION_NAMES = {
    sqlite3.SQLITE_CREATE_INDEX: "create-index",
    sqlite3.SQLITE_CREATE_TABLE: "create-table",
    sqlite3.SQLITE_CREATE_TEMP_INDEX: "create-temp-index",
    sqlite3.SQLITE_CREATE_TEMP_TABLE: "create-temp-table",
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER: "create-temp-trigger",
    sqlite3.SQLITE_CREATE_TEMP_VIEW: "create-temp-view",
    sqlite3.SQLITE_CREATE_TRIGGER: "create-trigger",
    sqlite3.SQLITE_CREATE_VIEW: "create-view",
    sqlite3.SQLITE_CREATE_VTABLE: "create-vtable",
    sqlite3.SQLITE_DELETE: "delete",
    sqlite3.SQLITE_DROP_INDEX: "drop-index",
    sqlite3.SQLITE_DROP_TABLE: "drop-table",
    sqlite3.SQLITE_DROP_TEMP_INDEX: "drop-temp-index",
    sqlite3.SQLITE_DROP_TEMP_TABLE: "drop-temp-table",
    sqlite3.SQLITE_DROP_TEMP_TRIGGER: "drop-temp-trigger",
    sqlite3.SQLITE_DROP_TEMP_VIEW: "drop-temp-view",
    sqlite3.SQLITE_DROP_TRIGGER: "drop-trigger",
    sqlite3.SQLITE_DROP_VIEW: "drop-view",
    sqlite3.SQLITE_DROP_VTABLE: "drop-vtable",
    sqlite3.SQLITE_INSERT: "insert",
    sqlite3.SQLITE_ALTER_TABLE: "alter-table",
    sqlite3.SQLITE_REINDEX: "reindex",
    sqlite3.SQLITE_ANALYZE: "analyze",
    sqlite3.SQLITE_TRANSACTION: "transaction",
    sqlite3.SQLITE_SAVEPOINT: "savepoint",
    sqlite3.SQLITE_ATTACH: "attach",
    sqlite3.SQLITE_DETACH: "detach",
    sqlite3.SQLITE_UPDATE: "update",
}


@dataclasses.dataclass(frozen=True, order=True)
class SourceColumn:
    """A single table column read by a report.

    :param database: schema the table lives in - ``main``, ``temp`` or the
      alias of an attached database
    :param table: underlying table the column was read from
    :param column: column that was read
    :param via: views the read passed through, outermost first - a direct read
      is an empty tuple, a read through ``v2`` -> ``v1`` is ``("v2", "v1")``
    """

    database: str
    table: str
    column: str
    via: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "table": self.table,
            "column": self.column,
            "via": list(self.via),
        }


@dataclasses.dataclass(frozen=True, order=True)
class SourceTable:
    """A table or view referenced by a report."""

    database: str
    name: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, order=True)
class FunctionUse:
    """A SQL function invoked by a report."""

    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclasses.dataclass(frozen=True)
class SourceVersion:
    """Version certification outcome for one attached schema."""

    database: str
    file: str | None
    expected: str | None
    actual: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ReportResult:
    """The validated SQL of one report together with its read manifest."""

    index: int
    sql: str
    columns_read: list[SourceColumn]
    tables_read: list[SourceTable]
    functions_used: list[FunctionUse]
    pragmas_used: list[str]
    rows: list[dict[str, Any]]

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sql": self.sql,
            "columns_read": [column.to_dict() for column in self.columns_read],
            "tables_read": [table.to_dict() for table in self.tables_read],
            "functions_used": [function.to_dict() for function in self.functions_used],
            "pragmas_used": list(self.pragmas_used),
        }


@dataclasses.dataclass
class ExportManifest:
    """The full manifest for a batch of reports."""

    reports: list[ReportResult]
    sources: list[SourceVersion]
    unauthenticated_tables: list[SourceTable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [report.manifest_dict() for report in self.reports],
            "sources": [source.to_dict() for source in self.sources],
            "unauthenticated_tables": [
                table.to_dict() for table in self.unauthenticated_tables
            ],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ReadOnlyExportError(Exception):
    """Raised when a batch cannot be safely exported.

    All problems discovered across the whole batch are collected into
    :attr:`reasons` so the caller gets one complete refusal instead of a
    sequence of partial failures. Tables belonging to sources that could not
    be version-certified are listed in :attr:`unauthenticated_tables`.
    """

    def __init__(
        self,
        reasons: str | Sequence[str],
        unauthenticated_tables: Sequence[SourceTable] = (),
    ):
        if isinstance(reasons, str):
            reasons = [reasons]
        self.reasons: list[str] = list(reasons)
        self.unauthenticated_tables: list[SourceTable] = list(
            unauthenticated_tables
        )
        super().__init__("; ".join(self.reasons))


@dataclasses.dataclass
class _ReadEvent:
    database: str
    table: str
    column: str
    source: str | None


@dataclasses.dataclass
class _Policy:
    # (schema, table) -> set of permitted columns
    allowed_columns: dict[tuple[str, str], frozenset[str]]
    allowed_functions: frozenset[str]
    # schema -> expected version string
    expected_versions: dict[str, str]
    # schema -> SQL returning one row/one column with the actual version
    version_queries: dict[str, str]


class _AuthorizingCollector:
    """SQLite authorizer callback plus the audit state for one statement.

    In observation mode (``enforce=False``) every callback returns
    ``SQLITE_OK`` so a single ``EXPLAIN`` surfaces every problem in the
    statement at once. In enforcement mode violations return ``SQLITE_DENY``;
    this is only used once a statement has already been validated.
    """

    def __init__(self, exporter: "ReadOnlyExporter", enforce: bool) -> None:
        self.exporter = exporter
        self.enforce = enforce
        self.read_events: list[_ReadEvent] = []
        self.functions: set[str] = set()
        self.pragmas: set[str] = set()
        self.violations: list[str] = []

    def _record_violation(self, message: str) -> int:
        # One statement can trigger the same authorizer violation several
        # times (a CREATE TABLE fires several events); keep each once.
        if message not in self.violations:
            self.violations.append(message)
        return self._verdict()

    def _verdict(self) -> int:
        # Observation mode keeps going so all violations are collected;
        # enforcement mode blocks the statement from running.
        return sqlite3.SQLITE_DENY if self.enforce else sqlite3.SQLITE_OK

    def __call__(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        dbname: str | None,
        source: str | None,
    ) -> int:
        if action in _WRITE_ACTIONS:
            return self._record_violation(
                self._describe_write(action, arg1, arg2, dbname)
            )
        if action == sqlite3.SQLITE_PRAGMA:
            return self._pragma(arg1, arg2)
        if action == sqlite3.SQLITE_FUNCTION:
            return self._function(arg2)
        if action == sqlite3.SQLITE_READ:
            return self._read(arg1, arg2, dbname, source)
        return sqlite3.SQLITE_OK

    @staticmethod
    def _action_name(action: int) -> str:
        return _ACTION_NAMES.get(action, f"action-{action}")

    def _describe_write(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        dbname: str | None,
    ) -> str:
        name = self._action_name(action)
        if action == sqlite3.SQLITE_TRANSACTION:
            return f"write operation rejected: transaction {arg1 or ''}".rstrip()
        if action == sqlite3.SQLITE_SAVEPOINT:
            detail = " ".join(part for part in (arg1, arg2) if part)
            return f"write operation rejected: savepoint {detail}".rstrip()
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return f"write operation rejected: {name} {arg1 or ''}".rstrip()
        target = ".".join(part for part in (dbname, arg1 or arg2) if part)
        return f"write operation rejected: {name} {target}".rstrip()

    def _pragma(self, name: str | None, value: str | None) -> int:
        pragma_name = name or ""
        self.pragmas.add(pragma_name)
        if value in (None, ""):
            return sqlite3.SQLITE_OK
        if pragma_name in _READ_PRAGMAS:
            # The argument names an object to introspect, not a value to set.
            return sqlite3.SQLITE_OK
        return self._record_violation(
            f"write operation rejected: PRAGMA {pragma_name} = {value}"
        )

    def _function(self, name: str | None) -> int:
        function_name = name or ""
        self.functions.add(function_name)
        if function_name not in self.exporter.policy.allowed_functions:
            return self._record_violation(
                "function not registered as an allowed pure function: "
                f"{function_name}()"
            )
        return sqlite3.SQLITE_OK

    def _read(
        self,
        table: str | None,
        column: str | None,
        dbname: str | None,
        source: str | None,
    ) -> int:
        table_name = table or ""
        # An empty column name means SQLite scanned the table without reading
        # a named column (e.g. SELECT 1 FROM t or SELECT count(*) FROM t).
        column_name = column or "*"
        database = dbname or "main"
        if table_name in _INTERNAL_TABLES or table_name.startswith("pragma_"):
            # pragma_* pseudo-tables back read-only introspection functions
            # such as pragma_table_info(); they expose no caller source data.
            return sqlite3.SQLITE_OK
        owner = self.exporter._shadow_virtual_table(database, table_name)
        if owner is not None:
            # Shadow tables are internal storage for a virtual table. Allow
            # them only when the owning virtual table itself is authorized;
            # they are implementation detail and excluded from the manifest.
            if (database, owner) in self.exporter.policy.allowed_columns:
                return sqlite3.SQLITE_OK
            return self._record_violation(
                "source table is not on the allowed list: "
                f"{database}.{table_name}"
            )
        self.read_events.append(
            _ReadEvent(database, table_name, column_name, source)
        )
        permitted = self.exporter.policy.allowed_columns.get(
            (database, table_name)
        )
        if permitted is None:
            return self._record_violation(
                f"source table is not on the allowed list: {database}.{table_name}"
            )
        if column_name == "*":
            # An existence/count scan reads no named column; authorizing the
            # table at all is sufficient. It is still recorded as "*".
            return sqlite3.SQLITE_OK
        # "*" in the permitted set whitelists every column of the table.
        if "*" in permitted or column_name in permitted:
            return sqlite3.SQLITE_OK
        return self._record_violation(
            "column is not on the allowed list: "
            f"{database}.{table_name}.{column_name}"
        )


class ReadOnlyExporter:
    """Run reports under a read-only authorizer and build an export manifest.

    :param db: :class:`~sqlite_utils.db.Database` to export from; attached
      databases added with :meth:`~sqlite_utils.db.Database.attach` are treated
      as declared sources
    :param allowed_columns: mapping of ``(schema, table)`` to the columns the
      caller permits reads of; a bare table name in the mapping is treated as
      belonging to ``main``
    :param allowed_functions: names of SQL functions the caller has registered
      as pure functions; every other function call is rejected
    :param expected_versions: mapping of schema name to the version that schema
      must certify
    :param version_queries: mapping of schema name to a SQL query returning one
      row with one column: the actual version of that schema
    :param query_only: also set ``PRAGMA query_only=ON`` as defence in depth
    """

    def __init__(
        self,
        db: Database,
        allowed_columns: dict[str | tuple[str, str], Sequence[str]]
        | None = None,
        allowed_functions: Sequence[str] | None = None,
        expected_versions: dict[str, str] | None = None,
        version_queries: dict[str, str] | None = None,
        query_only: bool = True,
    ) -> None:
        self.db = db
        normalized_columns: dict[tuple[str, str], frozenset[str]] = {}
        for key, columns in (allowed_columns or {}).items():
            if isinstance(key, tuple):
                schema, table = key
            else:
                schema, table = "main", key
            normalized_columns[(schema, table)] = frozenset(columns)
        self.policy = _Policy(
            allowed_columns=normalized_columns,
            allowed_functions=frozenset(allowed_functions or ()),
            expected_versions=dict(expected_versions or {}),
            version_queries=dict(version_queries or {}),
        )
        # Cache of schema name -> {object name: "table"/"view"}
        self._schema_objects: dict[str, dict[str, str]] = {}
        # Cache of schema name -> names of virtual tables (FTS5, ...), whose
        # <vtab>_data, <vtab>_idx, ... shadow tables are implementation detail.
        self._virtual_tables: dict[str, frozenset[str]] = {}
        self._databases: list[tuple[str, str | None]] | None = None
        self.query_only = query_only
        self._query_only_restored = False
        if query_only:
            db.conn.execute("PRAGMA query_only=ON")

    def __enter__(self) -> "ReadOnlyExporter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Detach the authorizer and restore the connection's query-only state."""
        self.db.conn.set_authorizer(None)
        if self.query_only and not self._query_only_restored:
            self.db.conn.execute("PRAGMA query_only=OFF")
            self._query_only_restored = True

    @contextlib.contextmanager
    def _authorizer(self, collector: _AuthorizingCollector) -> Generator[None, None, None]:
        self.db.conn.set_authorizer(collector)
        try:
            yield
        finally:
            self.db.conn.set_authorizer(None)

    def _database_list(self) -> list[tuple[str, str | None]]:
        # ATTACH is refused during an export, so the database list is stable
        # for the lifetime of the exporter and can be cached.
        if self._databases is None:
            rows = self.db.conn.execute("PRAGMA database_list").fetchall()
            self._databases = [(row[1], row[2] or None) for row in rows]
        return self._databases

    def _schema_object_kind(self, database: str, name: str) -> str | None:
        if database not in self._schema_objects:
            sql = (
                f"SELECT name, type, sql FROM "
                f"{quote_identifier(database)}.sqlite_master"
            )
            try:
                rows = self.db.conn.execute(sql).fetchall()
            except sqlite3.Error:
                rows = []
            objects: dict[str, str] = {}
            virtual: set[str] = set()
            for object_name, object_type, object_sql in rows:
                if object_type in ("table", "view"):
                    objects[object_name] = object_type
                if (
                    object_type == "table"
                    and isinstance(object_sql, str)
                    and object_sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE")
                ):
                    virtual.add(object_name)
            self._schema_objects[database] = objects
            self._virtual_tables[database] = frozenset(virtual)
        return self._schema_objects[database].get(name)

    def _shadow_virtual_table(
        self, database: str, table_name: str
    ) -> str | None:
        """Return the virtual table owning a shadow table, if any.

        FTS5 creates ``<vtab>_data``, ``<vtab>_idx``, ``<vtab>_docsize`` and
        ``<vtab>_config`` shadow tables that only appear as reads at execution
        time, not during the EXPLAIN preparation.
        """
        self._schema_object_kind(database, "")
        for suffix in ("_data", "_idx", "_docsize", "_config"):
            if table_name.endswith(suffix):
                candidate = table_name[: -len(suffix)]
                if candidate in self._virtual_tables.get(database, frozenset()):
                    return candidate
        return None

    def _find_object(self, name: str) -> tuple[str | None, str | None]:
        """Find which schema an unqualified object name belongs to."""
        for database, _ in self._database_list():
            kind = self._schema_object_kind(database, name)
            if kind is not None:
                return database, kind
        return None, None

    def _resolve_view(
        self, name: str, table_database: str
    ) -> tuple[str | None, str | None]:
        """Resolve an unqualified view name to its home schema.

        The authorizer's ``source`` argument carries no schema, so views with
        the same name in different attached databases are ambiguous. A view
        reading a table normally lives in the same schema as that table, so
        the table's schema is preferred before falling back to a global
        lookup (which still resolves cross-schema views with unique names).
        """
        if self._schema_object_kind(table_database, name) == "view":
            return table_database, "view"
        # Fall back to a global lookup so cross-schema views with unique
        # names (e.g. a main-schema view over an attached table) still resolve.
        return self._find_object(name)

    def _certify_versions(self) -> tuple[list[SourceVersion], list[SourceTable]]:
        sources: list[SourceVersion] = []
        unauthenticated: list[SourceTable] = []
        for schema, file_path in self._database_list():
            expected = self.policy.expected_versions.get(schema)
            actual: str | None = None
            if schema in self.policy.version_queries:
                try:
                    cursor = self.db.conn.execute(
                        self.policy.version_queries[schema]
                    )
                    row = cursor.fetchone()
                    actual = None if row is None else str(row[0])
                except sqlite3.Error as exc:
                    raise ReadOnlyExportError(
                        f"version query for source {schema!r} failed: {exc}"
                    ) from exc
            if expected is None:
                # main and temp belong to this connection; only attached
                # databases without a declared version cannot be certified
                # and must fail closed.
                status = (
                    "unspecified"
                    if schema in ("main", "temp")
                    else "unauthenticated"
                )
            elif actual is None:
                status = "unauthenticated"
            elif str(expected) == actual:
                status = "verified"
            else:
                status = "mismatch"
            sources.append(
                SourceVersion(
                    database=schema,
                    file=file_path,
                    expected=expected,
                    actual=actual,
                    status=status,
                )
            )
            if status in ("unauthenticated", "mismatch"):
                for name, kind in self._schema_objects_cached(schema).items():
                    unauthenticated.append(SourceTable(schema, name, kind))
        return sources, unauthenticated

    def _schema_objects_cached(self, database: str) -> dict[str, str]:
        self._schema_object_kind(database, "")
        return self._schema_objects[database]

    def export_batch(
        self,
        sql: str | Sequence[str],
        params: Sequence[Any] | dict[str, Any] | None = None,
    ) -> ExportManifest:
        """Validate and run a batch of report SQL statements.

        Every statement is prepared first through ``EXPLAIN`` with an
        observing authorizer, which fires for every nested view without
        executing the statement, so writes and unauthorized sources abort the
        batch before any report query runs for real. Only when the whole
        batch and every source version check out are the statements executed
        and their rows materialized.

        :param sql: one SQL string (which may contain several statements) or a
          sequence of report SQL strings
        :param params: parameters bound to every statement
        :raises ReadOnlyExportError: if any statement writes, uses an
          unauthorized table, column or function, or reads a source whose
          version could not be certified - in that case nothing is exported
        """
        if isinstance(sql, str):
            statements = [
                statement.strip()
                for statement in _iter_complete_sql_statements(sql)
                if statement.strip()
            ]
        else:
            statements = list(sql)
        if not statements:
            raise ReadOnlyExportError("no SQL statements were provided")

        sources, unauthenticated = self._certify_versions()
        reasons: list[str] = []
        bad_schemas = {
            source.database
            for source in sources
            if source.status in ("unauthenticated", "mismatch")
        }
        if bad_schemas:
            reasons.append(
                "sources could not be certified as the required version: "
                + ", ".join(sorted(bad_schemas))
            )

        # Phase 1: observe every statement through EXPLAIN. The authorizer
        # returns SQLITE_OK even for violations so one pass collects the
        # complete list of problems; EXPLAIN prepares but never executes.
        dry_run_collectors: list[_AuthorizingCollector] = []
        bound_params = {} if params is None else params
        for statement in statements:
            collector = _AuthorizingCollector(self, enforce=False)
            dry_run_collectors.append(collector)
            try:
                with self._authorizer(collector):
                    cursor = self.db.conn.execute(
                        f"EXPLAIN {statement}", bound_params
                    )
                    cursor.fetchall()
            except sqlite3.Error as exc:
                if not collector.violations:
                    collector.violations.append(f"SQL error: {exc}")
            reasons.extend(collector.violations)

        # Every table or view that belongs to a source which could not be
        # version-certified is marked, not just the ones the reports read:
        # none of them can be treated as belonging to the required version.
        unauthenticated_tables = sorted(set(unauthenticated))

        if reasons:
            raise ReadOnlyExportError(reasons, unauthenticated_tables)

        # Phase 2: every statement is proven read-only and authorized. Run
        # them for real under the enforcing authorizer and materialize the
        # rows of each statement so no partial output can be streamed out.
        report_results: list[ReportResult] = []
        for index, statement in enumerate(statements, start=1):
            collector = _AuthorizingCollector(self, enforce=True)
            with self._authorizer(collector):
                cursor = self.db.conn.execute(statement, bound_params)
                if cursor.description is None:
                    rows: list[dict[str, Any]] = []
                else:
                    keys = [description[0] for description in cursor.description]
                    rows = [dict(zip(keys, row)) for row in cursor.fetchall()]
            if collector.violations:
                # Defensive: phase 1 should have caught all of these.
                raise ReadOnlyExportError(collector.violations)
            report_results.append(
                self._build_report(index, statement, collector, rows)
            )

        return ExportManifest(
            reports=report_results,
            sources=sources,
            unauthenticated_tables=unauthenticated_tables,
        )

    def _build_report(
        self,
        index: int,
        sql: str,
        collector: _AuthorizingCollector,
        rows: list[dict[str, Any]],
    ) -> ReportResult:
        # (view database, view name) -> set of (database, object) it reads.
        # A view can live in a different schema from the tables it reads
        # (e.g. a main-schema view over an attached table), so view objects
        # are resolved to their home schema instead of assuming event.database.
        edges: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for event in collector.read_events:
            if event.source is None:
                continue
            view_database, view_kind = self._resolve_view(
                event.source, event.database
            )
            if view_kind == "view":
                edges[(view_database, event.source)].add(
                    (event.database, event.table)
                )

        # Views referenced directly by the statement are the chain roots.
        roots = sorted(
            {
                (event.database, event.table)
                for event in collector.read_events
                if event.source is None
                and self._schema_object_kind(
                    event.database, event.table
                )
                == "view"
            }
        )

        columns: set[SourceColumn] = set()
        tables: set[SourceTable] = set()
        for event in collector.read_events:
            kind = self._schema_object_kind(event.database, event.table)
            tables.add(
                SourceTable(event.database, event.table, kind or "table")
            )
            if kind == "view":
                # View reads are documented through the "via" chains on the
                # underlying table columns, not as columns of their own.
                continue
            via = self._view_chain(event, edges, roots)
            columns.add(
                SourceColumn(
                    event.database, event.table, event.column, tuple(via)
                )
            )

        return ReportResult(
            index=index,
            sql=sql,
            columns_read=sorted(columns),
            tables_read=sorted(tables),
            functions_used=sorted(
                FunctionUse(name) for name in collector.functions
            ),
            pragmas_used=sorted(collector.pragmas),
            rows=rows,
        )

    def _view_chain(
        self,
        event: _ReadEvent,
        edges: dict[tuple[str, str], set[tuple[str, str]]],
        roots: list[tuple[str, str]],
    ) -> list[str]:
        if event.source is None:
            return []
        view_database, view_kind = self._resolve_view(
            event.source, event.database
        )
        if view_kind != "view":
            # CTE names and table aliases also appear as source but are not
            # registered views, so they are not part of a view chain.
            return []
        target = (view_database, event.source)
        for root in roots:
            path = self._find_view_path(root, target, edges)
            if path is not None:
                return [
                    name
                    if database == event.database
                    else f"{database}.{name}"
                    for database, name in path
                ]
        # Fallback: the source view could not be linked to a root (it is
        # still a real view the read passed through).
        return [
            event.source
            if view_database == event.database
            else f"{view_database}.{event.source}"
        ]

    @staticmethod
    def _find_view_path(
        start: tuple[str, str],
        target: tuple[str, str],
        edges: dict[tuple[str, str], set[tuple[str, str]]],
    ) -> list[tuple[str, str]] | None:
        # Deterministic depth-first walk; sorted neighbours keep output stable
        # even when a view is reachable through more than one route.
        stack: list[tuple[tuple[str, str], list[tuple[str, str]]]] = [
            (start, [start])
        ]
        while stack:
            node, path = stack.pop()
            if node == target:
                return path
            for neighbour in sorted(edges.get(node, ()), reverse=True):
                if neighbour not in path:
                    stack.append((neighbour, path + [neighbour]))
        return None
