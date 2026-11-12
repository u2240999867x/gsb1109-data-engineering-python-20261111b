import json

import pytest
from click.testing import CliRunner

from sqlite_utils import Database, cli
from sqlite_utils.readonly_export import (
    ExportManifest,
    ReadOnlyExportError,
    ReadOnlyExporter,
    SourceColumn,
    SourceTable,
)


@pytest.fixture
def warehouse(tmpdir):
    """Main database with tables, a CTE-friendly schema and nested views."""
    db = Database(memory=True)
    _build_warehouse(db)
    return db


def _build_warehouse(db):
    db.table("customers").insert_all(
        [
            {"id": 1, "name": "Cleo", "tier": "gold", "ssn": "secret-1"},
            {"id": 2, "name": "Pancakes", "tier": "silver", "ssn": "secret-2"},
        ]
    )
    db.table("orders").insert_all(
        [
            {"id": 10, "customer_id": 1, "amount": 50, "note": "a"},
            {"id": 11, "customer_id": 1, "amount": 70, "note": "b"},
        ]
    )
    db.execute("CREATE VIEW v_customers AS SELECT id, name, tier FROM customers")
    db.execute(
        "CREATE VIEW v_orders AS "
        "SELECT o.id AS order_id, c.name AS customer, o.amount "
        "FROM orders o JOIN v_customers c ON o.customer_id = c.id"
    )
    return db


@pytest.fixture
def warehouse_path(tmpdir):
    """File-backed version of the warehouse schema for CLI tests."""
    path = str(tmpdir / "warehouse.db")
    _build_warehouse(Database(path))
    return path


@pytest.fixture
def attached_sources(tmpdir):
    """Two attached databases that both contain a table called ``shared``."""
    first_path = str(tmpdir / "first.db")
    second_path = str(tmpdir / "second.db")
    first = Database(first_path)
    first.table("shared").insert_all(
        [{"id": 1, "label": "first", "hidden": "h1"}]
    )
    first.execute(
        "CREATE VIEW shared_view AS SELECT id, label FROM shared"
    )
    first.table("meta").insert_all([{"key": "version", "value": "v-2026"}])
    first.conn.commit()
    first.close()
    second = Database(second_path)
    second.table("shared").insert_all(
        [{"id": 1, "label": "second", "hidden": "h2"}]
    )
    second.table("meta").insert_all([{"key": "version", "value": "v-2026"}])
    second.conn.commit()
    second.close()
    return first_path, second_path


def attach_both(db, attached_sources):
    db.attach("first", attached_sources[0])
    db.attach("second", attached_sources[1])


VERSION_QUERIES = {
    "first": "SELECT value FROM first.meta WHERE key = 'version'",
    "second": "SELECT value FROM second.meta WHERE key = 'version'",
}


def test_records_columns_actually_read(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={"customers": ["id", "name", "tier", "ssn"]},
    )
    manifest = exporter.export_batch("SELECT id, name FROM customers")
    report = manifest.reports[0]
    assert report.rows == [
        {"id": 1, "name": "Cleo"},
        {"id": 2, "name": "Pancakes"},
    ]
    assert report.columns_read == [
        SourceColumn("main", "customers", "id"),
        SourceColumn("main", "customers", "name"),
    ]
    # ssn was permitted but never read, so it must not appear in the manifest
    assert {column.column for column in report.columns_read} == {"id", "name"}
    assert report.tables_read == [SourceTable("main", "customers", "table")]


def test_star_expansion_records_all_read_columns(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["*"]}
    )
    manifest = exporter.export_batch("SELECT * FROM customers")
    assert {column.column for column in manifest.reports[0].columns_read} == {
        "id",
        "name",
        "tier",
        "ssn",
    }


def test_count_query_records_wildcard_column(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={"orders": ["id", "amount"]},
        allowed_functions=["count"],
    )
    report = exporter.export_batch(
        "SELECT count(*) AS n FROM orders"
    ).reports[0]
    assert report.rows == [{"n": 2}]
    assert report.columns_read == [SourceColumn("main", "orders", "*")]
    assert [function.name for function in report.functions_used] == ["count"]


def test_cte_and_aliases_resolve_to_real_tables(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            "customers": ["id", "name"],
            "orders": ["id", "customer_id", "amount"],
        },
    )
    sql = (
        "WITH chosen AS (SELECT id, name FROM customers) "
        "SELECT c.name, o.amount FROM chosen c "
        "JOIN orders o ON o.customer_id = c.id"
    )
    report = exporter.export_batch(sql).reports[0]
    tables = {(table.database, table.name) for table in report.tables_read}
    assert tables == {
        ("main", "customers"),
        ("main", "orders"),
    }
    # The CTE name must not show up as a table or a view chain
    assert all(column.via == () for column in report.columns_read)
    assert all("chosen" not in column.via for column in report.columns_read)


def test_nested_views_are_resolved_two_levels_deep(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            "customers": ["id", "name", "tier"],
            "orders": ["id", "customer_id", "amount"],
            "v_customers": ["id", "name", "tier"],
            "v_orders": ["order_id", "customer", "amount"],
        },
    )
    report = exporter.export_batch("SELECT * FROM v_orders").reports[0]
    by_column = {
        (column.table, column.column): column.via
        for column in report.columns_read
    }
    # v_orders -> v_customers -> customers (two view layers)
    assert by_column[("customers", "id")] == ("v_orders", "v_customers")
    assert by_column[("customers", "name")] == ("v_orders", "v_customers")
    # v_orders -> orders (one view layer)
    assert by_column[("orders", "amount")] == ("v_orders",)
    assert by_column[("orders", "customer_id")] == ("v_orders",)
    # Only base tables carry column entries; views are listed as tables
    assert {table.name for table in report.tables_read} == {
        "customers",
        "orders",
        "v_customers",
        "v_orders",
    }
    view_kinds = {table.name: table.kind for table in report.tables_read}
    assert view_kinds["v_orders"] == "view"
    assert view_kinds["orders"] == "table"


def test_attached_same_name_tables_are_distinguished(warehouse, attached_sources):
    attach_both(warehouse, attached_sources)
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("second", "shared"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
        },
        expected_versions={"first": "v-2026", "second": "v-2026"},
        version_queries=VERSION_QUERIES,
    )
    sql = (
        "SELECT f.label AS first_label, s.label AS second_label "
        "FROM first.shared f JOIN second.shared s ON f.id = s.id"
    )
    report = exporter.export_batch(sql).reports[0]
    assert report.rows == [{"first_label": "first", "second_label": "second"}]
    keys = {(column.database, column.table) for column in report.columns_read}
    assert keys == {("first", "shared"), ("second", "shared")}


def test_attached_view_is_resolved(warehouse, attached_sources):
    attach_both(warehouse, attached_sources)
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("first", "shared_view"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
            ("second", "shared"): ["id", "label"],
        },
        expected_versions={"first": "v-2026", "second": "v-2026"},
        version_queries=VERSION_QUERIES,
    )
    report = exporter.export_batch(
        "SELECT label FROM first.shared_view"
    ).reports[0]
    # SQLite reads every column defined by the view, even those the outer
    # statement does not project, so both id and label are accounted for.
    assert sorted(
        (column.table, column.column, column.via)
        for column in report.columns_read
    ) == [
        ("shared", "id", ("shared_view",)),
        ("shared", "label", ("shared_view",)),
    ]


def test_parameters_are_bound(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    report = exporter.export_batch(
        "SELECT name FROM customers WHERE id = :id", {"id": 2}
    ).reports[0]
    assert report.rows == [{"name": "Pancakes"}]


def test_batch_runs_every_statement_and_keeps_order(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    manifest = exporter.export_batch(
        [
            "SELECT id FROM customers WHERE id = 1",
            "SELECT name FROM customers WHERE id = 2",
        ]
    )
    assert [report.index for report in manifest.reports] == [1, 2]
    assert manifest.reports[0].rows == [{"id": 1}]
    assert manifest.reports[1].rows == [{"name": "Pancakes"}]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers (id) VALUES (9)",
        "UPDATE customers SET name = 'x'",
        "DELETE FROM customers",
        "CREATE TABLE new_table (id INTEGER)",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN extra TEXT",
        "ATTACH DATABASE ':memory:' AS other",
        "BEGIN",
        "SAVEPOINT sp1",
        "PRAGMA user_version = 5",
    ],
)
def test_write_statements_are_refused(warehouse, sql):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    before = list(warehouse.query("SELECT * FROM customers"))
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch(sql)
    assert exc_info.value.reasons
    after = list(warehouse.query("SELECT * FROM customers"))
    assert before == after


def test_unauthorized_table_is_refused(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id"]}
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT id FROM orders")
    assert "orders" in exc_info.value.reasons[0]


def test_unauthorized_column_is_refused_even_through_a_view(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            "customers": ["id", "name"],
            "orders": ["id", "customer_id", "amount"],
            "v_customers": ["id", "name"],
            "v_orders": ["order_id", "customer", "amount"],
        },
    )
    # tier is reachable through v_customers but deliberately not permitted
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT tier FROM v_customers")
    assert any("customers.tier" in reason for reason in exc_info.value.reasons)


def test_unregistered_function_is_refused(warehouse):
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={"customers": ["id", "name"]},
        allowed_functions=[],
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT lower(name) FROM customers")
    assert "lower()" in exc_info.value.reasons[0]


def test_registered_pure_function_is_allowed(warehouse):
    warehouse.register_function(
        lambda value: str(value).upper(), name="shout", deterministic=True
    )
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={"customers": ["id", "name"]},
        allowed_functions=["shout"],
    )
    report = exporter.export_batch(
        "SELECT shout(name) AS loud FROM customers WHERE id = 1"
    ).reports[0]
    assert report.rows == [{"loud": "CLEO"}]


def test_one_bad_statement_aborts_the_whole_batch(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch(
            [
                "SELECT name FROM customers",
                "DELETE FROM customers",
                "SELECT id FROM customers",
            ]
        )
    assert any("delete" in reason for reason in exc_info.value.reasons)
    # Nothing ran for real, so rows are untouched
    assert warehouse.table("customers").count == 2


def test_attached_source_without_version_is_unauthenticated(
    warehouse, attached_sources
):
    attach_both(warehouse, attached_sources)
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("second", "shared"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
        },
        expected_versions={"first": "v-2026"},
        version_queries={
            "first": VERSION_QUERIES["first"],
        },
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT label FROM second.shared")
    assert "second" in exc_info.value.reasons[0]
    unauthenticated = {
        (table.database, table.name)
        for table in exc_info.value.unauthenticated_tables
    }
    assert ("second", "shared") in unauthenticated


def test_version_mismatch_marks_every_table_in_the_source(
    warehouse, attached_sources
):
    attach_both(warehouse, attached_sources)
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("second", "shared"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
        },
        expected_versions={"first": "v-2026", "second": "different"},
        version_queries=VERSION_QUERIES,
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT label FROM first.shared")
    source_statuses = _source_statuses(exporter)
    assert source_statuses["first"] == "verified"
    assert source_statuses["second"] == "mismatch"
    marked = {
        table.name for table in exc_info.value.unauthenticated_tables
    }
    assert "shared" in marked and "meta" in marked


def test_verified_manifest_lists_sources(warehouse, attached_sources):
    attach_both(warehouse, attached_sources)
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("second", "shared"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
        },
        expected_versions={"first": "v-2026", "second": "v-2026"},
        version_queries=VERSION_QUERIES,
    )
    manifest = exporter.export_batch("SELECT id FROM first.shared")
    assert isinstance(manifest, ExportManifest)
    by_database = {source.database: source for source in manifest.sources}
    assert by_database["first"].status == "verified"
    assert by_database["second"].status == "verified"
    assert by_database["main"].status == "unspecified"
    assert manifest.unauthenticated_tables == []


def _source_statuses(exporter):
    sources, _ = exporter._certify_versions()
    return {source.database: source.status for source in sources}


def test_context_manager_restores_query_only(warehouse):
    warehouse.conn.execute("PRAGMA query_only=OFF")
    with ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id"]}
    ) as exporter:
        assert warehouse.conn.execute("PRAGMA query_only").fetchone()[0] == 1
        exporter.export_batch("SELECT id FROM customers")
    assert warehouse.conn.execute("PRAGMA query_only").fetchone()[0] == 0


def test_sql_error_is_reported_not_raised(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id"]}
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        exporter.export_batch("SELECT id FROM does_not_exist")
    assert "SQL error" in exc_info.value.reasons[0]


def test_empty_batch_is_rejected(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id"]}
    )
    with pytest.raises(ReadOnlyExportError):
        exporter.export_batch("")


def test_manifest_serializes_to_json(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    manifest = exporter.export_batch("SELECT id FROM customers")
    decoded = json.loads(manifest.to_json())
    assert decoded["reports"][0]["columns_read"][0] == {
        "database": "main",
        "table": "customers",
        "column": "id",
        "via": [],
    }
    assert "sources" in decoded and "unauthenticated_tables" in decoded


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def _policy_path(tmpdir, payload, name="policy.json"):
    path = str(tmpdir / name)
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


def test_cli_happy_path_outputs_envelope_and_manifest(
    tmpdir, warehouse_path, attached_sources
):
    db_path = warehouse_path
    policy = _policy_path(
        tmpdir,
        {
            "allowed_columns": {
                "customers": ["id", "name", "tier"],
                "orders": ["id", "customer_id", "amount"],
                "v_customers": ["id", "name", "tier"],
                "v_orders": ["order_id", "customer", "amount"],
                "first.shared": ["id", "label"],
                "second.shared": ["id", "label"],
                "first.meta": ["key", "value"],
                "second.meta": ["key", "value"],
            },
            "allowed_functions": ["count"],
            "versions": {
                "first": {
                    "expected": "v-2026",
                    "query": VERSION_QUERIES["first"],
                },
                "second": {
                    "expected": "v-2026",
                    "query": VERSION_QUERIES["second"],
                },
            },
        },
    )
    output_path = str(tmpdir / "out.json")
    manifest_path = str(tmpdir / "manifest.json")
    result = CliRunner().invoke(
        cli.cli,
        [
            "export-readonly",
            db_path,
            "--attach",
            "first",
            attached_sources[0],
            "--attach",
            "second",
            attached_sources[1],
            "--policy",
            policy,
            "--manifest",
            manifest_path,
            "-o",
            output_path,
            "SELECT * FROM v_orders",
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.load(open(output_path))
    assert envelope["reports"][0]["rows"]
    assert envelope["unauthenticated_tables"] == []
    manifest = json.load(open(manifest_path))
    assert "rows" not in json.dumps(manifest)
    tables = {
        (entry["database"], entry["name"])
        for entry in manifest["reports"][0]["tables_read"]
    }
    assert ("main", "v_orders") in tables


def test_cli_writes_nothing_on_refusal(tmpdir, warehouse_path):
    db_path = warehouse_path
    policy = _policy_path(
        tmpdir,
        {
            "allowed_columns": {"customers": ["id", "name"]},
        },
    )
    output_path = str(tmpdir / "out.json")
    manifest_path = str(tmpdir / "manifest.json")
    result = CliRunner().invoke(
        cli.cli,
        [
            "export-readonly",
            db_path,
            "--policy",
            policy,
            "-o",
            output_path,
            "--manifest",
            manifest_path,
            "DELETE FROM customers",
        ],
    )
    assert result.exit_code == 1
    assert "Read-only export refused" in result.output
    assert "delete" in result.output
    import os

    assert not os.path.exists(output_path)
    assert not os.path.exists(manifest_path)


def test_cli_reads_sql_from_file(tmpdir, warehouse_path):
    db_path = warehouse_path
    sql_path = str(tmpdir / "reports.sql")
    with open(sql_path, "w") as handle:
        handle.write("SELECT id FROM customers;\nSELECT name FROM customers;")
    policy = _policy_path(
        tmpdir, {"allowed_columns": {"customers": ["id", "name"]}}
    )
    result = CliRunner().invoke(
        cli.cli,
        [
            "export-readonly",
            db_path,
            "--policy",
            policy,
            "--sql-file",
            sql_path,
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert len(envelope["reports"]) == 2


def test_cli_inline_policy_options(tmpdir, warehouse_path):
    db_path = warehouse_path
    result = CliRunner().invoke(
        cli.cli,
        [
            "export-readonly",
            db_path,
            "--allow-column",
            "main",
            "customers",
            "id",
            "--allow-function",
            "count",
            "SELECT count(*) FROM customers",
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["reports"][0]["rows"] == [{"count(*)": 2}]


def test_cli_rejects_bad_policy_file(tmpdir, warehouse_path):
    db_path = warehouse_path
    bad_policy = str(tmpdir / "bad.json")
    with open(bad_policy, "w") as handle:
        handle.write('{"allowed_columns": {"customers": "not-a-list"}}')
    result = CliRunner().invoke(
        cli.cli,
        [
            "export-readonly",
            db_path,
            "--policy",
            bad_policy,
            "SELECT id FROM customers",
        ],
    )
    assert result.exit_code == 1
    assert "allowed_columns.customers" in result.output


def test_cli_requires_sql(tmpdir, warehouse_path):
    db_path = warehouse_path
    result = CliRunner().invoke(
        cli.cli, ["export-readonly", db_path]
    )
    assert result.exit_code == 1
    assert "SQL" in result.output


def test_temp_schema_is_trusted(warehouse):
    warehouse.conn.execute("CREATE TEMP TABLE scratch AS SELECT 1 AS x")
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={("temp", "scratch"): ["x"]}
    )
    manifest = exporter.export_batch("SELECT x FROM temp.scratch")
    assert manifest.reports[0].rows == [{"x": 1}]
    statuses = {
        source.database: source.status for source in manifest.sources
    }
    assert statuses["temp"] == "unspecified"


def test_pragma_read_is_allowed_and_recorded(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id"]}
    )
    report = exporter.export_batch("PRAGMA table_info(customers)").reports[0]
    assert report.pragmas_used == ["table_info"]


def test_pragma_table_valued_function_is_allowed(warehouse):
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"customers": ["id", "name"]}
    )
    report = exporter.export_batch(
        "SELECT name FROM pragma_table_info('customers')"
    ).reports[0]
    assert report.pragmas_used == ["table_info"]
    # The pragma pseudo-table is not caller-owned source data
    assert report.columns_read == []


def test_query_without_source_tables(warehouse):
    exporter = ReadOnlyExporter(warehouse, allowed_columns={})
    report = exporter.export_batch("SELECT 1 + 1 AS total").reports[0]
    assert report.rows == [{"total": 2}]
    assert report.columns_read == []
    assert report.tables_read == []


def test_same_named_views_in_attached_schemas_are_not_mixed(
    warehouse, attached_sources
):
    # Both attached databases define a view called shared_view
    attach_both(warehouse, attached_sources)
    second = Database(attached_sources[1])
    second.execute("CREATE VIEW shared_view AS SELECT id, label FROM shared")
    second.conn.commit()
    second.close()
    exporter = ReadOnlyExporter(
        warehouse,
        allowed_columns={
            ("first", "shared"): ["id", "label"],
            ("second", "shared"): ["id", "label"],
            ("first", "shared_view"): ["id", "label"],
            ("second", "shared_view"): ["id", "label"],
            ("first", "meta"): ["key", "value"],
            ("second", "meta"): ["key", "value"],
        },
        expected_versions={"first": "v-2026", "second": "v-2026"},
        version_queries=VERSION_QUERIES,
    )
    report = exporter.export_batch(
        "SELECT (SELECT label FROM first.shared_view) AS f, "
        "(SELECT label FROM second.shared_view) AS s"
    ).reports[0]
    by_database = {}
    for column in report.columns_read:
        by_database.setdefault(column.database, set()).add(column.table)
    assert by_database["first"] == {"shared"}
    assert by_database["second"] == {"shared"}
    # No read may attribute a second-schema table to the first-schema view
    assert all(
        column.via != ("first.shared_view",)
        for column in report.columns_read
        if column.database == "second"
    )


def test_recursive_cte_resolves_to_base_table(warehouse):
    warehouse.execute("CREATE TABLE node(id INTEGER, parent INTEGER)")
    warehouse.table("node").insert_all(
        [
            {"id": 1, "parent": None},
            {"id": 2, "parent": 1},
            {"id": 3, "parent": 2},
        ]
    )
    exporter = ReadOnlyExporter(
        warehouse, allowed_columns={"node": ["id", "parent"]}
    )
    sql = (
        "WITH RECURSIVE r(id, parent, depth) AS ("
        "SELECT id, parent, 0 FROM node WHERE parent IS NULL "
        "UNION ALL "
        "SELECT n.id, n.parent, r.depth + 1 FROM node n "
        "JOIN r ON n.parent = r.id) SELECT * FROM r"
    )
    report = exporter.export_batch(sql).reports[0]
    assert [row["id"] for row in report.rows] == [1, 2, 3]
    assert {table.name for table in report.tables_read} == {"node"}


def test_fts_virtual_table_requires_authorization_and_hides_shadow_tables():
    db = Database(memory=True)
    db.table("dogs").insert_all(
        [{"id": 1, "name": "Cleo"}, {"id": 2, "name": "Pancakes"}], pk="id"
    )
    db.table("dogs").enable_fts(["name"])
    sql = "SELECT rowid FROM dogs_fts WHERE dogs_fts MATCH 'Cleo'"

    allowed = ReadOnlyExporter(
        db,
        allowed_columns={
            "dogs": ["id", "name"],
            "dogs_fts": ["*"],
        },
        allowed_functions=["match"],
    )
    report = allowed.export_batch(sql).reports[0]
    assert report.rows == [{"rowid": 1}]
    table_names = {table.name for table in report.tables_read}
    assert "dogs_fts" in table_names
    # Internal FTS5 shadow tables must not pollute the source manifest
    assert not {name for name in table_names if name.startswith("dogs_fts_")}
    assert all(
        not column.table.startswith("dogs_fts_")
        for column in report.columns_read
    )

    # The virtual table must itself be authorized
    missing_table = ReadOnlyExporter(
        db, allowed_columns={"dogs": ["id", "name"]},
        allowed_functions=["match"],
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        missing_table.export_batch(sql)
    assert any("dogs_fts" in reason for reason in exc_info.value.reasons)

    # The match() function must itself be registered as allowed
    missing_function = ReadOnlyExporter(
        db,
        allowed_columns={"dogs": ["id", "name"], "dogs_fts": ["*"]},
    )
    with pytest.raises(ReadOnlyExportError) as exc_info:
        missing_function.export_batch(sql)
    assert any("match()" in reason for reason in exc_info.value.reasons)
