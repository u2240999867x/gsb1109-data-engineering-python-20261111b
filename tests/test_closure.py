"""Tests for foreign key closure extraction."""

import json

import pytest
from click.testing import CliRunner

from sqlite_utils import Database, cli
from sqlite_utils.db import (
    ClosureIncompleteError,
    ClosurePrimaryKeyError,
    NoTable,
)


@pytest.fixture
def shop_db():
    db = Database(memory=True)
    db.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id)
        );
        INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob');
        INSERT INTO products VALUES (10, 'Widget'), (20, 'Gadget');
        INSERT INTO orders VALUES
            (100, 1, 10),
            (101, NULL, 20),
            (102, 999, 10);
        """
    )
    return db


def test_closure_follows_single_column_foreign_keys(shop_db):
    result = shop_db.foreign_key_closure(table="orders", where="id = 100")
    assert result.complete is True
    assert result.rows_selected == 3
    assert result.tables["orders"].row_ids == (100,)
    assert result.tables["customers"].row_ids == (1,)
    assert result.tables["products"].row_ids == (10,)
    assert result.null_foreign_keys == []
    assert result.broken_foreign_keys == []
    assert result.boundaries == []


def test_closure_accepts_primary_key_seed_list(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [100, 101]})
    assert result.tables["orders"].row_ids == (100, 101)
    # Both products referenced across the two seed rows are pulled in once
    assert result.tables["products"].row_ids == (10, 20)
    assert result.tables["customers"].row_ids == (1,)


def test_closure_where_fragment_uses_parameters(shop_db):
    result = shop_db.foreign_key_closure(
        table="orders",
        where="id = :order_id",
        where_args={"order_id": 100},
    )
    assert result.tables["orders"].row_ids == (100,)


def test_null_foreign_key_is_reported_but_needs_no_parent(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [101]})
    assert len(result.null_foreign_keys) == 1
    item = result.null_foreign_keys[0]
    assert item.table == "orders"
    assert item.row_id == 101
    assert item.columns == ("customer_id",)
    assert item.key == (None,)
    assert item.foreign_key.other_table == "customers"
    # No customer row is fetched for the null reference
    assert "customers" not in result.tables
    # The non-null product reference still resolves
    assert result.tables["products"].row_ids == (20,)


def test_broken_foreign_key_is_reported_and_row_is_kept(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [102]})
    assert len(result.broken_foreign_keys) == 1
    item = result.broken_foreign_keys[0]
    assert item.table == "orders"
    assert item.row_id == 102
    assert item.columns == ("customer_id",)
    assert item.key == (999,)
    assert item.foreign_key.other_table == "customers"
    # The dangling customers row cannot be supplied
    assert "customers" not in result.tables
    # The order itself and its valid product dimension are still packaged
    assert result.tables["orders"].row_ids == (102,)
    assert result.tables["products"].row_ids == (10,)


def test_closure_terminates_on_cyclic_foreign_keys(fresh_db):
    fresh_db.executescript(
        """
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER REFERENCES nodes(id)
        );
        INSERT INTO nodes VALUES (1, NULL), (2, 1), (3, 2);
        """
    )
    result = fresh_db.foreign_key_closure(seeds={"nodes": [3]})
    assert result.complete is True
    assert result.tables["nodes"].row_ids == (1, 2, 3)


def test_compound_foreign_key_is_followed_as_a_unit(fresh_db):
    fresh_db.executescript(
        """
        CREATE TABLE departments (
            campus_name TEXT NOT NULL,
            dept_code TEXT NOT NULL,
            dept_name TEXT,
            PRIMARY KEY (campus_name, dept_code)
        );
        CREATE TABLE courses (
            course_code TEXT PRIMARY KEY,
            campus_name TEXT NOT NULL,
            dept_code TEXT NOT NULL,
            FOREIGN KEY (campus_name, dept_code)
                REFERENCES departments(campus_name, dept_code)
        );
        INSERT INTO departments VALUES ('main', 'CS', 'Computer Science');
        INSERT INTO courses VALUES ('CS101', 'main', 'CS');
        """
    )
    result = fresh_db.foreign_key_closure(seeds={"courses": ["CS101"]})
    assert result.tables["courses"].row_ids == ("CS101",)
    assert result.tables["departments"].row_ids == (("main", "CS"),)
    assert result.rows_for("departments") == [
        {"campus_name": "main", "dept_code": "CS", "dept_name": "Computer Science"}
    ]


def test_partially_null_compound_key_is_treated_as_null(fresh_db):
    fresh_db.executescript(
        """
        CREATE TABLE departments (
            campus_name TEXT NOT NULL,
            dept_code TEXT NOT NULL,
            PRIMARY KEY (campus_name, dept_code)
        );
        CREATE TABLE courses (
            course_code TEXT PRIMARY KEY,
            campus_name TEXT,
            dept_code TEXT,
            FOREIGN KEY (campus_name, dept_code)
                REFERENCES departments(campus_name, dept_code)
        );
        INSERT INTO courses VALUES ('CS101', NULL, 'CS');
        """
    )
    result = fresh_db.foreign_key_closure(seeds={"courses": ["CS101"]})
    assert "departments" not in result.tables
    assert len(result.null_foreign_keys) == 1
    assert result.null_foreign_keys[0].key == (None, "CS")


def test_table_without_primary_key_cannot_be_a_seed(fresh_db):
    fresh_db.execute("CREATE TABLE no_pk (name TEXT)")
    with pytest.raises(ClosurePrimaryKeyError, match="no declared primary key"):
        fresh_db.foreign_key_closure(seeds={"no_pk": ["anything"]})


def test_table_without_primary_key_cannot_be_referenced(fresh_db):
    fresh_db.executescript(
        """
        CREATE TABLE dimension (label TEXT);
        CREATE TABLE fact (
            id INTEGER PRIMARY KEY,
            dimension_label TEXT REFERENCES dimension(rowid)
        );
        """
    )
    fresh_db.execute("INSERT INTO fact (id, dimension_label) VALUES (1, 'x')")
    with pytest.raises(ClosurePrimaryKeyError, match="no declared primary key"):
        fresh_db.foreign_key_closure(seeds={"fact": [1]})


def test_missing_seed_table_raises(shop_db):
    with pytest.raises(NoTable, match="does not exist"):
        shop_db.foreign_key_closure(seeds={"nope": [1]})


def test_missing_referenced_table_raises(fresh_db):
    # A foreign key whose target table is absent from sqlite_master
    fresh_db.execute(
        "CREATE TABLE fact (id INTEGER PRIMARY KEY, ghost_id INTEGER)"
    )
    fresh_db.execute(
        "CREATE TABLE ghost (id INTEGER PRIMARY KEY)"
    )
    fresh_db.table("fact").add_foreign_key("ghost_id", "ghost", "id")
    fresh_db.execute("DROP TABLE ghost")
    with pytest.raises(NoTable, match="references missing table ghost"):
        fresh_db.foreign_key_closure(seeds={"fact": [1]})


def test_seed_key_arity_is_validated(fresh_db):
    fresh_db.execute(
        "CREATE TABLE departments (campus TEXT, code TEXT, PRIMARY KEY (campus, code))"
    )
    with pytest.raises(ValueError, match="needs 2 value"):
        fresh_db.foreign_key_closure(seeds={"departments": [("main",)]})


def test_requires_seeds(shop_db):
    with pytest.raises(ValueError, match="at least one seed"):
        shop_db.foreign_key_closure()


def test_table_requires_where(shop_db):
    with pytest.raises(ValueError, match="table= requires where="):
        shop_db.foreign_key_closure(table="orders")


def test_budget_must_be_positive(shop_db):
    with pytest.raises(ValueError, match="positive integer"):
        shop_db.foreign_key_closure(seeds={"orders": [100]}, budget=0)


def test_budget_truncation_reports_unexpanded_boundary(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [100]}, budget=2)
    assert result.complete is False
    assert result.rows_selected == 2
    assert result.boundaries
    boundary = result.boundaries[0]
    assert boundary.table == "orders"
    assert boundary.pending_rows == (100,)
    # The customers row (id 1) was discovered but could not be fetched
    assert boundary.pending_keys == (1,)


def test_budget_truncation_during_seed_phase(shop_db):
    result = shop_db.foreign_key_closure(
        seeds={"orders": [100, 101, 102]}, budget=2
    )
    assert result.complete is False
    assert result.tables["orders"].row_ids == (100, 101)
    seed_boundary = next(
        boundary for boundary in result.boundaries if not boundary.pending_rows
    )
    assert seed_boundary.table == "orders"
    assert seed_boundary.pending_keys == (102,)


def test_incomplete_closure_refuses_to_be_written(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [100]}, budget=2)
    with pytest.raises(ClosureIncompleteError):
        result.to_database(Database(memory=True))


def test_to_database_preserves_schema_foreign_keys_and_identity(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [100, 102]})
    dest = result.to_database(Database(memory=True))
    assert set(dest.table_names()) == {"customers", "products", "orders"}
    # Foreign key declarations survive in the new schema
    order_fks = {
        (fk.columns, fk.other_table, fk.other_columns)
        for fk in dest.table("orders").foreign_keys
    }
    assert (("customer_id",), "customers", ("id",)) in order_fks
    assert (("product_id",), "products", ("id",)) in order_fks
    # Primary key values are preserved rather than reassigned
    assert [row["id"] for row in dest.table("orders").rows] == [100, 102]
    assert dest.table("customers").get(1)["name"] == "Alice"


def test_extract_closure_writes_file(shop_db, tmp_path):
    output = tmp_path / "repro.db"
    result = shop_db.extract_closure(str(output), table="orders", where="id = 100")
    assert result.complete is True
    assert output.exists()
    dest = Database(output)
    assert set(dest.table_names()) == {"customers", "products", "orders"}
    assert list(dest.table("orders").rows) == [
        {"id": 100, "customer_id": 1, "product_id": 10}
    ]


def test_extract_closure_refuses_to_write_partial_package(shop_db, tmp_path):
    output = tmp_path / "partial.db"
    with pytest.raises(ClosureIncompleteError):
        shop_db.extract_closure(
            str(output), seeds={"orders": [100]}, budget=2
        )
    assert not output.exists()


def test_multiple_seed_tables_share_one_closure(shop_db):
    result = shop_db.foreign_key_closure(
        seeds={"orders": [100], "customers": [2]}
    )
    assert result.tables["orders"].row_ids == (100,)
    assert result.tables["customers"].row_ids == (1, 2)
    assert result.tables["products"].row_ids == (10,)


def test_row_ids_are_deterministic(shop_db):
    first = shop_db.foreign_key_closure(seeds={"orders": [102, 101, 100]})
    second = shop_db.foreign_key_closure(seeds={"orders": [100, 102, 101]})
    assert first.tables["orders"].row_ids == second.tables["orders"].row_ids
    assert first.tables["orders"].row_ids == (100, 101, 102)


def test_rows_for_unknown_table_returns_empty(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [101]})
    assert result.rows_for("customers") == []


def test_to_database_rejects_non_empty_destination(shop_db):
    result = shop_db.foreign_key_closure(seeds={"orders": [100]})
    dest = Database(memory=True)
    dest.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="already contains table"):
        result.to_database(dest)


# --- CLI -----------------------------------------------------------------


def test_cli_extract_closure_where(shop_db, tmp_path):
    source = tmp_path / "shop.db"
    shop_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    report = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "extract-closure",
            str(source),
            str(output),
            "orders",
            "--where",
            "id in (:o1, :o2)",
            "-p",
            "o1",
            "100",
            "-p",
            "o2",
            "101",
            "--report",
            str(report),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    dest = Database(output)
    assert set(dest.table_names()) == {"customers", "products", "orders"}
    assert [row["id"] for row in dest.table("orders").rows] == [100, 101]
    payload = json.loads(report.read_text())
    assert payload["complete"] is True
    assert payload["tables"]["orders"] == [100, 101]
    assert payload["null_foreign_keys"][0]["row_id"] == 101
    assert payload["broken_foreign_keys"] == []


def test_cli_extract_closure_pk_and_broken_reference(shop_db, tmp_path):
    source = tmp_path / "shop.db"
    shop_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    result = CliRunner().invoke(
        cli.cli,
        ["extract-closure", str(source), str(output), "orders", "--pk", "102"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "broken foreign key" in result.output
    assert "999" in result.output


def test_cli_extract_closure_budget_failure_writes_nothing(shop_db, tmp_path):
    source = tmp_path / "shop.db"
    shop_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    result = CliRunner().invoke(
        cli.cli,
        [
            "extract-closure",
            str(source),
            str(output),
            "orders",
            "--pk",
            "100",
            "--budget",
            "2",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "truncated" in result.output
    assert "pending parent keys [1]" in result.output
    assert not output.exists()


def test_cli_extract_closure_requires_replace(shop_db, tmp_path):
    source = tmp_path / "shop.db"
    shop_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    output.write_text("")
    result = CliRunner().invoke(
        cli.cli,
        ["extract-closure", str(source), str(output), "orders", "--pk", "100"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "--replace" in result.output


def test_cli_extract_closure_selector_validation(shop_db, tmp_path):
    source = tmp_path / "shop.db"
    shop_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    runner = CliRunner()
    for args in (
        [],
        ["--where", "id = 1", "--pk", "1"],
    ):
        result = runner.invoke(
            cli.cli,
            ["extract-closure", str(source), str(output), "orders", *args],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "exactly one of --where or --pk" in result.output


def test_cli_extract_closure_without_primary_key(fresh_db, tmp_path):
    fresh_db.execute("CREATE TABLE no_pk (name TEXT)")
    source = tmp_path / "source.db"
    fresh_db.conn.backup(Database(source).conn)
    output = tmp_path / "repro.db"
    result = CliRunner().invoke(
        cli.cli,
        [
            "extract-closure",
            str(source),
            str(output),
            "no_pk",
            "--where",
            "name = 'x'",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "no declared primary key" in result.output
    assert not output.exists()
