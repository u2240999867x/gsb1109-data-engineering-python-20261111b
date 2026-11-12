import sqlite3

import pytest
from click.testing import CliRunner

from sqlite_utils import Database, cli
from sqlite_utils.db import (
    DedupeError,
    DedupePlan,
    InvalidColumns,
)
from sqlite_utils.utils import sqlite3 as sqlite3_module


def _companies_with_contacts(fresh_db, *, unique_fk=False, partial_unique=False):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE companies(
            id INTEGER PRIMARY KEY, name TEXT, city TEXT
        );
        """
    )
    if partial_unique:
        fresh_db.executescript(
            """
            CREATE TABLE contacts(
                id INTEGER PRIMARY KEY,
                active INTEGER,
                company_id INTEGER REFERENCES companies(id)
            );
            CREATE UNIQUE INDEX contacts_company_id_active
                ON contacts(company_id) WHERE active = 1;
            """
        )
    else:
        constraint = "UNIQUE" if unique_fk else ""
        fresh_db.execute(
            f"""
            CREATE TABLE contacts(
                id INTEGER PRIMARY KEY,
                name TEXT,
                company_id INTEGER {constraint} REFERENCES companies(id)
            );
            """
        )
    fresh_db.conn.executemany(
        "INSERT INTO companies(id, name, city) VALUES (?, ?, ?)",
        [
            (1, "Acme", "London"),
            (2, "Acme", "London"),
            (3, "Globex", "Berlin"),
        ],
    )
    return fresh_db


def test_dedupe_requires_explicit_winner(fresh_db):
    _companies_with_contacts(fresh_db)
    with pytest.raises(DedupeError, match="no winner"):
        fresh_db.table("companies").dedupe_plan(columns=("name", "city"))


def test_dedupe_plan_previews_rows_to_delete(fresh_db):
    _companies_with_contacts(fresh_db)
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme", "London"): 1}, columns=("name", "city")
    )
    assert isinstance(plan, DedupePlan)
    assert plan.can_apply
    assert plan.delete_count == 1
    assert plan.groups[0].winner == (1,)
    assert plan.groups[0].losers == ((2,),)
    assert plan.rows_to_delete == (
        {"id": 2, "name": "Acme", "city": "London"},
    )
    # Building a plan changes nothing
    assert fresh_db.table("companies").count == 3


def test_dedupe_plan_does_not_pick_winner_for_solo_group(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y")]
    )
    with pytest.raises(DedupeError, match="no duplicate rows"):
        fresh_db.table("t").dedupe_plan({("x",): 1}, columns=("name",))


def test_dedupe_winner_must_belong_to_group(fresh_db):
    _companies_with_contacts(fresh_db)
    with pytest.raises(DedupeError, match="is not one of the rows"):
        fresh_db.table("companies").dedupe_plan(
            {("Acme", "London"): 3}, columns=("name", "city")
        )


def test_dedupe_unknown_winner_key(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "x"), (3, "y")]
    )
    # The real duplicate group is covered, plus a key that matches no row
    with pytest.raises(DedupeError, match="do not match any rows"):
        fresh_db.table("t").dedupe_plan(
            {("x",): 1, ("zzz",): 1}, columns=("name",)
        )


def test_dedupe_invalid_column(fresh_db):
    _companies_with_contacts(fresh_db)
    with pytest.raises(InvalidColumns):
        fresh_db.table("companies").dedupe_plan(
            {("Acme",): 1}, columns=("nope",)
        )


def test_dedupe_apply_deletes_losers_and_repoints_foreign_keys(fresh_db):
    _companies_with_contacts(fresh_db)
    fresh_db.conn.executemany(
        "INSERT INTO contacts(id, name, company_id) VALUES (?, ?, ?)",
        [(10, "a", 1), (11, "b", 2), (12, "c", 2)],
    )
    table = fresh_db.table("companies")
    plan = table.dedupe_plan(
        {("Acme", "London"): 1}, columns=("name", "city")
    )
    assert plan.repoint_count == 1
    repoint = plan.repoints[0]
    assert repoint.table == "contacts"
    assert repoint.columns == ("company_id",)
    assert repoint.old_value == (2,)
    assert repoint.new_value == (1,)
    assert repoint.affected_rows == ((11,), (12,))

    table.dedupe_apply(plan)
    assert [row["id"] for row in table.rows] == [1, 3]
    assert [row["company_id"] for row in fresh_db.table("contacts").rows] == [
        1,
        1,
        1,
    ]
    assert fresh_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_dedupe_compound_dedupe_columns_compare_every_column(fresh_db):
    # Rows equal on the first column but differing on the second must not merge
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT, b TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(1, "x", "1"), (2, "x", "2"), (3, "x", "1")],
    )
    table = fresh_db.table("t")
    groups = table.duplicate_groups(columns=("a", "b"))
    assert groups == [(("x", "1"), ((1,), (3,)))]
    plan = table.dedupe_plan({("x", "1"): 1}, columns=("a", "b"))
    assert plan.groups[0].losers == ((3,),)
    table.dedupe_apply(plan)
    assert [row["id"] for row in table.rows] == [1, 2]


def test_dedupe_default_columns_are_non_primary_key_columns(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, tag TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(1, "x", "a"), (2, "x", "a"), (3, "y", "b")],
    )
    plan = fresh_db.table("t").dedupe_plan({("x", "a"): 1})
    assert plan.columns == ("name", "tag")
    fresh_db.table("t").dedupe_apply(plan)
    assert [row["id"] for row in fresh_db.table("t").rows] == [1, 3]


def test_dedupe_rowid_table(fresh_db):
    fresh_db.execute("CREATE TABLE t(name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t(rowid, name) VALUES (?, ?)",
        [(1, "x"), (2, "x"), (3, "y")],
    )
    table = fresh_db.table("t")
    assert table.pks == ["rowid"]
    plan = table.dedupe_plan({("x",): 1}, columns=("name",))
    table.dedupe_apply(plan)
    assert [row["name"] for row in table.rows] == ["x", "y"]


def test_dedupe_compound_primary_key_and_compound_foreign_key(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE node(
            a TEXT, b TEXT, label TEXT, PRIMARY KEY(a, b)
        );
        CREATE TABLE edge(
            id INTEGER PRIMARY KEY,
            from_a TEXT, from_b TEXT,
            to_a TEXT, to_b TEXT,
            FOREIGN KEY(from_a, from_b) REFERENCES node(a, b),
            FOREIGN KEY(to_a, to_b) REFERENCES node(a, b)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO node(a, b, label) VALUES (?, ?, ?)",
        [("x", "1", "dup"), ("x", "2", "dup"), ("y", "3", "solo")],
    )
    fresh_db.conn.executemany(
        "INSERT INTO edge VALUES (?, ?, ?, ?, ?)",
        [(1, "x", "1", "y", "3"), (2, "y", "3", "x", "2")],
    )
    table = fresh_db.table("node")
    plan = table.dedupe_plan({("dup",): ("x", "1")})
    repoint = plan.repoints[0]
    assert repoint.is_compound
    assert repoint.columns == ("to_a", "to_b")
    assert repoint.old_value == ("x", "2")
    assert repoint.new_value == ("x", "1")
    table.dedupe_apply(plan)
    assert list(table.rows) == [
        {"a": "x", "b": "1", "label": "dup"},
        {"a": "y", "b": "3", "label": "solo"},
    ]
    edges = list(fresh_db.table("edge").rows)
    assert edges[1]["to_a"] == "x"
    assert edges[1]["to_b"] == "1"
    assert fresh_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_dedupe_handles_self_referential_foreign_key(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.execute(
        """
        CREATE TABLE employees(
            id INTEGER PRIMARY KEY,
            name TEXT,
            manager_id INTEGER REFERENCES employees(id)
        )
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?)",
        [(1, "Kim", None), (2, "Kim", 1), (3, "Pat", 2)],
    )
    table = fresh_db.table("employees")
    plan = table.dedupe_plan({("Kim",): 1}, columns=("name",))
    assert plan.repoints[0].table == "employees"
    table.dedupe_apply(plan)
    assert list(table.rows) == [
        {"id": 1, "name": "Kim", "manager_id": None},
        {"id": 3, "name": "Pat", "manager_id": 1},
    ]
    assert fresh_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_dedupe_detects_unique_conflict_on_foreign_key_column(fresh_db):
    _companies_with_contacts(fresh_db, unique_fk=True)
    # company_id is UNIQUE, so every contact points at a different company
    fresh_db.conn.execute(
        "INSERT INTO companies(id, name, city) VALUES (4, 'Acme', 'London')"
    )
    fresh_db.conn.executemany(
        "INSERT INTO contacts(id, name, company_id) VALUES (?, ?, ?)",
        [(10, "a", 1), (11, "b", 2), (12, "c", 4)],
    )
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme", "London"): 1}, columns=("name", "city")
    )
    assert not plan.can_apply
    assert plan.unique_conflicts
    with pytest.raises(DedupeError, match="unique constraint"):
        fresh_db.table("companies").dedupe_apply(plan)
    # Nothing was changed
    assert fresh_db.table("companies").count == 4
    assert fresh_db.table("contacts").count == 3


def test_dedupe_detects_conflict_on_compound_unique_index(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE memberships(
            id INTEGER PRIMARY KEY,
            role TEXT,
            company_id INTEGER REFERENCES companies(id),
            UNIQUE(role, company_id)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO companies VALUES (?, ?)", [(1, "Acme"), (2, "Acme")]
    )
    fresh_db.conn.executemany(
        "INSERT INTO memberships VALUES (?, ?, ?)",
        [(10, "admin", 1), (11, "admin", 2)],
    )
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme",): 1}, columns=("name",)
    )
    conflict = plan.unique_conflicts[0]
    assert conflict.table == "memberships"
    assert conflict.constraint_columns == ("role", "company_id")
    assert conflict.new_value == ("admin", 1)
    with pytest.raises(DedupeError):
        fresh_db.table("companies").dedupe_apply(plan)


def test_dedupe_compound_unique_no_conflict_when_other_column_differs(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE memberships(
            id INTEGER PRIMARY KEY,
            role TEXT,
            company_id INTEGER REFERENCES companies(id),
            UNIQUE(role, company_id)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO companies VALUES (?, ?)", [(1, "Acme"), (2, "Acme")]
    )
    fresh_db.conn.executemany(
        "INSERT INTO memberships VALUES (?, ?, ?)",
        [(10, "admin", 1), (11, "user", 2)],
    )
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme",): 1}, columns=("name",)
    )
    assert plan.can_apply
    fresh_db.table("companies").dedupe_apply(plan)
    assert [
        row["company_id"] for row in fresh_db.table("memberships").rows
    ] == [1, 1]


def test_dedupe_partial_unique_index_conflict_rolls_back(fresh_db):
    _companies_with_contacts(fresh_db, partial_unique=True)
    fresh_db.conn.executemany(
        "INSERT INTO contacts(id, active, company_id) VALUES (?, ?, ?)",
        [(10, 1, 1), (11, 1, 2)],
    )
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme", "London"): 1}, columns=("name", "city")
    )
    # The static check cannot evaluate the partial index predicate
    assert plan.unique_conflicts == ()
    with pytest.raises(sqlite3_module.IntegrityError):
        fresh_db.table("companies").dedupe_apply(plan)
    # The whole transaction rolled back
    assert fresh_db.table("companies").count == 3
    assert [
        row["company_id"] for row in fresh_db.table("contacts").rows
    ] == [1, 2]


def test_dedupe_null_values_match_and_do_not_create_unique_conflict(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE contacts(
            id INTEGER PRIMARY KEY,
            company_id INTEGER UNIQUE REFERENCES companies(id)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO companies VALUES (?, ?)", [(1, "Acme"), (2, "Acme")]
    )
    # NULL references are distinct in a unique index and never collide
    fresh_db.conn.executemany(
        "INSERT INTO contacts VALUES (?, ?)", [(10, None), (11, 2)]
    )
    plan = fresh_db.table("companies").dedupe_plan(
        {("Acme",): 1}, columns=("name",)
    )
    assert plan.can_apply
    fresh_db.table("companies").dedupe_apply(plan)
    assert [row["company_id"] for row in fresh_db.table("contacts").rows] == [
        None,
        1,
    ]


def test_dedupe_stale_plan_is_rejected(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "x")]
    )
    table = fresh_db.table("t")
    plan = table.dedupe_plan({("x",): 1}, columns=("name",))
    fresh_db.execute("INSERT INTO t (id, name) VALUES (3, 'x')")
    with pytest.raises(DedupeError, match="has changed"):
        table.dedupe_apply(plan)
    assert table.count == 3


def test_dedupe_convenience_method(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "x"), (3, "y")]
    )
    plan = fresh_db.table("t").dedupe({("x",): 1}, columns=("name",))
    assert isinstance(plan, DedupePlan)
    assert [row["id"] for row in fresh_db.table("t").rows] == [1, 3]


def test_duplicate_groups_is_read_only(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(1, "x"), (2, "x"), (3, "x"), (4, "y")],
    )
    groups = fresh_db.table("t").duplicate_groups(columns=("name",))
    assert groups == [(("x",), ((1,), (2,), (3,)))]
    assert fresh_db.table("t").count == 4


def test_dedupe_multiple_groups(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(1, "x"), (2, "x"), (3, "y"), (4, "y")],
    )
    plan = fresh_db.table("t").dedupe_plan(
        {("x",): 1, ("y",): 3}, columns=("name",)
    )
    assert plan.delete_count == 2
    fresh_db.table("t").dedupe_apply(plan)
    assert [row["id"] for row in fresh_db.table("t").rows] == [1, 3]


def test_dedupe_plan_rejects_wrong_table(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "x")]
    )
    plan = fresh_db.table("t").dedupe_plan(
        {("x",): 1}, columns=("name",)
    )
    fresh_db.execute("CREATE TABLE other(id INTEGER PRIMARY KEY, name TEXT)")
    with pytest.raises(DedupeError, match="not 'other'"):
        fresh_db.table("other").dedupe_apply(plan)


def test_dedupe_no_duplicates_is_a_noop(fresh_db):
    fresh_db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)")
    fresh_db.conn.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")]
    )
    plan = fresh_db.table("t").dedupe_plan(columns=("name",))
    assert plan.can_apply
    assert plan.delete_count == 0
    assert plan.repoints == ()
    fresh_db.table("t").dedupe_apply(plan)
    assert fresh_db.table("t").count == 2


def test_dedupe_without_rowid_table(fresh_db):
    fresh_db.conn.execute("PRAGMA foreign_keys=ON")
    fresh_db.executescript(
        """
        CREATE TABLE node(
            a TEXT, b TEXT, label TEXT, PRIMARY KEY(a, b)
        ) WITHOUT ROWID;
        CREATE TABLE edge(
            id INTEGER PRIMARY KEY,
            a TEXT, b TEXT,
            FOREIGN KEY(a, b) REFERENCES node(a, b)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO node VALUES (?, ?, ?)",
        [("x", "1", "d"), ("x", "2", "d")],
    )
    fresh_db.conn.execute("INSERT INTO edge VALUES (?, ?, ?)", (10, "x", "2"))
    plan = fresh_db.table("node").dedupe_plan({("d",): ("x", "1")})
    fresh_db.table("node").dedupe_apply(plan)
    assert list(fresh_db.table("node").rows) == [
        {"a": "x", "b": "1", "label": "d"}
    ]
    assert list(fresh_db.table("edge").rows) == [
        {"id": 10, "a": "x", "b": "1"}
    ]


def test_dedupe_rejects_foreign_key_to_non_primary_key_column(fresh_db):
    fresh_db.executescript(
        """
        CREATE TABLE p(
            id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT
        );
        CREATE TABLE c(
            id INTEGER PRIMARY KEY, p_code TEXT REFERENCES p(code)
        );
        """
    )
    fresh_db.conn.executemany(
        "INSERT INTO p VALUES (?, ?, ?)",
        [(1, "A", "dup"), (2, "B", "dup")],
    )
    with pytest.raises(DedupeError, match="non-primary-key column"):
        fresh_db.table("p").dedupe_plan({("dup",): 1}, columns=("name",))


def _write_cli_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE contacts(
            id INTEGER PRIMARY KEY,
            name TEXT,
            company_id INTEGER REFERENCES companies(id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO companies VALUES (?, ?, ?)",
        [(1, "Acme", "London"), (2, "Acme", "London"), (3, "Globex", "Berlin")],
    )
    conn.executemany(
        "INSERT INTO contacts VALUES (?, ?, ?)",
        [(10, "a", 1), (11, "b", 2)],
    )
    conn.commit()
    conn.close()


def test_cli_dedupe_without_winners_lists_groups(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        ["dedupe", str(path), "companies", "--column", "name", "--column", "city"],
    )
    assert result.exit_code == 0
    assert "candidate primary keys" in result.output
    assert "Acme, London" in result.output
    # No changes were made
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 3
    conn.close()


def test_cli_dedupe_preview_does_not_modify_database(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--column",
            "city",
            "--winner",
            "Acme,London",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Preview only" in result.output
    assert "Foreign keys to repoint" in result.output
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 3
    conn.close()


def test_cli_dedupe_apply_confirmation(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--column",
            "city",
            "--winner",
            "Acme,London",
            "1",
            "--apply",
        ],
        input="y\n",
    )
    assert result.exit_code == 0, result.exception
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT id FROM companies ORDER BY id").fetchall() == [
        (1,),
        (3,),
    ]
    assert conn.execute(
        "SELECT company_id FROM contacts ORDER BY id"
    ).fetchall() == [(1,), (1,)]
    conn.close()


def test_cli_dedupe_declined_confirmation_changes_nothing(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--column",
            "city",
            "--winner",
            "Acme,London",
            "1",
            "--apply",
        ],
        input="n\n",
    )
    assert result.exit_code != 0
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 3
    conn.close()


def test_cli_dedupe_yes_applies_without_prompt(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--column",
            "city",
            "--winner",
            "Acme,London",
            "1",
            "-y",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "Done:" in result.output
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 2
    conn.close()


def test_cli_dedupe_unique_conflict_blocks_application(tmpdir):
    path = tmpdir / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE contacts(
            id INTEGER PRIMARY KEY,
            company_id INTEGER UNIQUE REFERENCES companies(id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO companies VALUES (?, ?)",
        [(1, "Acme"), (2, "Acme"), (3, "Acme")],
    )
    conn.executemany(
        "INSERT INTO contacts VALUES (?, ?)", [(10, 1), (11, 2), (12, 3)]
    )
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--winner",
            "Acme",
            "1",
            "-y",
        ],
    )
    assert result.exit_code != 0
    assert "unique constraint" in result.output
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 3
    conn.close()


def test_cli_dedupe_missing_winner_errors(tmpdir):
    path = tmpdir / "test.db"
    _write_cli_db(path)
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--column",
            "city",
            "--winner",
            "Acme,London",
            "99",
        ],
    )
    assert result.exit_code != 0
    assert "is not one of the rows" in result.output


def test_cli_dedupe_null_key(tmpdir):
    path = tmpdir / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE contacts(
            id INTEGER PRIMARY KEY,
            company_id INTEGER REFERENCES companies(id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO companies VALUES (?, ?)",
        [(1, None), (2, None), (3, "z")],
    )
    conn.executemany(
        "INSERT INTO contacts VALUES (?, ?)", [(10, 1), (11, 2)]
    )
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        cli.cli,
        [
            "dedupe",
            str(path),
            "companies",
            "--column",
            "name",
            "--winner",
            "NULL",
            "1",
            "-y",
        ],
    )
    assert result.exit_code == 0, result.exception
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT id FROM companies ORDER BY id").fetchall() == [
        (1,),
        (3,),
    ]
    assert conn.execute(
        "SELECT company_id FROM contacts ORDER BY id"
    ).fetchall() == [(1,), (1,)]
    conn.close()
