"""Tests for incremental foreign key violation auditing."""

import pytest
from hypothesis import HealthCheck, given, settings, assume
from hypothesis import strategies as st

from sqlite_utils import Database, RowChanges
from sqlite_utils.db import (
    ForeignKey,
    ForeignKeyAuditor,
    ForeignKeyViolation,
)
from sqlite_utils.utils import sqlite3


def _build_db(without_rowid=False):
    db = Database(memory=True)
    if without_rowid:
        db.executescript(
            """
            CREATE TABLE parents (
                k1 TEXT NOT NULL, k2 TEXT NOT NULL, v TEXT,
                PRIMARY KEY (k1, k2)
            ) WITHOUT ROWID;
            CREATE TABLE children (
                id INTEGER NOT NULL, parent_k1 TEXT, parent_k2 TEXT,
                backup_k1 TEXT, backup_k2 TEXT,
                PRIMARY KEY (id)
            ) WITHOUT ROWID;
            """
        )
        db.add_foreign_keys(
            [
                ("children", ("parent_k1", "parent_k2"),
                 "parents", ("k1", "k2")),
                ("children", ("backup_k1", "backup_k2"),
                 "parents", ("k1", "k2")),
            ]
        )
    else:
        db.executescript(
            """
            CREATE TABLE parents (id INTEGER PRIMARY KEY, label TEXT);
            CREATE TABLE backups (id INTEGER PRIMARY KEY);
            CREATE TABLE children (
                id INTEGER PRIMARY KEY,
                pid INTEGER REFERENCES parents(id),
                bid INTEGER REFERENCES backups(id)
            );
            """
        )
    db.conn.execute("PRAGMA foreign_keys=OFF")
    db.index_foreign_keys()
    return db


# --------------------------------------------------------------------------
# Full audit
# --------------------------------------------------------------------------

def test_full_audit_reports_violation_details_rowid():
    db = _build_db()
    db["parents"].insert({"id": 1, "label": "ok"})
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 10, "pid": 1, "bid": 9},
            {"id": 11, "pid": 2, "bid": 9},
            {"id": 12, "pid": 1, "bid": 8},
        ]
    )
    report = db.audit_foreign_keys()
    by_row = {}
    for violation in report.violations:
        by_row.setdefault(violation.row_pk, []).append(violation)

    assert sorted(by_row) == [11, 12]
    orphan = by_row[11][0]
    assert isinstance(orphan, ForeignKeyViolation)
    assert orphan.table == "children"
    assert orphan.row_pk == 11
    assert orphan.key_values == (2,)
    assert orphan.source_values == {"id": 11, "pid": 2, "bid": 9}
    assert orphan.foreign_key.other_table == "parents"
    assert orphan.foreign_key.columns == ("pid",)
    # Row 12 fails the backups constraint, not the parents one
    assert by_row[12][0].foreign_key.other_table == "backups"


def test_same_row_two_constraints_stay_separate():
    db = _build_db()
    db["parents"].insert({"id": 1})
    db["children"].insert({"id": 1, "pid": 77, "bid": 88})
    report = db.audit_foreign_keys()
    assert len(report.violations) == 2
    targets = sorted(v.foreign_key.other_table for v in report.violations)
    assert targets == ["backups", "parents"]
    assert all(v.row_pk == 1 for v in report.violations)


def test_null_keys_are_not_violations_match_none():
    db = _build_db()
    db["children"].insert({"id": 1, "pid": None, "bid": None})
    report = db.audit_foreign_keys()
    assert report.violations == ()


def test_without_rowid_compound_keys_match_back():
    db = _build_db(without_rowid=True)
    db["parents"].insert({"k1": "a", "k2": "b", "v": "x"})
    db["children"].insert_all(
        [
            {"id": 1, "parent_k1": "a", "parent_k2": "b",
             "backup_k1": "a", "backup_k2": "b"},
            {"id": 2, "parent_k1": "a", "parent_k2": "zzz",
             "backup_k1": "a", "backup_k2": "b"},
            {"id": 3, "parent_k1": None, "parent_k2": None,
             "backup_k1": "a", "backup_k2": "b"},
        ]
    )
    report = db.audit_foreign_keys()
    assert len(report.violations) == 1
    violation = report.violations[0]
    # The single-column WITHOUT ROWID primary key still maps back by itself;
    # the compound parent key is exercised through key_values.
    assert violation.row_pk == 2
    assert violation.key_values == ("a", "zzz")
    assert violation.foreign_key.columns == ("parent_k1", "parent_k2")
    assert violation.source_values["parent_k2"] == "zzz"


# --------------------------------------------------------------------------
# Incremental classification
# --------------------------------------------------------------------------

def test_incremental_new_resolved_and_remaining():
    db = _build_db()
    db["parents"].insert_all([{"id": 1}, {"id": 3}])
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 1, "pid": 2, "bid": 9},   # orphan: missing parent 2
            {"id": 2, "pid": 3, "bid": 9},   # clean
            {"id": 3, "pid": 4, "bid": 9},   # orphan: stays unresolved
        ]
    )
    baseline = db.audit_foreign_keys()
    assert {v.row_pk for v in baseline.violations} == {1, 3}

    # Parent 2 is added (resolves row 1); parent 3 is deleted (row 2 breaks).
    db["parents"].insert({"id": 2})
    db["parents"].delete_where("id = 3")
    changes = RowChanges(
        inserted={"parents": {2: {"id": 2}}},
        deleted={"parents": {3: None}},
        previous={"parents": {3: {"id": 3}}},
    )
    report = db.recheck_foreign_keys(baseline.violations, changes)

    assert [v.row_pk for v in report.new_violations] == [2]
    assert [v.row_pk for v in report.resolved_violations] == [1]
    assert [v.row_pk for v in report.remaining_violations] == [3]
    # The three buckets partition the old/new evidence without overlap
    current = set(report.new_violations) | set(report.remaining_violations)
    assert len(current) == len(report.violations)


def test_child_row_update_can_resolve_or_create_violation():
    db = _build_db()
    db["parents"].insert_all([{"id": 1}, {"id": 5}])
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 1, "pid": 2, "bid": 9},
            {"id": 2, "pid": 1, "bid": 9},
        ]
    )
    baseline = db.audit_foreign_keys()
    assert [v.row_pk for v in baseline.violations] == [1]

    # Row 1 fixed, row 2 broken
    db["children"].update(1, {"pid": 1})
    db["children"].update(2, {"pid": 6})
    changes = RowChanges(
        updated={
            "children": {
                1: {"id": 1, "pid": 1, "bid": 9},
                2: {"id": 2, "pid": 6, "bid": 9},
            }
        }
    )
    report = db.recheck_foreign_keys(baseline.violations, changes)
    assert [v.row_pk for v in report.resolved_violations] == [1]
    assert [v.row_pk for v in report.new_violations] == [2]
    assert report.remaining_violations == ()


def test_parent_key_update_uses_previous_interval():
    db = _build_db()
    db["parents"].insert({"id": 10})
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 1, "pid": 10, "bid": 9},
            {"id": 2, "pid": 11, "bid": 9},
        ]
    )
    baseline = db.audit_foreign_keys()
    assert [v.row_pk for v in baseline.violations] == [2]

    # Parent moves from key 10 to key 12: row 1 breaks, nothing references 12
    db["parents"].update(10, {"id": 12})
    changes = RowChanges(
        updated={"parents": {12: {"id": 12}}},
        previous={"parents": {10: {"id": 10}}},
    )
    report = db.recheck_foreign_keys(baseline.violations, changes)
    assert [v.row_pk for v in report.new_violations] == [1]
    # Row 2 was not in either key interval {10, 12}, so it is untouched and
    # carried forward as still unresolved.
    assert [v.row_pk for v in report.remaining_violations] == [2]
    assert report.resolved_violations == ()


def test_unrelated_change_leaves_violations_untouched_and_unread():
    db = _build_db()
    db["backups"].insert({"id": 9})
    db["children"].insert({"id": 1, "pid": 2, "bid": 9})
    baseline = db.audit_foreign_keys()

    # A change on an unrelated table must not re-read the violating row.
    db["backups"].insert({"id": 10})
    changes = RowChanges(inserted={"backups": {10: {"id": 10}}})
    report = db.recheck_foreign_keys(baseline.violations, changes)
    assert report.remaining_violations == baseline.violations
    assert report.new_violations == ()
    assert report.resolved_violations == ()
    # No child/parents access at all - only the backups constraint was probed
    touched = {access.table for access in report.access}
    assert "children" not in touched
    assert "parents" not in touched


def test_empty_changes_carries_everything_forward():
    db = _build_db()
    db["children"].insert({"id": 1, "pid": 2, "bid": 3})
    baseline = db.audit_foreign_keys()
    report = db.recheck_foreign_keys(baseline.violations, RowChanges())
    assert report.remaining_violations == baseline.violations
    assert report.access == ()


# --------------------------------------------------------------------------
# Bounded access
# --------------------------------------------------------------------------

def test_access_is_bounded_to_affected_key_interval():
    db = _build_db()
    db["parents"].insert_all([{"id": i} for i in range(1, 6)])
    db["backups"].insert({"id": 9})
    # 100 children, all valid except none initially
    db["children"].insert_all(
        {"id": i, "pid": (i % 5) + 1, "bid": 9} for i in range(1, 101)
    )
    baseline = db.audit_foreign_keys()
    assert baseline.violations == ()

    # Delete parent 3: only the ~20 rows with pid=3 are visited.
    db["parents"].delete_where("id = 3")
    changes = RowChanges(
        deleted={"parents": {3: None}},
        previous={"parents": {3: {"id": 3}}},
    )
    report = db.recheck_foreign_keys(baseline.violations, changes)
    assert len(report.new_violations) == 20

    # The foreign key index is probed only inside the pid=3 interval.
    fk_access = next(
        a for a in report.access
        if a.table == "children" and a.index and "pid" in a.index
    )
    assert fk_access.rows_scanned == 20
    pid_range = next(r for r in fk_access.ranges if r.column == "pid")
    assert pid_range.access == "eq"
    assert pid_range.values == (3,)

    # The matched rows are then re-read through the primary key.
    pk_access = next(
        a for a in report.access
        if a.table == "children"
        and (a.index == "rowid" or a.index.startswith("PRIMARY KEY"))
        and a is not fk_access
    )
    assert pk_access.rows_scanned == 20

    parent_access = next(a for a in report.access if a.table == "parents")
    assert parent_access.rows_scanned == 20


def test_tracer_shows_no_unfiltered_child_scan():
    db = _build_db()
    db["parents"].insert({"id": 1})
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        {"id": i, "pid": 1, "bid": 9} for i in range(1, 51)
    )
    db["parents"].delete_where("id = 1")
    changes = RowChanges(
        deleted={"parents": {1: None}},
        previous={"parents": {1: {"id": 1}}},
    )
    seen = []
    with db.tracer(lambda sql, params: seen.append((sql, params))):
        db.recheck_foreign_keys((), changes)
    child_scans = [
        sql for sql, _ in seen
        if '"children"' in sql and sql.lstrip().lower().startswith("select")
    ]
    assert child_scans, "expected child lookups"
    assert all("where" in sql.lower() for sql in child_scans)


def test_describe_access_is_human_readable():
    db = _build_db()
    db["children"].insert({"id": 1, "pid": 2, "bid": 3})
    report = db.audit_foreign_keys()
    text = report.describe_access()
    assert "children" in text


# --------------------------------------------------------------------------
# Schema / collation evidence invalidation
# --------------------------------------------------------------------------

def test_foreign_key_declaration_change_invalidates_evidence():
    db = _build_db()
    db["parents"].insert({"id": 1})
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 1, "pid": 1, "bid": 9},
            {"id": 2, "pid": 1, "bid": 7},
        ]
    )
    baseline = db.audit_foreign_keys()
    assert len(baseline.violations) == 1
    assert baseline.violations[0].foreign_key.other_table == "backups"

    # Rebuild children keeping only the parents foreign key.
    db.conn.execute("ALTER TABLE children RENAME TO children_old")
    db.executescript(
        """
        CREATE TABLE children (
            id INTEGER PRIMARY KEY,
            pid INTEGER REFERENCES parents(id),
            bid INTEGER
        );
        """
    )
    db.conn.execute("INSERT INTO children SELECT id, pid, bid FROM children_old")
    report = db.recheck_foreign_keys(
        baseline.violations, RowChanges(updated={"children": {}})
    )
    assert len(report.invalidated_evidence) == 1
    invalid = report.invalidated_evidence[0]
    assert invalid.foreign_key.other_table == "backups"
    # The dropped constraint cannot be re-verified, so it is not resolved
    assert report.resolved_violations == ()


def test_collation_upgrade_invalidates_old_evidence():
    db = Database(memory=True)
    db.executescript(
        """
        CREATE TABLE p (id TEXT PRIMARY KEY COLLATE NOCASE);
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            pid TEXT REFERENCES p(id)
        );
        """
    )
    old_fk = ForeignKey(
        table="c", column="pid", other_table="p", other_column="id"
    )
    old_violation = ForeignKeyViolation(
        table="c",
        foreign_key=old_fk,
        row_pk=1,
        key_values=("a",),
        source_values={"id": 1, "pid": "a"},
        parent_collations=(("id", "NOCASE"),),
    )
    # Rebuild the parent with a different collation.
    db.conn.execute("ALTER TABLE p RENAME TO p_old")
    db.executescript("CREATE TABLE p (id TEXT PRIMARY KEY COLLATE BINARY);")
    db.conn.execute("INSERT INTO p SELECT id FROM p_old")
    report = db.recheck_foreign_keys(
        [old_violation], RowChanges(updated={"p": {}})
    )
    assert len(report.invalidated_evidence) == 1
    assert report.invalidated_evidence[0].row_pk == 1


def test_child_collation_change_invalidates_evidence():
    db = Database(memory=True)
    db.executescript(
        """
        CREATE TABLE p (id TEXT PRIMARY KEY);
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            pid TEXT COLLATE NOCASE REFERENCES p(id)
        );
        """
    )
    old_fk = db["c"].foreign_keys[0]
    old_violation = ForeignKeyViolation(
        table="c",
        foreign_key=old_fk,
        row_pk=1,
        key_values=("z",),
        source_values={"id": 1, "pid": "z"},
        child_collations=(("pid", "NOCASE"),),
    )
    db.conn.execute("ALTER TABLE c RENAME TO c_old")
    db.executescript(
        """
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            pid TEXT COLLATE BINARY REFERENCES p(id)
        );
        """
    )
    db.conn.execute("INSERT INTO c SELECT id, pid FROM c_old")
    report = db.recheck_foreign_keys(
        [old_violation], RowChanges(updated={"c": {}})
    )
    assert len(report.invalidated_evidence) == 1


# --------------------------------------------------------------------------
# Equivalence with a full re-scan (property test)
# --------------------------------------------------------------------------

@st.composite
def _scenario(draw):
    without_rowid = draw(st.booleans())
    db = _build_db(without_rowid)
    parent_rows = draw(
        st.lists(
            st.integers(min_value=1, max_value=8) if not without_rowid
            else st.text(alphabet="ab", min_size=1, max_size=2),
            min_size=1,
            max_size=6,
        )
    )
    assume(len(set(parent_rows)) == len(parent_rows))
    child_rows = draw(st.integers(min_value=1, max_value=12))
    return db, without_rowid, parent_rows, child_rows


def _seed(db, without_rowid, parent_rows, child_count):
    if without_rowid:
        db["parents"].insert_all(
            {"k1": value, "k2": "x", "v": None} for value in parent_rows
        )
        keys = [(value, "x") for value in parent_rows]
        db["children"].insert_all(
            {
                "id": i,
                "parent_k1": keys[i % len(keys)][0] if i % 4 else "ghost",
                "parent_k2": "x" if i % 4 else "x",
                "backup_k1": keys[i % len(keys)][0],
                "backup_k2": "x",
            }
            for i in range(child_count)
        )
    else:
        db["parents"].insert_all({"id": value} for value in parent_rows)
        db["backups"].insert_all({"id": value} for value in parent_rows)
        db["children"].insert_all(
            {
                "id": i,
                "pid": parent_rows[i % len(parent_rows)] if i % 4 else 999,
                "bid": parent_rows[i % len(parent_rows)],
            }
            for i in range(child_count)
        )


@given(scenario=_scenario())
@settings(
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_incremental_matches_full_rescan(scenario):
    db, without_rowid, parent_rows, child_count = scenario
    _seed(db, without_rowid, parent_rows, child_count)
    auditor = ForeignKeyAuditor(db)
    baseline = auditor.full_audit()

    # Apply a random parent deletion and a random child upsert, tracking both
    # current and previous row states explicitly.
    victim = parent_rows[0]
    if without_rowid:
        previous_parent = {"k1": victim, "k2": "x", "v": None}
        db["parents"].delete_where("k1 = ?", [victim])
        child_row = {
            "id": child_count,
            "parent_k1": "phantom",
            "parent_k2": "x",
            "backup_k1": parent_rows[-1],
            "backup_k2": "x",
        }
        changes = RowChanges(
            inserted={"children": {child_count: child_row}},
            deleted={"parents": {(victim, "x"): None}},
            previous={"parents": {(victim, "x"): previous_parent}},
        )
    else:
        previous_parent = {"id": victim, "label": None}
        db["parents"].delete_where("id = ?", [victim])
        child_row = {
            "id": child_count,
            "pid": 123456,
            "bid": parent_rows[-1],
        }
        changes = RowChanges(
            inserted={"children": {child_count: child_row}},
            deleted={"parents": {victim: None}},
            previous={"parents": {victim: previous_parent}},
        )
    db["children"].insert(child_row, replace=True)

    incremental = auditor.audit(baseline.violations, changes)
    full = auditor.full_audit()

    assert {auditor._identity(v) for v in incremental.violations} == {
        auditor._identity(v) for v in full.violations
    }
    # No schema changed, so no evidence can be invalidated.
    assert incremental.invalidated_evidence == ()
    # Old identities partition cleanly into resolved vs remaining.
    old_identities = {auditor._identity(v) for v in baseline.violations}
    resolved = {auditor._identity(v) for v in incremental.resolved_violations}
    remaining = {auditor._identity(v) for v in incremental.remaining_violations}
    assert resolved.isdisjoint(remaining)
    assert resolved | remaining == old_identities


def test_maximal_change_set_equals_full_scan():
    db = _build_db()
    db["parents"].insert_all([{"id": 1}, {"id": 2}])
    db["backups"].insert({"id": 9})
    db["children"].insert_all(
        [
            {"id": 1, "pid": 1, "bid": 9},
            {"id": 2, "pid": 3, "bid": 9},
            {"id": 3, "pid": 2, "bid": 4},
        ]
    )
    full = db.audit_foreign_keys()
    # Listing every child row as changed forces verification of the whole set.
    all_rows = {row["id"]: row for row in db["children"].rows}
    report = db.recheck_foreign_keys((), RowChanges(updated={"children": all_rows}))
    auditor = ForeignKeyAuditor(db)
    assert {auditor._identity(v) for v in report.violations} == {
        auditor._identity(v) for v in full.violations
    }


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def test_row_changes_delete_wins_and_table_union():
    changes = RowChanges(
        inserted={"t": {1: {"id": 1}}},
        updated={"t": {1: {"id": 2}}},
        deleted={"t": {1: None}},
    )
    assert changes.current_entries("t") == {1: None}
    assert changes.tables() == frozenset({"t"})


def test_table_level_audit_methods():
    db = _build_db()
    db["children"].insert({"id": 1, "pid": 2, "bid": 3})
    report = db["children"].audit_foreign_keys()
    assert len(report.violations) == 2
    again = db["children"].recheck_foreign_keys(
        report.violations, RowChanges()
    )
    assert again.remaining_violations == report.violations


def test_self_referencing_foreign_key():
    db = Database(memory=True)
    db.executescript(
        """
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER REFERENCES nodes(id)
        );
        """
    )
    db.conn.execute("PRAGMA foreign_keys=OFF")
    db.index_foreign_keys()
    db["nodes"].insert_all([{"id": 1, "parent_id": None}, {"id": 2, "parent_id": 99}])
    report = db.audit_foreign_keys()
    assert [v.row_pk for v in report.violations] == [2]

    db["nodes"].insert({"id": 99, "parent_id": None})
    changes = RowChanges(
        inserted={"nodes": {99: {"id": 99, "parent_id": None}}}
    )
    follow_up = db.recheck_foreign_keys(report.violations, changes)
    assert [v.row_pk for v in follow_up.resolved_violations] == [2]
    assert follow_up.new_violations == ()


def test_deleted_child_row_settles_its_violation():
    db = _build_db()
    db["children"].insert({"id": 1, "pid": 2, "bid": 3})
    baseline = db.audit_foreign_keys()
    assert len(baseline.violations) == 2

    db["children"].delete_where("id = 1")
    changes = RowChanges(deleted={"children": {1: None}})
    report = db.recheck_foreign_keys(baseline.violations, changes)
    assert {v.row_pk for v in report.resolved_violations} == {1}
    assert report.remaining_violations == ()
