import pytest

from sqlite_utils import Database
from sqlite_utils.db import BatchImportReceipt, ForeignKeyViolationError
from sqlite_utils.utils import sqlite3


@pytest.fixture
def refs_db():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT,
            featured_book_id INTEGER REFERENCES books(id)
        );
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author_id INTEGER REFERENCES authors(id)
        );
    """)
    return db


@pytest.fixture
def composite_db():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE dims (
            region TEXT,
            code INTEGER,
            label TEXT,
            PRIMARY KEY (region, code)
        );
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY,
            fact_region TEXT,
            fact_code INTEGER,
            FOREIGN KEY (fact_region, fact_code) REFERENCES dims(region, code)
        );
    """)
    return db


def insert_bad_row(db, table, row):
    "Insert a row that violates a foreign key, bypassing enforcement."
    foreign_keys_was_on = bool(db.execute("PRAGMA foreign_keys").fetchone()[0])
    db.execute("PRAGMA foreign_keys=OFF")
    try:
        db[table].insert(row)
    finally:
        if foreign_keys_was_on:
            db.execute("PRAGMA foreign_keys=ON")


def test_import_batch_mutually_referencing_tables(refs_db):
    # authors references books and books references authors - neither table
    # can be loaded first with foreign key enforcement on
    receipt = refs_db.import_batch(
        {
            "authors": [{"id": 1, "name": "Ann", "featured_book_id": 10}],
            "books": [{"id": 10, "title": "Book", "author_id": 1}],
        }
    )
    assert isinstance(receipt, BatchImportReceipt)
    assert receipt.committed is True
    assert receipt.status == "committed"
    assert receipt.rows_written == {"authors": 1, "books": 1}
    assert receipt.tables == ("authors", "books")
    assert receipt.resolved == ()
    assert refs_db["authors"].get(1)["featured_book_id"] == 10
    assert refs_db["books"].get(10)["author_id"] == 1
    # Foreign key enforcement was restored afterwards
    assert refs_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_import_batch_rejects_new_violation_and_rolls_back_everything(refs_db):
    with pytest.raises(ForeignKeyViolationError) as excinfo:
        refs_db.import_batch(
            {
                "authors": [{"id": 1, "name": "Ann", "featured_book_id": 10}],
                "books": [
                    {"id": 10, "title": "Book", "author_id": 1},
                    {"id": 11, "title": "Bad", "author_id": 999},
                ],
            }
        )
    assert isinstance(excinfo.value, sqlite3.IntegrityError)
    violations = excinfo.value.violations
    assert len(violations) == 1
    violation = violations[0]
    assert violation.table == "books"
    assert violation.child == {"author_id": 999}
    assert violation.parent == {"id": 999}
    assert violation.parent_table == "authors"
    assert violation.row["id"] == 11
    # The whole batch was rejected, not just the bad record
    assert refs_db["authors"].count == 0
    assert refs_db["books"].count == 0


def test_import_batch_tolerates_preexisting_violations(refs_db):
    insert_bad_row(refs_db, "books", {"id": 1, "title": "Old bad", "author_id": 999})
    receipt = refs_db.import_batch(
        {
            "authors": [{"id": 1, "name": "Ann", "featured_book_id": None}],
            "books": [{"id": 2, "title": "Good", "author_id": 1}],
        }
    )
    assert receipt.committed is True
    assert receipt.resolved == ()
    # The historical bad row is untouched
    assert refs_db["books"].get(1)["author_id"] == 999
    assert refs_db["books"].count == 2


def test_import_batch_rejects_aggravated_violation(refs_db):
    insert_bad_row(refs_db, "books", {"id": 1, "title": "Old bad", "author_id": 999})
    with pytest.raises(ForeignKeyViolationError) as excinfo:
        refs_db.import_batch(
            {"books": [{"id": 2, "title": "Another bad", "author_id": 999}]}
        )
    assert len(excinfo.value.violations) == 1
    assert excinfo.value.violations[0].child == {"author_id": 999}
    assert refs_db["books"].count == 1


def test_import_batch_corrections_cancel_before_commit(refs_db):
    # The first record puts the row into a bad state and the second record
    # fixes it again - only the net state at the end of the batch counts
    receipt = refs_db.import_batch(
        {
            "authors": [{"id": 1, "name": "Ann", "featured_book_id": None}],
            "books": [
                {"id": 1, "title": "Draft", "author_id": 999},
                {"id": 1, "title": "Fixed", "author_id": 1},
            ],
        },
        upsert=True,
    )
    assert receipt.committed is True
    assert refs_db["books"].get(1)["author_id"] == 1


def test_import_batch_outer_transaction_failure_rolls_back_only_batch(refs_db):
    refs_db.begin()
    refs_db["authors"].insert({"id": 1, "name": "Outer", "featured_book_id": None})
    with pytest.raises(ForeignKeyViolationError):
        refs_db.import_batch({"books": [{"id": 1, "title": "Bad", "author_id": 999}]})
    # The caller's uncommitted change survived, only the batch was undone
    assert refs_db["authors"].count == 1
    assert refs_db["books"].count == 0
    # defer_foreign_keys was restored
    assert refs_db.execute("PRAGMA defer_foreign_keys").fetchone()[0] == 0
    refs_db.rollback()
    assert refs_db["authors"].count == 0


def test_import_batch_outer_transaction_success_is_visible_not_committed(refs_db):
    refs_db.begin()
    refs_db["authors"].insert({"id": 1, "name": "Outer", "featured_book_id": None})
    receipt = refs_db.import_batch(
        {"books": [{"id": 1, "title": "Book", "author_id": 1}]}
    )
    assert receipt.committed is False
    assert receipt.status == "visible-in-outer-transaction"
    # Visible within the outer transaction...
    assert refs_db["books"].count == 1
    # ...but not durably committed - it goes away with the outer rollback
    refs_db.rollback()
    assert refs_db["books"].count == 0


def test_import_batch_committed_is_durable(tmpdir):
    path = str(tmpdir / "batch.db")
    db = Database(path)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author_id INTEGER REFERENCES authors(id)
        );
    """)
    receipt = db.import_batch(
        {
            "authors": [{"id": 1, "name": "Ann"}],
            "books": [{"id": 1, "title": "Book", "author_id": 1}],
        }
    )
    assert receipt.committed is True
    db.close()
    db2 = Database(path)
    assert db2["authors"].count == 1
    assert db2["books"].count == 1


def test_import_batch_resolved_records_for_fixed_rows(refs_db):
    refs_db["authors"].insert({"id": 1, "name": "Ann", "featured_book_id": None})
    insert_bad_row(refs_db, "books", {"id": 5, "title": "Broken", "author_id": 999})
    receipt = refs_db.import_batch(
        {"books": [{"id": 5, "title": "Fixed", "author_id": 1}]},
        upsert=True,
    )
    assert receipt.committed is True
    assert len(receipt.resolved) == 1
    resolved = receipt.resolved[0]
    assert resolved.table == "books"
    assert resolved.child == {"author_id": 999}
    assert resolved.parent == {"id": 999}
    assert resolved.parent_table == "authors"
    assert resolved.row["id"] == 5
    assert refs_db["books"].get(5)["author_id"] == 1


def test_import_batch_composite_foreign_key_maps_both_ends(composite_db):
    with pytest.raises(ForeignKeyViolationError) as excinfo:
        composite_db.import_batch(
            {"facts": [{"id": 1, "fact_region": "eu", "fact_code": 7}]}
        )
    violation = excinfo.value.violations[0]
    assert violation.foreign_key.is_compound
    assert violation.table == "facts"
    assert violation.child == {"fact_region": "eu", "fact_code": 7}
    assert violation.parent == {"region": "eu", "code": 7}
    assert violation.parent_table == "dims"
    assert violation.row == {"id": 1, "fact_region": "eu", "fact_code": 7}


def test_import_batch_composite_foreign_key_satisfied_within_batch(composite_db):
    receipt = composite_db.import_batch(
        {
            "facts": [{"id": 1, "fact_region": "eu", "fact_code": 7}],
            "dims": [{"region": "eu", "code": 7, "label": "EU-7"}],
        }
    )
    assert receipt.committed is True
    assert composite_db["facts"].count == 1
    assert composite_db["dims"].count == 1


def test_import_batch_empty(refs_db):
    receipt = refs_db.import_batch({})
    assert receipt.committed is True
    assert receipt.rows_written == {}
    assert receipt.resolved == ()


def test_foreign_key_violations_match_pragma_foreign_key_check(refs_db):
    insert_bad_row(refs_db, "authors", {"id": 1, "name": "A", "featured_book_id": 42})
    insert_bad_row(refs_db, "books", {"id": 1, "title": "B", "author_id": 7})
    refs_db["books"].insert({"id": 2, "title": "C", "author_id": 1})
    violations = refs_db._foreign_key_violations()
    pragma_rows = refs_db.execute("PRAGMA foreign_key_check").fetchall()
    assert len(violations) == len(pragma_rows) == 2
    assert {(v.table, v.values) for v in violations} == {
        ("authors", (42,)),
        ("books", (7,)),
    }
