import pytest

from sqlite_utils import Database
from sqlite_utils.db import ImportReport, ImportStatus, RowResult
from sqlite_utils.utils import OperationalError


def test_import_all_returns_report_and_table_still_writable(fresh_db):
    table = fresh_db.table("items")
    report = table.import_all([{"id": 1, "name": "Cleo"}], pk="id")
    assert isinstance(report, ImportReport)
    assert table.last_import_report is report
    assert list(table.rows) == [{"id": 1, "name": "Cleo"}]


def test_report_totals_and_status_constants(fresh_db):
    table = fresh_db.table("items")
    table.insert({"id": 1, "name": "exists"}, pk="id")
    report = table.import_all(
        [
            {"id": 1, "name": "ignored"},
            {"id": 2, "name": "new"},
        ],
        pk="id",
        ignore=True,
    )
    assert report.totals == {
        ImportStatus.INSERTED: 1,
        ImportStatus.UPDATED: 0,
        ImportStatus.IGNORED: 1,
        ImportStatus.FAILED: 0,
        ImportStatus.UNKNOWN: 0,
    }
    assert report.inserted == 1
    assert report.ignored == 1
    assert report.updated == 0
    assert report.failed == 0
    assert report.unknown == 0


def test_every_input_gets_a_result_in_order_with_global_index(fresh_db):
    table = fresh_db.table("items")
    records = [{"id": i, "name": str(i)} for i in range(5)]
    report = table.import_all(records, pk="id", batch_size=2)
    assert [row.index for row in report.rows] == [0, 1, 2, 3, 4]
    assert all(isinstance(row, RowResult) for row in report.rows)
    assert all(row.inserted for row in report.rows)
    # Three batches: sizes 2, 2, 1, all committed
    assert [(batch.index, batch.size, batch.committed) for batch in report.batches] == [
        (0, 2, True),
        (1, 2, True),
        (2, 1, True),
    ]


def test_ignore_distinguishes_existing_and_duplicate_business_keys(fresh_db):
    table = fresh_db.table("items")
    table.insert({"id": 1, "name": "existing"}, pk="id")
    report = table.import_all(
        [
            {"id": 1, "name": "conflicts with stored row"},
            {"id": 2, "name": "first occurrence"},
            {"id": 2, "name": "duplicate inside the import"},
        ],
        pk="id",
        ignore=True,
        batch_size=1,
    )
    statuses = [(row.index, row.status, row.key) for row in report.rows]
    assert statuses == [
        (0, ImportStatus.IGNORED, 1),
        (1, ImportStatus.INSERTED, 2),
        (2, ImportStatus.IGNORED, 2),
    ]
    # Each occurrence of the duplicate key is accounted for separately
    assert report.rows[0].detail == "a row with this business key already exists"
    # With batch_size=1 the earlier occurrence already committed, so the later
    # one conflicts with a stored row rather than an uncommitted sibling
    assert report.rows[2].detail == "a row with this business key already exists"
    # The ignored inputs never overwrote stored data
    assert list(table.rows_where("id = 1")) == [{"id": 1, "name": "existing"}]
    assert list(table.rows_where("id = 2")) == [{"id": 2, "name": "first occurrence"}]


def test_ignore_duplicate_within_single_batch(fresh_db):
    table = fresh_db.table("items")
    report = table.import_all(
        [
            {"id": 5, "name": "a"},
            {"id": 5, "name": "b"},
            {"id": 6, "name": "c"},
        ],
        pk="id",
        ignore=True,
        batch_size=10,
    )
    assert [row.status for row in report.rows] == [
        ImportStatus.INSERTED,
        ImportStatus.IGNORED,
        ImportStatus.INSERTED,
    ]
    # The second occurrence loses to its sibling inside the same batch, before
    # either is committed
    assert report.rows[1].detail == (
        "duplicate business key within the import; an earlier occurrence was "
        "written instead"
    )


def test_upsert_distinguishes_inserted_from_updated(fresh_db):
    table = fresh_db.table("items")
    table.insert({"id": 1, "v": 1}, pk="id")
    report = table.import_upsert_all(
        [
            {"id": 1, "v": 2},
            {"id": 2, "v": 9},
            {"id": 2, "v": 10},
        ],
        pk="id",
        batch_size=1,
    )
    assert [(row.index, row.status, row.key) for row in report.rows] == [
        (0, ImportStatus.UPDATED, 1),
        (1, ImportStatus.INSERTED, 2),
        (2, ImportStatus.UPDATED, 2),
    ]
    assert report.inserted == 1
    assert report.updated == 2
    assert list(table.rows_where("id = 2")) == [{"id": 2, "v": 10}]


def test_replace_reports_updated_for_existing_keys(fresh_db):
    table = fresh_db.table("items")
    table.insert({"id": 1, "v": "old"}, pk="id")
    report = table.import_all(
        [{"id": 1, "v": "new"}, {"id": 2, "v": "brand new"}],
        pk="id",
        replace=True,
    )
    assert [row.status for row in report.rows] == [
        ImportStatus.UPDATED,
        ImportStatus.INSERTED,
    ]


def test_compound_primary_key_attribution(fresh_db):
    table = fresh_db.table("items")
    table.insert_all([{"a": 1, "b": 2, "v": "first"}], pk=("a", "b"))
    report = table.import_upsert_all(
        [
            {"a": 1, "b": 2, "v": "second"},
            {"a": 3, "b": 4, "v": "new"},
        ],
        pk=("a", "b"),
    )
    assert [row.key for row in report.rows] == [(1, 2), (3, 4)]
    assert [row.status for row in report.rows] == [
        ImportStatus.UPDATED,
        ImportStatus.INSERTED,
    ]


def test_hash_id_attribution(fresh_db):
    table = fresh_db.table("dogs")
    table.insert({"name": "Cleo"}, hash_id="id")
    stored_id = next(iter(table.rows))["id"]
    report = table.import_all(
        [{"name": "Cleo"}, {"name": "Pancakes"}],
        hash_id="id",
        ignore=True,
    )
    assert [row.status for row in report.rows] == [
        ImportStatus.IGNORED,
        ImportStatus.INSERTED,
    ]
    assert report.rows[0].key == stored_id


def test_list_mode_attribution(fresh_db):
    table = fresh_db.table("items")
    table.insert_all([["id", "v"], [1, "old"]], pk="id")
    report = table.import_all(
        [["id", "v"], [1, "new"], [2, "fresh"]],
        pk="id",
        ignore=True,
    )
    assert [(row.status, row.key) for row in report.rows] == [
        (ImportStatus.IGNORED, 1),
        (ImportStatus.INSERTED, 2),
    ]


def test_rowid_table_without_business_key_is_unknown(fresh_db):
    table = fresh_db.table("items")
    report = table.import_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    assert report.unknown == 2
    assert all(row.status == ImportStatus.UNKNOWN for row in report.rows)
    assert all(row.key is None for row in report.rows)
    assert table.count == 2


def test_unique_column_conflict_without_pk_is_unknown(fresh_db):
    # A UNIQUE column is not the declared primary key, so the row change
    # cannot be reliably attributed to an input
    fresh_db.execute("create table u (title text unique)")
    table = fresh_db.table("u")
    table.insert({"title": "x"})
    report = table.import_all(
        [{"title": "x"}, {"title": "y"}], ignore=True
    )
    assert all(row.status == ImportStatus.UNKNOWN for row in report.rows)
    assert table.count == 2


def test_trigger_changes_are_marked_unknown(fresh_db):
    table = fresh_db.table("items")
    table.create({"id": int, "v": str}, pk="id")
    fresh_db.execute("create table audit_log (id integer primary key, note text)")
    fresh_db.execute(
        """
        create trigger items_ai after insert on items
        begin
            insert into audit_log (note) values ('inserted ' || new.id);
        end
        """
    )
    report = table.import_all(
        [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}], pk="id"
    )
    assert all(row.status == ImportStatus.UNKNOWN for row in report.rows)
    assert all("trigger" in (row.detail or "") for row in report.rows)
    # The inputs were still written; only the attribution is uncertain
    assert table.count == 2


def test_failed_batch_marks_commit_and_unexecuted_batches(fresh_db):
    table = fresh_db.table("items")
    table.create({"id": int, "v": str}, pk="id", not_null={"v"})
    table.insert({"id": 1, "v": "seed"})

    def records():
        yield {"id": 2, "v": "ok"}
        yield {"id": 3, "v": None}  # NOT NULL failure, batch_size=1
        yield {"id": 4, "v": "never executed"}

    with pytest.raises(Exception, match="NOT NULL"):
        table.import_all(records(), pk="id", batch_size=1)

    report = table.last_import_report
    assert [(batch.index, batch.committed, batch.size) for batch in report.batches] == [
        (0, True, 1),
        (1, False, 1),
        (2, False, 1),
    ]
    assert len(report.committed_batches) == 1
    assert len(report.errored_batches) == 1
    assert len(report.failed_batches) == 2
    assert len(report.unexecuted_batches) == 1
    assert report.errored_batches[0].index == 1
    assert report.unexecuted_batches[0].index == 2

    failed_row = report.rows[1]
    assert failed_row.status == ImportStatus.FAILED
    assert failed_row.error is not None
    unexecuted_row = report.rows[2]
    assert unexecuted_row.status == ImportStatus.FAILED
    assert unexecuted_row.error is None
    assert "never executed" in (unexecuted_row.detail or "")

    # The committed batch persisted; the failed and unexecuted rows did not
    assert [row["id"] for row in table.rows] == [1, 2]


def test_failed_multi_record_batch_attributes_failure_to_every_input(fresh_db):
    table = fresh_db.table("items")
    table.create({"id": int, "v": str}, pk="id")
    table.insert({"id": 1, "v": "seed"})
    # Both records share id=1 inside one batch: the whole batch rolls back
    with pytest.raises(Exception):
        table.import_all(
            [{"id": 1, "v": "a"}, {"id": 1, "v": "b"}],
            pk="id",
            batch_size=10,
        )
    report = table.last_import_report
    assert len(report.batches) == 1
    assert report.batches[0].committed is False
    assert all(row.status == ImportStatus.FAILED for row in report.rows)
    assert all(row.error is not None for row in report.rows)
    assert table.count == 1


def test_generator_is_consumed_exactly_once_on_failure(fresh_db):
    table = fresh_db.table("items")
    table.create({"id": int, "v": str}, pk="id", not_null={"v"})
    consumed = []

    def records():
        for i in range(4):
            consumed.append(i)
            yield {"id": i, "v": None if i == 2 else str(i)}

    with pytest.raises(Exception):
        table.import_all(records(), pk="id", batch_size=1)
    # Every input was read once, even though the last batch never executed
    assert consumed == [0, 1, 2, 3]
    # The unexecuted tail still appears in the report with global indices
    assert [row.index for row in table.last_import_report.rows] == [0, 1, 2, 3]


def test_non_audit_insert_all_returns_table_and_empty_report(fresh_db):
    table = fresh_db.table("items")
    result = table.insert_all([{"id": 1}], pk="id")
    assert result is table
    assert isinstance(table.last_import_report, ImportReport)
    assert table.last_import_report.totals == {
        ImportStatus.INSERTED: 0,
        ImportStatus.UPDATED: 0,
        ImportStatus.IGNORED: 0,
        ImportStatus.FAILED: 0,
        ImportStatus.UNKNOWN: 0,
    }


def test_empty_input_returns_empty_report(fresh_db):
    report = fresh_db.table("items").import_all([], pk="id")
    assert report.rows == []
    assert report.batches == []
    assert report.totals[ImportStatus.INSERTED] == 0


@pytest.mark.parametrize("use_old_upsert", (False, True))
def test_audit_works_with_old_upsert_path(use_old_upsert):
    db = Database(memory=True, use_old_upsert=use_old_upsert)
    table = db.table("items")
    table.insert({"id": 1, "v": 1}, pk="id")
    report = table.import_upsert_all(
        [{"id": 1, "v": 2}, {"id": 2, "v": 3}], pk="id"
    )
    assert [row.status for row in report.rows] == [
        ImportStatus.UPDATED,
        ImportStatus.INSERTED,
    ]


def test_audit_survives_too_many_variables_split(fresh_db, monkeypatch):
    # Force the "too many SQL variables" recovery path: the first INSERT in a
    # batch raises, after which the batch is retried as two halves. The audit
    # indices and classifications must still line up with the original input.
    table = fresh_db.table("items")
    real_execute = fresh_db.execute
    raised = {"done": False}

    def flaky_execute(sql, parameters=None):
        if not raised["done"] and sql.lstrip().upper().startswith("INSERT"):
            raised["done"] = True
            raise OperationalError("too many SQL variables")
        return real_execute(sql, parameters)

    monkeypatch.setattr(fresh_db, "execute", flaky_execute)
    report = table.import_all(
        [{"id": i, "v": i} for i in range(4)],
        pk="id",
        batch_size=4,
    )
    assert raised["done"] is True
    assert [row.index for row in report.rows] == [0, 1, 2, 3]
    assert [row.key for row in report.rows] == [0, 1, 2, 3]
    assert all(row.status == ImportStatus.INSERTED for row in report.rows)
    assert all(batch.committed for batch in report.batches)
    assert table.count == 4
