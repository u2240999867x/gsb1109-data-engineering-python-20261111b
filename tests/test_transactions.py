import io

import pytest

import sqlparse
from sqlparse.transactions import (
    TransactionBatch,
    TransactionStreamReader,
    iter_transaction_batches,
)


class TrackingStream:
    """File-like object that records how often it was read."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.reads = 0

    def read(self, size=-1):
        if self.reads >= len(self._chunks):
            return ""
        chunk = self._chunks[self.reads]
        self.reads += 1
        return chunk


def statements_of(batches):
    return [batch.statements for batch in batches]


def test_committed_transaction_yields_single_batch():
    sql = "BEGIN;\nINSERT INTO t VALUES (1);\nCOMMIT;\n"
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.transactional
    assert batch.statements == ["INSERT INTO t VALUES (1);"]
    (span,) = batch.spans
    assert sql[span[0]:span[1]] == "INSERT INTO t VALUES (1);"
    assert sql[batch.start:batch.end] == sql.rstrip("\n")


def test_first_commit_delivered_before_stream_is_consumed():
    chunks = [
        "BEGIN;\nINSERT INTO t VALUES (1);\nCOMMIT;\n",
        "BEGIN;\nINSERT INTO t VALUES (2);\nCOMMIT;\n",
        "INSERT INTO t VALUES (3);\n",
    ]
    stream = TrackingStream(chunks)
    reader = iter_transaction_batches(stream)
    it = iter(reader)
    batch = next(it)
    assert batch.statements == ["INSERT INTO t VALUES (1);"]
    assert stream.reads == 1
    remaining = list(it)
    assert statements_of(remaining) == [
        ["INSERT INTO t VALUES (2);"],
        ["INSERT INTO t VALUES (3);"],
    ]
    assert stream.reads == len(chunks)


def test_early_stop_does_not_pull_more_source():
    chunks = [
        "BEGIN;\nINSERT INTO t VALUES (1);\nCOMMIT;\n",
        "BEGIN;\nINSERT INTO t VALUES (2);\nCOMMIT;\n",
    ]
    stream = TrackingStream(chunks)
    for _batch in iter_transaction_batches(stream):
        break
    assert stream.reads == 1


def test_standalone_dml_forms_its_own_batch():
    sql = "INSERT INTO t VALUES (1);\nUPDATE t SET a = 2;\n"
    batches = list(iter_transaction_batches(sql))
    assert len(batches) == 2
    assert all(not batch.transactional for batch in batches)
    assert batches[0].statements == ["INSERT INTO t VALUES (1);"]
    assert batches[1].statements == ["UPDATE t SET a = 2;"]
    for batch in batches:
        (span,) = batch.spans
        assert sql[span[0]:span[1]] == batch.statements[0]
        assert (batch.start, batch.end) == span


def test_fake_commits_inside_constructs_are_ignored():
    sql = (
        "BEGIN;\n"
        "INSERT INTO t VALUES ('COMMIT;');\n"
        "-- COMMIT;\n"
        "INSERT INTO t VALUES ($$COMMIT;$$);\n"
        "/* COMMIT; */\n"
        'INSERT INTO t VALUES ("COMMIT;");\n'
        "COMMIT;\n"
    )
    (batch,) = list(iter_transaction_batches(sql))
    assert len(batch.statements) == 3
    assert batch.statements[0] == "INSERT INTO t VALUES ('COMMIT;');"
    assert batch.statements[1].endswith("INSERT INTO t VALUES ($$COMMIT;$$);")
    assert batch.statements[2].endswith('INSERT INTO t VALUES ("COMMIT;");')


def test_constructs_spanning_chunk_boundaries():
    sql = (
        "BEGIN;\n"
        "INSERT INTO t VALUES ('co;mm''it');\n"
        "/* multi\nline ; comment */\n"
        "INSERT INTO t VALUES ($tag$begin;commit;$tag$);\n"
        "COMMIT;\n"
    )
    whole = list(iter_transaction_batches(sql))
    assert len(whole) == 1
    for size in (1, 2, 3, 5, 7, 13):
        chunked = list(iter_transaction_batches(sql, chunk_size=size))
        assert statements_of(chunked) == statements_of(whole)
        assert [b.spans for b in chunked] == [b.spans for b in whole]
        assert [(b.start, b.end) for b in chunked] == [
            (b.start, b.end) for b in whole
        ]


def test_chunk_iterable_source():
    chunks = ["BEGIN;\nINSERT INTO t", " VALUES (1);\nCOM", "MIT;\n"]
    (batch,) = list(iter_transaction_batches(chunks))
    assert batch.statements == ["INSERT INTO t VALUES (1);"]


def test_savepoint_rollback_excludes_statements():
    sql = (
        "BEGIN;\n"
        "INSERT INTO t VALUES (1);\n"
        "SAVEPOINT sp1;\n"
        "INSERT INTO t VALUES (2);\n"
        "ROLLBACK TO sp1;\n"
        "INSERT INTO t VALUES (3);\n"
        "COMMIT;\n"
    )
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.statements == [
        "INSERT INTO t VALUES (1);",
        "INSERT INTO t VALUES (3);",
    ]
    for stmt, (start, end) in zip(
            batch.statements, batch.spans, strict=True):
        assert sql[start:end] == stmt
    assert not any("VALUES (2)" in stmt for stmt in batch.statements)


def test_nested_savepoints_rollback_to_outer():
    sql = (
        "BEGIN;"
        "INSERT INTO t VALUES (1);"
        "SAVEPOINT a;"
        "INSERT INTO t VALUES (2);"
        "SAVEPOINT b;"
        "INSERT INTO t VALUES (3);"
        "ROLLBACK TO SAVEPOINT a;"
        "INSERT INTO t VALUES (4);"
        "COMMIT;"
    )
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.statements == [
        "INSERT INTO t VALUES (1);",
        "INSERT INTO t VALUES (4);",
    ]


def test_savepoint_survives_rollback_to_it():
    sql = (
        "BEGIN;"
        "SAVEPOINT sp1;"
        "INSERT INTO t VALUES (1);"
        "ROLLBACK TO sp1;"
        "INSERT INTO t VALUES (2);"
        "ROLLBACK TO sp1;"
        "INSERT INTO t VALUES (3);"
        "COMMIT;"
    )
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.statements == ["INSERT INTO t VALUES (3);"]


def test_quoted_savepoint_name():
    sql = (
        'BEGIN;SAVEPOINT "My Sp";INSERT INTO t VALUES (1);'
        'ROLLBACK TO "My Sp";INSERT INTO t VALUES (2);COMMIT;'
    )
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.statements == ["INSERT INTO t VALUES (2);"]


def test_full_rollback_discards_transaction():
    sql = (
        "BEGIN;\nINSERT INTO t VALUES (1);\nROLLBACK;\n"
        "INSERT INTO t VALUES (2);\n"
    )
    batches = list(iter_transaction_batches(sql))
    assert statements_of(batches) == [["INSERT INTO t VALUES (2);"]]
    assert not batches[0].transactional


def test_eof_with_unclosed_transaction_keeps_batch_start():
    sql = "INSERT INTO t VALUES (0);\nBEGIN;\nINSERT INTO t VALUES (1);\n"
    reader = iter_transaction_batches(sql)
    batches = list(reader)
    assert statements_of(batches) == [["INSERT INTO t VALUES (0);"]]
    assert reader.pending_offset == sql.index("BEGIN")
    assert sql[reader.pending_offset:] == (
        "BEGIN;\nINSERT INTO t VALUES (1);\n")


def test_pending_offset_is_none_on_clean_eof():
    reader = iter_transaction_batches("BEGIN; INSERT INTO t VALUES (1); COMMIT;")
    assert len(list(reader)) == 1
    assert reader.pending_offset is None


def test_binary_stream_reports_byte_offsets():
    sql = "BEGIN;\nINSERT INTO t VALUES ('héllo中文');\nCOMMIT;\n"
    data = sql.encode("utf-8")
    reader = iter_transaction_batches(
        io.BytesIO(data), encoding="utf-8", chunk_size=7)
    (batch,) = list(reader)
    (span,) = batch.spans
    assert data[span[0]:span[1]].decode("utf-8") == batch.statements[0]
    assert data[batch.start:batch.end] == data.rstrip(b"\n")


def test_bytes_input_defaults_to_utf8():
    sql = "INSERT INTO t VALUES ('héllo');\n"
    data = sql.encode("utf-8")
    (batch,) = list(iter_transaction_batches(data, chunk_size=3))
    (span,) = batch.spans
    assert data[span[0]:span[1]].decode("utf-8") == batch.statements[0]


def test_rollback_to_unknown_savepoint_raises():
    with pytest.raises(ValueError):
        list(iter_transaction_batches("BEGIN; ROLLBACK TO nope;"))


def test_comment_only_and_empty_statements_are_skipped():
    sql = "-- just a comment;\n;\nINSERT INTO t VALUES (1);/* done */;\n"
    batches = list(iter_transaction_batches(sql))
    assert len(batches) == 1
    assert batches[0].statements[0].endswith("INSERT INTO t VALUES (1);")


def test_stray_control_statements_are_ignored():
    sql = "COMMIT; INSERT INTO t VALUES (1); ROLLBACK;"
    batches = list(iter_transaction_batches(sql))
    assert statements_of(batches) == [["INSERT INTO t VALUES (1);"]]


def test_nested_begin_is_tolerated():
    sql = ("BEGIN; INSERT INTO t VALUES (1); BEGIN; "
           "INSERT INTO t VALUES (2); COMMIT;")
    (batch,) = list(iter_transaction_batches(sql))
    assert batch.statements == [
        "INSERT INTO t VALUES (1);",
        "INSERT INTO t VALUES (2);",
    ]


def test_mixed_sequence_of_batches():
    sql = (
        "INSERT INTO t VALUES (1);\n"
        "BEGIN;\nINSERT INTO t VALUES (2);\nCOMMIT;\n"
        "BEGIN;\nINSERT INTO t VALUES (3);\nROLLBACK;\n"
        "UPDATE t SET a = 4;\n"
    )
    batches = list(iter_transaction_batches(sql))
    assert statements_of(batches) == [
        ["INSERT INTO t VALUES (1);"],
        ["INSERT INTO t VALUES (2);"],
        ["UPDATE t SET a = 4;"],
    ]
    assert [b.transactional for b in batches] == [False, True, False]


def test_module_is_exposed_at_top_level():
    assert sqlparse.transactions.TransactionBatch is TransactionBatch
    reader = sqlparse.transactions.iter_transaction_batches("SELECT 1;")
    assert isinstance(reader, TransactionStreamReader)
    (batch,) = list(reader)
    assert isinstance(batch, TransactionBatch)
