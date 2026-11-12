"""
Tests for the resumable batch cleanup helper and the duplicate-ID
removal regression.
"""

from pathlib import Path

import pytest

from tinydb import TinyDB, Cleanup, CleanupResult, JSONStorage
from tinydb.storages import MemoryStorage, Storage
from tinydb.table import Document
from tinydb.cleanup import (
    CLEANUP_STATE_TABLE,
    STATUS_MISSING,
    STATUS_REMOVED,
    STATUS_SKIPPED,
)


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=['memory', 'json'])
def db(request, tmp_path: Path) -> TinyDB:
    if request.param == 'json':
        database = TinyDB(tmp_path / 'cleanup.db', storage=JSONStorage)
    else:
        database = TinyDB(storage=MemoryStorage)
    database.drop_tables()
    return database


class FailingStorage(Storage):
    """MemoryStorage wrapper that fails the Nth write.

    TinyDB instantiates the storage class itself, so the failing write
    index is configured through a class attribute. Data is copied on every
    read and successful write so a failed ``write`` really leaves the
    previously committed state untouched -- the same commit boundary
    JSONStorage provides via json load/dump.
    """

    fail_on = 2

    def __init__(self):
        import copy

        self._copy = copy.deepcopy
        self._committed = None
        self.write_calls = 0

    def read(self):
        return self._copy(self._committed)

    def write(self, data):
        self.write_calls += 1
        if self.write_calls == self.fail_on:
            raise RuntimeError('simulated write failure')
        self._committed = self._copy(data)

    def close(self):
        pass


def failing_storage(fail_on: int):
    """Return a zero-argument storage class failing on the given write."""
    return type(
        'ConfiguredFailingStorage',
        (FailingStorage,),
        {'fail_on': fail_on},
    )


class CountingStorage(Storage):
    """MemoryStorage wrapper that counts committed writes."""

    def __init__(self):
        self._inner = MemoryStorage()
        self.write_count = 0

    def read(self):
        return self._inner.read()

    def write(self, data):
        self.write_count += 1
        self._inner.write(data)

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Regression: duplicate doc_ids must not raise KeyError
# ---------------------------------------------------------------------------


def test_remove_duplicate_doc_ids_does_not_raise(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(3))

    removed = db.remove(doc_ids=[1, 1, 2, 2, 2])

    assert sorted(removed) == [1, 2]
    assert db.get(doc_id=1) is None
    assert db.get(doc_id=2) is None
    assert db.get(doc_id=3) is not None


def test_remove_duplicate_mixed_with_missing(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(2))

    assert sorted(db.remove(doc_ids=[1, 1, 99, 99, 2])) == [1, 2]
    assert len(db) == 0


# ---------------------------------------------------------------------------
# Per-source-position results
# ---------------------------------------------------------------------------


def test_every_source_position_has_a_result(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(3))

    result = Cleanup(db, [1, 1, 2, 99], batch_size=10, cleanup_id='c1').run()

    assert isinstance(result, CleanupResult)
    assert result.done is True
    assert [entry['position'] for entry in result.results] == [0, 1, 2, 3]
    assert [entry['doc_id'] for entry in result.results] == [1, 1, 2, 99]
    assert [entry['status'] for entry in result.results] == [
        STATUS_REMOVED,
        STATUS_REMOVED,
        STATUS_REMOVED,
        STATUS_MISSING,
    ]
    assert result.removed_ids == [1, 2]
    assert result.missing_ids == [99]
    assert result.skipped_ids == []
    assert len(db) == 1


def test_duplicate_id_is_associated_with_first_request(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(2))

    result = Cleanup(db, [1, 1], batch_size=1, cleanup_id='dup').run()

    # The second occurrence must inherit the first request's verdict even
    # though the document is gone by the time the second batch runs.
    assert all(entry['status'] == STATUS_REMOVED for entry in result.results)
    assert result.missing_ids == []


# ---------------------------------------------------------------------------
# Batching, interruption and resumability
# ---------------------------------------------------------------------------


def test_run_batch_commits_one_write_per_batch():
    database = TinyDB(storage=CountingStorage)
    database.insert_multiple({'i': i} for i in range(5))
    cleanup = Cleanup(database, [1, 2, 3, 4, 5], batch_size=2,
                      cleanup_id='b')

    writes_before = database.storage.write_count
    first = cleanup.run_batch()
    assert first.done is False
    assert database.storage.write_count - writes_before == 1
    assert [entry['position'] for entry in first.results] == [0, 1]
    assert len(database) == 3

    second = cleanup.run_batch()
    assert [entry['position'] for entry in second.results] == [0, 1, 2, 3]
    assert len(database) == 1

    third = cleanup.run_batch()
    assert third.done is True
    assert len(database) == 0

    # A finished cleanup has nothing left to commit.
    assert cleanup.run_batch() is None
    assert cleanup.is_done()
    database.close()


def test_resume_continues_with_original_results(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(4))
    cleanup = Cleanup(db, [1, 2, 3, 4], batch_size=2, cleanup_id='r')

    cleanup.run_batch()
    assert sorted(doc.doc_id for doc in db.all()) == [3, 4]

    # Simulate an interruption: a fresh Cleanup instance resumes the same
    # cleanup run using its persisted recovery state.
    resumed = Cleanup(db, [1, 2, 3, 4], batch_size=2, cleanup_id='r')
    assert resumed.is_done() is False
    result = resumed.run()

    assert result.done is True
    assert len(result.results) == 4
    assert result.removed_ids == [1, 2, 3, 4]
    assert result.missing_ids == []
    assert len(db) == 0


def test_retry_does_not_report_deleted_docs_as_never_existed(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(3))
    cleanup = Cleanup(db, [1, 2, 3], batch_size=1, cleanup_id='retry')

    cleanup.run_batch()
    assert db.get(doc_id=1) is None

    # From the live table's point of view doc 1 no longer exists. The
    # resumed cleanup must still honour the original 'removed' verdict.
    result = Cleanup(db, [1, 2, 3], batch_size=1, cleanup_id='retry').run()

    statuses = {
        entry['doc_id']: entry['status'] for entry in result.results
    }
    assert statuses == {
        1: STATUS_REMOVED,
        2: STATUS_REMOVED,
        3: STATUS_REMOVED,
    }
    assert result.missing_ids == []


def test_resume_with_different_request_raises(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(3))
    Cleanup(db, [1, 2], cleanup_id='same').run()

    with pytest.raises(ValueError, match='different list of document IDs'):
        Cleanup(db, [1, 3], cleanup_id='same').run()


def test_failed_write_leaves_consistent_state():
    # The second cleanup batch's write fails. Neither its deletions nor
    # its state update may be visible, and a retry picks up from the
    # committed point. The initial insert_multiple is write #1, the first
    # cleanup batch is write #2, so the failing one is write #3.
    database = TinyDB(storage=failing_storage(fail_on=3))
    database.insert_multiple({'i': i} for i in range(4))

    cleanup = Cleanup(database, [1, 2, 3, 4], batch_size=2, cleanup_id='f')
    cleanup.run_batch()

    with pytest.raises(RuntimeError, match='simulated write failure'):
        cleanup.run_batch()

    # First batch committed; second batch left no trace.
    assert sorted(doc.doc_id for doc in database.all()) == [3, 4]
    persisted = database.storage.read()[CLEANUP_STATE_TABLE]
    state = persisted['_default::f']
    assert state['cursor'] == 2
    assert len(state['results']) == 2

    result = Cleanup(database, [1, 2, 3, 4], batch_size=2,
                     cleanup_id='f').run()
    assert result.done is True
    assert result.removed_ids == [1, 2, 3, 4]
    assert len(database) == 0
    database.close()


# ---------------------------------------------------------------------------
# Explicit rebuild between batches must be protected
# ---------------------------------------------------------------------------


def test_rebuilt_document_between_batches_is_not_deleted(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(2))
    cleanup = Cleanup(db, [1, 1], batch_size=1, cleanup_id='rebuild')

    cleanup.run_batch()
    assert db.get(doc_id=1) is None

    # The same ID is explicitly re-created as a new version.
    db.insert(Document({'i': 999, 'version': 2}, doc_id=1))
    assert db.get(doc_id=1) is not None

    result = Cleanup(db, [1, 1], batch_size=1, cleanup_id='rebuild').run()

    assert result.results[0]['status'] == STATUS_REMOVED
    assert result.results[1]['status'] == STATUS_SKIPPED
    assert result.skipped_ids == [1]
    # The rebuilt document survives the old cleanup run.
    rebuilt = db.get(doc_id=1)
    assert rebuilt is not None
    assert rebuilt['version'] == 2


def test_rebuild_then_remove_again_keeps_both_verdicts(db: TinyDB):
    db.insert({'v': 1})
    cleanup = Cleanup(db, [1, 1, 1], batch_size=1, cleanup_id='rb2')

    cleanup.run_batch()                      # removes v1
    db.insert(Document({'v': 2}, doc_id=1))  # explicit rebuild
    cleanup.run_batch()                      # must skip the rebuild
    db.remove(doc_ids=[1])                   # caller deletes rebuild itself
    final = cleanup.run_batch()              # nothing to delete now

    statuses = [entry['status'] for entry in final.results]
    assert statuses == [
        STATUS_REMOVED,
        STATUS_SKIPPED,
        STATUS_SKIPPED,
    ]


# ---------------------------------------------------------------------------
# Persistence across database reopens (JSON storage)
# ---------------------------------------------------------------------------


def test_json_storage_resume_across_reopen(tmp_path: Path):
    path = tmp_path / 'persistent.db'

    database = TinyDB(path, storage=JSONStorage)
    database.insert_multiple({'i': i} for i in range(4))
    Cleanup(
        database, [1, 2, 3, 4], batch_size=2, cleanup_id='persist'
    ).run_batch()
    database.close()

    reopened = TinyDB(path, storage=JSONStorage)
    assert sorted(doc.doc_id for doc in reopened.all()) == [3, 4]

    result = Cleanup(reopened, [1, 2, 3, 4], batch_size=2,
                     cleanup_id='persist').run()
    assert result.done is True
    assert result.removed_ids == [1, 2, 3, 4]
    assert len(reopened) == 0
    reopened.close()


def test_state_lives_next_to_user_data_in_storage(db: TinyDB):
    db.insert({'i': 1})
    Cleanup(db, [1], cleanup_id='meta').run()

    raw = db.storage.read()
    assert CLEANUP_STATE_TABLE in raw
    assert '_default::meta' in raw[CLEANUP_STATE_TABLE]


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def test_empty_request_completes_with_one_entry_per_position(db: TinyDB):
    result = Cleanup(db, [], cleanup_id='empty').run()
    assert result.done is True
    assert result.results == []
    assert Cleanup(db, [], cleanup_id='empty').is_done()


def test_invalid_batch_size(db: TinyDB):
    with pytest.raises(ValueError, match='batch_size'):
        Cleanup(db, [1], batch_size=0)


def test_result_before_start_raises(db: TinyDB):
    with pytest.raises(KeyError):
        Cleanup(db, [1], cleanup_id='never').result()


def test_discard_removes_state_only(db: TinyDB):
    db.insert_multiple({'i': i} for i in range(2))
    cleanup = Cleanup(db, [1, 2], cleanup_id='d')
    cleanup.run()
    assert len(db) == 0

    cleanup.discard()
    assert CLEANUP_STATE_TABLE not in db.storage.read()

    # A new run with the same ID starts from scratch; nothing is left to
    # delete, so every position is now reported as missing.
    result = Cleanup(db, [1, 2], cleanup_id='d').run()
    assert result.missing_ids == [1, 2]


def test_cleanup_on_named_table_does_not_touch_default(db: TinyDB):
    db.insert({'where': 'default'})
    other = db.table('other')
    other.insert_multiple({'i': i} for i in range(2))

    Cleanup(other, [1, 2], batch_size=1, cleanup_id='t').run()

    assert len(other) == 0
    assert len(db) == 1
    assert 'other::t' in db.storage.read()[CLEANUP_STATE_TABLE]
