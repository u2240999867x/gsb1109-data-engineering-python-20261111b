"""Tests for the optional operation-receipt machinery."""
import gc
import os
import sqlite3

import pytest

import sqlitedict
from sqlitedict import (
    OperationReceipt,
    OperationStatus,
    SqliteDict,
    SqliteDictError,
)
from accessories import norm_file


def _drop_table(db):
    db.conn.execute('DROP TABLE "%s"' % db.tablename)
    db.conn.commit(blocking=True)


def _recreate_table(db):
    db.conn.execute(
        'CREATE TABLE "%s" (key TEXT PRIMARY KEY, value BLOB)' % db.tablename
    )
    db.conn.commit(blocking=True)


class TestDefaultBehaviorUnchanged:
    def test_writes_return_none_without_receipts(self):
        db = SqliteDict()
        try:
            assert db.__setitem__('a', 1) is None
            assert db.update({'b': 2}) is None
            assert db.clear() is None
            assert db.commit() is None

            db['a'] = 1
            assert db.__delitem__('a') is None
            assert db.pending_receipts() == []
            assert db.all_receipts() == []
        finally:
            db.close()

    def test_mapping_interface_still_works(self):
        db = SqliteDict()
        try:
            db['a'] = 1
            db['b'] = 2
            db.commit()
            assert db['a'] == 1
            assert sorted(db) == ['a', 'b']
            assert len(db) == 2
            del db['a']
            assert 'a' not in db
        finally:
            db.close()

    def test_legacy_error_is_redelivered_to_next_call(self):
        db = SqliteDict()
        try:
            db.conn.execute('SELECT not_a_real_function()')
            with pytest.raises(sqlite3.OperationalError):
                db.conn.select_one('SELECT 1')
            # the legacy slot is consumed once, not raised again
            db.conn.check_raise_error()
        finally:
            db.close()


class TestReceiptHappyPath:
    def test_setitem_returns_succeeded_receipt(self):
        db = SqliteDict(receipts=True)
        try:
            receipt = db.__setitem__('a', 1)
            assert isinstance(receipt, OperationReceipt)
            assert receipt.operation == 'setitem'
            assert receipt.identity == 'a'
            assert receipt.wait() is receipt
            assert receipt.done
            assert receipt.status is OperationStatus.SUCCEEDED
            assert receipt.raise_for_status() is receipt
            assert receipt.error is None
        finally:
            db.close()

    def test_delitem_and_clear_receipts(self):
        db = SqliteDict(receipts=True)
        try:
            db['a'] = 1
            db.commit()
            receipt = db.__delitem__('a')
            assert receipt.operation == 'delitem'
            assert receipt.identity == 'a'
            receipt.raise_for_status()

            clear_receipt = db.clear()
            assert clear_receipt.operation == 'clear'
            assert clear_receipt.identity == db.tablename
            clear_receipt.raise_for_status()
        finally:
            db.close()

    def test_update_returns_one_receipt_per_item(self):
        db = SqliteDict(receipts=True)
        try:
            receipts = db.update([('a', 1), ('b', 2)], c=3)
            assert len(receipts) == 3
            assert [receipt.identity for receipt in receipts] == ['a', 'b', 'c']
            assert all(receipt.operation == 'update' for receipt in receipts)
            db.commit()
            for receipt in receipts:
                receipt.raise_for_status()
            assert db['a'] == 1 and db['b'] == 2 and db['c'] == 3
        finally:
            db.close()

    def test_commit_has_its_own_receipt(self):
        db = SqliteDict(receipts=True)
        try:
            db['a'] = 1
            receipt = db.commit()
            assert isinstance(receipt, OperationReceipt)
            assert receipt.operation == 'commit'
            assert receipt.identity == db.tablename
            receipt.raise_for_status()
        finally:
            db.close()

    def test_autocommit_receipt_is_durable(self):
        path = norm_file('tests/db/receipts-autocommit.sqlite')
        if os.path.exists(path):
            os.unlink(path)
        db = SqliteDict(path, autocommit=True, receipts=True)
        try:
            receipt = db.__setitem__('a', 1)
            receipt.raise_for_status()
        finally:
            db.close()

        other = SqliteDict(path)
        try:
            assert other['a'] == 1
        finally:
            other.close()
            os.unlink(path)

    def test_wait_timeout(self):
        receipt = OperationReceipt('a', 'setitem')
        assert not receipt.done
        assert receipt.wait(timeout=0.01) is None
        assert receipt.status is OperationStatus.PENDING


class TestReceiptFailureAttribution:
    def test_failure_keeps_identity_and_original_exception(self):
        db = SqliteDict(receipts=True)
        try:
            _drop_table(db)
            receipt = db.__setitem__('import-key', 1)
            receipt.wait()

            assert receipt.status is OperationStatus.FAILED
            assert receipt.done
            assert receipt.identity == 'import-key'
            assert receipt.operation == 'setitem'
            assert isinstance(receipt.error, sqlite3.OperationalError)

            with pytest.raises(SqliteDictError) as exc_info:
                receipt.raise_for_status()
            assert exc_info.value.receipt is receipt
            assert exc_info.value.__cause__ is receipt.error
        finally:
            db.close()

    def test_failed_receipt_does_not_poison_unrelated_requests(self):
        db = SqliteDict(receipts=True)
        try:
            _drop_table(db)
            failed = db.__setitem__('a', 1)
            failed.wait()
            assert failed.status is OperationStatus.FAILED

            # legacy shared exception slot must stay empty: no unrelated call
            # ever sees this error
            db.conn.check_raise_error()

            _recreate_table(db)
            ok = db.__setitem__('b', 2)
            ok.wait()
            assert ok.status is OperationStatus.SUCCEEDED
            ok.raise_for_status()

            # the original receipt keeps its own failure, nothing is re-delivered
            assert failed.status is OperationStatus.FAILED
            db.conn.check_raise_error()
            db.commit()
            assert db['b'] == 2
        finally:
            db.close()

    def test_update_failure_is_traced_to_the_failing_item(self):
        db = SqliteDict(receipts=True)
        try:
            receipts = db.update([('a', 1), ('b', 2)])
            db.commit()
            for receipt in receipts:
                receipt.raise_for_status()

            _drop_table(db)
            receipts = db.update([('good', 1), ('bad', 2)])
            db.commit()
            failed = [r for r in receipts if r.status is OperationStatus.FAILED]
            # every row was attempted against the missing table
            assert {r.identity for r in failed} == {'good', 'bad'}
            assert all(isinstance(r.error, sqlite3.OperationalError) for r in failed)
        finally:
            db.close()

    def test_commit_failure_is_reported_on_commit_receipt(self):
        db = SqliteDict(receipts=True)
        try:
            db['a'] = 1
            # an invalid statement exercises the same receipt-bound failure path
            receipt = OperationReceipt(db.tablename, 'commit')
            db.conn.execute('COMMIT_BROKEN_SYNTAX', receipt=receipt)
            receipt.wait()
            assert receipt.status is OperationStatus.FAILED
            assert isinstance(receipt.error, sqlite3.OperationalError)
            # again, the legacy slot is untouched
            db.conn.check_raise_error()
        finally:
            db.close()


class TestReceiptTerminalStateOnClose:
    def test_pending_receipt_is_cancelled_on_close(self):
        db = SqliteDict(receipts=True)
        receipt = OperationReceipt('queued-key', 'setitem')
        db.conn._track_receipt(receipt)
        assert receipt in db.pending_receipts()

        db.close()

        assert receipt.done
        assert receipt.status is OperationStatus.CANCELLED
        with pytest.raises(RuntimeError):
            receipt.raise_for_status()
        assert db.pending_receipts() == []

    def test_succeeded_receipts_keep_their_state_after_close(self):
        db = SqliteDict(receipts=True)
        receipt = db.__setitem__('a', 1)
        db.commit()
        receipt.wait()
        db.close()
        assert receipt.status is OperationStatus.SUCCEEDED
        receipt.raise_for_status()

    def test_force_close_also_terminates_receipts(self):
        db = SqliteDict(receipts=True)
        receipt = OperationReceipt('queued-key', 'setitem')
        db.conn._track_receipt(receipt)
        db.close(force=True)
        assert receipt.done
        assert receipt.status is OperationStatus.CANCELLED


class TestDiagnostics:
    def test_diagnostic_snapshots_run_no_sql(self):
        db = SqliteDict(receipts=True)
        try:
            first = db.__setitem__('a', 1)
            second = db.__setitem__('b', 2)
            db.commit()
            first.wait()
            second.wait()

            executed = []
            original_execute = db.conn.execute

            def spying_execute(req, *args, **kwargs):
                executed.append(req)
                return original_execute(req, *args, **kwargs)

            db.conn.execute = spying_execute
            db.pending_receipts()
            db.all_receipts()
            assert executed == []
        finally:
            db.close()

    def test_all_receipts_is_a_best_effort_snapshot(self):
        db = SqliteDict(receipts=True)
        try:
            kept = db.__setitem__('kept', 1)
            db.__setitem__('ephemeral', 2)
            db.commit()
            kept.wait()
            gc.collect()

            identities = {receipt.identity for receipt in db.all_receipts()}
            assert 'kept' in identities
            assert 'ephemeral' not in identities
            assert kept in db.all_receipts()
        finally:
            db.close()

    def test_repr_contains_identity_and_status(self):
        receipt = OperationReceipt('a', 'setitem')
        text = repr(receipt)
        assert 'setitem' in text and "'a'" in text and 'pending' in text
