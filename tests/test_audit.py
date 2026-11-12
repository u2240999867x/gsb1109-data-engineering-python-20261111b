"""Tests for commit-boundary audit snapshots.

An audit snapshot is an atomic backup of the whole database file taken at a
specific confirmed commit.  The tests pin down:

* the snapshot of commit N contains exactly the values visible at commit N,
  even when batch N+1 is already queued behind the commit;
* every reference table in the copied file is listed with its row count;
* readers only ever observe a previous complete copy or the new complete copy;
* a boundary that was already confirmed before subscribing is reported as
  unrecoverable -- the live database is never relabeled as an old commit;
* a failed snapshot is retriable at the next boundary.
"""
import io
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest

from sqlitedict import SqliteDict, AuditSnapshotError, AuditSnapshotHandle
from accessories import norm_file


class AuditSnapshotTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix='sqlitedict-audit-')
        self.source = norm_file(os.path.join(self.folder, 'source.sqlite'))
        self.backup = os.path.join(self.folder, 'copies', 'audit.sqlite')

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_commit_sequence_starts_at_zero_and_counts_user_commits(self):
        db = SqliteDict(self.source)
        self.assertEqual(db.commit_seq, 0)
        db['a'] = 1
        db.commit()
        self.assertEqual(db.commit_seq, 1)
        db['a'] = 2
        db.commit()
        self.assertEqual(db.commit_seq, 2)
        db.close()

    def test_snapshot_contains_all_tables_with_row_counts(self):
        db = SqliteDict(self.source, tablename='orders')
        other = SqliteDict(self.source, tablename='customers')

        # Populate and fully flush the second reference table first so the
        # snapshot boundary is defined solely by the subscribing connection.
        other['c1'] = 'acme'
        other.commit()

        handle = db.subscribe_audit_snapshot(1, self.backup)
        self.assertIsInstance(handle, AuditSnapshotHandle)
        self.assertFalse(handle.ready)
        self.assertEqual(handle.commit_seq, 1)

        db.update([('o1', 100), ('o2', 200)])
        db.commit()

        manifest = handle.wait(5)
        self.assertTrue(handle.ready)
        self.assertEqual(manifest['commit_seq'], 1)
        self.assertEqual(manifest['backup'], os.path.abspath(self.backup))
        self.assertEqual(manifest['source'], os.path.abspath(self.source))
        # Both reference tables live in the same file and both are reported.
        self.assertEqual(manifest['tables'], {'customers': 1, 'orders': 2})

        with io.open(handle.manifest_path) as manifest_file:
            on_disk = json.load(manifest_file)
        self.assertEqual(on_disk, manifest)

        # The copied database is independently readable and holds the data.
        with SqliteDict(self.backup, tablename='orders', flag='r') as copy:
            self.assertEqual(dict(copy.items()), {'o1': 100, 'o2': 200})
        with SqliteDict(self.backup, tablename='customers', flag='r') as copy:
            self.assertEqual(copy['c1'], 'acme')

        # The manifest is embedded in the copy itself, so the file is
        # self-describing even without the JSON sidecar.
        with sqlite3.connect(self.backup) as connection:
            rows = dict(connection.execute(
                'SELECT key, value FROM "sqlitedict_audit_manifest"'
            ).fetchall())
        self.assertEqual(rows['commit_seq'], '1')
        self.assertEqual(json.loads(rows['tables']),
                         {'customers': 1, 'orders': 2})

        db.close()
        other.close()

    def test_snapshot_pins_boundary_while_next_batch_is_queued(self):
        db = SqliteDict(self.source)
        db['k'] = 'first'
        db.commit()  # commit 1

        handle = db.subscribe_audit_snapshot(2, self.backup)

        db['k'] = 'second'
        db.commit(blocking=False)   # commit 2, queued but not yet awaited
        db['k'] = 'third'           # batch 3 already accepted by the dict
        db.commit(blocking=True)    # commit 3 confirms everything behind it

        manifest = handle.wait(5)
        self.assertEqual(manifest['commit_seq'], 2)
        with SqliteDict(self.backup, flag='r') as copy:
            # Boundary 2 must show the value of commit 2, not commit 3.
            self.assertEqual(copy['k'], 'second')
        self.assertEqual(db['k'], 'third')
        self.assertEqual(db.commit_seq, 3)
        db.close()

    def test_two_consecutive_commits_same_key_visible_values_and_labels(self):
        db = SqliteDict(self.source)

        first_backup = os.path.join(self.folder, 'first.sqlite')
        first_handle = db.subscribe_audit_snapshot(1, first_backup)
        db['shared_key'] = 'value-at-1'
        db.commit()
        first_manifest = first_handle.wait(5)

        second_backup = os.path.join(self.folder, 'second.sqlite')
        second_handle = db.subscribe_audit_snapshot(2, second_backup)
        db['shared_key'] = 'value-at-2'
        db.commit()
        second_manifest = second_handle.wait(5)

        self.assertEqual(first_manifest['commit_seq'], 1)
        self.assertEqual(second_manifest['commit_seq'], 2)
        with SqliteDict(first_backup, flag='r') as copy:
            self.assertEqual(copy['shared_key'], 'value-at-1')
        with SqliteDict(second_backup, flag='r') as copy:
            self.assertEqual(copy['shared_key'], 'value-at-2')
        self.assertEqual(db['shared_key'], 'value-at-2')
        self.assertEqual(db.commit_seq, 2)
        db.close()

    def test_missed_boundary_is_unrecoverable(self):
        db = SqliteDict(self.source)
        db['k'] = 'v1'
        db.commit()  # commit 1 is already confirmed

        with self.assertRaises(AuditSnapshotError) as context:
            db.subscribe_audit_snapshot(1, self.backup)
        message = str(context.exception)
        self.assertIn('no longer recoverable', message)
        self.assertIn('commit 1', message)
        # The current database must not be silently labeled as commit 1:
        # no backup file is produced.
        self.assertFalse(os.path.exists(self.backup))
        db.close()

    def test_subscribe_queued_after_commit_still_detects_missed_boundary(self):
        # The commit request and the subscription share one FIFO queue, so a
        # commit queued ahead of the subscription is honestly observed.
        db = SqliteDict(self.source)
        db['k'] = 'v1'
        db.commit(blocking=False)  # queued, not yet confirmed from our view
        with self.assertRaises(AuditSnapshotError):
            db.subscribe_audit_snapshot(1, self.backup)
        db.close()

    def test_failed_snapshot_can_be_retried_at_next_boundary(self):
        db = SqliteDict(self.source)

        # A file blocks creation of the destination directory -> backup fails.
        blocker = os.path.join(self.folder, 'blocker')
        with io.open(blocker, 'w') as blocker_file:
            blocker_file.write(u'x')
        broken_destination = os.path.join(blocker, 'audit.sqlite')

        doomed = db.subscribe_audit_snapshot(1, broken_destination)
        db['k'] = 'v1'
        db.commit()
        with self.assertRaises(AuditSnapshotError):
            doomed.wait(5)
        # Failure never leaves a partial file at the advertised destination.
        self.assertFalse(os.path.exists(broken_destination))
        self.assertEqual(db.commit_seq, 1)

        # Retry: subscribe to the next boundary with a writable destination.
        retried = db.subscribe_audit_snapshot(2, self.backup)
        db['k'] = 'v2'
        db.commit()
        manifest = retried.wait(5)
        self.assertEqual(manifest['commit_seq'], 2)
        with SqliteDict(self.backup, flag='r') as copy:
            self.assertEqual(copy['k'], 'v2')
        db.close()

    def test_pending_subscription_fails_when_connection_closes(self):
        db = SqliteDict(self.source)
        handle = db.subscribe_audit_snapshot(9, self.backup)
        db.close()
        with self.assertRaises(AuditSnapshotError):
            handle.wait(5)

    def test_repeated_snapshots_never_expose_partial_copy(self):
        db = SqliteDict(self.source)
        stop = threading.Event()
        seen_sequences = []

        def reader():
            while not stop.is_set():
                if not os.path.exists(self.backup):
                    continue
                try:
                    # Every observation must be a complete, openable database
                    # whose manifest matches its own contents.
                    with sqlite3.connect(self.backup) as connection:
                        count = connection.execute(
                            'SELECT COUNT(*) FROM "unnamed"'
                        ).fetchone()[0]
                    with io.open(self.backup + '.manifest.json') as handle:
                        manifest = json.load(handle)
                    self.assertEqual(
                        manifest['tables']['unnamed'], count
                    )
                    seen_sequences.append(manifest['commit_seq'])
                except sqlite3.DatabaseError:
                    raise AssertionError('reader observed a torn backup file')

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        try:
            for seq in range(1, 7):
                handle = db.subscribe_audit_snapshot(seq, self.backup)
                db['key'] = 'value-%d' % seq
                db.commit()
                handle.wait(5)
        finally:
            stop.set()
            reader_thread.join(5)

        self.assertFalse(reader_thread.is_alive())
        self.assertEqual(db.commit_seq, 6)
        # No temporary leftovers: the swap is fully atomic.
        leftovers = [
            name for name in os.listdir(os.path.dirname(self.backup))
            if name.endswith('.tmp')
        ]
        self.assertEqual(leftovers, [])
        with SqliteDict(self.backup, flag='r') as copy:
            self.assertEqual(copy['key'], 'value-6')
        db.close()

    def test_wait_timeout(self):
        db = SqliteDict(self.source)
        handle = db.subscribe_audit_snapshot(3, self.backup)
        with self.assertRaises(TimeoutError):
            handle.wait(timeout=0.1)
        self.assertFalse(handle.ready)
        db.close()

    def test_autocommit_counts_every_write_as_commit(self):
        db = SqliteDict(self.source, autocommit=True)
        handle = db.subscribe_audit_snapshot(2, self.backup)
        db['k'] = 'first'
        db['k'] = 'second'
        manifest = handle.wait(5)
        self.assertEqual(manifest['commit_seq'], 2)
        with SqliteDict(self.backup, flag='r') as copy:
            self.assertEqual(copy['k'], 'second')
        self.assertEqual(db.commit_seq, 2)
        db.close()

    def test_invalid_subscription_arguments(self):
        db = SqliteDict(self.source)
        with self.assertRaises(ValueError):
            db.subscribe_audit_snapshot(0, self.backup)
        with self.assertRaises(ValueError):
            db.subscribe_audit_snapshot(1, '')
        db.close()


if __name__ == '__main__':
    unittest.main()
