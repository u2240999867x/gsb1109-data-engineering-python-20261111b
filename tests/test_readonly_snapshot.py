# std imports
import os
import shutil
import sqlite3
import tempfile
import unittest

# local
import sqlitedict
from sqlitedict import SqliteDict, ReadonlySnapshot
from accessories import norm_file


class ReadonlySnapshotTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='sqlitedict-snapshot-test')
        self.fname = os.path.join(self.tmpdir, 'ref.sqlite')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_db(self, journal_mode='DELETE'):
        """Create a database with two tables and keep the writer open."""
        writer = SqliteDict(filename=self.fname, tablename='table_one',
                            journal_mode=journal_mode)
        writer['key'] = 'value_one'
        writer.commit()
        other = SqliteDict(filename=self.fname, tablename='table_two',
                           journal_mode=journal_mode)
        other['key'] = 'value_two'
        other.commit()
        return writer, other

    def test_missing_database_raises_and_creates_nothing(self):
        fname = os.path.join(self.tmpdir, 'missing.sqlite')
        with self.assertRaises(IOError):
            ReadonlySnapshot(fname)
        self.assertFalse(os.path.exists(fname))

    def test_missing_database_flag_r_creates_nothing(self):
        fname = os.path.join(self.tmpdir, 'missing.sqlite')
        with self.assertRaises(IOError):
            SqliteDict(filename=fname, flag='r')
        self.assertFalse(os.path.exists(fname))

    def test_missing_table_raises_and_creates_nothing(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            with self.assertRaises(RuntimeError):
                snapshot.table('table404')
            self.assertNotIn('table404', snapshot.tablenames())
        self.assertNotIn('table404', SqliteDict.get_tablenames(self.fname))

    def test_tables_share_one_version(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            first = snapshot.table('table_one')
            second = snapshot.table('table_two')
            self.assertEqual(first['key'], 'value_one')
            self.assertEqual(second['key'], 'value_two')
            self.assertEqual(first.snapshot_version, snapshot.version)
            self.assertEqual(second.snapshot_version, snapshot.version)
            self.assertTrue(snapshot.version.startswith('journal:'))
            first.close()
            second.close()

    def test_version_stable_without_commit(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as first:
            first_version = first.version
        with ReadonlySnapshot(self.fname) as second:
            second_version = second.version
        self.assertEqual(first_version, second_version)

    def test_version_changes_after_commit(self):
        writer, other = self._make_db()
        with ReadonlySnapshot(self.fname) as snapshot:
            version_before = snapshot.version
        writer['key'] = 'updated'
        writer.commit()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            version_after = snapshot.version
            self.assertEqual(snapshot.table('table_one')['key'], 'updated')
        self.assertNotEqual(version_before, version_after)

    def test_readonly_dict_rejects_writes(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            d = snapshot.table('table_one')
            with self.assertRaises(RuntimeError):
                d['key'] = 'new_value'
            with self.assertRaises(RuntimeError):
                d.update(key='new_value')
            with self.assertRaises(RuntimeError):
                del d['key']
            with self.assertRaises(RuntimeError):
                d.clear()
            with self.assertRaises(RuntimeError):
                d.terminate()
            d.close()

    def test_readonly_connection_rejects_writes(self):
        """The connection itself refuses writes, not just the dict layer."""
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            snapshot.conn.execute('CREATE TABLE evil (key TEXT)')
            with self.assertRaises(sqlite3.OperationalError):
                snapshot.conn.select_one('SELECT 1')
            # the snapshot survives the rejected write
            self.assertEqual(snapshot.table('table_one')['key'], 'value_one')

    def test_table_close_keeps_snapshot_alive(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        with ReadonlySnapshot(self.fname) as snapshot:
            first = snapshot.table('table_one')
            second = snapshot.table('table_two')
            first.close()
            self.assertEqual(second['key'], 'value_two')
            self.assertEqual(snapshot.table('table_one')['key'], 'value_one')
            second.close()

    def test_table_on_closed_snapshot_raises(self):
        writer, other = self._make_db()
        writer.close()
        other.close()
        snapshot = ReadonlySnapshot(self.fname)
        snapshot.close()
        with self.assertRaises(RuntimeError):
            snapshot.table('table_one')


class ReadonlySnapshotWalTest(unittest.TestCase):
    """Snapshots over WAL-mode sources (active writers keep the -wal alive)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='sqlitedict-snapshot-wal-test')
        self.fname = os.path.join(self.tmpdir, 'ref.sqlite')
        self.writer = SqliteDict(filename=self.fname, tablename='table_one',
                                 journal_mode='WAL')
        self.writer['key'] = 'value_one'
        self.writer.commit()
        self.other = SqliteDict(filename=self.fname, tablename='table_two',
                                journal_mode='WAL')
        self.other['key'] = 'value_two'
        self.other.commit()

    def tearDown(self):
        self.writer.close()
        self.other.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_active_wal_source_is_readable(self):
        with ReadonlySnapshot(self.fname) as snapshot:
            self.assertTrue(snapshot.version.startswith('wal:'))
            self.assertEqual(snapshot.table('table_one')['key'], 'value_one')
            self.assertEqual(snapshot.table('table_two')['key'], 'value_two')

    def test_snapshot_is_pinned_across_tables(self):
        """A commit between accessing the two tables must not leak in."""
        with ReadonlySnapshot(self.fname) as snapshot:
            first = snapshot.table('table_one')
            self.assertEqual(first['key'], 'value_one')
            version_before = snapshot.version

            # another connection commits while the backfill is in progress
            self.writer['key'] = 'updated'
            self.writer.commit()
            self.other['key'] = 'updated_too'
            self.other.commit()

            # the second table still sees the pinned version, not the new one
            second = snapshot.table('table_two')
            self.assertEqual(second['key'], 'value_two')
            self.assertEqual(first['key'], 'value_one')
            self.assertEqual(snapshot.version, version_before)

            # an accidental commit() must not release the pinned version
            first.commit()
            self.assertEqual(first['key'], 'value_one')
            first.close()
            second.close()

        # a fresh snapshot sees the new committed version, with a new token
        with ReadonlySnapshot(self.fname) as snapshot:
            self.assertNotEqual(snapshot.version, version_before)
            self.assertEqual(snapshot.table('table_one')['key'], 'updated')
            self.assertEqual(snapshot.table('table_two')['key'], 'updated_too')

    def test_detached_copy_without_wal_is_rejected(self):
        """An offline copy missing its -wal companion cannot be authenticated."""
        copy = os.path.join(self.tmpdir, 'copy.sqlite')
        shutil.copy(self.fname, copy)
        self.assertFalse(os.path.exists(copy + '-wal'))
        with self.assertRaises(RuntimeError):
            ReadonlySnapshot(copy)

    def test_copy_with_wal_companion_is_accepted(self):
        copy = os.path.join(self.tmpdir, 'copy.sqlite')
        shutil.copy(self.fname, copy)
        shutil.copy(self.fname + '-wal', copy + '-wal')
        with ReadonlySnapshot(copy) as snapshot:
            self.assertEqual(snapshot.table('table_one')['key'], 'value_one')


if __name__ == '__main__':
    unittest.main()
