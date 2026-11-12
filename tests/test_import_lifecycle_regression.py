# -*- coding: utf-8 -*-
"""Regression for a source-line-tagged import through the full SqliteDict lifecycle.

Every imported row carries its source line. The custom key codec deliberately
maps two different logical keys onto one storage key (a REPLACE collision), and
unrelated normal requests are interleaved before and after the failing rows.

For each source line three independent views are reconciled:

  * the exception the writing caller actually observes, and at which call;
  * what a *separate* connection really reads, both the raw SQLite bytes and a
    fresh SqliteDict decode attempt;
  * what remains on disk after the writer is closed.

Three shortest operation sequences are isolated:

  A. a request that is never dispatched to the background thread;
  B. SQL executed inside the background transaction but never committed;
  C. bytes that are committed but cannot be decoded afterwards.

Internal backend failures are never located by re-running a user codec: the
codecs keep their own external audit trail, and assertions prove the failed
row's codec is invoked exactly once. The background error surfaces through
SqliteMultithread.check_raise_error instead.
"""
import pickle
import shutil
import sqlite3
import tempfile
import time
import unittest

import sqlitedict
from sqlitedict import SqliteDict


TABLE = 'imports'
BAD_BLOB = b'\x00this-is-not-a-pickle-stream'

# Logical import keys -> physical storage keys. Lines 1 and 3 collide.
KEY_MAP = {
    'alpha': 'collision',    # source line 1
    'gamma': 'collision',    # source line 3, same storage key as line 1
    'beta': 'benign',        # source line 2, unrelated normal request
    'delta': 'benign-early',  # normal request issued before the failing row
}


class UnsupportedBindValue(object):
    """The value codec accepts this and returns it untouched; SQLite's
    parameter binder then rejects the type in the background thread."""

    def __init__(self, line):
        self.line = line


class OpaqueBlob(object):
    """The value codec emits raw bytes that the pickle decoder cannot parse,
    while SQLite itself stores them happily."""

    def __init__(self, line, blob):
        self.line = line
        self.blob = blob


class CodecAudit(object):
    """External record kept by the user codecs themselves.

    It is deliberately separate from SqliteDict's internal exception state:
    diagnosing a backend failure must not cause any extra codec invocation.
    """

    def __init__(self, key_map=None):
        self.key_map = key_map or {}
        self.encode_key_calls = []
        self.decode_key_calls = []
        self.encode_calls = []
        self.decode_calls = []

    def kwargs(self):
        return dict(
            encode_key=self.encode_key,
            decode_key=self.decode_key,
            encode=self.encode,
            decode=self.decode,
        )

    def encode_key(self, key):
        stored = self.key_map.get(key, key)
        self.encode_key_calls.append((key, stored))
        return stored

    def decode_key(self, stored):
        self.decode_key_calls.append(stored)
        return stored

    def encode(self, value):
        line = getattr(value, 'line', None)
        self.encode_calls.append((line, type(value).__name__))
        if isinstance(value, UnsupportedBindValue):
            return value
        if isinstance(value, OpaqueBlob):
            return value.blob
        return sqlitedict.encode(value)

    def decode(self, blob):
        self.decode_calls.append(bytes(blob))
        return sqlitedict.decode(blob)

    def line_encodes(self, line):
        return [entry for entry in self.encode_calls if entry[0] == line]


def wait_drained(writer, timeout=5.0):
    """Block until the background thread has dequeued every queued request.

    A request is fully handled before run() loops back to reqs.get(), so an
    empty queue means the SQL (and any exception capture) has happened.
    """
    deadline = time.monotonic() + timeout
    while writer.conn.reqs.qsize():
        if time.monotonic() > deadline:
            raise AssertionError('background request queue did not drain in time')
        time.sleep(0.001)


def raw_value(fname, stored_key):
    """Read committed bytes through an independent plain sqlite3 connection."""
    conn = sqlite3.connect(fname)
    try:
        row = conn.execute(
            'SELECT value FROM "%s" WHERE key = ?' % TABLE, (stored_key,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def backend_value(writer, stored_key):
    """Read through the writer's own background connection (sees uncommitted rows)."""
    row = writer.conn.select_one(
        'SELECT value FROM "%s" WHERE key = ?' % TABLE, (stored_key,)
    )
    return None if row is None else row[0]


def reader_get(fname, stored_key, reader_audit):
    """Open an independent read-only SqliteDict and attempt a real decode."""
    reader = SqliteDict(
        fname, tablename=TABLE, flag='r',
        encode_key=sqlitedict.identity, decode_key=sqlitedict.identity,
        encode=sqlitedict.encode, decode=reader_audit.decode,
    )
    try:
        return 'ok', reader[stored_key]
    except Exception as exc:
        return 'error', type(exc).__name__
    finally:
        reader.close()


class ImportLifecycleRegressionTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='sqldict-lifecycle-')
        self.db_index = 0

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def new_db(self):
        self.db_index += 1
        return '%s/import-%d.sqlite' % (self.tmpdir, self.db_index)

    def test_full_manifest_manual_commit_by_source_line(self):
        # Manifest order: failing row 1, benign row 2, colliding row 3.
        manifest = [
            ('manifest.csv:1', 'alpha', UnsupportedBindValue('manifest.csv:1')),
            ('manifest.csv:2', 'beta', ('normal', 2)),
            ('manifest.csv:3', 'gamma', OpaqueBlob('manifest.csv:3', BAD_BLOB)),
        ]
        line1, key1, value1 = manifest[0]
        line2, key2, value2 = manifest[1]
        line3, key3, value3 = manifest[2]

        fname = self.new_db()
        audit = CodecAudit(KEY_MAP)
        reader_audit = CodecAudit()
        writer = SqliteDict(fname, tablename=TABLE, **audit.kwargs())

        # Phase 0: an unrelated normal request before the failing row.
        writer['delta'] = ('early', 0)
        wait_drained(writer)

        # Phase 1 (line 1): both codecs run on the caller's thread and the
        # request is queued; the bind failure happens in the background thread.
        # Without autocommit, setitem is fire-and-forget, so the exception is
        # not visible yet.
        writer[key1] = value1
        wait_drained(writer)
        self.assertEqual(len(audit.line_encodes(line1)), 1)
        self.assertIn((key1, 'collision'), audit.encode_key_calls)

        # Phase 2 (line 2, first attempt): the pending backend error is
        # re-raised at check_raise_error(), before this request is queued.
        queued_before = writer.conn.reqs.qsize()
        with self.assertRaises(sqlite3.ProgrammingError):
            writer[key2] = value2
        self.assertEqual(writer.conn.reqs.qsize(), queued_before)
        # Locating the failure did not re-run the line-1 codec.
        self.assertEqual(len(audit.line_encodes(line1)), 1)

        # The pending exception was one-shot: the benign request now goes
        # through normally and executes inside the open transaction.
        writer[key2] = value2
        wait_drained(writer)

        # The failed line-1 REPLACE wrote nothing, not even inside the
        # backend transaction (the bind rejected the whole statement).
        self.assertIsNone(backend_value(writer, 'collision'))

        # Phase 3 (line 3): opaque blob, same storage key as line 1. The
        # REPLACE executes; line 1 never produced a row to replace.
        writer[key3] = value3
        wait_drained(writer)
        self.assertEqual(len(audit.line_encodes(line3)), 1)
        self.assertIn((key3, 'collision'), audit.encode_key_calls)
        self.assertEqual(backend_value(writer, 'collision'), BAD_BLOB)

        # Stage B evidence: SQL executed but not committed. A separate
        # connection still sees none of the transaction's rows.
        self.assertIsNone(raw_value(fname, 'collision'))
        self.assertIsNone(raw_value(fname, 'benign'))
        self.assertIsNone(raw_value(fname, 'benign-early'))

        # Commit: only now does the other connection see the rows.
        writer.commit(blocking=True)
        self.assertEqual(raw_value(fname, 'collision'), BAD_BLOB)
        self.assertEqual(raw_value(fname, 'benign'), sqlitedict.encode(value2))

        # Stage C evidence: the committed blob cannot be decoded.
        self.assertEqual(reader_get(fname, 'collision', reader_audit),
                         ('error', pickle.UnpicklingError.__name__))
        self.assertEqual(reader_get(fname, 'benign', reader_audit),
                         ('ok', value2))
        self.assertEqual(reader_get(fname, 'benign-early', reader_audit),
                         ('ok', ('early', 0)))

        # Per-source-line reconciliation after close and reopen.
        writer.close()
        # Line 1: nothing persisted under the collision key from this row; the
        # single row there belongs to line 3 and stays undecodable.
        self.assertEqual(raw_value(fname, 'collision'), BAD_BLOB)
        self.assertEqual(reader_get(fname, 'collision', reader_audit),
                         ('error', pickle.UnpicklingError.__name__))
        # Line 2: the retried normal request persisted and decodes fine.
        self.assertEqual(reader_get(fname, 'benign', reader_audit),
                         ('ok', value2))
        # The line-1 codec still ran exactly once over the whole lifecycle.
        self.assertEqual(len(audit.line_encodes(line1)), 1)
        self.assertEqual(len(audit.line_encodes(line3)), 1)

    def test_stage_a_request_never_dispatched(self):
        fname = self.new_db()
        audit = CodecAudit(KEY_MAP)
        writer = SqliteDict(fname, tablename=TABLE, **audit.kwargs())

        writer['delta'] = ('early', 0)
        wait_drained(writer)

        # The failing row is dispatched and dies on the SQLite bind.
        writer['alpha'] = UnsupportedBindValue('manifest.csv:1')
        wait_drained(writer)

        # Shortest sequence: the next call surfaces the error at
        # check_raise_error(), so its request never reaches reqs.
        queued_before = writer.conn.reqs.qsize()
        with self.assertRaises(sqlite3.ProgrammingError):
            writer['beta'] = ('normal', 2)
        self.assertEqual(writer.conn.reqs.qsize(), queued_before)

        # Neither the failed row nor the rejected benign row touched SQL.
        self.assertIsNone(raw_value(fname, 'collision'))
        self.assertIsNone(raw_value(fname, 'benign'))

        # After the one-shot exception is consumed, the queue works again.
        writer['beta'] = ('normal', 2)
        wait_drained(writer)
        writer.commit(blocking=True)
        self.assertEqual(raw_value(fname, 'benign'),
                         sqlitedict.encode(('normal', 2)))
        self.assertIsNone(raw_value(fname, 'collision'))

        writer.close()
        self.assertIsNone(raw_value(fname, 'collision'))
        self.assertEqual(len(audit.line_encodes('manifest.csv:1')), 1)

    def test_stage_b_executed_but_not_committed(self):
        # Branch 1, shortest rollback sequence: execute, no commit, close.
        fname = self.new_db()
        audit = CodecAudit()
        writer = SqliteDict(fname, tablename=TABLE, **audit.kwargs())
        writer['k'] = 'v'
        wait_drained(writer)

        # The writer's own connection executed the REPLACE.
        self.assertEqual(backend_value(writer, 'k'), sqlitedict.encode('v'))
        # A separate connection sees nothing: no commit happened.
        self.assertIsNone(raw_value(fname, 'k'))

        writer.close()  # sqlite closes the implicit transaction via rollback
        self.assertIsNone(raw_value(fname, 'k'))

        # Branch 2: the same sequence plus one blocking commit persists.
        fname_committed = self.new_db()
        committed = SqliteDict(fname_committed, tablename=TABLE,
                               **CodecAudit().kwargs())
        committed['k'] = 'v'
        wait_drained(committed)
        self.assertIsNone(raw_value(fname_committed, 'k'))
        committed.commit(blocking=True)
        self.assertEqual(raw_value(fname_committed, 'k'),
                         sqlitedict.encode('v'))
        committed.close()
        self.assertEqual(raw_value(fname_committed, 'k'),
                         sqlitedict.encode('v'))

    def test_stage_c_committed_but_undecodable(self):
        fname = self.new_db()
        audit = CodecAudit()
        reader_audit = CodecAudit()
        writer = SqliteDict(fname, tablename=TABLE, **audit.kwargs())

        writer['k'] = OpaqueBlob('manifest.csv:9', BAD_BLOB)
        wait_drained(writer)
        writer.commit(blocking=True)

        # The bytes are durably committed and readable as raw BLOB.
        self.assertEqual(raw_value(fname, 'k'), BAD_BLOB)
        # But every SqliteDict-level read fails inside the user decode codec.
        self.assertEqual(reader_get(fname, 'k', reader_audit),
                         ('error', pickle.UnpicklingError.__name__))
        self.assertEqual(len(reader_audit.decode_calls), 1)

        # Close and reopen: the corruption persists on disk and still fails.
        writer.close()
        self.assertEqual(raw_value(fname, 'k'), BAD_BLOB)
        reopen_audit = CodecAudit()
        self.assertEqual(reader_get(fname, 'k', reopen_audit),
                         ('error', pickle.UnpicklingError.__name__))

    def test_autocommit_collision_and_interleaved_failure(self):
        fname = self.new_db()
        audit = CodecAudit(KEY_MAP)
        reader_audit = CodecAudit()
        writer = SqliteDict(fname, tablename=TABLE, autocommit=True,
                            **audit.kwargs())

        # Normal request before the failure: immediately visible elsewhere.
        writer['delta'] = ('early', 0)
        wait_drained(writer)
        self.assertEqual(raw_value(fname, 'benign-early'),
                         sqlitedict.encode(('early', 0)))

        # Line 1: under autocommit the setitem internally issues a blocking
        # commit, so the bind failure from the background thread is re-raised
        # on the failing assignment itself, not at the next call.
        with self.assertRaises(sqlite3.ProgrammingError):
            writer['alpha'] = UnsupportedBindValue('manifest.csv:1')
        self.assertEqual(len(audit.line_encodes('manifest.csv:1')), 1)
        self.assertIsNone(raw_value(fname, 'collision'))

        # The exception was one-shot: an unrelated normal request goes
        # through immediately afterwards and is auto-committed.
        writer['beta'] = ('normal', 2)
        wait_drained(writer)
        self.assertEqual(raw_value(fname, 'benign'),
                         sqlitedict.encode(('normal', 2)))

        # Line 3 lands on line 1's storage key and auto-commits an
        # undecodable blob.
        writer['gamma'] = OpaqueBlob('manifest.csv:3', BAD_BLOB)
        wait_drained(writer)

        self.assertEqual(raw_value(fname, 'collision'), BAD_BLOB)
        self.assertEqual(reader_get(fname, 'collision', reader_audit),
                         ('error', pickle.UnpicklingError.__name__))
        self.assertEqual(reader_get(fname, 'benign', reader_audit),
                         ('ok', ('normal', 2)))
        self.assertEqual(len(audit.line_encodes('manifest.csv:1')), 1)

        writer.close()
        # Post-close persistence matches the last auto-committed state.
        self.assertEqual(raw_value(fname, 'collision'), BAD_BLOB)
        self.assertEqual(reader_get(fname, 'collision', reader_audit),
                         ('error', pickle.UnpicklingError.__name__))
        self.assertEqual(reader_get(fname, 'benign-early', reader_audit),
                         ('ok', ('early', 0)))


if __name__ == '__main__':
    unittest.main()
