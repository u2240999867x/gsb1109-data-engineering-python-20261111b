#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This code is distributed under the terms and conditions
# from the Apache License, Version 2.0
"""Tests for position-preserving, versioned batch reads (snapshot reader / getmany)."""
import glob
import os
import tempfile
import unittest

import sqlitedict
from sqlitedict import SqliteDict, MISSING, DecodeError
from accessories import norm_file


def _failing_decode(obj):
    raise ValueError('corrupt payload')


class SnapshotBatchReadTest(unittest.TestCase):

    def setUp(self):
        self.fname = norm_file('tests/db/batch-read.sqlite')
        self.db = SqliteDict(self.fname)
        self.db.update({'a': 1, 'b': None, 'c': [1, 2]})
        self.db.commit()

    def tearDown(self):
        self.db.close()
        if os.path.isfile(self.fname):
            os.unlink(self.fname)

    def test_positions_missing_none_and_values(self):
        """Missing keys, stored None and real values each keep their position."""
        keys = ['c', 'missing', 'a', 'b', 'missing']
        results = list(self.db.getmany(keys, max_read=10))

        self.assertEqual(len(results), 1)
        batch = results[0]
        self.assertEqual(len(batch), len(keys))
        self.assertEqual(batch[0], [1, 2])
        self.assertIs(batch[1], MISSING)
        self.assertEqual(batch[2], 1)
        self.assertIsNone(batch[3])
        self.assertIs(batch[4], MISSING)

    def test_none_is_distinct_from_missing(self):
        """A stored None must not be confused with an absent key."""
        batch = list(self.db.getmany(['b', 'nope']))[0]
        self.assertIsNone(batch[0])
        self.assertIs(batch[1], MISSING)

    def test_max_read_paginates_and_streams_early(self):
        """Results arrive in pages of max_read without consuming all keys first."""
        keys = ['k%03d' % i for i in range(25)]
        for i, key in enumerate(keys):
            self.db[key] = i
        self.db.commit()

        consumed = []

        def key_stream():
            for key in keys:
                consumed.append(key)
                yield key

        generator = self.db.getmany(key_stream(), max_read=10)
        first_batch = next(generator)
        # The first page is handed out as soon as 10 keys are seen, not after 25.
        self.assertEqual(len(first_batch), 10)
        self.assertEqual(len(consumed), 10)

        batches = [first_batch] + list(generator)

        self.assertEqual([len(batch) for batch in batches], [10, 10, 5])
        flat = [item for batch in batches for item in batch]
        self.assertEqual(flat, list(range(25)))

    def test_empty_keys_yields_nothing(self):
        self.assertEqual(list(self.db.getmany([])), [])

    def test_repeated_key_returns_same_version_across_pages(self):
        """The same key, pages apart, always resolves to the snapshot version."""
        writer = SqliteDict(self.fname)
        try:
            keys = ['a', 'b', 'a', 'c', 'a', 'b']
            batches = []
            for index, batch in enumerate(self.db.getmany(iter(keys), max_read=2)):
                batches.append(batch)
                if index == 0:
                    writer['a'] = 999
                    writer['b'] = 'changed'
                    writer.commit()
        finally:
            writer.close()

        flat = [item for batch in batches for item in batch]
        self.assertEqual(flat, [1, None, 1, [1, 2], 1, None])
        # The live connection, by contrast, observes the new committed value.
        self.assertEqual(self.db['a'], 999)

    def test_snapshot_ignores_concurrent_commits(self):
        """New values committed in the background never alter snapshot results."""
        keys = ['a', 'later', 'b', 'later']
        writer = SqliteDict(self.fname)
        try:
            batches = []
            for index, batch in enumerate(self.db.getmany(iter(keys), max_read=1)):
                batches.append(batch)
                writer['later'] = 'inserted'
                writer['a'] = -1
                writer.commit()
        finally:
            writer.close()

        flat = [item for batch in batches for item in batch]
        self.assertEqual(flat[0], 1)
        self.assertIs(flat[1], MISSING)
        self.assertIsNone(flat[2])
        self.assertIs(flat[3], MISSING)

    def test_uncommitted_writes_are_invisible(self):
        """The snapshot only contains committed state."""
        writer = SqliteDict(self.fname)
        writer['uncommitted'] = 'x'
        try:
            batch = list(self.db.getmany(['uncommitted', 'a']))[0]
            self.assertIs(batch[0], MISSING)
            self.assertEqual(batch[1], 1)
        finally:
            writer.close()

    def test_chunking_crosses_variable_limit(self):
        """Batches larger than chunk_size still return every position correctly."""
        total = 2500
        keys = ['key%04d' % i for i in range(total)]
        for i in range(0, total, 2):
            self.db[keys[i]] = i
        self.db.commit()

        with self.db.snapshot_reader(chunk_size=7) as reader:
            batch = reader.read(keys)

        self.assertEqual(len(batch), total)
        for i, item in enumerate(batch):
            if i % 2 == 0:
                self.assertEqual(item, i)
            else:
                self.assertIs(item, MISSING)

    def test_default_chunk_size_handles_large_batch(self):
        """The conservative default chunk size supports batches above the old 999 limit."""
        keys = ['k%d' % i for i in range(3000)]
        batch = list(self.db.getmany(iter(keys), max_read=3000))[0]
        self.assertEqual(len(batch), 3000)
        self.assertTrue(all(item is MISSING for item in batch))

    def test_decode_error_occupies_position(self):
        """Codec failures are returned in place instead of aborting the batch."""
        broken = SqliteDict(self.fname, decode=_failing_decode)
        try:
            batch = list(broken.getmany(['a', 'missing', 'c']))[0]
        finally:
            broken.close()

        self.assertIsInstance(batch[0], DecodeError)
        self.assertEqual(batch[0].key, 'a')
        self.assertIsInstance(batch[0].__cause__, ValueError)
        self.assertIs(batch[1], MISSING)
        self.assertIsInstance(batch[2], DecodeError)
        self.assertEqual(batch[2].key, 'c')

    def test_reader_close_removes_snapshot_file(self):
        """Closing the reader releases its private snapshot database."""
        reader = self.db.snapshot_reader()
        path = reader._path
        self.assertTrue(os.path.exists(path))
        self.assertEqual(reader.read(['a']), [1])
        reader.close()
        self.assertFalse(os.path.exists(path))
        with self.assertRaises(RuntimeError):
            reader.read(['a'])

    def test_early_generator_stop_releases_resources(self):
        """Abandoning getmany halfway still cleans up the snapshot."""
        pattern = os.path.join(tempfile.gettempdir(), 'sqlitedict-snapshot-*.sqlite')
        before = set(glob.glob(pattern))

        generator = self.db.getmany(('k%d' % i for i in range(100)), max_read=5)
        self.assertEqual(next(generator), [MISSING] * 5)
        generator.close()

        after = set(glob.glob(pattern))
        self.assertEqual(before, after)

    def test_open_reader_does_not_block_writer(self):
        """Holding a snapshot reader must not stop the writer from committing."""
        reader = self.db.snapshot_reader()
        try:
            writer = SqliteDict(self.fname)
            try:
                for i in range(20):
                    writer['counter'] = i
                    writer.commit()
            finally:
                writer.close()
            # The reader keeps serving the frozen version...
            self.assertEqual(reader.read(['counter', 'a']), [MISSING, 1])
        finally:
            reader.close()
        # ...while the live table has moved on.
        self.assertEqual(self.db['counter'], 19)

    def test_nonstring_keys(self):
        """Custom key codecs are honored by the batch reader."""
        fname = norm_file('tests/db/batch-read-encoded-keys.sqlite')
        encoded = SqliteDict(
            fname, encode_key=sqlitedict.encode_key, decode_key=sqlitedict.decode_key)
        try:
            encoded[(1, 2)] = 'tuple-key'
            encoded[7] = 'int-key'
            encoded.commit()
            batch = list(encoded.getmany([(1, 2), 'absent', 7]))[0]
            self.assertEqual(batch, ['tuple-key', MISSING, 'int-key'])
        finally:
            encoded.close()
            os.unlink(fname)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.db.snapshot_reader(chunk_size=0)
        with self.assertRaises(ValueError):
            self.db.getmany(['a'], max_read=0)
        with self.assertRaises(ValueError):
            self.db.getmany(['a'], max_read=-3)

    def test_in_memory_database_rejected(self):
        """A separate snapshot connection cannot read an in-memory database."""
        memory_db = SqliteDict(':memory:')
        try:
            with self.assertRaises(RuntimeError):
                memory_db.snapshot_reader()
        finally:
            memory_db.close()


if __name__ == '__main__':
    unittest.main()
