# -*- coding: utf-8 -*-
"""Tests for staged key renames: connected groups, cycles and concurrency."""
import json
import os
import sqlite3
import tempfile
import unittest

import sqlitedict
from sqlitedict import SqliteDict, RenameRequest, RenameConflict
from accessories import norm_file


class CountingCodec(object):
    """Value codec that records (de)serialization calls and marks bytes."""

    def __init__(self):
        self.encodes = 0
        self.decodes = 0

    def encode(self, value):
        self.encodes += 1
        return sqlitedict.encode('tagged:%r' % (value,))

    def decode(self, raw):
        self.decodes += 1
        return sqlitedict.decode(raw)


class RenameTestBase(unittest.TestCase):

    def setUp(self):
        self.fname = norm_file(os.path.join(
            tempfile.mkdtemp(prefix='sqlitedict-rename-'), 'db.sqlite'
        ))
        self.db = SqliteDict(filename=self.fname)

    def tearDown(self):
        self.db.close()
        os.remove(self.fname)
        os.rmdir(os.path.dirname(self.fname))


class PreviewRenameTest(RenameTestBase):

    def test_independent_groups_and_manifest_order(self):
        self.db.update({'a': 1, 'b': 2, 'c': 3})
        self.db.commit()
        requests = [
            RenameRequest('a', 'a2', 7),
            RenameRequest('b', 'b2', 2),
            RenameRequest('c', 'c2', 5),
        ]
        groups = self.db.preview_renames(requests)
        self.assertEqual(len(groups), 3)
        self.assertEqual(
            [min(r.source_line for r in group.requests) for group in groups],
            [2, 5, 7],
        )
        self.assertTrue(all(not group.blocked for group in groups))
        self.assertTrue(all(not group.cyclic for group in groups))

    def test_connected_group_via_release_then_occupy(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        requests = [
            RenameRequest('a', 'b', 1),
            RenameRequest('b', 'c', 2),
        ]
        groups = self.db.preview_renames(requests)
        self.assertEqual(len(groups), 1)
        self.assertEqual({r.source_line for r in groups[0].requests}, {1, 2})
        self.assertFalse(groups[0].blocked)
        self.assertFalse(groups[0].cyclic)

    def test_cycle_is_allowed_but_flagged(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        groups = self.db.preview_renames([
            RenameRequest('a', 'b', 3),
            RenameRequest('b', 'a', 4),
        ])
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0].cyclic)
        self.assertFalse(groups[0].blocked)

    def test_self_loop_is_cyclic(self):
        self.db['a'] = 1
        self.db.commit()
        groups = self.db.preview_renames([RenameRequest('a', 'a', 1)])
        self.assertTrue(groups[0].cyclic)
        self.assertFalse(groups[0].blocked)

    def test_missing_source_blocks_group(self):
        self.db['a'] = 1
        self.db.commit()
        groups = self.db.preview_renames([
            RenameRequest('a', 'a2', 1),
            RenameRequest('ghost', 'g2', 2),
        ])
        self.assertEqual(len(groups), 2)
        blocked = [group for group in groups if group.blocked]
        self.assertEqual(len(blocked), 1)
        self.assertTrue(any('missing source' in reason for reason in blocked[0].reasons))

    def test_occupied_destination_blocks_group(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        groups = self.db.preview_renames([RenameRequest('a', 'b', 1)])
        self.assertTrue(groups[0].blocked)
        self.assertTrue(any('occupied' in reason for reason in groups[0].reasons))

    def test_duplicate_destination_blocks_group(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        groups = self.db.preview_renames([
            RenameRequest('a', 'z', 1),
            RenameRequest('b', 'z', 2),
        ])
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0].blocked)
        self.assertTrue(groups[0].collision)
        self.assertTrue(any('destination' in reason for reason in groups[0].reasons))

    def test_encoded_key_collision_with_custom_codec(self):
        db = SqliteDict(
            filename=self.fname, tablename='collide',
            encode_key=lambda key: key[0], decode_key=lambda key: key,
        )
        db['ab'] = 1
        db.commit()
        groups = db.preview_renames([RenameRequest('ab', 'ax', 1)])
        self.assertTrue(groups[0].blocked)
        self.assertTrue(groups[0].collision)
        db.close()

    def test_distinct_logical_keys_sharing_encoding_are_one_group(self):
        db = SqliteDict(
            filename=self.fname, tablename='collide2',
            encode_key=lambda key: key[0], decode_key=lambda key: key,
        )
        db.update({'ab': 1, 'bc': 2})
        db.commit()
        groups = db.preview_renames([
            RenameRequest('ab', 'bc', 1),
            RenameRequest('bc', 'cd', 2),
        ])
        self.assertEqual(len(groups), 1)
        db.close()

    def test_invalid_manifests(self):
        with self.assertRaises(RenameConflict):
            self.db.preview_renames([])
        with self.assertRaises(RenameConflict):
            self.db.preview_renames([
                RenameRequest('a', 'b', 1),
                RenameRequest('c', 'd', 1),
            ])
        with self.assertRaises(RenameConflict):
            self.db.preview_renames([RenameRequest('a', 'b', 0)])

    def test_tuple_and_dict_inputs_are_coerced(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        groups = self.db.preview_renames([
            ('a', 'a2', 1),
            {'old_key': 'b', 'new_key': 'b2', 'source_line': 2},
        ])
        self.assertEqual(len(groups), 2)


class ApplyRenameTest(RenameTestBase):

    def test_simple_rename_preserves_row_and_bytes(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        before = self._raw_rows()
        results = self.db.apply_renames([RenameRequest('a', 'a2', 1)])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].applied)
        self.assertEqual(results[0].line_results, {1: before['a'][0]})
        after = self._raw_rows()
        self.assertEqual(after['a2'], before['a'])
        self.assertNotIn('a', after)

    def test_value_codec_is_not_rerun(self):
        codec = CountingCodec()
        db = SqliteDict(filename=self.fname, tablename='codec',
                        encode=codec.encode, decode=codec.decode)
        db['a'] = 100
        db.commit()
        encodes_after_write = codec.encodes
        db.apply_renames([RenameRequest('a', 'a2', 1)])
        self.assertEqual(codec.encodes, encodes_after_write)
        raw = sqlite3.connect(self.fname).execute(
            'SELECT value FROM codec WHERE key = ?', ('a2',)
        ).fetchone()[0]
        self.assertEqual(raw, sqlitedict.encode('tagged:100'))
        db.close()

    def test_chain_release_then_occupy(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        before = self._raw_rows()
        results = self.db.apply_renames([
            RenameRequest('a', 'b', 1),
            RenameRequest('b', 'c', 2),
        ])
        self.assertTrue(all(result.applied for result in results))
        after = self._raw_rows()
        self.assertEqual(after['b'], before['a'])
        self.assertEqual(after['c'], before['b'])
        self.assertEqual(set(after), {'b', 'c'})

    def test_cycle_swap(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        before = self._raw_rows()
        self.db.apply_renames([
            RenameRequest('a', 'b', 1),
            RenameRequest('b', 'a', 2),
        ])
        after = self._raw_rows()
        self.assertEqual(after['b'], before['a'])
        self.assertEqual(after['a'], before['b'])
        self.assertEqual(self.db['a'], 2)
        self.assertEqual(self.db['b'], 1)

    def test_retry_returns_original_line_attribution(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        requests = [
            RenameRequest('a', 'a2', 1),
            RenameRequest('b', 'b2', 2),
        ]
        first = self.db.apply_renames(requests)
        second = self.db.apply_renames(requests)
        self.assertEqual(
            [result.line_results for result in first],
            [result.line_results for result in second],
        )
        self.assertTrue(all(result.already_applied for result in second))
        self.assertTrue(all(not result.applied for result in second))
        self.assertEqual(len(self.db), 2)

    def test_retry_after_half_blocked_apply(self):
        self.db.update({'a': 1, 'b': 2, 'c': 3})
        self.db.commit()
        requests = [
            RenameRequest('a', 'a2', 1),
            RenameRequest('b', 'b2', 2),
        ]
        # Simulate another connection deleting one source before the first run.
        other = SqliteDict(filename=self.fname)
        del other['b']
        other.commit()
        other.close()

        results = self.db.apply_renames(requests)
        status = {result.group.requests[0].source_line: result for result in results}
        self.assertTrue(status[1].applied)
        self.assertTrue(status[2].blocked)

        # The other connection restores the missing key; retry applies the
        # remaining group while reporting the first one from the ledger.
        other = SqliteDict(filename=self.fname)
        other['b'] = 2
        other.commit()
        other.close()

        results = self.db.apply_renames(requests)
        status = {result.group.requests[0].source_line: result for result in results}
        self.assertTrue(status[1].already_applied)
        self.assertTrue(status[2].applied)

    def test_unrelated_groups_survive_concurrent_changes(self):
        self.db.update({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        self.db.commit()
        requests = [
            RenameRequest('a', 'a2', 1),
            RenameRequest('b', 'b2', 2),
            RenameRequest('c', 'c2', 3),
        ]
        other = SqliteDict(filename=self.fname)
        del other['b']
        other.commit()
        other.close()

        results = self.db.apply_renames(requests)
        blocked_lines = {
            request.source_line
            for result in results if result.blocked
            for request in result.group.requests
        }
        applied_lines = {
            line
            for result in results if result.applied
            for line in result.line_results
        }
        self.assertEqual(blocked_lines, {2})
        self.assertEqual(applied_lines, {1, 3})
        self.assertIn('a2', self.db)
        self.assertIn('c2', self.db)
        self.assertNotIn('b2', self.db)

    def test_concurrent_destination_takeover_blocks_only_that_group(self):
        self.db.update({'a': 1, 'b': 2, 'c': 3})
        self.db.commit()
        requests = [
            RenameRequest('a', 'a2', 1),
            RenameRequest('c', 'c2', 3),
        ]
        other = SqliteDict(filename=self.fname)
        other['c2'] = 99
        other.commit()
        other.close()

        results = self.db.apply_renames(requests)
        status = {result.group.requests[0].source_line: result for result in results}
        self.assertTrue(status[1].applied)
        self.assertTrue(status[3].blocked)
        self.assertTrue(
            any('already taken' in reason for reason in status[3].reasons)
        )

    def test_blocked_group_leaves_no_partial_writes(self):
        self.db.update({'a': 1, 'b': 2})
        self.db.commit()
        other = SqliteDict(filename=self.fname)
        del other['a']
        other.commit()
        other.close()

        results = self.db.apply_renames([RenameRequest('a', 'a2', 1)])
        self.assertTrue(results[0].blocked)
        self.assertEqual(results[0].line_results, {})
        self.assertEqual(dict(self.db), {'b': 2})

    def test_readonly_dict_refuses_renames(self):
        writer = SqliteDict(filename=self.fname, tablename='ro')
        writer['a'] = 1
        writer.commit()
        writer.close()
        reader = SqliteDict(filename=self.fname, tablename='ro', flag='r')
        with self.assertRaises(RuntimeError):
            reader.apply_renames([RenameRequest('a', 'a2', 1)])
        reader.close()

    def test_ledger_is_persisted_across_connections(self):
        self.db.update({'a': 1})
        self.db.commit()
        requests = [RenameRequest('a', 'a2', 1)]
        first = self.db.apply_renames(requests)[0]
        reopened = SqliteDict(filename=self.fname)
        second = reopened.apply_renames(requests)[0]
        self.assertTrue(second.already_applied)
        self.assertEqual(second.line_results, first.line_results)
        reopened.close()

    def _raw_rows(self):
        conn = sqlite3.connect(self.fname)
        try:
            return {
                key: (rowid, bytes(value))
                for rowid, key, value in conn.execute(
                    'SELECT rowid, key, value FROM "%s"' % self.db.tablename
                )
            }
        finally:
            conn.close()


if __name__ == '__main__':
    unittest.main()
