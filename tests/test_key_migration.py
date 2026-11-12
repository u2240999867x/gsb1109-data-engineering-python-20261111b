"""Tests for the typed-key batch migration (``KeyMigration``).

A legacy table stores raw, human-readable text keys.  These tests cover moving
those rows onto the pickle+base64 typed-key space, including ambiguous numeric
keys, caller-provided claims, interruption/resumption, no-resurrection of
deleted values, and byte-exact reconciliation.
"""
import os
import shutil
import sqlite3
import tempfile
import unittest

import sqlitedict
from sqlitedict import (
    SqliteDict,
    KeyMigration,
    MigrationClaim,
    encode_key,
    decode_key,
    encode,
    decode,
    looks_like_encoded_key,
)


class KeyMigrationTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='sqlitedict-migration-')
        self.path = os.path.join(self.dir, 'm.sqlite')
        self.table = 't'

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def build_legacy(self, mapping):
        """Create a legacy table written with the default text-only codec."""
        legacy = SqliteDict(self.path, tablename=self.table)
        for key, value in mapping.items():
            legacy[key] = value
        legacy.commit()
        legacy.close()

    def open_typed(self, **kwargs):
        kwargs.setdefault('tablename', self.table)
        kwargs.setdefault('encode_key', encode_key)
        kwargs.setdefault('decode_key', decode_key)
        return SqliteDict(self.path, **kwargs)

    def raw_connect(self):
        return sqlite3.connect(self.path)

    def raw_live_keys(self, conn):
        return [row[0] for row in conn.execute('SELECT key FROM "%s"' % self.table)]

    def work_table(self, conn, kind, generation):
        return '"_km_%s_%s_%s"' % (kind, self.table, generation)

    # ------------------------------------------------------------------ #
    # plain migration
    # ------------------------------------------------------------------ #

    def test_plain_text_keys_migrate_fully(self):
        self.build_legacy({'alpha': 1, 'beta': [1, 2], 'gamma': {'x': 'y'}})
        db = self.open_typed()
        report = db.migrate_typed_keys(batch_size=2)

        self.assertTrue(report.complete)
        self.assertEqual(report.migrated, 3)
        self.assertEqual(report.tombstoned, 3)
        self.assertEqual(report.ambiguous, [])
        self.assertEqual(db['alpha'], 1)
        self.assertEqual(db['beta'], [1, 2])
        self.assertEqual(db['gamma'], {'x': 'y'})
        # The live table now contains typed (encoded) keys only.
        conn = self.raw_connect()
        for stored_key in self.raw_live_keys(conn):
            self.assertTrue(looks_like_encoded_key(stored_key))
        conn.close()
        db.close()

    def test_rerun_after_completion_is_noop(self):
        self.build_legacy({'alpha': 1, 'beta': 2})
        db = self.open_typed()
        first = db.migrate_typed_keys()
        second = db.migrate_typed_keys()
        self.assertTrue(first.complete and second.complete)
        self.assertEqual(second.migrated, 0)
        self.assertEqual(second.generation, first.generation)
        self.assertEqual(len(db), 2)
        db.close()

    # ------------------------------------------------------------------ #
    # ambiguity and claims
    # ------------------------------------------------------------------ #

    def test_numeric_key_is_ambiguous_without_claim(self):
        self.build_legacy({'12': 'legacy-twelve', 'name': 'n'})
        db = self.open_typed()
        report = db.migrate_typed_keys(batch_size=10)

        self.assertFalse(report.complete)
        self.assertEqual(report.ambiguous, ['12'])
        self.assertEqual(report.migrated, 1)
        # Neither the integer nor the string typed key may read the old value.
        with self.assertRaises(KeyError):
            db[12]
        with self.assertRaises(KeyError):
            db['12']
        # The un-claimed value is parked out of the live table, not deleted.
        conn = self.raw_connect()
        held = self.work_table(conn, 'held', report.generation)
        self.assertEqual(
            conn.execute('SELECT legacy_key FROM %s' % held).fetchall(),
            [('12',)])
        conn.close()
        db.close()

    def test_claim_forms_are_equivalent(self):
        for claims in (
            {'12': 12},
            [MigrationClaim('12', 12)],
        ):
            path = os.path.join(self.dir, 'claim-%r.sqlite' % type(claims).__name__)
            legacy = SqliteDict(path, tablename=self.table)
            legacy['12'] = 'v'
            legacy.commit()
            legacy.close()

            db = SqliteDict(path, tablename=self.table,
                            encode_key=encode_key, decode_key=decode_key)
            report = db.migrate_typed_keys(claims=claims)
            self.assertTrue(report.complete)
            self.assertEqual(db[12], 'v')
            db.close()

    def test_claim_resolves_ambiguous_key_on_resume(self):
        self.build_legacy({'12': 'legacy-twelve', 'name': 'n'})
        db = self.open_typed()
        first = db.migrate_typed_keys()
        self.assertEqual(first.ambiguous, ['12'])

        second = db.migrate_typed_keys(claims={'12': 12})
        self.assertTrue(second.complete)
        self.assertEqual(second.generation, first.generation)
        self.assertEqual(second.ambiguous, [])
        self.assertEqual(db[12], 'legacy-twelve')
        with self.assertRaises(KeyError):
            db['12']
        db.close()

    def test_int_text_and_composite_keys_coexist(self):
        # The exact scenario: legacy text "12", plus new-source integer 12,
        # text "12" and a composite (12, "12") key.
        self.build_legacy({'12': 'legacy'})
        db = self.open_typed()
        db[12] = 'new-int'
        db['12'] = 'new-text'
        db[(12, '12')] = 'new-composite'
        db.commit()

        # Without a claim the legacy value is held rather than overwriting any
        # of the new-source rows.
        blocked = db.migrate_typed_keys()
        self.assertEqual(blocked.ambiguous, ['12'])
        self.assertEqual(db[12], 'new-int')

        # A fresh manifest claiming the legacy identity as the *string* key is
        # also possible; here we claim a distinct integer target that is free.
        db2_path = os.path.join(self.dir, 'coexist2.sqlite')
        legacy = SqliteDict(db2_path, tablename=self.table)
        legacy['12'] = 'legacy'
        legacy.commit()
        legacy.close()
        db2 = SqliteDict(db2_path, tablename=self.table,
                         encode_key=encode_key, decode_key=decode_key)
        db2['12'] = 'new-text'
        db2[(12, '12')] = 'new-composite'
        db2.commit()
        report = db2.migrate_typed_keys(claims={'12': 12})
        self.assertTrue(report.complete)
        self.assertEqual(db2[12], 'legacy')
        self.assertEqual(db2['12'], 'new-text')
        self.assertEqual(db2[(12, '12')], 'new-composite')
        db2.close()
        db.close()

    def test_collision_holds_loser_and_keeps_first_claimant(self):
        # Both legacy texts resolve to the same integer target.  Pending rows
        # are processed in a stable order, so the earlier key claims the target
        # and the other is held.
        self.build_legacy({'12': 'a', '012': 'b'})
        db = self.open_typed()
        report = db.migrate_typed_keys(claims={'12': 12, '012': 12})
        self.assertEqual(report.migrated, 1)
        self.assertEqual(report.ambiguous, ['12'])
        self.assertEqual(db[12], 'b')  # '012' sorts before '12'
        conn = self.raw_connect()
        held = self.work_table(conn, 'held', report.generation)
        self.assertEqual(
            conn.execute('SELECT legacy_key FROM %s' % held).fetchall(),
            [('12',)])
        conn.close()
        db.close()

    def test_preexisting_typed_row_blocks_legacy_value(self):
        self.build_legacy({'12': 'legacy'})
        db = self.open_typed()
        db[12] = 'new-source'
        db.commit()
        report = db.migrate_typed_keys(claims={'12': 12})
        self.assertEqual(report.ambiguous, ['12'])
        self.assertEqual(db[12], 'new-source')  # never overwritten
        db.close()

    # ------------------------------------------------------------------ #
    # interruption / resumption / no resurrection
    # ------------------------------------------------------------------ #

    def test_resumable_in_batches_keeps_one_generation(self):
        self.build_legacy({'k%d' % i: i for i in range(7)})
        db = self.open_typed()
        partial = db.migrate_typed_keys(batch_size=2, max_batches=1)
        self.assertEqual(partial.migrated, 2)
        self.assertFalse(partial.complete)
        db.close()

        # A brand-new instance resumes the persisted generation.
        db = self.open_typed()
        finished = db.migrate_typed_keys(batch_size=2)
        self.assertTrue(finished.complete)
        self.assertEqual(finished.generation, partial.generation)
        self.assertTrue(finished.resumed)
        self.assertEqual(finished.migrated, 5)
        self.assertEqual(len(db), 7)
        self.assertEqual({db['k%d' % i] for i in range(7)}, set(range(7)))
        db.close()

    def test_repeated_interruptions_preserve_every_row_once(self):
        self.build_legacy({'row%d' % i: i for i in range(10)})
        generation = None
        for _ in range(4):
            db = self.open_typed()
            report = db.migrate_typed_keys(batch_size=2, max_batches=1)
            if generation is None:
                generation = report.generation
            else:
                self.assertEqual(report.generation, generation)
            db.close()

        db = self.open_typed()
        finished = db.migrate_typed_keys(batch_size=2)
        self.assertTrue(finished.complete)
        self.assertEqual(len(db), 10)
        self.assertEqual(
            {db['row%d' % i] for i in range(10)}, set(range(10)))
        # Every original value is tombstoned exactly once.
        conn = self.raw_connect()
        tomb = self.work_table(conn, 'tomb', generation)
        self.assertEqual(
            conn.execute('SELECT COUNT(*) FROM %s' % tomb).fetchone()[0], 10)
        conn.close()
        db.close()

    def test_held_rows_are_not_reprocessed_without_new_claims(self):
        self.build_legacy({'12': 'x', 'a': 1})
        db = self.open_typed()
        first = db.migrate_typed_keys()
        self.assertEqual(first.ambiguous, ['12'])

        # Re-running with identical claims must leave the held row parked and
        # not re-report it each time.
        second = db.migrate_typed_keys()
        self.assertEqual(second.ambiguous, [])
        self.assertFalse(second.complete)

        # A changed manifest releases the held row for another pass.
        third = db.migrate_typed_keys(claims={'12': 12})
        self.assertTrue(third.complete)
        self.assertEqual(db[12], 'x')
        db.close()

    def test_deleted_legacy_value_never_resurrects(self):
        self.build_legacy({'12': 'held', 'alpha': 1, 'beta': 2})
        db = self.open_typed()
        db.migrate_typed_keys(batch_size=1, max_batches=1)
        db.close()

        conn = self.raw_connect()
        # No raw-text keys may remain in the live table: migrated and pending
        # rows alike have been moved into generation-scoped work tables.
        for stored_key in self.raw_live_keys(conn):
            self.assertTrue(looks_like_encoded_key(stored_key))
        self.assertNotIn('12', self.raw_live_keys(conn))

        # Resuming must not copy any legacy row back into the old key space.
        conn.close()
        db = self.open_typed()
        report = db.migrate_typed_keys(batch_size=1)
        conn = self.raw_connect()
        for stored_key in self.raw_live_keys(conn):
            self.assertTrue(looks_like_encoded_key(stored_key))
        conn.close()
        self.assertEqual(report.migrated, 2)  # alpha, beta only; '12' stays held
        db.close()

    def test_held_rows_preserve_raw_bytes(self):
        payload = {'nested': ['bytes', 1, 2]}
        self.build_legacy({'12': payload})
        db = self.open_typed()
        report = db.migrate_typed_keys()
        generation = report.generation

        conn = self.raw_connect()
        held = self.work_table(conn, 'held', generation)
        row = conn.execute('SELECT value FROM %s WHERE legacy_key = ?' % held,
                           ('12',)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(decode(row[0]), payload)
        conn.close()

        db.migrate_typed_keys(claims={'12': 12})
        self.assertEqual(db[12], payload)
        db.close()

    # ------------------------------------------------------------------ #
    # generation alignment
    # ------------------------------------------------------------------ #

    def test_all_work_tables_target_persisted_generation(self):
        self.build_legacy({'12': 'x', 'a': 1})
        db = self.open_typed()
        report = db.migrate_typed_keys(batch_size=1)
        generation = report.generation
        conn = self.raw_connect()
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for kind in ('pending', 'held', 'tomb'):
            self.assertIn('_km_%s_%s_%s' % (kind, self.table, generation), names)
        # Metadata pins the same generation.
        meta = conn.execute(
            'SELECT generation FROM "%s" WHERE tablename = ?'
            % sqlitedict._MIGRATION_META_TABLE, (self.table,)).fetchone()
        self.assertEqual(meta[0], generation)
        conn.close()
        db.close()

    # ------------------------------------------------------------------ #
    # byte-exact reconciliation
    # ------------------------------------------------------------------ #

    def test_reconcile_ok_for_claimed_migrated_keys(self):
        self.build_legacy({'12': 'twelve', '7': 'seven', 'name': 'n'})
        db = self.open_typed()
        db.migrate_typed_keys(claims={'12': 12, '7': '7', 'name': 'name'})
        km = KeyMigration(db, claims={'12': 12, '7': '7', 'name': 'name'})
        statuses = {legacy: status for legacy, _, status in km.reconcile()}
        self.assertEqual(statuses, {'12': 'ok', '7': 'ok', 'name': 'ok'})
        db.close()

    def test_reconcile_detects_mismatch_and_missing(self):
        self.build_legacy({'name': 'original'})
        db = self.open_typed()
        db.migrate_typed_keys(claims={'name': 'name'})

        # Tamper with the live typed value: tombstone bytes no longer match.
        db['name'] = 'tampered'
        db.commit()
        km = KeyMigration(db, claims={'name': 'name'})
        self.assertEqual(km.reconcile(), [('name', 'name', 'mismatch')])

        # Removing the typed row surfaces as missing-typed.
        del db['name']
        db.commit()
        self.assertEqual(km.reconcile(), [('name', 'name', 'missing-typed')])
        db.close()

    def test_reconcile_reports_unmigrated_held_key(self):
        self.build_legacy({'12': 'x', 'name': 'n'})
        db = self.open_typed()
        db.migrate_typed_keys()  # '12' stays held
        km = KeyMigration(db, claims={'12': 12})
        self.assertEqual(km.reconcile(), [('12', 12, 'not-migrated')])
        db.close()

    def test_tombstone_bytes_equal_original_value_bytes(self):
        self.build_legacy({'name': 'n'})
        db = self.open_typed()
        report = db.migrate_typed_keys(claims={'name': 'name'})
        conn = self.raw_connect()
        tomb = self.work_table(conn, 'tomb', report.generation)
        original = encode('n')
        stored = conn.execute(
            'SELECT value FROM %s WHERE legacy_key = ?' % tomb,
            ('name',)).fetchone()[0]
        self.assertEqual(bytes(stored), bytes(original))
        conn.close()
        db.close()

    # ------------------------------------------------------------------ #
    # validation / guards
    # ------------------------------------------------------------------ #

    def test_guards(self):
        self.build_legacy({'a': 1})

        readonly = self.open_typed(flag='r')
        with self.assertRaises(RuntimeError):
            KeyMigration(readonly)
        readonly.close()

        plain = SqliteDict(os.path.join(self.dir, 'plain.sqlite'))
        with self.assertRaises(RuntimeError):
            KeyMigration(plain)
        with self.assertRaises(RuntimeError):
            plain.migrate_typed_keys()
        plain.close()

        typed = self.open_typed()
        with self.assertRaises(ValueError):
            KeyMigration(typed, batch_size=0)
        with self.assertRaises(TypeError):
            KeyMigration(typed, claims=[MigrationClaim(12, 12)])
        with self.assertRaises(TypeError):
            KeyMigration('not a dict')
        typed.close()

    def test_codec_detector(self):
        self.assertTrue(looks_like_encoded_key(encode_key(12)))
        self.assertTrue(looks_like_encoded_key(encode_key('12')))
        self.assertTrue(looks_like_encoded_key(encode_key((12, '12'))))
        for not_encoded in ('12', 'abc', 'gAVLDC4', '!!!!', '', 'aW5jb2JyZWN0'):
            self.assertFalse(looks_like_encoded_key(not_encoded), not_encoded)


if __name__ == '__main__':
    unittest.main()
