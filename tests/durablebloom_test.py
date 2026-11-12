#!/usr/bin/env python3
"""Tests for the durable, cross-process :class:`DurableBloomFilter`.

The tests pin the persistence contract described in the module docstring:
exactly-once batch confirmation, lossless concurrent byte-OR merging, explicit
partition-generation fencing (no timeouts), stable generation snapshots, and
all-or-nothing recovery after a machine interruption.
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from struct import Struct

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import DurableBloomFilter, DurableBloomRecoveryError, GenerationReader, StaleWriterError  # noqa: E402
from probables.blooms.durablebloom import (  # noqa: E402
    _BATCH_TRAILER,
    _RECORD_PREFIX,
    _RECORD_STRUCT,
)
from probables.hashes import default_fnv_1a  # noqa: E402

EST = 1000
FPR = 0.05


def _layer_geometry(directory, generation):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    meta = manifest["layers"][str(generation)]
    return meta["num_bits"], meta["number_hashes"], meta["byte_length"]


def _craft_journal(path, generation, key, geometry, event_count=1):
    """Write a valid (but unconfirmed) batch journal for one key."""
    num_bits, number_hashes, _byte_length = geometry
    touched = {}
    for hsh in default_fnv_1a(key, number_hashes):
        bit = hsh % num_bits
        touched[bit // 8] = touched.get(bit // 8, 0) | (1 << (bit % 8))
    payload = Struct("<QQ").pack(event_count, len(touched))
    for byte_index, mask in sorted(touched.items()):
        payload += Struct("<QB").pack(byte_index, mask)
    payload += _BATCH_TRAILER
    crc = zlib.crc32(_RECORD_PREFIX.pack(generation, len(payload)))
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    Path(path).write_bytes(_RECORD_STRUCT.pack(generation, len(payload), crc, 0) + payload)
    return touched


def _keys_sharing_one_byte(dbf):
    """Find two keys that set different bits inside the same layer byte."""
    num_bits = dbf._active._num_bits  # noqa: SLF001
    chosen = []
    target_byte = None
    index = 0
    while len(chosen) < 2:
        key = f"shared-byte-{index}"
        index += 1
        bit = default_fnv_1a(key, 1)[0] % num_bits
        byte_index, bit_index = bit // 8, bit % 8
        if target_byte is None:
            target_byte = byte_index
            chosen.append((key, bit_index))
        elif byte_index == target_byte and all(bit_index != b for _k, b in chosen):
            chosen.append((key, bit_index))
        if index > 100000:
            raise AssertionError("could not find two keys sharing a byte")
    return target_byte, chosen


class TestDurableBloomFilter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _open(self, partition_generation=0, **kwargs):
        params = {"est_elements": EST, "false_positive_rate": FPR}
        params.update(kwargs)
        return DurableBloomFilter(self.dir, partition_generation=partition_generation, **params)

    # ------------------------------------------------------------------ #
    def test_create_layout_and_byte_boundaries(self):
        dbf = self._open()
        self.assertEqual(dbf.generation, 0)
        self.assertEqual(dbf.confirmed_batches, 0)
        self.assertEqual(dbf.confirmed_events, 0)
        manifest = json.loads((Path(self.dir) / "manifest.json").read_text())
        meta = manifest["layers"]["0"]
        layer = Path(self.dir) / "gen" / "g0.layer"
        # The layer file occupies exactly the metadata-declared byte boundary.
        self.assertEqual(layer.stat().st_size, meta["byte_length"])
        self.assertTrue((Path(self.dir) / "lock").exists())
        self.assertTrue((Path(self.dir) / "gen" / "g0.meta").exists())
        dbf.close()

    def test_batch_is_confirmed_exactly_once_on_replay(self):
        dbf = self._open()
        keys = ["a", "b", "c", "d"]
        self.assertTrue(dbf.submit_batch(keys, "batch-1"))
        # Re-sending the same batch id changes neither membership nor counts.
        self.assertFalse(dbf.submit_batch(keys, "batch-1"))
        self.assertEqual(dbf.confirmed_batches, 1)
        self.assertEqual(dbf.confirmed_events, 4)
        for key in keys:
            self.assertIn(key, dbf)
        self.assertNotIn("never-inserted-0001", dbf)
        dbf.close()

        reopened = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertEqual(reopened.confirmed_batches, 1)
        self.assertEqual(reopened.confirmed_events, 4)
        # Even after a process restart the replay stays a no-op.
        self.assertFalse(reopened.submit_batch(keys, "batch-1"))
        self.assertEqual(reopened.confirmed_batches, 1)
        reopened.close()

    def test_persists_across_clean_shutdown(self):
        dbf = self._open()
        dbf.submit_batch(["persist-me", "and-me"], "p1")
        dbf.close()

        reopened = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertTrue(reopened.check("persist-me"))
        self.assertTrue(reopened.check("and-me"))
        self.assertEqual(reopened.confirmed_batches, 1)
        reopened.close()

    def test_concurrent_writers_merge_bits_in_same_byte(self):
        writer_a = self._open()
        byte_index, (first, second) = _keys_sharing_one_byte(writer_a)
        writer_a.submit_batch([first[0]], "writer-a")

        writer_b = DurableBloomFilter(self.dir, partition_generation=0)
        writer_b.submit_batch([second[0]], "writer-b")

        layer = (Path(self.dir) / "gen" / "g0.layer").read_bytes()
        expected = (1 << first[1]) | (1 << second[1])
        # Neither writer overwrote the other's bit inside the shared byte.
        self.assertEqual(layer[byte_index], expected)
        self.assertTrue(writer_a.check(first[0]))
        self.assertTrue(writer_b.check(second[0]))
        writer_a.close()
        writer_b.close()

    # ------------------------------------------------------------------ #
    def test_takeover_fences_late_writer_without_timeout(self):
        original = self._open()
        original.submit_batch(["old-era-event"], "old")
        original.close()

        late_writer = DurableBloomFilter(self.dir, partition_generation=1)
        new_owner = DurableBloomFilter(self.dir, partition_generation=1)
        self.assertEqual(new_owner.takeover(5), 5)
        self.assertEqual(new_owner.sealed_generations, [0])

        # The caller explicitly bumped the generation; the stale writer fails.
        with self.assertRaises(StaleWriterError):
            late_writer.submit_batch(["late-event"], "late")
        new_owner.submit_batch(["new-era-event"], "new")
        new_owner.close()

        # A stale writer closing must not roll the persisted generation back.
        late_writer.close()
        self.assertEqual(json.loads((Path(self.dir) / "manifest.json").read_text())["generation"], 5)

    def test_open_with_stale_generation_is_rejected(self):
        dbf = self._open()
        dbf.takeover(3)
        dbf.close()
        with self.assertRaises(StaleWriterError):
            DurableBloomFilter(self.dir, partition_generation=2)

    def test_takeover_requires_strictly_greater_generation(self):
        dbf = self._open(partition_generation=4)
        with self.assertRaises(StaleWriterError):
            dbf.takeover(4)
        dbf.close()

    def test_late_writer_cannot_overwrite_takeover_data(self):
        first = self._open()
        first.takeover(2)
        first.submit_batch(["owned-by-gen-2"], "g2")
        first.close()

        # A writer presenting the superseded generation is fenced at open time.
        with self.assertRaises(StaleWriterError):
            DurableBloomFilter(self.dir, partition_generation=1)

        checker = DurableBloomFilter(self.dir, partition_generation=2)
        self.assertTrue(checker.check("owned-by-gen-2"))
        self.assertFalse(checker.check("must-not-land"))
        checker.close()

    # ------------------------------------------------------------------ #
    def test_grow_keeps_old_layers_membership_valid(self):
        dbf = self._open()
        dbf.submit_batch(["first-generation-key"], "g0-batch")
        old_layer = (Path(self.dir) / "gen" / "g0.layer").read_bytes()

        new_generation = dbf.grow(est_elements=5000, false_positive_rate=0.02)
        self.assertEqual(new_generation, 1)
        self.assertEqual(dbf.sealed_generations, [0])
        # Growing never rewrites or copies the old bit table.
        self.assertEqual((Path(self.dir) / "gen" / "g0.layer").read_bytes(), old_layer)

        # The old key stays a member even though its key was never re-read.
        self.assertTrue(dbf.check("first-generation-key"))
        dbf.submit_batch(["second-generation-key"], "g1-batch")
        self.assertTrue(dbf.check("second-generation-key"))
        dbf.close()

        reopened = DurableBloomFilter(self.dir, partition_generation=1)
        self.assertTrue(reopened.check("first-generation-key"))
        self.assertTrue(reopened.check("second-generation-key"))
        reopened.close()

    def test_reader_snapshot_sees_complete_old_or_new_generation(self):
        dbf = self._open()
        dbf.submit_batch(["snapshot-key"], "snap")
        old_reader = dbf.open_reader(generation=0)
        self.assertIsInstance(old_reader, GenerationReader)
        self.assertTrue(old_reader.check("snapshot-key"))

        # A grow publishes a new generation while the reader stays on the old.
        dbf.grow(est_elements=5000, false_positive_rate=0.02)
        dbf.submit_batch(["new-only-key"], "new")

        # The held reader still observes the complete, immutable old generation.
        self.assertEqual(old_reader.generation, 0)
        self.assertTrue(old_reader.check("snapshot-key"))
        self.assertFalse(old_reader.check("new-only-key"))
        old_reader.close()

        # A freshly opened reader observes the complete new generation.
        new_reader = dbf.open_reader()
        self.assertEqual(new_reader.generation, 1)
        self.assertTrue(new_reader.check("new-only-key"))
        new_reader.close()
        dbf.close()

    def test_open_reader_unknown_generation_raises(self):
        dbf = self._open()
        with self.assertRaises(Exception):
            dbf.open_reader(generation=99)
        dbf.close()

    # ------------------------------------------------------------------ #
    def test_recovery_intact_unconfirmed_batch_is_whole_in(self):
        dbf = self._open()
        dbf.submit_batch(["confirmed-key"], "confirmed")
        dbf.close()

        geometry = _layer_geometry(self.dir, 0)
        _craft_journal(Path(self.dir) / "journal" / "pending-good.dbf", 0, "pending-good", geometry)

        recovered = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertTrue(recovered.check("confirmed-key"))
        self.assertTrue(recovered.check("pending-good"))
        entry = recovered._manifest["confirmed"]["pending-good"]  # noqa: SLF001
        self.assertTrue(entry["recovered"])
        # Both batches are counted exactly once, including the recovered one.
        self.assertEqual(recovered.confirmed_batches, 2)
        self.assertEqual(recovered.confirmed_events, 2)
        recovered.close()

    def test_recovery_torn_unconfirmed_batch_is_whole_out(self):
        dbf = self._open()
        dbf.submit_batch(["confirmed-key"], "confirmed")
        dbf.close()

        geometry = _layer_geometry(self.dir, 0)
        journal = Path(self.dir) / "journal" / "pending-torn.dbf"
        _craft_journal(journal, 0, "pending-torn", geometry)
        journal.write_bytes(journal.read_bytes()[: len(journal.read_bytes()) // 2])

        recovered = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertFalse(recovered.check("pending-torn"))
        self.assertFalse(journal.exists())
        # The confirmed batch survives and is still counted exactly once.
        self.assertTrue(recovered.check("confirmed-key"))
        self.assertEqual(recovered.confirmed_batches, 1)
        recovered.close()

    def test_recovery_crc_corrupt_batch_is_discarded(self):
        dbf = self._open()
        dbf.close()
        geometry = _layer_geometry(self.dir, 0)
        journal = Path(self.dir) / "journal" / "pending-crc.dbf"
        _craft_journal(journal, 0, "pending-crc", geometry)
        corrupted = bytearray(journal.read_bytes())
        corrupted[-1] ^= 0xFF
        journal.write_bytes(bytes(corrupted))

        recovered = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertFalse(recovered.check("pending-crc"))
        self.assertFalse(journal.exists())
        recovered.close()

    def test_recovery_detects_truncated_layer_byte_boundary(self):
        dbf = self._open()
        dbf.submit_batch(["confirmed-key"], "confirmed")
        dbf.close()

        layer = Path(self.dir) / "gen" / "g0.layer"
        original = layer.read_bytes()
        layer.write_bytes(original[:-5])
        with self.assertRaises(DurableBloomRecoveryError):
            DurableBloomFilter(self.dir, partition_generation=0)
        layer.write_bytes(original)

    def test_superseded_writer_journal_is_not_applied(self):
        dbf = self._open()
        dbf.close()
        geometry = _layer_geometry(self.dir, 0)
        # A journal tagged to a generation that was never published.
        journal = Path(self.dir) / "journal" / "ghost.dbf"
        _craft_journal(journal, 42, "ghost-key", geometry)
        recovered = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertFalse(recovered.check("ghost-key"))
        self.assertFalse(journal.exists())
        recovered.close()

    def test_context_manager_closes_cleanly(self):
        with self._open() as dbf:
            dbf.submit_batch(["ctx"], "ctx-batch")
        # Reopening works, proving close() left a coherent state.
        reopened = DurableBloomFilter(self.dir, partition_generation=0)
        self.assertTrue(reopened.check("ctx"))
        reopened.close()

    # ------------------------------------------------------------------ #
    # Reviewable byte boundaries and a fully controlled takeover interleave
    # ------------------------------------------------------------------ #
    def test_journal_persistence_byte_boundary_and_crc(self):
        dbf = self._open()
        dbf.submit_batch(["boundary-key"], "boundary")
        dbf.close()

        journal = Path(self.dir) / "journal" / "boundary.dbf"
        raw = journal.read_bytes()
        header_size = _RECORD_STRUCT.size
        # Fixed header, then exactly the declared payload byte count.
        generation, length, crc, reserved = _RECORD_STRUCT.unpack(raw[:header_size])
        self.assertEqual(len(raw), header_size + length)
        self.assertEqual(generation, 0)
        self.assertEqual(reserved, 0)
        # The stored CRC must be independently reproducible from the bytes.
        payload = raw[header_size:]
        recomputed = zlib.crc32(_RECORD_PREFIX.pack(generation, length))
        recomputed = zlib.crc32(payload, recomputed) & 0xFFFFFFFF
        self.assertEqual(crc, recomputed)
        self.assertTrue(payload.endswith(_BATCH_TRAILER))

    def test_small_batch_does_not_copy_the_whole_bit_table(self):
        dbf = DurableBloomFilter(self.dir, est_elements=10000, false_positive_rate=0.01)
        layer_size = (Path(self.dir) / "gen" / "g0.layer").stat().st_size
        dbf.submit_batch(["one-little-key"], "small")
        dbf.close()
        journal_size = (Path(self.dir) / "journal" / "small.dbf").stat().st_size
        # A single-key batch only journals its handful of touched bytes.
        self.assertLess(journal_size, layer_size // 4)

    def test_controlled_takeover_interleave_counts(self):
        # Step 1: the original writer confirms batch A on generation 0.
        original = self._open()
        original.submit_batch(["event-A"], "batch-A")
        original.close()

        # Step 2: a long-lived writer (the soon-to-be-late writer) is open.
        late = DurableBloomFilter(self.dir, partition_generation=1)

        # Step 3: another process explicitly takes over at generation 4.
        new_owner = DurableBloomFilter(self.dir, partition_generation=1)
        new_owner.takeover(4)
        new_owner.submit_batch(["event-B"], "batch-B")

        # Step 4: the late writer's resend is fenced, not silently overwritten.
        with self.assertRaises(StaleWriterError):
            late.submit_batch(["event-late"], "batch-late")
        new_owner.close()
        late.close()

        # Step 5: reconcile membership and the exactly-once batch counters.
        final = DurableBloomFilter(self.dir, partition_generation=4)
        self.assertEqual(final.confirmed_batches, 2)
        self.assertEqual(final.confirmed_events, 2)
        self.assertTrue(final.check("event-A"))
        self.assertTrue(final.check("event-B"))
        self.assertFalse(final.check("event-late"))
        final.close()


def _mp_worker(directory, proc, batches):
    dbf = DurableBloomFilter(directory, partition_generation=0)
    for batch in range(batches):
        batch_id = f"p{proc}-b{batch}"
        keys = [f"ev-{proc}-{batch}-{i}" for i in range(5)]
        assert dbf.submit_batch(keys, batch_id) is True
        assert dbf.submit_batch(keys, batch_id) is False
    dbf.close()


@unittest.skipUnless("fork" in mp.get_all_start_methods(), "fork start method required")
class TestDurableBloomMultiprocess(unittest.TestCase):
    def test_flock_serialises_writers_and_keeps_exactly_once_counts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = tmp.name
        bootstrap = DurableBloomFilter(directory, est_elements=5000, false_positive_rate=0.02)
        bootstrap.close()

        nproc, batches = 6, 20
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=_mp_worker, args=(directory, p, batches)) for p in range(nproc)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        self.assertEqual([proc.exitcode for proc in procs], [0] * nproc)

        checker = DurableBloomFilter(directory, partition_generation=0)
        self.assertEqual(checker.confirmed_batches, nproc * batches)
        self.assertEqual(checker.confirmed_events, nproc * batches * 5)
        missing = [
            f"ev-{p}-{b}-{i}"
            for p in range(nproc)
            for b in range(batches)
            for i in range(5)
            if not checker.check(f"ev-{p}-{b}-{i}")
        ]
        self.assertEqual(missing, [])
        checker.close()


if __name__ == "__main__":
    unittest.main()
