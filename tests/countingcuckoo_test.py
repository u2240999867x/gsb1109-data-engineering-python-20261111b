#!/usr/bin/env python3
"""Unittest class"""

import hashlib
import os
import random
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import CountingCuckooFilter, CuckooFilterFullError  # noqa: E402
from probables.cuckoo.countingcuckoo import CountingCuckooBin  # noqa: E402
from tests.utilities import (  # noqa: E402
    calc_file_md5,
    find_unpaired_kicks,
    reconcile_counting_cuckoo,
    trace_cuckoo_writes,
)

DELETE_TEMP_FILES = True


class TestCountingCuckooFilter(unittest.TestCase):
    """base Cuckoo Filter test"""

    def test_c_cuckoo_filter_default(self):
        """test counting cuckoo filter default properties"""
        cko = CountingCuckooFilter()
        self.assertEqual(10000, cko.capacity)
        self.assertEqual(4, cko.bucket_size)
        self.assertEqual(500, cko.max_swaps)
        self.assertEqual(2, cko.expansion_rate)
        self.assertEqual(True, cko.auto_expand)

    def test_c_cuckoo_filter_diff(self):
        """test counting cuckoo filter non-standard properties"""
        cko = CountingCuckooFilter(
            capacity=100,
            bucket_size=2,
            max_swaps=5,
            expansion_rate=4,
            auto_expand=False,
        )
        self.assertEqual(100, cko.capacity)
        self.assertEqual(2, cko.bucket_size)
        self.assertEqual(5, cko.max_swaps)
        self.assertEqual(4, cko.expansion_rate)
        self.assertEqual(False, cko.auto_expand)

    def test_c_cuckoo_filter_add(self):
        """test adding to the counting cuckoo filter"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 1)
        cko.add("this is another test")
        self.assertEqual(cko.elements_added, 2)
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)

    def test_c_cuckoo_filter_remove(self):
        """test removing from the counting cuckoo filter"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 1)
        cko.add("this is another test")
        self.assertEqual(cko.elements_added, 2)
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)
        self.assertEqual(cko.unique_elements, 3)
        cko.add("this is a test")
        cko.add("this is a test")
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 6)
        self.assertEqual(cko.unique_elements, 3)

        res = cko.remove("this is another test")
        self.assertTrue(res)
        self.assertEqual(cko.elements_added, 5)
        self.assertEqual(cko.unique_elements, 2)

        self.assertTrue(cko.check("this is a test"))
        self.assertFalse(cko.check("this is another test"))
        self.assertTrue(cko.check("this is yet another test"))

    def test_c_cuckoo_filter_rmv_miss(self):
        """test removing from the counting cuckoo filter when not present"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 1)
        cko.add("this is another test")
        self.assertEqual(cko.elements_added, 2)
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)

        res = cko.remove("this is still a test")
        self.assertFalse(res)
        self.assertEqual(cko.elements_added, 3)
        self.assertTrue(cko.check("this is a test"))
        self.assertTrue(cko.check("this is another test"))
        self.assertTrue(cko.check("this is yet another test"))

    def test_c_cuckoo_filter_lots(self):
        """test inserting lots into the counting cuckoo filter"""
        cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=100)
        for i in range(125):
            cko.add(str(i))
        self.assertEqual(cko.elements_added, 125)

    def test_c_cuckoo_filter_full(self):
        """test inserting until counting cuckoo filter is full"""

        def runner():
            """runner"""
            cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=100, auto_expand=False)
            for i in range(175):
                cko.add(str(i))

        self.assertRaises(CuckooFilterFullError, runner)

    def test_c_cuckoo_full_msg(self):
        """test exception message for full counting cuckoo filter"""
        try:
            cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=100, auto_expand=False)
            for i in range(175):
                cko.add(str(i))
        except CuckooFilterFullError as ex:
            msg = "The CountingCuckooFilter is currently full"
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_c_cuckoo_filter_full_preserves_elements(self):
        """test that a failed insertion does not remove previously added elements"""
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)
        cko = CountingCuckooFilter(capacity=4, bucket_size=1, max_swaps=1, auto_expand=False)
        added = ["k54", "k24", "k48", "k56"]
        for key in added:
            cko.add(key)
        self.assertRaises(CuckooFilterFullError, cko.add, "k36")
        self.assertEqual(len(added), cko.elements_added)
        for key in added:
            self.assertTrue(cko.check(key))

    def test_c_cuckoo_filter_failed_expand_preserves_elements(self):
        """test that a failed expansion does not remove previously added elements"""
        self.addCleanup(random.setstate, random.getstate())
        random.seed(15)
        cko = CountingCuckooFilter(capacity=8, bucket_size=1, max_swaps=2)
        added = ["k{}".format(i) for i in range(18)]
        for key in added:
            cko.add(key)
        self.assertEqual(32, cko.capacity)  # earlier expansions succeeded
        self.assertRaises(CuckooFilterFullError, cko.add, "k18")
        self.assertEqual(32, cko.capacity)  # the failed expansion is rolled back
        for key in added:
            self.assertTrue(cko.check(key))

    def test_c_cuckoo_idx(self):
        """test that the indexing works correctly for counting cuckoo filter
        swap"""
        cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=5)
        txt = "this is a test"
        idx_1, idx_2, fingerprint = cko._generate_fingerprint_info(txt)
        index_1, index_2 = cko._indicies_from_fingerprint(fingerprint)
        self.assertEqual(idx_1, index_1)
        self.assertEqual(idx_2, index_2)

    def test_c_cuckoo_filter_check(self):
        """test checking if element in counting cuckoo filter"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        cko.add("this is another test")
        cko.add("this is yet another test")
        self.assertEqual(cko.check("this is a test"), True)
        self.assertEqual(cko.check("this is another test"), True)
        self.assertEqual(cko.check("this is yet another test"), True)
        self.assertEqual(cko.check("this is not another test"), False)
        self.assertEqual(cko.check("this is not a test"), False)

    def test_c_cuckoo_filter_in(self):
        """test checking using 'in' counting cuckoo filter"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        cko.add("this is another test")
        cko.add("this is yet another test")
        self.assertEqual("this is a test" in cko, True)
        self.assertEqual("this is another test" in cko, True)
        self.assertEqual("this is yet another test" in cko, True)
        self.assertEqual("this is not another test" in cko, False)
        self.assertEqual("this is not a test" in cko, False)

    def test_c_cuckoo_filter_dup_add(self):
        """test adding same item multiple times counting cuckoo filter"""
        cko = CountingCuckooFilter()
        cko.add("this is a test")
        cko.add("this is another test")
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)
        cko.add("this is a test")
        cko.add("this is another test")
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 6)
        self.assertEqual(cko.unique_elements, 3)

    def test_c_cuckoo_filter_l_fact(self):
        """test the load factor of the counting cuckoo filter"""
        cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=10)
        self.assertEqual(cko.load_factor(), 0.0)
        for i in range(50):
            cko.add(str(i))
        self.assertEqual(cko.load_factor(), 0.25)
        for i in range(50):
            cko.add(str(i + 50))

        if cko.capacity == 200:  # self expanded
            self.assertEqual(cko.load_factor(), 0.25)
        else:
            self.assertEqual(cko.load_factor(), 0.50)

        for i in range(100):
            cko.add(str(i))
        if cko.capacity == 200:  # self expanded
            self.assertEqual(cko.load_factor(), 0.25)
        else:
            self.assertEqual(cko.load_factor(), 0.50)

    def test_c_cuckoo_filter_export(self):
        """test exporting a counting cuckoo filter"""
        md5sum = "6a98c2df1ec9fbb4f75f8e6392696b9b"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cck", delete=DELETE_TEMP_FILES) as fobj:
            cko = CountingCuckooFilter(capacity=1000, bucket_size=2, auto_expand=False)
            for i in range(100):
                cko.add(str(i))

            cko.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5sum, md5_out)

    def test_c_cuckoo_filter_bytes(self):
        """test exporting a counting cuckoo filter"""
        md5sum = "6a98c2df1ec9fbb4f75f8e6392696b9b"
        cko = CountingCuckooFilter(capacity=1000, bucket_size=2, auto_expand=False)
        for i in range(100):
            cko.add(str(i))
        md5_out = hashlib.md5(bytes(cko)).hexdigest()
        self.assertEqual(md5sum, md5_out)

    def test_c_cuckoo_filter_frombytes(self):
        """test initializing a counting cuckoo filter frombytes"""
        cko = CountingCuckooFilter(capacity=1000, bucket_size=2, auto_expand=False)
        for i in range(100):
            cko.add(str(i))
        bytes_out = bytes(cko)

        cko2 = CountingCuckooFilter.frombytes(bytes_out)

        self.assertEqual(bytes_out, bytes(cko2))
        for i in range(100):
            self.assertTrue(cko2.check(str(i)))
        self.assertFalse(cko2.check("999"))

    def test_c_cuckoo_filter_load(self):
        """test loading a saved counting cuckoo filter"""
        md5sum = "6a98c2df1ec9fbb4f75f8e6392696b9b"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cck", delete=DELETE_TEMP_FILES) as fobj:
            cko = CountingCuckooFilter(capacity=1000, bucket_size=2, auto_expand=False)
            for i in range(100):
                cko.add(str(i))

            cko.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5sum, md5_out)

            ckf = CountingCuckooFilter(filepath=fobj.name)
            for i in range(100):
                self.assertEqual(ckf.check(str(i)), 1)

            self.assertEqual(1000, ckf.capacity)
            self.assertEqual(2, ckf.bucket_size)
            self.assertEqual(500, ckf.max_swaps)
            self.assertEqual(0.05, ckf.load_factor())

    def test_c_cuckoo_filter_expand_els(self):
        """test out the expansion of the counting cuckoo filter"""
        cko = CountingCuckooFilter()
        for i in range(200):
            cko.add(str(i))
        cko.expand()
        for i in range(200):
            self.assertGreater(cko.check(str(i)), 0)
        self.assertEqual(20000, cko.capacity)

    def test_c_cuckoo_filter_expand_duplicates(self):
        """test that expanding preserves the count of duplicate keys"""
        cko = CountingCuckooFilter(capacity=10)
        for _ in range(3):
            cko.add("a")
        for i in range(20):
            cko.add(str(i))
        total = cko.elements_added
        cko.expand()
        self.assertEqual(total, cko.elements_added)
        self.assertEqual(3, cko.check("a"))
        for _ in range(3):
            cko.remove("a")
        self.assertEqual(total - 3, cko.elements_added)

    def test_c_cuckoo_filter_auto_exp(self):
        """test inserting until counting cuckoo filter is full"""
        cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=100)
        for i in range(375):  # this would fail if it doesn't expand
            cko.add(str(i))
        self.assertEqual(400, cko.capacity)
        self.assertEqual(375, cko.elements_added)
        for i in range(375):
            self.assertGreater(cko.check(str(i)), 0)

    def test_c_cuckoo_filter_bin(self):
        """test the cuckoo bin repr"""
        cko = CountingCuckooFilter(capacity=1, bucket_size=2, max_swaps=100)
        cko.add("this is a test")
        self.assertEqual("[(fingerprint:4280557824 count:1)]", str(cko.buckets[0]))

    def test_c_cuckoo_filter_str(self):
        """test the str representation of the counting cuckoo filter"""
        cko = CountingCuckooFilter(capacity=100, bucket_size=2, max_swaps=100)
        for i in range(75):
            cko.add(str(i))
        msg = (
            "CountingCuckooFilter:\n"
            "\tCapacity: 100\n"
            "\tTotal Bins: 200\n"
            "\tLoad Factor: 37.5%\n"
            "\tInserted Elements: 75\n"
            "\tMax Swaps: 100\n"
            "\tExpansion Rate: 2\n"
            "\tAuto Expand: True"
        )
        self.assertEqual(str(cko), msg)


class TestCuckooFilterErrorRate(unittest.TestCase):
    """Test CountingCuckooFilter using Error Rate"""

    def test_c_cuckoo_filter_er_default(self):
        """test cuckoo filter default properties"""
        cko = CountingCuckooFilter.init_error_rate(0.00001)
        self.assertEqual(10000, cko.capacity)
        self.assertEqual(4, cko.bucket_size)
        self.assertEqual(500, cko.max_swaps)
        self.assertEqual(2, cko.expansion_rate)
        self.assertEqual(True, cko.auto_expand)
        self.assertEqual(3, cko.fingerprint_size)
        self.assertEqual(20, cko.fingerprint_size_bits)
        self.assertEqual(0.00001, cko.error_rate)

    def test_c_cuckoo_filter_er_add_check(self):
        """test adding to the cuckoo filter"""
        cko = CountingCuckooFilter.init_error_rate(0.00001)
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 1)
        cko.add("this is another test")
        self.assertEqual(cko.elements_added, 2)
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)

        # check
        self.assertEqual(cko.check("this is a test"), True)
        self.assertEqual(cko.check("this is another test"), True)
        self.assertEqual(cko.check("this is yet another test"), True)
        self.assertEqual(cko.check("this is not another test"), False)
        self.assertEqual(cko.check("this is not a test"), False)

        # use of `in`
        self.assertEqual("this is a test" in cko, True)
        self.assertEqual("this is another test" in cko, True)
        self.assertEqual("this is yet another test" in cko, True)
        self.assertEqual("this is not another test" in cko, False)
        self.assertEqual("this is not a test" in cko, False)

    def test_c_cuckoo_filter_er_export(self):
        """test exporting a cuckoo filter"""
        md5sum = "f68767bd97b21426f5d2315fb38961ad"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cko", delete=DELETE_TEMP_FILES) as fobj:
            cko = CountingCuckooFilter.init_error_rate(0.00001)
            for i in range(1000):
                cko.add(str(i))
            cko.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5sum, md5_out)

    def test_c_cuckoo_filter_load(self):
        """test loading a saved cuckoo filter"""
        md5sum = "88bc3a08bfc967f9ba60e9d57c21207f"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cko", delete=DELETE_TEMP_FILES) as fobj:
            cko = CountingCuckooFilter.init_error_rate(0.00001)
            for i in range(1000):
                cko.add(str(i))
                if i % 2 == 1:
                    cko.add(str(i))
            cko.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5sum, md5_out)

            ckf = CountingCuckooFilter.load_error_rate(error_rate=0.00001, filepath=fobj.name)
            for i in range(1000):
                self.assertEqual(ckf.check(str(i)), (i % 2) + 1)

            self.assertEqual(10000, ckf.capacity)
            self.assertEqual(4, ckf.bucket_size)
            self.assertEqual(500, ckf.max_swaps)
            self.assertEqual(2, ckf.expansion_rate)
            self.assertEqual(True, ckf.auto_expand)
            self.assertEqual(20, ckf.fingerprint_size_bits)
            self.assertEqual(3, ckf.fingerprint_size)
            self.assertEqual(0.00001, ckf.error_rate)
            self.assertEqual(0.025, ckf.load_factor())

    def test_c_cuckoo_filter_er_bytes(self):
        """test exporting a cuckoo filter to bytes"""
        md5sum = "f68767bd97b21426f5d2315fb38961ad"
        cko = CountingCuckooFilter.init_error_rate(0.00001)
        for i in range(1000):
            cko.add(str(i))
        md5_out = hashlib.md5(bytes(cko)).hexdigest()
        self.assertEqual(md5sum, md5_out)

    def test_c_cuckoo_filter_er_frombytes(self):
        """test initializing a couting cuckoo filter from bytes"""
        cko = CountingCuckooFilter.init_error_rate(0.00001, capacity=3000)
        for i in range(1000):
            cko.add(str(i))
        bytes_out = bytes(cko)

        cko2 = CountingCuckooFilter.frombytes(bytes_out, error_rate=0.00001)

        self.assertEqual(bytes_out, bytes(cko2))
        for i in range(1000):
            self.assertTrue(cko2.check(str(i)))
        self.assertFalse(cko2.check("9999"))
        self.assertEqual(cko2.capacity, 3000)

    def test_c_cuckoo_filter_er_remove(self):
        """test removing from the counting cuckoo filter"""
        cko = CountingCuckooFilter.init_error_rate(0.00001)
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 1)
        cko.add("this is another test")
        self.assertEqual(cko.elements_added, 2)
        cko.add("this is yet another test")
        self.assertEqual(cko.elements_added, 3)
        self.assertEqual(cko.unique_elements, 3)
        cko.add("this is a test")
        cko.add("this is a test")
        cko.add("this is a test")
        self.assertEqual(cko.elements_added, 6)
        self.assertEqual(cko.unique_elements, 3)

        res = cko.remove("this is another test")
        self.assertTrue(res)
        self.assertEqual(cko.elements_added, 5)
        self.assertEqual(cko.unique_elements, 2)

        self.assertTrue(cko.check("this is a test"))
        self.assertFalse(cko.check("this is another test"))
        self.assertTrue(cko.check("this is yet another test"))


class TestCountingCuckooFilterWriteRegression(unittest.TestCase):
    """replayable write regression for the counting event dedup table

    Every test replays a fixed event stream (each event arriving a fixed
    number of times) into a tiny filter so that a new event is kicked
    several times, wraps back to an already visited bucket, and still does
    not fit; the three paths (expansion forbidden, expansion succeeds,
    expansion interrupted) must each leave exactly the last committed state,
    including the per-fingerprint arrival counts"""

    SEED = 0
    CAPACITY = 4
    BUCKET_SIZE = 2
    MAX_SWAPS = 5  # more swaps than buckets guarantees the kick path wraps around
    FINGER_SIZE = 1

    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())

    def _new_filter(self, auto_expand):
        return CountingCuckooFilter(
            capacity=self.CAPACITY,
            bucket_size=self.BUCKET_SIZE,
            max_swaps=self.MAX_SWAPS,
            auto_expand=auto_expand,
            finger_size=self.FINGER_SIZE,
        )

    @staticmethod
    def _arrivals(idx):
        """deterministic arrival count for the idx-th event"""
        return (idx + 1) % 3 + 1

    def _fill_until_full(self):
        """replay the fixed event stream until an insert fails

        Returns the filter, the confirmed-event ledger (fingerprint ->
        committed arrival count, where truly colliding keys merge into one
        observation unit), the committed keys in arrival order, and the
        event that did not fit"""
        random.seed(self.SEED)
        cko = self._new_filter(auto_expand=False)
        ledger = {}
        keys = []
        idx = 0
        while True:
            key = f"event-{idx:04d}"
            arrivals = self._arrivals(idx)
            idx += 1
            _, _, fingerprint = cko._generate_fingerprint_info(key)
            try:
                for _ in range(arrivals):
                    cko.add(key)
            except CuckooFilterFullError:
                return cko, ledger, keys, key
            ledger[fingerprint] = ledger.get(fingerprint, 0) + arrivals
            keys.append(key)

    def _replay_committed(self, keys, auto_expand):
        """replay the confirmed event stream into a fresh filter"""
        random.seed(self.SEED)
        cko = self._new_filter(auto_expand=auto_expand)
        for idx, key in enumerate(keys):
            for _ in range(self._arrivals(idx)):
                cko.add(key)
        return cko

    def _assert_ledger_committed(self, cko, ledger):
        """every committed observation unit is present once, with its count"""
        self.assertEqual(len(ledger), cko.unique_elements)
        self.assertEqual(sum(ledger.values()), cko.elements_added)
        anomalies = reconcile_counting_cuckoo(ledger, cko.buckets)
        self.assertEqual({"missing": [], "foreign": [], "duplicate": [], "count_drift": []}, anomalies)

    def test_failed_insert_multi_kick_wraparound_rolls_back(self):
        """expansion forbidden: every kicked bin is rolled back, counts intact"""
        cko, ledger, _, key = self._fill_until_full()
        _, _, fingerprint = cko._generate_fingerprint_info(key)
        self.assertNotIn(fingerprint, ledger)  # the new event is a new observation unit
        snapshot = [[(b.finger, b.count) for b in bucket] for bucket in cko.buckets]
        trace = trace_cuckoo_writes(cko)

        self.assertRaises(CuckooFilterFullError, cko.add, key)

        kicks = trace[: self.MAX_SWAPS]
        self.assertEqual(self.MAX_SWAPS, len(kicks))  # multiple kicks happened
        visited = [idx for idx, _, _, _ in kicks]
        self.assertLess(len(set(visited)), len(visited))  # the path wrapped back to a bucket
        self.assertEqual([], find_unpaired_kicks(trace, self.MAX_SWAPS))  # no kick rollback lost
        after = [[(b.finger, b.count) for b in bucket] for bucket in cko.buckets]
        self.assertEqual(snapshot, after)
        self._assert_ledger_committed(cko, ledger)

    def test_failed_insert_next_event_sees_committed_state(self):
        """the event after a failure observes the last committed counts"""
        cko, ledger, keys, key = self._fill_until_full()
        self.assertRaises(CuckooFilterFullError, cko.add, key)

        _, _, fingerprint = cko._generate_fingerprint_info(keys[0])
        committed = ledger[fingerprint]
        cko.add(keys[0])  # a repeat arrival increments exactly once
        ledger[fingerprint] = committed + 1
        self.assertEqual(committed + 1, cko.check(keys[0]))
        self._assert_ledger_committed(cko, ledger)

    def test_expand_success_commits_whole_ledger(self):
        """expansion succeeds: all committed fingerprints migrate exactly once
        and every arrival count is preserved"""
        _, ledger, keys, key = self._fill_until_full()
        cko = self._replay_committed(keys, auto_expand=True)
        trace = trace_cuckoo_writes(cko)

        cko.add(key)
        self.assertEqual(self.CAPACITY * 2, cko.capacity)

        kicks = trace[: self.MAX_SWAPS]
        self.assertEqual(self.MAX_SWAPS, len(kicks))
        visited = [idx for idx, _, _, _ in kicks]
        self.assertLess(len(set(visited)), len(visited))  # wrapped around before expanding
        self.assertEqual([], find_unpaired_kicks(trace, self.MAX_SWAPS))

        _, _, fingerprint = cko._generate_fingerprint_info(key)
        ledger[fingerprint] = 1
        self._assert_ledger_committed(cko, ledger)

    def test_interrupted_expand_restores_committed_state(self):
        """expansion interrupted mid-migration: membership and counts roll
        back to the last committed state and the next event still commits"""
        _, ledger, keys, key = self._fill_until_full()
        cko = self._replay_committed(keys, auto_expand=True)
        snapshot = [[(b.finger, b.count) for b in bucket] for bucket in cko.buckets]
        capacity = self.CAPACITY

        original = CountingCuckooFilter._insert_fingerprint_alt
        migrations = []

        def flaky_insert(self, fingerprint, idx_1, idx_2, count=1):
            if self.capacity > capacity:
                migrations.append(fingerprint)
                if len(migrations) == 3:  # interrupt mid-migration
                    raise RuntimeError("simulated expansion interruption")
            return original(self, fingerprint, idx_1, idx_2, count)

        CountingCuckooFilter._insert_fingerprint_alt = flaky_insert
        try:
            with self.assertRaises(RuntimeError) as ctx:
                cko.add(key)
        finally:
            CountingCuckooFilter._insert_fingerprint_alt = original
        self.assertEqual("simulated expansion interruption", str(ctx.exception))
        self.assertEqual(3, len(migrations))  # the expansion really started

        self.assertEqual(capacity, cko.capacity)
        after = [[(b.finger, b.count) for b in bucket] for bucket in cko.buckets]
        self.assertEqual(snapshot, after)
        self._assert_ledger_committed(cko, ledger)

        cko.add(key)  # the next event sees the committed state, then commits
        _, _, fingerprint = cko._generate_fingerprint_info(key)
        ledger[fingerprint] = 1
        self.assertEqual(capacity * 2, cko.capacity)
        self._assert_ledger_committed(cko, ledger)

    def test_true_collision_merges_observation_unit(self):
        """keys that truly collide on a fingerprint share one arrival count"""
        cko = CountingCuckooFilter(finger_size=1)
        seen = {}
        pair = None
        fingerprint = None
        for i in range(300):
            candidate = f"collision-{i:04d}"
            _, _, fingerprint = cko._generate_fingerprint_info(candidate)
            if fingerprint in seen:
                pair = (seen[fingerprint], candidate)
                break
            seen[fingerprint] = candidate
        self.assertIsNotNone(pair)

        cko.add(pair[0])
        cko.add(pair[0])
        cko.add(pair[1])  # same fingerprint: merged into the same unit
        self.assertEqual(1, cko.unique_elements)
        self.assertEqual(3, cko.elements_added)
        self.assertEqual(3, cko.check(pair[0]))
        self.assertEqual(3, cko.check(pair[1]))
        ledger = {fingerprint: 3}
        anomalies = reconcile_counting_cuckoo(ledger, cko.buckets)
        self.assertEqual({"missing": [], "foreign": [], "duplicate": [], "count_drift": []}, anomalies)

    def test_reconcile_distinguishes_failure_modes(self):
        """the ledger reconciliation separates a lost kick rollback, a
        duplicate migration, and an unchanged membership with an extra count"""
        # a lost kick rollback: a bin ends up holding a fingerprint that was
        # never committed while the committed occupant is gone
        cko, ledger, _, _ = self._fill_until_full()
        occupant = cko.buckets[0][0].finger
        novel = max(b.finger for bucket in cko.buckets for b in bucket) + 1
        cko.buckets[0][0] = CountingCuckooBin(novel, 1)
        anomalies = reconcile_counting_cuckoo(ledger, cko.buckets)
        self.assertEqual([occupant], anomalies["missing"])
        self.assertEqual([novel], anomalies["foreign"])
        self.assertEqual([], anomalies["duplicate"])
        self.assertEqual([], anomalies["count_drift"])

        # a duplicate migration: a committed fingerprint lands in the table twice
        cko = self._new_filter(auto_expand=False)
        ledger = {}
        for i in range(3):
            key = f"dup-{i:04d}"
            _, _, fingerprint = cko._generate_fingerprint_info(key)
            cko.add(key)
            ledger[fingerprint] = ledger.get(fingerprint, 0) + 1
        target = next(bucket for bucket in cko.buckets if len(bucket) < cko.bucket_size)
        target.append(CountingCuckooBin(next(iter(ledger)), 1))
        anomalies = reconcile_counting_cuckoo(ledger, cko.buckets)
        self.assertEqual([], anomalies["missing"])
        self.assertEqual([], anomalies["foreign"])
        self.assertEqual(1, len(anomalies["duplicate"]))
        self.assertEqual([], anomalies["count_drift"])

        # membership unchanged but a count was incremented once too many
        cko, ledger, _, _ = self._fill_until_full()
        drifted = next(iter(ledger))
        for bucket in cko.buckets:
            for ccb in bucket:
                if ccb.finger == drifted:
                    ccb.increment()
        anomalies = reconcile_counting_cuckoo(ledger, cko.buckets)
        self.assertEqual([], anomalies["missing"])
        self.assertEqual([], anomalies["foreign"])
        self.assertEqual([], anomalies["duplicate"])
        self.assertEqual([drifted], anomalies["count_drift"])


if __name__ == "__main__":
    unittest.main()
