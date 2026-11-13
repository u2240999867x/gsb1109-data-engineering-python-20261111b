#!/usr/bin/env python3
"""Unittest class"""

import hashlib
import itertools
import os
import sys
import unittest
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import ExpandingBloomFilter, RotatingBloomFilter  # noqa: E402
from probables.exceptions import InitializationError, RotatingBloomFilterError  # noqa: E402
from tests.utilities import calc_file_md5, different_hash  # noqa: E402

DELETE_TEMP_FILES = True


class TestExpandingBloomFilter(unittest.TestCase):
    """Test ExpandingBloomFilter"""

    def test_ebf_init(self):
        """test the initialization of an expanding bloom filter"""
        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.expansions, 0)
        self.assertEqual(blm.false_positive_rate, 0.05)
        self.assertEqual(blm.estimated_elements, 10)
        self.assertEqual(blm.elements_added, 0)

    def test_ebf_add_lots(self):
        """test adding "lots" of elements to force the expansion"""
        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.05)
        for i in range(100):
            blm.add(f"{i}", True)
        self.assertEqual(blm.expansions, 9)

    def test_ebf_add_lots_diff_hash(self):
        """test adding "lots" of elements to force the expansion using a different hash"""
        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.05, hash_function=different_hash)
        for i in range(100):
            blm.add(f"{i}", True)
        self.assertEqual(blm.expansions, 9)

    def test_ebf_add_lots_without_force(self):
        """testing adding "lots" but force them to be inserted multiple times"""
        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.05)
        # simulate false positives... notice it didn't grow a few...
        for i in range(120):
            blm.add(f"{i}")
        self.assertEqual(blm.expansions, 8)
        self.assertEqual(blm.elements_added, 120)

    def test_ebf_check(self):
        """ensure that checking the expanding bloom filter works"""
        blm = ExpandingBloomFilter(est_elements=30, false_positive_rate=0.05)
        # expand it out some first!
        for i in range(100):
            blm.add(f"{i}")
        blm.add("this is a test")
        blm.add("this is another test")
        self.assertGreater(blm.expansions, 1)
        self.assertEqual(blm.check("this is a test"), True)
        self.assertEqual(blm.check("this is another test"), True)
        self.assertEqual(blm.check("this is yet another test!"), False)
        self.assertEqual(blm.check("this is not another test"), False)
        self.assertEqual(blm.elements_added, 102)

    def test_ebf_contains(self):
        """ensure that "in" functionality for the expanding bloom filter works"""
        blm = ExpandingBloomFilter(est_elements=30, false_positive_rate=0.05)
        # expand it out some first!
        for i in range(100):
            blm.add(f"{i}")
        blm.add("this is a test")
        blm.add("this is another test")
        self.assertGreater(blm.expansions, 1)
        self.assertEqual("this is a test" in blm, True)
        self.assertEqual("this is another test" in blm, True)
        self.assertEqual("this is yet another test!" in blm, False)
        self.assertEqual("this is not another test" in blm, False)

    def test_ebf_push(self):
        """ensure that we are able to push new Bloom Filters"""
        blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
        self.assertEqual(blm.expansions, 0)
        blm.push()
        self.assertEqual(blm.expansions, 1)
        self.assertEqual(blm.elements_added, 0)
        blm.push()
        self.assertEqual(blm.expansions, 2)
        self.assertEqual(blm.elements_added, 0)
        blm.push()
        self.assertEqual(blm.expansions, 3)
        self.assertEqual(blm.elements_added, 0)

    def test_ebf_export(self):
        """basic expanding Bloom Filter export test"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".ebf", delete=DELETE_TEMP_FILES) as fobj:
            blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
            blm.export(fobj.name)
            self.assertEqual(calc_file_md5(fobj.name), "eb5769ae9babdf7b37d6ce64d58812bc")

    def test_ebf_bytes(self):
        """basic expanding Bloom Filter export bytes test"""
        blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
        self.assertEqual(hashlib.md5(bytes(blm)).hexdigest(), "eb5769ae9babdf7b37d6ce64d58812bc")

    def test_ebf_frombytes(self):
        """expanding Bloom Filter load bytes test"""
        blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
        for i in range(105):
            blm.add(str(i))
        bytes_out = bytes(blm)

        blm2 = ExpandingBloomFilter.frombytes(bytes_out)
        self.assertEqual(blm2.expansions, 3)
        self.assertEqual(blm2.false_positive_rate, 0.05000000074505806)
        self.assertEqual(blm2.estimated_elements, 25)
        self.assertEqual(blm2.elements_added, 105)
        self.assertEqual(bytes(blm2), bytes(blm))

        for i in range(105):
            self.assertTrue(blm.check(str(i)))

    def test_ebf_import_empty(self):
        """test that expanding Bloom Filter is correct on import"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".ebf", delete=DELETE_TEMP_FILES) as fobj:
            blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
            blm.export(fobj.name)
            self.assertEqual(calc_file_md5(fobj.name), "eb5769ae9babdf7b37d6ce64d58812bc")

            blm2 = ExpandingBloomFilter(filepath=fobj.name)
            for bloom in blm2._blooms:
                self.assertEqual(bloom.elements_added, 0)

    def test_ebf_import_non_empty(self):
        """test expanding Bloom Filter import when non-empty"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".ebf", delete=DELETE_TEMP_FILES) as fobj:
            blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
            for i in range(15):
                blm.add(f"{i}")
                blm.push()

            blm.export(fobj.name)

            blm2 = ExpandingBloomFilter(filepath=fobj.name)
            self.assertEqual(blm2.expansions, 15)
            for i in range(15):
                self.assertEqual(f"{i}" in blm2, True)

            # check for things that are not there!
            for i in range(99, 125):
                self.assertEqual(f"{i}" in blm2, False)


class TestExpandingBloomFilterBudget(unittest.TestCase):
    """Test explicit overall false positive budget mode"""

    def test_default_mode_is_per_layer(self):
        """without opting in, false_positive_rate remains the per layer rate"""
        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertFalse(blm.budget_mode)
        self.assertIsNone(blm.overall_false_positive_rate)
        self.assertIsNone(blm.budget_strategy)
        self.assertEqual(len(blm.layer_false_positive_rates), 1)
        self.assertAlmostEqual(blm.layer_false_positive_rates[0], 0.05)
        self.assertAlmostEqual(blm.false_positive_rate_bound, 0.05)

    def test_overall_budget_initial_layer(self):
        """the first layer receives half of the overall budget"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        self.assertTrue(blm.budget_mode)
        self.assertEqual(blm.budget_strategy, "geometric")
        self.assertEqual(blm.overall_false_positive_rate, 0.05)
        self.assertEqual(blm.false_positive_rate, 0.05)
        self.assertEqual(len(blm.layer_false_positive_rates), 1)
        self.assertAlmostEqual(blm.layer_false_positive_rates[0], 0.025)
        self.assertLessEqual(blm.false_positive_rate_bound, 0.05)
        self.assertEqual(blm.next_layer_index, 1)
        self.assertAlmostEqual(blm.next_layer_false_positive_rate, 0.0125)

    def test_overall_budget_layers_halve(self):
        """each layer receives half of the previous layer's budget"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(100):
            blm.add(f"{i}", True)
        self.assertEqual(blm.expansions, 9)
        rates = blm.layer_false_positive_rates
        self.assertEqual(len(rates), 10)
        for idx, rate in enumerate(rates):
            self.assertAlmostEqual(rate, 0.05 / (2 ** (idx + 1)), places=6)
            self.assertLessEqual(rate, 0.05 / (2 ** (idx + 1)))
        self.assertLessEqual(blm.false_positive_rate_bound, 0.05)
        self.assertLess(blm.false_positive_rate_bound, 0.05)
        self.assertEqual(blm.next_layer_index, 10)

    def test_overall_budget_growth_schedule(self):
        """a new layer opens exactly when the newest layer reaches est_elements"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(9):
            blm.add(f"{i}", True)
        self.assertEqual(blm.expansions, 0)
        self.assertEqual(blm.elements_added, 9)
        blm.add("9", True)
        self.assertEqual(blm.expansions, 0)
        blm.add("10", True)
        self.assertEqual(blm.expansions, 1)
        self.assertEqual(blm.layer_metrics()[0]["elements_added"], 10)
        self.assertEqual(blm.layer_metrics()[1]["elements_added"], 1)

    def test_budget_mode_alias_keywords(self):
        """explicit budget can be enabled with the documented aliases"""
        for kwargs in (
            {"overall_false_positive_rate": 0.02},
            {"overall_fpr": 0.02},
            {"overall_budget": 0.02},
            {"budget": 0.02},
        ):
            blm = ExpandingBloomFilter(est_elements=10, **kwargs)
            self.assertTrue(blm.budget_mode)
            self.assertEqual(blm.overall_false_positive_rate, 0.02)

        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.02, budget_mode=True)
        self.assertTrue(blm.budget_mode)
        self.assertEqual(blm.overall_false_positive_rate, 0.02)

        blm = ExpandingBloomFilter(est_elements=10, false_positive_rate=0.02, mode="per-layer")
        self.assertFalse(blm.budget_mode)

    def test_budget_mode_requires_budget(self):
        """enabling budget mode without a numeric budget raises"""
        with self.assertRaises(InitializationError):
            ExpandingBloomFilter(est_elements=10, budget_mode=True)

    def test_conflicting_budgets_raise(self):
        """two different explicit overall budgets are rejected"""
        with self.assertRaises(InitializationError):
            ExpandingBloomFilter(est_elements=10, overall_fpr=0.02, budget=0.03)

    def test_invalid_overall_budget(self):
        """overall budget must be strictly between zero and one"""
        for value in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(InitializationError):
                ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=value)

    def test_layer_parameters_grow(self):
        """tighter layer budgets imply more hashes and more bits"""
        blm = ExpandingBloomFilter(est_elements=25, overall_false_positive_rate=0.05)
        for i in range(100):
            blm.add(f"{i}", True)
        metrics = blm.layer_metrics()
        for prev, cur in itertools.pairwise(metrics):
            self.assertGreater(cur["number_hashes"], prev["number_hashes"])
            self.assertGreater(cur["number_bits"], prev["number_bits"])

    def test_check_covers_all_layers_with_growing_hashes(self):
        """every layer is queried and no layer is queried with a short hash list"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        added = [f"item-{i}" for i in range(50)]
        for item in added:
            blm.add(item, True)
        self.assertGreater(blm.expansions, 1)
        for item in added:
            self.assertTrue(blm.check(item))
            self.assertIn(item, blm)

        hashes = blm.hashes("item-3")
        self.assertGreaterEqual(len(hashes), max(metric["number_hashes"] for metric in blm.layer_metrics()))
        for blm_layer in blm._blooms:
            self.assertGreaterEqual(len(hashes), blm_layer.number_hashes)

    def test_short_hash_list_is_rejected(self):
        """a hash list that is too short for a layer raises instead of reusing values"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(20):
            blm.add(f"{i}", True)
        short = blm.hashes("value")[: blm._blooms[-1].number_hashes - 1]
        with self.assertRaises(ValueError):
            blm.check_alt(short)
        with self.assertRaises(ValueError):
            blm.add_alt(short, True)

    def test_add_alt_with_full_depth_hashes(self):
        """add_alt accepts hash lists generated at the maximum required depth"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(20):
            hashes = blm.hashes(f"{i}")
            blm.add_alt(hashes, True)
        for i in range(20):
            self.assertTrue(blm.check(f"{i}"))

    def test_budget_round_trip_bytes(self):
        """bytes preserve the budget mode, every layer rate, and the schedule"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(35):
            blm.add(f"{i}", True)
        data = bytes(blm)
        loaded = ExpandingBloomFilter.frombytes(data)
        self.assertTrue(loaded.budget_mode)
        self.assertEqual(loaded.overall_false_positive_rate, 0.05)
        self.assertEqual(loaded.budget_strategy, "geometric")
        self.assertEqual(loaded.layer_false_positive_rates, blm.layer_false_positive_rates)
        self.assertEqual(loaded.next_layer_index, blm.next_layer_index)
        self.assertEqual(loaded.elements_added, blm.elements_added)
        self.assertEqual(bytes(loaded), data)
        for i in range(35):
            self.assertTrue(loaded.check(f"{i}"))

    def test_budget_round_trip_file_ignores_constructor_budget(self):
        """a loaded file's budget metadata wins over constructor arguments"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".ebf", delete=DELETE_TEMP_FILES) as fobj:
            blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
            for i in range(30):
                blm.add(f"{i}", True)
            blm.export(fobj.name)

            loaded = ExpandingBloomFilter(est_elements=99, overall_false_positive_rate=0.9, filepath=fobj.name)
            self.assertEqual(loaded.estimated_elements, 10)
            self.assertEqual(loaded.overall_false_positive_rate, 0.05)
            self.assertEqual(loaded.layer_false_positive_rates, blm.layer_false_positive_rates)

    def test_legacy_file_still_loads(self):
        """files written in per layer mode keep the legacy footer layout"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".ebf", delete=DELETE_TEMP_FILES) as fobj:
            blm = ExpandingBloomFilter(est_elements=25, false_positive_rate=0.05)
            for i in range(30):
                blm.add(f"{i}", True)
            blm.export(fobj.name)
            loaded = ExpandingBloomFilter(filepath=fobj.name)
            self.assertFalse(loaded.budget_mode)
            self.assertIsNone(loaded.overall_false_positive_rate)
            self.assertEqual(loaded.layer_false_positive_rates, (loaded.false_positive_rate,) * len(loaded._blooms))

    def test_budget_bound_never_exceeds_commitment(self):
        """the union bound stays at or below the overall budget as layers grow"""
        for count in (1, 2, 5, 10, 25, 60):
            blm = ExpandingBloomFilter(est_elements=7, overall_false_positive_rate=0.03)
            for i in range(count * 7):
                blm.add(f"{i}", True)
            self.assertLessEqual(blm.false_positive_rate_bound, 0.03)
            self.assertEqual(blm.false_positive_rate_bound, sum(blm.layer_false_positive_rates))

    def test_empirical_false_positive_rate_is_only_sanity_check(self):
        """a single sample frequency is checked loosely; the math bound is authoritative"""
        budget = 0.05
        blm = ExpandingBloomFilter(est_elements=200, overall_false_positive_rate=budget)
        for i in range(2000):
            blm.add(f"present-{i}", True)
        self.assertGreater(blm.expansions, 1)

        probes = 5000
        false_positives = sum(1 for i in range(probes) if blm.check(f"absent-{i}"))
        observed = false_positives / probes
        self.assertLessEqual(blm.false_positive_rate_bound, budget)
        self.assertLess(observed, budget * 2)

    def test_rotating_budget_mode(self):
        """rotating filters allocate within the budget and preserve it on reload"""
        blm = RotatingBloomFilter(est_elements=5, overall_false_positive_rate=0.04, max_queue_size=3)
        self.assertTrue(blm.budget_mode)
        worst_bound = 0.0
        for i in range(60):
            blm.add(f"{i}", True)
            worst_bound = max(worst_bound, blm.false_positive_rate_bound)
        self.assertEqual(blm.current_queue_size, 3)
        self.assertLessEqual(worst_bound, 0.04)
        self.assertLessEqual(blm.false_positive_rate_bound, 0.04)

        loaded = RotatingBloomFilter.frombytes(bytes(blm), max_queue_size=3)
        self.assertTrue(loaded.budget_mode)
        self.assertEqual(loaded.overall_false_positive_rate, 0.04)
        self.assertEqual(loaded.layer_false_positive_rates, blm.layer_false_positive_rates)
        self.assertEqual(bytes(loaded), bytes(blm))

    def test_rotating_queue_size_budget_bound(self):
        """the active rotating queue stays inside the budget for several queue sizes"""
        for queue_size in (2, 3, 5, 10):
            blm = RotatingBloomFilter(est_elements=5, overall_false_positive_rate=0.04, max_queue_size=queue_size)
            worst_bound = 0.0
            for i in range(300):
                blm.add(f"{i}", True)
                worst_bound = max(worst_bound, blm.false_positive_rate_bound)
            self.assertLessEqual(worst_bound, 0.04)
            self.assertEqual(blm.current_queue_size, queue_size)

    def test_budget_exhaustion_raises(self):
        """when the geometric schedule underflows, growth raises instead of mis-allocating"""
        blm = ExpandingBloomFilter(est_elements=1, overall_false_positive_rate=0.5)
        with self.assertRaises(InitializationError):
            for i in range(200):
                blm.add(f"{i}", True)

    def test_invalid_file_budget_metadata(self):
        """corrupt budget metadata is rejected instead of silently trusted"""
        blm = ExpandingBloomFilter(est_elements=10, overall_false_positive_rate=0.05)
        for i in range(15):
            blm.add(f"{i}", True)
        data = bytearray(bytes(blm))
        footer = Struct("<QQQQdBB4s")
        size, est_els, els_added, next_idx, overall, mode, strategy, magic = footer.unpack(bytes(data[-footer.size :]))
        data[-footer.size + 32 : -footer.size + 40] = Struct("<d").pack(1.5)
        with self.assertRaises(InitializationError):
            ExpandingBloomFilter.frombytes(bytes(data))

    def test_rotating_rotation_opens_new_layer(self):
        """rotation uses the next layer budget and queries all queued layers"""
        blm = RotatingBloomFilter(est_elements=5, overall_false_positive_rate=0.04, max_queue_size=3)
        for i in range(10):
            blm.add(f"{i}", True)
        self.assertEqual(blm.current_queue_size, 2)
        self.assertNotEqual(blm.layer_false_positive_rates[0], blm.layer_false_positive_rates[1])
        self.assertLessEqual(blm.false_positive_rate_bound, 0.04)
        for i in range(10):
            self.assertTrue(blm.check(f"{i}"))


class TestRotatingBloomFilter(unittest.TestCase):
    """Test RotatingBloomFilter"""

    def test_rbf_init(self):
        """test the initialization of an rotating bloom filter"""
        blm = RotatingBloomFilter(est_elements=10, false_positive_rate=0.05, max_queue_size=10)
        self.assertEqual(blm.expansions, 0)
        self.assertEqual(blm.max_queue_size, 10)

    def test_rbf_rotate(self):
        """test that the bloom filter rotates the first bloom off the stack"""
        blm = RotatingBloomFilter(est_elements=10, false_positive_rate=0.05, max_queue_size=5)
        self.assertEqual(blm.expansions, 0)
        blm.add("test")
        self.assertEqual(blm.expansions, 0)
        for i in range(10):
            blm.add(f"{i}", force=True)
        self.assertEqual(blm.expansions, 1)
        self.assertEqual(blm.current_queue_size, 2)
        self.assertEqual(blm.check("test"), True)

        for i in range(10, 20):
            blm.add(f"{i}", force=True)
        self.assertEqual(blm.check("test"), True)
        self.assertEqual(blm.current_queue_size, 3)

        for i in range(20, 30):
            blm.add(f"{i}", force=True)
        self.assertEqual(blm.check("test"), True)
        self.assertEqual(blm.current_queue_size, 4)

        for i in range(30, 40):
            blm.add(f"{i}", force=True)
        self.assertEqual(blm.check("test"), True)
        self.assertEqual(blm.current_queue_size, 5)

        for i in range(40, 50):
            blm.add(f"{i}", force=True)
        self.assertEqual(blm.check("test"), False)  # it should roll off
        self.assertEqual(blm.current_queue_size, 5)

        self.assertEqual(blm.elements_added, 51)

    def test_rbf_push_pop(self):
        """test forcing push and pop"""
        blm = RotatingBloomFilter(est_elements=10, false_positive_rate=0.05, max_queue_size=5)
        self.assertEqual(blm.current_queue_size, 1)
        blm.add("test")
        blm.push()
        self.assertEqual(blm.current_queue_size, 2)
        self.assertEqual("test" in blm, True)
        blm.push()
        self.assertEqual(blm.current_queue_size, 3)
        self.assertEqual("test" in blm, True)
        blm.push()
        self.assertEqual(blm.current_queue_size, 4)
        self.assertEqual("test" in blm, True)
        blm.push()
        self.assertEqual(blm.current_queue_size, 5)
        self.assertEqual("test" in blm, True)
        blm.push()
        self.assertEqual(blm.current_queue_size, 5)
        self.assertEqual("test" in blm, False)

        # test popping
        blm.add("that")
        blm.pop()
        self.assertEqual(blm.current_queue_size, 4)
        self.assertEqual("that" in blm, True)
        blm.pop()
        self.assertEqual(blm.current_queue_size, 3)
        self.assertEqual("that" in blm, True)
        blm.pop()
        self.assertEqual(blm.current_queue_size, 2)
        self.assertEqual("that" in blm, True)
        blm.pop()
        self.assertEqual(blm.current_queue_size, 1)
        self.assertEqual("that" in blm, True)

    def test_rbf_pop_exception(self):
        """ensure the correct exception is thrown"""
        blm = RotatingBloomFilter(est_elements=10, false_positive_rate=0.05, max_queue_size=5)
        self.assertRaises(RotatingBloomFilterError, lambda: blm.pop())

    def test_rbf_pop_exception_msg(self):
        """rotating bloom filter error: check the resulting error message"""
        blm = RotatingBloomFilter(est_elements=10, false_positive_rate=0.05, max_queue_size=5)
        try:
            blm.pop()
        except RotatingBloomFilterError as ex:
            msg = "Popping a Bloom Filter will result in an unusable system!"
            self.assertEqual(str(ex), msg)
        except:  # noqa: E722
            self.assertEqual(True, False)

    def test_rfb_basic_export(self):
        """basic rotating Bloom Filter export test"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".rbf", delete=DELETE_TEMP_FILES) as fobj:
            blm = RotatingBloomFilter(est_elements=25, false_positive_rate=0.05)
            blm.export(fobj.name)
            self.assertEqual(calc_file_md5(fobj.name), "eb5769ae9babdf7b37d6ce64d58812bc")

    def test_rfb_basic_bytes(self):
        """basic rotating Bloom Filter export bytes test"""
        blm = RotatingBloomFilter(est_elements=25, false_positive_rate=0.05)
        self.assertEqual(hashlib.md5(bytes(blm)).hexdigest(), "eb5769ae9babdf7b37d6ce64d58812bc")

    def test_rfb_from_bytes(self):
        """basic rotating Bloom Filter export bytes test"""
        blm = RotatingBloomFilter(est_elements=25, false_positive_rate=0.05, max_queue_size=3)
        for i in range(105):
            blm.add(str(i))
        bytes_out = bytes(blm)

        blm2 = RotatingBloomFilter.frombytes(bytes_out, max_queue_size=3)
        self.assertEqual(blm2.expansions, 2)
        self.assertEqual(blm2.false_positive_rate, 0.05000000074505806)
        self.assertEqual(blm2.estimated_elements, 25)
        self.assertEqual(blm2.elements_added, 105)
        self.assertEqual(blm2.current_queue_size, 3)
        self.assertEqual(bytes(blm2), bytes(blm))
        for i in range(105):
            self.assertEqual(blm.check(str(i)), blm2.check(str(i)))

    def test_rbf_import_empty(self):
        """test that rotating Bloom Filter is correct on import"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".rbf", delete=DELETE_TEMP_FILES) as fobj:
            blm = RotatingBloomFilter(est_elements=25, false_positive_rate=0.05)
            blm.export(fobj.name)
            self.assertEqual(calc_file_md5(fobj.name), "eb5769ae9babdf7b37d6ce64d58812bc")

            blm2 = ExpandingBloomFilter(filepath=fobj.name)
            for bloom in blm2._blooms:
                self.assertEqual(bloom.elements_added, 0)

    def test_rbf_non_basic_import(self):
        """test that the imported rotating Bloom filter is correct"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".rbf", delete=DELETE_TEMP_FILES) as fobj:
            blm = RotatingBloomFilter(est_elements=25, false_positive_rate=0.05)
            for i in range(15):
                blm.add(f"{i}")
                blm.push()
            blm.export(fobj.name)

            blm2 = RotatingBloomFilter(filepath=fobj.name)
            # test those that should be popped off...
            for i in range(5):
                self.assertEqual(f"{i}" in blm2, False)
            # test things that would not be popped
            for i in range(6, 15):
                self.assertEqual(f"{i}" in blm2, True)
            self.assertEqual(blm2.current_queue_size, 10)
            self.assertEqual(blm2.expansions, 9)
            self.assertEqual(blm2.elements_added, 15)


if __name__ == "__main__":
    unittest.main()
