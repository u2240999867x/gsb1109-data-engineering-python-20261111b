#!/usr/bin/env python3
"""Unittest class for partition-aware merging"""

import sys
import unittest
from pathlib import Path

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import (  # noqa: E402
    BloomFilter,
    CountingBloomFilter,
    CountMinSketch,
    CoverageInterval,
    HeavyHitters,
    MergeGap,
    PartitionRecord,
    merge_partitions,
    merge_results,
    register_legacy_hash_convention,
)
from probables.merging.intervals import normalize_intervals, subtract_intervals  # noqa: E402
from tests.utilities import different_hash  # noqa: E402


def _cms(width=100, depth=3, items=None):
    sketch = CountMinSketch(width=width, depth=depth)
    for key, count in items or []:
        sketch.add(key, count)
    return sketch


def _bloom(items=None):
    sketch = BloomFilter(est_elements=100, false_positive_rate=0.05)
    for item in items or []:
        sketch.add(item)
    return sketch


def _counting_bloom(items=None):
    sketch = CountingBloomFilter(est_elements=100, false_positive_rate=0.05)
    for item in items or []:
        sketch.add(item)
    return sketch


class TestCoverageIntervals(unittest.TestCase):
    """Test coverage interval helpers"""

    def test_validation(self):
        """intervals must be ordered integer ranges"""
        self.assertRaises(ValueError, lambda: CoverageInterval(5, 4))
        self.assertRaises(TypeError, lambda: CoverageInterval(0, 1.5))  # type: ignore

    def test_normalize_merges_touching(self):
        """disjoint, adjacent and overlapping ranges normalize correctly"""
        intervals = [
            CoverageInterval(10, 19),
            CoverageInterval(0, 4),
            CoverageInterval(5, 9),
            CoverageInterval(7, 12),
        ]
        self.assertEqual(normalize_intervals(intervals), [CoverageInterval(0, 19)])

    def test_normalize_keeps_gap(self):
        """a missing position stays a gap between normalized ranges"""
        self.assertEqual(
            normalize_intervals([CoverageInterval(0, 4), CoverageInterval(6, 9)]),
            [CoverageInterval(0, 4), CoverageInterval(6, 9)],
        )

    def test_subtract(self):
        """subtraction returns only the uncovered residuals"""
        self.assertEqual(
            subtract_intervals(CoverageInterval(0, 19), [CoverageInterval(5, 9)]),
            [CoverageInterval(0, 4), CoverageInterval(10, 19)],
        )
        self.assertEqual(subtract_intervals(CoverageInterval(0, 9), [CoverageInterval(0, 9)]), [])


class TestPartitionRecord(unittest.TestCase):
    """Test partition record validation and legacy convention binding"""

    def test_invalid_record(self):
        """ids, generations and intervals are validated"""
        sketch = _cms()
        self.assertRaises(
            ValueError,
            lambda: PartitionRecord("", 1, [CoverageInterval(0, 1)], sketch),
        )
        self.assertRaises(
            ValueError,
            lambda: PartitionRecord("p1", 1, [], sketch),
        )
        self.assertRaises(
            TypeError,
            lambda: PartitionRecord("p1", True, [CoverageInterval(0, 1)], sketch),  # type: ignore
        )

    def test_unknown_legacy_convention(self):
        """legacy conventions must be explicitly registered first"""
        with self.assertRaises(KeyError):
            PartitionRecord("p1", 1, [CoverageInterval(0, 1)], _cms(), legacy_hash_convention="missing")

    def test_double_binding_rejected(self):
        """a record cannot name a convention and pass a function"""
        register_legacy_hash_convention("test-legacy", different_hash)
        with self.assertRaises(ValueError):
            PartitionRecord(
                "p1",
                1,
                [CoverageInterval(0, 1)],
                _cms(),
                legacy_hash_convention="test-legacy",
                hash_function=different_hash,
            )


class TestCountMinPartitionMerge(unittest.TestCase):
    """Test partition merging with additive Count-Min sketches"""

    def test_disjoint_partitions_sum(self):
        """disjoint coverage stitches and frequencies add"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5), ("b", 2)])),
                PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _cms(items=[("a", 3), ("c", 7)])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertEqual(result.included_partitions, ("p1", "p2"))
        self.assertEqual(result.covered_intervals, (CoverageInterval(0, 19),))
        self.assertEqual(result.sketch.check("a"), 8)
        self.assertEqual(result.sketch.check("b"), 2)
        self.assertEqual(result.sketch.check("c"), 7)
        self.assertEqual(result.sketch.elements_added, 17)
        self.assertEqual(result.superseded, {"p1": 1, "p2": 1})

    def test_redelivery_is_idempotent(self):
        """re-sending the same partition generation never doubles counts"""
        payload = _cms(items=[("a", 5)])
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], payload),
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertEqual(result.duplicate_deliveries, (("p1", 1),))
        self.assertEqual(result.sketch.check("a"), 5)
        self.assertEqual(result.sketch.elements_added, 5)

    def test_partial_overlap_leaves_gap(self):
        """additive overlap without a finer source withholds the payload"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)])),
                PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _cms(items=[("c", 3)])),
            ]
        )
        self.assertFalse(result.total_known)
        gap_ranges = sorted((gap.interval.start, gap.interval.end) for gap in result.gaps)
        self.assertEqual(gap_ranges, [(5, 9), (10, 14)])
        self.assertTrue(any(gap.partitions == ("p2",) for gap in result.gaps))
        self.assertEqual(result.sketch.check("c"), 0)
        self.assertEqual(result.sketch.check("a"), 5)
        self.assertIn("overlap", result.explain_gaps())
        self.assertNotIn("p2", result.included_partitions)

    def test_new_generation_supersedes(self):
        """a newer generation of the same partition replaces the old bins"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)])),
                PartitionRecord("p1", 2, [CoverageInterval(0, 9)], _cms(items=[("a", 9)])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertEqual(result.stale_deliveries, (("p1", 1),))
        self.assertEqual(result.superseded, {"p1": 2})
        self.assertEqual(result.sketch.check("a"), 9)
        self.assertEqual(result.sketch.elements_added, 9)

    def test_stale_generation_uncovered_range_gap(self):
        """ranges only covered by a superseded generation remain unknown"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 19)], _cms(items=[("a", 5)])),
                PartitionRecord("p1", 2, [CoverageInterval(0, 9)], _cms(items=[("a", 9)])),
            ]
        )
        self.assertFalse(result.total_known)
        self.assertEqual(result.gaps[0].interval, CoverageInterval(10, 19))
        self.assertIn("superseded", result.gaps[0].reason)

    def test_dimension_mismatch_rejected(self):
        """buckets of different dimensions never share a result"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(width=100)),
                PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _cms(width=50)),
            ]
        )
        self.assertFalse(result.total_known)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("width or depth", result.rejected[0])
        self.assertEqual(result.included_partitions, ("p1",))

    def test_hash_convention_mismatch_rejected(self):
        """a different hashing convention cannot feed the same bins"""
        legacy = CountMinSketch(width=100, depth=3, hash_function=different_hash)
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms()),
                PartitionRecord("p2", 1, [CoverageInterval(10, 19)], legacy),
            ]
        )
        self.assertFalse(result.total_known)
        self.assertIn("hashing conventions", result.rejected[0])

    def test_legacy_convention_accepted_when_bound(self):
        """caller-bound matching conventions merge and are recorded"""
        register_legacy_hash_convention("cms-legacy", different_hash)
        first = CountMinSketch(width=100, depth=3, hash_function=different_hash)
        first.add("a", 4)
        second = CountMinSketch(width=100, depth=3, hash_function=different_hash)
        second.add("b", 6)
        result = merge_partitions(
            [
                PartitionRecord(
                    "p1", 1, [CoverageInterval(0, 9)], first, legacy_hash_convention="cms-legacy"
                ),
                PartitionRecord(
                    "p2", 1, [CoverageInterval(10, 19)], second, legacy_hash_convention="cms-legacy"
                ),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertEqual(result.legacy_conventions, ("cms-legacy",))
        self.assertEqual(result.sketch.check("a"), 4)
        self.assertEqual(result.sketch.check("b"), 6)

    def test_wrong_binding_rejected(self):
        """claiming a convention that the payload does not match is rejected"""
        register_legacy_hash_convention("cms-legacy-2", different_hash)
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms()),
                PartitionRecord(
                    "p2",
                    1,
                    [CoverageInterval(10, 19)],
                    _cms(),
                    legacy_hash_convention="cms-legacy-2",
                ),
            ]
        )
        self.assertFalse(result.total_known)
        self.assertIn("hashes do not match", result.rejected[0])

    def test_unsupported_payload(self):
        """HeavyHitters cannot be merged because joining is unsupported"""
        hh = HeavyHitters(num_hitters=5, width=100, depth=3)
        self.assertRaises(
            TypeError,
            lambda: merge_partitions([PartitionRecord("p1", 1, [CoverageInterval(0, 9)], hh)]),
        )

    def test_empty_input(self):
        """merging nothing is undefined"""
        self.assertRaises(ValueError, lambda: merge_partitions([]))


class TestBloomPartitionMerge(unittest.TestCase):
    """Test partition merging with set-semantics Bloom filters"""

    def test_disjoint_bloom_merge(self):
        """disjoint bloom payloads union cleanly"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _bloom(["a", "b"])),
                PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _bloom(["c"])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertTrue(all(item in result.sketch for item in ("a", "b", "c")))

    def test_overlap_is_idempotent(self):
        """set union means overlapping membership is never doubled"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _bloom(["a"])),
                PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _bloom(["a", "b"])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertEqual(result.gaps, ())
        self.assertEqual(len(result.notices), 1)
        self.assertIn("p2", result.included_partitions)
        self.assertTrue("a" in result.sketch and "b" in result.sketch)

    def test_redelivery_recorded(self):
        """duplicate bloom deliveries are flagged and skipped"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _bloom(["a"])),
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _bloom(["a"])),
            ]
        )
        self.assertEqual(result.duplicate_deliveries, (("p1", 1),))
        self.assertTrue("a" in result.sketch)


class TestCountingBloomPartitionMerge(unittest.TestCase):
    """Test partition merging with additive counting Bloom filters"""

    def test_disjoint_counters_add(self):
        """disjoint counting blooms add their counters"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _counting_bloom(["a"])),
                PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _counting_bloom(["a", "b"])),
            ]
        )
        self.assertTrue(result.total_known)
        self.assertTrue(result.sketch.check("a"))
        self.assertTrue(result.sketch.check("b"))

    def test_overlap_withheld(self):
        """counting bloom overlap would double count, so it becomes a gap"""
        result = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _counting_bloom(["a"])),
                PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _counting_bloom(["a"])),
            ]
        )
        self.assertFalse(result.total_known)
        self.assertTrue(any(isinstance(gap, MergeGap) for gap in result.gaps))
        self.assertNotIn("p2", result.included_partitions)


class TestMergeResults(unittest.TestCase):
    """Test merging finished merge results together"""

    def test_disjoint_results_combine(self):
        """finished results combine and retain the original partition ledger"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)]))]
        )
        second = merge_partitions(
            [PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _cms(items=[("a", 3)]))]
        )
        combined = merge_results(first, second)
        self.assertTrue(combined.total_known)
        self.assertEqual(combined.included_partitions, ("p1", "p2"))
        self.assertEqual(combined.covered_intervals, (CoverageInterval(0, 19),))
        self.assertEqual(combined.sketch.check("a"), 8)

    def test_result_redelivery_no_double_count(self):
        """re-merging the same result does not add the bins twice"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)]))]
        )
        combined = merge_results(first, first)
        self.assertTrue(combined.total_known)
        self.assertEqual(combined.duplicate_deliveries, (("p1", 1),))
        self.assertEqual(combined.sketch.check("a"), 5)
        self.assertEqual(combined.sketch.elements_added, 5)

    def test_result_overlap_gap(self):
        """overlapping additive results without finer sources stay unknown"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)]))]
        )
        second = merge_partitions(
            [PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _cms(items=[("b", 2)]))]
        )
        combined = merge_results(first, second)
        self.assertFalse(combined.total_known)
        ranges = sorted((gap.interval.start, gap.interval.end) for gap in combined.gaps)
        self.assertIn((5, 9), ranges)
        self.assertEqual(combined.sketch.check("b"), 0)

    def test_result_generation_conflict_gap(self):
        """a result carrying a newer overlapping generation cannot supersede bins"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)]))]
        )
        second = merge_partitions(
            [PartitionRecord("p1", 2, [CoverageInterval(0, 9)], _cms(items=[("a", 2)]))]
        )
        combined = merge_results(first, second)
        self.assertFalse(combined.total_known)
        self.assertEqual(combined.gaps[0].interval, CoverageInterval(0, 9))
        self.assertIn("generation", combined.gaps[0].reason)
        self.assertEqual(combined.sketch.check("a"), 5)

    def test_bloom_results_idempotent(self):
        """bloom results union idempotently on overlap"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _bloom(["a"]))]
        )
        second = merge_partitions(
            [PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _bloom(["a", "b"]))]
        )
        combined = merge_results(first, second)
        self.assertTrue(combined.total_known)
        self.assertTrue("a" in combined.sketch and "b" in combined.sketch)

    def test_incompatible_kinds_rejected(self):
        """bloom and count-min results cannot share an aggregate"""
        first = merge_partitions(
            [PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms())]
        )
        second = merge_partitions(
            [PartitionRecord("p2", 1, [CoverageInterval(10, 19)], _bloom())]
        )
        combined = merge_results(first, second)
        self.assertFalse(combined.total_known)
        self.assertEqual(len(combined.rejected), 1)

    def test_gaps_propagate(self):
        """undetermined totals in either input stay undetermined"""
        first = merge_partitions(
            [
                PartitionRecord("p1", 1, [CoverageInterval(0, 9)], _cms(items=[("a", 5)])),
                PartitionRecord("p2", 1, [CoverageInterval(5, 14)], _cms(items=[("b", 2)])),
            ]
        )
        second = merge_partitions(
            [PartitionRecord("p3", 1, [CoverageInterval(20, 29)], _cms(items=[("c", 1)]))]
        )
        combined = merge_results(first, second)
        self.assertFalse(combined.total_known)
        self.assertEqual(combined.included_partitions, ("p1", "p3"))
        self.assertEqual(combined.sketch.check("c"), 1)


if __name__ == "__main__":
    unittest.main()
