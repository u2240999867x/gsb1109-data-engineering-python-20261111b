#!/usr/bin/env python3
"""Unittest class for the mergeable windowed approximate de-duplication summary"""

import sys
import unittest
from pathlib import Path

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import Coverage, LateEventPolicy, WindowInsertResult, WindowedSummary, WindowSummaryError  # noqa: E402
from probables.hashes import default_md5  # noqa: E402
from tests.utilities import different_hash  # noqa: E402


class TestWindowedSummaryInit(unittest.TestCase):
    """Test WindowedSummary initialization and validation"""

    def test_defaults(self):
        """test default construction values"""
        summary = WindowedSummary(10, 100, 0.05)
        self.assertEqual(summary.bucket_width, 10)
        self.assertEqual(summary.estimated_elements_per_bucket, 100)
        self.assertEqual(summary.false_positive_rate, 0.05)
        self.assertEqual(summary.allowed_lateness, 0)
        self.assertIs(summary.late_policy, LateEventPolicy.REJECT)
        self.assertEqual(summary.partitions, ())
        self.assertEqual(summary.bucket_starts, ())
        self.assertEqual(summary.elements_added, 0)
        self.assertEqual(summary.inserts, 0)

    def test_invalid_bucket_width(self):
        """test non-positive and non-integer bucket widths are rejected"""
        for bad in (0, -5, 2.5):
            with self.assertRaises(WindowSummaryError):
                WindowedSummary(bad, 100, 0.05)

    def test_invalid_capacity(self):
        """test invalid per-bucket capacity is rejected"""
        for bad in (0, -1):
            with self.assertRaises(WindowSummaryError):
                WindowedSummary(10, bad, 0.05)

    def test_invalid_fpr(self):
        """test false positive rate bounds are enforced"""
        for bad in (-0.1, 1.0):
            with self.assertRaises(WindowSummaryError):
                WindowedSummary(10, 100, bad)

    def test_invalid_lateness(self):
        """test negative lateness is rejected"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary(10, 100, 0.05, allowed_lateness=-1)

    def test_preregistered_partitions(self):
        """test partitions may be declared up front"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["b", "a"])
        self.assertEqual(summary.partitions, ("a", "b"))
        self.assertIsNone(summary.watermark("a"))
        summary.register_partition("c")
        self.assertEqual(summary.partitions, ("a", "b", "c"))


class TestWindowedSummaryIngestion(unittest.TestCase):
    """Test event ingestion, watermarks and out-of-order handling"""

    def test_bucket_assignment(self):
        """test events land in the bucket owning their event time"""
        summary = WindowedSummary(10, 100, 0.05)
        result = summary.add("p1", 12, "a")
        self.assertEqual(result, WindowInsertResult(True, False, False, 10))
        self.assertEqual(summary.bucket_starts, (10,))
        self.assertEqual(summary.bucket_start_for(19), 10)
        self.assertEqual(summary.bucket_start_for(20), 20)

    def test_out_of_order_within_watermark_accepted(self):
        """test out-of-order events not past the watermark are accepted"""
        summary = WindowedSummary(10, 100, 0.05, allowed_lateness=5)
        summary.add("p1", 12, "a")
        summary.advance_watermark("p1", 15)
        result = summary.add("p1", 11, "b")
        self.assertTrue(result.accepted)
        self.assertFalse(result.late)
        self.assertEqual(result.bucket_start, 10)
        self.assertTrue(summary.check("a"))
        self.assertTrue(summary.check("b"))

    def test_lateness_boundary(self):
        """test the inclusive lateness boundary timestamp + lateness >= watermark"""
        summary = WindowedSummary(10, 100, 0.05, allowed_lateness=5)
        summary.advance_watermark("p1", 15)
        accepted = summary.add("p1", 10, "edge")
        self.assertTrue(accepted.accepted)
        with self.assertRaises(WindowSummaryError):
            summary.add("p1", 9, "too-late")

    def test_late_reject_policy(self):
        """test the default reject policy raises on late events"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.advance_watermark("p1", 20)
        with self.assertRaises(WindowSummaryError):
            summary.add("p1", 0, "late")
        self.assertEqual(summary.late_stats["late_rejected"], 1)
        self.assertFalse(summary.check("late"))

    def test_late_drop_policy(self):
        """test the drop policy discards late events deterministically"""
        summary = WindowedSummary(10, 100, 0.05, late_policy=LateEventPolicy.DROP)
        summary.advance_watermark("p1", 20)
        result = summary.add("p1", 0, "late")
        self.assertFalse(result.accepted)
        self.assertTrue(result.late)
        self.assertIsNone(result.bucket_start)
        self.assertIn("watermark", result.reason)
        self.assertEqual(summary.late_stats["late_dropped"], 1)
        self.assertFalse(summary.check("late"))

    def test_late_side_output_policy(self):
        """test the side output policy collects late events"""
        summary = WindowedSummary(10, 100, 0.05, late_policy=LateEventPolicy.SIDE_OUTPUT)
        summary.advance_watermark("p1", 20)
        result = summary.add("p1", 0, "late")
        self.assertFalse(result.accepted)
        self.assertEqual(summary.side_output, [("p1", 0, "late")])
        self.assertEqual(summary.late_stats["late_dropped"], 1)

    def test_add_late_within_bound(self):
        """test explicit late acceptance still validates the lateness bound"""
        summary = WindowedSummary(10, 100, 0.05, allowed_lateness=5, late_policy=LateEventPolicy.DROP)
        summary.advance_watermark("p1", 14)
        result = summary.add_late("p1", 9, "late-ok")
        self.assertTrue(result.accepted)
        self.assertTrue(result.late)
        self.assertEqual(summary.late_stats["late_accepted"], 1)
        with self.assertRaises(WindowSummaryError):
            summary.add_late("p1", 0, "too-late")

    def test_watermark_monotonic(self):
        """test watermarks cannot regress"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.advance_watermark("p1", 10)
        with self.assertRaises(WindowSummaryError):
            summary.advance_watermark("p1", 9)

    def test_duplicate_detection_shared_window(self):
        """test duplicates are detected across partitions and buckets"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 2, "same")
        result = summary.add("p2", 25, "same")
        self.assertTrue(result.duplicate)
        self.assertEqual(summary.elements_added, 2)
        self.assertEqual(summary.inserts, 1)
        forced = summary.add("p2", 25, "same", force=True)
        self.assertTrue(forced.duplicate)
        self.assertEqual(summary.inserts, 2)

    def test_check_bucket_requires_retained_bucket(self):
        """test checking a missing bucket raises instead of returning false"""
        summary = WindowedSummary(10, 100, 0.05)
        with self.assertRaises(WindowSummaryError):
            summary.check_bucket(0, "missing")


class TestWindowedSummarySealing(unittest.TestCase):
    """Test partition sealing, reopening and deterministic late handling"""

    def test_seal_requires_watermark(self):
        """test sealing without a watermark requires an explicit value"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 1, "a")
        with self.assertRaises(WindowSummaryError):
            summary.seal("p1")
        summary.seal("p1", watermark=10)
        self.assertTrue(summary.is_sealed("p1"))
        self.assertEqual(summary.watermark("p1"), 10)

    def test_sealed_partition_rejects_writes(self):
        """test sealed partitions reject every write until reopened"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.advance_watermark("p1", 10)
        summary.seal("p1")
        with self.assertRaises(WindowSummaryError):
            summary.add("p1", 11, "a")
        with self.assertRaises(WindowSummaryError):
            summary.advance_watermark("p1", 12)
        self.assertEqual(summary.late_stats["sealed_rejected"], 1)

    def test_reopen_judges_late_events_deterministically(self):
        """test late events after recovery follow the configured policy"""
        summary = WindowedSummary(10, 100, 0.05, late_policy=LateEventPolicy.DROP)
        summary.advance_watermark("p1", 20)
        summary.seal("p1")
        summary.reopen("p1")
        self.assertEqual(summary.watermark("p1"), 20)
        result = summary.add("p1", 5, "old-event")
        self.assertFalse(result.accepted)
        self.assertTrue(result.late)

    def test_reopen_unsealed_raises(self):
        """test reopening a non-sealed partition raises"""
        summary = WindowedSummary(10, 100, 0.05)
        with self.assertRaises(WindowSummaryError):
            summary.reopen("p1")

    def test_seal_releases_retention(self):
        """test sealing the lagging partition allows bucket rotation"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        summary.add("p1", 2, "a")
        summary.advance_watermark("p1", 30)
        self.assertEqual(summary.bucket_starts, (0,))
        summary.seal("p2", 30)
        self.assertEqual(summary.bucket_starts, ())
        self.assertEqual(summary.coverage().gaps, ((0, 10),))


class TestWindowedSummaryCoverage(unittest.TestCase):
    """Test complete versus uncertain coverage reporting"""

    def test_coverage_empty(self):
        """test coverage of an unused summary"""
        summary = WindowedSummary(10, 100, 0.05)
        coverage = summary.coverage()
        self.assertIsInstance(coverage, Coverage)
        self.assertIsNone(coverage.covered_start)
        self.assertIsNone(coverage.covered_end)
        self.assertIsNone(coverage.uncertain_end)
        self.assertEqual(summary.complete_intervals(), [])
        self.assertEqual(summary.uncertain_intervals(), [])

    def test_unknown_partition_means_uncertain(self):
        """test a registered partition without a watermark blocks completeness"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        summary.advance_watermark("p1", 30)
        coverage = summary.coverage()
        self.assertIsNone(coverage.covered_end)
        self.assertEqual(coverage.uncertain_end, 30)
        self.assertEqual(summary.complete_intervals(), [])

    def test_complete_and_uncertain_intervals(self):
        """test the min watermark delimits complete and uncertain spans"""
        summary = WindowedSummary(10, 100, 0.05, allowed_lateness=10)
        summary.add("p1", 5, "a")
        summary.add("p1", 25, "b")
        summary.advance_watermark("p1", 30)
        summary.advance_watermark("p2", 40)
        coverage = summary.coverage()
        self.assertEqual(coverage.covered_start, 20)
        self.assertEqual(coverage.covered_end, 30)
        self.assertEqual(coverage.uncertain_end, 40)
        self.assertEqual(summary.complete_intervals(), [(20, 30)])
        self.assertEqual(summary.uncertain_intervals(), [(30, 40)])

    def test_gaps_subtracted_from_intervals(self):
        """test rotated-out gaps are excluded from complete intervals"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 2, "a")
        summary.add("p1", 12, "b")
        summary.add("p1", 22, "c")
        summary.advance_watermark("p1", 15)
        coverage = summary.coverage()
        self.assertEqual(coverage.gaps, ((0, 10),))
        self.assertEqual(summary.bucket_starts, (10, 20))
        self.assertEqual(summary.complete_intervals(), [(10, 15)])

    def test_manual_eviction(self):
        """test explicit eviction rotates buckets and records a gap"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 2, "a")
        summary.add("p1", 12, "b")
        removed = summary.evict_before(10)
        self.assertEqual(removed, 1)
        self.assertEqual(summary.bucket_starts, (10,))
        self.assertFalse(summary.check("a"))
        self.assertEqual(summary.coverage().gaps, ((0, 10),))

    def test_evict_before_safety_guard(self):
        """test eviction past the global safe bound is refused"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        summary.add("p1", 2, "a")
        summary.advance_watermark("p1", 30)
        summary.advance_watermark("p2", 5)
        with self.assertRaises(WindowSummaryError):
            summary.evict_before(20)

    def test_no_false_negative_from_fast_partition(self):
        """test a fast partition cannot rotate a bucket a slow partition needs"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["fast", "slow"])
        summary.add("fast", 2, "keep-me")
        summary.advance_watermark("fast", 100)
        self.assertEqual(summary.bucket_starts, (0,))
        self.assertTrue(summary.check("keep-me"))
        summary.advance_watermark("slow", 100)
        self.assertEqual(summary.bucket_starts, ())


class TestWindowedSummaryMerge(unittest.TestCase):
    """Test merging per-partition summaries"""

    def _pair(self):
        first = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        second = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        first.add("p1", 2, "e1")
        first.add("p1", 12, "e2")
        first.advance_watermark("p1", 100)
        second.add("p2", 3, "e1")
        second.add("p2", 25, "e3")
        second.advance_watermark("p2", 30)
        return first, second

    def test_merge_requires_two(self):
        """test merging a single summary is rejected"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary.merge(WindowedSummary(10, 100, 0.05))

    def test_merge_unions_buckets_and_preserves_elements(self):
        """test confirmed elements survive merging even after one side rotates"""
        first, second = self._pair()
        self.assertEqual(first.bucket_starts, (0, 10))
        merged = WindowedSummary.merge(first, second)
        self.assertEqual(merged.bucket_starts, (0, 10, 20))
        for key in ("e1", "e2", "e3"):
            self.assertTrue(merged.check(key), key)

    def test_merge_coverage_complete_vs_uncertain(self):
        """test merged coverage uses the conservative minimum watermark"""
        first, second = self._pair()
        merged = WindowedSummary.merge(first, second)
        coverage = merged.coverage()
        self.assertEqual(coverage.covered_end, 30)
        self.assertEqual(coverage.uncertain_end, 100)
        self.assertEqual(merged.complete_intervals(), [(0, 30)])
        self.assertEqual(merged.uncertain_intervals(), [(30, 100)])

    def test_merge_combines_stats(self):
        """test merge aggregates ingestion and late-event counters"""
        first, second = self._pair()
        second.seal("p2", 30)
        merged = WindowedSummary.merge(first, second)
        self.assertEqual(merged.elements_added, 4)
        self.assertIn("p1", merged.partitions)
        self.assertIn("p2", merged.partitions)

    def test_merge_incompatible_width(self):
        """test summaries with different bucket widths cannot merge"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary.merge(
                WindowedSummary(10, 100, 0.05),
                WindowedSummary(20, 100, 0.05),
            )

    def test_merge_incompatible_hash(self):
        """test summaries with incompatible hash functions cannot merge"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary.merge(
                WindowedSummary(10, 100, 0.05),
                WindowedSummary(10, 100, 0.05, hash_function=different_hash),
            )

    def test_merge_incompatible_bucket_sizing(self):
        """test summaries with different bucket capacities cannot merge"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary.merge(
                WindowedSummary(10, 100, 0.05),
                WindowedSummary(10, 200, 0.05),
            )

    def test_merge_incompatible_lateness(self):
        """test summaries with different allowed lateness cannot merge"""
        with self.assertRaises(WindowSummaryError):
            WindowedSummary.merge(
                WindowedSummary(10, 100, 0.05),
                WindowedSummary(10, 100, 0.05, allowed_lateness=5),
            )

    def test_merge_sealed_only_when_every_input_sealed(self):
        """test a partition is sealed in the merge only if all inputs sealed it"""
        first = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        second = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        first.advance_watermark("p1", 40)
        second.advance_watermark("p1", 30)
        first.seal("p2", 50)
        second.seal("p2", 60)
        merged = WindowedSummary.merge(first, second)
        self.assertTrue(merged.is_sealed("p2"))
        self.assertEqual(merged.watermark("p2"), 50)
        self.assertFalse(merged.is_sealed("p1"))
        self.assertEqual(merged.watermark("p1"), 30)

        third = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        third.advance_watermark("p2", 70)
        merged_again = WindowedSummary.merge(first, third)
        self.assertFalse(merged_again.is_sealed("p2"))
        self.assertEqual(merged_again.watermark("p2"), 50)

    def test_merge_gap_does_not_erase_retained_bucket(self):
        """test a gap in one input cannot blank a bucket retained by another"""
        first = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        first.add("p1", 2, "a")
        first.add("p1", 12, "b")
        first.advance_watermark("p1", 30)
        self.assertEqual(first.evict_before(10), 1)
        second = WindowedSummary(10, 100, 0.05, partitions=["p1", "p2"])
        second.add("p2", 5, "c")
        second.advance_watermark("p2", 30)
        self.assertEqual(second.bucket_starts, (0,))
        merged = WindowedSummary.merge(first, second)
        self.assertIn(0, merged.bucket_starts)
        self.assertTrue(merged.check("c"))
        self.assertNotIn((0, 10), merged.coverage().gaps)


class TestWindowedSummaryRegranulate(unittest.TestCase):
    """Test bucket granularity changes"""

    def test_coarsen_aligned_reuses_buckets(self):
        """test aligned old buckets are reused on the coarser grid"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 2, "a")
        summary.add("p1", 22, "b")
        coarse = summary.regranulate(20)
        self.assertEqual(coarse.bucket_width, 20)
        self.assertEqual(coarse.bucket_starts, (0, 20))
        for key in ("a", "b"):
            self.assertTrue(coarse.check(key), key)

    def test_refine_aligned(self):
        """test refining to an aligned divisor keeps bucket boundaries"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 2, "a")
        fine = summary.regranulate(5)
        self.assertEqual(fine.bucket_width, 5)
        self.assertEqual(fine.bucket_starts, (0,))
        self.assertTrue(fine.check("a"))

    def test_unaligned_width_rejected(self):
        """test non-multiple widths are rejected"""
        summary = WindowedSummary(10, 100, 0.05)
        with self.assertRaises(WindowSummaryError):
            summary.regranulate(7)

    def test_same_width_rejected(self):
        """test regranulating to the current width is rejected"""
        summary = WindowedSummary(10, 100, 0.05)
        with self.assertRaises(WindowSummaryError):
            summary.regranulate(10)

    def test_straddling_bucket_rejected(self):
        """test retained buckets off the new grid block regranulation"""
        summary = WindowedSummary(10, 100, 0.05)
        summary.add("p1", 12, "a")
        with self.assertRaises(WindowSummaryError):
            summary.regranulate(20)

    def test_regranulate_preserves_partition_state(self):
        """test watermarks and sealing survive a granularity change"""
        summary = WindowedSummary(10, 100, 0.05, partitions=["p1"])
        summary.advance_watermark("p1", 20)
        summary.seal("p1")
        coarse = summary.regranulate(20)
        self.assertTrue(coarse.is_sealed("p1"))
        self.assertEqual(coarse.watermark("p1"), 20)


class TestWindowedSummaryFalsePositiveRate(unittest.TestCase):
    """Test that probability estimates are separate from time coverage"""

    def test_empty_fpr_is_zero(self):
        """test empty buckets report zero false positive rate"""
        summary = WindowedSummary(10, 100, 0.05)
        self.assertEqual(summary.window_false_positive_rate(), 0.0)

    def test_bucket_and_window_fpr(self):
        """test per-bucket and window-wide false positive estimates"""
        summary = WindowedSummary(10, 100, 0.05)
        for idx in range(50):
            summary.add("p1", 2, f"key-{idx}")
        for idx in range(25):
            summary.add("p1", 12, f"other-{idx}")
        bucket_rate = summary.bucket_false_positive_rate(0)
        self.assertGreater(bucket_rate, 0.0)
        self.assertLess(bucket_rate, 1.0)
        window_rate = summary.window_false_positive_rate()
        self.assertGreater(window_rate, bucket_rate)
        self.assertLess(window_rate, 1.0)

    def test_missing_bucket_fpr_raises(self):
        """test requesting the FPR of a missing bucket raises"""
        summary = WindowedSummary(10, 100, 0.05)
        with self.assertRaises(WindowSummaryError):
            summary.bucket_false_positive_rate(0)

    def test_custom_hash_function(self):
        """test summaries accept a pluggable hash function"""
        summary = WindowedSummary(10, 100, 0.05, hash_function=default_md5)
        summary.add("p1", 1, "hashed")
        self.assertTrue(summary.check("hashed"))
        self.assertIs(summary.hash_function, default_md5)


if __name__ == "__main__":
    unittest.main()
