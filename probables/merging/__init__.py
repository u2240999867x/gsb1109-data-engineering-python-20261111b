"""Partition-aware merging of probabilistic data structures.

This package merges per-partition Bloom filters and Count-Min sketches while
keeping the provenance (partition id, generation, coverage intervals and
caller-bound legacy hash conventions) attached to every contribution.
"""

from probables.merging.intervals import (
    CoverageInterval,
    find_overlaps,
    normalize_intervals,
    subtract_intervals,
)
from probables.merging.merger import merge_partitions, merge_results
from probables.merging.records import (
    LEGACY_HASH_CONVENTIONS,
    MergeGap,
    MergeResult,
    PartitionRecord,
    register_legacy_hash_convention,
)

__all__ = [
    "CoverageInterval",
    "MergeGap",
    "MergeResult",
    "PartitionRecord",
    "LEGACY_HASH_CONVENTIONS",
    "find_overlaps",
    "merge_partitions",
    "merge_results",
    "normalize_intervals",
    "register_legacy_hash_convention",
    "subtract_intervals",
]
