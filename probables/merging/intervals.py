"""Coverage interval arithmetic for partitioned sketch merging.

Intervals live on a discrete, totally ordered domain (for example hourly
bucket ids or event offsets) and are inclusive on both ends. Two intervals
that touch (``end + 1 == start``) are considered adjacent and may be stitched
together without leaving a hole.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class CoverageInterval:
    """Inclusive interval ``[start, end]`` on an ordered integer domain.

    Args:
        start (int): First covered position, inclusive.
        end (int): Last covered position, inclusive.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("CoverageInterval bounds must be integers")
        if self.start > self.end:
            raise ValueError("CoverageInterval requires start <= end")

    def intersects(self, other: "CoverageInterval") -> bool:
        """Return True when the two intervals share at least one position."""
        return self.start <= other.end and other.start <= self.end

    def intersection(self, other: "CoverageInterval") -> "CoverageInterval | None":
        """Return the shared sub-interval, or None when they are disjoint."""
        lo = max(self.start, other.start)
        hi = min(self.end, other.end)
        if lo > hi:
            return None
        return CoverageInterval(lo, hi)


def normalize_intervals(intervals: list[CoverageInterval] | tuple[CoverageInterval, ...]) -> list[CoverageInterval]:
    """Merge overlapping or adjacent intervals into a sorted, disjoint list."""
    ordered = sorted(intervals, key=lambda iv: (iv.start, iv.end))
    merged: list[CoverageInterval] = []
    for interval in ordered:
        if not merged:
            merged.append(interval)
            continue
        last = merged[-1]
        if interval.start <= last.end + 1:
            merged[-1] = CoverageInterval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def find_overlaps(
    interval: CoverageInterval, covered: list[CoverageInterval]
) -> list[CoverageInterval]:
    """Return the portions of ``interval`` already inside ``covered``.

    ``covered`` is expected to be normalized (see :func:`normalize_intervals`).
    """
    overlaps: list[CoverageInterval] = []
    for other in covered:
        if other.end < interval.start:
            continue
        if other.start > interval.end:
            break
        shared = interval.intersection(other)
        if shared is not None:
            overlaps.append(shared)
    return overlaps


def subtract_intervals(
    target: CoverageInterval, covered: list[CoverageInterval]
) -> list[CoverageInterval]:
    """Return the portions of ``target`` not covered by ``covered``.

    ``covered`` is expected to be normalized. Adjacency does not count as
    coverage, so an interval ending immediately before ``cursor`` leaves no
    residual.
    """
    residuals: list[CoverageInterval] = []
    cursor = target.start
    for other in covered:
        if other.end < cursor:
            continue
        if other.start > target.end:
            break
        if other.start > cursor:
            residuals.append(CoverageInterval(cursor, min(other.start - 1, target.end)))
        cursor = max(cursor, other.end + 1)
        if cursor > target.end:
            break
    if cursor <= target.end:
        residuals.append(CoverageInterval(cursor, target.end))
    return residuals
