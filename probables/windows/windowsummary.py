"""Mergeable, event-time windowed approximate de-duplication summaries.

A :class:`WindowedSummary` lets several collection streams (partitions) share a
single approximate de-duplication window while each partition keeps its own
event-time watermark. Events are placed into fixed-width event-time buckets,
each backed by a :class:`~probables.BloomFilter`, so out-of-order events that
have not crossed their partition watermark (minus an explicit allowed
lateness) are still accepted into the correct bucket.

License: MIT
Author: Tyler Barrus (barrust@gmail.com)
URL: https://github.com/barrust/pyprobables
"""

from enum import Enum
from math import exp
from typing import Iterable, NamedTuple, Optional

from probables.blooms.bloom import BloomFilter
from probables.exceptions import SimilarityError, WindowSummaryError
from probables.hashes import HashFuncT, KeyT, default_fnv_1a


class LateEventPolicy(Enum):
    """Deterministic handling for events that arrive after their partition
    watermark has advanced past the allowed lateness.

    REJECT: raise :class:`~probables.exceptions.WindowSummaryError`.
    DROP: silently discard the event and report it in the late-event stats.
    SIDE_OUTPUT: collect the event in a side output list instead of the window.
    """

    REJECT = "reject"
    DROP = "drop"
    SIDE_OUTPUT = "side_output"


class Coverage(NamedTuple):
    """Time coverage of a (possibly merged) windowed summary.

    All bounds are event-time timestamps; half-open intervals are used
    everywhere (``[start, end)``).

    Args:
        covered_start: Lower edge of the oldest retained bucket, or ``None``
            when no bucket has been created yet.
        covered_end: Global completeness watermark. Every partition has
            observed all events with timestamps below this value, so the span
            ``[covered_start, covered_end)`` (minus ``gaps``) can be answered
            definitively. ``None`` means no span is provably complete yet.
        uncertain_end: The most advanced partition watermark. The span
            ``[covered_end, uncertain_end)`` can only be answered as "not yet
            certain" because at least one partition may still deliver events
            there. ``None`` when no watermark has been seen.
        gaps: Retained-window ranges that were rotated out (garbage collected)
            and can no longer answer membership queries.
    """

    covered_start: Optional[int]
    covered_end: Optional[int]
    uncertain_end: Optional[int]
    gaps: tuple[tuple[int, int], ...] = ()


class WindowInsertResult(NamedTuple):
    """Outcome of a single :meth:`WindowedSummary.add` call.

    Args:
        accepted: ``True`` when the event was admitted to the window.
        late: ``True`` when the event took a late-event path: it was diverted
            by the late policy, or explicitly admitted through
            :meth:`WindowedSummary.add_late`. Out-of-order events still inside
            the lateness bound are admitted normally and report ``False``.
        duplicate: ``True`` when an admitted event matched an element already
            present in the shared window.
        bucket_start: Event-time start of the target bucket, or ``None`` when
            the event was not admitted.
        reason: Machine readable explanation when the event was not admitted.
    """

    accepted: bool
    late: bool
    duplicate: bool
    bucket_start: Optional[int]
    reason: Optional[str] = None


def _hash_functions_compatible(first: HashFuncT, second: HashFuncT) -> bool:
    """Return whether two hash functions produce identical digest lists."""
    if first is second:
        return True
    probe_depth = 3
    try:
        return first("__window_summary_probe__", probe_depth) == second("__window_summary_probe__", probe_depth)
    except Exception:  # pragma: no cover - defensive guard for exotic callables
        return False


class WindowedSummary:
    """Mergeable approximate de-duplication window with per-partition
    watermarks.

    Multiple collection streams (partitions) share one de-duplication window.
    Each partition tracks its own event-time watermark and may be sealed while
    idle. Events are bucketed by event time into Bloom Filters; an event is
    accepted as long as it has not crossed its partition watermark by more
    than ``allowed_lateness``. Buckets are only rotated out once *every*
    partition has advanced past them, so an event already admitted can never
    become a false negative because a faster partition rotated its bucket.

    Args:
        bucket_width (int): Width, in event-time units, of each bucket. Must
            be a positive integer.
        est_elements_per_bucket (int): Number of elements each bucket is sized
            to hold at the configured false positive rate.
        false_positive_rate (float): Per-bucket target false positive rate.
        allowed_lateness (int): Non-negative number of event-time units an
            event may trail its partition watermark and still be accepted.
        late_policy (LateEventPolicy): Deterministic handling of events that
            miss the lateness bound.
        hash_function (function): Hashing strategy function to use
            ``hf(key, number)``.
        partitions (Iterable[str]): Partition identifiers known up front, for
            example when per-partition summaries are built independently and
            merged later. A pre-registered partition without a watermark holds
            back bucket rotation, so a fast partition cannot rotate a bucket a
            slower partition may still need.

    Raises:
        WindowSummaryError: When an initialization parameter is invalid.
    """

    __slots__ = (
        "_width",
        "_est_elements",
        "_fpr",
        "_lateness",
        "_policy",
        "_hash_func",
        "_buckets",
        "_counts",
        "_partitions",
        "_watermarks",
        "_sealed",
        "_seal_watermarks",
        "_gaps",
        "_observed",
        "_inserts",
        "_late_accepted",
        "_late_dropped",
        "_late_rejected",
        "_sealed_rejected",
        "_side_output",
    )

    def __init__(
        self,
        bucket_width: int,
        est_elements_per_bucket: int,
        false_positive_rate: float,
        allowed_lateness: int = 0,
        late_policy: LateEventPolicy = LateEventPolicy.REJECT,
        hash_function: HashFuncT | None = None,
        partitions: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(bucket_width, int) or bucket_width <= 0:
            raise WindowSummaryError("bucket_width must be a positive integer")
        if not isinstance(est_elements_per_bucket, int) or est_elements_per_bucket <= 0:
            raise WindowSummaryError("est_elements_per_bucket must be a positive integer")
        if not isinstance(false_positive_rate, (int, float)) or not 0.0 <= float(false_positive_rate) < 1.0:
            raise WindowSummaryError("false_positive_rate must be between 0.0 (inclusive) and 1.0 (exclusive)")
        if not isinstance(allowed_lateness, int) or allowed_lateness < 0:
            raise WindowSummaryError("allowed_lateness must be a non-negative integer")
        if not isinstance(late_policy, LateEventPolicy):
            raise WindowSummaryError("late_policy must be a LateEventPolicy value")

        self._width = bucket_width
        self._est_elements = est_elements_per_bucket
        self._fpr = float(false_positive_rate)
        self._lateness = allowed_lateness
        self._policy = late_policy
        self._hash_func: HashFuncT = hash_function if hash_function is not None else default_fnv_1a

        self._buckets: dict[int, BloomFilter] = {}
        self._counts: dict[int, int] = {}
        self._partitions: set[str] = set()
        self._watermarks: dict[str, int] = {}
        self._sealed: dict[str, bool] = {}
        self._seal_watermarks: dict[str, int] = {}
        self._gaps: list[tuple[int, int]] = []
        if partitions is not None:
            for partition in partitions:
                self._register(partition)
        self._observed = 0
        self._inserts = 0
        self._late_accepted = 0
        self._late_dropped = 0
        self._late_rejected = 0
        self._sealed_rejected = 0
        self._side_output: list[tuple[str, int, KeyT]] = []

    def __contains__(self, key: KeyT) -> bool:
        return self.check(key)

    def __len__(self) -> int:
        return len(self._buckets)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def bucket_width(self) -> int:
        """int: Event-time width of every bucket."""
        return self._width

    @property
    def estimated_elements_per_bucket(self) -> int:
        """int: The per-bucket capacity the Bloom Filters were sized for."""
        return self._est_elements

    @property
    def false_positive_rate(self) -> float:
        """float: The configured per-bucket target false positive rate."""
        return self._fpr

    @property
    def allowed_lateness(self) -> int:
        """int: Event-time units an event may trail its watermark."""
        return self._lateness

    @property
    def late_policy(self) -> LateEventPolicy:
        """LateEventPolicy: Policy applied to events past the lateness bound."""
        return self._policy

    @property
    def hash_function(self) -> HashFuncT:
        """function: The hashing strategy used by every bucket."""
        return self._hash_func

    @property
    def partitions(self) -> tuple[str, ...]:
        """tuple[str, ...]: All partitions known to this summary."""
        return tuple(sorted(self._partitions))

    @property
    def bucket_starts(self) -> tuple[int, ...]:
        """tuple[int, ...]: Event-time starts of the retained buckets."""
        return tuple(sorted(self._buckets))

    @property
    def elements_added(self) -> int:
        """int: Number of events admitted to the window, including duplicates."""
        return self._observed

    @property
    def inserts(self) -> int:
        """int: Number of elements actually written into a bucket."""
        return self._inserts

    @property
    def late_stats(self) -> dict[str, int]:
        """dict[str, int]: Counters for late and sealed-partition handling."""
        return {
            "late_accepted": self._late_accepted,
            "late_dropped": self._late_dropped,
            "late_rejected": self._late_rejected,
            "sealed_rejected": self._sealed_rejected,
        }

    @property
    def side_output(self) -> list[tuple[str, int, KeyT]]:
        """list: Events diverted by the ``SIDE_OUTPUT`` late policy."""
        return list(self._side_output)

    # ------------------------------------------------------------------ #
    # Partition and watermark management
    # ------------------------------------------------------------------ #
    def _register(self, partition: str) -> None:
        if partition not in self._partitions:
            self._partitions.add(partition)
            self._sealed[partition] = False

    def register_partition(self, partition: str) -> None:
        """Declare a partition before it emits any event or watermark.

        A registered partition without a watermark participates in coverage
        and rotation checks as an unknown-lag consumer, holding back bucket
        rotation until it advances or is sealed.

        Args:
            partition (str): The partition identifier.
        """
        self._register(partition)

    def _effective_watermark(self, partition: str) -> Optional[int]:
        """Watermark in force for a partition, including a sealed value."""
        if self._sealed.get(partition, False):
            return self._seal_watermarks[partition]
        return self._watermarks.get(partition)

    def watermark(self, partition: str) -> Optional[int]:
        """Return the effective event-time watermark of a partition.

        Args:
            partition (str): The partition identifier.
        Returns:
            int|None: The watermark timestamp, or ``None`` if none has been
            observed (sealed partitions report their seal watermark).
        """
        return self._effective_watermark(partition)

    def watermarks(self) -> dict[str, Optional[int]]:
        """dict: Effective watermark of every known partition."""
        return {partition: self._effective_watermark(partition) for partition in self.partitions}

    def is_sealed(self, partition: str) -> bool:
        """bool: Whether the partition is explicitly sealed."""
        return self._sealed.get(partition, False)

    def advance_watermark(self, partition: str, timestamp: int) -> None:
        """Advance a partition's event-time watermark.

        Watermarks are monotonic; advancing them may rotate buckets that every
        partition has moved past.

        Args:
            partition (str): The partition identifier.
            timestamp (int): The new watermark timestamp.
        Raises:
            WindowSummaryError: When the partition is sealed, the timestamp is
                not an integer, or it regresses an existing watermark.
        """
        if not isinstance(timestamp, int):
            raise WindowSummaryError("watermark timestamps must be integers")
        self._register(partition)
        if self._sealed[partition]:
            raise WindowSummaryError(f"partition '{partition}' is sealed; reopen it before advancing its watermark")
        current = self._watermarks.get(partition)
        if current is not None and timestamp < current:
            raise WindowSummaryError(
                f"watermark for partition '{partition}' cannot regress from {current} to {timestamp}"
            )
        self._watermarks[partition] = timestamp
        self._garbage_collect()

    def seal(self, partition: str, watermark: Optional[int] = None) -> None:
        """Explicitly seal an idle partition at its current (or a given)
        watermark.

        Sealed partitions reject all writes until :meth:`reopen` is called, but
        their seal watermark still participates in global completeness and
        bucket rotation. This lets an idle, caught-up partition release shared
        window state instead of holding it forever.

        Args:
            partition (str): The partition identifier.
            watermark (int|None): Seal watermark; defaults to the current one.
        Raises:
            WindowSummaryError: When no watermark is available, the value is
                invalid, or the partition is already sealed.
        """
        self._register(partition)
        if self._sealed[partition]:
            raise WindowSummaryError(f"partition '{partition}' is already sealed")
        current = self._watermarks.get(partition)
        if watermark is None:
            if current is None:
                raise WindowSummaryError(
                    f"partition '{partition}' has no watermark; provide an explicit seal watermark"
                )
            seal_wm = current
        else:
            if not isinstance(watermark, int):
                raise WindowSummaryError("seal watermark must be an integer")
            if current is not None and watermark < current:
                raise WindowSummaryError(
                    f"seal watermark cannot regress partition '{partition}' from {current} to {watermark}"
                )
            seal_wm = watermark
        self._sealed[partition] = True
        self._seal_watermarks[partition] = seal_wm
        self._garbage_collect()

    def reopen(self, partition: str) -> None:
        """Reopen a sealed partition when it resumes emitting events.

        The seal watermark is retained as the partition's watermark, so events
        arriving after the idle period are judged against it by the configured
        :class:`LateEventPolicy`.

        Args:
            partition (str): The partition identifier.
        Raises:
            WindowSummaryError: When the partition is not currently sealed.
        """
        if not self._sealed.get(partition, False):
            raise WindowSummaryError(f"partition '{partition}' is not sealed")
        seal_wm = self._seal_watermarks.pop(partition)
        self._sealed[partition] = False
        self._watermarks[partition] = seal_wm

    # ------------------------------------------------------------------ #
    # Event ingestion
    # ------------------------------------------------------------------ #
    def bucket_start_for(self, timestamp: int) -> int:
        """Return the event-time bucket start that owns ``timestamp``."""
        return (timestamp // self._width) * self._width

    def add(self, partition: str, timestamp: int, key: KeyT, force: bool = False) -> WindowInsertResult:
        """Add an event from ``partition`` at event time ``timestamp``.

        Out-of-order events are accepted while they remain within the
        partition's effective watermark minus :attr:`allowed_lateness`, and
        every accepted event is written to the bucket owning its event time.
        Events past the bound are handled deterministically by
        :attr:`late_policy`. Membership is checked against every retained
        bucket of the shared window, so duplicates are detected regardless of
        which partition or bucket observed them first.

        Args:
            partition (str): The originating partition.
            timestamp (int): The event-time timestamp.
            key (str|bytes): The element to de-duplicate.
            force (bool): Insert even when the element is already present.
        Returns:
            WindowInsertResult: The deterministic outcome of the insertion.
        Raises:
            WindowSummaryError: When the timestamp is invalid, the partition is
                sealed, or the late policy is ``REJECT`` for a late event.
        """
        if not isinstance(timestamp, int):
            raise WindowSummaryError("event timestamps must be integers")
        self._register(partition)

        if self._sealed[partition]:
            self._sealed_rejected += 1
            raise WindowSummaryError(
                f"partition '{partition}' is sealed at {self._seal_watermarks[partition]}; "
                "reopen it before adding events"
            )

        watermark = self._effective_watermark(partition)
        is_late = watermark is not None and timestamp + self._lateness < watermark
        bucket_start = self.bucket_start_for(timestamp)

        if is_late:
            return self._handle_late(partition, timestamp, key, bucket_start, watermark)

        self._observed += 1
        bucket = self._buckets.get(bucket_start)
        if bucket is None:
            bucket = self._new_bucket()
            self._buckets[bucket_start] = bucket
            self._counts[bucket_start] = 0
        hashes = bucket.hashes(key)
        is_duplicate = any(other.check_alt(hashes) for other in self._buckets.values())
        if is_duplicate and not force:
            return WindowInsertResult(True, False, True, bucket_start, "duplicate")
        bucket.add_alt(hashes)
        self._counts[bucket_start] += 1
        self._inserts += 1
        return WindowInsertResult(True, False, is_duplicate, bucket_start)

    def _handle_late(
        self,
        partition: str,
        timestamp: int,
        key: KeyT,
        bucket_start: int,
        watermark: Optional[int],
    ) -> WindowInsertResult:
        reason = (
            f"event at {timestamp} is past partition '{partition}' watermark {watermark} "
            f"by more than allowed_lateness={self._lateness}"
        )
        if self._policy is LateEventPolicy.REJECT:
            self._late_rejected += 1
            raise WindowSummaryError(reason)
        if self._policy is LateEventPolicy.SIDE_OUTPUT:
            self._side_output.append((partition, timestamp, key))
        self._late_dropped += 1
        return WindowInsertResult(False, True, False, None, reason)

    def add_late(self, partition: str, timestamp: int, key: KeyT, force: bool = False) -> WindowInsertResult:
        """Admit an event known to be late, provided its bucket is still
        retained and the event stays within the allowed lateness.

        This gives callers an explicit, deterministic way to accept late
        events under any :class:`LateEventPolicy`: lateness is still validated,
        but a policy rejection/drop is bypassed for events within the bound.
        Events whose bucket was already rotated out cannot be admitted without
        creating a false-negative risk and raise instead.

        Args:
            partition (str): The originating partition.
            timestamp (int): The event-time timestamp.
            key (str|bytes): The element to de-duplicate.
            force (bool): Insert even when the element is already present.
        Returns:
            WindowInsertResult: The outcome of the insertion.
        Raises:
            WindowSummaryError: When the event exceeds the lateness bound or
                its event-time bucket has already been rotated out.
        """
        if not isinstance(timestamp, int):
            raise WindowSummaryError("event timestamps must be integers")
        self._register(partition)
        if self._sealed[partition]:
            self._sealed_rejected += 1
            raise WindowSummaryError(
                f"partition '{partition}' is sealed at {self._seal_watermarks[partition]}; "
                "reopen it before adding events"
            )
        watermark = self._effective_watermark(partition)
        if watermark is not None and timestamp + self._lateness < watermark:
            self._late_rejected += 1
            raise WindowSummaryError(
                f"event at {timestamp} exceeds allowed_lateness={self._lateness} behind watermark {watermark}"
            )
        bucket_start = self.bucket_start_for(timestamp)
        if bucket_start not in self._buckets and self._is_below_retention(bucket_start):
            self._late_rejected += 1
            raise WindowSummaryError(
                f"bucket starting at {bucket_start} was rotated out; the late event cannot be admitted"
            )
        self._observed += 1
        self._late_accepted += 1
        bucket = self._buckets.get(bucket_start)
        if bucket is None:
            bucket = self._new_bucket()
            self._buckets[bucket_start] = bucket
            self._counts[bucket_start] = 0
        hashes = bucket.hashes(key)
        is_duplicate = any(other.check_alt(hashes) for other in self._buckets.values())
        if is_duplicate and not force:
            return WindowInsertResult(True, True, True, bucket_start, "duplicate")
        bucket.add_alt(hashes)
        self._counts[bucket_start] += 1
        self._inserts += 1
        return WindowInsertResult(True, True, is_duplicate, bucket_start)

    # ------------------------------------------------------------------ #
    # Membership queries
    # ------------------------------------------------------------------ #
    def check(self, key: KeyT) -> bool:
        """Check the shared window for ``key`` across every retained bucket.

        Args:
            key (str|bytes): The element to look up.
        Returns:
            bool: ``True`` if the element is likely present in any retained
            bucket; ``False`` if definitely absent from the retained window.
        """
        if not self._buckets:
            return False
        hashes = next(iter(self._buckets.values())).hashes(key)
        return any(bucket.check_alt(hashes) for bucket in self._buckets.values())

    def check_bucket(self, bucket_start: int, key: KeyT) -> bool:
        """Check a single event-time bucket for ``key``.

        Raises:
            WindowSummaryError: When ``bucket_start`` is not a retained bucket.
        """
        bucket = self._buckets.get(bucket_start)
        if bucket is None:
            raise WindowSummaryError(f"no retained bucket starts at {bucket_start}")
        return bucket.check(key)

    # ------------------------------------------------------------------ #
    # Coverage: what is complete versus uncertain
    # ------------------------------------------------------------------ #
    def coverage(self) -> Coverage:
        """Describe the time coverage of the summary.

        The completeness watermark is the minimum effective watermark across
        all partitions: below it every partition has delivered all events it
        will deliver for those timestamps. Anything between that minimum and
        the most advanced partition watermark is only partially observed and
        is reported as uncertain.
        """
        effective = {partition: self._effective_watermark(partition) for partition in self._partitions}
        known = [wm for wm in effective.values() if wm is not None]
        if len(known) == len(effective) and known:
            covered_end: Optional[int] = min(known)
        else:
            covered_end = None
        uncertain_end = max(known) if known else None
        if self._buckets:
            covered_start = min(self._buckets)
        else:
            covered_start = None
        return Coverage(covered_start, covered_end, uncertain_end, tuple(self._gaps))

    def complete_intervals(self) -> list[tuple[int, int]]:
        """list[tuple[int, int]]: Retained intervals every partition covers
        completely, with rotated-out gaps subtracted."""
        info = self.coverage()
        if info.covered_start is None or info.covered_end is None:
            return []
        return self._subtract_gaps(info.covered_start, info.covered_end)

    def uncertain_intervals(self) -> list[tuple[int, int]]:
        """list[tuple[int, int]]: Intervals that can only be answered as "not
        yet certain", because a slower partition may still deliver events."""
        info = self.coverage()
        if info.covered_end is None or info.uncertain_end is None:
            return []
        if info.covered_end >= info.uncertain_end:
            return []
        return self._subtract_gaps(info.covered_end, info.uncertain_end)

    def _subtract_gaps(self, start: int, end: int) -> list[tuple[int, int]]:
        intervals = [(start, end)]
        for gap_start, gap_end in self._gaps:
            remaining: list[tuple[int, int]] = []
            for seg_start, seg_end in intervals:
                if gap_end <= seg_start or gap_start >= seg_end:
                    remaining.append((seg_start, seg_end))
                    continue
                if gap_start > seg_start:
                    remaining.append((seg_start, min(gap_start, seg_end)))
                if gap_end < seg_end:
                    remaining.append((max(gap_end, seg_start), seg_end))
            intervals = remaining
        return [(seg_start, seg_end) for seg_start, seg_end in intervals if seg_end > seg_start]

    @staticmethod
    def _union_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping or adjacent half-open ranges into disjoint spans."""
        if not ranges:
            return []
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _subtract_retained_spans(self, start: int, end: int) -> list[tuple[int, int]]:
        """Subtract the span of every retained bucket from ``[start, end)``."""
        intervals = [(start, end)]
        for bucket_start in sorted(self._buckets):
            bucket_end = bucket_start + self._width
            remaining: list[tuple[int, int]] = []
            for seg_start, seg_end in intervals:
                if bucket_end <= seg_start or bucket_start >= seg_end:
                    remaining.append((seg_start, seg_end))
                    continue
                if bucket_start > seg_start:
                    remaining.append((seg_start, min(bucket_start, seg_end)))
                if bucket_end < seg_end:
                    remaining.append((max(bucket_end, seg_start), seg_end))
            intervals = remaining
        return [(seg_start, seg_end) for seg_start, seg_end in intervals if seg_end > seg_start]

    # ------------------------------------------------------------------ #
    # Bucket rotation
    # ------------------------------------------------------------------ #
    def _rotation_bound(self) -> Optional[int]:
        """Timestamps below this value need no bucket for any partition.

        Returns ``None`` while any known partition has no effective watermark,
        because its lag is unknown and it may still need every retained bucket.
        """
        watermarks = [self._effective_watermark(partition) for partition in self._partitions]
        if not watermarks or any(watermark is None for watermark in watermarks):
            return None
        return min(watermarks) - self._lateness

    def _is_below_retention(self, bucket_start: int) -> bool:
        bound = self._rotation_bound()
        return bound is not None and bucket_start + self._width <= bound

    def _garbage_collect(self) -> None:
        bound = self._rotation_bound()
        if bound is None:
            return
        for start in sorted(self._buckets):
            if start + self._width <= bound:
                self._record_gap(start, start + self._width)
                del self._buckets[start]
                del self._counts[start]
            else:
                break

    def evict_before(self, timestamp: int) -> int:
        """Manually rotate buckets ending at or before ``timestamp``.

        Args:
            timestamp (int): Exclusive eviction bound.
        Returns:
            int: Number of buckets rotated out.
        Raises:
            WindowSummaryError: When eviction would remove a bucket that some
                partition may still need (within its lateness bound), which
                would introduce false negatives.
        """
        safe_bound = self._rotation_bound()
        if safe_bound is not None and timestamp > safe_bound:
            raise WindowSummaryError(
                f"refusing to evict through {timestamp}: partitions may still accept events "
                f"up to {safe_bound} (global watermark minus allowed_lateness)"
            )
        removed = 0
        for start in sorted(self._buckets):
            if start + self._width <= timestamp:
                self._record_gap(start, start + self._width)
                del self._buckets[start]
                del self._counts[start]
                removed += 1
            else:
                break
        return removed

    def _record_gap(self, start: int, end: int) -> None:
        gaps = list(self._gaps)
        inserted = False
        for idx, (gap_start, gap_end) in enumerate(gaps):
            if end < gap_start:
                gaps.insert(idx, (start, end))
                inserted = True
                break
            if start <= gap_end and end >= gap_start:
                gaps[idx] = (min(start, gap_start), max(end, gap_end))
                inserted = True
                break
        if not inserted:
            gaps.append((start, end))
        merged: list[tuple[int, int]] = []
        for gap in gaps:
            if merged and gap[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], gap[1]))
            else:
                merged.append(gap)
        self._gaps = merged

    # ------------------------------------------------------------------ #
    # False positive estimation, kept separate from time coverage
    # ------------------------------------------------------------------ #
    def bucket_false_positive_rate(self, bucket_start: int) -> float:
        """Estimate the actual false positive rate of one retained bucket
        from its current load.

        Raises:
            WindowSummaryError: When ``bucket_start`` is not retained.
        """
        bucket = self._buckets.get(bucket_start)
        if bucket is None:
            raise WindowSummaryError(f"no retained bucket starts at {bucket_start}")
        inserted = self._counts[bucket_start]
        if inserted == 0:
            return 0.0
        hashes = bucket.number_hashes
        bits = bucket.number_bits
        return (1.0 - exp(-hashes * inserted / bits)) ** hashes

    def window_false_positive_rate(self) -> float:
        """float: Estimated false positive rate of a membership query over all
        retained buckets (the per-bucket rates combined as a logical OR)."""
        if not self._buckets:
            return 0.0
        no_false_positive = 1.0
        for start in self._buckets:
            no_false_positive *= 1.0 - self.bucket_false_positive_rate(start)
        return 1.0 - no_false_positive

    # ------------------------------------------------------------------ #
    # Merging
    # ------------------------------------------------------------------ #
    def _validate_mergeable(self, other: "WindowedSummary") -> None:
        if not isinstance(other, WindowedSummary):
            raise WindowSummaryError("merging requires another WindowedSummary instance")
        if self._width != other._width:
            raise WindowSummaryError(
                f"bucket width mismatch: {self._width} versus {other._width}; regranulate before merging"
            )
        if self._est_elements != other._est_elements or self._fpr != other._fpr:
            raise WindowSummaryError("per-bucket capacity and false positive rate must match for merging")
        if self._lateness != other._lateness:
            raise WindowSummaryError(
                f"allowed lateness mismatch: {self._lateness} versus {other._lateness}; align before merging"
            )
        if not _hash_functions_compatible(self._hash_func, other._hash_func):
            raise WindowSummaryError("hash functions are not compatible; bucket bit patterns cannot be combined")

    @classmethod
    def merge(cls, *summaries: "WindowedSummary") -> "WindowedSummary":
        """Merge two or more summaries into a new shared-window summary.

        Buckets at the same event-time start are combined with a Bloom Filter
        union, so elements confirmed by either side remain detectable
        (fast-partition bucket rotation in one input cannot erase them as long
        as the bucket is retained by another input). Per-partition watermarks
        are combined conservatively: the minimum watermark wins and a
        partition is only considered sealed when every input sealed it.

        Raises:
            WindowSummaryError: When fewer than two summaries are given or
                their bucket widths, capacities, false positive rates or hash
                functions are incompatible.
        """
        if len(summaries) < 2:
            raise WindowSummaryError("merge requires at least two WindowedSummary instances")
        first = summaries[0]
        for other in summaries[1:]:
            first._validate_mergeable(other)

        merged = cls(
            bucket_width=first._width,
            est_elements_per_bucket=first._est_elements,
            false_positive_rate=first._fpr,
            allowed_lateness=first._lateness,
            late_policy=first._policy,
            hash_function=first._hash_func,
        )

        raw_gaps: list[tuple[int, int]] = []
        for summary in summaries:
            merged._observed += summary._observed
            merged._inserts += summary._counts_total()
            merged._late_accepted += summary._late_accepted
            merged._late_dropped += summary._late_dropped
            merged._late_rejected += summary._late_rejected
            merged._sealed_rejected += summary._sealed_rejected
            merged._side_output.extend(summary._side_output)
            for start, bucket in summary._buckets.items():
                merged._merge_bucket(start, bucket, summary._counts[start])
            raw_gaps.extend(summary._gaps)
            merged._merge_partition_state(summary)

        for gap_start, gap_end in merged._union_ranges(raw_gaps):
            for seg_start, seg_end in merged._subtract_retained_spans(gap_start, gap_end):
                merged._record_gap(seg_start, seg_end)

        return merged

    def _counts_total(self) -> int:
        return sum(self._counts.values())

    def _merge_bucket(self, start: int, bucket: BloomFilter, count: int) -> None:
        if start not in self._buckets:
            self._buckets[start] = bucket
            self._counts[start] = count
            return
        try:
            self._buckets[start] = self._buckets[start].union(bucket)
        except SimilarityError as exc:
            raise WindowSummaryError(f"cannot union buckets starting at {start}: {exc.message}") from exc
        self._counts[start] += count

    def _merge_partition_state(self, other: "WindowedSummary") -> None:
        for partition in other._partitions:
            self._partitions.add(partition)
            other_wm = other._effective_watermark(partition)
            other_sealed = other._sealed.get(partition, False)
            if partition not in self._sealed:
                # First time this partition is seen: adopt its state directly.
                self._sealed[partition] = other_sealed
                if other_sealed:
                    self._seal_watermarks[partition] = other_wm
                candidate = other._watermarks.get(partition)
                if candidate is None:
                    candidate = other_wm
                if candidate is not None:
                    self._watermarks[partition] = candidate
                continue
            self_sealed = self._sealed[partition]
            combined_sealed = self_sealed and other_sealed
            if combined_sealed:
                self._seal_watermarks[partition] = min(self._seal_watermarks[partition], other_wm)
            else:
                self._seal_watermarks.pop(partition, None)
            self._sealed[partition] = combined_sealed
            if other_wm is not None:
                current = self._watermarks.get(partition)
                if current is None or other_wm < current:
                    self._watermarks[partition] = other_wm
        # Merging never garbage collects: a bucket retained by any input must
        # stay answerable in the union. Callers rotate explicitly afterwards.

    # ------------------------------------------------------------------ #
    # Granularity changes
    # ------------------------------------------------------------------ #
    def regranulate(self, new_bucket_width: int) -> "WindowedSummary":
        """Return a copy whose buckets use ``new_bucket_width``.

        Old buckets are reused only when their boundaries align with the new
        grid (the widths must be integer multiples and every retained bucket
        start must fall on a new boundary). Aligned buckets are combined with a
        Bloom Filter union; nothing is silently dropped. Rotated-out gap
        metadata is carried over unchanged.

        Raises:
            WindowSummaryError: When the new width is invalid, the grids do not
                align, or retained buckets straddle new boundaries.
        """
        if not isinstance(new_bucket_width, int) or new_bucket_width <= 0:
            raise WindowSummaryError("new_bucket_width must be a positive integer")
        if new_bucket_width == self._width:
            raise WindowSummaryError("new_bucket_width equals the current bucket width")
        if new_bucket_width % self._width != 0 and self._width % new_bucket_width != 0:
            raise WindowSummaryError(
                f"bucket widths {self._width} and {new_bucket_width} are not integer multiples; "
                "their boundaries cannot align"
            )
        for start in self._buckets:
            if start % new_bucket_width != 0:
                raise WindowSummaryError(
                    f"retained bucket starting at {start} does not align with the new width "
                    f"{new_bucket_width}; rotate it before regranulating"
                )

        regranulated = WindowedSummary(
            bucket_width=new_bucket_width,
            est_elements_per_bucket=self._est_elements,
            false_positive_rate=self._fpr,
            allowed_lateness=self._lateness,
            late_policy=self._policy,
            hash_function=self._hash_func,
        )
        groups: dict[int, list[tuple[BloomFilter, int]]] = {}
        for start, bucket in self._buckets.items():
            new_start = (start // new_bucket_width) * new_bucket_width
            groups.setdefault(new_start, []).append((bucket, self._counts[start]))
        for start, members in sorted(groups.items()):
            combined = members[0][0]
            total = members[0][1]
            for bucket, count in members[1:]:
                try:
                    combined = combined.union(bucket)
                except SimilarityError as exc:
                    raise WindowSummaryError(f"cannot union buckets at {start}: {exc.message}") from exc
                total += count
            regranulated._buckets[start] = combined
            regranulated._counts[start] = total

        regranulated._partitions = set(self._partitions)
        regranulated._watermarks = dict(self._watermarks)
        regranulated._sealed = dict(self._sealed)
        regranulated._seal_watermarks = dict(self._seal_watermarks)
        regranulated._gaps = list(self._gaps)
        regranulated._observed = self._observed
        regranulated._inserts = self._inserts
        regranulated._late_accepted = self._late_accepted
        regranulated._late_dropped = self._late_dropped
        regranulated._late_rejected = self._late_rejected
        regranulated._sealed_rejected = self._sealed_rejected
        regranulated._side_output = list(self._side_output)
        return regranulated

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _new_bucket(self) -> BloomFilter:
        return BloomFilter(
            est_elements=self._est_elements,
            false_positive_rate=self._fpr,
            hash_function=self._hash_func,
        )
