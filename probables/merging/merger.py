"""Partition-aware merging for Bloom filters and Count-Min sketches.

The engine merges per-partition payloads while tracking partition generation,
coverage intervals and optional, caller-bound legacy hash conventions.

Only payloads whose buckets carry identical meaning (same dimensions and the
same hashing convention) are combined. Re-deliveries of the same
``(partition_id, generation)`` are ignored, disjoint coverage is stitched
together, and partial overlaps that cannot be attributed to a finer source
produce explicit :class:`MergeGap` entries instead of silently inflating
frequencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from probables.blooms import BloomFilter, CountingBloomFilter
from probables.countminsketch import CountMinSketch, HeavyHitters, StreamThreshold
from probables.merging.intervals import (
    CoverageInterval,
    find_overlaps,
    normalize_intervals,
    subtract_intervals,
)
from probables.merging.records import MergeGap, MergeResult, PartitionRecord

PROBE_KEY = "merge-compatibility-probe"

REASON_OVERLAP = "partially overlapping coverage from distinct partitions without a finer-grained source"
REASON_RESIDUAL = "payload withheld because another part of the same contribution overlaps already-merged coverage"
REASON_GENERATION = "overlapping generation change cannot supersede bins already counted in the aggregate"
REASON_SUPERSEDED = "superseded by a newer generation whose coverage does not include this range"


class _SketchAdapter(ABC):
    """Strategy object capturing the merge semantics of one payload kind."""

    additive = False
    kind = "sketch"

    @staticmethod
    @abstractmethod
    def supports(payload: object) -> bool:
        """Whether this adapter owns the payload type."""

    @staticmethod
    @abstractmethod
    def empty_like(payload: object) -> object:
        """Return a zeroed payload with the same dimensions and hash."""

    @staticmethod
    @abstractmethod
    def check_compatibility(reference: object, candidate: object) -> str | None:
        """Return an error message when buckets do not share meaning."""

    @staticmethod
    @abstractmethod
    def combine(accumulator: object, payload: object) -> object:
        """Fold a compatible payload into the accumulator."""


class _BloomAdapter(_SketchAdapter):
    """Set-semantics Bloom filters: union is bitwise OR and idempotent."""

    additive = False
    kind = "bloom-filter"

    @staticmethod
    def supports(payload: object) -> bool:
        return type(payload) is BloomFilter

    @staticmethod
    def empty_like(payload: BloomFilter) -> BloomFilter:
        return type(payload)(
            est_elements=payload.estimated_elements,
            false_positive_rate=payload.false_positive_rate,
            hash_function=payload.hash_function,
        )

    @staticmethod
    def check_compatibility(reference: BloomFilter, candidate: object) -> str | None:
        if type(candidate) is not type(reference):
            return f"cannot combine {type(reference).__name__} with {type(candidate).__name__}"
        if not reference._verify_bloom_similarity(candidate):  # noqa: SLF001
            return "bloom filters differ in dimensions or hashing convention"
        return None

    @staticmethod
    def combine(accumulator: BloomFilter, payload: BloomFilter) -> BloomFilter:
        return accumulator.union(payload)


class _CountingBloomAdapter(_SketchAdapter):
    """Counting Bloom filters: union adds counters, so overlaps double count."""

    additive = True
    kind = "counting-bloom-filter"

    @staticmethod
    def supports(payload: object) -> bool:
        return type(payload) is CountingBloomFilter

    @staticmethod
    def empty_like(payload: CountingBloomFilter) -> CountingBloomFilter:
        return type(payload)(
            est_elements=payload.estimated_elements,
            false_positive_rate=payload.false_positive_rate,
            hash_function=payload.hash_function,
        )

    @staticmethod
    def check_compatibility(reference: CountingBloomFilter, candidate: object) -> str | None:
        if type(candidate) is not type(reference):
            return f"cannot combine {type(reference).__name__} with {type(candidate).__name__}"
        if not reference._verify_bloom_similarity(candidate):  # noqa: SLF001
            return "counting bloom filters differ in dimensions or hashing convention"
        return None

    @staticmethod
    def combine(accumulator: CountingBloomFilter, payload: CountingBloomFilter) -> CountingBloomFilter:
        return accumulator.union(payload)


class _CountMinAdapter(_SketchAdapter):
    """Count-Min style sketches: joining adds every counter bin."""

    additive = True
    kind = "count-min-sketch"

    @staticmethod
    def supports(payload: object) -> bool:
        return isinstance(payload, CountMinSketch) and not isinstance(
            payload, HeavyHitters | StreamThreshold
        )

    @staticmethod
    def empty_like(payload: CountMinSketch) -> CountMinSketch:
        return type(payload)(  # noqa: SLF001
            width=payload.width, depth=payload.depth, hash_function=payload._hash_function
        )

    @staticmethod
    def check_compatibility(reference: CountMinSketch, candidate: object) -> str | None:
        if type(candidate) is not type(reference):
            return f"cannot combine {type(reference).__name__} with {type(candidate).__name__}"
        if reference.width != candidate.width or reference.depth != candidate.depth:
            return "count-min sketches differ in width or depth"
        if reference.hashes(PROBE_KEY) != candidate.hashes(PROBE_KEY):
            return "count-min sketches use different hashing conventions"
        return None

    @staticmethod
    def combine(accumulator: CountMinSketch, payload: CountMinSketch) -> CountMinSketch:
        accumulator.join(payload)
        return accumulator


_ADAPTERS: tuple[type[_SketchAdapter], ...] = (_BloomAdapter, _CountingBloomAdapter, _CountMinAdapter)


def _adapter_for(payload: object) -> _SketchAdapter:
    for adapter in _ADAPTERS:
        if adapter.supports(payload):
            return adapter()
    raise TypeError(
        f"payload type {type(payload).__name__} is not supported for partition merging"
    )


def _verify_binding(record: PartitionRecord) -> str | None:
    """Check that an explicitly bound convention really matches the payload."""
    bound = record.bound_hash_function()
    if bound is None:
        return None
    depth = record.sketch.number_hashes if hasattr(record.sketch, "number_hashes") else record.sketch.depth
    if bound(PROBE_KEY, depth) != record.sketch.hashes(PROBE_KEY):
        label = record.legacy_hash_convention or "explicit hash_function"
        return f"partition '{record.partition_id}' binds '{label}' but the payload hashes do not match it"
    return None


def _fold_ledger(
    ledger: dict[str, list[CoverageInterval]],
    generations: dict[str, int],
    pid: str,
    intervals: list[CoverageInterval],
    generation: int | None,
) -> None:
    ledger.setdefault(pid, [])
    ledger[pid] = normalize_intervals([*ledger[pid], *intervals])
    if generation is not None:
        generations[pid] = max(generations.get(pid, generation), generation)


def _classify_overlap(
    intervals: list[CoverageInterval],
    covered: list[CoverageInterval],
    ledger: dict[str, list[CoverageInterval]],
    pid: str,
) -> tuple[list[CoverageInterval], bool]:
    """Return overlapping portions and whether a same-partition generation
    change is involved."""
    overlaps: list[CoverageInterval] = []
    generation_conflict = False
    owned = normalize_intervals(ledger.get(pid, []))
    for interval in intervals:
        for shared in find_overlaps(interval, covered):
            overlaps.append(shared)
            if find_overlaps(shared, owned):
                generation_conflict = True
    return normalize_intervals(overlaps), generation_conflict


def _withhold_gaps(
    intervals: list[CoverageInterval],
    overlaps: list[CoverageInterval],
    reason: str,
    partitions: tuple[str, ...],
) -> list[MergeGap]:
    gaps = [MergeGap(interval=iv, reason=reason, partitions=partitions) for iv in overlaps]
    for interval in intervals:
        for residual in subtract_intervals(interval, overlaps):
            gaps.append(MergeGap(interval=residual, reason=REASON_RESIDUAL, partitions=partitions))
    return gaps


def _stale_coverage_gaps(records: list[PartitionRecord]) -> list[MergeGap]:
    """Gap ranges of older generations not re-covered by the latest one."""
    latest_generation: dict[str, int] = {}
    for record in records:
        latest_generation[record.partition_id] = max(
            latest_generation.get(record.partition_id, record.generation), record.generation
        )

    latest_coverage: dict[str, list[CoverageInterval]] = {}
    for record in records:
        if record.generation == latest_generation[record.partition_id]:
            latest_coverage.setdefault(record.partition_id, [])
            latest_coverage[record.partition_id].extend(record.normalized_intervals)
    latest_coverage = {pid: normalize_intervals(ivs) for pid, ivs in latest_coverage.items()}

    gaps: list[MergeGap] = []
    stale_ranges: dict[str, list[CoverageInterval]] = {}
    for record in records:
        if record.generation >= latest_generation[record.partition_id]:
            continue
        stale_ranges.setdefault(record.partition_id, [])
        for interval in record.normalized_intervals:
            stale_ranges[record.partition_id].extend(
                subtract_intervals(interval, latest_coverage.get(record.partition_id, []))
            )
    for pid, ranges in sorted(stale_ranges.items()):
        for residual in normalize_intervals(ranges):
            gaps.append(
                MergeGap(interval=residual, reason=REASON_SUPERSEDED, partitions=(pid,))
            )
    return gaps


def merge_partitions(records: list[PartitionRecord] | tuple[PartitionRecord, ...]) -> MergeResult:
    """Merge per-partition sketch payloads into a single :class:`MergeResult`.

    Args:
        records (list[PartitionRecord]): Partition contributions. Each record
            carries its partition id, generation, inclusive coverage intervals
            and an optional caller-bound legacy hash convention.
    Returns:
        MergeResult: The aggregate payload plus a full accounting of which
            partitions were counted, which deliveries were duplicates or
            stale, and which coverage remains undetermined.
    Raises:
        ValueError: When no records are supplied.
        TypeError: When a payload type cannot be merged.
    Note:
        Records are processed in input order. Additive payloads (Count-Min
        sketches and counting Bloom filters) cannot be split by interval:
        when one partially overlaps already-merged coverage, the whole
        contribution is withheld and every affected range is reported as a
        gap instead of inflating frequencies.
    """
    if not records:
        raise ValueError("at least one PartitionRecord is required")

    adapter = _adapter_for(records[0].sketch)
    accumulator = adapter.empty_like(records[0].sketch)

    covered: list[CoverageInterval] = []
    ledger: dict[str, list[CoverageInterval]] = {}
    generations: dict[str, int] = {}
    conventions: set[str] = set()

    duplicate_deliveries: list[tuple[str, int]] = []
    stale_deliveries: list[tuple[str, int]] = []
    gaps: list[MergeGap] = _stale_coverage_gaps(list(records))
    rejected: list[str] = []
    notices: list[str] = []

    latest_generation: dict[str, int] = {}
    for record in records:
        latest_generation[record.partition_id] = max(
            latest_generation.get(record.partition_id, record.generation), record.generation
        )

    seen_deliveries: set[tuple[str, int]] = set()
    for record in records:
        delivery = (record.partition_id, record.generation)
        if delivery in seen_deliveries:
            duplicate_deliveries.append(delivery)
            continue
        seen_deliveries.add(delivery)

        if record.generation < latest_generation[record.partition_id]:
            stale_deliveries.append(delivery)
            continue

        if not adapter.supports(record.sketch):
            rejected.append(
                f"partition '{record.partition_id}': payload type {type(record.sketch).__name__} "
                f"does not match aggregate kind {adapter.kind}"
            )
            continue

        binding_error = _verify_binding(record)
        if binding_error is not None:
            rejected.append(binding_error)
            continue

        mismatch = adapter.check_compatibility(accumulator, record.sketch)
        if mismatch is not None:
            rejected.append(f"partition '{record.partition_id}': {mismatch}")
            continue

        intervals = record.normalized_intervals
        overlaps, generation_conflict = _classify_overlap(intervals, covered, ledger, record.partition_id)

        if overlaps and adapter.additive:
            reason = REASON_GENERATION if generation_conflict else REASON_OVERLAP
            gaps.extend(_withhold_gaps(intervals, overlaps, reason, (record.partition_id,)))
            continue
        if overlaps:
            notices.append(
                f"partition '{record.partition_id}' overlaps already-merged coverage on "
                f"{', '.join(f'[{iv.start}, {iv.end}]' for iv in overlaps)}; "
                "set union keeps membership idempotent"
            )

        accumulator = adapter.combine(accumulator, record.sketch)
        if record.legacy_hash_convention is not None:
            conventions.add(record.legacy_hash_convention)
        covered = normalize_intervals([*covered, *intervals])
        _fold_ledger(ledger, generations, record.partition_id, intervals, record.generation)

    return MergeResult(
        sketch=accumulator,
        covered_intervals=tuple(covered),
        partition_intervals={pid: tuple(ivs) for pid, ivs in sorted(ledger.items())},
        superseded=dict(sorted(generations.items())),
        duplicate_deliveries=tuple(duplicate_deliveries),
        stale_deliveries=tuple(stale_deliveries),
        gaps=tuple(gaps),
        rejected=tuple(rejected),
        notices=tuple(notices),
        legacy_conventions=tuple(sorted(conventions)),
    )


def _result_overlap(
    result: MergeResult,
    covered: list[CoverageInterval],
    ledger: dict[str, list[CoverageInterval]],
    generations: dict[str, int],
) -> tuple[list[CoverageInterval], bool]:
    overlaps: list[CoverageInterval] = []
    for interval in result.covered_intervals:
        overlaps.extend(find_overlaps(interval, covered))
    overlaps = normalize_intervals(overlaps)

    safe_redelivery = True
    for pid, ivs in result.partition_intervals.items():
        for interval in ivs:
            for shared in find_overlaps(interval, covered):
                owned = normalize_intervals(ledger.get(pid, []))
                same_owner = normalize_intervals(find_overlaps(shared, owned))
                if same_owner != [shared] or result.superseded.get(pid) != generations.get(pid):
                    safe_redelivery = False
    return overlaps, safe_redelivery


def merge_results(first: MergeResult, second: MergeResult) -> MergeResult:
    """Merge two finished :class:`MergeResult` objects.

    Args:
        first (MergeResult): Aggregate produced by :func:`merge_partitions` or
            by a previous call to this function.
        second (MergeResult): Another aggregate to fold in.
    Returns:
        MergeResult: A combined aggregate whose ledger states which original
            partitions are counted and whose gaps state which totals still
            cannot be determined.
    Note:
        A second result that only re-delivers already-counted
        ``(partition, generation)`` contributions contributes no new bins.
        For additive payloads, any other overlap cannot be split or
        superseded in place, so the second payload is withheld and the
        affected ranges are reported as gaps.
    """
    adapter = _adapter_for(first.sketch)
    accumulator = adapter.empty_like(first.sketch)
    accumulator = adapter.combine(accumulator, first.sketch)

    covered: list[CoverageInterval] = list(first.covered_intervals)
    ledger: dict[str, list[CoverageInterval]] = {
        pid: list(ivs) for pid, ivs in first.partition_intervals.items()
    }
    generations: dict[str, int] = dict(first.superseded)
    conventions: set[str] = set(first.legacy_conventions)

    gaps: list[MergeGap] = list(first.gaps) + list(second.gaps)
    rejected: list[str] = list(first.rejected) + list(second.rejected)
    notices: list[str] = list(first.notices)
    duplicate_deliveries: list[tuple[str, int]] = []

    mismatch = adapter.check_compatibility(accumulator, second.sketch)
    if mismatch is not None:
        rejected.append(f"second merge result rejected: {mismatch}")
    else:
        overlaps, safe_redelivery = _result_overlap(second, covered, ledger, generations)
        second_intervals = list(second.covered_intervals)

        if overlaps and adapter.additive and not (
            safe_redelivery and normalize_intervals(second_intervals) == overlaps
        ):
            reason = REASON_GENERATION if _generation_conflict(first, second) else REASON_OVERLAP
            partitions = tuple(sorted(second.partition_intervals))
            gaps.extend(_withhold_gaps(second_intervals, overlaps, reason, partitions))
        else:
            if overlaps:
                duplicate_deliveries.extend(
                    (pid, gen)
                    for pid, gen in sorted(second.superseded.items())
                    if generations.get(pid) == gen
                )
                notices.append(
                    "second result overlaps already-merged coverage; "
                    + ("identical bins skipped as a re-delivery" if adapter.additive else
                       "set union keeps membership idempotent")
                )
            if not (overlaps and adapter.additive):
                accumulator = adapter.combine(accumulator, second.sketch)
            conventions.update(second.legacy_conventions)
            covered = normalize_intervals([*covered, *second_intervals])
            for pid, ivs in second.partition_intervals.items():
                _fold_ledger(ledger, generations, pid, list(ivs), second.superseded.get(pid))

    return MergeResult(
        sketch=accumulator,
        covered_intervals=tuple(covered),
        partition_intervals={pid: tuple(ivs) for pid, ivs in sorted(ledger.items())},
        superseded=dict(sorted(generations.items())),
        duplicate_deliveries=tuple(dict.fromkeys(duplicate_deliveries)),
        stale_deliveries=first.stale_deliveries + second.stale_deliveries,
        gaps=tuple(gaps),
        rejected=tuple(rejected),
        notices=tuple(notices),
        legacy_conventions=tuple(sorted(conventions)),
    )


def _generation_conflict(first: MergeResult, second: MergeResult) -> bool:
    return any(
        pid in first.superseded and first.superseded[pid] != generation
        for pid, generation in second.superseded.items()
    )
