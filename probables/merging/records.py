"""Record types describing partitioned sketches and merge outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field

from probables.hashes import HashFuncT
from probables.merging.intervals import CoverageInterval, normalize_intervals

# Registry of legacy hash conventions. A convention is only usable when the
# caller explicitly binds its name to a concrete hashing function; nothing
# here ever assumes an unnamed historical default.
LEGACY_HASH_CONVENTIONS: dict[str, HashFuncT] = {}


def register_legacy_hash_convention(name: str, hash_function: HashFuncT) -> None:
    """Bind a legacy hash convention name to a concrete hashing function.

    Args:
        name (str): Stable identifier of the historical convention.
        hash_function (function): Hashing strategy with signature
            ``hf(key, number)`` matching the sketch's hashing interface.
    Note:
        Legacy conventions are never applied implicitly; callers must still
        name the convention on every partition that needs it.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Legacy hash convention name must be a non-empty string")
    if not callable(hash_function):
        raise TypeError("hash_function must be callable")
    LEGACY_HASH_CONVENTIONS[name] = hash_function


def resolve_hash_function(
    hash_function: HashFuncT | None,
    legacy_convention: str | None,
) -> HashFuncT | None:
    """Resolve an explicit function or a caller-bound legacy convention name."""
    if hash_function is not None and legacy_convention is not None:
        raise ValueError("Provide either hash_function or legacy_convention, not both")
    if legacy_convention is not None:
        if legacy_convention not in LEGACY_HASH_CONVENTIONS:
            raise KeyError(
                f"Unknown legacy hash convention '{legacy_convention}'; "
                "register it with register_legacy_hash_convention first"
            )
        return LEGACY_HASH_CONVENTIONS[legacy_convention]
    return hash_function


@dataclass(frozen=True)
class PartitionRecord:
    """One partition contribution handed in for merging.

    Args:
        partition_id (str): Stable identifier of the emitting partition.
        generation (int): Partition generation; higher generations supersede
            earlier ones for the same ``partition_id``.
        intervals (list[CoverageInterval]): Inclusive coverage of the
            contribution on the shared ordered domain.
        sketch (BloomFilter | CountMinSketch): The payload.
        legacy_hash_convention (str): Optional name of a caller-registered
            historical hashing convention used by the payload.
        hash_function (function): Optional explicit hashing function for the
            payload, mutually exclusive with ``legacy_hash_convention``.
    Note:
        Old files that only recorded payload dimensions are supported: leave
        both hash arguments unset and the payload keeps the function it was
        loaded with, but such a record can only join a merge whose other
        payloads are verified compatible against that same function.
    """

    partition_id: str
    generation: int
    intervals: list[CoverageInterval]
    sketch: object
    legacy_hash_convention: str | None = None
    hash_function: HashFuncT | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.partition_id, str) or not self.partition_id:
            raise ValueError("partition_id must be a non-empty string")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise TypeError("generation must be an integer")
        if not self.intervals:
            raise ValueError("at least one coverage interval is required")
        for interval in self.intervals:
            if not isinstance(interval, CoverageInterval):
                raise TypeError("intervals must contain CoverageInterval instances")
        resolve_hash_function(self.hash_function, self.legacy_hash_convention)

    @property
    def normalized_intervals(self) -> list[CoverageInterval]:
        """Sorted, disjoint coverage of the contribution."""
        return normalize_intervals(self.intervals)

    def bound_hash_function(self) -> HashFuncT | None:
        """Return the hashing function explicitly bound to this record."""
        return resolve_hash_function(self.hash_function, self.legacy_hash_convention)


@dataclass(frozen=True)
class MergeGap:
    """Description of coverage that could not be merged deterministically.

    Args:
        interval (CoverageInterval): Positions left without a sound result.
        reason (str): Why the gap exists.
        partitions (tuple[str, ...]): Partition ids involved, when relevant.
    """

    interval: CoverageInterval
    reason: str
    partitions: tuple[str, ...] = ()


@dataclass
class MergeResult:
    """Outcome of merging one or more :class:`PartitionRecord` objects.

    Args:
        sketch: The merged payload; only buckets with identical meaning feed
            into it.
        covered_intervals (tuple[CoverageInterval, ...]): Normalized coverage
            represented by the counted payloads.
        partition_intervals (dict): Counted coverage per original partition
            id; enables later result-to-result merges to attribute every bin
            back to its source partitions.
        superseded (dict[str, int]): Latest generation counted per
            partition id.
        duplicate_deliveries (tuple[tuple[str, int], ...]): ``(partition_id,
            generation)`` pairs received more than once and ignored after the
            first delivery.
        stale_deliveries (tuple[tuple[str, int], ...]): Older generations
            superseded by a newer delivery of the same partition.
        gaps (tuple[MergeGap, ...]): Coverage whose total cannot be
            determined.
        rejected (tuple[str, ...]): Human-readable notes for records whose
            payload could not be counted at all.
        notices (tuple[str, ...]): Benign merge notes, such as idempotent
            bloom overlaps.
        legacy_conventions (tuple[str, ...]): Legacy convention names bound
            into the result.
    Note:
        ``total_known`` is True only when no gaps and no fully rejected
        records remain; otherwise the aggregate total cannot be determined.
    """

    sketch: object
    covered_intervals: tuple[CoverageInterval, ...] = ()
    partition_intervals: dict[str, tuple[CoverageInterval, ...]] = field(default_factory=dict)
    superseded: dict[str, int] = field(default_factory=dict)
    duplicate_deliveries: tuple[tuple[str, int], ...] = ()
    stale_deliveries: tuple[tuple[str, int], ...] = ()
    gaps: tuple[MergeGap, ...] = ()
    rejected: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    legacy_conventions: tuple[str, ...] = ()

    @property
    def included_partitions(self) -> tuple[str, ...]:
        """Partition ids whose contributions were counted at least once."""
        return tuple(sorted(self.partition_intervals.keys()))

    @property
    def total_known(self) -> bool:
        """True when every requested contribution was counted without gaps."""
        return not self.gaps and not self.rejected

    def explain_gaps(self) -> str:
        """Render the gap and rejection notes as a human-readable summary."""
        lines: list[str] = []
        for gap in self.gaps:
            ids = f" (partitions: {', '.join(gap.partitions)})" if gap.partitions else ""
            lines.append(f"[{gap.interval.start}, {gap.interval.end}]{ids}: {gap.reason}")
        for note in self.rejected:
            lines.append(f"rejected: {note}")
        return "\n".join(lines)
