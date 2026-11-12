# Needed for pre-3.10 versions
from __future__ import annotations

__all__ = [
    "RuleChange",
    "BatchResult",
    "apply",
    "plan",
    "replay",
]

from collections.abc import Sequence as AbstractSequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, List, Sequence, Tuple

from dpath import segments
from dpath.exceptions import InvalidGlob
from dpath.types import Filter, PathSegment


@dataclass
class RuleChange:
    """
    A single record resolved (and, for an applied batch, written) for one
    distinct value at one target of a manifest.

    rules       : indices of every manifest rule that resolved to this target
                  asking for new_value (empty for records produced by
                  replay(), whose hits carry no rule identity).
    sources     : the globs (or literal paths) authored by those rules, in
                  rule order.
    target      : the concrete path tuple inside the original document. String
                  dictionary keys stay strings and list indices become plain
                  ints, so a '0' key can never be confused with index 0 and
                  the tuple stays hashable.
    path        : target rendered back into a separator joined path string.
    old_value   : value found at target in the original document.
    new_value   : value the rules want written at target.
    conflicts   : sibling RuleChange records that resolved to the same target
                  with a different new_value. Empty when there is no conflict.
    """

    rules: List[int]
    sources: List[Any]
    target: Tuple[PathSegment, ...]
    path: str
    old_value: Any
    new_value: Any
    conflicts: List["RuleChange"] = field(default_factory=list)


@dataclass
class BatchResult:
    """
    Outcome of a batch write.

    changed   : number of targets actually written.
    applied   : one RuleChange per written target, in document walk order.
    conflicts : conflicting records that were isolated (never written). A
                conflict contributes one RuleChange per disagreeing value,
                each cross linked through RuleChange.conflicts and carrying
                every rule (and authored glob) asking for that value.
    """

    changed: int
    applied: List[RuleChange]
    conflicts: List[RuleChange]


def _split(path: Any, separator: str) -> Tuple[PathSegment, ...]:
    """
    Split a string glob on the separator (a leading separator is stripped).
    Sequence paths pass through unchanged so separators inside a key name
    are preserved.
    """
    if isinstance(path, str):
        return tuple(path.lstrip(separator).split(separator))
    return tuple(path)


def _compile(manifest: Sequence, separator: str):
    """
    Parse every rule of the manifest against nothing but its own text, before
    any write is attempted. Each compiled rule is a tuple of:

        (rule index, authored glob, split glob segments, value, literal)

    A rule is either a (glob, value) pair or a mapping carrying the keys
    'glob' and 'value', optionally 'literal' (when true, glob characters are
    treated as ordinary key characters and '**' is a key literally named
    '**').
    """
    compiled = []
    for rule_index, entry in enumerate(manifest):
        literal = False
        if isinstance(entry, dict):
            try:
                glob = entry["glob"]
                value = entry["value"]
            except KeyError as error:
                raise TypeError(
                    "Manifest mapping rules require both 'glob' and 'value' "
                    f"keys, got: {entry!r}"
                ) from error
            literal = bool(entry.get("literal", False))
        else:
            try:
                glob, value = entry
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "Manifest rules must be (glob, value) pairs or mappings "
                    f"with 'glob' and 'value' keys, got: {entry!r}"
                ) from error

        glob_segments = _split(glob, separator)
        if not literal and glob_segments.count("**") > 1:
            raise InvalidGlob(
                "Invalid glob. Only one '**' is permitted per glob: "
                f"{glob!r}"
            )
        compiled.append((rule_index, glob, glob_segments, value, literal))
    return compiled


def _has_magic(pattern: PathSegment) -> bool:
    magic = b"*?[" if isinstance(pattern, bytes) else "*?["
    return any(character in pattern for character in magic)


def _segment_matches(segment: PathSegment, pattern: PathSegment) -> bool:
    """
    Type aware counterpart of segments.match() for a single segment.

    Integer segments are list indices. They match integer patterns
    numerically (negative patterns honor ListIndex semantics). A decimal
    string pattern also reaches a list index, matching ordinary dpath path
    conventions; string patterns carrying glob magic are fnmatch tested
    against the index's string form.

    String (and bytes) dictionary keys are only ever fnmatch tested against
    string or bytes patterns and are never coerced to integers. Walk targets
    keep their segment types regardless, so the string key '0' and the list
    index 0 always resolve to distinct, type tagged targets even when their
    rendered paths look identical. Use an integer path segment, or a literal
    rule, to address exactly one of the two kinds.
    """
    if isinstance(segment, int):
        if isinstance(pattern, int):
            return segment == pattern
        if isinstance(pattern, str):
            if not _has_magic(pattern):
                # Signed and unsigned numeric strings address list indices
                # (ListIndex itself understands negative indices); anything
                # else cannot name a list position.
                try:
                    return segment == int(pattern)
                except ValueError:
                    return False
            try:
                return fnmatchcase(str(segment), pattern)
            except Exception:
                return False
        if isinstance(pattern, bytes) and _has_magic(pattern):
            try:
                pattern = pattern.decode()
            except UnicodeDecodeError:
                return False
            try:
                return fnmatchcase(str(segment), pattern)
            except Exception:
                return False
        return False

    if isinstance(pattern, int):
        return False
    if isinstance(segment, bytes) and isinstance(pattern, str):
        pattern = pattern.encode()
    elif isinstance(segment, str) and isinstance(pattern, bytes):
        try:
            pattern = pattern.decode()
        except UnicodeDecodeError:
            return False
    try:
        return fnmatchcase(segment, pattern)
    except Exception:
        return False


def _literal_matches(segment: PathSegment, pattern: PathSegment) -> bool:
    """
    Exact equality matching for literal rules. Matching is strict about the
    kind of segment: a list index is only addressed by an integer pattern and
    a dictionary key only by an equal string or bytes pattern. The decimal
    string '0' and the integer 0 are never the same target.
    """
    if isinstance(segment, int):
        return isinstance(pattern, int) and segment == pattern
    if isinstance(pattern, int):
        return False
    return segment == pattern


def _match_typed(path: Sequence[PathSegment], glob: Sequence[PathSegment], literal: bool) -> bool:
    """
    Match a concrete walk path against a rule glob, preserving the segment
    types carried by the walk. A '**' glob segment expands to zero or more
    path segments of any type; in literal mode '**' is an ordinary key.
    """
    matcher = _literal_matches if literal else _segment_matches

    if literal or "**" not in glob:
        if len(path) != len(glob):
            return False
        return all(matcher(segment, pattern) for segment, pattern in zip(path, glob))

    star_star = glob.index("**")
    prefix = glob[:star_star]
    suffix = glob[star_star + 1:]

    if len(path) < len(prefix) + len(suffix):
        return False

    head = path[:star_star]
    tail = path[len(path) - len(suffix):] if suffix else ()

    if not all(matcher(segment, pattern) for segment, pattern in zip(head, prefix)):
        return False
    if not all(matcher(segment, pattern) for segment, pattern in zip(tail, suffix)):
        return False
    return True


def _values_differ(left: Any, right: Any) -> bool:
    try:
        return left != right
    except Exception:
        return True


def _normalize_target(target: Sequence[PathSegment]) -> Tuple[PathSegment, ...]:
    """
    Normalize a concrete walk target into a hashable tuple.

    ListIndex (an int subclass) disables hashing because it overrides
    __eq__, so index segments are normalized to plain ints. String and bytes
    dictionary keys keep both their value and their type, hence the string
    key '0' and the list index 0 still produce different tuples even though
    their rendered paths are identical.
    """
    return tuple(int(segment) if isinstance(segment, int) else segment for segment in target)


def _resolve(obj, rules, afilter: Filter | None):
    """
    Walk the original document once and resolve every rule against that
    immutable snapshot. No mutation happens here, so list positions referenced
    by later rules are exactly the positions the original document had.

    Returns (original values by target, rule hits by target, target walk
    order).
    """
    snapshot = tuple(segments.walk(obj))
    original = {}
    hits = {}

    for target, value in snapshot:
        original[_normalize_target(target)] = value

    for rule_index, source, glob_segments, value, literal in rules:
        for target, current in snapshot:
            if not _match_typed(target, glob_segments, literal):
                continue
            if afilter is not None and not (segments.leaf(current) and afilter(current)):
                continue
            key = _normalize_target(target)
            hits.setdefault(key, []).append((rule_index, source, value))

    # Report targets in the original document's walk order, independent of
    # which rule reached each one first.
    order = []
    for target, _ in snapshot:
        key = _normalize_target(target)
        if key in hits:
            order.append(key)

    return original, hits, order


def _join(target: Sequence[PathSegment], separator: str) -> str:
    rendered = []
    for segment in target:
        if isinstance(segment, bytes):
            rendered.append(segment.decode("utf-8", "surrogateescape"))
        else:
            rendered.append(segments.int_str(segment))
    return separator.join(rendered)


def _build_changes(target, target_hits, old_value, separator):
    """
    Split the rules hitting one target into groups of distinct new values.
    Equal values collapse onto one RuleChange carrying every contributing
    rule and its authored source glob; disagreeing values become isolated
    conflicts cross linked through RuleChange.conflicts.
    """
    grouped = []
    for rule_index, source, value in target_hits:
        for group in grouped:
            if not _values_differ(value, group[2]):
                group[0].append(rule_index)
                group[1].append(source)
                break
        else:
            grouped.append(([rule_index], [source], value))

    path = _join(target, separator)
    changes = [
        RuleChange(rules, sources, target, path, old_value, value)
        for rules, sources, value in grouped
    ]

    if len(changes) > 1:
        for change in changes:
            change.conflicts = [other for other in changes if other is not change]
    return changes


def plan(obj, manifest: Sequence, separator: str = "/", afilter: Filter | None = None) -> List[RuleChange]:
    """
    Dry run a manifest: parse every rule and resolve it against the original
    document without writing anything.

    Returns one RuleChange per (target, distinct value) pair in document walk
    order. Disagreeing rules come back as conflict records cross linked through
    RuleChange.conflicts; everything else is a preview of what apply() would
    write.
    """
    rules = _compile(manifest, separator)
    original, hits, order = _resolve(obj, rules, afilter)

    preview = []
    for target in order:
        preview.extend(_build_changes(target, hits[target], original[target], separator))
    return preview


def apply(obj, manifest: Sequence, separator: str = "/", afilter: Filter | None = None) -> BatchResult:
    """
    Apply a manifest of column rules to obj in one batch.

    Every rule is parsed and resolved against the original document first;
    only afterwards are values written, and writes never insert or delete
    containers, so list moves cannot shift the rows later rules pointed at.

    Targets reached by several rules with the same value are written once.
    Targets reached with different values are left untouched and reported per
    disagreeing rule (each record keeps its own authored source glob and
    original target path) instead of aborting the whole batch.
    """
    rules = _compile(manifest, separator)
    original, hits, order = _resolve(obj, rules, afilter)

    applied = []
    conflicts = []

    for target in order:
        changes = _build_changes(target, hits[target], original[target], separator)
        if changes[0].conflicts:
            conflicts.extend(changes)
            continue

        change = changes[0]
        # A previous write could have replaced a parent container. Such a
        # target no longer denotes the node resolved from the snapshot, so it
        # is skipped rather than silently recreated.
        if not segments.has(obj, target):
            continue
        segments.set(obj, target, change.new_value, creator=None)
        applied.append(change)

    return BatchResult(len(applied), applied, conflicts)


def _resolve_path_types(obj, target: Sequence[PathSegment]) -> Tuple[PathSegment, ...]:
    """
    Coerce the string segments of a replayed path into the same typed form the
    document walk would produce: decimal string segments addressing a sequence
    become plain ints, while mapping keys keep their string form. Segments that
    cannot be reached (the path is missing) keep their original string form.
    """
    resolved = []
    current = obj
    for segment in target:
        if (
            isinstance(segment, str)
            and isinstance(current, AbstractSequence)
            and not isinstance(current, (str, bytes))
            and segment.lstrip("-").isdecimal()
        ):
            index = int(segment)
            # Normalize negative indices to the positive form the document
            # walk reports, so replayed targets match apply() targets.
            if index < 0:
                index += len(current)
            chosen = index
        else:
            chosen = segment

        resolved.append(chosen)
        try:
            current = current[chosen]
        except (KeyError, IndexError, TypeError):
            current = None
    return tuple(resolved)


def replay(obj, hits, separator: str = "/") -> BatchResult:
    """
    Replay (path, value) pairs, such as the tuples yielded by dpath.search(),
    back onto a document of the same shape.

    The same path arriving with different values is handled exactly like a
    manifest conflict: the disagreeing records are reported and isolated and
    nothing is written at that path. Reported targets carry the document's
    actual segment types (list indices as ints, dictionary keys as strings),
    so they reconcile directly with the changes reported by apply().
    """
    values_by_target = {}
    order = []

    for path, value in hits:
        target = _resolve_path_types(obj, _split(path, separator))
        if target not in values_by_target:
            values_by_target[target] = []
            order.append(target)
        values_by_target[target].append(value)

    applied = []
    conflicts = []

    for target in order:
        if not segments.has(obj, target):
            continue

        old_value = segments.get(obj, target)
        distinct = []
        for value in values_by_target[target]:
            if all(_values_differ(value, kept) for kept in distinct):
                distinct.append(value)

        path = _join(target, separator)
        if len(distinct) > 1:
            changes = [
                RuleChange([], [], target, path, old_value, value)
                for value in distinct
            ]
            for change in changes:
                change.conflicts = [other for other in changes if other is not change]
            conflicts.extend(changes)
            continue

        value = distinct[0]
        segments.set(obj, target, value, creator=None)
        applied.append(RuleChange([], [], target, path, old_value, value))

    return BatchResult(len(applied), applied, conflicts)
