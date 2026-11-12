"""
Whole-row planning and transactional materialization of dpath targets.

A flat row maps several columns onto nested destinations and the columns
may share parent nodes that do not exist yet. The shape of every shared
node is therefore planned across *all* columns before the target
document is touched:

* :class:`Text` forces a dictionary key (the string ``"0"`` stays text).
* :class:`Index` forces a sequence subscript (``0`` is an array index).
* :class:`Auto` leaves both interpretations open until another column,
  or the live document, decides the matter.

A sequence subscript may be declared :class:`Sparse` (holes are allowed)
or :class:`Dense` (the resulting array must be contiguous).

The row is materialized only when exactly one feasible shape remains.
When several shapes remain feasible the distinct target structures are
reported on :class:`~dpath.exceptions.AmbiguousShapeError`; when none
remain feasible the columns that cannot be satisfied together are
reported on :class:`~dpath.exceptions.InfeasibleShapeError`. Nothing is
written in either failure case. The actual write is transactional: every
adopted node is created at most once through the creator, and any failure
restores the document to its exact prior state.
"""

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from copy import deepcopy
from enum import Enum
from typing import Any, Tuple

from dpath import segments
from dpath.exceptions import AmbiguousShapeError, InfeasibleShapeError, PathNotFound

_DENSITY_UNSET = object()


def _loc_key(key):
    """Normalize a segment key for planner-internal location identity.

    Integers become strings so an explicit ``Index(0)`` and an
    ``Auto("0")`` constrain the same edge; bytes and other keys pass
    through unchanged.
    """
    if isinstance(key, int):
        return str(key)
    return key


class NodeKind(str, Enum):
    MAPPING = "mapping"
    SEQUENCE = "sequence"


class Density(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class Segment:
    """A path segment carrying an explicit declaration of its role.

    The declaration constrains the kind of container that holds the
    segment: a mapping key or a sequence subscript.
    """

    __slots__ = ("key", "roles", "density")

    def __init__(self, key, roles, density=_DENSITY_UNSET):
        self.key = key
        self.roles = frozenset(roles)
        self.density = density

    def __repr__(self):
        extras = ""
        if self.density is not _DENSITY_UNSET:
            extras = f", {self.density.value}"
        return f"{type(self).__name__}({self.key!r}{extras})"


class Text(Segment):
    """Declare ``key`` as a dictionary key (e.g. the string ``"0"``)."""

    def __init__(self, key):
        super().__init__(key, (NodeKind.MAPPING,))


class Index(Segment):
    """Declare ``key`` as a sequence subscript (e.g. the integer ``0``)."""

    def __init__(self, key, density=_DENSITY_UNSET):
        super().__init__(int(key), (NodeKind.SEQUENCE,), density)


class Sparse(Index):
    """Sequence subscript that allows holes in the resulting array."""

    def __init__(self, key):
        super().__init__(key, Density.SPARSE)


class Dense(Index):
    """Sequence subscript requiring a contiguous, hole-free array."""

    def __init__(self, key):
        super().__init__(key, Density.DENSE)


class Auto(Segment):
    """Ambiguous segment: dictionary key and array index stay both open.

    Only decimal keys are ambiguous (``"0"`` could be either reading);
    any other key is unambiguously a mapping key.
    """

    def __init__(self, key):
        if isinstance(key, str) and key.isdecimal():
            super().__init__(key, (NodeKind.MAPPING, NodeKind.SEQUENCE))
            self.alt_key = int(key)
        elif isinstance(key, int):
            text_key = str(key)
            super().__init__(text_key, (NodeKind.MAPPING, NodeKind.SEQUENCE))
            self.alt_key = int(key)
        else:
            # Bytes and non-decimal strings are unambiguously mapping keys.
            super().__init__(key, (NodeKind.MAPPING,))
            self.alt_key = None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class _Edge:
    """Accumulated constraint on the container at one planned location."""

    __slots__ = ("roles", "sources", "indices", "index_sources",
                 "density", "density_source", "child_keys")

    def __init__(self):
        self.roles = {NodeKind.MAPPING, NodeKind.SEQUENCE}
        self.sources = []
        self.indices = set()
        self.index_sources = set()
        self.density = _DENSITY_UNSET
        self.density_source = None
        # Raw child key per passing column, needed to reject non-decimal
        # keys when the container ends up being a sequence.
        self.child_keys = []


class _Planner:
    def __init__(self, root):
        self.root = root
        self.edges = {}
        self.columns = []
        # Locations that are the terminus of some column (leaf assignment).
        self.termini = {}

    # -- registration ------------------------------------------------------

    def add(self, source, path, value):
        declared = tuple(coerce(path))
        if not declared:
            raise ValueError(f"column {source!r} has an empty path")
        self.columns.append((source, declared, value))

        for depth, segment in enumerate(declared):
            parent_loc = tuple(_loc_key(seg.key) for seg in declared[:depth])
            edge = self.edges.setdefault(parent_loc, _Edge())
            self._constrain(edge, source, segment)
            edge.child_keys.append((source, segment.key))

        terminus = tuple(_loc_key(seg.key) for seg in declared)
        if terminus in self.edges:
            edge = self.edges[terminus]
            raise InfeasibleShapeError(
                f"column {source!r} ends at {terminus!r} but column "
                f"{edge.sources[0]!r} continues through that location, so it "
                "cannot be both a leaf and a container",
                sources=(source, edge.sources[0]),
            )
        for depth in range(1, len(terminus)):
            prefix = terminus[:depth]
            if prefix in self.termini:
                other = self.termini[prefix]
                raise InfeasibleShapeError(
                    f"column {source!r} continues through {prefix!r} but "
                    f"column {other!r} ends there, so it cannot be both a "
                    "container and a leaf",
                    sources=(source, other),
                )
        self.termini.setdefault(terminus, source)
        return declared

    def _constrain(self, edge, source, segment):
        if source not in edge.sources:
            edge.sources.append(source)

        surviving = edge.roles & set(segment.roles)
        if not surviving:
            raise InfeasibleShapeError(
                f"column {source!r} requires {segment!r} but the container at "
                f"that location was already declared as "
                f"{next(iter(edge.roles)).value} by column "
                f"{edge.sources[0]!r}",
                sources=(source, edge.sources[0]),
            )
        edge.roles = surviving

        if NodeKind.SEQUENCE in segment.roles:
            edge.indices.add(int(segment.key))
            edge.index_sources.add(source)
        if segment.density is not _DENSITY_UNSET:
            if edge.density is not _DENSITY_UNSET and edge.density != segment.density:
                raise InfeasibleShapeError(
                    f"column {source!r} declares the array at that location "
                    f"{segment.density.value} but column {edge.density_source!r} "
                    f"already declared it {edge.density.value}",
                    sources=(source, edge.density_source),
                )
            edge.density = segment.density
            edge.density_source = source

    # -- live document -----------------------------------------------------

    def _live(self, loc):
        if not loc:
            return self.root, True
        try:
            return segments.get(self.root, loc), True
        except (KeyError, IndexError, TypeError):
            return None, False
        except PathNotFound:
            # An intermediate node on the way exists but is a leaf, so the
            # requested descent cannot be planned at all.
            edge = self.edges[loc] if loc in self.edges else None
            sources = tuple(edge.sources) if edge is not None else ()
            raise InfeasibleShapeError(
                f"the path through {loc!r} is blocked by a live leaf value",
                sources=sources,
            )

    def _pin_to_live_document(self):
        for loc, edge in self.edges.items():
            node, exists = self._live(loc)
            if not exists:
                continue
            if isinstance(node, Mapping):
                live_role = NodeKind.MAPPING
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                live_role = NodeKind.SEQUENCE
            else:
                raise InfeasibleShapeError(
                    f"column {edge.sources[0]!r} continues through location "
                    f"{loc!r} but the live value there is a leaf: {node!r}",
                    sources=tuple(edge.sources),
                )
            if live_role not in edge.roles:
                raise InfeasibleShapeError(
                    f"the live container at {loc!r} is a "
                    f"{type(node).__name__} but column {edge.sources[0]!r} "
                    f"declares it as {next(iter(edge.roles)).value}",
                    sources=tuple(edge.sources),
                )
            edge.roles = {live_role}

    # -- shape enumeration -------------------------------------------------

    def resolve(self):
        self._pin_to_live_document()
        self._check_density()

        ambiguous = [loc for loc, edge in self.edges.items()
                     if edge.roles == {NodeKind.MAPPING, NodeKind.SEQUENCE}]

        assignments = []
        seen = set()
        for bits in range(1 << len(ambiguous)):
            assignment = {
                loc: (NodeKind.SEQUENCE if (bits >> i) & 1 else NodeKind.MAPPING)
                for i, loc in enumerate(ambiguous)
            }
            if not self._assignment_valid(assignment):
                continue
            fingerprint = tuple(sorted(
                (loc, kind.value) for loc, kind in assignment.items()
            ))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            assignments.append(assignment)
        return ambiguous, assignments

    def _role_at(self, loc, assignment):
        edge = self.edges.get(loc)
        if edge is not None and len(edge.roles) == 1:
            return next(iter(edge.roles))
        return assignment.get(loc, NodeKind.MAPPING)

    def _assignment_valid(self, assignment):
        for loc, edge in self.edges.items():
            role = self._role_at(loc, assignment)
            if role is NodeKind.SEQUENCE:
                for source, key in edge.child_keys:
                    if not (isinstance(key, int) or
                            (isinstance(key, str) and key.isdecimal())):
                        return False
        return True

    def _check_density(self):
        for loc, edge in self.edges.items():
            if edge.density is not Density.DENSE:
                continue
            node, exists = self._live(loc)
            existing = set(range(len(node))) if exists else set()
            required = existing | edge.indices
            if required and required != set(range(max(required) + 1)):
                holes = sorted(set(range(max(required) + 1)) - required)
                raise InfeasibleShapeError(
                    f"column {edge.density_source!r} requires a dense array at "
                    f"{loc!r} but indices {holes} are missing",
                    sources=tuple(sorted(edge.index_sources)),
                )

    # -- rendering & materialization --------------------------------------

    def concrete_paths(self, assignment):
        prepared = []
        for source, declared, value in self.columns:
            concrete = []
            for depth, segment in enumerate(declared):
                parent_loc = tuple(_loc_key(seg.key) for seg in declared[:depth])
                role = self._role_at(parent_loc, assignment)
                if role is NodeKind.SEQUENCE:
                    concrete.append(int(segment.key))
                else:
                    concrete.append(segment.key)
            prepared.append((source, tuple(concrete), value))
        return prepared

    def render_structure(self, assignment):
        if self._role_at((), assignment) is NodeKind.SEQUENCE:
            preview = []
        else:
            preview = {}
        for _, concrete, value in self.concrete_paths(assignment):
            cursor = preview
            loc = ()
            for depth, key in enumerate(concrete):
                last = depth == len(concrete) - 1
                child_loc = loc + (_loc_key(key),)
                if self._role_at(loc, assignment) is NodeKind.SEQUENCE:
                    if len(cursor) <= key:
                        cursor.extend([None] * (key + 1 - len(cursor)))
                    if last:
                        cursor[key] = value
                        break
                    if cursor[key] is None:
                        cursor[key] = [] if self._role_at(
                            child_loc, assignment
                        ) is NodeKind.SEQUENCE else {}
                    cursor = cursor[key]
                else:
                    if last:
                        cursor[key] = value
                        break
                    if key not in cursor:
                        cursor[key] = [] if self._role_at(
                            child_loc, assignment
                        ) is NodeKind.SEQUENCE else {}
                    cursor = cursor[key]
                loc = child_loc
        return preview


# ---------------------------------------------------------------------------
# Helpers / public API
# ---------------------------------------------------------------------------

def coerce(path):
    """Turn shorthand path elements into explicit :class:`Segment` objects.

    Existing segments pass through. Integers become :class:`Index`, decimal
    strings become :class:`Auto` (the historical dpath ambiguity), every
    other string becomes :class:`Text`.
    """
    for element in path:
        if isinstance(element, Segment):
            yield element
        elif isinstance(element, int):
            yield Index(element)
        elif isinstance(element, str) and element.isdecimal():
            yield Auto(element)
        else:
            yield Text(element)


def plan(target: MutableMapping, columns: Sequence[Tuple[Any, Any, Any]]):
    """Return the distinct target structures feasible for one row.

    Each column is a ``(source, path, value)`` triple. ``path`` is a
    sequence of :class:`Segment` objects (or shorthand strings/ints).
    The returned structures are preview trees, one per feasible shape;
    an empty list means no shape can satisfy all columns.
    """
    planner = _Planner(target)
    for source, path, value in columns:
        planner.add(source, path, value)
    _, assignments = planner.resolve()
    return [planner.render_structure(assignment) for assignment in assignments]


def apply_row(
        target: MutableMapping,
        columns: Sequence[Tuple[Any, Any, Any]],
        creator=None,
) -> MutableMapping:
    """Materialize a whole flat row into nested detail on ``target``.

    The row is applied only when the column declarations admit exactly
    one feasible shape. Ambiguity raises
    :class:`~dpath.exceptions.AmbiguousShapeError` carrying the distinct
    feasible target structures; contradiction raises
    :class:`~dpath.exceptions.InfeasibleShapeError` carrying the
    conflicting columns. On any failure ``target`` is restored exactly.

    ``creator`` follows the same contract as in :func:`dpath.new` and is
    invoked at most once for each adopted intermediate node.
    """
    planner = _Planner(target)
    for source, path, value in columns:
        planner.add(source, path, value)

    _, assignments = planner.resolve()
    if not assignments:
        raise InfeasibleShapeError(
            "the columns cannot be satisfied by any shared target shape",
            sources=tuple(source for source, _, _ in planner.columns),
        )
    if len(assignments) > 1:
        structures = [planner.render_structure(a) for a in assignments]
        raise AmbiguousShapeError(
            f"{len(assignments)} distinct target structures remain feasible; "
            "declare the ambiguous segments with Text or Index",
            structures=structures,
            sources=tuple(source for source, _, _ in planner.columns),
        )

    snapshot = deepcopy(target)
    try:
        roles = {loc: planner._role_at(loc, assignments[0]) for loc in planner.edges}
        created = set()
        for _, concrete, value in planner.concrete_paths(assignments[0]):
            _materialize(target, concrete, value, creator, created, roles)
    except Exception:
        target.clear()
        target.update(snapshot)
        raise
    return target


def _default_create(current, key, child_location, roles):
    """Create ``current[key]`` using the role planned for ``child_location``."""
    if roles.get(child_location) is NodeKind.SEQUENCE:
        child = []
    else:
        child = {}
    if isinstance(current, MutableSequence):
        segments.extend(current, key)
    current[key] = child


def _materialize(target, concrete_path, value, creator, created, roles):
    current = target
    for depth, key in enumerate(concrete_path[:-1]):
        location = tuple(_loc_key(k) for k in concrete_path[:depth + 1])
        try:
            current[key]
        except (KeyError, IndexError, TypeError):
            if location not in created:
                if creator is None:
                    _default_create(current, key, location, roles)
                else:
                    creator(current, concrete_path, depth, ())
                created.add(location)
        current = current[key]

    last_key = concrete_path[-1]
    if isinstance(last_key, int) and isinstance(current, MutableSequence):
        segments.extend(current, last_key)
    current[last_key] = value
