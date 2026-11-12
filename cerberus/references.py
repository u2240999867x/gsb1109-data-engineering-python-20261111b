# -*- coding: utf-8 -*-
"""
Versioned reference tables with half-open validity intervals.

A reference table holds the versions of a referenced entity (e.g. a product)
that were effective at a given time. Each version is valid on the half-open
interval ``[effective_from, effective_until)``: a point in time matches the
version when ``effective_from <= point < effective_until``. An open-ended
version (``effective_until is None``) stays valid indefinitely.

Tables are addressed by ``(source, code)`` and may be corrected retroactively:
a correction can split one former interval into two, leave a gap or make two
versions overlap. Such ambiguities are never raised silently, they are reported
by :meth:`ReferenceTable.audit` and on every lookup performed through
:func:`verify`.

A :class:`BatchGate` snapshots a table when a validation batch starts. Versions
that arrive while the batch is in flight are invisible to the gate and only
apply to the next batch. :func:`recheck` compares the verdicts of a frozen
batch gate with a later, completed table and returns only the lines whose
verdict changed, including gap or overlap ambiguities.
"""

from __future__ import absolute_import

from collections import namedtuple

from cerberus import errors
from cerberus.platform import Hashable, Mapping
from cerberus.validator import Validator


class ReferenceError(ValueError):
    """Raised when a reference entry or table is malformed."""


ReferenceEntry = namedtuple(
    'ReferenceEntry',
    (
        'source',
        'code',
        'version',
        'effective_from',
        'effective_until',
        'payload',
    ),
    defaults=(None, None),
)
"""
One version of a referenced entity, valid on the half-open interval
``[effective_from, effective_until)``. ``effective_until`` of :obj:`None`
marks an open-ended interval. ``payload`` may carry arbitrary version data.
"""


CoverageIssue = namedtuple(
    'CoverageIssue',
    ('kind', 'source', 'code', 'start', 'end', 'versions'),
)
"""
A coverage problem for one ``(source, code)``. ``kind`` is ``'gap'`` or
``'overlap'``, ``start``/``end`` delineate the affected half-open interval and
``versions`` names the involved version labels.
"""


Verdict = namedtuple(
    'Verdict',
    (
        'identifier',
        'source',
        'code',
        'point',
        'hit',
        'versions',
        'ambiguity',
        'entries',
    ),
)
"""
The outcome of verifying one line. ``hit`` is the uniquely matched version or
:obj:`None`, ``versions`` lists all matching versions (more than one means an
overlap) and ``ambiguity`` is :obj:`None`, ``'gap'`` or ``'overlap'``.
"""


RecheckItem = namedtuple(
    'RecheckItem',
    (
        'identifier',
        'source',
        'code',
        'point',
        'old_hit',
        'new_hit',
        'old_versions',
        'new_versions',
        'ambiguity',
        'note',
    ),
)
"""
A persisted line whose verdict changed between the frozen batch gate and the
completed reference table."""


def _compare(left, right, label, source, code):
    try:
        return left < right
    except TypeError:
        raise ReferenceError(
            "Cannot order {label} bounds {left!r} and {right!r} for "
            "reference ({source!r}, {code!r}).".format(
                label=label,
                left=left,
                right=right,
                source=source,
                code=code,
            )
        )


class ReferenceTable(object):
    """
    A collection of :class:`ReferenceEntry` instances indexed by
    ``(source, code)``.
    """

    def __init__(self, entries=(), name=None):
        self.name = name
        self._entries = []
        self._index = {}
        self.extend(entries)

    def add(self, entry):
        """Validate and store a reference entry."""
        if not isinstance(entry, ReferenceEntry):
            raise ReferenceError(
                'Expected a ReferenceEntry, got {!r}.'.format(type(entry))
            )
        self._validate(entry)
        self._entries.append(entry)
        self._index.setdefault((entry.source, entry.code), []).append(entry)

    def extend(self, entries):
        """Add several entries at once."""
        for entry in entries:
            self.add(entry)

    @staticmethod
    def _validate(entry):
        if entry.source is None or entry.code is None:
            raise ReferenceError(
                'source and code are required: {!r}'.format(entry)
            )
        if entry.version is None or not isinstance(entry.version, Hashable):
            raise ReferenceError(
                'version must be a hashable, non-null value: {!r}'.format(entry)
            )
        if entry.effective_from is None:
            raise ReferenceError(
                'effective_from is required: {!r}'.format(entry)
            )
        if entry.effective_until is not None and not _compare(
            entry.effective_from,
            entry.effective_until,
            'validity',
            entry.source,
            entry.code,
        ):
            raise ReferenceError(
                'effective_from must precede effective_until on the half-open '
                'interval: {!r}'.format(entry)
            )

    def entries(self, source=None, code=None):
        """Return entries, optionally filtered by source and code."""
        if source is None or code is None:
            return list(self._entries)
        return list(self._index.get((source, code), ()))

    def keys(self):
        """All known ``(source, code)`` pairs."""
        return list(self._index)

    def versions(self, source, code):
        """All versions registered for one reference, ordered by validity."""
        return [entry.version for entry in self._sorted(source, code)]

    def __len__(self):
        return len(self._entries)

    def __contains__(self, entry):
        return entry in self._entries

    def _sorted(self, source, code):
        entries = list(self._index.get((source, code), ()))
        try:
            entries.sort(
                key=lambda entry: (
                    _sort_key(entry.effective_from),
                    _sort_key(entry.effective_until),
                )
            )
        except TypeError:
            raise ReferenceError(
                'Validity bounds of reference ({!r}, {!r}) use incompatible '
                'types and cannot be ordered.'.format(source, code)
            )
        return entries

    def lookup_all(self, source, code, point):
        """
        Return every entry that covers ``point`` on its half-open interval.
        More than one result means an overlap ambiguity.
        """
        result = []
        for entry in self._index.get((source, code), ()):
            if _covers(entry, point, source, code):
                result.append(entry)
        result.sort(key=lambda entry: _sort_key(entry.effective_from))
        return result

    def lookup(self, source, code, point):
        """
        Return the unique entry covering ``point`` or :obj:`None`. Overlaps are
        not resolved here; use :meth:`lookup_all` and :meth:`audit` to detect
        them.
        """
        matches = self.lookup_all(source, code, point)
        if len(matches) == 1:
            return matches[0]
        return None

    def audit(self, source=None, code=None):
        """
        Check coverage of every reference (or one reference) and report gaps
        and overlapping intervals.
        """
        if source is not None and code is not None:
            keys = [(source, code)]
        else:
            keys = list(self._index)

        issues = []
        for key in keys:
            entries = self._sorted(*key)
            for previous, current in zip(entries, entries[1:]):
                ref_source, ref_code = key
                if previous.effective_until is None:
                    issues.append(
                        CoverageIssue(
                            'overlap',
                            ref_source,
                            ref_code,
                            current.effective_from,
                            None,
                            (previous.version, current.version),
                        )
                    )
                elif _compare(
                    previous.effective_until,
                    current.effective_from,
                    'validity',
                    ref_source,
                    ref_code,
                ):
                    issues.append(
                        CoverageIssue(
                            'gap',
                            ref_source,
                            ref_code,
                            previous.effective_until,
                            current.effective_from,
                            (previous.version, current.version),
                        )
                    )
                elif _compare(
                    current.effective_from,
                    previous.effective_until,
                    'validity',
                    ref_source,
                    ref_code,
                ):
                    issues.append(
                        CoverageIssue(
                            'overlap',
                            ref_source,
                            ref_code,
                            current.effective_from,
                            previous.effective_until,
                            (previous.version, current.version),
                        )
                    )
        return issues

    def gap_at(self, source, code, point):
        """Return the gap issue containing ``point``, or :obj:`None`."""
        for issue in self.audit(source, code):
            if issue.kind != 'gap':
                continue
            if not _compare(
                point, issue.start, 'validity', source, code
            ) and _compare(point, issue.end, 'validity', source, code):
                return issue
        return None


def _sort_key(value):
    # Values of the same key are validated to be mutually orderable; the repr
    # fallback only keeps sorting deterministic for mixed display output.
    return (0, value) if value is not None else (1, repr(value))


def _covers(entry, point, source, code):
    if _compare(point, entry.effective_from, 'validity', source, code):
        return False
    if entry.effective_until is None:
        return True
    return _compare(point, entry.effective_until, 'validity', source, code)


class BatchGate(object):
    """
    An immutable view of a :class:`ReferenceTable` as of a batch's start.

    Entries added to the underlying table after the gate was frozen are not
    visible; they only apply to the next batch.
    """

    def __init__(self, table, at=None):
        self._source_table = table
        self.frozen_at = at
        self.frozen = False
        self.freeze(at)

    def freeze(self, at=None):
        """Snapshot the current table state and lock the gate."""
        if self.frozen:
            raise ReferenceError('This batch gate is already frozen.')
        if at is not None:
            self.frozen_at = at
        self._snapshot = ReferenceTable(
            self._source_table.entries(), name=self._source_table.name
        )
        self.frozen = True
        return self

    @property
    def table(self):
        """The frozen snapshot table."""
        return self._snapshot

    def entries(self, source=None, code=None):
        return self._snapshot.entries(source, code)

    def lookup(self, source, code, point):
        return self._snapshot.lookup(source, code, point)

    def lookup_all(self, source, code, point):
        return self._snapshot.lookup_all(source, code, point)

    def audit(self, source=None, code=None):
        return self._snapshot.audit(source, code)

    def gap_at(self, source, code, point):
        return self._snapshot.gap_at(source, code, point)

    def __len__(self):
        return len(self._snapshot)


def _line_identity(line, index, id_field):
    if isinstance(line, Mapping) and id_field in line:
        return line[id_field]
    return index


def verify(
    details,
    table,
    source_field='source',
    code_field='code',
    point_field='at',
    id_field='id',
):
    """
    Verify every line of a batch against a reference table or batch gate.

    :param details: An iterable of mappings, each holding ``source``, ``code``
                    and the validity point (``at`` by default).
    :param table: A :class:`ReferenceTable` or :class:`BatchGate`.
    :return: A list of :class:`Verdict` in input order.
    """
    verdicts = []
    for index, line in enumerate(list(details)):
        source = line[source_field]
        code = line[code_field]
        point = line[point_field]
        identifier = _line_identity(line, index, id_field)

        matches = table.lookup_all(source, code, point)
        versions = tuple(sorted((entry.version for entry in matches), key=repr))
        ambiguity = None
        hit = None

        if len(matches) > 1:
            ambiguity = 'overlap'
        elif len(matches) == 1:
            hit = matches[0].version
        elif table.gap_at(source, code, point) is not None:
            ambiguity = 'gap'

        verdicts.append(
            Verdict(
                identifier,
                source,
                code,
                point,
                hit,
                versions,
                ambiguity,
                tuple(matches),
            )
        )
    return verdicts


def _verdict_signature(verdict):
    return verdict.versions, verdict.ambiguity


def _verdict_index(details, table, field_names):
    index = {}
    for verdict in verify(details, table, **field_names):
        if verdict.identifier in index:
            raise ReferenceError(
                'Duplicate line identifier {!r}; recheck needs unique '
                'identifiers.'.format(verdict.identifier)
            )
        index[verdict.identifier] = verdict
    return index


def _describe_change(old, new):
    sides = []
    if old.ambiguity:
        sides.append('{} in old reference'.format(old.ambiguity))
    if new.ambiguity and new.ambiguity != old.ambiguity:
        sides.append('{} in new reference'.format(new.ambiguity))
    elif new.ambiguity:
        sides.append('{} also in new reference'.format(new.ambiguity))
    if not sides:
        sides.append(
            'hit changed from {old!r} to {new!r}'.format(
                old=old.hit, new=new.hit
            )
        )
    return '; '.join(sides)


def recheck(
    details,
    old_gate,
    new_table,
    source_field='source',
    code_field='code',
    point_field='at',
    id_field='id',
):
    """
    Re-verify persisted batch lines against a completed reference table.

    Only lines whose verdict differs from the frozen batch gate are returned.
    Each :class:`RecheckItem` states the old and new hit (or all candidates on
    overlap) and whether the difference is a gap or overlap ambiguity.
    """
    field_names = dict(
        source_field=source_field,
        code_field=code_field,
        point_field=point_field,
        id_field=id_field,
    )
    details = list(details)
    old_verdicts = _verdict_index(details, old_gate, field_names)
    new_verdicts = _verdict_index(details, new_table, field_names)

    affected = []
    for identifier, new in new_verdicts.items():
        old = old_verdicts[identifier]
        if _verdict_signature(old) == _verdict_signature(new):
            continue
        affected.append(
            RecheckItem(
                identifier,
                new.source,
                new.code,
                new.point,
                old.hit,
                new.hit,
                old.versions,
                new.versions,
                new.ambiguity or old.ambiguity,
                _describe_change(old, new),
            )
        )
    return affected


class ReferenceCheckingValidator(Validator):
    """
    A :class:`~cerberus.Validator` that supports a ``reference`` rule.

    The rule consults the :class:`BatchGate` (or :class:`ReferenceTable`)
    passed as ``reference_gate``::

        schema = {'sku': {'type': 'string',
                          'reference': {'source': 'catalog',
                                        'at_field': 'issued_at'}}}
        validator = ReferenceCheckingValidator(
            schema, reference_gate=batch_gate
        )

    ``source`` is the literal reference source, ``code_field`` names the
    document field holding the code (defaults to the validated field) and the
    validity point is taken from ``at`` (a literal) or ``at_field``.
    """

    _valid_schemas = set()
    """An isolated rule-schema cache, as this subclass adds the reference rule."""

    @property
    def reference_gate(self):
        return self._config.get('reference_gate')

    def _lookup_document_value(self, field_name):
        wanted_field, value = self._lookup_field(field_name)
        if wanted_field is None:
            raise ReferenceError(
                "Field {!r} used by a reference rule is missing from the "
                "document.".format(field_name)
            )
        return value

    def _validate_reference(self, constraint, field, value):
        """
        {'type': 'dict',
         'schema': {'source': {'required': True, 'type': 'hashable'},
                    'code_field': {'type': 'string'},
                    'at': {'type': 'hashable', 'excludes': 'at_field'},
                    'at_field': {'type': 'string', 'excludes': 'at'}}}
        """
        gate = self.reference_gate
        if gate is None:
            self._error(
                field,
                errors.CUSTOM,
                'No reference gate configured for the reference rule.',
            )
            return

        if 'at' in constraint:
            point = constraint['at']
        elif 'at_field' in constraint:
            try:
                point = self._lookup_document_value(constraint['at_field'])
            except ReferenceError as exc:
                self._error(field, errors.CUSTOM, str(exc))
                return
        else:
            self._error(
                field,
                errors.CUSTOM,
                "The reference rule requires an 'at' or 'at_field' constraint.",
            )
            return

        code_field = constraint.get('code_field', field)
        try:
            code = self._lookup_document_value(code_field)
        except ReferenceError as exc:
            self._error(field, errors.CUSTOM, str(exc))
            return

        source = constraint['source']
        matches = gate.lookup_all(source, code, point)

        if len(matches) > 1:
            versions = tuple(
                sorted((entry.version for entry in matches), key=repr)
            )
            self._error(
                field,
                errors.REFERENCE_AMBIGUOUS,
                'overlapping versions {}'.format(versions),
            )
            return

        if len(matches) == 1:
            return

        gap = gate.gap_at(source, code, point)
        if gap is not None:
            self._error(
                field,
                errors.REFERENCE_AMBIGUOUS,
                'gap between versions {} on [{}, {})'.format(
                    gap.versions, gap.start, gap.end
                ),
            )
        else:
            self._error(field, errors.REFERENCE_NOT_FOUND)
