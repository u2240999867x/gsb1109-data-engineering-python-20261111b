"""Validation of product change streams on top of Cerberus.

An on-sale interval is identified by the pair (``source``, ``external_id``).
The same identity may reappear in disjoint validity periods; only overlapping
periods are reported as conflicts.  Besides ``upsert`` events the stream
carries ``terminate`` events (closing a previous interval) and ``replace``
events (superseding a previous event).  Processing is idempotent by event
number: re-applying an already seen ``event_no`` is a no-op.

Rows whose identity cannot be converted (a blank ``source`` or an
``external_id`` that does not coerce to an integer) are quarantined and kept
apart from the valid changes.  Replaying the valid changes onto an empty
state yields the same conflicts as validating the final version in one pass.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Mapping, Optional

from cerberus import Validator

__all__ = [
    'ChangeReport',
    'ChangeStreamValidator',
    'Conflict',
    'QuarantinedRow',
    'validate_change_stream',
]

OP_UPSERT = 'upsert'
OP_TERMINATE = 'terminate'
OP_REPLACE = 'replace'


def _coerce_external_id(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a valid external id')
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise ValueError('cannot convert to an integer external id')


def _coerce_moment(value):
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str) and value.strip():
        moment = datetime.fromisoformat(value.strip())
    else:
        raise ValueError('cannot convert to a datetime')
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


_BASE_SCHEMA = {
    'op': {
        'type': 'string',
        'required': True,
        'allowed': (OP_UPSERT, OP_TERMINATE, OP_REPLACE),
    },
    'source': {'type': 'string', 'empty': False},
    'external_id': {'coerce': _coerce_external_id},
    'valid_from': {'coerce': _coerce_moment},
    'valid_to': {'coerce': _coerce_moment, 'nullable': True},
    'terminated_at': {'coerce': _coerce_moment},
    'replaces_event_no': {'type': 'integer'},
    'payload': {'type': 'dict'},
}

_ROW_SCHEMA = dict(
    _BASE_SCHEMA,
    event_no={'type': 'integer', 'required': True},
    row_no={'type': 'integer', 'required': True},
)

_PAYLOAD_SCHEMA = dict(
    _BASE_SCHEMA,
    op={'type': 'string', 'required': True, 'allowed': (OP_UPSERT, OP_TERMINATE)},
)

_ROW_VALIDATOR = Validator(_ROW_SCHEMA, allow_unknown=True)
_PAYLOAD_VALIDATOR = Validator(_PAYLOAD_SCHEMA, allow_unknown=True)


@dataclass(frozen=True)
class Conflict:
    """One truly conflicting time segment of an identity."""

    source: str
    external_id: int
    start: datetime
    end: Optional[datetime]  # None means open-ended
    row_nos: tuple  # sorted original row numbers participating


@dataclass(frozen=True)
class QuarantinedRow:
    """A row set aside because it failed validation."""

    row_no: Optional[int]
    event_no: Optional[int]
    errors: dict


@dataclass
class ChangeReport:
    """Outcome of applying a batch of change rows."""

    introduced: list = field(default_factory=list)
    resolved: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    quarantined: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)


@dataclass
class _Event:
    event_no: Optional[int]
    row_no: Optional[int]
    op: str
    source: Optional[str] = None
    external_id: Optional[int] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    replaces: Optional[int] = None
    payload: Optional['_Event'] = None


@dataclass
class _Interval:
    start: datetime
    end: Optional[datetime]
    row_no: int
    event_no: int


def _validate_payload(raw):
    doc = _PAYLOAD_VALIDATOR.validated(dict(raw))
    if doc is None:
        return None, dict(_PAYLOAD_VALIDATOR.errors)
    errors = _check_semantics(doc)
    if errors:
        return None, errors
    return _build_event(doc, is_payload=True), {}


def _check_semantics(doc):
    errors = defaultdict(list)
    op = doc.get('op')
    if op in (OP_UPSERT, OP_TERMINATE):
        for name in ('source', 'external_id'):
            if doc.get(name) is None:
                errors[name].append('required for %s' % op)
    if op == OP_UPSERT:
        if doc.get('valid_from') is None:
            errors['valid_from'].append('required for upsert')
        start, end = doc.get('valid_from'), doc.get('valid_to')
        if start is not None and end is not None and end <= start:
            errors['valid_to'].append('must be after valid_from')
    elif op == OP_TERMINATE:
        if doc.get('terminated_at') is None:
            errors['terminated_at'].append('required for terminate')
    return dict(errors)


def _build_event(doc, is_payload=False):
    payload = None
    if doc.get('op') == OP_REPLACE and doc.get('payload') is not None:
        payload, _ = _validate_payload(doc['payload'])
    return _Event(
        event_no=doc.get('event_no'),
        row_no=doc.get('row_no'),
        op=doc['op'],
        source=doc.get('source'),
        external_id=doc.get('external_id'),
        start=doc.get('valid_from') or doc.get('terminated_at'),
        end=doc.get('valid_to'),
        replaces=doc.get('replaces_event_no'),
        payload=payload,
    )


def _validate_row(row):
    """Return ``(event, errors)``; exactly one of both is truthy."""
    if not isinstance(row, Mapping):
        return None, {'row': ['must be a mapping']}
    doc = _ROW_VALIDATOR.validated(dict(row))
    if doc is None:
        return None, dict(_ROW_VALIDATOR.errors)
    errors = _check_semantics(doc)
    if doc['op'] == OP_REPLACE:
        if doc.get('replaces_event_no') is None:
            errors.setdefault('replaces_event_no', []).append(
                'required for replace'
            )
        if doc.get('payload') is None:
            errors.setdefault('payload', []).append('required for replace')
        else:
            _, payload_errors = _validate_payload(doc['payload'])
            if payload_errors:
                errors['payload'] = payload_errors
    if errors:
        return None, errors
    return _build_event(doc), {}


def _conflict_sort_key(conflict):
    return (
        conflict.source,
        conflict.external_id,
        conflict.start,
        conflict.end is None,
        conflict.end or conflict.start,
        conflict.row_nos,
    )


def _segments_for_intervals(intervals):
    """Sweep-line over one identity's intervals, yielding conflict segments."""
    points = []
    for idx, interval in enumerate(intervals):
        points.append((interval.start, 1, idx))
        if interval.end is not None:
            points.append((interval.end, 0, idx))
    points.sort(key=lambda point: (point[0], point[1]))

    segments = []
    active = set()
    previous = None
    for moment, is_start, idx in points:
        if previous is not None and previous < moment and len(active) >= 2:
            segments.append([previous, moment, frozenset(active)])
        if is_start:
            active.add(idx)
        else:
            active.discard(idx)
        previous = moment
    if previous is not None and len(active) >= 2:
        segments.append([previous, None, frozenset(active)])

    merged = []
    for segment in segments:
        if merged and merged[-1][2] == segment[2] and merged[-1][1] == segment[0]:
            merged[-1][1] = segment[1]
        else:
            merged.append(segment)
    return merged


def _fold(events):
    """Replay events in event-number order onto an empty state."""
    ordered = sorted(events, key=lambda event: event.event_no)
    known = {event.event_no for event in ordered}

    superseded = set()
    replaced_by = {}
    for event in ordered:
        if event.op != OP_REPLACE or event.replaces is None:
            continue
        if event.replaces == event.event_no:
            continue
        current = event.replaces
        seen = {event.event_no}
        while current in replaced_by and current not in seen:
            seen.add(current)
            current = replaced_by[current]
        if current in seen:
            continue
        if current in known:
            superseded.add(current)
        replaced_by[event.replaces] = event.event_no

    intervals = defaultdict(list)
    for event in ordered:
        if event.event_no in superseded:
            continue
        effective = event
        if event.op == OP_REPLACE:
            if event.payload is None:
                continue
            effective = _Event(
                event_no=event.event_no,
                row_no=event.row_no,
                op=event.payload.op,
                source=event.payload.source,
                external_id=event.payload.external_id,
                start=event.payload.start,
                end=event.payload.end,
            )
        key = (effective.source, effective.external_id)
        if effective.op == OP_UPSERT:
            intervals[key].append(
                _Interval(
                    start=effective.start,
                    end=effective.end,
                    row_no=effective.row_no,
                    event_no=effective.event_no,
                )
            )
        elif effective.op == OP_TERMINATE:
            for interval in intervals.get(key, ()):
                if interval.start < effective.start and (
                    interval.end is None or effective.start < interval.end
                ):
                    interval.end = effective.start
    return intervals


def _compute_conflicts(events):
    conflicts = []
    for (source, external_id), intervals in _fold(events).items():
        for start, end, participants in _segments_for_intervals(intervals):
            row_nos = tuple(sorted({intervals[i].row_no for i in participants}))
            conflicts.append(
                Conflict(
                    source=source,
                    external_id=external_id,
                    start=start,
                    end=end,
                    row_nos=row_nos,
                )
            )
    return sorted(conflicts, key=_conflict_sort_key)


class ChangeStreamValidator:
    """Stateful, idempotent validator for a stream of product changes."""

    def __init__(self):
        self._journal = {}
        self._conflicts = []
        self._quarantined = []

    @property
    def conflicts(self):
        return list(self._conflicts)

    @property
    def quarantined(self):
        return list(self._quarantined)

    def apply(self, rows):
        """Apply a batch of raw change rows, returning a ChangeReport."""
        quarantined = []
        duplicates = []
        accepted = []
        seen = set(self._journal)
        for row in rows:
            event, errors = _validate_row(row)
            if errors:
                raw = row if isinstance(row, Mapping) else {}
                quarantined.append(
                    QuarantinedRow(
                        row_no=raw.get('row_no'),
                        event_no=raw.get('event_no'),
                        errors=errors,
                    )
                )
                continue
            if event.event_no in seen:
                duplicates.append(event.event_no)
                continue
            seen.add(event.event_no)
            accepted.append(event)

        for event in accepted:
            self._journal[event.event_no] = event
        self._quarantined.extend(quarantined)

        before = self._conflicts
        self._conflicts = _compute_conflicts(self._journal.values())
        introduced = [c for c in self._conflicts if c not in before]
        resolved = [c for c in before if c not in self._conflicts]
        return ChangeReport(
            introduced=introduced,
            resolved=resolved,
            conflicts=list(self._conflicts),
            quarantined=quarantined,
            duplicates=duplicates,
        )

    def replay(self):
        """Replay the accepted changes onto an empty state.

        The result matches validating the final journal in one pass, because
        conflicts are always derived by folding the journaled events in
        event-number order.
        """
        fresh = ChangeStreamValidator()
        fresh._journal = dict(self._journal)
        fresh._conflicts = _compute_conflicts(fresh._journal.values())
        return fresh


def validate_change_stream(rows: Iterable[Mapping]):
    """Validate a whole change stream in one pass."""
    validator = ChangeStreamValidator()
    return validator.apply(rows)
