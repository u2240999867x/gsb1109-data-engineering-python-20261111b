"""Timezone-aware alignment of wall-clock scheduling records.

A scheduling record carries a local date, a time-zone offset and a ``fold``
marker. It is aligned, by the wall-clock time shown on a clock, to reference
marks stored in a :class:`~sortedcollections.NearestDict`.

Candidates are generated per local day. By default only the record's own
calendar day is considered; callers may opt in to the previous and next day so
that a late-evening record can align to an early-morning mark (and vice
versa). Distances for the nearest, previous and next modes are compared using
*true elapsed time* in UTC, so daylight-saving transitions are handled
correctly: a fall-back wall time names two distinct instants and a
spring-forward gap names none.

The record's original local date, UTC offset and ``fold`` are echoed back on
the result as evidence rather than being silently normalised.
"""

from collections import namedtuple
from datetime import datetime, timedelta, timezone

from .nearestdict import NearestDict

PREVIOUS = NearestDict.NEAREST_PREV
NEAREST = NearestDict.NEAREST
NEXT = NearestDict.NEAREST_NEXT

AlignmentResult = namedtuple(
    'AlignmentResult',
    'key wall_time distance date offset fold',
)


class NoCandidate(KeyError):
    """Raised when no admissible reference mark exists.

    This is distinct from :class:`ToleranceExceeded`: there was nothing to
    compare, either because the reference set was empty or because every
    candidate wall time fell in a spring-forward gap or outside the allowed
    calendar days.
    """


class ToleranceExceeded(KeyError):
    """Raised when candidates exist but all are farther than the tolerance."""


def _instant(local):
    """Return the fold-aware UTC instant named by an aware datetime."""
    return local.astimezone(timezone.utc)


def _wall_time_exists(local):
    """Return ``True`` unless ``local`` falls in a spring-forward gap.

    A non-existent local time round-trips through UTC to a different wall
    clock. Ambiguous (fall-back) times round-trip to themselves for each
    fold and are therefore considered to exist.
    """
    mirrored = _instant(local).astimezone(local.tzinfo)
    return mirrored.replace(tzinfo=None) == local.replace(tzinfo=None)


def _wall_on_day(key, record_date, day_offset):
    """Place reference ``key`` on the record's day plus ``day_offset`` days."""
    day = record_date + timedelta(days=day_offset)
    return datetime.combine(day, key.timetz())


def _elapsed(left, right):
    """Absolute true elapsed time between two aware datetimes."""
    delta = _instant(left) - _instant(right)
    if delta < timedelta():
        delta = -delta
    return delta


def ordinary_linear_nearest(references, request):
    """Return the nearest key by plain wall-clock subtraction.

    This mirrors the existing :class:`NearestDict` behaviour using an
    ordinary linear scan: distance is ``abs(key - request)``, which for aware
    datetimes compares wall-clock components and ignores ``fold``. It is kept
    deliberately naive so callers can contrast it with the true-time
    alignment performed by :class:`ScheduleAlignment`.
    """
    keys = list(references)
    if not keys:
        raise KeyError('NearestDict is empty')
    return min(keys, key=lambda key: abs(key - request))


class ScheduleAlignment:
    """Align scheduling records to reference marks on a clock face.

    :param references: :class:`NearestDict` whose keys are aware reference
        datetimes; only their wall time and time zone matter when generating
        candidates.
    :param rounding: one of :data:`PREVIOUS`, :data:`NEAREST` or
        :data:`NEXT` (default :data:`NEAREST`).
    :param allow_cross_day: when true the previous and next calendar days are
        searched as well as the record's own day.
    :param tolerance: optional maximum true elapsed :class:`datetime.timedelta`
        between the record and the chosen mark.
    """

    def __init__(
        self,
        references,
        rounding=NEAREST,
        allow_cross_day=False,
        tolerance=None,
    ):
        self.references = references
        self.rounding = rounding
        self.allow_cross_day = allow_cross_day
        self.tolerance = tolerance

    def _day_offsets(self):
        if self.allow_cross_day:
            return (-1, 0, 1)
        return (0,)

    def candidates(self, record, apply_tolerance=True):
        """Return admissible ``(distance, wall_time, key)`` triples.

        Spring-forward gap marks are omitted. With ``apply_tolerance`` false
        the tolerance filter is bypassed, which is used internally to tell a
        tolerance failure apart from having no candidate at all.
        """
        candidates = []
        record_date = record.date()
        for key in self.references:
            for day_offset in self._day_offsets():
                wall_time = _wall_on_day(
                    key, record_date, day_offset
                )
                if not _wall_time_exists(wall_time):
                    continue
                distance = _elapsed(wall_time, record)
                if apply_tolerance and self.tolerance is not None:
                    if distance > self.tolerance:
                        continue
                candidates.append((distance, wall_time, key))
        return candidates

    def _select(self, candidates, record):
        record_instant = _instant(record)
        if self.rounding == PREVIOUS:
            earlier = [
                item
                for item in candidates
                if _instant(item[1]) <= record_instant
            ]
            if not earlier:
                raise NoCandidate(f'No mark at or before {record!r} found')
            return max(earlier, key=lambda item: _instant(item[1]))
        if self.rounding == NEXT:
            later = [
                item
                for item in candidates
                if _instant(item[1]) >= record_instant
            ]
            if not later:
                raise NoCandidate(f'No mark at or after {record!r} found')
            return min(later, key=lambda item: _instant(item[1]))
        closest = min(distance for distance, _, _ in candidates)
        tied = [item for item in candidates if item[0] == closest]
        return max(tied, key=lambda item: _instant(item[1]))

    def align(self, record):
        """Align one aware scheduling ``record`` and return the evidence.

        :param record: timezone-aware :class:`datetime.datetime`; its
            ``fold`` selects which instant an ambiguous wall time names.
        :return: :class:`AlignmentResult` with the chosen key, the candidate
            wall time, true elapsed distance and the record's original date,
            UTC offset and fold.
        :raises NoCandidate: if no admissible mark exists.
        :raises ToleranceExceeded: if marks exist but none is within
            ``tolerance``.
        """
        candidates = self.candidates(record)
        if not candidates:
            if self.candidates(record, apply_tolerance=False):
                raise ToleranceExceeded(
                    f'Every mark is farther than {self.tolerance!r} from '
                    f'{record!r}'
                )
            raise NoCandidate(f'No admissible mark for {record!r} found')
        distance, wall_time, key = self._select(candidates, record)
        return AlignmentResult(
            key=key,
            wall_time=wall_time,
            distance=distance,
            date=record.date(),
            offset=record.utcoffset(),
            fold=record.fold,
        )
