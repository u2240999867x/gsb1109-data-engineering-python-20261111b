"""Compatibility checks for composed :class:`relativedelta` operations.

The three forms compared by this module are:

``sequential``
    Apply the deltas in the caller's order, one at a time.
``merged``
    Add the deltas together with ``relativedelta.__add__`` and apply the
    result once.  This is the form produced by a scheduler that collapses a
    chain of rules into a single delta.
``swapped``
    Apply the same deltas in reverse order.

Only dates on which those three results are not all equal are reported.
Contiguous dates with the same equality pattern and the same set of causes
are grouped into one interval.

Ranges whose rules contain only relative fields are evaluated by scanning
one Gregorian cycle (400 calendar years) and tiling the result, so labels
are attached to real boundary dates in that cycle rather than inferred for
an entire month.

The module also provides a small command line interface::

    python -m dateutil.relativedelta_compat \
        --start 2019-01-01 --end 2021-12-31 \
        --delta 'months=1' --delta 'months=1'

Use ``--format json`` for machine-readable output and ``--days`` to emit the
un-grouped, per-date result.  ``--fail-on-divergence`` makes the command exit
with status 1 when at least one divergent date is found.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

from .relativedelta import relativedelta, weekday

__all__ = [
    "CAUSE_ABSOLUTE_FIELD",
    "CAUSE_LEAP_DAY",
    "CAUSE_MONTH_END",
    "CAUSE_WEEKDAY",
    "CAUSES",
    "Analysis",
    "DivergenceInterval",
    "analyze_range",
    "find_divergences",
    "parse_delta",
]

CAUSE_ABSOLUTE_FIELD = "absolute-field-override"
CAUSE_LEAP_DAY = "leap-day"
CAUSE_MONTH_END = "month-end-clamp"
CAUSE_WEEKDAY = "weekday-interaction"

CAUSES = (
    CAUSE_ABSOLUTE_FIELD,
    CAUSE_LEAP_DAY,
    CAUSE_MONTH_END,
    CAUSE_WEEKDAY,
)
_CAUSE_INDEX = {cause: index for index, cause in enumerate(CAUSES)}

_ABSOLUTE_FIELDS = (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
)
_RELATIVE_FIELDS = (
    "years",
    "months",
    "days",
    "leapdays",
    "hours",
    "minutes",
    "seconds",
    "microseconds",
)
_DELTA_FIELDS = _ABSOLUTE_FIELDS + _RELATIVE_FIELDS + ("weekday",)

_GREGORIAN_CYCLE_DAYS = 146097
_CYCLE_ANCHOR = date(2000, 1, 1)
_FORM_NAMES = ("sequential", "merged", "swapped")
_PAIR_NAMES = (
    ("sequential", "merged"),
    ("sequential", "swapped"),
    ("merged", "swapped"),
)
_CYCLE_CACHE: dict[tuple, tuple[tuple[int, int, "Label"], ...]] = {}
_CYCLE_CACHE_LIMIT = 16

Label = tuple[tuple[int, int, int], tuple[str, ...]]


@dataclass(frozen=True)
class DivergenceInterval:
    """A maximal run of start dates with the same divergence signature."""

    start: date
    end: date
    causes: tuple[str, ...]
    differing_pairs: tuple[str, ...]
    first_results: tuple[object, object, object]
    last_results: tuple[object, object, object]

    @property
    def days(self) -> int:
        return self.end.toordinal() - self.start.toordinal() + 1

    def as_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
            "causes": list(self.causes),
            "differing_pairs": list(self.differing_pairs),
            "first_results": {
                name: _isoformat_value(value)
                for name, value in zip(_FORM_NAMES, self.first_results)
            },
            "last_results": {
                name: _isoformat_value(value)
                for name, value in zip(_FORM_NAMES, self.last_results)
            },
        }

@dataclass(frozen=True)
class Analysis:
    """Result of :func:`analyze_range`."""

    start: date
    end: date
    deltas: tuple[relativedelta, ...]
    periodic: bool
    mode: str
    intervals: tuple[DivergenceInterval, ...]

    @property
    def interval_count(self) -> int:
        return len(self.intervals)

    @property
    def divergent_days(self) -> int:
        return sum(interval.days for interval in self.intervals)

    def as_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "deltas": [repr(delta) for delta in self.deltas],
            "periodic": self.periodic,
            "mode": self.mode,
            "divergent_days": self.divergent_days,
            "interval_count": self.interval_count,
            "intervals": [interval.as_dict() for interval in self.intervals],
        }


def _isoformat_value(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return repr(value)


def _clone_delta(delta: relativedelta, **overrides) -> relativedelta:
    values = {field: getattr(delta, field) for field in _DELTA_FIELDS}
    values.update(overrides)
    return type(delta)(**values)


def _strip_absolute(delta: relativedelta) -> relativedelta:
    return _clone_delta(delta, **{field: None for field in _ABSOLUTE_FIELDS})


def _strip_weekday(delta: relativedelta) -> relativedelta:
    return _clone_delta(delta, weekday=None)


def _strip_leapdays(delta: relativedelta) -> relativedelta:
    return _clone_delta(delta, leapdays=0)


def _has_absolute(deltas: Sequence[relativedelta]) -> bool:
    return any(
        getattr(delta, field) is not None
        for delta in deltas
        for field in _ABSOLUTE_FIELDS
    )


def _has_weekday(deltas: Sequence[relativedelta]) -> bool:
    return any(delta.weekday is not None for delta in deltas)


def _has_leapdays(deltas: Sequence[relativedelta]) -> bool:
    return any(delta.leapdays for delta in deltas)


def _merge(deltas: Sequence[relativedelta]) -> relativedelta:
    merged = deltas[0]
    for delta in deltas[1:]:
        merged = merged + delta
    return merged


def _apply(start: date, deltas: Sequence[relativedelta]):
    current = start
    for delta in deltas:
        current = current + delta
    return current


def _apply_with_steps(start: date, deltas: Sequence[relativedelta]):
    current = start
    steps = []
    for delta in deltas:
        before = current
        current = current + delta
        steps.append((before, current, delta))
    return current, steps


def _formulations(start: date, deltas: Sequence[relativedelta], merged):
    sequential, sequential_steps = _apply_with_steps(start, deltas)
    merged_result, merged_steps = _apply_with_steps(start, (merged,))
    swapped, swapped_steps = _apply_with_steps(start, tuple(reversed(deltas)))
    return (
        (sequential, merged_result, swapped),
        (sequential_steps, merged_steps, swapped_steps),
    )


def _final_results(start: date, deltas: Sequence[relativedelta]):
    merged = _merge(deltas)
    return (
        _apply(start, deltas),
        _apply(start, (merged,)),
        _apply(start, tuple(reversed(deltas))),
    )


def _partition(results: Sequence[object]) -> tuple[int, int, int]:
    ids = []
    for index, value in enumerate(results):
        for previous, previous_value in enumerate(results[:index]):
            if value == previous_value:
                ids.append(previous)
                break
        else:
            ids.append(index)
    return tuple(ids)


def _differing_pairs(partition: tuple[int, int, int]) -> tuple[str, ...]:
    pairs = []
    for index, (left, right) in enumerate(
        ((0, 1), (0, 2), (1, 2))
    ):
        if partition[left] != partition[right]:
            pairs.append("%s!=%s" % _PAIR_NAMES[index])
    return tuple(pairs)


def _month_shift(before: date, delta: relativedelta) -> tuple[int, int, int]:
    year = (delta.year or before.year) + delta.years
    month = delta.month or before.month
    if delta.months:
        month += delta.months
        if month > 12:
            year += 1
            month -= 12
        elif month < 1:
            year -= 1
            month += 12
    day = delta.day if delta.day is not None else before.day
    return year, month, day


def _clamp_event(before: date, after: date, delta: relativedelta):
    year, month, nominal_day = _month_shift(before, delta)
    if not (1 <= year <= 9999 and 1 <= month <= 12):
        return None
    month_days = calendar.monthrange(year, month)[1]
    if nominal_day <= month_days:
        return None

    # The library applies weekday after clamped date construction, so compare
    # with the same delta with weekday removed to see the clamped day.
    pre_weekday = before
    if delta.weekday is not None:
        pre_weekday = before + _strip_weekday(delta)
    else:
        pre_weekday = after
    return (year, month, int(nominal_day), pre_weekday.day)


def _clamp_events(steps) -> tuple[tuple[int, int, int, int], ...]:
    events = []
    for before, after, delta in steps:
        event = _clamp_event(before, after, delta)
        if event is not None:
            events.append(event)
    return tuple(events)


def _bundle(deltas: Sequence[relativedelta], start: date):
    return _final_results(start, deltas)


def _ablation_changes(
    start: date,
    deltas: Sequence[relativedelta],
    transform,
) -> bool:
    alternative = tuple(transform(delta) for delta in deltas)
    return _bundle(alternative, start) != _bundle(deltas, start)


def _day_replaced(start: date, day: int):
    return start.replace(day=day)


def _common_year_start(start: date) -> date | None:
    for year in (2019, 2021, 2018, 2022):
        try:
            return _day_replaced(start.replace(year=year), start.day)
        except ValueError:
            continue
    return None


def _replace_absolute_year(delta: relativedelta, year: int) -> relativedelta:
    if delta.year is None or not calendar.isleap(delta.year):
        return delta
    return _clone_delta(delta, year=year)


def _leap_changes_outcome(
    start: date,
    deltas: Sequence[relativedelta],
    partition: tuple[int, int, int],
    leap_target_year: int | None,
) -> bool:
    analog = _common_year_start(start)
    if analog is not None:
        try:
            analog_partition = _partition(_final_results(analog, deltas))
        except ValueError:
            analog_partition = None
        if analog_partition != partition:
            return True

    if leap_target_year is not None:
        alternative = tuple(
            _replace_absolute_year(delta, 2021) for delta in deltas
        )
        if alternative != deltas:
            try:
                alternative_partition = _partition(
                    _final_results(start, alternative)
                )
            except ValueError:
                alternative_partition = None
            if alternative_partition != partition:
                return True
    return False


def _leap_ablation_changes(
    start: date,
    deltas: Sequence[relativedelta],
    partition: tuple[int, int, int],
) -> bool:
    if _has_leapdays(deltas) and _ablation_changes(
        start, deltas, _strip_leapdays
    ):
        return True

    if start.month == 2 and start.day == 29:
        # On a leap day itself, distinguish an ordinary divergence (which
        # also happens on 28 February) from one that needs the extra day.
        try:
            feb28_partition = _partition(
                _final_results(_day_replaced(start, 28), deltas)
            )
        except ValueError:
            return True
        if feb28_partition != partition:
            return True

    leap_target_years = set()
    for steps in _formulations_for_causes(start, deltas):
        for before, after, _delta in steps:
            if (before.month, before.day) == (2, 29):
                leap_target_years.add(before.year)
            if (after.month, after.day) == (2, 29):
                leap_target_years.add(after.year)
    for year in leap_target_years:
        if _leap_changes_outcome(start, deltas, partition, year):
            return True
    return False


def _cause_labels(
    start: date,
    deltas: Sequence[relativedelta],
    partition: tuple[int, int, int],
) -> tuple[str, ...]:
    causes = set()
    absolute = _has_absolute(deltas)
    has_weekday = _has_weekday(deltas)

    if absolute and _ablation_changes(start, deltas, _strip_absolute):
        causes.add(CAUSE_ABSOLUTE_FIELD)

    if has_weekday and _ablation_changes(start, deltas, _strip_weekday):
        causes.add(CAUSE_WEEKDAY)

    if absolute and has_weekday and not causes:
        if _ablation_changes(
            start,
            deltas,
            lambda delta: _strip_weekday(_strip_absolute(delta)),
        ):
            causes.add(CAUSE_ABSOLUTE_FIELD)
            causes.add(CAUSE_WEEKDAY)

    steps_by_form = _formulations_for_causes(start, deltas)
    clamp_events = tuple(
        _clamp_events(steps) for steps in steps_by_form
    )
    if any(clamp_events) and not (
        clamp_events[0] == clamp_events[1] == clamp_events[2]
    ):
        causes.add(CAUSE_MONTH_END)

    if _leap_ablation_changes(start, deltas, partition):
        causes.add(CAUSE_LEAP_DAY)

    if not causes:
        if absolute:
            causes.add(CAUSE_ABSOLUTE_FIELD)
        elif has_weekday:
            causes.add(CAUSE_WEEKDAY)
        else:
            causes.add(CAUSE_MONTH_END)

    return tuple(sorted(causes, key=_CAUSE_INDEX.__getitem__))


def _formulations_for_causes(start: date, deltas: Sequence[relativedelta]):
    merged = _merge(deltas)
    return _formulations(start, deltas, merged)[1]


def _classify(
    start: date, deltas: Sequence[relativedelta], merged
) -> Label | None:
    results = _final_results(start, deltas)
    partition = _partition(results)
    if partition == (0, 0, 0):
        return None

    causes = _cause_labels(start, deltas, partition)
    return (partition, causes)


def _is_periodic_rule(deltas: Sequence[relativedelta]) -> bool:
    return not _has_absolute(deltas)


def _validate_range(start: date, end: date) -> None:
    if not isinstance(start, (date, datetime)) or not isinstance(
        end, (date, datetime)
    ):
        raise TypeError("start and end must be date or datetime instances")
    if isinstance(start, datetime) != isinstance(end, datetime):
        raise TypeError("start and end must have the same type")
    if isinstance(start, datetime):
        if (
            start.hour,
            start.minute,
            start.second,
            start.microsecond,
            start.fold,
        ) != (
            end.hour,
            end.minute,
            end.second,
            end.microsecond,
            end.fold,
        ):
            raise ValueError(
                "datetime ranges must use the same time of day at both ends"
            )
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise TypeError(
                "datetime ranges must be both naive or both timezone-aware"
            )
        if start.utcoffset() != end.utcoffset():
            raise ValueError(
                "datetime ranges must use the same UTC offset at both ends"
            )
    if start > end:
        raise ValueError("start must not be later than end")


def _template_date(template: date, ordinal: int) -> date:
    value = date.fromordinal(ordinal)
    if isinstance(template, datetime):
        return template.replace(
            year=value.year,
            month=value.month,
            day=value.day,
        )
    return value


def _canonical_start(template: date) -> date:
    if isinstance(template, datetime):
        return datetime(
            2000,
            1,
            1,
            template.hour,
            template.minute,
            template.second,
            template.microsecond,
            tzinfo=template.tzinfo,
            fold=template.fold,
        )
    return _CYCLE_ANCHOR


def _cycle_runs(
    deltas: Sequence[relativedelta], template: date
) -> tuple[tuple[int, int, Label], ...]:
    canonical = _canonical_start(template)
    key = (
        tuple(repr(delta) for delta in deltas),
        type(canonical),
        canonical.isoformat(),
    )
    cached = _CYCLE_CACHE.get(key)
    if cached is not None:
        return cached

    merged = _merge(deltas)
    runs = []
    run_start = None
    run_label = None
    current = canonical
    for offset in range(_GREGORIAN_CYCLE_DAYS):
        label = _classify(current, deltas, merged)
        if label is None:
            if run_label is not None:
                runs.append((run_start, offset - 1, run_label))
                run_start = run_label = None
        elif label != run_label:
            if run_label is not None:
                runs.append((run_start, offset - 1, run_label))
            run_start = offset
            run_label = label
        current = current + timedelta(days=1)
    if run_label is not None:
        runs.append((run_start, _GREGORIAN_CYCLE_DAYS - 1, run_label))

    result = tuple(runs)
    if len(_CYCLE_CACHE) >= _CYCLE_CACHE_LIMIT:
        _CYCLE_CACHE.pop(next(iter(_CYCLE_CACHE)))
    _CYCLE_CACHE[key] = result
    return result


def _build_interval(
    start_ordinal: int,
    end_ordinal: int,
    label: Label,
    template: date,
    deltas: Sequence[relativedelta],
    merged,
) -> DivergenceInterval:
    start = _template_date(template, start_ordinal)
    end = _template_date(template, end_ordinal)
    first_results = _formulations(start, deltas, merged)[0]
    last_results = _formulations(end, deltas, merged)[0]
    partition, causes = label
    return DivergenceInterval(
        start=start,
        end=end,
        causes=causes,
        differing_pairs=_differing_pairs(partition),
        first_results=first_results,
        last_results=last_results,
    )


def _scan_runs(start: date, end: date, deltas, merged):
    runs = []
    run_start = None
    run_label = None
    start_ordinal = start.toordinal()
    end_ordinal = end.toordinal()
    for ordinal in range(start_ordinal, end_ordinal + 1):
        current = _template_date(start, ordinal)
        label = _classify(current, deltas, merged)
        if label is None:
            if run_label is not None:
                runs.append((run_start, ordinal - 1, run_label))
                run_start = run_label = None
        elif label != run_label:
            if run_label is not None:
                runs.append((run_start, ordinal - 1, run_label))
            run_start = ordinal
            run_label = label
    if run_label is not None:
        runs.append((run_start, end_ordinal, run_label))
    return tuple(runs)


def _periodic_runs(start: date, end: date, deltas, merged):
    template = start
    if isinstance(template, datetime) and template.tzinfo is not None:
        if template.tzinfo.utcoffset(template) != template.tzinfo.utcoffset(
            _template_date(template, template.toordinal() + 1)
        ):
            return None

    start_ordinal = start.toordinal()
    end_ordinal = end.toordinal()
    anchor_ordinal = _CYCLE_ANCHOR.toordinal()
    cycle_runs = _cycle_runs(deltas, template)

    k_start = (start_ordinal - anchor_ordinal) // _GREGORIAN_CYCLE_DAYS
    k_end = (end_ordinal - anchor_ordinal) // _GREGORIAN_CYCLE_DAYS

    pieces = []
    for cycle in range(k_start, k_end + 1):
        shift = anchor_ordinal + cycle * _GREGORIAN_CYCLE_DAYS
        for run_start, run_end, label in cycle_runs:
            first = shift + run_start
            last = shift + run_end
            if last < start_ordinal or first > end_ordinal:
                continue
            pieces.append(
                (max(first, start_ordinal), min(last, end_ordinal), label)
            )

    pieces.sort(key=lambda piece: piece[0])
    merged_pieces = []
    for first, last, label in pieces:
        if (
            merged_pieces
            and merged_pieces[-1][1] + 1 == first
            and merged_pieces[-1][2] == label
        ):
            merged_pieces[-1] = (merged_pieces[-1][0], last, label)
        else:
            merged_pieces.append((first, last, label))
    return tuple(merged_pieces)


def analyze_range(
    start: date,
    end: date,
    deltas: Iterable[relativedelta],
    *,
    use_cycle: bool = True,
) -> Analysis:
    """Find all divergent start dates in ``[start, end]``.

    ``deltas`` are applied in the given order for the ``sequential`` form.
    The ``merged`` form uses ``relativedelta`` addition (``a + b``), and the
    ``swapped`` form applies the same deltas in reverse order.

    For rules made only of relative fields, the Gregorian calendar and
    ``relativedelta`` behavior repeat every 400 years.  Ranges longer than one
    cycle are resolved by scanning one canonical cycle and tiling its runs,
    rather than scanning every day of every cycle.

    :param start: First start date to test.
    :param end: Last start date to test, inclusive.
    :param deltas: Two or more :class:`relativedelta` instances.
    :param use_cycle: Set to ``False`` to force a literal day-by-day scan.
    """
    deltas = tuple(deltas)
    if len(deltas) < 2:
        raise ValueError("at least two relativedelta instances are required")
    if any(not isinstance(delta, relativedelta) for delta in deltas):
        raise TypeError("deltas must contain relativedelta instances")
    _validate_range(start, end)

    merged = _merge(deltas)
    periodic = _is_periodic_rule(deltas)
    optimized = (
        use_cycle
        and periodic
        and (end.toordinal() - start.toordinal() + 1)
        > _GREGORIAN_CYCLE_DAYS
    )

    if optimized:
        runs = _periodic_runs(start, end, deltas, merged)
        if runs is None:
            runs = _scan_runs(start, end, deltas, merged)
            optimized = False
    else:
        runs = _scan_runs(start, end, deltas, merged)

    intervals = tuple(
        _build_interval(
            run_start,
            run_end,
            label,
            start,
            deltas,
            merged,
        )
        for run_start, run_end, label in runs
    )
    return Analysis(
        start=start,
        end=end,
        deltas=deltas,
        periodic=periodic,
        mode="cycle" if optimized else "scan",
        intervals=intervals,
    )


def find_divergences(
    start: date,
    end: date,
    deltas: Iterable[relativedelta],
    *,
    use_cycle: bool = True,
) -> tuple[DivergenceInterval, ...]:
    """Return only the intervals from :func:`analyze_range`."""
    return analyze_range(
        start, end, deltas, use_cycle=use_cycle
    ).intervals


_WEEKDAY_RE = re.compile(
    r"^(MO|TU|WE|TH|FR|SA|SU)(?:\(([+-]?\d+)\))?$"
)
_INTEGER_FIELDS = {
    "years",
    "months",
    "days",
    "leapdays",
    "weeks",
    "hours",
    "minutes",
    "seconds",
    "microseconds",
}
_VALUE_FIELDS = set(_DELTA_FIELDS) | {"weeks"}


def _parse_value(name: str, text: str):
    text = text.strip()
    if name == "weekday":
        match = _WEEKDAY_RE.match(text)
        if match:
            weekday_cls = {
                "MO": 0,
                "TU": 1,
                "WE": 2,
                "TH": 3,
                "FR": 4,
                "SA": 5,
                "SU": 6,
            }[match.group(1)]
            n = int(match.group(2)) if match.group(2) else None
            return weekday(weekday_cls, n)
        return int(text)
    try:
        return int(text)
    except ValueError:
        return float(text)


def parse_delta(spec: str) -> relativedelta:
    """Build a :class:`relativedelta` from a CLI-friendly string.

    Examples: ``months=1``, ``day=31,weekday=FR(-1)``, or a JSON object such
    as ``{"months": 1, "days": 2}``.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty relativedelta specification")
    if spec.startswith("{"):
        values = json.loads(spec)
        if not isinstance(values, dict):
            raise ValueError("JSON delta must be an object")
    else:
        values = {}
        for item in spec.split(","):
            if "=" not in item:
                raise ValueError(
                    "delta items must have the form name=value: %r" % item
                )
            name, text = item.split("=", 1)
            name = name.strip()
            if name not in _VALUE_FIELDS:
                raise ValueError("unknown relativedelta field %r" % name)
            values[name] = _parse_value(name, text)

    normalized = {}
    for name, value in values.items():
        if name not in _VALUE_FIELDS:
            raise ValueError("unknown relativedelta field %r" % name)
        if name == "weekday" and isinstance(value, str):
            value = _parse_value(name, value)
        elif name in _INTEGER_FIELDS and not isinstance(value, int):
            value = int(value)
        normalized[name] = value
    return relativedelta(**normalized)


def _parse_date(text: str):
    if "T" in text or " " in text:
        return datetime.fromisoformat(text)
    return date.fromisoformat(text)


def _write_days(analysis: Analysis, deltas, merged) -> None:
    template = analysis.start
    for ordinal in range(
        analysis.start.toordinal(), analysis.end.toordinal() + 1
    ):
        current = _template_date(template, ordinal)
        label = _classify(current, deltas, merged)
        if label is None:
            continue
        partition, causes = label
        results = _formulations(current, deltas, merged)[0]
        print(
            json.dumps(
                {
                    "date": current.isoformat(),
                    "causes": list(causes),
                    "differing_pairs": list(_differing_pairs(partition)),
                    "results": {
                        name: _isoformat_value(value)
                        for name, value in zip(_FORM_NAMES, results)
                    },
                },
                sort_keys=True,
            )
        )


def _human_line(interval: DivergenceInterval) -> str:
    causes = ",".join(interval.causes)
    pairs = ",".join(interval.differing_pairs)
    return "%s..%s (%d day%s) causes=%s differs=%s" % (
        interval.start.isoformat(),
        interval.end.isoformat(),
        interval.days,
        "" if interval.days == 1 else "s",
        causes,
        pairs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find dates where sequential, merged and reversed "
            "relativedelta application disagree."
        )
    )
    parser.add_argument("--start", required=True, metavar="DATE")
    parser.add_argument("--end", required=True, metavar="DATE")
    parser.add_argument(
        "--delta",
        action="append",
        required=True,
        metavar="SPEC",
        help=(
            "relativedelta specification; repeat once per delta. "
            "Example: --delta 'months=1' --delta 'day=31'"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "jsonl"),
        default="text",
    )
    parser.add_argument(
        "--days",
        action="store_true",
        help="emit every divergent date instead of grouped intervals",
    )
    parser.add_argument(
        "--no-cycle",
        action="store_true",
        help="disable 400-year tiling for purely relative rules",
    )
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="exit with status 1 if any divergent date is found",
    )
    args = parser.parse_args(argv)

    try:
        start = _parse_date(args.start)
        end = _parse_date(args.end)
        deltas = tuple(parse_delta(spec) for spec in args.delta)
        analysis = analyze_range(
            start,
            end,
            deltas,
            use_cycle=not args.no_cycle,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.days:
        _write_days(analysis, deltas, _merge(deltas))
    elif args.format == "json":
        print(json.dumps(analysis.as_dict(), indent=2, sort_keys=True))
    elif args.format == "jsonl":
        for interval in analysis.intervals:
            print(json.dumps(interval.as_dict(), sort_keys=True))
    else:
        print(
            "periodic=%s mode=%s divergent_days=%d intervals=%d"
            % (
                str(analysis.periodic).lower(),
                analysis.mode,
                analysis.divergent_days,
                analysis.interval_count,
            )
        )
        for interval in analysis.intervals:
            print(_human_line(interval))

    if args.fail_on_divergence and analysis.intervals:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
