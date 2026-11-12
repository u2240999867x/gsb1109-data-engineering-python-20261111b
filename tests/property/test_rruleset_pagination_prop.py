"""Reconcile ``rruleset`` pagination against an independent baseline.

A calendar view is partitioned by UTC date and expanded from an
``rruleset``: a preview page takes the first three instances, then
background pagination interleaves ``between``/``after`` queries to fetch
further pages.  These tests expand the same schedule independently --
enumerating the calendar conditions over a finite date domain by hand,
including RDATE/RRULE overlaps removed by EXDATE and COUNT/UNTIL
truncation -- and then exhaustively run short query sequences with the
cache toggled on and off.  Every returned instance is reconciled against
the baseline set, its UTC partition and its repetition count.

When the cache toggle produces a divergence, the query sequence is
shrunk to a minimal form that still distinguishes the three failure
classes:

* ``CURSOR_ADVANCE`` -- the divergence only manifests when earlier
  queries ran first (iterator/cache state was advanced incorrectly);
* ``SET_DEDUP`` -- an instance is returned more than once;
* ``TIME_BOUNDARY`` -- a single query misinterprets its time bounds
  (interval endpoints, UNTIL or a UTC partition edge).
"""

import itertools
from collections import Counter, namedtuple
from datetime import date, datetime, time, timedelta
from enum import Enum

import pytest

from dateutil import tz
from dateutil.rrule import DAILY, WEEKLY, rrule, rruleset, weekday

DOMAIN_START = date(2026, 1, 1)
# Far enough out that COUNT-limited sparse weekly rules terminate inside
# the domain; UNTIL-limited schedules truncate earlier than this.
DOMAIN_END = date(2026, 3, 15)

Spec = namedtuple(
    "Spec",
    [
        "name",
        "freq",
        "interval",
        "byweekday",
        "dtstart",
        "count",
        "until",
        "rdates",
        "exdates",
    ],
)


class FailureClass(Enum):
    CURSOR_ADVANCE = "cursor-advance"
    SET_DEDUP = "set-dedup"
    TIME_BOUNDARY = "time-boundary"


def rule_occurrences(spec):
    """Expand the RRULE part of a schedule over the finite date domain.

    This is deliberately independent of ``dateutil.rrule``'s iterator:
    candidate days are stepped one by one and matched against the
    calendar conditions, then COUNT and UNTIL truncate the result.
    """
    start = spec.dtstart.date()
    week_start = start - timedelta(days=start.weekday())
    occurrences = []
    day = start
    # One day of slack: near a UTC partition edge the last in-domain
    # instant can fall on the next local date (and vice versa).
    while day <= DOMAIN_END + timedelta(days=1):
        if spec.freq == DAILY:
            match = (day - start).days % spec.interval == 0
        else:
            week = (day - week_start).days // 7
            match = (
                week % spec.interval == 0
                and day.weekday() in spec.byweekday
            )
        if match:
            candidate = datetime.combine(day, spec.dtstart.timetz())
            if spec.until is not None and candidate > spec.until:
                break
            occurrences.append(candidate)
            if spec.count is not None and len(occurrences) >= spec.count:
                break
        day += timedelta(days=1)
    return occurrences


def baseline(spec):
    """The expected recurrence set: (RRULE | RDATE) - EXDATE, sorted."""
    included = set(rule_occurrences(spec)) | set(spec.rdates)
    return sorted(included - set(spec.exdates))


def make_schedule(name, freq, interval, tzinfo, start_time, limit):
    dtstart = datetime.combine(
        DOMAIN_START + timedelta(days=1), start_time, tzinfo=tzinfo
    )
    count = until = None
    if limit == "count":
        count = 5
    elif limit == "until":
        until = datetime(2026, 1, 20, 12, 0, tzinfo=tz.UTC)
    else:
        # Keep the expansion finite: bound it at the edge of the date
        # domain without truncating any instance inside it.
        until = datetime(2026, 2, 2, 0, 0, tzinfo=tz.UTC)
    byweekday = (0, 2, 4) if freq == WEEKLY else None
    spec = Spec(
        name, freq, interval, byweekday, dtstart, count, until, (), ()
    )
    occurrences = rule_occurrences(spec)
    # An RDATE off the rule pattern and an EXDATE matching nothing.
    rdates = [dtstart + timedelta(hours=5, minutes=15)]
    exdates = [dtstart + timedelta(days=9, minutes=7)]
    if len(occurrences) > 1:
        # An RDATE duplicating an RRULE instance, removed by EXDATE.
        rdates.append(occurrences[1])
        exdates.append(occurrences[1])
    if len(occurrences) > 3:
        exdates.append(occurrences[3])
    return spec._replace(rdates=tuple(rdates), exdates=tuple(exdates))


_TZ_VARIANTS = [
    ("utc", tz.UTC, time(12, 0)),
    # Local midnight half-hours that fall on a different UTC date.
    ("ist", tz.tzoffset("IST", 19800), time(0, 30)),
    ("nst", tz.tzoffset("NST", -12600), time(23, 30)),
]

SCHEDULES = [
    make_schedule(
        name="{}-i{}-{}-{}".format(freq_name, interval, tz_name, limit),
        freq=freq,
        interval=interval,
        tzinfo=tzinfo,
        start_time=start_time,
        limit=limit,
    )
    for (freq, freq_name) in [(DAILY, "daily"), (WEEKLY, "weekly")]
    for interval in (1, 3)
    for (tz_name, tzinfo, start_time) in _TZ_VARIANTS
    for limit in ("domain", "count", "until")
]


def build_rruleset(spec, cache):
    kwargs = {
        "freq": spec.freq,
        "dtstart": spec.dtstart,
        "interval": spec.interval,
        "wkst": 0,
    }
    if spec.byweekday is not None:
        kwargs["byweekday"] = [weekday(day) for day in spec.byweekday]
    if spec.count is not None:
        kwargs["count"] = spec.count
    if spec.until is not None:
        kwargs["until"] = spec.until
    rset = rruleset(cache=cache)
    rset.rrule(rrule(**kwargs))
    for rdate in spec.rdates:
        rset.rdate(rdate)
    for exdate in spec.exdates:
        rset.exdate(exdate)
    return rset


def schedule_ops(spec, base):
    """The query vocabulary for one schedule, with probes on instances.

    ``p1``/``p2`` sit exactly on recurrence instances so that the
    inclusive/exclusive flavours of ``after``/``between`` exercise the
    interval endpoints.
    """
    p1 = base[0] if base else spec.dtstart
    p2 = (
        base[len(base) // 2]
        if len(base) > 1
        else spec.dtstart + timedelta(days=2)
    )
    return [
        ("preview",),
        ("count",),
        ("after", p1, True),
        ("after", p1, False),
        ("after", p2, True),
        ("between", p1, p2, True),
        ("between", p1, p2, False),
    ]


def query_sequences(ops):
    """All length-1/2 sequences plus preview-first length-3 sequences."""
    sequences = [[op] for op in ops]
    sequences += [list(pair) for pair in itertools.product(ops, repeat=2)]
    preview = ops[0]
    sequences += [
        [preview, second, third] for second in ops for third in ops
    ]
    return sequences


def run_op(rset, op):
    kind = op[0]
    if kind == "preview":
        return list(rset[:3])
    if kind == "count":
        return [rset.count()]
    if kind == "after":
        return [rset.after(op[1], inc=op[2])]
    if kind == "between":
        return list(rset.between(op[1], op[2], inc=op[3]))
    raise ValueError("unknown op: %r" % (op,))


def expected_op(base, op):
    kind = op[0]
    if kind == "preview":
        return base[:3]
    if kind == "count":
        return [len(base)]
    if kind == "after":
        _, dt, inc = op
        probe = (lambda x: x >= dt) if inc else (lambda x: x > dt)
        return [next((x for x in base if probe(x)), None)]
    if kind == "between":
        _, start, end, inc = op
        if inc:
            return [x for x in base if start <= x <= end]
        return [x for x in base if start < x < end]
    raise ValueError("unknown op: %r" % (op,))


def execute_sequence(spec, ops, cache, fault=None):
    rset = build_rruleset(spec, cache=cache)
    results = []
    for position, op in enumerate(ops):
        result = run_op(rset, op)
        if fault is not None:
            result = fault(ops, position, result, cache)
        results.append(result)
    return results


def utc_partition(dt):
    return dt.astimezone(tz.UTC).date()


def check_partitions(ops, results, base):
    """Reconcile each returned instance's UTC partition and repetition."""
    instants = set(base)
    partitions = Counter(utc_partition(x) for x in base)
    for op, result in zip(ops, results):
        seen = Counter()
        for item in result:
            if not isinstance(item, datetime):
                continue
            partition = utc_partition(item)
            assert item in instants, (
                "%r returned %r, which is not in the baseline set"
                % (op, item)
            )
            seen[partition] += 1
            assert seen[partition] <= partitions[partition], (
                "%r returned partition %s more times than the baseline"
                % (op, partition)
            )


Discrepancy = namedtuple(
    "Discrepancy", ["expected", "cached", "plain", "offender"]
)


def discrepancy(spec, base, ops, fault=None):
    """Run a sequence with the cache toggled and compare to the baseline."""
    expected = [expected_op(base, op) for op in ops]
    cached = execute_sequence(spec, ops, cache=True, fault=fault)
    plain = execute_sequence(spec, ops, cache=False, fault=fault)
    if cached == expected and plain == expected:
        return None
    offender = cached if cached != expected else plain
    return Discrepancy(expected, cached, plain, offender)


def classify_sequence(ops, expected, actual, base):
    """Classify the first diverging op of a sequence.

    Precedence: duplicated instances mean a set-dedup error; an instance
    outside the baseline set means a time-boundary error; a divergence
    that needs earlier queries to manifest means a cursor-advance error;
    a single self-diverging query is a time-boundary error.
    """
    instants = set(base)
    for op, exp, act in zip(ops, expected, actual):
        if exp == act:
            continue
        items = [x for x in act if isinstance(x, datetime)]
        if len(set(items)) != len(items):
            return FailureClass.SET_DEDUP
        if any(x not in instants for x in items):
            return FailureClass.TIME_BOUNDARY
        if len(ops) > 1:
            return FailureClass.CURSOR_ADVANCE
        return FailureClass.TIME_BOUNDARY
    return None


def classify_at(spec, base, ops, fault=None):
    disc = discrepancy(spec, base, ops, fault)
    if disc is None:
        return None
    return classify_sequence(ops, disc.expected, disc.offender, base)


def _greedy_shrink(seq, pred):
    seq = list(seq)
    changed = True
    while changed:
        changed = False
        for i in range(len(seq)):
            candidate = seq[:i] + seq[i + 1:]
            if candidate and pred(candidate):
                seq = candidate
                changed = True
                break
    return seq


def minimize(spec, base, seq, fault=None):
    """Shrink a diverging sequence, preserving its failure class."""
    shrunk = _greedy_shrink(
        seq, lambda s: classify_at(spec, base, s, fault) is not None
    )
    cls = classify_at(spec, base, shrunk, fault)
    shrunk = _greedy_shrink(
        shrunk, lambda s: classify_at(spec, base, s, fault) is cls
    )
    return shrunk, cls


@pytest.mark.rruleset
@pytest.mark.parametrize(
    "spec", SCHEDULES, ids=[spec.name for spec in SCHEDULES]
)
def test_rruleset_pagination_reconciliation(spec):
    base = baseline(spec)
    # Reconcile the independent baseline against full expansion first.
    probe = build_rruleset(spec, cache=False)
    assert list(probe) == base
    assert probe.count() == len(base)

    ops = schedule_ops(spec, base)
    for seq in query_sequences(ops):
        expected = [expected_op(base, op) for op in seq]
        cached = execute_sequence(spec, seq, cache=True)
        plain = execute_sequence(spec, seq, cache=False)
        if cached != expected or plain != expected:
            minimal, cls = minimize(spec, base, seq)
            disc = discrepancy(spec, base, minimal)
            pytest.fail(
                "rruleset pagination divergence (%s)\n"
                "schedule: %s\n"
                "minimal sequence: %r\n"
                "expected: %r\ncached: %r\nuncached: %r"
                % (
                    cls.value,
                    spec.name,
                    minimal,
                    disc.expected,
                    disc.cached,
                    disc.plain,
                ),
                pytrace=False,
            )
        check_partitions(seq, cached, base)
        check_partitions(seq, plain, base)


@pytest.mark.rruleset
def test_classifier_distinguishes_failure_classes():
    base = [
        datetime(2026, 1, 1, 12, tzinfo=tz.UTC) + timedelta(days=i)
        for i in range(4)
    ]
    between = ("between", base[0], base[2], True)
    # A duplicated instance is a set-dedup error.
    assert (
        classify_sequence(
            [between], [base[:3]], [base[:3] + [base[0]]], base
        )
        is FailureClass.SET_DEDUP
    )
    # An instance outside the baseline set is a time-boundary error.
    alien = base[0] + timedelta(hours=3)
    assert (
        classify_sequence([between], [base[:3]], [[base[0], alien]], base)
        is FailureClass.TIME_BOUNDARY
    )
    # A single query returning the wrong window of valid instances is a
    # time-boundary error.
    assert (
        classify_sequence([between], [base[:3]], [base[1:3]], base)
        is FailureClass.TIME_BOUNDARY
    )
    # The same wrong window behind a preview page is a cursor error.
    ops = [("preview",), between]
    expected = [base[:3], base[:3]]
    actual = [base[:3], base[1:3]]
    assert (
        classify_sequence(ops, expected, actual, base)
        is FailureClass.CURSOR_ADVANCE
    )


FAULT_SPEC = make_schedule(
    "fault-probe", DAILY, 1, tz.UTC, time(12, 0), "domain"
)


def _cursor_fault(ops, position, result, cache):
    # A stale cursor left by the preview page skips the first instance
    # of a later paginated query, but only with the cache enabled.
    if (
        cache
        and ops[position][0] == "between"
        and any(op[0] == "preview" for op in ops[:position])
    ):
        return result[1:]
    return result


def _dedup_fault(ops, position, result, cache):
    if cache and ops[position][0] == "between" and result:
        return result + [result[0]]
    return result


def _boundary_fault(ops, position, result, cache):
    # inc=True is misread as exclusive when dt is itself an instance.
    op = ops[position]
    if (
        cache
        and op[0] == "after"
        and op[2]
        and result
        and result[0] == op[1]
    ):
        return [None]
    return result


@pytest.mark.rruleset
@pytest.mark.parametrize(
    "fault, expected_class, minimal_length",
    [
        (_cursor_fault, FailureClass.CURSOR_ADVANCE, 2),
        (_dedup_fault, FailureClass.SET_DEDUP, 1),
        (_boundary_fault, FailureClass.TIME_BOUNDARY, 1),
    ],
)
def test_shrunk_divergence_distinguishes_faults(
    fault, expected_class, minimal_length
):
    base = baseline(FAULT_SPEC)
    ops = schedule_ops(FAULT_SPEC, base)
    preview, count, after_inc, between_inc = ops[0], ops[1], ops[2], ops[5]
    seq = [preview, between_inc, after_inc, count]
    assert classify_at(FAULT_SPEC, base, seq, fault) is not None

    minimal, cls = minimize(FAULT_SPEC, base, seq, fault=fault)
    assert cls is expected_class
    assert len(minimal) == minimal_length
    assert classify_at(FAULT_SPEC, base, minimal, fault) is cls
    # 1-minimality: no proper subsequence diverges with the same class.
    for i in range(len(minimal)):
        reduced = minimal[:i] + minimal[i + 1:]
        assert classify_at(FAULT_SPEC, base, reduced, fault) is not cls
