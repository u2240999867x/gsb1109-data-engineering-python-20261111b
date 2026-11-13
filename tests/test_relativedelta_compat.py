"""Executable compatibility regressions for composed relativedelta rules."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timedelta

import pytest

from dateutil.relativedelta import TU, FR, relativedelta
from dateutil.relativedelta_compat import (
    CAUSE_ABSOLUTE_FIELD,
    CAUSE_LEAP_DAY,
    CAUSE_MONTH_END,
    CAUSE_WEEKDAY,
    analyze_range,
    find_divergences,
    main,
    parse_delta,
)


def _forms(start, deltas):
    sequential = start
    for delta in deltas:
        sequential = sequential + delta
    merged = start + (deltas[0] + deltas[1])
    swapped = start
    for delta in reversed(deltas):
        swapped = swapped + delta
    return sequential, merged, swapped


def _brute_force_divergent_days(start, end, deltas):
    found = []
    current = start
    while current <= end:
        results = _forms(current, deltas)
        if not results[0] == results[1] == results[2]:
            found.append(current)
        current += timedelta(days=1)
    return found


def _range_days(intervals):
    days = set()
    for interval in intervals:
        current = interval.start
        while current <= interval.end:
            days.add(current)
            current += timedelta(days=1)
    return days


def test_repeated_month_delta_reports_month_end_clamp():
    intervals = find_divergences(
        date(2021, 1, 1),
        date(2021, 12, 31),
        [relativedelta(months=1), relativedelta(months=1)],
    )

    assert [(item.start, item.end) for item in intervals] == [
        (date(2021, 1, 29), date(2021, 1, 31)),
        (date(2021, 3, 31), date(2021, 3, 31)),
        (date(2021, 5, 31), date(2021, 5, 31)),
        (date(2021, 8, 31), date(2021, 8, 31)),
        (date(2021, 10, 31), date(2021, 10, 31)),
    ]
    for interval in intervals:
        assert interval.causes == (CAUSE_MONTH_END,)

    assert intervals[0].first_results == (
        date(2021, 3, 28),
        date(2021, 3, 29),
        date(2021, 3, 28),
    )
    assert intervals[0].last_results == (
        date(2021, 3, 28),
        date(2021, 3, 31),
        date(2021, 3, 28),
    )


def test_repeated_month_delta_does_not_extrapolate_january_to_whole_year():
    intervals = find_divergences(
        date(2021, 1, 1),
        date(2021, 12, 31),
        [relativedelta(months=1), relativedelta(months=1)],
    )
    found = _range_days(intervals)

    assert date(2021, 1, 29) in found
    assert date(2021, 2, 1) not in found
    assert date(2021, 4, 1) not in found
    assert date(2021, 10, 31) in found


def test_merged_two_month_rule_misses_month_end_clamp():
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    month = relativedelta(months=1)
    pair_days = _range_days(find_divergences(start, end, [month, month]))

    assert date(2021, 1, 31) in pair_days
    assert date(2021, 3, 31) + month + month == date(2021, 5, 30)
    assert date(2021, 3, 31) + relativedelta(months=2) == date(2021, 5, 31)
    assert date(2021, 1, 31) + month + month == date(2021, 3, 28)
    assert date(2021, 1, 31) + relativedelta(months=2) == date(2021, 3, 31)


def test_absolute_year_and_month_are_both_applied_in_merged_form():
    deltas = [relativedelta(year=2020), relativedelta(month=2)]
    interval = [
        item
        for item in find_divergences(
            date(2021, 5, 1), date(2021, 5, 31), deltas
        )
        if item.start <= date(2021, 5, 31) <= item.end
    ][0]

    assert CAUSE_ABSOLUTE_FIELD in interval.causes
    assert interval.first_results[0] == date(2020, 2, 29)
    assert interval.first_results[1] == date(2020, 2, 29)
    assert interval.first_results[2] == date(2020, 2, 28)


def test_weekday_is_applied_to_the_already_shifted_day():
    deltas = [relativedelta(months=1), relativedelta(weekday=TU)]
    interval = [
        item
        for item in find_divergences(
            date(2021, 1, 1), date(2021, 1, 31), deltas
        )
        if item.start <= date(2021, 1, 1) <= item.end
    ][0]

    assert interval.causes == (CAUSE_WEEKDAY,)
    assert interval.first_results[0] == date(2021, 2, 2)
    assert interval.first_results[1] == date(2021, 2, 2)
    assert interval.first_results[2] == date(2021, 2, 5)


def test_leap_day_start_is_distinguished_from_ordinary_clamp():
    deltas = [relativedelta(years=1), relativedelta(months=1)]
    intervals = find_divergences(
        date(2019, 1, 1), date(2021, 12, 31), deltas
    )

    assert intervals == find_divergences(
        date(2019, 1, 1),
        date(2021, 12, 31),
        deltas,
        use_cycle=False,
    )
    labeled = {interval.start: interval.causes for interval in intervals}

    assert labeled[date(2020, 2, 29)] == (CAUSE_LEAP_DAY, CAUSE_MONTH_END)
    assert labeled[date(2019, 1, 29)] == (CAUSE_MONTH_END,)

    ordinary = find_divergences(
        date(2019, 1, 1),
        date(2019, 12, 31),
        [relativedelta(months=1), relativedelta(months=1)],
    )
    assert ordinary
    assert all(
        CAUSE_LEAP_DAY not in interval.causes for interval in ordinary
    )


def test_absolute_year_before_month_override():
    deltas = [relativedelta(year=2020), relativedelta(month=2)]
    intervals = find_divergences(
        date(2019, 1, 1), date(2021, 12, 31), deltas
    )

    assert intervals
    assert all(CAUSE_ABSOLUTE_FIELD in item.causes for item in intervals)
    boundary = [
        item for item in intervals if item.start == date(2021, 1, 29)
    ][0]
    assert boundary.end == date(2021, 1, 31)
    assert boundary.differing_pairs == (
        "sequential!=swapped",
        "merged!=swapped",
    )


def test_weekday_after_month_rule_is_labeled():
    delta_month = relativedelta(months=1)
    delta_weekday = relativedelta(weekday=TU)
    intervals = find_divergences(
        date(2021, 1, 1), date(2021, 12, 31), [delta_month, delta_weekday]
    )

    assert intervals
    found = _range_days(intervals)
    assert date(2021, 1, 1) in found
    assert date(2021, 2, 1) not in found
    assert all(
        CAUSE_WEEKDAY in item.causes
        for item in intervals
        if date(2021, 1, 1) <= item.start <= date(2021, 1, 10)
    )
    january_first = [
        item
        for item in intervals
        if item.start <= date(2021, 1, 1) <= item.end
    ][0]
    assert date(2021, 1, 1) + delta_month + delta_weekday == date(2021, 2, 2)
    assert date(2021, 1, 1) + (delta_month + delta_weekday) == date(2021, 2, 2)
    assert date(2021, 1, 1) + delta_weekday + delta_month == date(2021, 2, 5)


def test_leapdays_ablation_is_labeled_leap_day():
    deltas = [relativedelta(leapdays=1), relativedelta(months=1)]
    intervals = find_divergences(
        date(2019, 1, 1), date(2021, 12, 31), deltas
    )

    assert intervals
    assert any(CAUSE_LEAP_DAY in item.causes for item in intervals)
    assert intervals[0].start == date(2020, 2, 1)


def test_prefix_boundaries_match_brute_force():
    deltas = [relativedelta(months=1), relativedelta(months=1)]
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    expected = set(_brute_force_divergent_days(start, end, deltas))
    analysis = analyze_range(start, end, deltas)
    found = _range_days(analysis.intervals)

    assert found == expected
    assert analysis.divergent_days == len(expected)

    for ordinal in range(start.toordinal(), end.toordinal() + 1):
        current = date.fromordinal(ordinal)
        present = any(
            item.start <= current <= item.end for item in analysis.intervals
        )
        assert present == (current in expected)


def test_interval_end_is_inclusive_and_maximal():
    deltas = [relativedelta(months=1), relativedelta(months=1)]
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    intervals = find_divergences(start, end, deltas)
    expected = set(_brute_force_divergent_days(start, end, deltas))

    for interval in intervals:
        assert interval.start in expected
        assert interval.end in expected
        assert interval.start - timedelta(days=1) not in expected
        assert interval.end + timedelta(days=1) not in expected


def test_cycle_tiling_matches_prefix_scans_over_400_years():
    deltas = [relativedelta(months=1), relativedelta(months=1)]
    whole = analyze_range(date(2000, 1, 1), date(2400, 12, 31), deltas)
    prefix = analyze_range(
        date(2000, 1, 1), date(2400, 1, 31), deltas
    )
    suffix = analyze_range(
        date(2400, 2, 1), date(2400, 12, 31), deltas
    )

    assert whole.mode == "cycle"
    whole_days = _range_days(whole.intervals)
    prefix_days = _range_days(prefix.intervals)
    suffix_days = _range_days(suffix.intervals)
    assert whole_days == prefix_days | suffix_days
    assert whole.divergent_days == len(whole_days)


def test_cycle_output_does_not_double_count_across_or_inside_bounds():
    deltas = [relativedelta(months=1), relativedelta(months=1)]
    analysis = analyze_range(date(2100, 3, 1), date(2550, 3, 1), deltas)
    assert analysis.mode == "cycle"
    days = _range_days(analysis.intervals)
    assert analysis.divergent_days == len(days)


def test_literal_scan_can_be_forced():
    deltas = [relativedelta(months=1), relativedelta(months=1)]
    optimized = analyze_range(date(2100, 1, 1), date(2550, 1, 1), deltas)
    literal = analyze_range(
        date(2100, 1, 1),
        date(2550, 1, 1),
        deltas,
        use_cycle=False,
    )

    assert optimized.mode == "cycle"
    assert literal.mode == "scan"
    assert optimized.intervals == literal.intervals


def test_datetime_range_preserves_time_and_supports_cycle_mode():
    delta = relativedelta(months=1)
    deltas = [delta, delta]
    start = datetime(2021, 1, 1, 9, 30, 15)
    end = datetime(2021, 12, 31, 9, 30, 15)
    analysis = analyze_range(start, end, deltas)

    assert analysis.intervals
    assert analysis.intervals[0].start == datetime(2021, 1, 29, 9, 30, 15)
    assert analysis.intervals[0].end == datetime(2021, 1, 31, 9, 30, 15)
    assert analysis.divergent_days == len(_range_days(analysis.intervals))

    start = datetime(2100, 1, 1, 9, 30, 15)
    end = datetime(2550, 1, 1, 9, 30, 15)
    wide = analyze_range(start, end, deltas)
    assert wide.mode == "cycle"
    assert wide.divergent_days == len(_range_days(wide.intervals))


def test_datetime_range_rejects_mismatched_time_of_day():
    with pytest.raises(ValueError):
        analyze_range(
            datetime(2021, 1, 1, 9),
            datetime(2021, 12, 31, 10),
            [relativedelta(months=1), relativedelta(months=1)],
        )


def test_parse_delta_and_cli_json(capsys):
    spec = "year=2020,month=2"
    delta = parse_delta(spec)
    assert delta.year == 2020
    assert delta.month == 2
    assert parse_delta('{"months": 1, "day": 31}') == relativedelta(
        months=1, day=31
    )
    assert parse_delta("weekday=TU") == relativedelta(weekday=TU)
    assert parse_delta("weekday=FR(-1)") == relativedelta(weekday=FR(-1))

    status = main(
        [
            "--start",
            "2021-01-01",
            "--end",
            "2021-12-31",
            "--delta",
            "months=1",
            "--delta",
            "months=1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["interval_count"] == 5
    assert payload["intervals"][0]["start"] == "2021-01-29"
    assert payload["intervals"][0]["end"] == "2021-01-31"
    assert payload["intervals"][0]["causes"] == [CAUSE_MONTH_END]


def test_cli_fail_on_divergence_and_days(capsys):
    status = main(
        [
            "--start",
            "2021-01-01",
            "--end",
            "2021-01-31",
            "--delta",
            "months=1",
            "--delta",
            "months=1",
            "--days",
            "--fail-on-divergence",
        ]
    )
    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert status == 1
    assert [row["date"] for row in rows] == [
        "2021-01-29",
        "2021-01-30",
        "2021-01-31",
    ]


def test_module_cli_runs_as_executable():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dateutil.relativedelta_compat",
            "--start",
            "2021-03-01",
            "--end",
            "2021-03-31",
            "--delta",
            "months=1",
            "--delta",
            "months=1",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["intervals"][0]["start"] == "2021-03-31"


def test_single_delta_is_rejected():
    with pytest.raises(ValueError):
        analyze_range(
            date(2021, 1, 1), date(2021, 1, 1), [relativedelta(months=1)]
        )


def test_invalid_parse_spec_rejected():
    with pytest.raises(ValueError):
        parse_delta("months=1,bogus=2")
