from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from sortedcollections import (
    NEAREST,
    NEXT,
    PREVIOUS,
    NearestDict,
    NoCandidate,
    ScheduleAlignment,
    ToleranceExceeded,
    ordinary_linear_nearest,
)

NY = ZoneInfo('America/New_York')


def dt(year, month, day, hour=0, minute=0, fold=0, tzinfo=NY):
    return datetime(year, month, day, hour, minute, fold=fold, tzinfo=tzinfo)


def mark(hour, minute=0, fold=0):
    # Reference keys are authored on an arbitrary ordinary January day; only
    # their wall time, zone and fold are projected onto the record's day.
    return dt(2024, 1, 1, hour, minute, fold=fold)


def test_nearest_aligns_on_wall_time_and_preserves_evidence():
    references = NearestDict({mark(8): 'open', mark(10): 'close'})
    alignment = ScheduleAlignment(references)
    record = dt(2024, 11, 4, 9, 30)  # EST, UTC-05:00

    result = alignment.align(record)

    assert result.key == mark(10)
    assert result.wall_time == dt(2024, 11, 4, 10)
    assert result.distance == timedelta(minutes=30)
    assert result.date == record.date()
    assert result.offset == timedelta(hours=-5)
    assert result.fold == 0


def test_previous_and_next_rounding():
    references = NearestDict({mark(8): 'open', mark(10): 'close'})
    record = dt(2024, 11, 4, 9, 30)

    previous = ScheduleAlignment(references, rounding=PREVIOUS).align(record)
    following = ScheduleAlignment(references, rounding=NEXT).align(record)

    assert previous.key == mark(8)
    assert following.key == mark(10)


def test_local_day_only_ignores_adjacent_days():
    references = NearestDict({mark(23): 'late'})
    record = dt(2024, 11, 4, 0, 15)

    result = ScheduleAlignment(references).align(record)

    # The only same-day candidate is 23:00 on the record's own day.
    assert result.wall_time.date() == record.date()
    assert result.distance == timedelta(hours=22, minutes=45)


def test_cross_day_allows_previous_and_next_day_marks():
    references = NearestDict({mark(23): 'late'})
    record = dt(2024, 11, 4, 0, 15)

    result = ScheduleAlignment(
        references, allow_cross_day=True
    ).align(record)

    # The previous evening's 23:00 mark is only 75 minutes away in true time.
    assert result.wall_time == dt(2024, 11, 3, 23)
    assert result.distance == timedelta(hours=1, minutes=15)


def test_tolerance_keeps_close_marks_and_reports_far_ones_separately():
    references = NearestDict({mark(8): 'open', mark(10): 'close'})
    record = dt(2024, 11, 4, 9, 30)

    within = ScheduleAlignment(
        references, tolerance=timedelta(hours=1)
    ).align(record)
    assert within.key == mark(10)  # 08:00 is 90 minutes away and filtered.

    with pytest.raises(ToleranceExceeded):
        ScheduleAlignment(
            references, tolerance=timedelta(minutes=15)
        ).align(record)


def test_no_candidate_is_distinct_from_tolerance_failure():
    record = dt(2024, 11, 4, 9, 30)

    with pytest.raises(NoCandidate):
        ScheduleAlignment(NearestDict()).align(record)

    # The only mark lands inside the spring-forward gap and is skipped.
    gap_references = NearestDict({mark(2, 30): 'ghost'})
    with pytest.raises(NoCandidate):
        ScheduleAlignment(gap_references).align(dt(2024, 3, 10, 4))


def test_spring_forward_gap_mark_is_skipped_but_real_marks_remain():
    references = NearestDict(
        {mark(1): 'real', mark(2, 30): 'ghost'}
    )
    record = dt(2024, 3, 10, 4)  # EDT after the 02:00 -> 03:00 jump

    result = ScheduleAlignment(references).align(record)

    assert result.key == mark(1)


def test_fall_back_fold_names_two_distinct_instants():
    references = NearestDict({mark(1, 15): 'shift'})  # fold=0, EDT
    earlier = dt(2024, 11, 3, 1, 30, fold=0)  # 05:30 UTC
    later = dt(2024, 11, 3, 1, 30, fold=1)  # 06:30 UTC

    first = ScheduleAlignment(references).align(earlier)
    second = ScheduleAlignment(references).align(later)

    assert first.distance == timedelta(minutes=15)
    assert second.distance == timedelta(minutes=75)
    assert first.offset == timedelta(hours=-4)
    assert second.offset == timedelta(hours=-5)
    assert first.fold == 0 and second.fold == 1


def test_direction_uses_true_instant_not_wall_order_across_fold():
    # 01:15 fold=1 (06:15 UTC) is later than 01:30 fold=0 (05:30 UTC) even
    # though its clock reading is earlier.
    references = NearestDict(
        {mark(0, 30): 'early', mark(1, 15, fold=1): 'late'}
    )
    record = dt(2024, 11, 3, 1, 30, fold=0)

    previous = ScheduleAlignment(
        references, rounding=PREVIOUS
    ).align(record)
    following = ScheduleAlignment(
        references, rounding=NEXT
    ).align(record)

    assert previous.key == mark(0, 30)
    assert following.key == mark(1, 15, fold=1)


def test_previous_and_next_raise_when_nothing_on_that_side():
    record = dt(2024, 11, 4, 9, 30)
    only_later = NearestDict({mark(1, 15, fold=1): 'late'})
    only_earlier = NearestDict({mark(8): 'open'})

    with pytest.raises(NoCandidate):
        ScheduleAlignment(only_later, rounding=PREVIOUS).align(
            dt(2024, 11, 3, 1, 30, fold=0)
        )
    with pytest.raises(NoCandidate):
        ScheduleAlignment(only_earlier, rounding=NEXT).align(record)


def test_nearest_tie_prefers_later_instant():
    references = NearestDict({mark(9): 'a', mark(10): 'b'})
    record = dt(2024, 11, 4, 9, 30)

    result = ScheduleAlignment(references, rounding=NEAREST).align(record)

    assert result.key == mark(10)


def test_wall_closer_can_be_truly_farther_across_fall_back():
    # A is 15 minutes away on the clock but 45 true minutes (second pass of
    # the repeated hour); B is 40 minutes on the clock and 40 true minutes.
    mark_a = mark(1, 15, fold=1)  # 06:15 UTC
    mark_b = mark(0, 50)  # 04:50 UTC
    references = NearestDict({mark_a: 'A', mark_b: 'B'})
    record = dt(2024, 11, 3, 1, 30, fold=0)  # 05:30 UTC

    true_choice = ScheduleAlignment(references).align(record)
    linear_choice = ordinary_linear_nearest(references, record)

    assert true_choice.key == mark_b
    assert true_choice.distance == timedelta(minutes=40)
    assert linear_choice == mark_a
    # The stock NearestDict linear lookup keeps its existing wall behaviour.
    assert NearestDict(references)[record] == 'A'


def test_ordinary_linear_nearest_empty_raises():
    with pytest.raises(KeyError):
        ordinary_linear_nearest(NearestDict(), dt(2024, 11, 4, 9))


def test_true_distance_matches_utc_elapsed_seconds():
    references = NearestDict({mark(1, 15): 'shift'})
    record = dt(2024, 11, 3, 1, 30, fold=1)

    result = ScheduleAlignment(references).align(record)
    elapsed = abs(
        result.wall_time.astimezone(timezone.utc)
        - record.astimezone(timezone.utc)
    )

    assert result.distance == elapsed


def test_candidates_omit_gap_and_apply_tolerance():
    references = NearestDict(
        {mark(1): 'real', mark(2, 30): 'ghost', mark(10): 'far'}
    )
    alignment = ScheduleAlignment(
        references, tolerance=timedelta(hours=2)
    )
    record = dt(2024, 3, 10, 4)

    filtered = alignment.candidates(record)
    assert [key for _, _, key in filtered] == [mark(1)]

    unrestricted = alignment.candidates(record, apply_tolerance=False)
    # The gap mark is still absent; only the far but real mark is restored.
    assert [key for _, _, key in unrestricted] == [mark(1), mark(10)]


def test_rounding_constants_match_nearestdict_modes():
    assert PREVIOUS == NearestDict.NEAREST_PREV
    assert NEAREST == NearestDict.NEAREST
    assert NEXT == NearestDict.NEAREST_NEXT