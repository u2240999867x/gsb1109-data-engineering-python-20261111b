"""Offline regressions for choosing between TZif sources of the same key.

When the system time zone files are replaced, whether local rows that were
already partitioned into UTC have to be recomputed cannot be decided from the
tzdata version number: two releases can describe the same zone identically
everywhere except for a finite interval in which some transitions move. The
answer has to come from the transitions themselves, compared as UTC
partitions.

Every source here is built by hand with ``construct_zone`` and installed in a
temporary TZPATH, so the tests need no network and no particular installed
version of the time zone database. Each pair of sources differs only inside a
limited window that contains a transition present in only one of the files:
a fall-back transition (one wall time, two UTC solutions) and a spring-forward
transition (a wall time with no UTC solution).
"""

import io
import os
from datetime import datetime, timedelta

import pytest

from dateutil import tz

from ._common import set_tzpath
from .test_tzfile import ZoneOffset, construct_zone

ZERO = timedelta(0)
ONE_H = timedelta(hours=1)
EPOCH = datetime(1970, 1, 1, tzinfo=tz.UTC)

KEY = "Fictional/Repartition"

# The same transition, scheduled at a different UTC instant in each source.
# Everything outside the interval between these two instants is identical.
T_OLD = 1_700_000_000
T_NEW = T_OLD + 2 * 86400

STD = ZoneOffset("STD", ZERO)
DST = ZoneOffset("DST", ONE_H, ONE_H)
NST = ZoneOffset("NST", -ONE_H)


@pytest.fixture(params=["fold", "gap"])
def source_pair(request):
    """Two TZif blobs differing by one transition inside a finite window."""
    if request.param == "fold":
        # +00:00 -> -01:00: the clock goes backwards, so a wall time repeats.
        alternative = NST
    else:
        # +00:00 -> +01:00: the clock jumps forwards, so a wall time is skipped.
        alternative = DST

    old_data = construct_zone([(T_OLD, STD, alternative)], after="").read()
    new_data = construct_zone([(T_NEW, STD, alternative)], after="").read()
    return old_data, new_data, request.param


def _install(root, key, data):
    path = root.joinpath(*key.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _offset_at(zone, utc_timestamp):
    dt_utc = EPOCH + timedelta(seconds=utc_timestamp)
    return dt_utc.astimezone(zone).utcoffset()


def _utc_difference_windows(old_zone, new_zone):
    """Exact half-open UTC windows where the two sources give different offsets.

    The timeline is cut at every transition known to either source, and each
    resulting interval is classified by sampling its interior, so the window
    boundaries are the transition instants themselves rather than probe times.
    """
    bounds = sorted(set(old_zone._trans_utc) | set(new_zone._trans_utc))
    intervals = [(None, bounds[0])]
    intervals.extend(zip(bounds, bounds[1:]))
    intervals.append((bounds[-1], None))

    windows = []
    start = None
    for lo, hi in intervals:
        if lo is None:
            probe = hi - 1
        elif hi is None:
            probe = lo + 1
        else:
            probe = (lo + hi) // 2

        differs = _offset_at(old_zone, probe) != _offset_at(new_zone, probe)
        if differs and start is None:
            start = lo
        elif not differs and start is not None:
            windows.append((start, lo))
            start = None
    if start is not None:
        windows.append((start, intervals[-1][0]))

    return windows


def _check_window_with_witnesses(old_zone, new_zone, start, end):
    """Witnesses on both sides of each transition boundary pin the window."""
    # Just outside: the sources agree.
    assert _offset_at(old_zone, start - 1) == _offset_at(
        new_zone, start - 1
    )
    assert _offset_at(old_zone, end) == _offset_at(new_zone, end)
    # Just inside: the sources disagree.
    assert _offset_at(old_zone, start) != _offset_at(new_zone, start)
    assert _offset_at(old_zone, end - 1) != _offset_at(new_zone, end - 1)


def _transitions(zone):
    """The loaded transitions as ``(utc_ts, offset_before, offset_after)``."""
    previous = zone._tti_before
    for ts, tti in zip(zone._trans_utc, zone._ttinfos):
        yield ts, previous.utcoff, tti.utcoff
        previous = tti


def _anomaly_segments(zone, windows):
    """Local wall-time segments made anomalous by a transition in ``windows``.

    Returns ``(kind, local_start, local_end)`` triples of seconds since the
    epoch in wall time, where ``kind`` is ``"ambiguous"`` (two UTC solutions)
    or ``"imaginary"`` (no UTC solution).
    """
    segments = []
    for ts, offset_before, offset_after in _transitions(zone):
        # The window endpoints are themselves the moved transitions, so a
        # transition at either boundary is part of the difference.
        if not any(start <= ts <= end for start, end in windows):
            continue

        local_start = ts + min(offset_before, offset_after).total_seconds()
        local_end = ts + max(offset_before, offset_after).total_seconds()
        kind = "ambiguous" if offset_before > offset_after else "imaginary"
        segments.append((kind, int(local_start), int(local_end)))

    return sorted(segments)


def _expected_anomaly_segments(kind):
    if kind == "fold":
        return [
            ("ambiguous", T_OLD - 3600, T_OLD),
            ("ambiguous", T_NEW - 3600, T_NEW),
        ]
    return [
        ("imaginary", T_OLD, T_OLD + 3600),
        ("imaginary", T_NEW, T_NEW + 3600),
    ]


def _local_images(zone, windows):
    """A UTC difference window expressed as wall times under ``zone``.

    The zone's offset is constant across the interior of each window, so the
    image of the half-open UTC interval is another half-open interval.
    """
    images = []
    for start, end in windows:
        offset = _offset_at(zone, (start + end) // 2)
        delta = int(offset.total_seconds())
        images.append((start + delta, end + delta))
    return images


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _contains(intervals, timestamp):
    return any(start <= timestamp < end for start, end in intervals)


def _utc_solutions(zone, wall_timestamp):
    """The valid UTC timestamps a wall-time row maps to in this zone."""
    naive = EPOCH + timedelta(seconds=wall_timestamp)
    dt = naive.replace(tzinfo=zone)
    if not tz.datetime_exists(dt):
        # A skipped wall time has no UTC solution in either fold.
        return frozenset()

    return frozenset(
        wall_timestamp
        - int(tz.enfold(dt, fold=fold).utcoffset().total_seconds())
        for fold in (0, 1)
    )


def _classify(zone, wall_timestamp):
    naive = EPOCH + timedelta(seconds=wall_timestamp)
    dt = naive.replace(tzinfo=zone)
    return tz.datetime_ambiguous(dt), tz.datetime_exists(dt)


def test_fresh_object_partitions_by_actual_utc_transitions(
    tmp_path, source_pair
):
    """A freshly loaded object is governed by its file's own transitions."""
    old_data, new_data, kind = source_pair
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _install(old_root, KEY, old_data)
    _install(new_root, KEY, new_data)

    with set_tzpath((str(old_root),), block_tzdata=True):
        old_zone = tz.gettz(KEY)
    with set_tzpath((str(new_root),), block_tzdata=True):
        new_zone = tz.gettz(KEY)

    assert isinstance(old_zone, tz.tzfile)
    assert isinstance(new_zone, tz.tzfile)
    assert old_zone != new_zone

    # The exact UTC partition differs on one window, and witnesses on both
    # sides of its transition boundaries confirm where it opens and closes.
    windows = _utc_difference_windows(old_zone, new_zone)
    assert windows == [(T_OLD, T_NEW)]
    for start, end in windows:
        _check_window_with_witnesses(old_zone, new_zone, start, end)

    # Each source contributes the anomalous local segment of the transition
    # the other one lacks: the fold gives a two-solution segment, the gap a
    # no-solution segment. These are the rows whose partition is not just
    # shifted but structurally different.
    segments = sorted(
        _anomaly_segments(old_zone, windows)
        + _anomaly_segments(new_zone, windows)
    )
    assert segments == _expected_anomaly_segments(kind)

    for zone in (old_zone, new_zone):
        for segment_kind, start, end in _anomaly_segments(zone, windows):
            ambiguous, exists = _classify(zone, (start + end) // 2)
            if segment_kind == "ambiguous":
                assert (ambiguous, exists) == (True, True)
            else:
                assert (ambiguous, exists) == (False, False)


def test_cached_object_keeps_the_old_partition_after_the_source_changes(
    tmp_path, source_pair
):
    """A cached tzfile is not silently re-read when the file is replaced."""
    old_data, new_data, _ = source_pair
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _install(old_root, KEY, old_data)
    _install(new_root, KEY, new_data)

    with set_tzpath((str(old_root),), block_tzdata=True):
        old_zone = tz.gettz(KEY)
        interior = (T_OLD + T_NEW) // 2
        old_offset = _offset_at(old_zone, interior)

        # Same key, new directory, but the cache is deliberately kept.
        with set_tzpath(
            (str(new_root),), block_tzdata=True, clear_cache=False
        ):
            cached_zone = tz.gettz(KEY)
            assert cached_zone is old_zone
            assert _offset_at(cached_zone, interior) == old_offset

    # Clearing the cache and querying again finally reads the new file.
    with set_tzpath((str(new_root),), block_tzdata=True):
        new_zone = tz.gettz(KEY)
        assert new_zone is not old_zone
        assert _offset_at(new_zone, interior) != old_offset


def test_switching_source_reprocesses_exactly_the_affected_local_rows(
    tmp_path, source_pair
):
    """Re-processing raw local rows changes only rows in the recompute set."""
    old_data, new_data, kind = source_pair
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _install(old_root, KEY, old_data)
    _install(new_root, KEY, new_data)

    with set_tzpath((str(old_root),), block_tzdata=True):
        old_zone = tz.gettz(KEY)
    with set_tzpath((str(new_root),), block_tzdata=True):
        new_zone = tz.gettz(KEY)

    windows = _utc_difference_windows(old_zone, new_zone)
    assert windows == [(T_OLD, T_NEW)]

    # The recompute set is the union of the anomalous segments (rows that
    # gain or lose a UTC solution) with the wall-time images of the UTC
    # difference window (ordinary rows whose single UTC solution moves).
    anomaly_intervals = [
        (start, end)
        for _, start, end in (
            _anomaly_segments(old_zone, windows)
            + _anomaly_segments(new_zone, windows)
        )
    ]
    expected_intervals = _merge_intervals(
        anomaly_intervals
        + _local_images(old_zone, windows)
        + _local_images(new_zone, windows)
    )

    row_timestamps = range(T_OLD - 2 * 3600, T_NEW + 2 * 3600, 1800)
    recomputed = {
        wall_timestamp
        for wall_timestamp in row_timestamps
        if _utc_solutions(old_zone, wall_timestamp)
        != _utc_solutions(new_zone, wall_timestamp)
    }

    expected = {
        wall_timestamp
        for wall_timestamp in row_timestamps
        if _contains(expected_intervals, wall_timestamp)
    }
    assert recomputed == expected
    assert recomputed  # The difference window really reaches the rows.

    # Every anomalous segment is inside the recompute set, with the fold's
    # two solutions distinguished from the gap's missing solution.
    for zone in (old_zone, new_zone):
        for segment_kind, start, end in _anomaly_segments(zone, windows):
            midpoint = (start + end) // 2
            assert midpoint in recomputed
            if kind == "fold":
                assert len(_utc_solutions(zone, midpoint)) == 2
            else:
                assert _utc_solutions(zone, midpoint) == frozenset()

    # Rows well outside the window keep exactly the same UTC partition.
    far_before, far_after = T_OLD - 2 * 3600, T_NEW + 2 * 3600 - 1800
    assert not _contains(expected_intervals, far_before)
    assert not _contains(expected_intervals, far_after)
    assert _utc_solutions(old_zone, far_before) == _utc_solutions(
        new_zone, far_before
    )
    assert _utc_solutions(old_zone, far_after) == _utc_solutions(
        new_zone, far_after
    )


def test_same_offsets_from_another_source_remain_traceable(tmp_path):
    """Identical UTC offsets do not make two differently sourced files equal."""
    std_a = ZoneOffset("STD", ZERO)
    dst_a = ZoneOffset("DST", ONE_H, ONE_H)
    std_b = ZoneOffset("S00", ZERO)
    dst_b = ZoneOffset("S01", ONE_H, ONE_H)

    data_a = construct_zone([(T_OLD, std_a, dst_a)], after="").read()
    data_b = construct_zone([(T_OLD, std_b, dst_b)], after="").read()

    root_a = tmp_path / "source_a"
    root_b = tmp_path / "source_b"
    path_a = _install(root_a, KEY, data_a)
    path_b = _install(root_b, KEY, data_b)

    with set_tzpath((str(root_a),), block_tzdata=True):
        zone_a = tz.gettz(KEY)
    with set_tzpath((str(root_b),), block_tzdata=True):
        zone_b = tz.gettz(KEY)

    # The UTC offsets, including on both sides of the transition, match.
    assert _utc_difference_windows(zone_a, zone_b) == []
    for ts in (T_OLD - 1, T_OLD + 1):
        assert _offset_at(zone_a, ts) == _offset_at(zone_b, ts)

    # The abbreviations and the files they came from do not, so the source
    # can still be traced after the fact despite the equal offsets.
    assert zone_a != zone_b
    assert zone_a.key == zone_b.key == KEY
    assert os.path.normpath(zone_a._filename) == str(path_a)
    assert os.path.normpath(zone_b._filename) == str(path_b)
    assert zone_a._filename != zone_b._filename

    dt_utc = EPOCH + timedelta(seconds=T_OLD + 1)
    assert dt_utc.astimezone(zone_a).tzname() == "DST"
    assert dt_utc.astimezone(zone_b).tzname() == "S01"


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(b"not a TZif file\n", id="not_tzif"),
        pytest.param(b"truncated", id="truncated"),
    ],
)
def test_corrupt_source_falls_back_and_is_accounted_separately(
    tmp_path, source_pair, corrupt
):
    """A broken source is skipped without poisoning lookup of the good one."""
    old_data, new_data, _ = source_pair
    if corrupt == b"truncated":
        corrupt = old_data[:30]

    broken_root = tmp_path / "broken"
    good_root = tmp_path / "good"
    good_path = _install(good_root, KEY, new_data)
    _install(broken_root, KEY, corrupt)

    reference = tz.tzfile(io.BytesIO(new_data), key=KEY)

    # The broken first entry is skipped and the good second entry is used.
    with set_tzpath((str(broken_root), str(good_root)), block_tzdata=True):
        zone = tz.gettz(KEY)
        assert isinstance(zone, tz.tzfile)
        assert os.path.normpath(zone._filename) == str(good_path)
        assert zone == reference
        interior = (T_OLD + T_NEW) // 2
        assert _offset_at(zone, interior) == _offset_at(reference, interior)

        # The fallback object is what gets cached, so clearing and re-querying
        # does not resurrect the broken file.
        tz.gettz.cache_clear()
        assert tz.gettz(KEY) == reference

    # With the broken source left on its own, the key resolves to None rather
    # than to a partially parsed object or a cached answer from elsewhere.
    with set_tzpath((str(broken_root),), block_tzdata=True):
        assert tz.gettz(KEY) is None
