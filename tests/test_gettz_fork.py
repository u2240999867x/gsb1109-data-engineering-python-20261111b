"""Regression tests for using ``tz.gettz`` across ``os.fork``.

A date-cleaning pool warms up some zones in the parent process and then
forks workers. These tests pin down which UTC instant ("partition") a
forked child assigns local rows to when the fork happens:

- while another thread is inside ``gettz`` loading rules, so the cache
  lock is held,
- on a cache hit, so the child inherits the warmed zone object, and
- right after the rule source was swapped for a new version.

Two versions of the rule source define the same zone with a different
fall-back transition, so a child that silently keeps using the old rules
is visible in the rows it emits. Every child must answer within a bounded
time, either with attributed rows or with a recognizable error; the
synchronization trace then distinguishes an inherited lock that no thread
will release, an inherited stale zone object and a legitimate snapshot of
the old rule version.
"""

import calendar
import contextlib
import json
import os
import select
import signal
import struct
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime

import pytest

from dateutil import tz

from ._common import set_tzpath

pytestmark = [
    pytest.mark.gettz,
    pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork"),
]

UTC = tz.UTC
KEY = "Cleaning/Pool"
WARM_KEY = "Cleaning/Other"

# The zone has standard time ("PLS", UTC+0) and DST ("PLD", UTC+1). The
# two rule versions agree except on the 2020 fall-back, so the version a
# worker used can be read off the UTC instants it assigns rows to.
_STD = (0, 0, "PLS")
_DST = (3600, 1, "PLD")


def _ts(year, month, day, hour=0):
    return calendar.timegm((year, month, day, hour, 0, 0))


RULES = {
    "A": [
        (_ts(2020, 1, 1), 0),
        (_ts(2020, 3, 29, 1), 1),
        (_ts(2020, 11, 1, 1), 0),
    ],
    "B": [
        (_ts(2020, 1, 1), 0),
        (_ts(2020, 3, 29, 1), 1),
        (_ts(2020, 11, 8, 1), 0),
    ],
}

# Local wall-clock rows the pool hands to each worker, as (wall, fold).
ROWS = [
    (datetime(2020, 10, 31, 12, 0), 0),
    (datetime(2020, 11, 1, 1, 30), 0),
    (datetime(2020, 11, 1, 1, 30), 1),
    (datetime(2020, 11, 5, 12, 0), 0),
    (datetime(2020, 11, 9, 12, 0), 0),
]


def _build_tzif(transitions):
    """Builds a minimal version-1 TZif file for a two-type zone."""
    types = [_STD, _DST]
    abbrs = b""
    ttinfos = []
    for utcoff, isdst, abbr in types:
        ttinfos.append((utcoff, isdst, len(abbrs)))
        abbrs += abbr.encode("ascii") + b"\x00"

    out = bytearray()
    out += b"TZif\x00" + b"\x00" * 15
    out += struct.pack(
        ">6l", 0, 0, 0, len(transitions), len(types), len(abbrs)
    )
    out += struct.pack(">%dl" % len(transitions), *(t for t, _ in transitions))
    out += struct.pack(">%dB" % len(transitions), *(i for _, i in transitions))
    for ttinfo in ttinfos:
        out += struct.pack(">lbb", *ttinfo)
    out += abbrs
    return bytes(out)


@pytest.fixture
def rule_sources(tmp_path):
    """Two versions of the rule source, differing in one fall-back."""
    dirs = {}
    for version, transitions in RULES.items():
        source_dir = tmp_path / ("rules_" + version.lower())
        zone_dir = source_dir / "Cleaning"
        zone_dir.mkdir(parents=True)
        data = _build_tzif(transitions)
        (zone_dir / "Pool").write_bytes(data)
        (zone_dir / "Other").write_bytes(data)
        dirs[version] = str(source_dir)
    return dirs


def _attribute_rows(zone):
    """The UTC instant ("partition") the pool assigns each local row."""
    attributed = []
    for wall, fold in ROWS:
        dt = tz.enfold(wall.replace(tzinfo=zone), fold=fold)
        utc = dt.astimezone(UTC)
        attributed.append([wall.isoformat(), fold, utc.isoformat()])
    return attributed


@pytest.fixture
def expected_rows(rule_sources):
    """The rows each rule version produces, read straight from the files."""
    expected = {}
    for version, source_dir in rule_sources.items():
        zone = tz.tzfile(os.path.join(source_dir, *KEY.split("/")))
        expected[version] = _attribute_rows(zone)
    return expected


def test_rule_versions_disagree_on_the_fallback(expected_rows):
    """The two rule sources attribute the fall-back rows differently."""
    assert expected_rows["A"] != expected_rows["B"]


def _signature(rows, expected_rows):
    """Which rule version a worker's rows were attributed with, if any."""
    for version, expected in expected_rows.items():
        if rows == expected:
            return version
    return "unknown"


def _child_report(report_fd, key, warm_zone, probe):
    """Worker entry point, run in the forked child.

    The child inherits every lock in the state the fork left it in, so it
    must not touch pytest, logging or stdout -- only the report pipe.
    """
    try:
        lock_state = cache_state = None
        if probe:
            lock = tz.gettz._cache_lock
            if lock.acquire(blocking=False):
                lock.release()
                lock_state = "free"
            else:
                lock_state = "held"

        if lock_state == "held":
            # gettz would block forever on the inherited lock, so
            # attribute the rows straight from the rule source.
            zone = tz.gettz.nocache(key)
            cache_state = "bypassed"
        else:
            zone = tz.gettz(key)
            if probe:
                cache_state = "hit" if zone is warm_zone else "miss"

        report = {
            "lock": lock_state,
            "cache": cache_state,
            "rows": _attribute_rows(zone),
        }
        status = 0
    except Exception:
        report = {"error": traceback.format_exc()}
        status = 3

    os.write(report_fd, json.dumps(report).encode("ascii") + b"\n")
    os._exit(status)


class ForkResult:
    def __init__(self, report, exit_code, timed_out):
        self.report = report
        self.exit_code = exit_code
        self.timed_out = timed_out

    @property
    def rows(self):
        return self.report.get("rows")


def _fork_worker(key, warm_zone, probe=True, timeout=10.0):
    """Forks one cleaning worker and collects its report within ``timeout``."""
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        # Forking a process with live threads is exactly what is under
        # test here; the interpreter's own warning about it must not
        # become an error under the suite's -W error configuration.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()

    if pid == 0:  # child
        try:
            os.close(read_fd)
            _child_report(write_fd, key, warm_zone, probe)
        finally:
            # Only reached if _child_report itself crashed before exiting.
            os._exit(4)

    os.close(write_fd)
    chunks = []
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        ready, _, _ = select.select([read_fd], [], [], 0.05)
        if ready:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break  # the child closed its end of the pipe by exiting
            chunks.append(chunk)
        if not timed_out and time.monotonic() > deadline:
            timed_out = True
            os.kill(pid, signal.SIGKILL)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    if os.WIFEXITED(status):
        exit_code = os.WEXITSTATUS(status)
    else:
        exit_code = -os.WTERMSIG(status)

    lines = b"".join(chunks).splitlines()
    report = json.loads(lines[0]) if lines else {}
    return ForkResult(report, exit_code, timed_out)


def _classify(result, source_version, expected_rows):
    """Reads the synchronization trace of a finished worker.

    One of:
    - "timeout":        no answer within the bound (a hang, e.g. a
                        deadlock on the inherited cache lock)
    - "error":          the worker reported a recognizable error
    - "inherited-lock": the cache lock was inherited held; the rows came
                        from a lock-free read of the rule source
    - "stale-object":   an inherited cached zone object was used although
                        the rule source had moved on
    - "snapshot":       an inherited cached zone object matching the rule
                        source as of the fork
    - "fresh-load":     rules loaded from the current source after the fork
    """
    if result.timed_out:
        return "timeout"
    if "error" in result.report:
        return "error"
    if result.report.get("lock") == "held":
        return "inherited-lock"

    signature = _signature(result.rows, expected_rows)
    if signature == "unknown":
        return "unexplained"

    cache = result.report.get("cache")
    if cache == "hit":
        return "snapshot" if signature == source_version else "stale-object"
    if cache == "miss":
        return "fresh-load" if signature == source_version else "unexplained"
    return "unexplained"


def test_fork_on_cache_hit_uses_parent_snapshot(rule_sources, expected_rows):
    """A worker forked on a cache hit attributes rows with the warmed zone."""
    with set_tzpath((rule_sources["A"],), block_tzdata=True):
        tz.gettz(WARM_KEY)
        warm = tz.gettz(KEY)

        result = _fork_worker(KEY, warm)

    assert result.exit_code == 0, result.report
    assert _classify(result, "A", expected_rows) == "snapshot"
    assert result.rows == expected_rows["A"]


def test_swap_without_clear_reuses_stale_zone(rule_sources, expected_rows):
    """Swapping the rule source without clearing the cache leaves workers
    on the old rules, and the row attribution gives it away."""
    with set_tzpath((rule_sources["A"],), block_tzdata=True):
        warm = tz.gettz(KEY)
        with set_tzpath((rule_sources["B"],), clear_cache=False):
            result = _fork_worker(KEY, warm)

    assert result.exit_code == 0, result.report
    assert _classify(result, "B", expected_rows) == "stale-object"
    assert result.rows == expected_rows["A"]


def test_swap_and_clear_loads_new_rules(rule_sources, expected_rows):
    """After the source swap and a cache clear, workers load the new rules."""
    with set_tzpath((rule_sources["A"],), block_tzdata=True):
        warm = tz.gettz(KEY)
        with set_tzpath((rule_sources["B"],)):
            result = _fork_worker(KEY, warm)

    assert result.exit_code == 0, result.report
    assert _classify(result, "B", expected_rows) == "fresh-load"
    assert result.rows == expected_rows["B"]


@contextlib.contextmanager
def loading_thread(monkeypatch):
    """Runs a thread blocked inside ``gettz`` while holding the cache lock.

    Must be entered inside a ``set_tzpath`` context, so that the load
    reads the test's rule source. Yields an event that is set once the
    loader holds the lock; the loader is released and joined on exit.
    """
    tz_module = sys.modules["dateutil.tz.tz"]
    real_tzfile = tz_module.tzfile
    parent_pid = os.getpid()
    entered = threading.Event()
    proceed = threading.Event()

    def blocking_tzfile(*args, **kwargs):
        # Only the loader thread in the parent blocks; a forked child
        # falls straight through to the real tzfile.
        if os.getpid() == parent_pid and threading.current_thread() is loader:
            entered.set()
            proceed.wait(10)
        return real_tzfile(*args, **kwargs)

    monkeypatch.setattr(tz_module, "tzfile", blocking_tzfile)
    loader = threading.Thread(target=lambda: tz.gettz(KEY), name="rule-loader")
    loader.start()
    try:
        yield entered
    finally:
        proceed.set()
        loader.join(10)


def test_fork_while_loading_reports_inherited_lock(
    rule_sources, expected_rows, monkeypatch
):
    """Forking mid-load hands the child a cache lock no thread will
    release; the worker must notice and still attribute its rows."""
    with set_tzpath((rule_sources["A"],), block_tzdata=True):
        tz.gettz(WARM_KEY)  # the pool warmed some zones before forking
        with loading_thread(monkeypatch) as entered:
            assert entered.wait(10)
            assert tz.gettz._cache_lock.locked()

            result = _fork_worker(KEY, warm_zone=None)

    assert result.exit_code == 0, result.report
    assert _classify(result, "A", expected_rows) == "inherited-lock"
    assert result.rows == expected_rows["A"]


def test_fork_while_loading_deadlocks_unprepared_worker(
    rule_sources, expected_rows, monkeypatch
):
    """A worker that calls ``gettz`` without probing the inherited lock
    hangs, and the trace shows it as a timeout rather than a result."""
    with set_tzpath((rule_sources["A"],), block_tzdata=True):
        with loading_thread(monkeypatch) as entered:
            assert entered.wait(10)
            assert tz.gettz._cache_lock.locked()

            result = _fork_worker(KEY, warm_zone=None, probe=False, timeout=3.0)

    assert _classify(result, "A", expected_rows) == "timeout"
    assert result.exit_code == -signal.SIGKILL
