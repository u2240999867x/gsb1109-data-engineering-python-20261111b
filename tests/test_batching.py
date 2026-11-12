from concurrent.futures import ThreadPoolExecutor, TimeoutError
import heapq
import threading
import time
from unittest import mock

import pytest

import pydash as _
from pydash import functions
from pydash.batching import BatchFlushError, KeyedBatcher


class ControlledClock:
    """Deterministic replacement for ``time.monotonic_ns`` and ``threading.Timer``."""

    def __init__(self, monkeypatch, start=0):
        self.now = start
        self._queue = []
        self.timers = []
        self._monkeypatch = monkeypatch

        clock = self

        class FakeTimer:
            def __init__(self, interval, function, args):
                self.when = clock.now + round(interval * 1000)
                self.function = function
                self.args = args
                self.cancelled = False
                self.timer_id = len(clock.timers)
                clock.timers.append(self)

            def start(self):
                heapq.heappush(clock._queue, (self.when, self.timer_id, self))

            def cancel(self):
                self.cancelled = True

        monkeypatch.setattr(time, "monotonic_ns", lambda: self.now * 1_000_000)
        monkeypatch.setattr(functions.threading, "Timer", FakeTimer)

    def advance(self, milliseconds):
        target = self.now + milliseconds
        while self._queue and self._queue[0][0] <= target:
            when, _timer_id, timer = heapq.heappop(self._queue)
            self.now = max(self.now, when)
            if not timer.cancelled:
                timer.function(*timer.args)
        self.now = target


@pytest.fixture
def clock(monkeypatch):
    return ControlledClock(monkeypatch)


def make_batcher(writer, wait=10, **kwargs):
    return KeyedBatcher(writer, wait, **kwargs)


def test_flush_writes_latest_version_of_each_key_up_to_boundary():
    commits = []
    batcher = make_batcher(lambda batch: commits.append(dict(batch)), wait=1000)

    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    batcher.add("a", 5, "a5")

    batcher.flush(3)
    assert commits == [{"a": (1, "a1"), "b": (2, "b2")}]
    assert batcher.receipts == {"a": 1, "b": 2}
    assert batcher.pending == {"a": [(5, "a5")]}

    batcher.flush()
    assert commits == [
        {"a": (1, "a1"), "b": (2, "b2")},
        {"a": (5, "a5")},
    ]
    assert batcher.receipts == {"a": 5, "b": 2}
    assert batcher.pending == {}


def test_versions_after_boundary_remain_for_next_batch(clock):
    commits = []
    batcher = make_batcher(lambda batch: commits.append(dict(batch)), wait=10)

    batcher.add("a", 1, "a1")
    batcher.add("a", 2, "a2")
    clock.advance(5)
    batcher.add("b", 4, "b4")

    batcher.flush(2)
    assert commits == [{"a": (2, "a2")}]

    # A boundary ahead of the latest observed sequence still cannot deliver future versions.
    batcher.flush(99)
    assert commits == [{"a": (2, "a2")}, {"b": (4, "b4")}]
    assert batcher.pending == {}


def test_debounce_timer_flushes_after_quiet_period(clock):
    commits = []
    batcher = make_batcher(lambda batch: commits.append(dict(batch)), wait=10)

    batcher.add("a", 1, "a1")
    clock.advance(5)
    batcher.add("a", 2, "a2")
    clock.advance(9)
    assert commits == []

    clock.advance(2)
    assert commits == [{"a": (2, "a2")}]
    assert batcher.receipts == {"a": 2}

    # Versions arriving after the timer fired wait for the next batch.
    batcher.add("a", 3, "a3")
    assert batcher.pending == {"a": [(3, "a3")]}
    clock.advance(10)
    assert commits == [{"a": (2, "a2")}, {"a": (3, "a3")}]


def test_timer_uses_boundary_snapshot_and_keeps_later_versions(clock):
    commits = []
    batcher = make_batcher(lambda batch: commits.append(dict(batch)), wait=10)

    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    clock.advance(10)
    assert commits == [{"a": (1, "a1"), "b": (2, "b2")}]

    batcher.add("a", 7, "a7")
    batcher.add("b", 8, "b8")
    clock.advance(6)
    # Manual flush at an old boundary: the newest versions stay pending.
    batcher.flush(2)
    assert commits == [{"a": (1, "a1"), "b": (2, "b2")}]
    assert batcher.pending == {"a": [(7, "a7")], "b": [(8, "b8")]}

    clock.advance(4)
    assert commits == [
        {"a": (1, "a1"), "b": (2, "b2")},
        {"a": (7, "a7"), "b": (8, "b8")},
    ]


def test_add_from_inside_writer_waits_for_next_batch():
    commits = []
    batcher = None

    def writer(batch):
        commits.append(dict(batch))
        if not any(seq == 6 for seq, _ in batch.values()):
            batcher.add("a", 6, "a6")
        return len(commits)

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")
    batcher.add("a", 3, "a3")
    batcher.add("b", 4, "b4")

    batcher.flush(4)
    assert commits == [{"a": (3, "a3"), "b": (4, "b4")}]
    assert batcher.pending == {"a": [(6, "a6")]}
    assert batcher.in_flight == {}

    batcher.flush()
    assert commits == [
        {"a": (3, "a3"), "b": (4, "b4")},
        {"a": (6, "a6")},
    ]


def test_timer_callback_while_writer_running_reconciles_same_boundary():
    entered = threading.Event()
    release = threading.Event()
    commits = []

    def writer(batch):
        commits.append(dict(batch))
        entered.set()
        assert release.wait(5)
        return len(commits)

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")

    with ThreadPoolExecutor(max_workers=2) as pool:
        timer_flush = pool.submit(batcher._flush_latest)
        assert entered.wait(5)

        # The timer wake is inside the writer; new versions block until the writer finishes.
        late_add = pool.submit(batcher.add, "a", 2, "a2")
        with pytest.raises(TimeoutError):
            late_add.result(timeout=0.1)

        release.set()
        timer_flush.result(5)
        late_add.result(5)

    # The timer flush was bounded at sequence 1; sequence 2 stays for the next batch.
    assert commits == [{"a": (1, "a1")}]
    assert batcher.pending == {"a": [(2, "a2")]}
    batcher.flush()
    assert commits == [{"a": (1, "a1")}, {"a": (2, "a2")}]


def test_manual_flush_overlaps_running_writer_does_not_rewrite_stale_version():
    entered = threading.Event()
    release = threading.Event()
    commits = []

    def writer(batch):
        commits.append(dict(batch))
        if len(commits) == 1:
            entered.set()
            assert release.wait(5)
        return len(commits)

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(batcher.flush)
        assert entered.wait(5)

        late_add = pool.submit(batcher.add, "a", 2, "a2")
        second = pool.submit(batcher.flush, 1)
        with pytest.raises(TimeoutError):
            second.result(timeout=0.1)

        release.set()
        assert first.result(5) == 1
        assert second.result(5) is None
        late_add.result(5)

    # The overlapping flush at boundary 1 delivers nothing new; only the final flush writes v2.
    assert commits == [{"a": (1, "a1")}]
    batcher.flush()
    assert commits == [{"a": (1, "a1")}, {"a": (2, "a2")}]


def test_cancel_releases_only_undelivered_versions():
    commits = []
    batcher = make_batcher(lambda batch: commits.append(dict(batch)), wait=1000)

    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    batcher.add("a", 3, "a3")

    batcher.flush(1)
    assert commits == [{"a": (1, "a1")}]

    dropped = batcher.cancel(2)
    assert dropped == [(2, "b2")]
    assert batcher.pending == {"a": [(3, "a3")]}
    assert batcher.receipts == {"a": 1}

    batcher.flush()
    assert commits == [{"a": (1, "a1")}, {"a": (3, "a3")}]


def test_cancel_without_boundary_drops_full_backlog_and_re_arms_timer(clock):
    batcher = make_batcher(mock.Mock(return_value=None), wait=10)
    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    batcher.add("a", 3, "a3")

    dropped = batcher.cancel()
    assert dropped == [(1, "a1"), (2, "b2"), (3, "a3")]
    assert batcher.pending == {}
    assert all(timer.cancelled for timer in clock.timers)

    # A later add starts a fresh debounce cycle.
    batcher.add("a", 4, "a4")
    clock.advance(10)
    batcher.writer.assert_called_once_with({"a": (4, "a4")})


def test_cancel_inside_writer_keeps_inflight_versions():
    observed = []

    def writer(batch):
        dropped = batcher.cancel()
        observed.append((dict(batch), dropped))

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    batcher.add("a", 3, "a3")

    batcher.flush(2)

    inflight_batch, dropped = observed[0]
    assert inflight_batch == {"a": (1, "a1"), "b": (2, "b2")}
    assert dropped == [(3, "a3")]
    # The in-flight batch still completes and issues receipts on success.
    assert batcher.receipts == {"a": 1, "b": 2}
    assert batcher.pending == {}


def test_close_inside_writer_defers_timer_cancellation(clock):
    events = []

    def writer(batch):
        events.append(dict(batch))
        batcher.add("a", 6, "a6")
        batcher.close()

    batcher = make_batcher(writer, wait=10)
    batcher.add("a", 1, "a1")
    batcher.flush()

    assert events == [{"a": (1, "a1")}]
    assert batcher.pending == {"a": [(6, "a6")]}
    assert all(timer.cancelled for timer in clock.timers)
    with pytest.raises(RuntimeError, match="closed"):
        batcher.add("a", 7, "a7")


def test_nested_flush_reconciles_within_same_writer_cycle():
    commits = []

    def writer(batch):
        commits.append(dict(batch))
        if len(commits) == 1:
            batcher.add("a", 6, "a6")
            # The nested flush delivers the reentrant version immediately and re-arms the timer.
            nested_result = batcher.flush()
            assert nested_result == len(commits)
        return len(commits)

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")

    assert batcher.flush() == 2
    assert commits == [{"a": (1, "a1")}, {"a": (6, "a6")}]
    assert batcher.pending == {}
    assert batcher.receipts == {"a": 6}


def test_reentrant_cancel_partial_backlog_keeps_timer_armed(clock):
    observed = []

    def writer(batch):
        observed.append(dict(batch))
        if len(observed) == 1:
            batcher.add("c", 7, "c7")
            # Cancel only an older undelivered key; c7 remains, so the timer must stay alive.
            batcher.cancel(5)

    batcher = make_batcher(writer, wait=10)
    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")
    batcher.add("a", 3, "a3")
    batcher.add("d", 4, "d4")

    batcher.flush(3)
    assert observed == [{"a": (3, "a3"), "b": (2, "b2")}]
    # d4 (seq 4 <= 5) is released; only the reentrant c7 stays for the next batch.
    assert batcher.pending == {"c": [(7, "c7")]}
    assert not all(timer.cancelled for timer in clock.timers)

    clock.advance(10)
    assert batcher.pending == {}


def test_failed_flush_keeps_reentrant_versions_pending(clock):
    commits = []
    attempts = {"count": 0}

    def writer(batch):
        attempts["count"] += 1
        if attempts["count"] == 1:
            batcher.add("a", 6, "a6")
            raise RuntimeError("index unavailable")
        commits.append(dict(batch))

    batcher = make_batcher(writer, wait=10)
    batcher.add("a", 1, "a1")

    with pytest.raises(BatchFlushError):
        batcher.flush()

    assert batcher.pending == {"a": [(1, "a1"), (6, "a6")]}
    assert batcher.in_flight == {}

    # The reentrant version re-armed the timer; quiet period delivers both as the latest.
    clock.advance(10)
    assert commits == [{"a": (6, "a6")}]
    assert batcher.receipts == {"a": 6}
    assert batcher.pending == {}


def test_writer_failure_exposes_keys_and_pending_versions():
    commits = []
    attempts = {"count": 0}

    def writer(batch):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("index unavailable")
        commits.append(dict(batch))

    batcher = make_batcher(writer, wait=1000)
    batcher.add("a", 1, "a1")
    batcher.add("b", 2, "b2")

    with pytest.raises(BatchFlushError) as exc_info:
        batcher.flush()

    error = exc_info.value
    assert error.batch == {"a": (1, "a1"), "b": (2, "b2")}
    message = str(error)
    assert "a" in message and "b" in message
    assert "(1, 'a1')" in message and "(2, 'b2')" in message
    assert isinstance(error.__cause__, RuntimeError)
    assert batcher.in_flight == {}
    assert batcher.pending == {"a": [(1, "a1")], "b": [(2, "b2")]}

    batcher.flush()
    assert commits == [{"a": (1, "a1"), "b": (2, "b2")}]
    assert batcher.pending == {}


def test_duplicate_and_out_of_order_sequences_are_rejected():
    batcher = make_batcher(mock.Mock(return_value=None), wait=1000)
    batcher.add("a", 1, "a1")

    with pytest.raises(ValueError, match="duplicate sequence"):
        batcher.add("b", 1, "b1")

    batcher.add("a", 3, "a3")
    with pytest.raises(ValueError, match="not greater than 3"):
        batcher.add("a", 2, "a2")

    # A new key may not reuse a sequence already acknowledged for another key.
    batcher.flush()
    with pytest.raises(ValueError, match="duplicate sequence"):
        batcher.add("c", 1, "c1")


def test_flush_without_changes_raises():
    batcher = make_batcher(mock.Mock(return_value=None), wait=1000)
    with pytest.raises(ValueError, match="no changes"):
        batcher.flush()


def test_close_cancels_timer_and_rejects_adds(clock):
    batcher = make_batcher(mock.Mock(return_value=None), wait=10)
    batcher.add("a", 1, "a1")

    batcher.close()
    assert all(timer.cancelled for timer in clock.timers)
    assert batcher.pending == {"a": [(1, "a1")]}

    with pytest.raises(RuntimeError, match="closed"):
        batcher.add("a", 2, "a2")


def test_controlled_clock_replay_produces_identical_commits(monkeypatch):
    wait = 10

    def run():
        clock = ControlledClock(monkeypatch)
        commits = []
        batcher = None

        def writer(batch):
            commits.append(dict(batch))
            # Reentrant delivery from inside the running writer.
            if all(seq < 6 for seq, _ in batch.values()):
                batcher.add("a", 6, "a6")

        batcher = make_batcher(writer, wait=wait)
        batcher.add("a", 1, "a1")
        batcher.add("b", 2, "b2")
        clock.advance(5)
        batcher.add("a", 3, "a3")
        clock.advance(wait)
        # First timer flush bounded at sequence 3; reentrant sequence 6 stays pending.
        batcher.flush(4)  # nothing new at this old boundary
        clock.advance(wait)
        batcher.add("b", 8, "b8")
        clock.advance(wait)
        return commits

    first_run = run()
    second_run = run()
    assert first_run == [
        {"a": (3, "a3"), "b": (2, "b2")},
        {"a": (6, "a6")},
        {"b": (8, "b8")},
    ]
    assert second_run == first_run


def test_debounce_flush_runs_pending_call_immediately(monkeypatch):
    clock = ControlledClock(monkeypatch)
    func = mock.Mock(return_value="done")
    debounced = _.debounce(func, 10)

    assert debounced("first") is None
    assert debounced.flush() == "done"
    func.assert_called_once_with("first")

    # The pending timer was consumed; advancing the clock does not fire it again.
    clock.advance(50)
    func.assert_called_once_with("first")


def test_debounce_flush_without_pending_call_returns_last_result():
    func = mock.Mock(return_value="done")
    debounced = _.debounce(func, 10)
    assert debounced.flush() is None

    debounced("value")
    debounced.flush()
    assert debounced.flush() == "done"
    func.assert_called_once_with("value")


def test_debounce_cancel_discards_pending_arguments(monkeypatch):
    clock = ControlledClock(monkeypatch)
    func = mock.Mock()
    debounced = _.debounce(func, 10)

    debounced("first")
    debounced.cancel()
    clock.advance(50)
    func.assert_not_called()

    # A fresh burst after cancel still works.
    debounced("second")
    clock.advance(10)
    func.assert_called_once_with("second")
