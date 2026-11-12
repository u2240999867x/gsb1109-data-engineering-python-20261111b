"""
Batching of per-key changes with debounced, source-sequence-aligned flushes.

.. versionadded:: 8.2.0
"""

from __future__ import annotations

from collections import deque
import threading
import typing as t
import uuid

from .functions import debounce


__all__ = (
    "BatchFlushError",
    "KeyedBatcher",
)

K = t.TypeVar("K")
V = t.TypeVar("V")

# A per-key change: the source sequence number and the version payload.
Change = t.Tuple[int, V]
# Snapshot handed to a writer: one entry per key, the latest version up to the flush boundary.
Batch = t.Dict[K, Change[V]]
Writer = t.Callable[[Batch[K, V]], t.Any]


class BatchFlushError(RuntimeError):
    """Raised when a batch writer fails. The rejected batch stays pending and retryable."""

    def __init__(self, batch: Batch[t.Any, t.Any]) -> None:
        super().__init__(
            f"batch flush failed for keys {sorted(batch)} with pending versions {dict(batch)}"
        )
        self.batch = batch


class KeyedBatcher(t.Generic[K, V]):
    """
    Accumulate per-key change versions and flush them through a debounced writer.

    Every change is tagged with a source sequence number. A flush is bounded by a sequence: for
    each key, only the latest version whose sequence is at or before that boundary is written,
    while later versions stay pending for the next batch. Receipts remember the highest
    acknowledged sequence per key, so overlapping or retried flushes never deliver a stale
    version twice.

    The same reentrant lock guards the timer callback, a running writer, and changes delivered
    from inside that writer, so timer wakes, in-flight writes, and reentrant additions all
    reconcile against the same boundary.

    Args:
        writer: Receives a mapping of ``key -> (sequence, version)`` with the latest deliverable
            version of each key in the batch.
        wait: Debounce quiet period in milliseconds, forwarded to :func:`pydash.debounce`.
        max_wait (optional): Maximum debounce wait in milliseconds.

    .. versionadded:: 8.2.0
    """

    def __init__(
        self,
        writer: Writer[K, V],
        wait: int,
        max_wait: t.Union[int, t.Literal[False]] = False,
    ) -> None:
        self.writer = writer
        self.wait = wait
        self.max_wait = max_wait

        self._lock = threading.RLock()
        # Versions per key, oldest first; pruned once acknowledged up to a flush boundary.
        self._changes: t.Dict[K, t.Deque[Change[V]]] = {}
        # Every sequence ever accepted, so duplicates can be rejected.
        self._seen: t.Set[int] = set()
        # Highest sequence observed; the default flush/cancel boundary.
        self._latest: t.Optional[int] = None
        # Highest sequence acknowledged by a successful writer, per key.
        self._receipts: t.Dict[K, int] = {}
        # Batches currently running inside their writer, keyed by an internal flush id.
        self._in_flight: t.Dict[str, Batch[K, V]] = {}
        # Identity and nesting depth of the thread running a writer, if any.
        self._writer_ident: t.Optional[int] = None
        self._writer_depth = 0
        # A reentrant add() cannot touch the debounce lock; the timer is re-armed once the
        # outermost flush cycle finishes. A reentrant cancel()/close() requests cancellation
        # instead, and cancellation wins over re-arming.
        self._rearm_pending = False
        self._cancel_pending = False
        self._closed = False

        self._debounced = debounce(self._flush_latest, wait, max_wait=max_wait)

    # ------------------------------------------------------------------ input
    def add(self, key: K, sequence: int, version: V) -> None:
        """
        Record ``version`` of ``key`` at source ``sequence`` and (re)arm the debounce timer.

        When called from inside the running writer, the change is recorded without touching the
        debounce timer (the writer already holds this batcher's lock); the timer is re-armed
        when the flush cycle finishes.

        Raises:
            RuntimeError: If the batcher has been closed.
            ValueError: If ``sequence`` was already used or is not greater than the latest
                sequence seen for ``key``.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")
            if sequence in self._seen:
                raise ValueError(f"duplicate sequence number: {sequence}")

            high_water: t.Optional[int] = self._receipts.get(key)
            if key in self._changes:
                latest_pending = self._changes[key][-1][0]
                high_water = (
                    latest_pending if high_water is None else max(high_water, latest_pending)
                )
            if high_water is not None and sequence <= high_water:
                raise ValueError(
                    f"sequence {sequence} for key {key!r} is not greater than {high_water}"
                )

            self._seen.add(sequence)
            self._changes.setdefault(key, deque()).append((sequence, version))
            self._latest = sequence if self._latest is None else max(self._latest, sequence)

            reentrant = self._writer_depth > 0 and self._writer_ident == threading.get_ident()
            if reentrant:
                self._rearm_pending = True

        if not reentrant:
            self._debounced(key, sequence, version)

    # ------------------------------------------------------------------ flushing
    def flush(self, sequence: t.Optional[int] = None) -> t.Any:
        """
        Flush the latest version of each key at or before ``sequence``.

        Without an explicit boundary, the highest observed sequence is used. A timer that has not
        fired yet stays scheduled and simply finds nothing new when it later runs; a timer whose
        writer is already running serializes on the same lock and its receipts make this call a
        no-op for anything already delivered. Returns the writer's result, or ``None`` when
        nothing new is deliverable.
        """
        actions: t.Tuple[bool, bool, t.Optional[int]] = (False, False, None)
        try:
            with self._lock:
                boundary = self._boundary(sequence)
                try:
                    result = self._flush_through(boundary)
                finally:
                    actions = self._take_timer_actions()
        finally:
            self._apply_timer_actions(*actions)
        return result

    def _flush_latest(self, *_args: t.Any) -> None:
        """Debounce target: flush through the newest sequence observed when the timer fired."""
        actions: t.Tuple[bool, bool, t.Optional[int]] = (False, False, None)
        try:
            with self._lock:
                if self._latest is None:  # pragma: no cover
                    return
                try:
                    self._flush_through(self._latest)
                finally:
                    actions = self._take_timer_actions()
        finally:
            self._apply_timer_actions(*actions)

    def _flush_through(self, boundary: int) -> t.Any:
        """Deliver batches until no unacknowledged version remains at or before ``boundary``.

        A nested flush runs inside the outermost writer cycle and leaves its pending timer
        actions for that cycle to apply.
        """
        self._enter_writer()
        result: t.Any = None
        try:
            while True:
                batch = self._build_batch(boundary)
                if not batch:
                    return result

                flush_id = uuid.uuid4().hex
                self._in_flight[flush_id] = batch
                try:
                    # The lock stays held across the writer; reentrant add()/flush()/cancel() on
                    # the same thread are allowed and reconciled by the loop and the receipts.
                    result = self.writer(batch)
                except Exception as exc:
                    self._in_flight.pop(flush_id, None)
                    raise BatchFlushError(batch) from exc
                else:
                    self._in_flight.pop(flush_id, None)
                    self._acknowledge(batch)
                    self._prune(boundary)
        finally:
            self._exit_writer()

    def _take_timer_actions(self) -> t.Tuple[bool, bool, t.Optional[int]]:
        """Consume pending timer actions for the outermost cycle; nested cycles defer them."""
        if self._writer_depth != 0:
            return False, False, self._latest
        rearm = self._rearm_pending
        cancel_timer = self._cancel_pending
        self._rearm_pending = False
        self._cancel_pending = False
        return rearm, cancel_timer, self._latest

    def _apply_timer_actions(
        self, rearm: bool, cancel_timer: bool, latest: t.Optional[int]
    ) -> None:
        """Touch the debounce wrapper only after releasing the batcher lock, so the lock order is
        always batcher-then-debounce and a timer callback can never deadlock against a writer."""
        if cancel_timer:
            self._debounced.cancel()
        elif rearm:
            # The debounced target ignores its arguments; only the timing matters.
            self._debounced(None, latest, None)

    def _enter_writer(self) -> None:
        if self._writer_depth == 0:
            self._writer_ident = threading.get_ident()
        self._writer_depth += 1

    def _exit_writer(self) -> None:
        self._writer_depth -= 1
        if self._writer_depth == 0:
            self._writer_ident = None

    def _build_batch(self, boundary: int) -> Batch[K, V]:
        """Latest not-yet-acknowledged version of each key whose sequence is <= ``boundary``."""
        batch: Batch[K, V] = {}
        for key, versions in self._changes.items():
            receipt = self._receipts.get(key)
            latest: t.Optional[Change[V]] = None
            for change in versions:
                sequence = change[0]
                if sequence > boundary:
                    break
                if receipt is None or sequence > receipt:
                    latest = change
            if latest is not None:
                batch[key] = latest
        return batch

    def _acknowledge(self, batch: Batch[K, V]) -> None:
        for key, (sequence, _version) in batch.items():
            current = self._receipts.get(key)
            if current is None or sequence > current:
                self._receipts[key] = sequence

    def _prune(self, boundary: int) -> None:
        """Drop acknowledged versions at or before ``boundary``; later versions stay pending."""
        for key in list(self._changes):
            versions = self._changes[key]
            receipt = self._receipts.get(key)
            while versions:
                sequence = versions[0][0]
                if sequence > boundary or receipt is None or sequence > receipt:
                    break
                versions.popleft()
            if not versions:
                del self._changes[key]

    # ------------------------------------------------------------------ cancel/close
    def cancel(self, sequence: t.Optional[int] = None) -> t.List[Change[V]]:
        """
        Release undelivered versions at or before ``sequence`` (default: every pending version).

        Versions already acknowledged or currently inside a writer are retained. Returns the
        dropped ``(sequence, version)`` pairs ordered by sequence. Cancels the debounce timer
        when nothing remains pending.
        """
        with self._lock:
            boundary = self._boundary(sequence)
            reentrant = self._writer_depth > 0 and self._writer_ident == threading.get_ident()
            in_flight = {
                (key, change[0])
                for batch in self._in_flight.values()
                for key, change in batch.items()
            }
            dropped: t.List[Change[V]] = []

            for key in list(self._changes):
                versions = self._changes[key]
                receipt = self._receipts.get(key)
                kept: t.Deque[Change[V]] = deque()
                for change in versions:
                    seq = change[0]
                    delivered = (receipt is not None and seq <= receipt) or (
                        key,
                        seq,
                    ) in in_flight
                    if seq <= boundary and not delivered:
                        dropped.append(change)
                    else:
                        kept.append(change)
                if kept:
                    self._changes[key] = kept
                else:
                    del self._changes[key]

            empty = not self._changes
            if reentrant and empty:  # pragma: no cover
                # Unreachable while a writer runs: the in-flight batch's versions are always
                # retained, so the backlog cannot empty. Kept defensively for future refactors.
                # Defer the debounce lock until the outermost writer cycle releases this lock.
                self._cancel_pending = True
                self._rearm_pending = False

        if not reentrant and empty:
            self._debounced.cancel()

        return sorted(dropped, key=lambda change: change[0])

    def close(self) -> None:
        """Cancel the debounce timer and reject further :meth:`add` calls. Pending data is kept."""
        with self._lock:
            self._closed = True
            reentrant = self._writer_depth > 0 and self._writer_ident == threading.get_ident()
            if reentrant:
                self._cancel_pending = True
                self._rearm_pending = False

        if not reentrant:
            self._debounced.cancel()

    # ------------------------------------------------------------------ inspection
    @property
    def pending(self) -> t.Dict[K, t.List[Change[V]]]:
        """Versions not yet acknowledged, including those currently inside a writer."""
        with self._lock:
            snapshot: t.Dict[K, t.List[Change[V]]] = {}
            for key, versions in self._changes.items():
                receipt = self._receipts.get(key)
                unacknowledged = [
                    change for change in versions if receipt is None or change[0] > receipt
                ]
                if unacknowledged:
                    snapshot[key] = unacknowledged
            return snapshot

    @property
    def receipts(self) -> t.Dict[K, int]:
        """Highest acknowledged sequence per key."""
        with self._lock:
            return dict(self._receipts)

    @property
    def in_flight(self) -> t.Dict[str, Batch[K, V]]:
        """Batches currently running inside their writer, keyed by an internal flush id."""
        with self._lock:
            return {flush_id: dict(batch) for flush_id, batch in self._in_flight.items()}

    def _boundary(self, sequence: t.Optional[int]) -> int:
        if sequence is not None:
            return int(sequence)
        if self._latest is None:
            raise ValueError("no changes have been added")
        return self._latest
