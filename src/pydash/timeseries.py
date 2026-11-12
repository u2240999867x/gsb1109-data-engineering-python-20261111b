"""
Time-series sensor imputation with revision-aware, per-device state.

Sensor gaps are filled by carrying the most recent observed reading forward.
Because source events can be retransmitted, corrected, or delivered late, every
event is identified by its stable ``(event_time, event_seq)`` key and carries a
monotonic ``revision``. Only the highest revision seen for a key is retained.

Filled readings remember the observed event they were derived from so that a
late correction (for example, a shutdown that invalidates a stretch of fills)
only forces recomputation of the affected device interval.

.. versionadded:: 8.2.0
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import typing as t


__all__ = (
    "ACTION_ADDED",
    "ACTION_SOURCE_CHANGED",
    "ACTION_WITHDRAWN",
    "DeviceImputer",
    "ImputerBatch",
    "ImputerReading",
    "SensorEvent",
    "SOURCE_FILLED",
    "SOURCE_OBSERVED",
)


#: Reading was published for the first time for its event position.
ACTION_ADDED = "added"
#: A previously published reading is no longer valid and must be retracted.
ACTION_WITHDRAWN = "withdrawn"
#: The reading value or its originating source event changed.
ACTION_SOURCE_CHANGED = "source_changed"

#: Reading value was reported by the sensor (including explicit zero values).
SOURCE_OBSERVED = "observed"
#: Reading value was carried forward from an earlier observed event.
SOURCE_FILLED = "filled"


KeyT = t.Tuple[t.Any, int]


@dataclass(frozen=True)
class SensorEvent:
    """A single sensor event.

    Args:
        device_id: Stable identifier of the device that produced the event.
        event_time: Timestamp of the event. Values must support ordering and
            subtraction (for example :class:`datetime.datetime` paired with a
            :class:`datetime.timedelta` ``max_gap``).
        event_seq: Stable sequence number distinguishing events sharing an
            ``event_time``.
        revision: Monotonic revision number for the ``(event_time, event_seq)``
            key. Retransmissions reuse the current revision while corrections
            publish a higher one.
        value: Measured value. ``0`` is a valid measurement. Ignored for gap
            events.
        observed: ``True`` for measured values, ``False`` for gap markers that
            can only receive carried-forward values.
    """

    device_id: t.Hashable
    event_time: t.Any
    event_seq: int
    revision: int = 0
    value: t.Any = None
    observed: bool = True


@dataclass(frozen=True)
class ImputerReading:
    """A published reading for one event position.

    Args:
        device_id: Device the reading belongs to.
        event_time: Timestamp of the event position.
        event_seq: Sequence number of the event position.
        value: Published value (measured or carried forward).
        source: Either :data:`SOURCE_OBSERVED` or :data:`SOURCE_FILLED`.
        source_event_time: Timestamp of the observed event the value points to.
        source_event_seq: Sequence number of the observed event the value
            points to.
        action: In an :class:`ImputerBatch`, one of the ``ACTION_*`` constants.
            ``None`` for snapshot reads.
    """

    device_id: t.Hashable
    event_time: t.Any
    event_seq: int
    value: t.Any
    source: str
    source_event_time: t.Any
    source_event_seq: int
    action: t.Optional[str] = None


@dataclass(frozen=True)
class ImputerBatch:
    """The readings touched by one :meth:`DeviceImputer.ingest` call.

    Args:
        device_id: Device the batch belongs to.
        readings: Touched readings in ``(event_time, event_seq)`` order.
            Withdrawn readings carry their previous value and source.
    """

    device_id: t.Hashable
    readings: t.Tuple[ImputerReading, ...]

    @property
    def added(self) -> t.Tuple[ImputerReading, ...]:
        """Readings published for the first time."""
        return tuple(reading for reading in self.readings if reading.action == ACTION_ADDED)

    @property
    def withdrawn(self) -> t.Tuple[ImputerReading, ...]:
        """Previously published readings that were retracted."""
        return tuple(reading for reading in self.readings if reading.action == ACTION_WITHDRAWN)

    @property
    def source_changed(self) -> t.Tuple[ImputerReading, ...]:
        """Readings whose value or originating source changed."""
        return tuple(
            reading for reading in self.readings if reading.action == ACTION_SOURCE_CHANGED
        )


@dataclass
class _StoredEvent:
    """Latest retained revision of an event key."""

    revision: int
    value: t.Any
    observed: bool


@dataclass(frozen=True)
class _Output:
    """A computed reading and the observed event it points back to."""

    value: t.Any
    source: str
    source_key: KeyT


class DeviceImputer:
    """Per-device incremental imputer for revision-aware sensor events.

    Args:
        max_gap: Maximum distance a value may be carried forward. The distance
            is always measured from the original observed event's timestamp,
            never from the gap position being filled. Use a
            :class:`datetime.timedelta` for datetime events or a number for
            numeric events.

    Example:

        >>> imputer = DeviceImputer(max_gap=10)
        >>> batch = imputer.ingest(
        ...     [
        ...         SensorEvent("dev-1", 0, 1, value=5),
        ...         SensorEvent("dev-1", 5, 2, observed=False),
        ...     ]
        ... )
        >>> [reading.action for reading in batch["dev-1"].readings]
        ['added', 'added']
        >>> filled = batch["dev-1"].readings[1]
        >>> filled.value, filled.source, filled.source_event_seq
        (5, 'filled', 1)
    """

    def __init__(self, max_gap: t.Any) -> None:
        self.max_gap = max_gap
        self._events: t.Dict[t.Hashable, t.Dict[KeyT, _StoredEvent]] = {}
        self._outputs: t.Dict[t.Hashable, t.Dict[KeyT, _Output]] = {}
        self._event_log: t.List[SensorEvent] = []

    def ingest(
        self, events: Iterable[t.Union[SensorEvent, t.Mapping[str, t.Any]]]
    ) -> t.Dict[t.Hashable, ImputerBatch]:
        """Apply one delivery of events and return the per-device changes.

        Retransmissions of an already seen revision are ignored. Corrections
        replace the stored event and only the affected interval of the device
        is recomputed.

        Args:
            events: Sensor events as :class:`SensorEvent` instances or mappings
                with the same fields.

        Returns:
            Mapping of device id to the :class:`ImputerBatch` describing the
            added, withdrawn, and source-changed readings.
        """
        normalized = [self._coerce_event(event) for event in events]

        grouped: t.Dict[t.Hashable, t.List[SensorEvent]] = {}
        for event in normalized:
            grouped.setdefault(event.device_id, []).append(event)

        result: t.Dict[t.Hashable, ImputerBatch] = {}
        for device_id, device_events in grouped.items():
            candidates = self._merge(device_id, device_events)
            if candidates:
                result[device_id] = self._rebuild(device_id, candidates)
            else:
                result[device_id] = ImputerBatch(device_id, ())

        self._event_log.extend(normalized)
        return result

    def reprocess(
        self,
        events: t.Optional[Iterable[t.Union[SensorEvent, t.Mapping[str, t.Any]]]] = None,
    ) -> t.Dict[t.Hashable, ImputerBatch]:
        """Replay an event stream from scratch.

        Feeding a complete history (including late events) in one pass produces
        the same final readings and source pointers as processing the same
        events incrementally.

        Args:
            events: Full event history. When ``None``, every event previously
                accepted by this imputer is replayed.

        Returns:
            Per-device batches containing every resulting reading, each marked
            :data:`ACTION_ADDED`.
        """
        normalized = (
            [self._coerce_event(event) for event in events]
            if events is not None
            else list(self._event_log)
        )

        fresh = DeviceImputer(self.max_gap)
        batches: t.Dict[t.Hashable, t.List[ImputerReading]] = {}
        for event in normalized:
            batch = fresh.ingest([event])[event.device_id]
            batches.setdefault(batch.device_id, []).extend(batch.readings)

        return {
            device_id: ImputerBatch(device_id, tuple(readings))
            for device_id, readings in batches.items()
        }

    def readings(self, device_id: t.Optional[t.Hashable] = None) -> t.List[ImputerReading]:
        """Return the current published readings.

        Args:
            device_id: Device to read. ``None`` returns readings for every
                device, in device registration order and event position order.

        Raises:
            KeyError: If ``device_id`` is unknown to the imputer.
        """
        if device_id is not None:
            return self._device_readings(device_id)

        result: t.List[ImputerReading] = []
        for known_device in self._outputs:
            result.extend(self._device_readings(known_device))
        return result

    def get_reading(
        self, device_id: t.Hashable, event_time: t.Any, event_seq: int
    ) -> ImputerReading:
        """Return the current reading at an event position.

        Raises:
            KeyError: If the device or event position has no published reading.
        """
        key = (event_time, event_seq)
        try:
            output = self._outputs[device_id][key]
        except KeyError as exc:
            raise KeyError(
                f"No reading for device {device_id!r} at "
                f"(event_time={event_time!r}, event_seq={event_seq!r})"
            ) from exc
        return self._to_reading(device_id, key, output, None)

    def devices(self) -> t.Tuple[t.Hashable, ...]:
        """Return the ids of every device that has accepted events."""
        return tuple(self._events)

    def _merge(self, device_id: t.Hashable, events: t.List[SensorEvent]) -> t.List[KeyT]:
        store = self._events.setdefault(device_id, {})
        candidates: t.List[KeyT] = []

        for event in events:
            key = (event.event_time, event.event_seq)
            existing = store.get(key)
            if existing is not None and event.revision <= existing.revision:
                continue

            store[key] = _StoredEvent(event.revision, event.value, event.observed)
            candidates.append(key)

        return candidates

    def _rebuild(self, device_id: t.Hashable, candidates: t.List[KeyT]) -> ImputerBatch:
        store = self._events[device_id]
        keys = sorted(store)

        low = min(candidates)
        high = max(candidates)

        # Recomputation covers every position from the first touched key up to
        # the observed anchor following the touched interval (or the stream
        # end). Positions before it keep their anchor and are left untouched.
        begin = bisect_left(keys, low)

        anchor_index = begin - 1
        while anchor_index >= 0 and not store[keys[anchor_index]].observed:
            anchor_index -= 1

        end = bisect_right(keys, high)
        probe = end
        while probe < len(keys) and not store[keys[probe]].observed:
            probe += 1
        end = probe + 1 if probe < len(keys) else len(keys)

        if anchor_index >= 0:
            anchor_key = keys[anchor_index]
            anchor_time: t.Any = anchor_key[0]
            anchor_value: t.Any = store[anchor_key].value
            anchor_seq = anchor_key[1]
        else:
            anchor_time = None
            anchor_value = None
            anchor_seq = 0

        new_outputs: t.Dict[KeyT, _Output] = {}
        for index in range(begin, end):
            key = keys[index]
            record = store[key]
            if record.observed:
                new_outputs[key] = _Output(record.value, SOURCE_OBSERVED, key)
                anchor_time = key[0]
                anchor_value = record.value
                anchor_seq = key[1]
            elif anchor_time is not None and key[0] - anchor_time <= self.max_gap:
                new_outputs[key] = _Output(anchor_value, SOURCE_FILLED, (anchor_time, anchor_seq))

        return self._diff(device_id, keys[begin], keys[end - 1], new_outputs)

    def _diff(
        self,
        device_id: t.Hashable,
        low_key: KeyT,
        high_key: KeyT,
        new_outputs: t.Dict[KeyT, _Output],
    ) -> ImputerBatch:
        outputs = self._outputs.setdefault(device_id, {})
        changes: t.List[ImputerReading] = []

        old_keys = [key for key in outputs if low_key <= key <= high_key]
        for key in sorted(set(old_keys).union(new_outputs)):
            old = outputs.get(key)
            new = new_outputs.get(key)

            if old is None and new is not None:
                changes.append(self._to_reading(device_id, key, new, ACTION_ADDED))
            elif old is not None and new is None:
                changes.append(self._to_reading(device_id, key, old, ACTION_WITHDRAWN))
            elif new is not None and old != new:
                changes.append(self._to_reading(device_id, key, new, ACTION_SOURCE_CHANGED))

        for key in old_keys:
            if key not in new_outputs:
                del outputs[key]
        outputs.update(new_outputs)

        return ImputerBatch(device_id, tuple(changes))

    def _device_readings(self, device_id: t.Hashable) -> t.List[ImputerReading]:
        if device_id not in self._events:
            raise KeyError(f"Unknown device: {device_id!r}")

        outputs = self._outputs.get(device_id, {})
        return [self._to_reading(device_id, key, outputs[key], None) for key in sorted(outputs)]

    @staticmethod
    def _to_reading(
        device_id: t.Hashable,
        key: KeyT,
        output: _Output,
        action: t.Optional[str],
    ) -> ImputerReading:
        return ImputerReading(
            device_id=device_id,
            event_time=key[0],
            event_seq=key[1],
            value=output.value,
            source=output.source,
            source_event_time=output.source_key[0],
            source_event_seq=output.source_key[1],
            action=action,
        )

    @staticmethod
    def _coerce_event(event: t.Union[SensorEvent, t.Mapping[str, t.Any]]) -> SensorEvent:
        if isinstance(event, SensorEvent):
            return event

        if not isinstance(event, Mapping):
            raise TypeError(
                f"events must be SensorEvent instances or mappings, got {type(event).__name__}"
            )

        try:
            device_id = event["device_id"]
            event_time = event["event_time"]
            event_seq = event["event_seq"]
        except KeyError as exc:
            raise ValueError(f"Sensor event is missing required field: {exc.args[0]}") from exc

        revision = event.get("revision", 0)
        observed = event.get("observed", True)
        value = event.get("value")

        if not isinstance(event_seq, int) or isinstance(event_seq, bool):
            raise ValueError("Sensor event event_seq must be an integer")
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise ValueError("Sensor event revision must be an integer")
        if revision < 0:
            raise ValueError("Sensor event revision must be non-negative")
        if not isinstance(observed, bool):
            raise ValueError("Sensor event observed must be a boolean")
        if observed and "value" not in event:
            raise ValueError("Observed sensor events require a value")

        return SensorEvent(
            device_id=device_id,
            event_time=event_time,
            event_seq=event_seq,
            revision=revision,
            value=value,
            observed=observed,
        )
