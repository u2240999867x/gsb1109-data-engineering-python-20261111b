from datetime import datetime, timedelta
import random

import pytest

import pydash as _
from pydash.timeseries import (
    ACTION_ADDED,
    ACTION_SOURCE_CHANGED,
    ACTION_WITHDRAWN,
    SOURCE_FILLED,
    SOURCE_OBSERVED,
    DeviceImputer,
    SensorEvent,
)


parametrize = pytest.mark.parametrize


def make_imputer(events=None, max_gap=10):
    imputer = DeviceImputer(max_gap=max_gap)
    if events:
        imputer.ingest(events)
    return imputer


def reading_tuples(imputer, device_id):
    return [
        (
            reading.event_time,
            reading.event_seq,
            reading.value,
            reading.source,
            reading.source_event_time,
            reading.source_event_seq,
        )
        for reading in imputer.readings(device_id)
    ]


def test_forward_fill_carries_last_observed_value():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 3, 2, observed=False),
            SensorEvent("d1", 6, 3, observed=False),
        ]
    )

    readings = imputer.readings("d1")
    assert [reading.value for reading in readings] == [5, 5, 5]
    assert [reading.source for reading in readings] == [
        SOURCE_OBSERVED,
        SOURCE_FILLED,
        SOURCE_FILLED,
    ]
    filled = readings[1]
    assert filled.source_event_time == 0
    assert filled.source_event_seq == 1
    assert imputer.get_reading("d1", 6, 3).source_event_seq == 1


def test_zero_is_an_observed_value_and_is_carried_forward():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=0),
            SensorEvent("d1", 4, 2, observed=False),
        ]
    )

    observed = imputer.get_reading("d1", 0, 1)
    assert observed.value == 0
    assert observed.source == SOURCE_OBSERVED

    filled = imputer.get_reading("d1", 4, 2)
    assert filled.value == 0
    assert filled.source == SOURCE_FILLED
    assert filled.source_event_seq == 1


def test_max_gap_is_measured_from_the_original_observed_time():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 6, 2, observed=False),
            SensorEvent("d1", 10, 3, observed=False),
            SensorEvent("d1", 11, 4, observed=False),
        ],
        max_gap=10,
    )

    assert imputer.get_reading("d1", 10, 3).value == 5
    with pytest.raises(KeyError):
        imputer.get_reading("d1", 11, 4)


def test_max_gap_with_datetimes_and_timedelta():
    origin = datetime(2026, 1, 1)
    imputer = DeviceImputer(max_gap=timedelta(minutes=10))
    imputer.ingest(
        [
            SensorEvent("d1", origin, 1, value=7),
            SensorEvent("d1", origin + timedelta(minutes=9), 2, observed=False),
            SensorEvent("d1", origin + timedelta(minutes=10), 3, observed=False),
            SensorEvent("d1", origin + timedelta(minutes=11), 4, observed=False),
        ]
    )

    assert imputer.get_reading("d1", origin + timedelta(minutes=9), 2).value == 7
    assert imputer.get_reading("d1", origin + timedelta(minutes=10), 3).value == 7
    with pytest.raises(KeyError):
        imputer.get_reading("d1", origin + timedelta(minutes=11), 4)


def test_gap_before_first_observed_event_has_no_reading():
    imputer = make_imputer(
        [
            SensorEvent("d1", 5, 2, observed=False),
            SensorEvent("d1", 9, 1, value=3),
        ]
    )

    assert [reading.event_seq for reading in imputer.readings("d1")] == [1]


def test_retransmission_of_same_revision_is_ignored():
    imputer = make_imputer([SensorEvent("d1", 0, 1, value=5)])
    batch = imputer.ingest([SensorEvent("d1", 0, 1, revision=0, value=5)])

    assert batch["d1"].readings == ()
    assert batch["d1"].added == ()
    assert imputer.get_reading("d1", 0, 1).value == 5


def test_older_revision_is_ignored():
    imputer = make_imputer([SensorEvent("d1", 0, 1, revision=2, value=5)])
    batch = imputer.ingest([SensorEvent("d1", 0, 1, revision=1, value=99)])

    assert batch["d1"].readings == ()
    assert imputer.get_reading("d1", 0, 1).value == 5


def test_duplicate_key_in_one_delivery_keeps_highest_revision():
    imputer = DeviceImputer(max_gap=10)
    batch = imputer.ingest(
        [
            SensorEvent("d1", 0, 1, revision=0, value=5),
            SensorEvent("d1", 4, 2, observed=False),
            SensorEvent("d1", 0, 1, revision=1, value=9),
        ]
    )

    by_seq = {reading.event_seq: reading for reading in batch["d1"].readings}
    assert by_seq[1].value == 9
    assert by_seq[2].value == 9
    assert by_seq[1].action == ACTION_ADDED
    assert imputer.get_reading("d1", 0, 1).value == 9


def test_stale_retransmission_alongside_correction_in_one_delivery():
    imputer = make_imputer([SensorEvent("d1", 0, 1, revision=1, value=5)])
    batch = imputer.ingest(
        [
            SensorEvent("d1", 0, 1, revision=1, value=5),
            SensorEvent("d1", 0, 1, revision=2, value=8),
        ]
    )

    assert [reading.event_seq for reading in batch["d1"].readings] == [1]
    assert batch["d1"].readings[0].action == ACTION_SOURCE_CHANGED
    assert imputer.get_reading("d1", 0, 1).value == 8


def test_correction_of_observed_value_emits_source_changed_and_reanchors_fills():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 4, 2, observed=False),
            SensorEvent("d1", 8, 3, observed=False),
        ]
    )

    batch = imputer.ingest([SensorEvent("d1", 0, 1, revision=1, value=9)])
    actions = {reading.event_seq: reading.action for reading in batch["d1"].readings}

    assert actions == {1: ACTION_SOURCE_CHANGED, 2: ACTION_SOURCE_CHANGED, 3: ACTION_SOURCE_CHANGED}
    assert [reading.value for reading in batch["d1"].readings] == [9, 9, 9]
    assert all(reading.source_event_seq == 1 for reading in batch["d1"].readings)


def test_correction_observed_to_gap_withdraws_anchor_and_dependent_fills():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 4, 2, observed=False),
            SensorEvent("d1", 8, 3, observed=False),
            SensorEvent("d1", 12, 4, value=2),
        ]
    )

    batch = imputer.ingest([SensorEvent("d1", 0, 1, revision=1, observed=False)])
    actions = {reading.event_seq: reading.action for reading in batch["d1"].readings}

    assert actions == {1: ACTION_WITHDRAWN, 2: ACTION_WITHDRAWN, 3: ACTION_WITHDRAWN}
    withdrawn_fill = next(reading for reading in batch["d1"].readings if reading.event_seq == 2)
    assert withdrawn_fill.value == 5
    assert withdrawn_fill.source == SOURCE_FILLED

    remaining = {reading.event_seq: reading for reading in imputer.readings("d1")}
    assert set(remaining) == {4}
    assert remaining[4].value == 2


def test_late_observed_event_becomes_new_anchor_for_later_gaps():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 10, 3, observed=False),
        ]
    )

    batch = imputer.ingest([SensorEvent("d1", 6, 2, value=7)])
    actions = {reading.event_seq: reading.action for reading in batch["d1"].readings}

    assert actions == {2: ACTION_ADDED, 3: ACTION_SOURCE_CHANGED}
    changed = next(reading for reading in batch["d1"].readings if reading.event_seq == 3)
    assert changed.value == 7
    assert changed.source_event_seq == 2


def test_correction_gap_to_observed_reanchors_downstream():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 4, 2, observed=False),
            SensorEvent("d1", 8, 3, observed=False),
        ]
    )

    batch = imputer.ingest([SensorEvent("d1", 4, 2, revision=1, value=9)])
    actions = {reading.event_seq: reading.action for reading in batch["d1"].readings}

    assert actions == {2: ACTION_SOURCE_CHANGED, 3: ACTION_SOURCE_CHANGED}
    by_seq = {reading.event_seq: reading for reading in batch["d1"].readings}
    assert by_seq[2].source == SOURCE_OBSERVED
    assert by_seq[3].source == SOURCE_FILLED
    assert by_seq[3].source_event_seq == 2
    assert by_seq[3].value == 9


def test_only_affected_interval_is_recomputed():
    imputer = make_imputer(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 3, 2, observed=False),
            SensorEvent("d1", 20, 3, value=8),
            SensorEvent("d1", 23, 4, observed=False),
        ]
    )

    batch = imputer.ingest([SensorEvent("d1", 20, 3, revision=1, value=80)])
    touched = {reading.event_seq for reading in batch["d1"].readings}

    assert touched == {3, 4}
    assert imputer.get_reading("d1", 3, 2).value == 5
    assert imputer.get_reading("d1", 3, 2).action is None


def test_same_timestamp_sequence_numbers_define_order():
    imputer = DeviceImputer(max_gap=0)
    imputer.ingest(
        [
            SensorEvent("d1", 5, 2, value=1),
            SensorEvent("d1", 5, 1, observed=False),
        ]
    )

    readings = imputer.readings("d1")
    assert [reading.event_seq for reading in readings] == [2]
    assert readings[0].source == SOURCE_OBSERVED


def test_devices_are_imputed_independently():
    imputer = make_imputer(
        [
            SensorEvent("a", 0, 1, value=1),
            SensorEvent("b", 0, 1, value=2),
            SensorEvent("b", 5, 2, observed=False),
        ]
    )

    batch = imputer.ingest([SensorEvent("a", 0, 1, revision=1, value=100)])

    assert set(batch) == {"a"}
    assert batch["a"].readings[0].value == 100
    assert imputer.get_reading("b", 5, 2).value == 2
    assert set(imputer.devices()) == {"a", "b"}


def test_batch_convenience_groupings():
    imputer = DeviceImputer(max_gap=10)
    imputer.ingest(
        [
            SensorEvent("d1", 0, 1, value=5),
            SensorEvent("d1", 4, 2, observed=False),
        ]
    )
    batch = imputer.ingest(
        [
            SensorEvent("d1", 0, 1, revision=1, observed=False),
            SensorEvent("d1", 9, 3, observed=False),
        ]
    )

    assert {reading.event_seq for reading in batch["d1"].added} == set()
    assert {reading.event_seq for reading in batch["d1"].withdrawn} == {1, 2}
    assert {reading.event_seq for reading in batch["d1"].source_changed} == set()


def test_readings_returns_all_devices_in_registration_order():
    imputer = make_imputer(
        [
            SensorEvent("a", 0, 1, value=1),
            SensorEvent("b", 0, 1, value=2),
            SensorEvent("b", 5, 2, observed=False),
        ]
    )

    readings = imputer.readings()
    assert [(reading.device_id, reading.event_seq) for reading in readings] == [
        ("a", 1),
        ("b", 1),
        ("b", 2),
    ]
    assert all(reading.action is None for reading in readings)


def test_unknown_device_and_missing_reading_raise_key_error():
    imputer = make_imputer([SensorEvent("d1", 0, 1, value=5)])

    with pytest.raises(KeyError):
        imputer.readings("ghost")
    with pytest.raises(KeyError):
        imputer.get_reading("d1", 99, 9)


def test_mapping_events_are_coerced():
    imputer = DeviceImputer(max_gap=10)
    batch = imputer.ingest(
        [
            {"device_id": "x", "event_time": 0, "event_seq": 1, "value": 3},
            {
                "device_id": "x",
                "event_time": 4,
                "event_seq": 2,
                "revision": 2,
                "observed": False,
            },
        ]
    )

    assert [reading.value for reading in batch["x"].readings] == [3, 3]
    assert batch["x"].readings[1].source == SOURCE_FILLED


@parametrize(
    "event,message",
    [
        ({"event_time": 0, "event_seq": 1, "value": 1}, "device_id"),
        ({"device_id": "x", "event_seq": 1, "value": 1}, "event_time"),
        ({"device_id": "x", "event_time": 0, "value": 1}, "event_seq"),
    ],
)
def test_missing_required_field_raises_value_error(event, message):
    imputer = DeviceImputer(max_gap=10)
    with pytest.raises(ValueError, match=message):
        imputer.ingest([event])


@parametrize(
    "event",
    [
        {"device_id": "x", "event_time": 0, "event_seq": 1},
        {"device_id": "x", "event_time": 0, "event_seq": 1, "revision": -1, "value": 1},
        {"device_id": "x", "event_time": 0, "event_seq": "1", "value": 1},
        {"device_id": "x", "event_time": 0, "event_seq": 1, "revision": "1", "value": 1},
        {"device_id": "x", "event_time": 0, "event_seq": 1, "revision": True, "value": 1},
        {
            "device_id": "x",
            "event_time": 0,
            "event_seq": 1,
            "revision": 1,
            "observed": "yes",
            "value": 1,
        },
    ],
)
def test_invalid_fields_raise_value_error(event):
    imputer = DeviceImputer(max_gap=10)
    with pytest.raises(ValueError):
        imputer.ingest([event])


def test_non_mapping_event_raises_type_error():
    imputer = DeviceImputer(max_gap=10)
    with pytest.raises(TypeError):
        imputer.ingest([42])


def test_empty_ingest_returns_empty_result():
    imputer = DeviceImputer(max_gap=10)
    assert imputer.ingest([]) == {}
    assert imputer.devices() == ()
    assert imputer.readings() == []


def test_stale_only_delivery_returns_empty_batch_per_device():
    imputer = make_imputer([SensorEvent("d1", 0, 1, value=5)])
    result = imputer.ingest(
        [
            SensorEvent("d1", 0, 1, revision=0, value=5),
            SensorEvent("d2", 0, 1, revision=0, value=1),
        ]
    )

    assert result["d1"].readings == ()
    assert result["d2"].readings[0].action == ACTION_ADDED


def reference_outputs(events, max_gap):
    """Brute-force reference: keep highest revision per key then sweep once."""
    latest = {}
    for event in events:
        key = (event.event_time, event.event_seq)
        if key not in latest or event.revision > latest[key].revision:
            latest[key] = event

    outputs = {}
    anchor_key = None
    anchor_value = None
    for key in sorted(latest):
        record = latest[key]
        if record.observed:
            outputs[key] = (record.value, SOURCE_OBSERVED, key)
            anchor_key = key
            anchor_value = record.value
        elif anchor_key is not None and key[0] - anchor_key[0] <= max_gap:
            outputs[key] = (anchor_value, SOURCE_FILLED, anchor_key)
    return outputs


def build_history(rng):
    events = []
    for seq in range(1, 9):
        event_time = seq * 3
        observed = rng.random() < 0.6
        revisions = rng.sample(range(4), k=rng.randint(1, 3))
        for revision in sorted(revisions):
            if observed:
                event = SensorEvent(
                    "d1",
                    event_time,
                    seq,
                    revision=revision,
                    value=rng.randint(0, 5),
                )
            else:
                event = SensorEvent("d1", event_time, seq, revision=revision, observed=False)
            events.append(event)
    return events


def test_incremental_matches_reference_under_random_delivery():
    rng = random.Random(20261113)
    history = build_history(rng)
    shuffled = history[:]
    rng.shuffle(shuffled)

    imputer = DeviceImputer(max_gap=10)
    for event in shuffled:
        imputer.ingest([event])

    expected = reference_outputs(history, 10)
    actual = {
        (reading.event_time, reading.event_seq): (
            reading.value,
            reading.source,
            (reading.source_event_time, reading.source_event_seq),
        )
        for reading in imputer.readings("d1")
    }
    assert actual == expected


def test_one_shot_reprocess_matches_incremental_processing():
    rng = random.Random(424242)
    history = build_history(rng)
    shuffled = history[:]
    rng.shuffle(shuffled)

    incremental = DeviceImputer(max_gap=10)
    for event in shuffled:
        incremental.ingest([event])

    replay = DeviceImputer(max_gap=10).reprocess(history)["d1"]
    final = {}
    for reading in replay.readings:
        final[(reading.event_time, reading.event_seq)] = (
            reading.value,
            reading.source,
            (reading.source_event_time, reading.source_event_seq),
        )

    incremental_view = {
        (reading.event_time, reading.event_seq): (
            reading.value,
            reading.source,
            (reading.source_event_time, reading.source_event_seq),
        )
        for reading in incremental.readings("d1")
    }
    assert final == incremental_view


def test_reprocess_without_arguments_replays_the_event_log():
    imputer = DeviceImputer(max_gap=10)
    history = [
        SensorEvent("d1", 0, 1, value=5),
        SensorEvent("d1", 4, 2, observed=False),
        SensorEvent("d1", 0, 1, revision=1, value=9),
    ]
    for event in history:
        imputer.ingest([event])

    replay = imputer.reprocess()
    final = {}
    for reading in replay["d1"].readings:
        final[(reading.event_time, reading.event_seq)] = reading.value

    assert final == {(0, 1): 9, (4, 2): 9}


def test_reprocess_accepts_mappings_and_multiple_devices():
    imputer = DeviceImputer(max_gap=10)
    batches = imputer.reprocess(
        [
            {"device_id": "a", "event_time": 0, "event_seq": 1, "value": 1},
            {"device_id": "a", "event_time": 5, "event_seq": 2, "observed": False},
            {"device_id": "b", "event_time": 0, "event_seq": 1, "value": 2},
        ]
    )

    assert set(batches) == {"a", "b"}
    assert [reading.value for reading in batches["a"].readings] == [1, 1]
    assert all(reading.action == ACTION_ADDED for reading in batches["b"].readings)


def test_public_api_symbols_are_exported():
    for symbol in (
        "DeviceImputer",
        "SensorEvent",
        "ImputerReading",
        "ImputerBatch",
        "ACTION_ADDED",
        "ACTION_WITHDRAWN",
        "ACTION_SOURCE_CHANGED",
        "SOURCE_OBSERVED",
        "SOURCE_FILLED",
    ):
        assert hasattr(_, symbol)
