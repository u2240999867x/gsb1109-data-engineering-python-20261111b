import random

import pytest

from sortedcontainers import (
    AccessCounts,
    InventoryEvent,
    InventoryRanking,
    RankChange,
)


def event(sku, version, quantity, batch=0):
    return InventoryEvent(sku, version, quantity, batch)


def test_initial_ranking_orders_by_descending_quantity():
    ranking = InventoryRanking(
        [
            event('sku-a', 1, 50),
            event('sku-b', 1, 70),
            event('sku-c', 1, 60),
        ]
    )
    assert list(ranking) == ['sku-b', 'sku-c', 'sku-a']
    assert len(ranking) == 3
    assert ranking[0] == 'sku-b'
    assert ranking[-1] == 'sku-a'
    assert ranking[:] == ['sku-b', 'sku-c', 'sku-a']
    assert list(ranking.items()) == [('sku-b', 70), ('sku-c', 60), ('sku-a', 50)]


def test_initial_ranking_reports_new_ranks():
    ranking = InventoryRanking()
    changes = ranking.rebuild([event('sku-a', 1, 50), event('sku-b', 1, 70)])
    assert changes == [
        RankChange('sku-b', None, 0),
        RankChange('sku-a', None, 1),
    ]


def test_ties_broken_by_first_in_sequence():
    ranking = InventoryRanking(
        [
            event('sku-a', 1, 100),
            event('sku-b', 1, 100),
            event('sku-c', 1, 100),
        ]
    )
    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c']

    # Correcting sku-c up to the same quantity does not let it pass the
    # rows that entered the ranking earlier.
    changes = ranking.rebuild([event('sku-c', 2, 100)])
    assert changes == []
    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c']

    # Correcting sku-c up to a shared higher quantity: first-in wins ties.
    ranking.rebuild([event('sku-a', 2, 200), event('sku-c', 3, 200)])
    assert list(ranking) == ['sku-a', 'sku-c', 'sku-b']


def test_correction_reports_crossed_rows():
    ranking = InventoryRanking(
        [
            event('sku-a', 1, 90),
            event('sku-b', 1, 80),
            event('sku-c', 1, 70),
            event('sku-d', 1, 60),
        ]
    )
    changes = ranking.rebuild([event('sku-d', 2, 95)])
    assert list(ranking) == ['sku-d', 'sku-a', 'sku-b', 'sku-c']
    assert changes == [
        RankChange('sku-d', 3, 0),
        RankChange('sku-a', 0, 1),
        RankChange('sku-b', 1, 2),
        RankChange('sku-c', 2, 3),
    ]

    changes = ranking.rebuild([event('sku-d', 3, 65)])
    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c', 'sku-d']
    assert changes == [
        RankChange('sku-a', 1, 0),
        RankChange('sku-b', 2, 1),
        RankChange('sku-c', 3, 2),
        RankChange('sku-d', 0, 3),
    ]


def test_correction_without_rank_change_reports_nothing():
    ranking = InventoryRanking(
        [event('sku-a', 1, 100), event('sku-b', 1, 50), event('sku-c', 1, 10)]
    )
    changes = ranking.rebuild([event('sku-b', 2, 49)])
    assert changes == []
    assert ranking.version('sku-b') == 2
    assert ranking.quantity('sku-b') == 49
    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c']


def test_stale_redelivery_changes_nothing():
    ranking = InventoryRanking([event('sku-a', 2, 100), event('sku-b', 1, 50)])
    before = list(ranking)

    # Older version.
    assert ranking.rebuild([event('sku-a', 1, 999)]) == []
    # Same version, different quantity: still a stale re-delivery.
    assert ranking.rebuild([event('sku-a', 2, 999)]) == []

    assert list(ranking) == before
    assert ranking.quantity('sku-a') == 100
    assert ranking.version('sku-a') == 2


def test_stale_redelivery_makes_no_key_calls_or_comparisons():
    ranking = InventoryRanking([event('sku-a', 2, 100), event('sku-b', 1, 50)])
    ranking.reset_access_counts()
    assert ranking.rebuild([event('sku-a', 1, 999)]) == []
    assert ranking.access_counts == AccessCounts(0, 0)


def test_batch_watermark():
    events = [
        event('sku-a', 1, 90, batch=1),
        event('sku-b', 1, 80, batch=2),
        event('sku-c', 1, 70, batch=3),
    ]
    ranking = InventoryRanking()
    ranking.rebuild(events, batch=2)
    assert list(ranking) == ['sku-a', 'sku-b']
    ranking.rebuild(events, batch=3)
    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c']

    replayed = InventoryRanking(events)
    assert list(replayed) == list(ranking)


def test_new_sku_shifts_crossed_rows():
    ranking = InventoryRanking([event('sku-a', 1, 90), event('sku-b', 1, 70)])
    changes = ranking.rebuild([event('sku-c', 1, 80)])
    assert list(ranking) == ['sku-a', 'sku-c', 'sku-b']
    assert changes == [
        RankChange('sku-c', None, 1),
        RankChange('sku-b', 1, 2),
    ]


def test_multiple_corrections_in_one_rebuild():
    ranking = InventoryRanking(
        [event('sku-a', 1, 90), event('sku-b', 1, 80), event('sku-c', 1, 70)]
    )
    changes = ranking.rebuild(
        [
            event('sku-c', 2, 95),
            event('sku-a', 2, 60),
            event('sku-c', 3, 85),
        ]
    )
    assert list(ranking) == ['sku-c', 'sku-b', 'sku-a']
    # sku-b crossed sku-a but was itself crossed back: no net change.
    assert changes == [
        RankChange('sku-c', 2, 0),
        RankChange('sku-a', 0, 2),
    ]
    assert ranking.version('sku-c') == 3
    assert ranking.quantity('sku-c') == 85


def test_key_error_publishes_no_half_ranking():
    def key(quantity):
        if quantity == 'bad':
            raise ValueError('bad quantity')
        return quantity

    ranking = InventoryRanking([event('sku-a', 1, 90), event('sku-b', 1, 80)], key=key)
    with pytest.raises(ValueError, match='bad quantity'):
        ranking.rebuild([event('sku-a', 2, 'bad')])

    assert list(ranking) == ['sku-a', 'sku-b']
    assert ranking.quantity('sku-a') == 90
    assert ranking.version('sku-a') == 1

    # The ranking still works afterwards.
    changes = ranking.rebuild([event('sku-a', 3, 85)])
    assert changes == []
    assert list(ranking) == ['sku-a', 'sku-b']


def test_comparison_error_publishes_no_half_ranking():
    class Guarded:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            if 'boom' in (self.value, other.value):
                raise TypeError('cannot compare')
            return self.value < other.value

        def __eq__(self, other):
            return self.value == other.value

    ranking = InventoryRanking(
        [event('sku-a', 1, 90), event('sku-b', 1, 80), event('sku-c', 1, 70)],
        key=Guarded,
    )
    with pytest.raises(TypeError, match='cannot compare'):
        ranking.rebuild([event('sku-b', 2, 'boom')])

    assert list(ranking) == ['sku-a', 'sku-b', 'sku-c']
    assert ranking.quantity('sku-b') == 80
    assert ranking.version('sku-b') == 1

    # The ranking still works afterwards.
    changes = ranking.rebuild([event('sku-b', 3, 95)])
    assert list(ranking) == ['sku-b', 'sku-a', 'sku-c']
    assert changes == [
        RankChange('sku-b', 1, 0),
        RankChange('sku-a', 0, 1),
    ]


def test_error_in_one_event_keeps_whole_rebuild_unpublished():
    def key(quantity):
        if quantity == 'bad':
            raise ValueError('bad quantity')
        return quantity

    ranking = InventoryRanking([event('sku-a', 1, 90), event('sku-b', 1, 80)], key=key)
    with pytest.raises(ValueError, match='bad quantity'):
        ranking.rebuild([event('sku-a', 2, 95), event('sku-b', 2, 'bad')])

    # Even the valid first correction is not published.
    assert list(ranking) == ['sku-a', 'sku-b']
    assert ranking.version('sku-a') == 1
    assert ranking.version('sku-b') == 1


def test_access_counts_stay_small_for_small_corrections():
    count = 2000
    ranking = InventoryRanking([event(num, 1, num) for num in range(count)])
    assert ranking.access_counts.key_calls == count

    ranking.reset_access_counts()
    changes = ranking.rebuild([event(1000, 2, 1000.5)])
    assert changes == []
    counts = ranking.access_counts
    assert counts.key_calls == 1
    # Logarithmic in the ranking size, nowhere near a full rebuild.
    assert counts.comparisons < 100

    ranking.reset_access_counts()
    changes = ranking.rebuild([event(1000, 3, 2 * count)])
    assert changes != []
    counts = ranking.access_counts
    assert counts.key_calls == 1
    assert counts.comparisons < 100


def test_custom_key_function():
    ranking = InventoryRanking(
        [event('sku-a', 1, 19), event('sku-b', 1, 91), event('sku-c', 1, 25)],
        key=lambda quantity: quantity % 10,
    )
    assert list(ranking) == ['sku-a', 'sku-c', 'sku-b']
    assert ranking.key is not None


def test_accessors():
    ranking = InventoryRanking([event('sku-a', 1, 90), event('sku-b', 2, 80)])
    assert 'sku-a' in ranking
    assert 'sku-c' not in ranking
    assert ranking.rank('sku-a') == 0
    assert ranking.rank('sku-b') == 1
    assert ranking.quantity('sku-a') == 90
    assert ranking.version('sku-b') == 2
    with pytest.raises(KeyError):
        ranking.rank('sku-c')
    with pytest.raises(KeyError):
        ranking.quantity('sku-c')
    with pytest.raises(KeyError):
        ranking.version('sku-c')
    assert repr(ranking) == "InventoryRanking([('sku-a', 90), ('sku-b', 80)])"


def test_repr_empty():
    assert repr(InventoryRanking()) == 'InventoryRanking([])'


def model_apply(state, seq, events, batch=None):
    "Naive reference model: apply events, return (state, seq, changes)."
    before = model_order(state)
    state = dict(state)
    for sku, version, quantity, event_batch in events:
        if batch is not None and event_batch > batch:
            continue
        current = state.get(sku)
        if current is None:
            state[sku] = (version, quantity, seq)
            seq += 1
        elif version > current[0]:
            state[sku] = (version, quantity, current[2])
    after = model_order(state)
    changes = {}
    for sku, new_rank in after.items():
        old_rank = before.get(sku)
        if old_rank != new_rank:
            changes[sku] = (old_rank, new_rank)
    return state, seq, changes


def model_order(state):
    "Map of sku to rank ordered by descending quantity, ties by sequence."
    skus = sorted(state, key=lambda sku: (-state[sku][1], state[sku][2]))
    return {sku: rank for rank, sku in enumerate(skus)}


def test_random_rebuilds_match_reference_model():
    rng = random.Random(20261111)
    ranking = InventoryRanking()
    state = {}
    seq = 0
    sku_pool = [f'sku-{num}' for num in range(50)]

    for batch in range(1, 101):
        events = []
        for _ in range(rng.randrange(10)):
            sku = rng.choice(sku_pool)
            current = state.get(sku)
            if current is not None and rng.random() < 0.3:
                # Stale re-delivery at or below the recorded version.
                version = rng.randrange(0, current[0] + 1)
            elif current is None:
                version = rng.randrange(1, 4)
            else:
                version = current[0] + rng.randrange(1, 3)
            quantity = rng.randrange(0, 200)
            events.append(event(sku, version, quantity, batch=batch))

        watermark = batch if rng.random() < 0.9 else batch - 1
        changes = ranking.rebuild(events, batch=watermark)
        state, seq, expected = model_apply(state, seq, events, batch=watermark)

        reported = {
            change.sku: (change.old_rank, change.new_rank) for change in changes
        }
        assert reported == expected

        order = model_order(state)
        assert list(ranking) == sorted(order, key=order.get)
        assert len(ranking) == len(state)
        for sku, (version, quantity, _) in state.items():
            assert ranking.rank(sku) == order[sku]
            assert ranking.version(sku) == version
            assert ranking.quantity(sku) == quantity
