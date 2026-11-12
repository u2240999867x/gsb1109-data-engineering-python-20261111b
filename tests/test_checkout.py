"""Tests for the SortedList range-checkout ledger."""

import random
import time

import pytest

from sortedcontainers import (
    STATUS_AVAILABLE,
    STATUS_COMMITTED,
    STATUS_PENDING,
    CheckoutError,
    CheckoutLedger,
    SortedKeyList,
    SortedList,
)


class Row:
    """Distinct rows that share a group key for equal-key ordering tests."""

    def __init__(self, group, name):
        self.group = group
        self.name = name

    def __repr__(self):
        return f'Row({self.group!r}, {self.name!r})'


def make_key_ledger():
    rows = [
        Row(1, 'a'),
        Row(2, 'old1'),
        Row(2, 'old2'),
        Row(3, 'x'),
    ]
    return CheckoutLedger(SortedKeyList(rows, key=lambda row: row.group))


def test_claim_takes_whole_equal_key_range():
    ledger = CheckoutLedger(SortedList([1, 2, 2, 2, 3]))
    checkout = ledger.claim(2)

    assert checkout.rows == [2, 2, 2]
    assert ledger.available() == [1, 3]
    assert ledger.pending() == [2, 2, 2]
    assert ledger.committed() == []
    assert len(ledger) == 5


def test_concurrent_claims_never_share_rows_but_may_share_keys():
    ledger = CheckoutLedger(SortedList([1, 2, 3, 4, 5]))
    first = ledger.claim(2, 3)
    assert first.rows == [2, 3]

    with pytest.raises(CheckoutError):
        ledger.claim(2)

    second = ledger.claim(4, 5)
    assert second.rows == [4, 5]

    ledger.add(2)
    third = ledger.claim(2)
    assert third.rows == [2]

    assert {c.id for c in ledger.pending_checkouts()} == {
        first.id,
        second.id,
        third.id,
    }
    assert ledger.available() == [1]
    assert sorted(ledger.pending()) == [2, 2, 3, 4, 5]
    assert set(first.seqs).isdisjoint(third.seqs)


def test_new_equal_key_rows_stay_with_main_collection():
    ledger = CheckoutLedger(SortedList([2, 2]))
    checkout = ledger.claim(2)
    assert checkout.rows == [2, 2]

    ledger.add(2)
    assert ledger.available() == [2]
    assert ledger.pending() == [2, 2]

    second = ledger.claim(2)
    assert second.rows == [2]
    assert ledger.available() == []


def test_commit_moves_attribution_once_and_is_idempotent():
    ledger = CheckoutLedger(SortedList([1, 2, 2, 3]))
    checkout = ledger.claim(2)

    assert ledger.commit(checkout).state == 'committed'
    assert ledger.committed() == [2, 2]
    assert ledger.available() == [1, 3]

    committed_before = ledger.committed()
    assert ledger.commit(checkout).state == 'committed'
    assert ledger.return_checkout(checkout).state == 'committed'
    assert ledger.committed() == committed_before
    assert ledger.pending() == []


def test_return_restores_equal_key_order_before_later_arrivals():
    ledger = make_key_ledger()
    checkout = ledger.claim(2)
    assert [row.name for row in checkout.rows] == ['old1', 'old2']

    late = Row(2, 'late')
    ledger.add(late)
    group_two_available = [row.name for row in ledger.available() if row.group == 2]
    assert group_two_available == ['late']

    ledger.return_checkout(checkout)

    names = [row.name for row in ledger.available() if row.group == 2]
    assert names == ['old1', 'old2', 'late']

    available_seqs, pending_seqs, committed_seqs = ledger.reconcile()
    assert [row.name for _, row in available_seqs if row.group == 2] == [
        'old1',
        'old2',
        'late',
    ]
    assert pending_seqs == []
    assert committed_seqs == []


def test_return_then_reclaim_keeps_original_order():
    ledger = make_key_ledger()
    checkout = ledger.claim(2)
    ledger.add(Row(2, 'late'))
    ledger.return_checkout(checkout)

    again = ledger.claim(2)
    assert [row.name for row in again.rows] == ['old1', 'old2', 'late']


def test_failed_return_loses_no_rows():
    ledger = CheckoutLedger(SortedList([1, 2, 2, 3]))
    checkout = ledger.claim(2)

    tampered = ledger._by_seq[checkout.seqs[0]]
    tampered.status = STATUS_COMMITTED

    with pytest.raises(CheckoutError):
        ledger.return_checkout(checkout)

    assert checkout.state == 'pending'
    available, pending, committed = ledger.reconcile()
    assert [row for _, row in available] == [1, 3]
    assert [row for _, row in pending] == [2]
    assert [row for _, row in committed] == [2]

    tampered.status = STATUS_PENDING
    ledger.return_checkout(checkout)
    assert checkout.state == 'returned'
    assert ledger.available() == [1, 2, 2, 3]
    assert ledger.committed() == []


def test_resolve_rejects_foreign_and_bad_handles():
    ledger = CheckoutLedger(SortedList([1]))
    other = CheckoutLedger(SortedList([1]))
    checkout = other.claim(1)

    with pytest.raises(CheckoutError):
        ledger.commit(checkout)
    with pytest.raises(TypeError):
        ledger.commit('not-a-checkout')


def test_reconcile_partitions_every_original_row_number():
    ledger = CheckoutLedger(SortedList(range(10)))
    low = ledger.claim(1, 2)
    high = ledger.claim(7, 8)
    ledger.commit(low)
    ledger.add(10)

    available, pending, committed = ledger.reconcile()
    all_seqs = sorted(
        seq for group in (available, pending, committed) for seq, _ in group
    )
    assert all_seqs == list(range(11))
    assert sorted(seq for seq, _ in committed) == sorted(low.seqs)
    assert sorted(seq for seq, _ in pending) == sorted(high.seqs)
    assert [row for _, row in available] == [0, 3, 4, 5, 6, 9, 10]


def test_small_claim_does_not_copy_other_records():
    ledger = CheckoutLedger(SortedList(range(5000)))
    entry_ids = {seq: id(entry) for seq, entry in ledger._by_seq.items()}

    checkout = ledger.claim(100, 101)
    claimed_seqs = set(checkout.seqs)

    for seq, entry_id in entry_ids.items():
        if seq not in claimed_seqs:
            assert id(ledger._by_seq[seq]) == entry_id

    ledger.return_checkout(checkout)
    for seq, entry_id in entry_ids.items():
        assert id(ledger._by_seq[seq]) == entry_id
    assert len(ledger) == 5000


def test_small_claim_and_return_cost_is_independent_of_total_size():
    ledger = CheckoutLedger(SortedList(range(20000)))
    started = time.perf_counter()
    for _ in range(100):
        checkout = ledger.claim(10000)
        ledger.return_checkout(checkout)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0
    assert ledger.pending() == []


def test_half_open_bounds_and_empty_range_errors():
    ledger = CheckoutLedger(SortedList([1, 2, 2, 3]))
    checkout = ledger.claim(1, 3, inclusive=(True, False))
    assert checkout.rows == [1, 2, 2]

    with pytest.raises(CheckoutError):
        ledger.claim(9, 10)
    with pytest.raises(CheckoutError):
        ledger.claim(3, 4, inclusive=(False, False))


def test_backing_list_and_iterable_input():
    source = SortedList([1, 2])
    ledger = CheckoutLedger(source)
    ledger.add(3)
    assert ledger.source == [1, 2, 3]
    assert source == [1, 2, 3]

    from_iterable = CheckoutLedger(iter([5, 4, 4]))
    assert from_iterable.available() == [4, 4, 5]


def test_keyed_ledger_uses_key_bounds():
    ledger = make_key_ledger()
    checkout = ledger.claim(3)
    assert [row.name for row in checkout.rows] == ['x']
    assert [row.name for row in ledger.available()] == ['a', 'old1', 'old2']


def test_close_blocks_new_claims_and_adds():
    ledger = CheckoutLedger(SortedList([1]))
    checkout = ledger.claim(1)
    ledger.close()

    with pytest.raises(CheckoutError):
        ledger.claim(1)
    with pytest.raises(CheckoutError):
        ledger.add(2)

    ledger.commit(checkout)
    assert ledger.committed() == [1]


def test_status_constants_are_distinct():
    assert len({STATUS_AVAILABLE, STATUS_PENDING, STATUS_COMMITTED}) == 3


def test_randomized_interleaving_matches_reference_model():
    rng = random.Random(20261113)
    ledger = CheckoutLedger(SortedList())

    # Reference model: seq -> (key, status). A row never changes key, and
    # equal-key rows must always stay in ascending seq order.
    model = {}
    pending = []
    next_seq = 0

    def snapshot_statuses():
        return {seq: entry.status for seq, entry in ledger._by_seq.items()}

    for _ in range(4000):
        action = rng.random()

        if action < 0.45:
            key = rng.randrange(5)
            seq = next_seq
            next_seq += 1
            ledger.add(key)
            model[seq] = (key, STATUS_AVAILABLE)
        elif action < 0.75:
            key = rng.randrange(5)
            try:
                checkout = ledger.claim(key)
            except CheckoutError:
                continue
            for seq in checkout.seqs:
                assert model[seq][1] == STATUS_AVAILABLE
                model[seq] = (model[seq][0], STATUS_PENDING)
            pending.append(checkout)
        elif pending:
            checkout = rng.choice(pending)
            if rng.random() < 0.5:
                ledger.commit(checkout)
                target = STATUS_COMMITTED
            else:
                ledger.return_checkout(checkout)
                target = STATUS_AVAILABLE
            for seq in checkout.seqs:
                model[seq] = (model[seq][0], target)
            pending.remove(checkout)

        assert snapshot_statuses() == {seq: state for seq, (_, state) in model.items()}

        # In store order, equal-key entries must keep ascending original
        # sequence numbers even after returns interleaved with late arrivals.
        store_entries = list(ledger._store)
        for index in range(len(store_entries) - 1):
            prev = store_entries[index]
            nxt = store_entries[index + 1]
            if prev.row == nxt.row:
                assert prev.seq < nxt.seq

        available, pending_view, committed = ledger.reconcile()
        for view in (available, pending_view, committed):
            seqs = [seq for seq, _ in view]
            assert seqs == sorted(seqs)

    for checkout in pending:
        ledger.return_checkout(checkout)
        for seq in checkout.seqs:
            model[seq] = (model[seq][0], STATUS_AVAILABLE)

    assert ledger.pending() == []
    assert sorted(row for _, row in ledger.reconcile()[0]) == sorted(
        key for key, state in model.values() if state == STATUS_AVAILABLE
    )
