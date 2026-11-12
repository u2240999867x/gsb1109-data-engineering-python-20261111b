"""Dual-index regression tests for :class:`SortedSet`.

Scenario: image labels are de-duplicated by a *stable entity identity* but
ranked by *hotness*.  ``SortedSet`` maintains two independent indices for
that:

* ``_set``  -- the identity ledger (hash table, membership / merge rules)
* ``_list`` -- the rank ledger (``SortedKeyList``, hotness order plus a
  cached ``_keys`` table)

The tests below never trust the final label values alone.  An independent
identity ledger and rank ledger are rebuilt from scratch after every
operation and used to cross-check member counts, representatives and
positions for intersection, union, difference and symmetric difference.

A second group separates two failure modes that look superficially alike:

* legitimate equal sort keys (distinct entities, same hotness) -- healthy;
* an externally mutated hash (broken identity key) -- the stored object
  becomes unfindable by its own hash, which corrupts identity merges and
  index write-backs.

A final group replays interleaved operation logs and locates the *first*
diverging step, classifying it as an identity-merge fault, a sort-key
extraction fault or an index write-back fault.
"""

import platform
from functools import total_ordering

import pytest

from sortedcontainers import SortedSet


def hotness_key(tag):
    """Rank ledger key: labels are ordered by hotness."""
    return tag.hotness


@total_ordering
class ImageTag:
    """Image label with a stable identity and mutable hotness.

    Equality and the healthy hash depend only on ``identity`` so that two
    labels fetched from different sources for the same entity collapse into
    one set member.  Setting ``corrupt_hash`` simulates external mutation of
    the hash without touching the stored object's position.
    """

    def __init__(self, identity, hotness, source='seed'):
        self.identity = identity
        self.hotness = hotness
        self.source = source
        self.corrupt_hash = None

    def __eq__(self, other):
        if not isinstance(other, ImageTag):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self):
        if self.corrupt_hash is None:
            return hash(self.identity)
        return self.corrupt_hash

    def __lt__(self, other):
        return (self.hotness, self.identity) < (other.hotness, other.identity)

    def __repr__(self):
        return f'ImageTag(id={self.identity}, hot={self.hotness}, src={self.source!r})'


# ---------------------------------------------------------------------------
# Independent reference ledgers
# ---------------------------------------------------------------------------


def flatten_rank_keys(sorted_set):
    """Flatten the rank ledger's cached key table (one entry per member)."""
    return [key for sublist in sorted_set._list._keys for key in sublist]


def physical_identities(sorted_set):
    """Identities physically held in the identity ledger."""
    return [tag.identity for tag in sorted_set._set]


def fresh_rank_order(sorted_set):
    """Rank order rebuilt independently from the identity ledger.

    Ties on hotness are broken by identity so the expected position of every
    member is fully determined without consulting ``SortedSet``.
    """
    members = sorted(sorted_set._set, key=lambda tag: (tag.hotness, tag.identity))
    return [tag.identity for tag in members]


def assert_dual_index(sorted_set, expected_ids):
    """Cross-check both ledgers against an independently built oracle.

    ``expected_ids`` is the plain set of identities the operation log says
    the set ought to hold.
    """
    expected_ids = set(expected_ids)

    # --- identity ledger -------------------------------------------------
    physical = list(sorted_set._set)
    phys_ids = [tag.identity for tag in physical]
    assert len(phys_ids) == len(set(phys_ids)), 'duplicate identity in identity ledger'
    assert set(phys_ids) == expected_ids
    assert len(sorted_set) == len(sorted_set._set)
    # Hash contract: every stored member must still find its own bucket.
    assert all(tag in sorted_set._set for tag in physical)

    # --- index write-back: the rank ledger mirrors the identity ledger ---
    list_ids = [tag.identity for tag in sorted_set._list]
    assert len(sorted_set._list) == len(sorted_set._set)
    assert sorted(list_ids) == sorted(phys_ids)
    assert all(tag in sorted_set._set for tag in sorted_set._list)

    # --- rank ledger: order and cached sort keys -------------------------
    visible_ids = [tag.identity for tag in sorted_set]
    assert visible_ids == fresh_rank_order(sorted_set)
    fresh_keys = sorted(hotness_key(tag) for tag in sorted_set._set)
    assert flatten_rank_keys(sorted_set) == fresh_keys

    # Position round-trips through the rank ledger.
    for position, tag in enumerate(sorted_set):
        assert sorted_set.index(tag) == position
        assert sorted_set[position] is tag

    sorted_set._check()


# ---------------------------------------------------------------------------
# De-duplication by identity, ranking by hotness
# ---------------------------------------------------------------------------


def test_same_identity_from_two_sources_collapses_to_one_representative():
    primary = ImageTag(7, hotness=10, source='primary')
    replica = ImageTag(7, hotness=99, source='replica')

    sorted_set = SortedSet([primary], key=hotness_key)
    sorted_set.add(replica)

    # The first inserted object wins the identity merge; the replica must
    # never become a second member even though its hotness differs.
    assert len(sorted_set) == 1
    representative = next(iter(sorted_set._set))
    assert representative is primary
    assert_dual_index(sorted_set, {7})
    assert [tag.identity for tag in sorted_set] == [7]
    assert sorted_set[0] is primary


def test_equal_hotness_distinct_entities_are_both_retained():
    left = ImageTag(1, hotness=50, source='a')
    right = ImageTag(2, hotness=50, source='b')

    sorted_set = SortedSet([right, left], key=hotness_key)
    sorted_set._reset(4)

    assert len(sorted_set) == 2
    # Equal sort keys are legitimate: both entities survive, ties broken by
    # identity in the independent oracle.
    assert [tag.identity for tag in sorted_set] == [1, 2]
    cached = flatten_rank_keys(sorted_set)
    assert cached == [50, 50]
    # Both members remain individually addressable inside the equal-key run.
    assert sorted_set.index(left) == 0
    assert sorted_set.index(right) == 1
    assert sorted_set.bisect_key_left(50) == 0
    assert sorted_set.bisect_key_right(50) == 2
    assert_dual_index(sorted_set, {1, 2})


def test_rank_positions_follow_hotness_under_shuffled_inserts():
    tags = [ImageTag(identity, hotness=identity * 7) for identity in range(12)]
    shuffled = [tags[index] for index in (5, 0, 11, 1, 9, 3, 7, 2, 10, 4, 8, 6)]

    sorted_set = SortedSet(key=hotness_key)
    for tag in shuffled:
        sorted_set.add(tag)
    sorted_set._reset(4)

    expected_order = sorted(range(12), key=lambda identity: identity * 7)
    assert [tag.identity for tag in sorted_set] == expected_order
    for position, identity in enumerate(expected_order):
        tag = tags[identity]
        assert sorted_set.index(tag) == position
        assert sorted_set[position] is tag
    assert_dual_index(sorted_set, set(range(12)))


def test_interleaved_add_and_discard_keeps_both_ledgers_in_sync():
    sorted_set = SortedSet(key=hotness_key)
    oracle = set()
    tags = {
        identity: ImageTag(identity, hotness=100 - identity) for identity in range(10)
    }

    for identity in (3, 1, 4, 1, 5, 9, 2, 6):
        sorted_set.add(tags[identity])
        oracle.add(identity)
        assert_dual_index(sorted_set, oracle)

    for identity in (4, 9, 1, 100):  # 100 is absent: discard must be a no-op
        sorted_set.discard(tags.get(identity, ImageTag(identity, 0)))
        oracle.discard(identity)
        assert_dual_index(sorted_set, oracle)

    assert sorted(oracle) == [2, 3, 5, 6]


# ---------------------------------------------------------------------------
# Set operations: independent identity and rank ledgers
# ---------------------------------------------------------------------------


def build_source(identities, source, hotness_offset):
    tags = [
        ImageTag(identity, hotness=identity * 10 + hotness_offset, source=source)
        for identity in identities
    ]
    sorted_set = SortedSet(tags, key=hotness_key)
    sorted_set._reset(4)
    return sorted_set, {tag.identity: tag for tag in tags}


def assert_operation_ledgers(result, expected_ids, source_sets):
    assert_dual_index(result, expected_ids)
    # Every adopted representative must be traceable, object identity and
    # all, back to one of the operand ledgers.
    for tag in result:
        donors = [tag is member for source in source_sets for member in source._set]
        assert any(donors), f'representative {tag!r} comes from no operand'


def test_intersection_union_difference_symdiff_counts_and_positions():
    alpha, alpha_by_id = build_source(range(20), 'A', hotness_offset=0)
    beta, beta_by_id = build_source(range(10, 30), 'B', hotness_offset=5)

    alpha_ids = set(range(20))
    beta_ids = set(range(10, 30))

    intersect = alpha & beta
    assert_operation_ledgers(intersect, alpha_ids & beta_ids, [alpha, beta])
    assert len(intersect) == 10

    union = alpha | beta
    assert_operation_ledgers(union, alpha_ids | beta_ids, [alpha, beta])
    assert len(union) == 30

    difference = alpha - beta
    assert_operation_ledgers(difference, alpha_ids - beta_ids, [alpha, beta])
    assert len(difference) == 10

    symdiff = alpha ^ beta
    assert_operation_ledgers(symdiff, alpha_ids ^ beta_ids, [alpha, beta])
    assert len(symdiff) == 20

    # Non-mutating operations leave the operand ledgers untouched.
    assert_dual_index(alpha, alpha_ids)
    assert_dual_index(beta, beta_ids)


def test_intersection_representative_traces_to_an_operand():
    big, big_by_id = build_source(range(20), 'A', hotness_offset=0)
    small, small_by_id = build_source(range(3, 5), 'B', hotness_offset=5)

    result = big.intersection(small)
    assert set(tag.identity for tag in result) == {3, 4}

    # Guaranteed contract: each adopted representative is an exact object
    # held by one of the operands -- never a synthesized value. A plain list
    # is required because a set would itself collapse the equal-identity
    # twins into one member.
    operand_objects = [*big._set, *small._set]
    assert all(any(tag is member for member in operand_objects) for tag in result)
    for identity in (3, 4):
        representative = next(tag for tag in result if tag.identity == identity)
        assert (
            representative is big_by_id[identity]
            or representative is small_by_id[identity]
        )

    # The adopted representative's hotness (not the discarded twin's) is the
    # sort key actually cached in the rank ledger.
    expected_keys = sorted(hotness_key(tag) for tag in result)
    assert flatten_rank_keys(result) == expected_keys


@pytest.mark.skipif(
    platform.python_implementation() != 'CPython',
    reason='CPython set intersection scans the smaller operand and adopts its objects',
)
def test_cpython_intersection_scans_the_smaller_operand():
    big, big_by_id = build_source(range(20), 'A', hotness_offset=0)
    small, small_by_id = build_source(range(3, 5), 'B', hotness_offset=5)

    result = big.intersection(small)
    for identity in (3, 4):
        representative = next(tag for tag in result if tag.identity == identity)
        assert representative is small_by_id[identity]
        assert representative.source == 'B'

    reverse = small.intersection(big)
    for identity in (3, 4):
        representative = next(tag for tag in reverse if tag.identity == identity)
        assert representative is big_by_id[identity]
        assert representative.source == 'A'

    assert [tag.identity for tag in result] == [3, 4]
    assert flatten_rank_keys(result) == [35, 45]


def test_union_and_difference_keep_the_left_operand_representative():
    alpha, alpha_by_id = build_source(range(8), 'A', hotness_offset=0)
    beta, beta_by_id = build_source(range(4, 12), 'B', hotness_offset=9)

    union = alpha.union(beta)
    union_reps = {tag.identity: tag for tag in union}
    for identity in range(4, 8):  # overlapping identities
        assert union_reps[identity] is alpha_by_id[identity]

    difference = alpha.difference(beta)
    assert all(tag.source == 'A' for tag in difference)
    assert {tag.identity for tag in difference} == set(range(4))

    symdiff = alpha.symmetric_difference(beta)
    symdiff_reps = {tag.identity: tag for tag in symdiff}
    for identity in range(4):
        assert symdiff_reps[identity] is alpha_by_id[identity]
    for identity in range(8, 12):
        assert symdiff_reps[identity] is beta_by_id[identity]


def test_in_place_set_operations_rebuild_both_ledgers():
    alpha, _ = build_source(range(20), 'A', hotness_offset=0)
    beta, _ = build_source(range(10, 30), 'B', hotness_offset=5)
    beta_ids = set(range(10, 30))

    alpha &= beta
    assert_dual_index(alpha, set(range(10, 20)))
    assert_dual_index(beta, beta_ids)

    alpha |= SortedSet([ImageTag(40, 400, 'C')], key=hotness_key)
    assert_dual_index(alpha, set(range(10, 20)) | {40})

    # Remove through the representative actually indexed in the rank ledger
    # (intersection may have adopted B's object with B's hotness); an equal
    # identity with a mismatched sort key is a separate fault class covered
    # by the corruption tests below.
    indexed_15 = next(tag for tag in alpha if tag.identity == 15)
    alpha -= SortedSet([indexed_15], key=hotness_key)
    assert_dual_index(alpha, (set(range(10, 20)) | {40}) - {15})

    indexed_11 = next(tag for tag in alpha if tag.identity == 11)
    alpha ^= SortedSet([indexed_11, ImageTag(99, 990, 'Z')], key=hotness_key)
    expected = (set(range(10, 20)) | {40}) - {15}
    assert_dual_index(alpha, expected ^ {11, 99})


# ---------------------------------------------------------------------------
# Legitimate equal keys vs. externally corrupted hash
# ---------------------------------------------------------------------------


def test_legitimate_equal_keys_survive_set_operations():
    left = ImageTag(1, hotness=50, source='a')
    right = ImageTag(2, hotness=50, source='b')
    sorted_set = SortedSet([left, right], key=hotness_key)
    other = SortedSet([right, ImageTag(3, hotness=10, source='c')], key=hotness_key)

    assert len(sorted_set) == 2
    assert left in sorted_set and right in sorted_set
    result = sorted_set | other
    assert_dual_index(result, {1, 2, 3})
    assert flatten_rank_keys(result) == [10, 50, 50]


def test_corrupted_hash_breaks_membership_but_not_physical_storage():
    tag = ImageTag(7, hotness=50)
    sorted_set = SortedSet([tag], key=hotness_key)
    assert_dual_index(sorted_set, {7})

    tag.corrupt_hash = 9999  # external mutation of the identity hash

    # The object is physically still stored in both ledgers...
    assert len(sorted_set._set) == 1
    assert len(sorted_set._list) == 1
    assert list(sorted_set._set)[0] is tag
    # ...but it can no longer find its own hash bucket.
    assert tag not in sorted_set
    assert tag not in sorted_set._set
    assert sorted_set.count(tag) == 0


def test_corrupted_hash_discard_is_a_silent_identity_ledger_noop():
    tag = ImageTag(7, hotness=50)
    sorted_set = SortedSet([tag], key=hotness_key)
    intended = {7}

    tag.corrupt_hash = 9999
    sorted_set.discard(tag)  # membership test misses, so removal is skipped

    # The label is still visible: final label values alone cannot reveal the
    # fault; only the identity ledger / operation intent can.
    assert [t.identity for t in sorted_set] == [7]
    failures = stage_failures(sorted_set, intended - {7})
    assert failures == {'identity'}


def test_corrupted_hash_re_add_duplicates_the_identity():
    tag = ImageTag(4, hotness=40)
    sorted_set = SortedSet([tag], key=hotness_key)

    tag.corrupt_hash = 9999
    sorted_set.add(tag)  # looks like a brand-new member under the new hash

    physical = list(sorted_set._set)
    assert len(physical) == 2
    assert all(member is tag for member in physical)
    assert physical_identities(sorted_set) == [4, 4]
    assert len(sorted_set._list) == 2
    failures = stage_failures(sorted_set, {4})
    assert failures == {'identity'}


def test_corrupted_hash_pop_splits_the_two_ledgers_writeback_fault():
    tag = ImageTag(8, hotness=60)
    sorted_set = SortedSet([tag], key=hotness_key)

    tag.corrupt_hash = 9999
    with pytest.raises(KeyError):
        sorted_set.pop(0)  # rank ledger removed first, identity write-back fails

    # The two ledgers now disagree about whether the member exists.
    assert len(sorted_set._list) == 0
    assert len(sorted_set._set) == 1
    assert list(sorted_set._set)[0] is tag
    failures = stage_failures(sorted_set, set())
    assert 'writeback' in failures
    with pytest.raises(AssertionError):
        sorted_set._check()


def test_hotness_mutation_is_a_sort_key_fault_not_an_identity_fault():
    moved = ImageTag(9, hotness=10)
    members = [ImageTag(identity, hotness=identity * 10) for identity in range(1, 6)]
    sorted_set = SortedSet(members + [moved], key=hotness_key)

    moved.hotness = 999  # hash stays stable: identity is untouched

    # Identity ledger remains healthy...
    assert moved in sorted_set
    assert len(sorted_set._set) == 6
    # ...but the rank ledger cached the stale sort key and stale position.
    failures = stage_failures(sorted_set, {1, 2, 3, 4, 5, 9})
    assert failures == {'sort_key'}
    cached = flatten_rank_keys(sorted_set)
    fresh = sorted(hotness_key(tag) for tag in sorted_set._set)
    assert cached != fresh
    assert fresh == [10, 20, 30, 40, 50, 999]
    # The visible order contradicts the order rebuilt from current hotness.
    assert [tag.identity for tag in sorted_set] != fresh_rank_order(sorted_set)


def test_equal_final_labels_still_diverge_at_the_identity_ledger():
    # Two runs end with the same visible label sequence; only the identity
    # ledger distinguishes the corrupted run from the healthy one.
    healthy_tag = ImageTag(7, hotness=50)
    healthy = SortedSet([healthy_tag], key=hotness_key)
    healthy.discard(ImageTag(999, 1))  # no-op, absent member

    corrupt_tag = ImageTag(7, hotness=50)
    corrupt = SortedSet([corrupt_tag], key=hotness_key)
    corrupt_tag.corrupt_hash = 9999
    corrupt.discard(corrupt_tag)  # intended removal, silently swallowed

    assert [t.identity for t in healthy] == [t.identity for t in corrupt] == [7]
    assert stage_failures(healthy, {7}) == set()
    assert stage_failures(corrupt, set()) == {'identity'}


# ---------------------------------------------------------------------------
# Interleaved operation logs: locate the first diverging stage
# ---------------------------------------------------------------------------


def hash_contract_breaks(sorted_set):
    """A stored object can no longer find its own hash bucket (latent fault).

    This signal appears the moment the external hash mutation happens, even
    before any SortedSet operation moves a ledger entry.
    """
    return any(tag not in sorted_set._set for tag in sorted_set._set)


def stage_failures(sorted_set, intended_ids):
    """Return the set of ledger stages that contradict the operation log.

    Stages:

    * ``'writeback'`` -- identity and rank ledgers disagree about members;
    * ``'sort_key'`` -- cached rank keys drift from freshly extracted keys;
    * ``'identity'``  -- duplicated identities or a membership set that
      contradicts the intended identity set.

    A merely latent hash mutation is reported separately by
    :func:`hash_contract_breaks`, not counted as an operational divergence.
    """
    failures = set()
    physical = list(sorted_set._set)
    phys_ids = [tag.identity for tag in physical]
    list_ids = [tag.identity for tag in sorted_set._list]

    if len(sorted_set._list) != len(sorted_set._set) or sorted(list_ids) != sorted(
        phys_ids
    ):
        failures.add('writeback')

    cached_keys = flatten_rank_keys(sorted_set)
    fresh_keys = sorted(hotness_key(tag) for tag in physical)
    if cached_keys != fresh_keys:
        failures.add('sort_key')

    if len(phys_ids) != len(set(phys_ids)) or set(phys_ids) != set(intended_ids):
        failures.add('identity')

    return failures


ROOT_PRECEDENCE = ('writeback', 'sort_key', 'identity')


def root_cause(sorted_set, intended_ids):
    failures = stage_failures(sorted_set, intended_ids)
    for stage in ROOT_PRECEDENCE:
        if stage in failures:
            return stage, failures
    return None, failures


class Replay:
    """Operation log with an independent identity oracle, step by step."""

    def __init__(self):
        self.sorted_set = SortedSet(key=hotness_key)
        self.oracle = set()
        self.tags = {
            identity: ImageTag(identity, hotness=identity * 10)
            for identity in range(1, 9)
        }

    def step(self, label, action, intended_ids):
        action()
        root, failures = root_cause(self.sorted_set, intended_ids)
        self.oracle = set(intended_ids)
        return root, failures

    def add(self, identity):
        return self.step(
            f'add {identity}',
            lambda: self.sorted_set.add(self.tags[identity]),
            self.oracle | {identity},
        )

    def discard(self, identity):
        return self.step(
            f'discard {identity}',
            lambda: self.sorted_set.discard(self.tags[identity]),
            self.oracle - {identity},
        )


def test_healthy_interleaved_log_never_diverges():
    replay = Replay()
    healthy_log = (
        replay.add(3),
        replay.add(1),
        replay.add(4),
        replay.add(1),
        replay.add(5),
        replay.discard(3),
        replay.add(2),
        replay.discard(1),
    )
    for step_number, (root, failures) in enumerate(healthy_log):
        assert root is None, f'divergence at step {step_number}: {failures}'
        assert_dual_index(replay.sorted_set, replay.oracle)


def test_first_divergence_identity_merge_on_re_add_after_hash_mutation():
    replay = Replay()
    assert replay.add(3)[0] is None
    assert replay.add(1)[0] is None
    assert replay.add(4)[0] is None

    # External event: the hash is mutated while the object is stored. The
    # ledgers themselves do not move, so no operational stage diverges yet;
    # only the latent hash-contract probe can see it.
    replay.tags[1].corrupt_hash = 9999
    assert hash_contract_breaks(replay.sorted_set)
    assert root_cause(replay.sorted_set, replay.oracle)[0] is None

    # Next identity merge treats the same object as a new member.
    root, failures = replay.add(1)
    assert root == 'identity'
    assert failures == {'identity'}
    assert physical_identities(replay.sorted_set).count(1) == 2


def test_first_divergence_sort_key_extraction_on_hotness_mutation():
    replay = Replay()
    assert replay.add(2)[0] is None
    assert replay.add(5)[0] is None
    assert replay.add(3)[0] is None

    # External event: hotness changes, identity hash stays stable. The rank
    # ledger's cached sort key diverges at this exact step.
    replay.tags[5].hotness = 5
    root, failures = root_cause(replay.sorted_set, replay.oracle)
    assert root == 'sort_key'
    assert failures == {'sort_key'}

    # A subsequent structural operation does not paper over the stale key.
    root, failures = replay.add(7)
    assert root == 'sort_key'
    assert failures == {'sort_key'}


def test_first_divergence_writeback_on_pop_after_hash_mutation():
    replay = Replay()
    assert replay.add(2)[0] is None
    assert replay.add(5)[0] is None
    assert replay.add(3)[0] is None

    replay.tags[5].corrupt_hash = 9999  # latent until a hash-based op
    assert hash_contract_breaks(replay.sorted_set)
    assert root_cause(replay.sorted_set, replay.oracle)[0] is None

    position = replay.sorted_set.index(replay.tags[5])
    intended_after = replay.oracle - {5}
    with pytest.raises(KeyError):
        replay.sorted_set.pop(position)

    root, failures = root_cause(replay.sorted_set, intended_after)
    assert root == 'writeback'
    assert 'writeback' in failures
    # Rank ledger dropped the label, identity ledger still holds it.
    assert replay.tags[5] not in replay.sorted_set._list
    assert any(member is replay.tags[5] for member in replay.sorted_set._set)


def test_three_fault_classes_are_distinguished_on_the_same_log():
    def run(fault):
        replay = Replay()
        replay.add(4)
        replay.add(2)
        replay.add(6)
        if fault == 'identity':
            replay.tags[2].corrupt_hash = 9999
            return replay.add(2)
        if fault == 'sort_key':
            replay.tags[2].hotness = 1
            return root_cause(replay.sorted_set, replay.oracle)
        replay.tags[2].corrupt_hash = 9999
        position = replay.sorted_set.index(replay.tags[2])
        with pytest.raises(KeyError):
            replay.sorted_set.pop(position)
        return root_cause(replay.sorted_set, replay.oracle - {2})

    identity_root, identity_failures = run('identity')
    sort_root, sort_failures = run('sort_key')
    write_root, write_failures = run('writeback')

    assert identity_root == 'identity' and identity_failures == {'identity'}
    assert sort_root == 'sort_key' and sort_failures == {'sort_key'}
    assert write_root == 'writeback' and 'writeback' in write_failures
