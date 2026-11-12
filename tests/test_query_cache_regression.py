"""
Differential regression tests for the Table query cache around dimension
matching rules built with ``Query.one_of`` and composable predicates.

Every scenario is run against two branches holding identical rows:

* the default table, which caches search results keyed by the query hash;
* a mirror table opened with ``cache_size=0`` (the *cache-free branch*), which
  re-evaluates the predicate closure on every search and never stores a slot.

Hits are compared through an independent business row-number list
(``line_no``), not TinyDB's internal ``doc_id``.

Two failure modes can each make a cached search diverge from the cache-free
branch, and the tests below isolate them so they are separately identifiable
even when they happen to return the same rows:

* **cache-key collision** -- two retained rule *versions* built from separate,
  value-equal lists hash to the same frozen ``one_of`` key. Revising the second
  version changes its closure semantics but not its hash, so its first search
  is served from the first version's cache slot.
* **parameter rewritten after build** -- one rule's closure captures the
  *mutable* list passed to ``one_of``. Mutating that list after the rule was
  built changes the predicate while the frozen hash (and the cached result)
  stays put.

Freezing contract:

* inputs ``freeze`` can normalise (list/tuple/set/dict and nested containers of
  hashable scalars) are snapshotted into the hash at rule-build time, so the
  cache key is immune to later mutation of the original object;
* closures carrying a mutable callable (``Query.map``), an unhashable
  parameter, or explicitly flagged via ``is_cacheable = False`` are treated as
  uncacheable and must never occupy a cache slot.

The two defect tests are *characterization* tests: they pin the current cache
behaviour. The cache-free branch always carries the correct result, so the
asserted ``cached != live`` divergence is the regression lock -- any future
change to the caching semantics makes these tests fail deliberately.
"""

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

Q = Query()

# ``line_no`` is an independent business identifier, deliberately distinct
# from TinyDB's internal document ID.
ROWS = [
    {'line_no': 1, 'region': 'NA'},
    {'line_no': 2, 'region': 'EU'},
    {'line_no': 3, 'region': 'APAC'},
]


def _build_db():
    """Return a cached default table and an independent cache-free mirror.

    The ``live`` table uses ``cache_size=0``: every search re-runs the
    predicate, so it is the no-cache reference branch for the same rules and
    the same documents.
    """
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple(dict(row) for row in ROWS)

    live = db.table('live', cache_size=0)
    live.insert_multiple(dict(row) for row in ROWS)
    return db, live


def _lines(docs):
    return [doc['line_no'] for doc in docs]


def test_one_of_repeated_query_agrees_with_cache_free_branch():
    db, live = _build_db()
    rule = Q.region.one_of(['NA', 'EU'])

    assert rule.is_cacheable()

    # First search populates the cache slot ...
    assert _lines(db.search(rule)) == _lines(live.search(rule)) == [1, 2]
    assert rule in db._query_cache

    # ... and the repeated identical query is served from that slot while the
    # cache-free branch keeps recomputing without storing anything.
    assert _lines(db.search(rule)) == [1, 2]
    assert len(db._query_cache) == 1
    assert len(live._query_cache) == 0


def test_history_backfill_invalidates_cache_on_both_branches():
    db, live = _build_db()
    rule = Q.region.one_of(['NA'])

    assert _lines(db.search(rule)) == _lines(live.search(rule)) == [1]
    assert rule in db._query_cache

    # Backfill a historical document into both branches.
    backfill = {'line_no': 4, 'region': 'NA'}
    db.insert(backfill)
    live.insert(dict(backfill))

    # Any write discards the whole query cache ...
    assert len(db._query_cache) == 0
    # ... so the backfilled row becomes visible on both branches.
    assert _lines(db.search(rule)) == _lines(live.search(rule)) == [1, 4]
    assert _lines(db.search(rule)) == [1, 4]


def test_rule_revision_cache_key_collision_keeps_two_versions():
    """Root cause A: two rule versions alias one frozen cache key.

    Discriminating signature (valid even if the stale rows coincidentally equal
    the rewrite case): there are *two* retained rule objects built from separate
    lists, and ``rule_v2`` is already ``in`` the cache before it is ever searched
    -- it aliases v1's slot. The unrevised v1 stays correct on both branches.
    """
    db, live = _build_db()

    # Two independent source lists with identical *content*.
    v1_source = ['NA', 'EU']
    v2_source = ['NA', 'EU']
    rule_v1 = Q.region.one_of(v1_source)
    rule_v2 = Q.region.one_of(v2_source)

    # freeze() snapshots content, so distinct mutable lists collapse onto one
    # hash / equality key.
    assert hash(rule_v1) == hash(rule_v2)
    assert rule_v1 == rule_v2

    # Revise ONLY v2; v1's source list must remain untouched.
    v2_source.append('APAC')
    assert v1_source == ['NA', 'EU']

    # The two retained versions now mean different things ...
    assert rule_v1({'region': 'APAC'}) is False
    assert rule_v2({'region': 'APAC'}) is True
    # ... yet still share a hash, i.e. the same cache slot.
    assert hash(rule_v1) == hash(rule_v2)

    # v1 runs first and fills the shared slot with v1's rows.
    assert _lines(db.search(rule_v1)) == [1, 2]
    assert rule_v2 in db._query_cache  # v2 aliases v1's slot before its own run

    # DEFECT (cache-key collision): v2's first search is served stale from
    # v1's slot, while the cache-free branch applies the revised v2 closure.
    cached_v2 = _lines(db.search(rule_v2))
    live_v2 = _lines(live.search(rule_v2))
    assert live_v2 == [1, 2, 3]
    assert cached_v2 == [1, 2]
    assert cached_v2 != live_v2

    # v1 was never revised, so it stays correct on both branches.
    assert _lines(db.search(rule_v1)) == _lines(live.search(rule_v1)) == [1, 2]


def test_mutable_parameter_rewritten_after_build_serves_stale_slot():
    """Root cause B: one rule's captured list is mutated later.

    Discriminating signature: a *single* rule object whose source list changes
    after build. The hash never moves, so the same slot stays stale until
    ``clear_cache()`` forces a fresh evaluation, after which the cached branch
    flips to the mutated result -- proving the closure read the rewritten list.
    """
    db, live = _build_db()
    allowed = ['NA', 'EU']
    rule = Q.region.one_of(allowed)

    assert _lines(db.search(rule)) == _lines(live.search(rule)) == [1, 2]
    assert rule in db._query_cache

    # Rewrite the captured parameter AFTER the rule was built.
    allowed.append('APAC')

    # The frozen hash is unchanged, so the key -- and its stale result -- is
    # reused ...
    assert _lines(db.search(rule)) == [1, 2]
    assert rule in db._query_cache
    # ... whereas the cache-free branch re-runs the closure over the mutated
    # list and now matches APAC as well.
    assert _lines(live.search(rule)) == [1, 2, 3]

    # Forcing a fresh evaluation converges the cached branch onto the mutated
    # closure too: the divergence was the stale slot, not the predicate.
    db.clear_cache()
    assert _lines(db.search(rule)) == [1, 2, 3]


def test_shared_mutable_closure_propagates_through_composed_predicate():
    db, live = _build_db()
    shared = ['NA']
    base = Q.region.one_of(shared)
    # The AND closure re-invokes the same base rule object.
    composed = base & (Q.line_no >= 1)
    assert composed.is_cacheable()

    assert _lines(db.search(composed)) == _lines(live.search(composed)) == [1]

    # Mutating the list captured by ``base`` changes ``composed`` as well.
    shared.append('EU')

    assert _lines(db.search(composed)) == [1]        # stale cached slot
    assert _lines(live.search(composed)) == [1, 2]   # re-evaluated closure


def test_composed_predicates_get_distinct_cache_keys():
    db, live = _build_db()
    r_na = Q.region.one_of(['NA'])
    r_eu = Q.region.one_of(['EU'])
    and_rule = r_na & (Q.line_no >= 1)
    or_rule = r_na | r_eu

    assert and_rule.is_cacheable()
    assert or_rule.is_cacheable()
    assert hash(and_rule) != hash(or_rule)

    assert _lines(db.search(and_rule)) == _lines(live.search(and_rule)) == [1]
    assert _lines(db.search(or_rule)) == _lines(live.search(or_rule)) == [1, 2]
    assert len(db._query_cache) == 2  # distinct keys, no aliasing


def test_freeze_snapshots_mutable_containers_into_the_hash():
    source = ['NA', 'EU']
    rule = Q.region.one_of(source)
    key_before = hash(rule)

    # Mutating the original after build time must not move the cache key.
    source.append('APAC')
    assert hash(rule) == key_before

    # Nested mutable containers are frozen recursively and stay hashable.
    nested = Q.region.test(
        lambda value, allowed: value in allowed,
        [['NA', 'EU'], {'group': ['APAC']}],
    )
    assert nested.is_cacheable()
    assert hash(nested)

    # tuple / set / dict inputs all normalise into stable, cacheable keys.
    assert Q.region.one_of(('NA', 'EU')).is_cacheable()
    assert Q.region.one_of({'NA', 'EU'}).is_cacheable()
    fragment = Q.fragment({'region': 'NA'})
    assert fragment.is_cacheable()
    db_fragment, _ = _build_db()
    assert _lines(db_fragment.search(fragment)) == [1]


def test_uncacheable_inputs_never_enter_the_query_cache():
    db, live = _build_db()

    # 1. Query.map places a (potentially mutable) callable in the path.
    mapped = Q.region.map(str).one_of(['NA', 'EU'])
    assert not mapped.is_cacheable()
    assert _lines(db.search(mapped)) == _lines(live.search(mapped)) == [1, 2]
    assert len(db._query_cache) == 0

    # 2. A parameter that stays unhashable after freeze() degrades to
    #    uncacheable instead of raising.
    class Unhashable:
        __hash__ = None

    doomed = Q.region.test(lambda value, _x: True, Unhashable())
    assert not doomed.is_cacheable()
    db.search(doomed)
    assert len(db._query_cache) == 0

    # 3. A plain callable explicitly opting out via is_cacheable=False.
    def plain(doc):
        return doc['region'] in ('NA', 'EU')

    plain.is_cacheable = lambda: False
    assert _lines(db.search(plain)) == [1, 2]
    assert len(db._query_cache) == 0


def test_cache_free_branch_recomputes_on_every_search():
    _db, live = _build_db()
    evaluations = 0

    def counting(doc):
        nonlocal evaluations
        evaluations += 1
        return doc['region'] == 'NA'

    counting.is_cacheable = lambda: False  # neither branch may cache it

    live.search(counting)
    live.search(counting)
    assert evaluations == 2 * len(ROWS)  # every row re-evaluated, every time


def test_interleaved_revision_backfill_repeat_and_shared_closure():
    """One timeline interleaving every operation with stepwise line checks."""
    db, live = _build_db()

    # Step 1: build rule v1 and run it on both branches.
    allow_v1 = ['NA', 'EU']
    rule_v1 = Q.region.one_of(allow_v1)
    assert _lines(db.search(rule_v1)) == _lines(live.search(rule_v1)) == [1, 2]

    # Step 2: repeated query -- cached and cache-free branches still agree.
    assert _lines(db.search(rule_v1)) == _lines(live.search(rule_v1)) == [1, 2]
    assert len(db._query_cache) == 1
    assert len(live._query_cache) == 0

    # Step 3: backfill a historical row (line 4, EU) into both branches.
    db.insert({'line_no': 4, 'region': 'EU'})
    live.insert({'line_no': 4, 'region': 'EU'})
    assert len(db._query_cache) == 0  # write invalidated the cache
    assert _lines(db.search(rule_v1)) == _lines(live.search(rule_v1)) == [1, 2, 4]

    # Step 4: revise into v2 (separate list, same content), keep both versions.
    allow_v2 = ['NA', 'EU']
    rule_v2 = Q.region.one_of(allow_v2)
    assert hash(rule_v1) == hash(rule_v2)  # frozen content -> shared key
    allow_v2.append('APAC')                # revise v2 only
    assert allow_v1 == ['NA', 'EU']
    # v2 aliases v1's slot (collision) ...
    assert _lines(db.search(rule_v2)) == [1, 2, 4]
    # ... while the cache-free branch applies the revised v2 closure.
    assert _lines(live.search(rule_v2)) == [1, 2, 3, 4]

    # Step 5: shared closure feeding a composed predicate, then mutated.
    shared = ['NA']
    base = Q.region.one_of(shared)
    composed = base & (Q.line_no >= 1)
    assert _lines(db.search(composed)) == _lines(live.search(composed)) == [1]
    shared.append('EU')
    assert _lines(db.search(composed)) == [1]            # stale slot
    assert _lines(live.search(composed)) == [1, 2, 4]    # NA/EU after backfill
