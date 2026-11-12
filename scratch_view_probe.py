"""Probe SortedDict view/iterator semantics under interleaved mutations.

Run: .venv/bin/python scratch_view_probe.py
"""
from sortedcontainers import SortedDict


def banner(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def structural(d):
    lst = d._list
    return {
        'len': lst._len,
        'block_ids': tuple(id(b) for b in lst._lists),
        'blocks': [list(b) for b in lst._lists],
        'maxes': list(lst._maxes),
        'index_built': bool(lst._index),
    }


# ---------------------------------------------------------------- 1. 改值
banner('1. value reassignment on existing key (sd["b"] = 20)')
sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
sd._list._reset(4)

live_k, live_v, live_it_view = sd.keys(), sd.values(), sd.items()
snap_k, snap_v, snap_it = list(sd.keys()), list(sd.values()), list(sd.items())
adv = iter(sd.items())
print('advanced iterator first yield:', next(adv))

before = structural(sd)
sd['b'] = 20                       # __setitem__: key already in dict -> dict-only write
after = structural(sd)

print('structure unchanged      :', before == after)
print('live values view         :', list(live_v))
print('live items view          :', list(live_it_view))
print('snapshot values (frozen) :', snap_v)
print('iterator resumes, sees   :', list(adv), '(value of b looked up at yield time)')


# ------------------------------------------------- 2. 删后复用同键
banner('2. delete then re-insert SAME keys (del 0,1; then sd[0],sd[1] back)')
sd = SortedDict({i: i for i in range(8)})
sd._list._reset(4)                       # blocks [0,1,2,3] [4,5,6,7]
live_v = sd.values()
snap_k = list(sd.keys())
s0 = structural(sd)
print('blocks before:', s0['blocks'])

del sd[1]
del sd[0]                                # block0 -> [2,3], len <= load/2 -> merge
print('live view len DIPS during churn :', len(live_v), '(snapshot still', len(snap_k), ')')
print('blocks after deletes (merged)   :', structural(sd)['blocks'])

sd[0] = 100                              # same keys, brand-new source rows
sd[1] = 101
s1 = structural(sd)
print('blocks after reinsert           :', s1['blocks'])

print('final keys / order identical    :', list(sd.keys()) == snap_k)
print('final length / key sets equal   :', len(sd) == len(snap_k), set(sd) == set(snap_k))
print('block object identities changed :', s0['block_ids'] != s1['block_ids'])
print('maxes changed                   :', s0['maxes'] != s1['maxes'], s0['maxes'], '->', s1['maxes'])
print('_index cleared by structural op :', sd._list._index == [])


# ------------------------------------------------- 3. 跨内部块插入 + 已前进迭代器
banner('3. cross-block insert triggers _expand split; iterator already advancing')
sd = SortedDict({i: i * 10 for i in range(16)})  # 0..15
sd._list._reset(4)                               # four blocks of 4
for k in range(-4, 0):                           # grow block 0 to length 8 (= 2*load)
    sd[k] = k * 10
print('blocks before:', structural(sd)['blocks'])

it = iter(sd)
consumed = [next(it) for _ in range(8)]          # fully past block 0, now inside block 1
print('iterator consumed from block 0:', consumed)

sd[-5] = -50                                     # block 0 -> length 9 > 2*load -> split
print('blocks after :', structural(sd)['blocks'])

rest = list(it)
all_seen = consumed + rest
dupes = sorted({k for k in all_seen if all_seen.count(k) > 1})
missing = sorted(set(sd) - set(all_seen))
print('iterator rest yields:', rest)
print('duplicated keys       :', dupes)
print('missing keys          :', missing)
print('no RuntimeError raised: iteration completed silently')


# ---------------------------------------- 3b. 块合并/重建导致重复或陈旧
banner('3b. block merge (iterator in merged block) skips; clear() goes stale')

# (i) iterator inside the block that gets merged away: stale yield + skipped block
sd = SortedDict({i: i for i in range(16)})
sd._list._reset(4)                          # [0..3] [4..7] [8..11] [12..15]
it = iter(sd)
consumed = [next(it) for _ in range(5)]     # 0..4, iterator now inside block 1
for k in (7, 6, 5):
    del sd[k]                               # block1 -> [4], len <= load/2 -> merge away
print('blocks after merge:', structural(sd)['blocks'])
rest = list(it)
seen = consumed + rest
print('(i) rest (stale 5, then skips [8..11]):', rest)
print('(i) duplicated keys :', sorted({k for k in seen if seen.count(k) > 1}))
print('(i) missing keys    :', sorted(set(sd) - set(seen)))
print('(i) yet live view is correct:', list(sd))

# (ii) clear() rebuilds the outer list in place; chain still references OLD blocks
sd = SortedDict({i: i for i in range(8)})
sd._list._reset(4)
it = iter(sd)
consumed = [next(it), next(it)]
sd.clear()
sd.update({'p': 1, 'q': 2})
rest = list(it)
print('(ii) live dict keys :', list(sd))
print('(ii) iterator keeps yielding deleted old keys:', consumed + rest,
      '-> stale, no RuntimeError')


# ------------------------------------------------- 4. 两个收敛历史
banner('4. two histories, identical final columns, different row provenance')


class Row:
    """Source row: reports compare by metric value, provenance kept separate."""

    def __init__(self, source_id, metric):
        self.source_id = source_id          # never shown in the report column
        self.metric = metric

    def __eq__(self, other):
        return isinstance(other, Row) and self.metric == other.metric

    def __hash__(self):
        return hash(self.metric)

    def __repr__(self):
        return f'Row(src={self.source_id}, m={self.metric})'


# History A: same two source rows, their metric values swapped in place (type-1 change)
hA = SortedDict({'a': Row('src-1', 1), 'b': Row('src-2', 2)})
dims_A = list(hA.keys())
hA['a'].metric, hA['b'].metric = 2, 1
metrics_A = [row.metric for row in hA.values()]
prov_A = {k: v.source_id for k, v in hA.items()}

# History B: original rows deleted; same keys rebound to two NEW source rows (type-2 change)
hB = SortedDict({'a': Row('src-1', 1), 'b': Row('src-2', 2)})
dims_B = list(hB.keys())
del hB['a']
del hB['b']
hB['b'] = Row('src-3', 1)
hB['a'] = Row('src-4', 2)
metrics_B = [row.metric for row in hB.values()]
prov_B = {k: v.source_id for k, v in hB.items()}

print('history A dims/metrics :', dims_A, metrics_A, prov_A)
print('history B dims/metrics :', dims_B, metrics_B, prov_B)
print('columns identical      :', dims_A == dims_B and metrics_A == metrics_B)
print('provenance differs     :', prov_A != prov_B,
      '-> value columns cannot tell A and B apart')


# ------------------------------------------------- 5. 最小复用指纹
banner('5. minimal reuse observation: block fingerprint + per-row version, no recompute')


def block_fingerprint(d):
    lst = d._list
    return (len(d), tuple(id(b) for b in lst._lists), tuple(lst._maxes))


# versions live in the source system; observing them is O(1), no metric recomputation
versions = {'a': 1, 'b': 1, 'c': 1}
sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
sd._list._reset(4)
f0 = block_fingerprint(sd)
v0 = dict(versions)

# type-1 change: metric of b changes, no structural movement
versions['b'] = 2
sd['b'] = 20
print('value change | block fingerprint stable :', block_fingerprint(sd) == f0)
print('             | dim column reusable      : True; stale metric rows:',
      [k for k in sd if versions[k] != v0[k]])

# structural change: insert x -> block ids/maxes change; bisect localizes the slice
sd['x'] = 99
lo, hi = sd.bisect_left('x'), sd.bisect_right('x')
print('insert       | block fingerprint stable :', block_fingerprint(sd) == f0)
print('             | only positional slice', (lo, hi), 'invalid; prefix 0..%d untouched' % lo)
print('per-row version scan is O(n) comparisons but recomputes metrics for',
      [k for k in ('a', 'b', 'c') if versions.get(k, 0) != v0.get(k, 0)], 'only')
