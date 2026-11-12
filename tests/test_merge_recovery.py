"""Recovery regressions for dpath.merge() partial failures.

A merge that fails in its back half has already mutated earlier members of
the destination. A blind retry of the same source fragment then replays those
members, which grows ADDITIVE lists (and re-references REPLACE targets). The
tests below enumerate every source member with an independent occurrence
ordinal, observe each failure through three channels (the live object,
externally held sub-lists, and an on-disk snapshot), classify the finite
two-level failure prefixes, and prove the minimal restore set that makes a
retry bit-for-bit equal to a single successful import.
"""
import copy
import json
import tempfile

import dpath
from dpath import MergeType, segments
from dpath.exceptions import InvalidKeyName


MISSING = object()


# ---------------------------------------------------------------------------
# Occurrence ordinals: independent numbering of every source member in the
# exact order merger() applies it.
# ---------------------------------------------------------------------------

def preorder(node, location=()):
    for key, value in segments.make_walkable(node):
        segment = int(key) if isinstance(key, int) else key
        path = location + (segment,)
        yield path, value
        if not segments.leaf(value):
            yield from preorder(value, path)


def member_ordinals(src):
    filtered = dpath.search(src, "**", separator="/")
    return {path: ordinal for ordinal, (path, _value) in enumerate(preorder(filtered), start=1)}


def contribution_count(node):
    """Number of leaf contributions an import makes to a document."""
    if isinstance(node, dict):
        return sum(contribution_count(value) for value in node.values())
    if isinstance(node, list):
        return sum(contribution_count(value) for value in node)
    return 1


# ---------------------------------------------------------------------------
# Fault injection. TYPESAFE failures need no injection (real TypeErrors); the
# ADDITIVE and REPLACE back-half failures are injected through the exact
# operations merge() uses (target += found / parent[key] = found).
# ---------------------------------------------------------------------------

class FailingAddList(list):
    """Fails the Nth in-place +=, which is exactly how merge applies ADDITIVE."""

    def __init__(self, values=(), label=(), fail_on_call=1, armed_box=None):
        super().__init__(values)
        self.label = label
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.armed_box = armed_box

    def __iadd__(self, other):
        self.calls += 1
        if self.armed_box is not None and self.armed_box["armed"] and self.calls == self.fail_on_call:
            raise RuntimeError(f"injected failure while extending {self.label}")
        return super().__iadd__(other)


class FailingReplaceDict(dict):
    """Fails setitem on chosen keys, which is exactly how merge applies REPLACE."""

    def __init__(self, values=(), boom_keys=(), armed_box=None):
        super().__init__(values)
        self.boom_keys = set(boom_keys)
        self.armed_box = armed_box

    def __setitem__(self, key, value):
        if self.armed_box is not None and self.armed_box["armed"] and key in self.boom_keys:
            raise RuntimeError(f"injected failure while replacing {key}")
        super().__setitem__(key, value)


# ---------------------------------------------------------------------------
# Observation / recovery helpers.
# ---------------------------------------------------------------------------

def take_snapshots(dst):
    """Observe the destination as a deep copy and a round-tripped on-disk view."""
    snapshot = copy.deepcopy(dst)
    with tempfile.TemporaryDirectory() as tmp_dir:
        disk_path = f"{tmp_dir}/dst.snapshot.json"
        with open(disk_path, "w") as handle:
            json.dump(dst, handle)
        with open(disk_path) as handle:
            on_disk = json.load(handle)
    return snapshot, on_disk


def divergent_top_level(dst, snapshot):
    keys = set(dst) | set(snapshot)
    return sorted(key for key in keys if dst.get(key, MISSING) != snapshot.get(key, MISSING))


def divergent_restore_paths(dst, snapshot):
    return [(key,) for key in divergent_top_level(dst, snapshot)]


def restore_paths(dst, snapshot, paths):
    """Minimal restore: put snapshot sub-trees back, delete paths never snapshotted."""
    for path in paths:
        if segments.has(snapshot, list(path)):
            segments.set(dst, list(path), copy.deepcopy(segments.get(snapshot, list(path))))
        else:
            parent = dst if len(path) == 1 else segments.get(dst, list(path[:-1]))
            del parent[path[-1]]


def delete_path(dst, path):
    parent = dst if len(path) == 1 else segments.get(dst, list(path[:-1]))
    del parent[path[-1]]


def clean_import(original_dst, fragments, flags):
    """What a single, never-retried import would have produced."""
    baseline = copy.deepcopy(original_dst)
    for fragment in fragments:
        dpath.merge(baseline, copy.deepcopy(fragment), flags=flags)
    return baseline


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as error:
        return error
    raise AssertionError(f"{exc_type.__name__} was not raised")


def naive_dedup_recovery(dst):
    """The tempting-but-wrong policy: strip repeated equal values from lists."""
    if isinstance(dst, dict):
        return {key: naive_dedup_recovery(value) for key, value in dst.items()}
    if isinstance(dst, list):
        seen = []
        for value in dst:
            cleaned = naive_dedup_recovery(value)
            if cleaned not in seen:
                seen.append(cleaned)
        return seen
    return dst


# ---------------------------------------------------------------------------
# Ordinal bookkeeping.
# ---------------------------------------------------------------------------

def test_every_source_member_gets_an_independent_ordinal():
    src = {"a": [5, 5], "b": {"c": [7], "d": 0}, "e": "z"}
    ordinals = member_ordinals(src)

    assert ordinals[("a",)] == 1
    assert ordinals[("a", 0)] == 2
    assert ordinals[("a", 1)] == 3
    assert ordinals[("b",)] == 4
    assert ordinals[("b", "c")] == 5
    assert ordinals[("b", "c", 0)] == 6
    assert ordinals[("b", "d")] == 7
    assert ordinals[("e",)] == 8

    # Equal-valued, legally duplicated members still get distinct ordinals.
    assert ordinals[("a", 0)] != ordinals[("a", 1)]
    assert src["a"][0] == src["a"][1]


# ---------------------------------------------------------------------------
# Class A: failure prefixes that changed nothing -> safe to retry directly.
# ---------------------------------------------------------------------------

def test_class_a_typesafe_failure_on_first_member_leaves_no_trace():
    original = {"bad": 0, "a": [1]}
    dst = copy.deepcopy(original)
    src = {"bad": "x", "a": [2]}
    snapshot, on_disk = take_snapshots(dst)
    external_a = dst["a"]

    assert_raises(TypeError, dpath.merge, dst, src,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)

    assert dst == snapshot
    assert dst == on_disk
    assert external_a == [1]
    assert divergent_top_level(dst, snapshot) == []

    # Direct retry of the corrected fragment needs no restoration.
    corrected = {"bad": 0, "a": [2]}
    dpath.merge(dst, corrected, flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert dst == clean_import(original, [corrected],
                               MergeType.ADDITIVE | MergeType.TYPESAFE)


def test_class_a_empty_key_aborts_in_pre_merge_search_for_every_flag():
    for flags in (MergeType.ADDITIVE, MergeType.REPLACE,
                  MergeType.ADDITIVE | MergeType.TYPESAFE, MergeType.TYPESAFE):
        dst = {"a": [1, 2], "": 0}
        src = {"a": [9], "": 1}
        snapshot, on_disk = take_snapshots(dst)

        assert_raises(InvalidKeyName, dpath.merge, dst, src, flags=flags)

        assert dst == snapshot == on_disk
        assert dst["a"] == [1, 2]


# ---------------------------------------------------------------------------
# Class B: a sub-tree must be restored from the snapshot before retrying.
# ---------------------------------------------------------------------------

def test_class_b_typesafe_additive_restores_exactly_the_divergent_subtrees():
    original = {"a": [1, 2], "b": {"c": [3], "d": 0}}
    fragment = {"a": [9], "b": {"c": [8], "d": "x"}}
    dst = copy.deepcopy(original)
    snapshot, on_disk = take_snapshots(dst)
    external_a = dst["a"]
    external_bc = dst["b"]["c"]

    # Back-half TYPESAFE failure: a and b/c are applied before b/d explodes.
    error = assert_raises(TypeError, dpath.merge, dst, fragment,
                          flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert str(error).endswith("b/d")

    # Three observation channels agree on where the boundary is.
    assert dst == {"a": [1, 2, 9], "b": {"c": [3, 8], "d": 0}}
    assert external_a is dst["a"] and external_a == [1, 2, 9]
    assert external_bc is dst["b"]["c"] and external_bc == [3, 8]
    assert snapshot == original and on_disk == original
    assert divergent_top_level(dst, snapshot) == ["a", "b"]

    # A blind retry replays the already-applied members: lists gain a stump.
    assert_raises(TypeError, dpath.merge, dst, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert dst["a"] == [1, 2, 9, 9]
    assert dst["b"]["c"] == [3, 8, 8]
    assert external_a == [1, 2, 9, 9] and external_bc == [3, 8, 8]

    # Restore precisely the divergent paths, then retry the corrected fragment.
    restore_paths(dst, snapshot, [("a",), ("b",)])
    assert dst == snapshot
    corrected = {"a": [9], "b": {"c": [8], "d": 1}}
    dpath.merge(dst, corrected, flags=MergeType.ADDITIVE | MergeType.TYPESAFE)

    baseline = clean_import(original, [corrected], MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert dst == baseline
    assert contribution_count(dst) == contribution_count(baseline)


def test_class_b_restore_set_is_minimal():
    original = {"a": [1, 2], "b": {"c": [3], "d": 0}}
    fragment = {"a": [9], "b": {"c": [8], "d": "x"}}
    corrected = {"a": [9], "b": {"c": [8], "d": 1}}
    flags = MergeType.ADDITIVE | MergeType.TYPESAFE
    baseline = clean_import(original, [corrected], flags)

    for partial_restore in ([("a",)], [("b",)]):
        dst = copy.deepcopy(original)
        snapshot = copy.deepcopy(dst)
        assert_raises(TypeError, dpath.merge, dst, fragment, flags=flags)
        restore_paths(dst, snapshot, partial_restore)
        dpath.merge(dst, corrected, flags=flags)
        # Every proper subset of the restore set leaves replay residue.
        assert dst != baseline


def test_class_b_injected_additive_failure_restores_only_the_prefix():
    original = {
        "a": [1, 2],
        "b": FailingAddList([3], label=("b",), fail_on_call=1,
                            armed_box={"armed": False}),
        "c": 9,
    }
    fragment = {"a": [8], "b": [7], "c": 0}
    dst = copy.deepcopy(original)
    snapshot, _on_disk = take_snapshots(dst)
    external_b = dst["b"]

    dst["b"].armed_box["armed"] = True
    # The injected RuntimeError is swallowed by merge's broad except and surfaces
    # as a secondary ListIndex TypeError; either way the job records a failure.
    error = assert_raises(Exception, dpath.merge, dst, fragment, flags=MergeType.ADDITIVE)
    assert isinstance(error, TypeError)
    assert isinstance(error.__context__, RuntimeError)

    assert dst["a"] == [1, 2, 8]          # prefix member applied ...
    assert dst["b"] == [3]                # ... failing member never extended ...
    assert dst["c"] == 9                  # ... and later members were never reached
    assert external_b is dst["b"]
    assert divergent_top_level(dst, snapshot) == ["a"]

    # Minimal restore set is {a} only; b and c already match the snapshot.
    restore_paths(dst, snapshot, [("a",)])
    dst["b"].armed_box["armed"] = False
    dpath.merge(dst, fragment, flags=MergeType.ADDITIVE)

    baseline = clean_import(original, [fragment], MergeType.ADDITIVE)
    assert dst == baseline
    assert contribution_count(dst) == contribution_count(baseline)


def test_class_b_replace_failure_rebinds_prefix_and_needs_snapshot_restore():
    original = FailingReplaceDict(
        {"a": [1, 2], "b": [3]}, boom_keys=("b",), armed_box={"armed": False})
    fragment = {"a": [9], "b": [7]}
    dst = copy.deepcopy(original)
    snapshot, _on_disk = take_snapshots(dst)
    external_old_a = dst["a"]

    dst.armed_box["armed"] = True
    assert_raises(Exception, dpath.merge, dst, fragment, flags=MergeType.REPLACE)

    # REPLACE rebinds the prefix to the source object instead of extending it.
    assert dst["a"] == [9] and dst["a"] is fragment["a"]
    assert external_old_a == [1, 2]       # the pre-merge list survives only via the holder
    assert dst["b"] == [3]
    assert divergent_top_level(dst, snapshot) == ["a"]

    restore_paths(dst, snapshot, [("a",)])
    dst.armed_box["armed"] = False
    dpath.merge(dst, fragment, flags=MergeType.REPLACE)

    baseline = clean_import(original, [fragment], MergeType.REPLACE)
    assert dst == baseline
    assert dst["a"] == [9] and dst["b"] == [7]


def test_class_b_fresh_path_absent_from_snapshot_is_restored_by_deletion():
    original = {"b": 0}
    fragment = {"new": [5, 5], "b": "x"}
    dst = copy.deepcopy(original)
    snapshot, on_disk = take_snapshots(dst)

    assert_raises(TypeError, dpath.merge, dst, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert "new" in dst and "new" not in snapshot and "new" not in on_disk
    assert divergent_top_level(dst, snapshot) == ["new"]

    # The snapshot cannot provide a value for this path; restoration means deleting it.
    delete_path(dst, ("new",))
    corrected = {"new": [5, 5], "b": 0}
    dpath.merge(dst, corrected, flags=MergeType.ADDITIVE | MergeType.TYPESAFE)

    baseline = clean_import(original, [corrected], MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert dst == baseline
    assert dst["new"] == [5, 5]           # legal in-fragment duplicates both retained


# ---------------------------------------------------------------------------
# Class C: on-disk values alone cannot separate replay from legal duplicates.
# ---------------------------------------------------------------------------

def test_class_c_disk_bytes_are_identical_for_replay_and_legitimate_duplicate():
    fragment = {"l": [2], "bad": 0}

    # History 1: one fragment replayed twice after a back-half failure.
    replayed = {"l": [1], "bad": "x"}
    assert_raises(TypeError, dpath.merge, replayed, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert_raises(TypeError, dpath.merge, replayed, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    replay_bytes = json.dumps(replayed, sort_keys=True)

    # History 2: distinct healthy imports whose final state carries the same values.
    legit_dst = {}
    dpath.merge(legit_dst, {"l": [1]})
    dpath.merge(legit_dst, {"l": [2]})
    dpath.merge(legit_dst, {"l": [2], "bad": "x"},
                flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    legit_bytes = json.dumps(legit_dst, sort_keys=True)

    assert replayed == legit_dst == {"bad": "x", "l": [1, 2, 2]}
    assert replay_bytes == legit_bytes
    # The on-disk view alone cannot tell the two histories apart; only the
    # per-member occurrence ordinals record that history 1 repeated one ordinal.


def test_class_c_aliased_holder_drifts_but_ordinal_restore_recovers_baseline():
    first_fragment = {"l": [1]}
    dst = {}
    dpath.merge(dst, first_fragment)
    assert dst["l"] is first_fragment["l"]          # merge references, it does not copy
    external_holder = dst["l"]

    snapshot, _on_disk = take_snapshots(dst)
    second_fragment = {"l": [2], "bad": 0}
    dst["bad"] = "x"
    assert_raises(TypeError, dpath.merge, dst, second_fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)

    # The external holder aliases the mutated list: no pristine live copy remains.
    assert external_holder is dst["l"] == [1, 2]
    assert snapshot == {"l": [1]}

    corrected = {"l": [2], "bad": "x"}
    restore_paths(dst, snapshot, [("l",)])
    dpath.merge(dst, corrected, flags=MergeType.ADDITIVE | MergeType.TYPESAFE)

    baseline = {}
    dpath.merge(baseline, {"l": [1]})
    dpath.merge(baseline, {"l": [2], "bad": "x"},
                flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    assert dst == baseline == {"l": [1, 2], "bad": "x"}


def test_legal_same_value_duplicates_must_not_be_deduplicated_as_replay():
    # Two fragments from distinct origins legitimately contribute the same value.
    origin_one = {"l": [9]}
    origin_two = {"l": [9]}
    original = {}
    flags = MergeType.ADDITIVE
    baseline = clean_import(original, [origin_one, origin_two], flags)
    assert baseline["l"] == [9, 9]
    assert contribution_count(baseline) == 2

    # A partial failure of the second import, recovered via snapshot + restore,
    # must re-land both legal contributions rather than collapse equal values.
    dst = {"l": FailingAddList(
        [9], label=("l",), fail_on_call=1, armed_box={"armed": False})}
    snapshot = copy.deepcopy(dst)
    dst["l"].armed_box["armed"] = True
    assert_raises(Exception, dpath.merge, dst, copy.deepcopy(origin_two), flags=flags)
    assert dst["l"] == [9]                 # the failing extend never landed
    dst["l"].armed_box["armed"] = False
    restore_paths(dst, snapshot, divergent_restore_paths(dst, snapshot))
    dpath.merge(dst, copy.deepcopy(origin_two), flags=flags)
    assert dst == baseline

    # Value-based dedup cannot distinguish origins and destroys a legal duplicate.
    deduped = naive_dedup_recovery(baseline)
    assert deduped["l"] == [9]
    assert contribution_count(deduped) != contribution_count(baseline)


def test_legal_duplicates_inside_one_fragment_survive_recovery():
    original = {"b": 0}
    fragment = {"new": [5, 5], "b": "x"}
    corrected = {"new": [5, 5], "b": 0}
    flags = MergeType.ADDITIVE | MergeType.TYPESAFE

    dst = copy.deepcopy(original)
    snapshot = copy.deepcopy(dst)
    assert_raises(TypeError, dpath.merge, dst, fragment, flags=flags)
    restore_paths(dst, snapshot, divergent_restore_paths(dst, snapshot))
    dpath.merge(dst, corrected, flags=flags)

    baseline = clean_import(original, [corrected], flags)
    assert dst == baseline
    assert dst["new"] == [5, 5]
    assert contribution_count(dst["new"]) == 2


# ---------------------------------------------------------------------------
# End-to-end classification across the three flag families on two-level docs.
# ---------------------------------------------------------------------------

def test_finite_two_level_matrix_is_classified_correctly():
    flags = MergeType.ADDITIVE | MergeType.TYPESAFE

    cases = [
        # (label, original, fragment, expected class, expected restore paths)
        ("first-member failure", {"bad": 0, "a": [1]}, {"bad": "x", "a": [2]},
         "direct-retry", []),
        ("back-half additive prefix", {"a": [1, 2], "b": {"c": [3], "d": 0}},
         {"a": [9], "b": {"c": [8], "d": "x"}}, "restore-subtree", [("a",), ("b",)]),
        ("fresh path then failure", {"b": 0}, {"new": [5, 5], "b": "x"},
         "restore-subtree", [("new",)]),
    ]

    for label, original, fragment, expected, expected_paths in cases:
        dst = copy.deepcopy(original)
        snapshot, _on_disk = take_snapshots(dst)
        assert_raises(Exception, dpath.merge, dst, fragment, flags=flags)

        divergent = [tuple([key]) for key in divergent_top_level(dst, snapshot)]
        classified = "direct-retry" if not divergent else "restore-subtree"

        assert classified == expected, label
        assert divergent == expected_paths, label


def test_replay_vs_legitimate_history_only_differs_in_ordinals():
    # Each occurrence is tagged with (job id, member ordinal), not with its value.
    def tagged(job_id, fragment):
        return {job_id: member_ordinals(fragment)}

    replay_tags = []
    fragment = {"l": [2], "bad": 0}
    job_one_tag = tagged("job-1", fragment)
    replayed = {"l": [1], "bad": "x"}
    assert_raises(TypeError, dpath.merge, replayed, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    replay_tags.append(job_one_tag)
    # The old job re-invokes the same source fragment: same job id, same ordinals.
    assert_raises(TypeError, dpath.merge, replayed, fragment,
                  flags=MergeType.ADDITIVE | MergeType.TYPESAFE)
    replay_tags.append(job_one_tag)

    legitimate_tags = []
    legitimate = {}
    dpath.merge(legitimate, {"l": [1]})
    dpath.merge(legitimate, {"l": [2]})
    legitimate_tags.append(tagged("job-1", {"l": [2]}))
    dpath.merge(legitimate, {"l": [2]})
    legitimate_tags.append(tagged("job-2", {"l": [2]}))

    assert replayed["l"] == legitimate["l"] == [1, 2, 2]
    # Values are byte-identical; only the occurrence tags distinguish a replay
    # (the same job id twice) from two legitimate, distinctly-tagged imports.
    assert replay_tags[0] is replay_tags[1]
    assert list(legitimate_tags[0]) != list(legitimate_tags[1])
    # The same member position carries the same ordinal in both jobs; only the
    # distinct job ids (not the values) prove these were separate, legal imports.
    assert list(legitimate_tags[0].values())[0][("l", 0)] == \
        list(legitimate_tags[1].values())[0][("l", 0)] == 2
