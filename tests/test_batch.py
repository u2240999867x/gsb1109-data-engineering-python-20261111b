import pytest

import dpath
import dpath.exceptions
from dpath.batch import apply, plan, replay


def test_literal_and_glob_rules_mix_on_same_columns():
    # The metadata really carries keys literally named 'a*b' and '**' next
    # to ordinary columns. A glob rule must keep globbing while a literal
    # rule addresses the strangely named columns themselves.
    doc = {
        "metadata": {
            "a*b": 1,
            "**": 2,
            "acb": 3,
            "keep": 4,
        },
    }
    original = {
        "metadata": {
            "a*b": 1,
            "**": 2,
            "acb": 3,
            "keep": 4,
        },
    }

    manifest = [
        ({"glob": ["metadata", "a*b"], "value": 10, "literal": True}),
        ({"glob": ["metadata", "**"], "value": 20, "literal": True}),
        ("metadata/a*b", 10),
    ]

    result = apply(doc, manifest)

    # The literal rules hit exactly the two strangely named columns.
    literal_changes = [
        change for change in result.applied
        if any(rule in (0, 1) for rule in change.rules)
    ]
    assert sorted(change.path for change in literal_changes) == [
        "metadata/**",
        "metadata/a*b",
    ]

    # The glob rule 'metadata/a*b' reaches both 'a*b' itself and 'acb'; at
    # 'a*b' it agrees with the literal rule, so nothing is isolated.
    glob_changes = [change for change in result.applied if 2 in change.rules]
    assert sorted(change.target[-1] for change in glob_changes) == ["a*b", "acb"]

    assert doc["metadata"]["a*b"] == 10
    assert doc["metadata"]["**"] == 20
    assert doc["metadata"]["acb"] == 10
    assert doc["metadata"]["keep"] == original["metadata"]["keep"]
    assert result.conflicts == []


def test_all_rules_resolve_against_original_document():
    doc = {
        "rows": [
            {"v": 1},
            {"v": 2},
            {"v": 3},
        ],
    }

    manifest = [
        ("rows/*/v", "X"),
        ("rows/1/v", "Y"),
    ]

    result = apply(doc, manifest)

    # Row 1 is the only record both rules reach with different values, so it
    # is the only isolated record; the other rows still get the glob value.
    assert doc["rows"][0]["v"] == "X"
    assert doc["rows"][1]["v"] == 2
    assert doc["rows"][2]["v"] == "X"

    assert len(result.conflicts) == 2
    conflict_paths = {change.path for change in result.conflicts}
    assert conflict_paths == {"rows/1/v"}

    # Each conflicting record reports its own authored rule and original path.
    by_rule = {change.rules[0]: change for change in result.conflicts}
    assert set(by_rule) == {0, 1}
    assert by_rule[0].sources == ["rows/*/v"]
    assert by_rule[1].sources == ["rows/1/v"]
    assert by_rule[0].old_value == 2
    assert by_rule[1].old_value == 2
    assert by_rule[0].target == ("rows", 1, "v")
    assert by_rule[1].target == ("rows", 1, "v")
    assert by_rule[0].conflicts[0].rules == [1]
    assert by_rule[1].conflicts[0].rules == [0]


def test_equal_values_on_same_target_collapse_to_one_write():
    doc = {"rows": [{"v": 1}, {"v": 2}]}

    result = apply(doc, [("rows/*/v", 9), (["rows", "*", "v"], 9)])

    assert result.conflicts == []
    assert [change.path for change in result.applied] == ["rows/0/v", "rows/1/v"]
    assert doc["rows"][0]["v"] == 9
    assert doc["rows"][1]["v"] == 9


def test_in_place_writes_keep_list_rows_stable():
    doc = {
        "rows": [
            {"v": 1},
            {"v": 2},
            {"v": 3},
            {"v": 4},
        ],
    }

    manifest = [
        ("rows/0/v", "first"),
        ("rows/3/v", "last"),
        ("rows/*/v", "globbed"),
    ]

    result = apply(doc, manifest)

    # Nothing is inserted or removed from the list, so the rows the later
    # rules resolved against never shift. Only rows 0 and 3 disagree with the
    # glob and therefore become isolated conflicts.
    assert len(doc["rows"]) == 4
    assert doc["rows"][0]["v"] == 1
    assert doc["rows"][3]["v"] == 4
    assert doc["rows"][1]["v"] == "globbed"
    assert doc["rows"][2]["v"] == "globbed"
    assert sorted({change.path for change in result.conflicts}) == [
        "rows/0/v",
        "rows/3/v",
    ]


def test_numeric_string_key_and_integer_index_are_distinct_targets():
    doc = {
        "named": {"0": "string key", "1": "another string key"},
        "indexed": ["list index 0", "list index 1"],
    }

    # An integer segment can only address the list index.
    result = apply(doc, [(["indexed", 0], "via int")])
    assert result.applied[0].target == ("indexed", 0)
    assert doc["indexed"][0] == "via int"
    assert doc["named"]["0"] == "string key"

    # A literal decimal string can only address the dictionary key.
    result = apply(
        doc,
        [({"glob": ["named", "0"], "value": "via literal", "literal": True})],
    )
    assert result.applied[0].target == ("named", "0")
    assert doc["named"]["0"] == "via literal"
    assert doc["indexed"][0] == "via int"

    # A glob decimal string follows ordinary dpath conventions and can reach
    # both kinds, but the resolved targets stay type tagged and distinct.
    preview = plan(doc, [("*/0", "both")])
    targets = {change.target for change in preview}
    assert ("named", "0") in targets
    assert ("indexed", 0) in targets
    assert all(change.path.endswith("/0") for change in preview)


def test_conflicts_isolated_per_target_keep_typed_paths():
    doc = {
        "named": {"0": "s"},
        "indexed": ["i"],
    }

    result = apply(doc, [("*/0", "one"), ("*/0", "two")])

    # The rendered paths collide, but the two records per disagreeing value
    # stay attached to their own typed original target.
    assert len(result.conflicts) == 4
    targets = {change.target for change in result.conflicts}
    assert targets == {("named", "0"), ("indexed", 0)}
    for target in targets:
        records = [change for change in result.conflicts if change.target == target]
        assert {change.new_value for change in records} == {"one", "two"}
        assert all(change.path.endswith("/0") for change in records)

    assert doc["named"]["0"] == "s"
    assert doc["indexed"][0] == "i"


def test_star_star_mixed_with_literal_segments():
    doc = {
        "a": {
            "b": {"k": 1, "other": 2},
            "c": {"k": 3},
            "literal": 4,
        },
    }

    result = apply(
        doc,
        [
            ("a/**/k", "found"),
            ({"glob": ["a", "literal"], "value": "exact", "literal": True}),
        ],
    )

    assert doc["a"]["b"]["k"] == "found"
    assert doc["a"]["c"]["k"] == "found"
    assert doc["a"]["b"]["other"] == 2
    assert doc["a"]["literal"] == "exact"
    assert sorted(change.path for change in result.applied) == [
        "a/b/k",
        "a/c/k",
        "a/literal",
    ]


def test_plan_does_not_mutate_and_previews_conflicts():
    doc = {"rows": [{"v": 1}, {"v": 2}]}
    snapshot = {"rows": [{"v": 1}, {"v": 2}]}

    preview = plan(doc, [("rows/*/v", "X"), ("rows/1/v", "Y")])

    assert doc == snapshot
    assert {change.path for change in preview} == {"rows/0/v", "rows/1/v"}
    conflicted = [change for change in preview if change.conflicts]
    assert {change.path for change in conflicted} == {"rows/1/v"}


def test_afilter_only_selects_matching_records():
    doc = {"a": {"b": 1, "c": 10, "d": 100}}

    result = apply(doc, [("a/*", 0)], afilter=lambda value: value >= 10)

    assert doc["a"] == {"b": 1, "c": 0, "d": 0}
    assert sorted(change.target[-1] for change in result.applied) == ["c", "d"]


def test_negative_index_and_custom_separator():
    doc = {"rows": [1, 2, 3]}

    result = apply(doc, [(";rows;-1", 30)], separator=";")

    assert doc["rows"] == [1, 2, 30]
    assert result.applied[0].target == ("rows", 2)
    assert result.applied[0].path == "rows;2"


def test_literal_string_path_with_glob_characters():
    doc = {"a": {"a*b": 1, "acb": 2}}

    result = apply(
        doc,
        [({"glob": "a/a*b", "value": 9, "literal": True})],
    )

    assert doc["a"]["a*b"] == 9
    assert doc["a"]["acb"] == 2
    assert result.applied[0].target == ("a", "a*b")


def test_glob_without_matches_changes_nothing():
    doc = {"a": {"b": 1}}

    result = apply(doc, [("a/missing/*", 2)])

    assert doc == {"a": {"b": 1}}
    assert result.changed == 0
    assert result.applied == []
    assert result.conflicts == []


def test_rules_resolve_before_any_write():
    # Rules resolve against the original document snapshot: a glob for a key
    # that exists nowhere originally must not create records just because
    # another rule's writes happen first.
    doc = {"rows": [{"v": 1}, {"v": 2}]}

    result = apply(doc, [("rows/*/v", 5), ("rows/*/other", 6)])

    assert doc == {"rows": [{"v": 5}, {"v": 5}]}
    assert all("other" not in row for row in doc["rows"])
    assert result.changed == 2
    assert result.conflicts == []


def test_replay_skips_paths_absent_from_target():
    target = {"a": {"b": 0}}

    # Stored search hits include a path that no longer exists in the target
    # document; replay must skip it instead of recreating the missing branch.
    result = replay(target, [("a/b", 1), ("a/missing/deep", 2)])

    assert target == {"a": {"b": 1}}
    assert [change.path for change in result.applied] == ["a/b"]
    assert result.conflicts == []


def test_bytes_keys_keep_their_type():
    doc = {"a": {b"k": 1, "k": 2}, "L": [7]}

    result = apply(doc, [("a/*", 9), (["L", 0], 8)])

    assert doc["a"][b"k"] == 9
    assert doc["a"]["k"] == 9
    assert doc["L"][0] == 8
    assert {change.target for change in result.applied} == {
        ("a", b"k"),
        ("a", "k"),
        ("L", 0),
    }


def test_star_star_expands_zero_or_more_segments():
    doc = {"a": {"k": 1, "b": {"k": 2, "b2": {"k": 3}}}}

    result = apply(doc, [("a/**/k", 0)])

    assert doc["a"]["k"] == 0
    assert doc["a"]["b"]["k"] == 0
    assert doc["a"]["b"]["b2"]["k"] == 0
    assert doc["a"]["b"]["b2"] == {"k": 0}
    assert sorted(change.path for change in result.applied) == [
        "a/b/b2/k",
        "a/b/k",
        "a/k",
    ]


def test_invalid_manifest_rules_are_rejected_before_writes():
    doc = {"a": {"b": 1}}

    with pytest.raises(TypeError):
        apply(doc, [("a/b",)])
    with pytest.raises(TypeError):
        apply(doc, [{"glob": "a/b"}])
    with pytest.raises(dpath.exceptions.InvalidGlob):
        apply(doc, [("a/**/**", 1)])

    assert doc == {"a": {"b": 1}}


def test_search_hits_replay_every_actual_change():
    original = {
        "rows": [
            {"v": 1},
            {"v": 2},
            {"v": 3},
        ],
        "meta": {"a*b": 9, "acb": 8},
    }
    manifest = [
        ("rows/*/v", "X"),
        ("rows/2/v", "Z"),
        ("meta/a*b", 7),
    ]

    applied_doc = {"rows": [dict(row) for row in original["rows"]], "meta": dict(original["meta"])}
    result = apply(applied_doc, manifest)

    # Rebuild the same manifest purely out of dpath.search hits, exactly as a
    # caller would when replaying a stored revision list.
    replayed_hits = []
    for glob, value in manifest:
        for path, _ in dpath.search(original, glob, yielded=True):
            replayed_hits.append((path, value))

    replayed_doc = {"rows": [dict(row) for row in original["rows"]], "meta": dict(original["meta"])}
    replay_result = replay(replayed_doc, replayed_hits)

    # The replayed writes reconcile one to one with the applied changes,
    # including the record that both runs isolate for the same reason.
    assert replay_result.changed == result.changed
    applied_by_path = {change.path: change.new_value for change in result.applied}
    replayed_by_path = {change.path: change.new_value for change in replay_result.applied}
    assert replayed_by_path == applied_by_path
    assert {change.path for change in replay_result.conflicts} == {
        change.path for change in result.conflicts
    }
    # Replayed list indices carry the same int segment types as apply(), even
    # though the search hits arrived as separator joined strings.
    assert any(
        change.target == ("rows", 0, "v") and isinstance(change.target[1], int)
        for change in replay_result.applied
    )
    assert replayed_doc == applied_doc

    # Replaying the isolated conflict on its own keeps the record untouched
    # and reports both disagreeing values with their original path.
    conflict_hits = [("rows/2/v", "X"), ("rows/2/v", "Z")]
    conflict_doc = {"rows": [{"v": 1}, {"v": 2}, {"v": 3}]}
    conflict_result = replay(conflict_doc, conflict_hits)

    assert conflict_doc["rows"][2]["v"] == 3
    assert len(conflict_result.conflicts) == 2
    assert {change.new_value for change in conflict_result.conflicts} == {"X", "Z"}
    assert all(change.path == "rows/2/v" for change in conflict_result.conflicts)
