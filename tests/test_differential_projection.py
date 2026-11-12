"""Differential regression for the two dpath projection plans.

The minimal (document, glob) combinations below are the shrunk outputs of
differential/diff_regression.py. Library defects are marked
``xfail(strict=True)``: they document today's behavior and will start failing
if the library is fixed, prompting this file to be updated. Engine mechanics
that are relied upon use plain assertions.

Comparison order used throughout: path (multi)set -> row count -> raw value.
"""
import dpath
import pytest
from dpath.exceptions import InvalidGlob


def flat(doc, glob):
    return list(dpath.search(doc, glob, yielded=True, dirs=False))


def tree(doc, glob):
    return dpath.search(doc, yielded=False, dirs=False, glob=glob)


# --------------------------------------------------------------------- #
# Engine mechanics that hold in both plans (plain assertions)
# --------------------------------------------------------------------- #
def test_star_star_zero_segment_collapse_matches_in_both_plans():
    doc = {"x": 1}
    assert flat(doc, "**/x") == [("x", 1)]
    assert tree(doc, "**/x") == {"x": 1}


def test_negative_index_resolves_to_last_element_in_flat_plan():
    assert flat({"a": [10, 20]}, "a/-1") == [("a/1", 20)]


def test_double_star_star_pattern_is_rejected_not_empty():
    with pytest.raises(InvalidGlob):
        flat({"a": 1}, "**/**/a")
    with pytest.raises(InvalidGlob):
        tree({"a": 1}, "**/**/a")


def test_legal_non_matching_glob_is_empty_not_rejected():
    # Distinct contract outcome from a rejection: no exception, zero rows.
    assert flat({"a": 1}, "x/**") == []
    assert tree({"a": 1}, "x/**") == {}


# --------------------------------------------------------------------- #
# C1: tree plan cannot rebuild a root-level list
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="ListIndex is unhashable as a dict key")
def test_c1_root_level_list_crashes_tree_plan_but_flat_survives():
    doc = [0]
    assert flat(doc, "*") == [("0", 0)]          # flat plan works
    tree(doc, "*")                                # tree plan raises TypeError


# --------------------------------------------------------------------- #
# E1: distinct int/text source positions collapse to one flat column.
# The flat API cannot express a type-aware ideal without changing its
# contract, so these are characterization tests that pin today's behavior;
# the tree plan's correct behavior is asserted below.
# --------------------------------------------------------------------- #
def test_e1_int_and_text_keys_share_one_flat_column_name():
    rows = flat({0: "x", "0": "y"}, "*")
    assert rows == [("0", "x"), ("0", "y")]      # repeated column identity
    assert len({path for path, _ in rows}) == 1  # one distinct column name


def test_e1_column_keyed_consumer_silently_overwrites_one_position():
    rows = flat({0: "x", "0": "y"}, "*")
    assert dict(rows) == {"0": "y"}              # the int-key value is lost


def test_e1_tree_plan_preserves_typed_int_and_text_keys():
    result = tree({0: "x", "0": "y"}, "*")
    assert set(result.keys()) == {0, "0"}


def test_e1_shared_value_siblings_produce_identical_flat_rows():
    rows = flat({0: "S", "0": "S"}, "*")
    assert rows == [("0", "S"), ("0", "S")]      # two source positions, identical rows
    assert len(rows) == 2 and len({(p, v) for p, v in rows}) == 1
    assert set(tree({0: "S", "0": "S"}, "*").keys()) == {0, "0"}


# --------------------------------------------------------------------- #
# E2: tree rebuild injects None padding absent from the source
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="tree list reconstruction pads earlier indices with None")
def test_e2_tree_plan_adds_phantom_none_padding_row():
    doc = {"a": [10, 20]}
    assert flat(doc, "a/-1") == [("a/1", 20)]    # one real row
    assert tree(doc, "a/-1") == {"a": [20]}      # today: {"a": [None, 20]}


# --------------------------------------------------------------------- #
# E3: bytes keys are silently invisible to str globs
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="bytes key dropped rather than matched or rejected")
def test_e3_bytes_key_invisible_to_str_glob():
    assert flat({b"k": 1}, "*") == [("k", 1)]
    assert tree({b"k": 1}, "*") == {b"k": 1}


def test_c3_list_form_bytes_glob_crashes_flat_but_tree_survives():
    doc = {b"k": 1}
    # Flat plan joins the bytes path with a str separator -> TypeError.
    with pytest.raises(TypeError):
        flat(doc, [b"k"])
    # Tree plan keeps bytes keys and works.
    assert tree(doc, [b"k"]) == {b"k": 1}


def test_c3_bytes_object_glob_crashes_both_plans():
    doc = {b"k": 1}
    with pytest.raises(TypeError):
        flat(doc, b"k")
    with pytest.raises(TypeError):
        tree(doc, b"k")


def test_c2_empty_string_key_is_rejected_in_both_plans():
    from dpath.exceptions import InvalidKeyName
    doc = {"": 1}
    with pytest.raises(InvalidKeyName):
        flat(doc, "*")
    with pytest.raises(InvalidKeyName):
        tree(doc, "*")


# --------------------------------------------------------------------- #
# E4: leading-zero text glob is coerced to a list index
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="'00' is int()-coerced and matches index 0")
def test_e4_leading_zero_glob_matches_list_index_zero():
    doc = {"a": [7]}
    assert flat(doc, "a/00") == []               # literal "00" should not match "0"
    assert tree(doc, "a/00") == {}


# --------------------------------------------------------------------- #
# Bounded-contract invariant guard against new, unclassified behavior
# --------------------------------------------------------------------- #
def test_small_contract_only_known_outcomes_occur():
    try:
        from differential.diff_regression import (
            gen_documents, gen_patterns, classify, collision_docs,
            COLLISION_PATTERNS,
        )
    except Exception:
        pytest.skip("differential harness not importable from this path")

    known = {"consistent", "empty", "rejected", "plan_error", "diverge"}
    # Depth-1 universe plus the curated collision documents; full depth-2
    # exhaustion lives in differential/diff_regression.py.
    sample = gen_documents(1) + [doc for _, doc in collision_docs()]
    for doc in sample:
        for pattern in list(gen_patterns()) + list(COLLISION_PATTERNS):
            kind, _ = classify(doc, pattern, dirs=False)
            assert kind in known
