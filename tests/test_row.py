import pytest

import dpath
from dpath.exceptions import AmbiguousShapeError, InfeasibleShapeError
from dpath.row import Auto, Density, Dense, Index, NodeKind, Sparse, Text, coerce


# ---------------------------------------------------------------------------
# Explicit declarations: text key "0" vs array index 0
# ---------------------------------------------------------------------------

def test_text_zero_stays_a_dictionary_key():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Text("0")], "v1"),
        ("c2", [Text("a"), Text("1")], "v2"),
    ])
    assert target == {"a": {"0": "v1", "1": "v2"}}
    assert isinstance(target["a"], dict)


def test_index_zero_creates_a_sequence():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Index(0)], "v0"),
        ("c2", [Text("a"), Index(1)], "v1"),
    ])
    assert target == {"a": ["v0", "v1"]}
    assert isinstance(target["a"], list)


def test_text_and_index_on_the_same_edge_are_infeasible():
    target = {}
    with pytest.raises(InfeasibleShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Text("0")], 1),
            ("c2", [Text("a"), Index(0)], 2),
        ])
    assert set(exc.value.sources) == {"c1", "c2"}
    assert target == {}


# ---------------------------------------------------------------------------
# Whole-row planning with shared, not-yet-existing parents
# ---------------------------------------------------------------------------

def test_later_column_pins_shared_parent_to_sequence():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Auto("0"), Text("x")], 1),
        ("c2", [Text("a"), Index(0), Text("y")], 2),
    ])
    assert target == {"a": [{"x": 1, "y": 2}]}


def test_later_column_pins_shared_parent_to_mapping():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Auto("0"), Text("k1")], 1),
        ("c2", [Text("a"), Text("name"), Text("k2")], 2),
    ])
    assert target == {"a": {"0": {"k1": 1}, "name": {"k2": 2}}}


def test_ambiguous_row_lists_two_distinct_structures():
    target = {}
    with pytest.raises(AmbiguousShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Auto("0")], 1),
        ])
    structures = exc.value.structures
    assert len(structures) == 2
    assert {"a": {"0": 1}} in structures
    assert {"a": [1]} in structures
    # Nothing is written while the shape is undecided.
    assert target == {}


def test_plan_row_is_a_dry_run():
    target = {}
    structures = dpath.plan_row(target, [
        ("c1", [Text("a"), Auto("0")], 1),
    ])
    assert len(structures) == 2
    assert target == {}


def test_two_autos_on_one_edge_remain_two_feasible_shapes():
    target = {}
    structures = dpath.plan_row(target, [
        ("c1", [Text("a"), Auto("0"), Text("x")], 1),
        ("c2", [Text("a"), Auto("1"), Text("y")], 2),
    ])
    assert structures == [
        {"a": {"0": {"x": 1}, "1": {"y": 2}}},
        {"a": [{"x": 1}, {"y": 2}]},
    ]


def test_non_decimal_key_forces_mapping_reading_of_auto_sibling():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Auto("0")], 1),
        ("c2", [Text("a"), Text("name")], 2),
    ])
    assert target == {"a": {"0": 1, "name": 2}}


# ---------------------------------------------------------------------------
# Sparse vs dense lists
# ---------------------------------------------------------------------------

def test_sparse_index_allows_holes():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Sparse(2)], "x"),
    ])
    assert target == {"a": [None, None, "x"]}


def test_dense_indices_reject_holes_and_report_sources():
    target = {}
    with pytest.raises(InfeasibleShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Dense(0)], "x"),
            ("c2", [Text("a"), Dense(2)], "y"),
        ])
    assert set(exc.value.sources) == {"c1", "c2"}
    assert target == {}


def test_dense_indices_accept_contiguous_layout():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Dense(0)], "x"),
        ("c2", [Text("a"), Dense(1)], "y"),
    ])
    assert target == {"a": ["x", "y"]}


def test_conflicting_density_declarations_are_infeasible():
    target = {}
    with pytest.raises(InfeasibleShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Sparse(1)], "x"),
            ("c2", [Text("a"), Dense(2)], "y"),
        ])
    assert set(exc.value.sources) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# Live document participates in the shape decision
# ---------------------------------------------------------------------------

def test_live_sequence_pins_auto_segment():
    target = {"a": []}
    dpath.apply_row(target, [("c1", [Text("a"), Auto("0")], "v")])
    assert target == {"a": ["v"]}


def test_live_mapping_pins_auto_segment():
    target = {"a": {"0": "old"}}
    dpath.apply_row(target, [("c1", [Text("a"), Auto("0")], "v")])
    assert target == {"a": {"0": "v"}}


def test_live_container_kind_conflicting_with_declaration_is_infeasible():
    target = {"a": {}}
    with pytest.raises(InfeasibleShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Index(0), Text("x")], 1),
        ])
    assert "c1" in exc.value.sources
    assert target == {"a": {}}


def test_descent_through_a_live_leaf_is_infeasible():
    target = {"keep": [10]}
    with pytest.raises(InfeasibleShapeError):
        dpath.apply_row(target, [
            ("c1", [Text("keep"), Text("deep")], 2),
        ])
    assert target == {"keep": [10]}


def test_leaf_and_container_at_one_location_cannot_coexist():
    target = {}
    with pytest.raises(InfeasibleShapeError) as exc:
        dpath.apply_row(target, [
            ("c1", [Text("a"), Text("b")], 1),
            ("c2", [Text("a"), Text("b"), Text("c")], 2),
        ])
    assert set(exc.value.sources) == {"c1", "c2"}
    assert target == {}


# ---------------------------------------------------------------------------
# Shorthand coercion
# ---------------------------------------------------------------------------

def test_shorthand_int_is_index_and_decimal_string_is_auto():
    target = {}
    dpath.apply_row(target, [
        ("c1", ["a", 0, "x"], 1),
        ("c2", ["a", "0", "y"], 2),
    ])
    assert target == {"a": [{"x": 1, "y": 2}]}


def test_coerce_maps_each_shorthand_kind():
    segments = list(coerce(["name", "0", 3]))
    assert isinstance(segments[0], Text)
    assert isinstance(segments[1], Auto)
    assert isinstance(segments[2], Index)


def test_explicit_segments_pass_through_coerce():
    explicit = Text("0")
    assert list(coerce([explicit])) == [explicit]


# ---------------------------------------------------------------------------
# Creator contract: once per adopted node, transactional rollback
# ---------------------------------------------------------------------------

def test_creator_runs_once_per_adopted_node():
    calls = []

    def creator(current, path, i, hints=()):
        calls.append((type(current).__name__, i))
        dpath.segments._default_creator(current, path, i, hints)

    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Index(0), Text("x")], 1),
        ("c2", [Text("a"), Index(0), Text("y")], 2),
        ("c3", [Text("a"), Index(1), Text("z")], 3),
    ], creator=creator)

    assert target == {"a": [{"x": 1, "y": 2}, {"z": 3}]}
    # Adopted intermediate nodes: a, a/0, a/1 -> exactly three calls.
    assert len(calls) == 3


def test_creator_failure_leaves_no_half_record():
    def bad_creator(current, path, i, hints=()):
        raise RuntimeError("boom")

    target = {"keep": 1}
    with pytest.raises(RuntimeError):
        dpath.apply_row(target, [
            ("c1", [Text("a"), Index(0), Text("x")], 1),
            ("c2", [Text("a"), Index(1), Text("y")], 2),
        ], creator=bad_creator)
    assert target == {"keep": 1}


def test_mid_row_failure_rolls_back_everything_written_by_the_row():
    def flaky_creator(current, path, i, hints=()):
        if i >= 1:
            raise RuntimeError("boom mid row")
        dpath.segments._default_creator(current, path, i, hints)

    target = {"keep": {"nested": 1}}
    with pytest.raises(RuntimeError):
        dpath.apply_row(target, [
            ("c1", [Text("a"), Index(0), Text("x")], 1),
            ("c2", [Text("a"), Index(1), Text("y")], 2),
        ], creator=flaky_creator)
    assert target == {"keep": {"nested": 1}}


def test_leaf_write_failure_also_rolls_back():
    class GuardedDict(dict):
        def __setitem__(self, key, value):
            if key == "bomb":
                raise RuntimeError("leaf boom")
            super().__setitem__(key, value)

    target = GuardedDict()
    with pytest.raises(RuntimeError):
        dpath.apply_row(target, [
            ("c1", [Text("a"), Text("ok")], 1),
            ("c2", [Text("bomb")], 2),
        ])
    assert dict(target) == {}


# ---------------------------------------------------------------------------
# Declarations and enums
# ---------------------------------------------------------------------------

def test_density_enum_values():
    assert Density.SPARSE.value == "sparse"
    assert Density.DENSE.value == "dense"


def test_node_kind_enum_values():
    assert NodeKind.MAPPING.value == "mapping"
    assert NodeKind.SEQUENCE.value == "sequence"


def test_index_normalizes_string_digits_to_int():
    assert Index("2").key == 2
    assert Dense("2").key == 2
    assert Sparse("2").key == 2


def test_auto_on_non_decimal_string_is_mapping_only():
    segment = Auto("name")
    assert segment.roles == frozenset({NodeKind.MAPPING})
    assert segment.alt_key is None


# ---------------------------------------------------------------------------
# Deeply nested and root-level containers
# ---------------------------------------------------------------------------

def test_nested_sequence_under_sequence_uses_planned_roles():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Index(0), Index(0)], 7),
    ])
    assert target == {"a": [[7]]}
    assert isinstance(target["a"][0], list)


def test_mixed_mapping_and_sequence_nesting():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Index(0), Text("b"), Index(1)], 7),
    ])
    assert target == {"a": [{"b": [None, 7]}]}


def test_siblings_inside_a_nested_sequence_share_one_created_node():
    target = {}
    dpath.apply_row(target, [
        ("c1", [Text("a"), Index(0), Index(0), Text("k")], 1),
        ("c2", [Text("a"), Index(0), Index(1), Text("j")], 2),
    ])
    assert target == {"a": [[{"k": 1}, {"j": 2}]]}


def test_row_can_target_a_sequence_root():
    target = []
    dpath.apply_row(target, [
        ("c1", [Index(0), Text("k")], 1),
        ("c2", [Index(1), Text("j")], 2),
    ])
    assert target == [{"k": 1}, {"j": 2}]
