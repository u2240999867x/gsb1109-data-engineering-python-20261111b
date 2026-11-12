"Test column-order reconciliation on top of OrderedSet."

import pytest

from sortedcollections import (
    ColumnReconcileError,
    OrderedSet,
    reconcile_columns,
)


def test_unique_rename_is_claimed_without_templates():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y'],
        {'a': [('x',)], 'b': [('y',)]},
    )
    assert result.feasible
    assert result.consistent == {'a': ('x',), 'b': ('y',)}
    assert result.ambiguous == ()
    assert result.witness_mappings == ()
    assert result.solution_count == 1
    assert result.mapping == result.consistent


def test_split_claims_a_consecutive_run_of_new_columns():
    result = reconcile_columns(
        ['a', 'b'], ['p', 'q', 'r'],
        {'a': [('p', 'q')], 'b': [('r',)]},
        [{'name': 't1', 'old': ['a', 'b'], 'new': ['p', 'q', 'r']}],
    )
    assert result.consistent == {'a': ('p', 'q'), 'b': ('r',)}


def test_split_must_keep_old_relative_order():
    result = reconcile_columns(
        ['a', 'b'], ['p', 'q', 'r'],
        {'a': [('p', 'q')], 'b': [('r',)]},
        [{'name': 't1', 'old': ['a', 'b'], 'new': ['r', 'p', 'q']}],
    )
    assert not result.feasible
    assert [template.name for template in result.conflict] == ['t1']
    assert result.reasons
    assert any('order' in reason for reason in result.reasons)


def test_template_order_disambiguates_rename():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',)], 'b': [('y',), ('z',)]},
        [{'name': 't1', 'old': ['a', 'b'], 'new': ['z', 'x', 'y']}],
    )
    assert result.feasible
    assert result.consistent == {'a': ('x',), 'b': ('y',)}
    assert result.ambiguous == ()


def test_adjacency_evidence_disambiguates():
    result = reconcile_columns(
        ['a', 'b'], ['w', 'x', 'z'],
        {'a': [('x',), ('w',)], 'b': [('z',)]},
        [{'name': 't1', 'old': ['a', 'b'], 'new': ['w', 'x', 'z']}],
    )
    assert result.consistent == {'a': ('x',), 'b': ('z',)}


def test_ambiguous_field_yields_two_full_witness_mappings():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',)], 'b': [('y',), ('z',)]},
    )
    assert result.feasible
    assert result.consistent == {'a': ('x',)}
    assert result.ambiguous == ('b',)
    first, second = result.witness_mappings
    assert set(first) == {'a', 'b'}
    assert set(second) == {'a', 'b'}
    assert first['b'] != second['b']
    assert {first['b'], second['b']} == {('y',), ('z',)}
    assert first['a'] == second['a'] == ('x',)


def test_repeated_template_submission_cannot_resolve_ambiguity():
    templates = [
        {'name': 't1', 'old': ['b'], 'new': ['z']},
        {'name': 't1', 'old': ['b'], 'new': ['z']},
        (['b'], ['z']),
        ('t1', ['b'], ['z']),
    ]
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',), ('y',)], 'b': [('z',)]},
        templates,
    )
    assert len(result.templates) == 1
    assert result.ambiguous == ('a',)
    assert result.solution_count == 2

    pinned = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',), ('y',)], 'b': [('z',)]},
        templates + [
            {'name': 't2', 'old': ['a', 'b'], 'new': ['y', 'x', 'z']},
        ],
    )
    assert pinned.consistent == {'a': ('x',), 'b': ('z',)}


def test_same_old_column_keeps_one_interpretation_across_templates():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',), ('z',)], 'b': [('y',)]},
        [
            {'name': 'forces-z', 'old': ['a', 'b'], 'new': ['z', 'y', 'x']},
            {'name': 'forces-x', 'old': ['a', 'b'], 'new': ['x', 'y', 'z']},
        ],
    )
    assert not result.feasible
    names = {template.name for template in result.conflict}
    assert names == {'forces-z', 'forces-x'}
    for template in result.conflict:
        relaxed = reconcile_columns(
            ['a', 'b'], ['x', 'y', 'z'],
            {'a': [('x',), ('z',)], 'b': [('y',)]},
            [item for item in result.templates if item is not template],
        )
        assert relaxed.feasible


def test_conflict_core_is_minimal_under_removal():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',), ('z',)], 'b': [('y',)]},
        [
            {'name': 'forces-z', 'old': ['a', 'b'], 'new': ['z', 'y', 'x']},
            {'name': 'forces-x', 'old': ['a', 'b'], 'new': ['x', 'y', 'z']},
            {'name': 'neutral', 'old': ['b'], 'new': ['y']},
        ],
    )
    assert not result.feasible
    assert {template.name for template in result.conflict} == {
        'forces-z', 'forces-x'}
    assert len(result.conflict_cores) == 1
    assert len(result.reasons) >= 2


def test_independent_conflicts_are_reported_as_separate_cores():
    result = reconcile_columns(
        ['a', 'b'], ['n0', 'n1', 'n2', 'n3'],
        {
            'a': [('n0', 'n2', 'n3'), ('n1',)],
            'b': [('n1',), ('n0', 'n1')],
        },
        [
            {'name': 't0', 'old': ['b', 'a'],
             'new': ['n0', 'n3', 'n2', 'n1']},
            {'name': 't1', 'old': ['b', 'a'],
             'new': ['n0', 'n1', 'n2']},
            {'name': 't2', 'old': ['a', 'b'],
             'new': ['n3', 'n0', 'n1']},
        ],
    )
    assert not result.feasible
    core_names = sorted(
        sorted(template.name for template in core)
        for core in result.conflict_cores)
    assert core_names == [['t0'], ['t1']]
    assert {template.name for template in result.conflict} == {'t0', 't1'}
    assert len(result.reasons) == 2
    for core in result.conflict_cores:
        alone = reconcile_columns(
            ['a', 'b'], ['n0', 'n1', 'n2', 'n3'],
            {
                'a': [('n0', 'n2', 'n3'), ('n1',)],
                'b': [('n1',), ('n0', 'n1')],
            },
            list(core),
        )
        assert not alone.feasible


def test_injective_claims_collision_without_templates():
    result = reconcile_columns(
        ['a', 'b'], ['x'],
        {'a': [('x',)], 'b': [('x',)]},
    )
    assert not result.feasible
    assert result.conflict == ()
    assert result.reasons


def test_partial_templates_combine_into_one_ordering():
    result = reconcile_columns(
        ['a', 'b', 'c'], ['w', 'x', 'y', 'z'],
        {'a': [('x',), ('w',)], 'b': [('y',)], 'c': [('z',)]},
        [
            {'name': 't1', 'old': ['a', 'b'], 'new': ['w', 'x', 'y']},
            {'name': 't2', 'old': ['b', 'c'], 'new': ['y', 'z']},
        ],
    )
    assert result.feasible
    assert result.consistent['b'] == ('y',)
    assert result.consistent['c'] == ('z',)
    assert result.consistent['a'] == ('x',)


def test_orderedset_classmethod_entry_matches_function():
    function_result = reconcile_columns(
        ['a'], ['x'], {'a': [('x',)]})
    method_result = OrderedSet.reconcile_columns(
        ['a'], ['x'], {'a': [('x',)]})
    assert method_result.consistent == function_result.consistent
    ordered = OrderedSet(method_result.consistent['a'])
    assert list(ordered) == ['x']


def test_template_input_forms_are_accepted():
    candidates = {'a': [('x',)], 'b': [('y',)]}
    dict_form = reconcile_columns(
        ['a', 'b'], ['x', 'y'], candidates,
        [{'name': 't', 'old': ['a', 'b'], 'new': ['x', 'y']}])
    pair_form = reconcile_columns(
        ['a', 'b'], ['x', 'y'], candidates,
        [(['a', 'b'], ['x', 'y'])])
    triple_form = reconcile_columns(
        ['a', 'b'], ['x', 'y'], candidates,
        [('t', ['a', 'b'], ['x', 'y'])])
    assert dict_form.consistent == pair_form.consistent
    assert pair_form.consistent == triple_form.consistent
    assert pair_form.templates[0].name == 'template-0'


def test_unknown_fields_are_rejected():
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(['a'], ['x'], {'a': [('q',)]})
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(['a'], ['x'], {'q': [('x',)]})
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(
            ['a'], ['x'], {'a': [('x',)]},
            [{'old': ['q'], 'new': ['x']}])
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(
            ['a'], ['x'], {'a': [('x',)]},
            [{'old': ['a'], 'new': ['q']}])


def test_empty_and_malformed_interpretations_are_rejected():
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(['a'], ['x'], {'a': [()]})
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(
            ['a'], ['x'], {'a': [('x',)]},
            [(['a'], ['x'], ['extra'])])


def test_string_arguments_are_treated_as_single_rename():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y'],
        {'a': ['x'], 'b': 'y'},
    )
    assert result.consistent == {'a': ('x',), 'b': ('y',)}


def test_results_are_deterministic():
    kwargs = dict(
        old_fields=['a', 'b'],
        new_fields=['x', 'y', 'z'],
        candidates={'a': [('x',)], 'b': [('y',), ('z',)]},
        templates=[{'old': ['a', 'b'], 'new': ['z', 'x', 'y']}],
    )
    first = reconcile_columns(**kwargs)
    second = reconcile_columns(**kwargs)
    assert first.consistent == second.consistent
    assert first.conflict == second.conflict


def test_result_and_template_reprs_are_readable():
    result = reconcile_columns(
        ['a'], ['x'], {'a': [('x',)]},
        [{'name': 't1', 'old': ['a'], 'new': ['x']}])
    assert 'feasible' in repr(result)
    assert 't1' in repr(result.templates[0])
    conflicting = reconcile_columns(
        ['a', 'b'], ['x'], {'a': [('x',)], 'b': [('x',)]})
    assert 'conflicting' in repr(conflicting)


def test_malformed_template_shapes_are_rejected():
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(['a'], ['x'], {'a': [('x',)]}, [42])
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(
            ['a'], ['x'], {'a': [('x',)]},
            [(['a'], ['x'], ['extra'], ['shape'])])


def test_more_columns_than_anchors_is_infeasible_without_templates():
    result = reconcile_columns(
        ['a', 'b', 'c'], ['x', 'y'],
        {'a': [('x',)], 'b': [('y',)], 'c': [('x',), ('y',)]},
    )
    assert not result.feasible
    assert result.conflict == ()
    assert result.reasons
    assert result.ambiguous == ('a', 'b', 'c')


def test_search_time_anchor_collision_prunes_branch():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y'],
        {'a': [('x',), ('y',)], 'b': [('x',), ('y',)]},
    )
    assert result.feasible
    assert result.solution_count == 2
    assert result.ambiguous == ('a', 'b')
    assert result.witness_mappings[0] != result.witness_mappings[1]


def test_adjacency_violation_makes_templates_infeasible():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y', 'z'],
        {'a': [('x',)], 'b': [('y',)]},
        [{'name': 't1', 'old': ['a', 'b'], 'new': ['x', 'z', 'y']}],
    )
    assert not result.feasible
    assert [template.name for template in result.conflict] == ['t1']
    assert any('back-to-back' in reason for reason in result.reasons)


def test_template_order_objects_pass_through_unchanged():
    from sortedcollections import TemplateOrder
    order = TemplateOrder('mine', 0, ['a'], ['x'])
    result = reconcile_columns(
        ['a'], ['x'], {'a': [('x',)]}, [order])
    assert result.templates[0].name == 'mine'
    assert result.templates[0].index == 0


def test_duplicate_fields_and_candidates_are_collapsed():
    result = reconcile_columns(
        ['a', 'a', 'b'], ['x', 'x', 'y'],
        {'a': [('x',), ('x',)], 'b': [('y',), ('y',)]},
    )
    assert result.consistent == {'a': ('x',), 'b': ('y',)}
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y'],
        {'a': [('x',)], 'b': [('y',)], 'a': [('x',)]})
    assert result.consistent == {'a': ('x',), 'b': ('y',)}


def test_empty_candidate_list_is_rejected():
    with pytest.raises(ColumnReconcileError):
        reconcile_columns(['a'], ['x'], {'a': []})


def test_template_without_any_claimed_anchor_is_neutral():
    result = reconcile_columns(
        ['a', 'b'], ['x', 'y'],
        {'a': [('x',)], 'b': [('y',)]},
        [{'name': 't1', 'old': ['a'], 'new': ['y']}],
    )
    assert result.feasible
    assert result.consistent == {'a': ('x',), 'b': ('y',)}


def test_three_column_template_exercises_non_adjacent_pairs():
    result = reconcile_columns(
        ['a', 'b', 'c'], ['x', 'y', 'z'],
        {'a': [('x',)], 'b': [('y',)], 'c': [('z',)]},
        [{'name': 't1',
          'old': ['a', 'b', 'c'],
          'new': ['x', 'y', 'z']}],
    )
    assert result.consistent == {
        'a': ('x',), 'b': ('y',), 'c': ('z',)}


def test_template_order_instances_are_deduplicated_like_mappings():
    from sortedcollections import TemplateOrder
    result = reconcile_columns(
        ['a'], ['x'], {'a': [('x',)]},
        [
            TemplateOrder('first', 0, ['a'], ['x']),
            TemplateOrder('again', 5, ['a'], ['x']),
        ],
    )
    assert len(result.templates) == 1
    assert result.templates[0].name == 'first'
