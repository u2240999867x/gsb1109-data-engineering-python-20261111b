import json
import os

import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.segmented import apply_field_ops, compile_condition


@pytest.fixture
def db():
    db_ = TinyDB(storage=MemoryStorage)
    db_.insert_multiple({'int': 1, 'char': c} for c in 'abc')
    return db_


def make_rules():
    return [
        {'cond': {'op': '==', 'path': 'char', 'value': 'a'},
         'fields': {'int': 2}},
        {'cond': {'op': '==', 'path': 'char', 'value': 'b'},
         'fields': {'int': 4}},
        {'cond': {'op': '==', 'path': 'char', 'value': 'zz'},
         'fields': {'int': 9}},
    ]


def test_segmented_matches_full_list(db: TinyDB):
    receipts = db.update_multiple_segmented(
        make_rules(), rule_segment_size=1, doc_segment_size=1)

    other = TinyDB(storage=MemoryStorage)
    other.insert_multiple({'int': 1, 'char': c} for c in 'abc')
    other.update_multiple([
        ({'int': 2}, where('char') == 'a'),
        ({'int': 4}, where('char') == 'b'),
        ({'int': 9}, where('char') == 'zz'),
    ])

    assert db.all() == other.all()
    assert receipts == [
        {'rule': 1, 'doc_ids': [1], 'count': 1},
        {'rule': 2, 'doc_ids': [2], 'count': 1},
        {'rule': 3, 'doc_ids': [], 'count': 0},
    ]


def test_segmented_one_shot_generator_source(db: TinyDB):
    receipts = db.update_multiple_segmented(
        ({'cond': {'op': '==', 'path': 'char', 'value': c},
          'fields': {'int': 2}} for c in 'abc'),
        rule_segment_size=2,
        doc_segment_size=2,
    )

    assert db.count(where('int') == 2) == 3
    assert [r['count'] for r in receipts] == [1, 1, 1]
    assert [r['doc_ids'] for r in receipts] == [[1], [2], [3]]


def test_segmented_later_rule_reads_earlier_writes(db: TinyDB):
    # Rule 1 rewrites 'char' on every row; rule 2 matches on the value
    # rule 1 wrote on the same row.
    receipts = db.update_multiple_segmented([
        {'cond': {'op': '==', 'path': 'int', 'value': 1},
         'fields': {'char': 'x'}},
        {'cond': {'op': '==', 'path': 'char', 'value': 'x'},
         'fields': {'flag': True}},
    ], rule_segment_size=1, doc_segment_size=2)

    assert db.count(where('flag') == True) == 3  # noqa: E712
    assert receipts[1] == {'rule': 2, 'doc_ids': [1, 2, 3], 'count': 3}


def test_segmented_field_operations(db: TinyDB):
    db.update_multiple_segmented([
        {'cond': {'op': '==', 'path': 'char', 'value': 'a'},
         'fields': {'int': {'$inc': 5}, 'char': {'$unset': True},
                    'extra': {'$set': 'y'}}},
    ])

    doc = db.get(doc_id=1)
    assert doc == {'int': 6, 'extra': 'y'}
    assert db.get(doc_id=2) == {'int': 1, 'char': 'b'}


def test_segmented_resume_after_interruption(db: TinyDB, tmp_path):
    state_path = str(tmp_path / 'state.json')

    rules = [
        {'cond': {'op': '==', 'path': 'char', 'value': c},
         'fields': {'int': i}}
        for i, c in enumerate('abc', start=1)
    ]

    # Interrupt the first run after the second confirmed boundary
    calls = []

    def abort_after_two(doc_segment, rule_segment):
        calls.append((doc_segment, rule_segment))
        if len(calls) == 2:
            raise KeyboardInterrupt('simulated interruption')

    with pytest.raises(KeyboardInterrupt):
        db.update_multiple_segmented(
            iter(rules),
            rule_segment_size=1,
            doc_segment_size=1,
            state_path=state_path,
            on_checkpoint=abort_after_two,
        )

    assert os.path.exists(state_path)

    # Resume: the rule source is not consumed again, the run continues
    # from the confirmed row/rule boundary
    receipts = db.update_multiple_segmented(
        iter([]),
        rule_segment_size=1,
        doc_segment_size=1,
        state_path=state_path,
    )

    # Compare with an uninterrupted run on a fresh copy
    other = TinyDB(storage=MemoryStorage)
    other.insert_multiple({'int': 1, 'char': c} for c in 'abc')
    expected = other.update_multiple_segmented(
        iter(rules), rule_segment_size=1, doc_segment_size=1)

    assert receipts == expected
    assert db.all() == other.all()
    # Checkpoint and staging area are cleaned up after success
    assert not os.path.exists(state_path)
    assert not os.path.exists(state_path + '.staging')


def test_segmented_resume_receipts_match_single_pass(db: TinyDB, tmp_path):
    state_path = str(tmp_path / 'state.json')

    rules = [
        {'cond': {'op': '>=', 'path': 'int', 'value': 1},
         'fields': {'int': {'$inc': 1}}},
        {'cond': {'op': '==', 'path': 'int', 'value': 2},
         'fields': {'seen': True}},
    ]

    def abort_immediately(doc_segment, rule_segment):
        raise RuntimeError('simulated crash')

    with pytest.raises(RuntimeError):
        db.update_multiple_segmented(
            rules, rule_segment_size=1, doc_segment_size=1,
            state_path=state_path, on_checkpoint=abort_immediately)

    receipts = db.update_multiple_segmented(
        rules, rule_segment_size=1, doc_segment_size=1,
        state_path=state_path)

    other = TinyDB(storage=MemoryStorage)
    other.insert_multiple({'int': 1, 'char': c} for c in 'abc')
    expected = other.update_multiple_segmented(rules)

    assert receipts == expected
    assert db.all() == other.all()
    assert receipts == [
        {'rule': 1, 'doc_ids': [1, 2, 3], 'count': 3},
        {'rule': 2, 'doc_ids': [1, 2, 3], 'count': 3},
    ]


def test_segmented_explicit_rule_numbers(db: TinyDB):
    receipts = db.update_multiple_segmented([
        {'number': 10,
         'cond': {'op': '==', 'path': 'char', 'value': 'a'},
         'fields': {'int': 2}},
        {'number': 20,
         'cond': {'op': '==', 'path': 'char', 'value': 'b'},
         'fields': {'int': 4}},
    ])

    assert [r['rule'] for r in receipts] == [10, 20]
    assert [r['doc_ids'] for r in receipts] == [[1], [2]]


def test_segmented_duplicate_rule_numbers_rejected(db: TinyDB):
    with pytest.raises(ValueError):
        db.update_multiple_segmented([
            {'number': 1,
             'cond': {'op': 'noop'}, 'fields': {'int': 2}},
            {'number': 1,
             'cond': {'op': 'noop'}, 'fields': {'int': 3}},
        ])


def test_segmented_rejects_callback_rules(db: TinyDB):
    # Callable conditions / transforms cannot be serialized; they keep
    # their original entry point (update_multiple)
    with pytest.raises(TypeError):
        db.update_multiple_segmented([
            {'cond': where('char') == 'a', 'fields': {'int': 2}},
        ])

    with pytest.raises(TypeError):
        db.update_multiple_segmented([
            {'cond': {'op': 'noop'},
             'fields': lambda doc: doc.update({'int': 2})},
        ])

    with pytest.raises(TypeError):
        db.update_multiple_segmented([
            ({'int': 2}, where('char') == 'a'),
        ])

    # The callback rules still work through the original entry point
    db.update_multiple([({'int': 2}, where('char') == 'a')])
    assert db.count(where('int') == 2) == 1


def test_segmented_non_serializable_rule_rejected(db: TinyDB):
    with pytest.raises(TypeError):
        db.update_multiple_segmented([
            {'cond': {'op': '==', 'path': 'char', 'value': object()},
             'fields': {'int': 2}},
        ])


def test_segmented_empty_rules_and_table(db: TinyDB):
    assert db.update_multiple_segmented([]) == []

    empty = TinyDB(storage=MemoryStorage)
    receipts = empty.update_multiple_segmented([
        {'cond': {'op': 'noop'}, 'fields': {'int': 2}},
    ])
    assert receipts == [{'rule': 1, 'doc_ids': [], 'count': 0}]


def test_segmented_invalid_segment_sizes(db: TinyDB):
    with pytest.raises(ValueError):
        db.update_multiple_segmented([], rule_segment_size=0)
    with pytest.raises(ValueError):
        db.update_multiple_segmented([], doc_segment_size=0)


def test_compile_condition_operators():
    doc = {'a': {'b': 5}, 's': 'hello world', 'lst': [1, 2, 3], 'x': 1}

    assert compile_condition({'op': '==', 'path': 'a.b', 'value': 5})(doc)
    assert compile_condition({'op': '!=', 'path': ['x'], 'value': 2})(doc)
    assert compile_condition({'op': '<', 'path': 'x', 'value': 2})(doc)
    assert compile_condition({'op': '<=', 'path': 'x', 'value': 1})(doc)
    assert compile_condition({'op': '>', 'path': 'x', 'value': 0})(doc)
    assert compile_condition({'op': '>=', 'path': 'x', 'value': 1})(doc)
    assert compile_condition({'op': 'exists', 'path': 'a'})(doc)
    assert compile_condition(
        {'op': 'matches', 'path': 's', 'pattern': r'\w+ \w+'})(doc)
    assert compile_condition(
        {'op': 'search', 'path': 's', 'pattern': r'world'})(doc)
    assert compile_condition(
        {'op': 'one_of', 'path': 'x', 'items': [1, 2]})(doc)
    assert compile_condition(
        {'op': 'any', 'path': 'lst', 'items': [2, 9]})(doc)
    assert compile_condition(
        {'op': 'all', 'path': 'lst', 'items': [1, 2, 3]})(doc)
    assert compile_condition(
        {'op': 'any', 'path': 'lst',
         'cond': {'op': '==', 'path': '', 'value': 2}})(doc)
    assert compile_condition(
        {'op': 'and', 'conds': [
            {'op': '==', 'path': 'x', 'value': 1},
            {'op': 'exists', 'path': 's'},
        ]})(doc)
    assert compile_condition(
        {'op': 'or', 'conds': [
            {'op': '==', 'path': 'x', 'value': 99},
            {'op': '==', 'path': 'x', 'value': 1},
        ]})(doc)
    assert compile_condition(
        {'op': 'not', 'cond': {'op': '==', 'path': 'x', 'value': 2}})(doc)
    assert compile_condition(
        {'op': 'fragment', 'document': {'x': 1}})(doc)
    assert compile_condition({'op': 'noop'})(doc)

    assert not compile_condition(
        {'op': '==', 'path': 'x', 'value': 2})(doc)

    with pytest.raises(ValueError):
        compile_condition({'op': 'unknown'})
    with pytest.raises(ValueError):
        compile_condition({'path': 'x'})
    with pytest.raises(TypeError):
        compile_condition(where('x') == 1)


def test_apply_field_ops():
    doc = {'a': 1, 'b': 2}
    apply_field_ops(doc, {
        'a': {'$inc': 4},
        'b': {'$unset': True},
        'c': {'$set': 'x'},
        'd': 9,
        'e': {'$inc': 1},
    })
    assert doc == {'a': 5, 'c': 'x', 'd': 9, 'e': 1}

    # A plain mapping value is stored as-is
    doc = {}
    apply_field_ops(doc, {'nested': {'k': 1}})
    assert doc == {'nested': {'k': 1}}


def test_segmented_rules_staged_with_stable_numbers(tmp_path):
    # The staged rule file assigns stable numbers in source order and is
    # itself JSON-serializable
    state_path = str(tmp_path / 'state.json')
    db = TinyDB(storage=MemoryStorage)
    db.insert({'int': 1})

    def stop(*args):
        raise RuntimeError

    with pytest.raises(RuntimeError):
        db.update_multiple_segmented(
            ({'cond': {'op': 'noop'}, 'fields': {'n': i}}
             for i in range(5)),
            rule_segment_size=2,
            state_path=state_path,
            on_checkpoint=stop,
        )

    with open(state_path + '.staging/rules-000000.jsonl') as fh:
        staged = [json.loads(line) for line in fh]
    assert [r['number'] for r in staged] == [1, 2]

    with open(state_path + '.staging/rules-000002.jsonl') as fh:
        staged = [json.loads(line) for line in fh]
    assert [r['number'] for r in staged] == [5]
