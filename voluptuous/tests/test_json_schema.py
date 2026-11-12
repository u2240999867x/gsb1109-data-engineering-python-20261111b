import pytest

from voluptuous import error as er
from voluptuous.json_schema import json_schema_accepts, to_json_schema
from voluptuous.schema_builder import (
    ALLOW_EXTRA,
    PREVENT_EXTRA,
    REMOVE_EXTRA,
    Extra,
    Exclusive,
    Inclusive,
    Msg,
    Object,
    Optional,
    Remove,
    Required,
    Schema,
)
from voluptuous.validators import All, Any, Coerce, Match, Range


def vol_accepts(schema, doc):
    try:
        schema(doc)
    except (er.MultipleInvalid, er.Invalid):
        return False
    return True


def assert_equivalent(schema, result, docs):
    for doc in docs:
        assert vol_accepts(schema, doc) == json_schema_accepts(
            result.schema, doc
        ), 'verdicts differ on %r' % (doc,)


# ---------------------------------------------------------------------------
# Exact exports
# ---------------------------------------------------------------------------


def test_scalar_types_exact():
    for node, expected, docs in [
        (str, {'type': 'string'}, ['a', '', 1, True, None, 1.5]),
        (bool, {'type': 'boolean'}, [True, False, 0, 1, 'x', None]),
        (type(None), {'type': 'null'}, [None, 0, False, '']),
        (None, {'const': None}, [None, 0, False]),
        ('lit', {'const': 'lit'}, ['lit', 'other', 1]),
        (1.5, {'const': 1.5}, [1.5, 1, True, 'x']),
    ]:
        schema = Schema(node)
        result = to_json_schema(schema)
        assert result.schema == expected
        assert result.complete, result.summary()
        assert_equivalent(schema, result, docs)


def test_dict_required_optional_prevent_extra():
    schema = Schema({Required('a'): str, Optional('b'): bool})
    result = to_json_schema(schema)
    assert result.schema == {
        'type': 'object',
        'properties': {'a': {'type': 'string'}, 'b': {'type': 'boolean'}},
        'required': ['a'],
        'additionalProperties': False,
    }
    assert result.complete
    assert result.covered == ['$.a', '$.b']
    assert_equivalent(
        schema,
        result,
        [
            {'a': 'x'},
            {'a': 'x', 'b': True},
            {},
            {'b': True},
            {'a': 1},
            {'a': 'x', 'c': 1},
            {'a': 'x', 'b': 1},
            [],
            'x',
        ],
    )


def test_schema_level_required_flag():
    schema = Schema({'a': str, Optional('b'): str}, required=True)
    result = to_json_schema(schema)
    assert result.schema['required'] == ['a']
    assert_equivalent(schema, result, [{'a': 'x'}, {'a': 'x', 'b': 'y'}, {}])


def test_extra_modes():
    docs = [{'a': 'x'}, {'a': 'x', 'junk': 1}, {}]
    schema = Schema({'a': str}, extra=ALLOW_EXTRA)
    result = to_json_schema(schema)
    assert result.schema['additionalProperties'] is True
    assert result.complete
    assert_equivalent(schema, result, docs)

    schema = Schema({'a': str}, extra=REMOVE_EXTRA)
    result = to_json_schema(schema)
    assert result.schema['additionalProperties'] is True
    # Acceptance is identical; only the output differs (noted).
    assert result.complete
    assert any('REMOVE_EXTRA' in n for n in result.notes)
    assert_equivalent(schema, result, docs)

    schema = Schema({'a': str}, extra=PREVENT_EXTRA)
    result = to_json_schema(schema)
    assert result.schema['additionalProperties'] is False
    assert_equivalent(schema, result, docs)


def test_required_with_default_is_optional_for_acceptance():
    schema = Schema({Required('a', default='x'): str})
    result = to_json_schema(schema)
    assert 'required' not in result.schema
    assert_equivalent(schema, result, [{}, {'a': 'y'}, {'a': 1}])


def test_wildcard_str_key():
    schema = Schema({'a': str, str: bool})
    result = to_json_schema(schema)
    assert result.schema == {
        'type': 'object',
        'properties': {'a': {'type': 'string'}},
        'additionalProperties': {'type': 'boolean'},
    }
    assert result.complete
    assert_equivalent(
        schema,
        result,
        [
            {'a': 'x'},
            {'a': 'x', 'b': True},
            {'a': 'x', 'b': 1},
            {'b': False},
            {'a': True},
        ],
    )


def test_extra_key_as_wildcard():
    schema = Schema({Extra: str})
    result = to_json_schema(schema)
    assert result.schema['additionalProperties'] == {'type': 'string'}
    assert_equivalent(schema, result, [{}, {'a': 'x'}, {'a': 1}])


def test_wildcard_shadowing():
    # Extra is callable (priority 3) and therefore precedes the str type
    # wildcard (priority 4) in voluptuous candidate order.
    schema = Schema({str: str, Extra: bool})
    result = to_json_schema(schema)
    assert result.schema['additionalProperties'] == {'type': 'boolean'}
    assert any('shadowed' in n for n in result.notes)
    assert_equivalent(schema, result, [{'a': 'x'}, {'a': True}, {}])


def test_non_str_keys_are_unreachable():
    schema = Schema({1: str, 'a': str})
    result = to_json_schema(schema)
    assert result.schema['properties'] == {'a': {'type': 'string'}}
    assert any('never appear in a JSON document' in n for n in result.notes)
    # {1: str} is dead for JSON input; "{'1': ...}" is an extra key on both
    # sides.
    assert_equivalent(schema, result, [{'a': 'x'}, {'a': 'x', '1': 'y'}, {}])


def test_match_key_single_pattern():
    schema = Schema({Match('^a'): str, str: bool})
    result = to_json_schema(schema)
    assert result.schema['patternProperties'] == {'^(?:^a)': {'type': 'string'}}
    assert result.schema['additionalProperties'] == {'type': 'boolean'}
    assert result.complete
    assert_equivalent(
        schema,
        result,
        [
            {'abc': 'x'},
            {'abc': 1},
            {'xy': True},
            {'xy': 'z'},
            {'abc': 'x', 'xy': False},
            {},
        ],
    )


def test_match_key_excludes_literal_keys():
    # The literal rule wins in voluptuous; the exported pattern must not
    # apply to the literal key.
    schema = Schema({'abc': bool, Match('^a'): str})
    result = to_json_schema(schema)
    (pattern,) = result.schema['patternProperties']
    assert pattern == '^(?!(?:abc)$)(?:^a)'
    assert result.complete
    assert_equivalent(
        schema,
        result,
        [
            {'abc': True},
            {'abc': 'x'},
            {'abd': 'x'},
            {'abd': True},
        ],
    )


def test_list_of_single_schema():
    schema = Schema([str])
    result = to_json_schema(schema)
    assert result.schema == {'type': 'array', 'items': {'type': 'string'}}
    assert result.complete
    assert_equivalent(schema, result, [[], ['a'], ['a', 'b'], [1], 'x', {}])


def test_list_of_multiple_schemas_is_anyof():
    schema = Schema([str, bool])
    result = to_json_schema(schema)
    assert result.schema == {
        'type': 'array',
        'items': {'anyOf': [{'type': 'string'}, {'type': 'boolean'}]},
    }
    assert result.complete
    assert_equivalent(schema, result, [[], ['a', True], [1], ['a', 1]])


def test_empty_list_schema():
    schema = Schema([])
    result = to_json_schema(schema)
    assert result.schema == {'type': 'array', 'maxItems': 0}
    assert_equivalent(schema, result, [[], [1], 'x'])


def test_any_and_all():
    schema = Schema(Any('a', 'b', bool))
    result = to_json_schema(schema)
    assert result.schema == {
        'anyOf': [{'const': 'a'}, {'const': 'b'}, {'type': 'boolean'}]
    }
    assert result.complete
    assert_equivalent(schema, result, ['a', 'b', True, 'c', 1])

    schema = Schema(All(str, Any('a', 'b')))
    result = to_json_schema(schema)
    assert result.schema == {
        'allOf': [{'type': 'string'}, {'anyOf': [{'const': 'a'}, {'const': 'b'}]}]
    }
    assert result.complete
    assert_equivalent(schema, result, ['a', 'b', 'c', 1])


def test_any_of_dicts_respects_member_required_flag():
    schema = Schema(Any({'a': str}, {'b': str}, required=True))
    result = to_json_schema(schema)
    first, second = result.schema['anyOf']
    assert first['required'] == ['a']
    assert second['required'] == ['b']
    assert_equivalent(
        schema,
        result,
        [{'a': 'x'}, {'b': 'y'}, {'a': 'x', 'b': 'y'}, {}, {'a': 1}],
    )


def test_nested_schema_and_msg():
    schema = Schema(
        {'inner': Schema({'a': str}, extra=ALLOW_EXTRA), 'm': Msg(str, 'nope')}
    )
    result = to_json_schema(schema)
    assert result.schema['properties']['inner'] == {
        'type': 'object',
        'properties': {'a': {'type': 'string'}},
        'additionalProperties': True,
    }
    assert result.schema['properties']['m'] == {'type': 'string'}
    assert_equivalent(
        schema,
        result,
        [
            {'inner': {'a': 'x'}, 'm': 'y'},
            {'inner': {'a': 'x', 'z': 1}, 'm': 'y'},
            {'inner': {'a': 1}, 'm': 'y'},
            {'inner': {}, 'm': 1},
        ],
    )


def test_raw_schema_with_kwargs():
    result = to_json_schema({'a': str}, required=True, extra=ALLOW_EXTRA)
    assert result.schema['required'] == ['a']
    assert result.schema['additionalProperties'] is True


# ---------------------------------------------------------------------------
# Divergences with witnesses
# ---------------------------------------------------------------------------


def check_witnesses(vol_schema, result):
    """Every reported witness must genuinely diverge on both sides."""
    assert result.divergences
    for divergence in result.divergences:
        assert divergence.witnesses
        for witness in divergence.witnesses:
            assert vol_accepts(vol_schema, witness.document) == (
                witness.voluptuous_accepts
            ), witness
            assert json_schema_accepts(divergence.exported, witness.document) == (
                witness.json_schema_accepts
            ), witness
            assert witness.voluptuous_accepts != witness.json_schema_accepts


def test_int_divergence():
    schema = Schema(int)
    result = to_json_schema(schema)
    assert result.schema == {'type': 'integer'}
    assert not result.complete
    docs = [(w.document, w.voluptuous_accepts) for d in result.divergences for w in d.witnesses]
    assert (True, True) in docs  # isinstance(True, int)
    assert (1.0, False) in docs  # JSON Schema treats 1.0 as an integer
    check_witnesses(schema, result)


def test_float_divergence():
    schema = Schema(float)
    result = to_json_schema(schema)
    assert result.schema == {'type': 'number'}
    docs = [(w.document, w.voluptuous_accepts) for d in result.divergences for w in d.witnesses]
    assert (1, False) in docs
    check_witnesses(schema, result)


def test_bool_literal_divergence():
    schema = Schema(True)
    result = to_json_schema(schema)
    assert result.schema == {'const': True}
    docs = [(w.document, w.voluptuous_accepts) for d in result.divergences for w in d.witnesses]
    assert (1, True) in docs  # 1 == True in Python
    check_witnesses(schema, result)


def test_int_literal_divergence_only_for_zero_and_one():
    result = to_json_schema(Schema(1))
    assert result.divergences
    check_witnesses(Schema(1), result)

    result = to_json_schema(Schema(42))
    assert result.schema == {'const': 42}
    assert result.complete
    assert_equivalent(Schema(42), result, [42, 42.0, True, '42'])


def test_exclusive_group_divergence():
    schema = Schema(
        {
            Exclusive('alpha', 'angles'): str,
            Exclusive('beta', 'angles'): str,
        }
    )
    result = to_json_schema(schema)
    assert not result.complete
    check_witnesses(schema, result)
    witness = result.divergences[0].witnesses[0].document
    assert 'alpha' in witness and 'beta' in witness


def test_inclusive_group_divergence():
    schema = Schema(
        {
            Inclusive('filename', 'file'): str,
            Inclusive('mimetype', 'file'): str,
        }
    )
    result = to_json_schema(schema)
    assert not result.complete
    check_witnesses(schema, result)
    witness = result.divergences[0].witnesses[0].document
    assert 'filename' in witness and 'mimetype' not in witness


def test_overlapping_match_keys_divergence():
    schema = Schema({Match('a'): str, Match('ab'): bool})
    result = to_json_schema(schema)
    assert not result.complete
    check_witnesses(schema, result)
    witness = result.divergences[0].witnesses[0].document
    (key,) = [k for k in witness]
    assert key.startswith('a')  # matched by both patterns
    # voluptuous applies the first matching pattern only.
    assert vol_accepts(schema, witness)


# ---------------------------------------------------------------------------
# Unsupported rules
# ---------------------------------------------------------------------------


def test_python_function_never_executed():
    calls = []

    def spy(value):
        calls.append(value)
        return value

    schema = Schema({'good': str, 'bad': spy})
    result = to_json_schema(schema)
    assert calls == []
    assert not result.complete
    assert result.covered == ['$.good']
    assert [u.path for u in result.unsupported] == ['$.bad']
    # The unsupported column is still present but unconstrained.
    assert result.schema['properties']['bad'] is True
    # Sibling column stays fully enforced.
    assert_equivalent(
        schema, result, [{'good': 'x'}, {'good': 1}, {'good': 'x', 'bad': 1}]
    )
    summary = result.summary()
    assert 'PARTIAL' in summary and '$.good' in summary and '$.bad' in summary


@pytest.mark.parametrize(
    'node',
    [
        lambda v: v,
        Coerce(int),
        Range(min=1),
        Match('^a'),
        (str,),
        {str},
        Object({'a': str}),
    ],
)
def test_unsupported_nodes(node):
    result = to_json_schema(Schema(node))
    assert not result.complete
    assert result.unsupported


def test_any_with_callable_member_is_unsupported():
    result = to_json_schema(Schema(Any(str, lambda v: v)))
    assert result.schema is True
    assert result.unsupported


def test_all_with_removing_member_is_unsupported():
    result = to_json_schema(
        Schema(All({'a': str}, Schema({'b': str}, extra=REMOVE_EXTRA)))
    )
    assert result.schema is True
    assert result.unsupported


def test_remove_key_is_unsupported():
    result = to_json_schema(Schema({Remove('a'): str, 'b': str}))
    assert not result.complete
    assert result.covered == ['$.b']
    assert [u.path for u in result.unsupported] == ['$']


def test_list_with_callable_element_drops_items():
    result = to_json_schema(Schema([str, lambda v: v]))
    assert result.schema == {'type': 'array'}
    assert result.unsupported


def test_complete_contract_summary():
    schema = Schema(
        {
            Required('name'): str,
            Optional('tags'): [str],
            Optional('state'): Any('on', 'off'),
        }
    )
    result = to_json_schema(schema)
    assert result.complete
    assert result.covered == ['$.name', '$.tags', '$.state']
    assert 'COMPLETE' in result.summary()
    assert_equivalent(
        schema,
        result,
        [
            {'name': 'x'},
            {'name': 'x', 'tags': ['a'], 'state': 'on'},
            {'name': 'x', 'state': 'bad'},
            {'tags': []},
            {'name': 'x', 'extra': 1},
        ],
    )
