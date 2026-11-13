import json

import pytest

from tinydb.queries import Query, QueryInstance, where
from tinydb.querydesc import (
    Catalog,
    ContractViolationError,
    ListField,
    ObjectField,
    QueryDescriptionError,
    QueryNotSerializableError,
    RuleComparison,
    ScalarField,
    UnsupportedQueryError,
    check_upgrade,
    compare_descriptions,
    compile_query,
    describe_query,
    description_from_json,
    description_to_json,
    validate_description,
)


def make_catalog():
    return Catalog({
        'category': ScalarField(['book', 'toy', 'food']),
        'price': ScalarField([1, 2, 3], nullable=True),
        'tags': ListField(['new', 'sale'], max_length=2, optional=True),
        'stock': ObjectField({
            'warehouse': ScalarField(['east', 'west']),
            'qty': ScalarField([0, 1], optional=True),
        }, optional=True),
        'a.b': ScalarField(['x', 'y'], optional=True),
    })


def describe(query):
    return describe_query(query)


# --- describe_query ----------------------------------------------------------


def test_describe_comparison():
    assert describe(where('price') <= 2) == {
        'version': 1,
        'root': {'op': 'cmp', 'path': ['price'], 'cmp': '<=', 'value': 2},
    }


def test_describe_all_comparison_operators():
    assert describe(where('p') == 1)['root']['cmp'] == '=='
    assert describe(where('p') != 1)['root']['cmp'] == '!='
    assert describe(where('p') < 1)['root']['cmp'] == '<'
    assert describe(where('p') <= 1)['root']['cmp'] == '<='
    assert describe(where('p') > 1)['root']['cmp'] == '>'
    assert describe(where('p') >= 1)['root']['cmp'] == '>='


def test_describe_boolean_combination_is_canonical():
    first = (where('category') == 'book') & (where('price') <= 2)
    second = (where('price') <= 2) & (where('category') == 'book')
    # frozenset iteration order must not leak into the description
    assert describe(first) == describe(second)
    root = describe(first)['root']
    assert root['op'] == 'and'
    assert len(root['args']) == 2


def test_describe_or_and_not():
    query = (where('category') == 'book') | ~(where('price') <= 2)
    root = describe(query)['root']
    assert root['op'] == 'or'
    assert {'op': 'not', 'arg': {'op': 'cmp', 'path': ['price'],
                                 'cmp': '<=', 'value': 2}} in root['args']


def test_describe_nested_and_flattens():
    query = ((where('a') == 1) & (where('b') == 2)) & (where('c') == 3)
    root = describe(query)['root']
    assert root['op'] == 'and'
    assert len(root['args']) == 3


def test_describe_any_all():
    assert describe(where('tags').any(['sale', 'new']))['root'] == {
        'op': 'any', 'path': ['tags'], 'values': ['sale', 'new']}
    assert describe(where('tags').all(['sale']))['root'] == {
        'op': 'all', 'path': ['tags'], 'values': ['sale']}


def test_describe_nested_field():
    root = describe(where('stock').warehouse == 'east')['root']
    assert root == {'op': 'cmp', 'path': ['stock', 'warehouse'],
                    'cmp': '==', 'value': 'east'}


def test_describe_literal_dot_key():
    root = describe(where('a.b') == 'x')['root']
    assert root == {'op': 'cmp', 'path': ['a.b'], 'cmp': '==', 'value': 'x'}


def test_describe_exists_and_one_of():
    assert describe(where('tags').exists())['root'] == {
        'op': 'exists', 'path': ['tags']}
    assert describe(where('category').one_of(['book', 'toy']))['root'] == {
        'op': 'one_of', 'path': ['category'], 'values': ['book', 'toy']}


def test_describe_noop():
    assert describe(Query().noop())['root'] == {'op': 'const', 'value': True}
    assert describe(~Query().noop())['root'] == {'op': 'const',
                                                 'value': False}


def test_describe_empty_query_rejected():
    with pytest.raises(QueryDescriptionError):
        describe(Query())
    with pytest.raises(QueryDescriptionError):
        describe(where('field'))


def test_describe_rejects_unknown_version():
    with pytest.raises(ValueError):
        describe_query(where('p') == 1, version=99)


def test_describe_python_test_points_to_source():
    def custom_test(value):
        return value == 42

    query = where('price').test(custom_test)
    with pytest.raises(QueryNotSerializableError) as exc_info:
        describe(query)
    error = exc_info.value
    assert error.func is custom_test
    assert error.location is not None
    file, line = error.location.rsplit(':', 1)
    assert file == __file__
    assert int(line) == custom_test.__code__.co_firstlineno
    assert 'custom_test' in str(error)
    assert error.location in str(error)


def test_describe_lambda_test_points_to_source():
    query = where('price').test(lambda value: value > 1)
    with pytest.raises(QueryNotSerializableError) as exc_info:
        describe(query)
    file, line = exc_info.value.location.rsplit(':', 1)
    assert file == __file__
    assert int(line) > 0


def test_describe_map_query_not_serializable():
    query = where('price').map(str) == '1'
    with pytest.raises(QueryNotSerializableError):
        describe(query)


def test_describe_custom_query_instance_not_serializable():
    query = QueryInstance(lambda doc: True, None)
    with pytest.raises(QueryNotSerializableError):
        describe(query)


def test_describe_regex_not_supported():
    with pytest.raises(UnsupportedQueryError):
        describe(where('category').matches(r'^b'))
    with pytest.raises(UnsupportedQueryError):
        describe(where('category').search(r'b'))


def test_describe_any_with_nested_query_not_supported():
    with pytest.raises(UnsupportedQueryError):
        describe(where('items').any(where('x') == 1))


def test_describe_fragment_not_supported():
    with pytest.raises(UnsupportedQueryError):
        describe(Query().fragment({'a': 1}))


def test_describe_ordering_comparison_with_none_rejected():
    with pytest.raises(UnsupportedQueryError):
        describe(where('p') < None)


# --- compile / round-trip ----------------------------------------------------

ROUNDTRIP_QUERIES = [
    where('category') == 'book',
    where('price') != 2,
    where('price') < 3,
    where('price') <= 3,
    where('price') > 1,
    where('price') >= 1,
    where('price') == None,  # noqa: E711
    where('tags').any(['sale', 'new']),
    where('tags').all(['sale']),
    where('tags').exists(),
    where('category').one_of(['book', 'toy']),
    where('stock').warehouse == 'east',
    where('a.b') == 'x',
    (where('category') == 'book') & (where('price') <= 2),
    (where('category') == 'book') | (where('price') <= 2),
    ~(where('tags').any(['sale'])),
    ((where('category') == 'book') & (where('price') <= 2))
    | ~(where('tags').any(['sale'])),
    Query().noop(),
]


@pytest.mark.parametrize('query', ROUNDTRIP_QUERIES)
def test_describe_compile_roundtrip(query):
    description = describe(query)
    assert describe(compile_query(description)) == description


def _safe_call(query, document):
    # TinyDB propagates TypeError (e.g. ordering comparisons against None)
    # depending on operand evaluation order; for selection purposes such
    # documents count as not matched.
    try:
        return bool(query(document))
    except TypeError:
        return False


@pytest.mark.parametrize('query', ROUNDTRIP_QUERIES)
def test_compiled_query_matches_original(query):
    documents = [
        {},
        {'category': 'book', 'price': 1, 'tags': ['sale']},
        {'category': 'toy', 'price': None, 'tags': []},
        {'category': 'food', 'price': 3, 'tags': ['new', 'sale'],
         'stock': {'warehouse': 'east', 'qty': 1}, 'a.b': 'x'},
        {'stock': {'warehouse': 'west'}},
    ]
    compiled = compile_query(describe(query))
    for document in documents:
        assert _safe_call(compiled, document) == _safe_call(query, document)


def test_description_is_json_serializable():
    description = describe((where('category') == 'book')
                           & where('tags').any(['sale']))
    payload = description_to_json(description)
    assert description_from_json(payload) == description
    assert json.loads(payload) == description


def test_validate_description_rejects_bad_input():
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 99, 'root': {}})
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 1})
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 1, 'root': {'op': 'bogus'}})
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 1, 'root': {
            'op': 'cmp', 'path': [], 'cmp': '==', 'value': 1}})
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 1, 'root': {
            'op': 'cmp', 'path': ['p'], 'cmp': '<', 'value': None}})
    with pytest.raises(QueryDescriptionError):
        validate_description({'version': 1, 'root': {
            'op': 'any', 'path': ['p'], 'values': [{'not': 'scalar'}]}})


# --- Catalog contract --------------------------------------------------------


def test_catalog_accepts_legal_documents():
    catalog = make_catalog()
    assert catalog.is_legal({'category': 'book', 'price': 1})
    assert catalog.is_legal({'category': 'toy', 'price': None,
                             'tags': [], 'stock': {'warehouse': 'east'},
                             'a.b': 'y'})
    assert catalog.is_legal({'category': 'food', 'price': 3,
                             'tags': ['new', 'sale'],
                             'stock': {'warehouse': 'west', 'qty': 0}})


def test_catalog_rejects_missing_required_field():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'price': 1})
    assert not catalog.is_legal({'price': 1})


def test_catalog_rejects_undeclared_field():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'book', 'price': 1, 'extra': 1})


def test_catalog_rejects_null_when_not_nullable():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': None, 'price': 1})


def test_catalog_rejects_value_outside_domain():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'music', 'price': 1})


def test_catalog_enforces_list_bounds():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'book', 'price': 1,
                          'tags': ['new', 'sale', 'new']})
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'book', 'price': 1,
                          'tags': ['bogus']})


def test_catalog_enforces_empty_list_rule():
    catalog = Catalog({
        'tags': ListField(['sale'], max_length=1, allow_empty=False),
    })
    assert catalog.is_legal({'tags': ['sale']})
    assert not catalog.is_legal({'tags': []})


def test_catalog_validates_nested_objects():
    catalog = make_catalog()
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'book', 'price': 1,
                          'stock': {'warehouse': 'north'}})
    with pytest.raises(ContractViolationError):
        catalog.validate({'category': 'book', 'price': 1,
                          'stock': {'warehouse': 'east', 'qty': 5}})


def test_catalog_literal_dot_key_is_not_a_path():
    catalog = make_catalog()
    assert catalog.is_legal({'category': 'book', 'price': 1, 'a.b': 'x'})
    # a literal dot key does not make nested documents legal
    assert not catalog.is_legal({'category': 'book', 'price': 1,
                                 'a': {'b': 'x'}})


def test_field_contract_validation():
    with pytest.raises(ValueError):
        ScalarField([])
    with pytest.raises(ValueError):
        ScalarField([None])
    with pytest.raises(ValueError):
        ScalarField([object()])
    with pytest.raises(ValueError):
        ListField(['a'], max_length=-1)
    with pytest.raises(ValueError):
        ListField([], max_length=2, allow_empty=False)
    # the same field is fine once it may be missing
    ListField([], max_length=2, allow_empty=False, optional=True)
    ScalarField([], optional=True, nullable=True)


# --- Equivalence checking ----------------------------------------------------


def test_equivalent_rules_are_reusable():
    catalog = make_catalog()
    old = describe((where('category') == 'book') & (where('price') <= 2))
    new = describe((where('price') <= 2) & (where('category') == 'book'))
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.EQUIVALENT
    assert result.equivalent
    assert result.reusable
    assert result.witness is None


def test_equivalent_rules_de_morgan():
    catalog = make_catalog()
    old = describe(~((where('category') == 'book')
                     | (where('price') <= 2)))
    new = describe(~(where('category') == 'book')
                   & ~(where('price') <= 2))
    assert compare_descriptions(old, new, catalog).equivalent


def test_different_rules_produce_minimal_legal_witness():
    catalog = make_catalog()
    old = describe(where('category') == 'book')
    new = describe(where('category') == 'toy')
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert not result.reusable
    witness = result.witness
    catalog.validate(witness)
    # minimal: only the required fields, no optional extras
    assert set(witness) == {'category', 'price'}
    assert witness['category'] in ('book', 'toy')
    old_query = compile_query(old)
    new_query = compile_query(new)
    assert old_query(witness) != new_query(witness)


def test_witness_respects_missing_field_semantics():
    catalog = Catalog({
        'category': ScalarField(['book', 'toy']),
        'price': ScalarField([1, 2], optional=True, nullable=True),
    })
    # != is False on missing fields, while ~(==) is True
    old = describe(~(where('price') == 1))
    new = describe(where('price') != 1)
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    witness = result.witness
    catalog.validate(witness)
    assert 'price' not in witness
    assert compile_query(old)(witness) != compile_query(new)(witness)


def test_witness_respects_nullable_semantics():
    catalog = Catalog({
        'category': ScalarField(['book', 'toy']),
        'price': ScalarField([1, 2], nullable=True),
    })
    old = describe(where('price') == None)  # noqa: E711
    new = describe(~where('price').exists())
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert result.witness['price'] is None
    catalog.validate(result.witness)


def test_witness_uses_empty_list_when_legal():
    catalog = Catalog({
        'sku': ScalarField(['a', 'b']),
        'tags': ListField(['sale'], max_length=1, optional=True),
    })
    old = describe(where('tags').all(['sale']))
    new = describe(where('tags').exists())
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert result.witness['tags'] == []
    catalog.validate(result.witness)


def test_empty_list_can_make_rules_equivalent():
    catalog = Catalog({
        'sku': ScalarField(['a', 'b']),
        'tags': ListField(['sale'], max_length=1, optional=True,
                          allow_empty=False),
    })
    old = describe(where('tags').all(['sale']))
    new = describe(where('tags').exists())
    assert compare_descriptions(old, new, catalog).equivalent


def test_any_all_singleton_equivalence():
    catalog = make_catalog()
    old = describe(where('tags').any(['sale']))
    new = describe(where('tags').all(['sale']))
    assert compare_descriptions(old, new, catalog).equivalent


def test_any_witness_prefers_shortest_list():
    catalog = make_catalog()
    old = describe(where('tags').any(['sale']))
    new = describe(where('tags').any(['new']))
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert len(result.witness['tags']) == 1
    catalog.validate(result.witness)


def test_nested_field_rules():
    catalog = make_catalog()
    old = describe(where('stock').warehouse == 'east')
    new = describe(where('stock').exists())
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert result.witness['stock'] == {'warehouse': 'west'}
    catalog.validate(result.witness)


def test_literal_dot_key_rules():
    catalog = make_catalog()
    old = describe(where('a.b') == 'x')
    new = describe(where('a.b') == 'y')
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.DIFFERENT
    assert result.witness['a.b'] in ('x', 'y')
    catalog.validate(result.witness)
    # equivalent rules over the literal dot key are decided too
    same = describe(where('a.b') == 'x')
    assert compare_descriptions(old, same, catalog).equivalent


def test_rules_over_field_subset_are_decided_not_unknown():
    catalog = make_catalog()
    # the rules only touch 'category' although the catalog declares
    # nested objects, lists and literal dot keys as well
    old = describe(where('category') == 'book')
    new = describe(where('category').one_of(['book']))
    result = compare_descriptions(old, new, catalog)
    assert result.status == RuleComparison.EQUIVALENT

    different = compare_descriptions(
        old, describe(where('category') == 'toy'), catalog)
    assert different.status == RuleComparison.DIFFERENT


def test_undeclared_field_is_unknown():
    catalog = make_catalog()
    old = describe(where('unknown_field') == 1)
    result = compare_descriptions(old, old, catalog)
    assert result.status == RuleComparison.UNKNOWN
    assert not result.equivalent
    assert not result.reusable
    assert 'unknown_field' in result.reason


def test_oversized_document_space_is_unknown():
    catalog = make_catalog()
    old = describe((where('category') == 'book') & (where('price') == 1))
    new = describe((where('category') == 'book') & (where('price') == 2))
    result = compare_descriptions(old, new, catalog, max_documents=5)
    assert result.status == RuleComparison.UNKNOWN
    assert 'document space' in result.reason


def test_check_upgrade_reuse_and_witness():
    catalog = make_catalog()
    old = describe(where('category') == 'book')
    equivalent_new = describe(where('category').one_of(['book']))
    changed_new = describe(where('category') == 'toy')

    reused = check_upgrade(old, equivalent_new, catalog)
    assert reused.reusable

    changed = check_upgrade(old, changed_new, catalog)
    assert not changed.reusable
    catalog.validate(changed.witness)
    assert (compile_query(old)(changed.witness)
            != compile_query(changed_new)(changed.witness))


def test_witness_is_minimal_among_legal_documents():
    catalog = make_catalog()
    old = describe(where('tags').any(['sale']))
    new = describe(where('tags').any(['new']))
    result = compare_descriptions(old, new, catalog)
    witness = result.witness
    # no legal document with fewer keys can distinguish the rules
    smaller = [{'category': category, 'price': price}
               for category in ('book', 'toy', 'food')
               for price in (None, 1, 2, 3)]
    old_query = compile_query(old)
    new_query = compile_query(new)
    for document in smaller:
        assert old_query(document) == new_query(document)
    assert witness is not None
