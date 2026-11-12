# -*- coding: utf-8 -*-
"""Computable idempotence regression for normalization.

The normalization pipeline of a mapping applies, in order,
``rename`` -> ``purge_unknown`` -> ``purge_readonly`` -> ``readonly`` ->
``default``/``default_setter`` -> ``coerce`` -> container normalization.
This module enumerates a finite column contract with a finite value domain and
checks whether a second normalization pass ever changes business values, then
reduces every failure to the minimal set of *native* rules that jointly cause
it.

Native rules that may carry a Python callable (``coerce``, ``rename_handler``,
``default_setter``) are not enumerable: a callable's output domain is unbounded.
The exhaustive search therefore only uses static ``default`` values, static
``rename`` targets and the boolean ``purge_unknown`` flag, while dedicated
tests below demonstrate why callables stay outside the search.
"""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

from pytest import mark

from cerberus import Validator

# --- Finite column contract -------------------------------------------------
# Three declared columns.  'a' and 'b' draw from a small integer domain, 'c'
# from a small string domain with an `allowed` constraint.  'ghost' is not
# declared, so it only interacts with `purge_unknown`.
ABSENT = object()
INTEGER_DOMAIN = (0, 7)
STRING_DOMAIN = ('s0', 's1')
# Each column has an explicit absent state, and declared columns additionally
# accept None (they are non-nullable, which interacts with `default`).
FIELD_STATES = {
    'a': (ABSENT, None) + INTEGER_DOMAIN,
    'b': (ABSENT, None) + INTEGER_DOMAIN,
    'c': (ABSENT, None) + STRING_DOMAIN,
    'ghost': (ABSENT, 0, 's0', None),
}
STATIC_DEFAULTS = {'a': 0, 'b': 7, 'c': 's0'}
RENAME_TARGETS = {'a': 'b', 'b': 'c'}

# One search atom per native rule instance plus the document-level flag.
RULE_ATOMS = (
    'rename_a',
    'rename_b',
    'default_a',
    'default_b',
    'default_c',
    'purge_unknown',
)

ILLEGAL = 'illegal'  # record does not validate under this rule configuration
STABLE = 'stable'  # legal, and a second pass changes nothing
FAILING = 'failing'  # legal, and the second pass changes business values


def build_schema(mask):
    schema = {
        'a': {'type': 'integer'},
        'b': {'type': 'integer'},
        'c': {'type': 'string', 'allowed': list(STRING_DOMAIN)},
    }
    for field in ('a', 'b'):
        atom = 'rename_' + field
        if atom in mask:
            schema[field]['rename'] = RENAME_TARGETS[field]
    for field in ('a', 'b', 'c'):
        atom = 'default_' + field
        if atom in mask:
            schema[field]['default'] = STATIC_DEFAULTS[field]
    return schema


def all_masks():
    return [
        frozenset(atom for atom, enabled in zip(RULE_ATOMS, bits) if enabled)
        for bits in product((False, True), repeat=len(RULE_ATOMS))
    ]


def all_documents():
    """Every partial mapping over the finite domains, including absent keys."""
    documents = []
    for values in product(*FIELD_STATES.values()):
        document = {}
        for field, value in zip(FIELD_STATES, values):
            if value is not ABSENT:
                document[field] = value
        documents.append(document)
    return documents


def observe(mask, document):
    validator = Validator(build_schema(mask), purge_unknown='purge_unknown' in mask)
    if not validator.validate(deepcopy(document)):
        return ILLEGAL
    first = validator.normalized(deepcopy(document), always_return_document=True)
    second = validator.normalized(deepcopy(first), always_return_document=True)
    return FAILING if first != second else STABLE


def passes_to_fixed_point(mask, document):
    validator = Validator(build_schema(mask), purge_unknown='purge_unknown' in mask)
    current = deepcopy(document)
    passes = 0
    while True:
        normalized = validator.normalized(
            deepcopy(current), always_return_document=True
        )
        passes += 1
        if normalized == current:
            return passes
        current = normalized


def kind_signature(mask):
    return frozenset(
        'purge_unknown' if atom == 'purge_unknown' else atom.split('_')[0]
        for atom in mask
    )


def exhaust():
    masks = all_masks()
    documents = all_documents()
    grid = {}
    for doc_index, document in enumerate(documents):
        for mask in masks:
            grid[(doc_index, mask)] = observe(mask, document)

    witnesses = [
        doc_index
        for doc_index in range(len(documents))
        if any(grid[(doc_index, mask)] == FAILING for mask in masks)
    ]

    signatures = set()
    two_atom_minima = set()
    minimal_pairs = 0
    driver_removals = 0
    precondition_removals = 0
    depth_distribution = {}

    for doc_index in witnesses:
        failing_masks = {
            mask for mask in masks if grid[(doc_index, mask)] == FAILING
        }
        minimal_masks = {
            mask
            for mask in failing_masks
            if not any(smaller < mask for smaller in failing_masks)
        }
        minimal_pairs += len(minimal_masks)
        for mask in minimal_masks:
            signatures.add(kind_signature(mask))
            if len(mask) == 2:
                two_atom_minima.add(mask)
            for atom in mask:
                if grid[(doc_index, mask - {atom})] == ILLEGAL:
                    # Removing the rule invalidates the record; the mechanical
                    # cause is still present, it merely lost its legal witness.
                    precondition_removals += 1
                else:
                    driver_removals += 1
        for mask in failing_masks:
            depth = passes_to_fixed_point(mask, documents[doc_index])
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1

    return SimpleNamespace(
        masks=masks,
        documents=documents,
        grid=grid,
        witnesses=witnesses,
        signatures=signatures,
        two_atom_minima=two_atom_minima,
        minimal_pairs=minimal_pairs,
        driver_removals=driver_removals,
        precondition_removals=precondition_removals,
        depth_distribution=depth_distribution,
    )


_EXHAUSTION_CACHE = None


def exhaust_cached():
    global _EXHAUSTION_CACHE
    if _EXHAUSTION_CACHE is None:
        _EXHAUSTION_CACHE = exhaust()
    return _EXHAUSTION_CACHE


def test_grid_covers_the_whole_finite_contract():
    result = exhaust_cached()
    assert len(result.documents) == 256
    assert len(result.masks) == 2 ** len(RULE_ATOMS) == 64
    assert len(result.grid) == 256 * 64
    # Every observation is one of the three exhaustive outcomes.
    assert set(result.grid.values()) <= {ILLEGAL, STABLE, FAILING}
    assert len(result.witnesses) == 208


def test_minimal_cause_signatures_are_complete():
    result = exhaust_cached()
    assert set(map(frozenset, result.signatures)) == {
        frozenset(('rename',)),
        frozenset(('rename', 'default')),
        frozenset(('rename', 'purge_unknown')),
        frozenset(('rename', 'default', 'purge_unknown')),
    }
    # `rename` is part of every minimal cause; `default` and `purge_unknown`
    # only appear as interacting companions, never as a standalone cause.
    assert all('rename' in signature for signature in result.signatures)
    assert result.minimal_pairs == 464


def test_three_two_rule_minimal_causes_exist_with_legal_records():
    result = exhaust_cached()
    assert set(map(frozenset, result.two_atom_minima)) == {
        frozenset(('rename_a', 'rename_b')),
        frozenset(('rename_a', 'default_a')),
        frozenset(('rename_b', 'default_b')),
    }
    # Each two-rule cause has at least one legal starting record.
    for mask in result.two_atom_minima:
        assert any(
            result.grid[(doc_index, mask)] == FAILING
            for doc_index in result.witnesses
        )


def test_rename_chain_legal_witness_changes_value_on_second_pass():
    schema = {
        'a': {'type': 'integer', 'rename': 'b'},
        'b': {'type': 'integer', 'rename': 'c'},
        'c': {'type': 'string', 'allowed': list(STRING_DOMAIN)},
    }
    record = {'a': 0}
    validator = Validator(schema)
    assert validator.validate(deepcopy(record))
    first = validator.normalized(deepcopy(record), always_return_document=True)
    second = validator.normalized(deepcopy(first), always_return_document=True)
    third = validator.normalized(deepcopy(second), always_return_document=True)
    assert first == {'b': 0}
    assert second == {'c': 0}  # second pass moves the business value again
    assert third == second


def test_same_field_rename_and_default_refills_and_overwrites():
    schema = {
        'a': {'type': 'integer', 'rename': 'b', 'default': 0},
        'b': {'type': 'integer'},
        'c': {'type': 'string', 'allowed': list(STRING_DOMAIN)},
    }
    record = {}
    validator = Validator(schema)
    assert validator.validate(deepcopy(record))
    first = validator.normalized(deepcopy(record), always_return_document=True)
    second = validator.normalized(deepcopy(first), always_return_document=True)
    third = validator.normalized(deepcopy(second), always_return_document=True)
    assert first == {'a': 0}
    assert second == {'b': 0, 'a': 0}  # default refills, rename overwrites
    assert third == second


def test_rename_with_target_default_refills_on_second_pass():
    schema = {
        'a': {'type': 'integer'},
        'b': {'type': 'integer', 'rename': 'c', 'default': 7},
        'c': {'type': 'string', 'allowed': list(STRING_DOMAIN)},
    }
    record = {}
    validator = Validator(schema)
    assert validator.validate(deepcopy(record))
    first = validator.normalized(deepcopy(record), always_return_document=True)
    second = validator.normalized(deepcopy(first), always_return_document=True)
    assert first == {'b': 7}
    assert second == {'c': 7, 'b': 7}


def test_rule_removal_is_classified_as_driver_or_legality_precondition():
    # A removed rule either stabilizes the legal record (it was a driver) or
    # makes the record illegal (it was only a legality precondition).  An
    # illegal outcome must not be misread as the cause having disappeared.
    result = exhaust_cached()
    for doc_index in result.witnesses:
        failing_masks = {
            mask
            for mask in result.masks
            if result.grid[(doc_index, mask)] == FAILING
        }
        minimal_masks = {
            mask
            for mask in failing_masks
            if not any(smaller < mask for smaller in failing_masks)
        }
        for mask in minimal_masks:
            for atom in mask:
                assert result.grid[(doc_index, mask - {atom})] in (
                    ILLEGAL,
                    STABLE,
                )
    assert result.driver_removals == 864
    assert result.precondition_removals == 900


def test_coerce_only_serves_as_legality_precondition_in_a_failing_case():
    schema = {
        'a': {'type': 'integer', 'coerce': int},
        'b': {'type': 'integer', 'rename': 'c', 'default': 7},
        'c': {'type': 'string', 'allowed': list(STRING_DOMAIN)},
    }
    record = {'a': '0', 'ghost': 0}
    validator = Validator(schema, purge_unknown=True)
    assert validator.validate(deepcopy(record))
    first = validator.normalized(deepcopy(record), always_return_document=True)
    second = validator.normalized(deepcopy(first), always_return_document=True)
    assert first == {'a': 0, 'b': 7}
    assert second == {'a': 0, 'c': 7, 'b': 7}

    # Without the coerce rule the same second-pass movement still happens;
    # the record merely stops being legal, so the cause has not vanished.
    weakened = deepcopy(schema)
    del weakened['a']['coerce']
    weakened_validator = Validator(weakened, purge_unknown=True)
    assert not weakened_validator.validate(deepcopy(record))
    weak_first = weakened_validator.normalized(
        deepcopy(record), always_return_document=True
    )
    weak_second = weakened_validator.normalized(
        deepcopy(weak_first), always_return_document=True
    )
    assert weak_first != weak_second


def test_every_failing_configuration_reaches_a_fixed_point_within_four_passes():
    result = exhaust_cached()
    assert result.depth_distribution == {3: 1420, 4: 255}
    assert max(result.depth_distribution) == 4


def test_purge_unknown_signature_requires_an_unknown_key_witness():
    # The purge-interaction signature can only fire on records that actually
    # carry an unknown key, which purge_unknown removes on the first pass.
    result = exhaust_cached()
    purge_masks = [
        mask
        for mask in result.masks
        if 'purge_unknown' in mask
        and any(
            result.grid[(doc_index, mask)] == FAILING
            for doc_index in result.witnesses
        )
    ]
    for mask in purge_masks:
        assert any(
            'ghost' in result.documents[doc_index]
            for doc_index in result.witnesses
            if result.grid[(doc_index, mask)] == FAILING
        )


@mark.parametrize('rule', ('coerce', 'rename_handler', 'default_setter'))
def test_callback_capable_rules_are_excluded_from_enumeration(rule):
    options = Validator.normalization_rules[rule]['oneof']
    assert any(option.get('type') == 'callable' for option in options)


def test_stateful_callable_breaks_the_finite_domain_guarantee():
    # A Python callback can depend on hidden state, so no finite input domain
    # bounds its outputs: the same normalized document changes again on the
    # next pass even though every static rule would already be idempotent.
    def stateful_coerce(value):
        stateful_coerce.calls += 1
        return value + stateful_coerce.calls

    stateful_coerce.calls = 0
    validator = Validator({'x': {'coerce': stateful_coerce}})
    first = validator.normalized({'x': 0})
    second = validator.normalized(first)
    assert first == {'x': 1}
    assert second == {'x': 3}
