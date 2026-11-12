"""Export a Voluptuous schema to an equivalent JSON Schema contract.

This module turns a :class:`~voluptuous.Schema` into a JSON Schema document
plus a machine-readable report, so a consumer that only understands JSON
Schema can validate the same payloads.

Scope of the export (acceptance-equivalent on both sides):

* finite JSON types: ``str``, ``bool``, ``None``/``type(None)`` exactly;
  ``int``/``float`` and numeric/bool literals are exported as
  ``integer``/``number``/``const`` with recorded divergences, because JSON
  Schema's numeric tower (``1.0`` is an ``integer``) and Python's
  ``isinstance(True, int)`` cannot be reconciled;
* fixed (literal) and pattern object keys, ``Required``/``Optional`` and the
  ``extra`` modes (``PREVENT_EXTRA``/``ALLOW_EXTRA``/``REMOVE_EXTRA``);
* list element schemas (``[s]`` -> ``items``, several schemas -> ``anyOf``);
* conversion-free ``Any``/``All`` combinations -> ``anyOf``/``allOf``.

Overlapping rules keep the source-side matching semantics: literal keys win
over pattern keys (``properties`` before ``patternProperties``), the first
``str``/``Extra`` wildcard shadows later key rules, and a single ``Match``
key is exported with ``re.match`` anchoring.  Where equivalence is impossible
(e.g. two overlapping ``Match`` key patterns, ``Exclusive``/``Inclusive``
groups) the rule is exported best-effort and a *witness* JSON value on which
the two sides disagree is attached to the report.

Any Python callable (functions, lambdas, ``Coerce``, ``Range``, ``Match`` as
a value, ...) is marked as *unsupported* statically, before it could ever be
executed: the exporter never calls user code.  Unsupported sibling columns
are exported independently (their constraint becomes ``true``) and listed in
``ExportResult.unsupported``; ``ExportResult.covered`` states exactly which
columns are fully covered, so a partial contract is never mistaken for a
complete one.

Divergence witnesses are verified at export time against the exported JSON
Schema with the bundled mini validator (:func:`json_schema_accepts`); the
Voluptuous-side verdict of a witness is known by construction.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import typing
from collections.abc import Mapping

from voluptuous.schema_builder import (
    PREVENT_EXTRA,
    REMOVE_EXTRA,
    Extra,
    Exclusive,
    Inclusive,
    Marker,
    Msg,
    Object,
    Optional,
    Remove,
    Required,
    Schema,
    Self,
    Undefined,
    _sort_item,
)
from voluptuous.validators import All, Any, Match, Union, _WithSubValidators

__all__ = [
    'to_json_schema',
    'json_schema_accepts',
    'ExportResult',
    'Divergence',
    'Unsupported',
    'Witness',
]


@dataclasses.dataclass
class Witness:
    """A JSON value on which the two sides reach different verdicts."""

    document: typing.Any
    voluptuous_accepts: bool
    json_schema_accepts: bool


@dataclasses.dataclass
class Divergence:
    """A rule exported best-effort whose acceptance sets differ."""

    path: str
    rule: str
    exported: typing.Any
    reason: str
    witnesses: typing.List[Witness]


@dataclasses.dataclass
class Unsupported:
    """A rule that cannot be exported at all (never executed)."""

    path: str
    rule: str
    reason: str


@dataclasses.dataclass
class ExportResult:
    """The exported JSON Schema plus the coverage report."""

    schema: typing.Any
    covered: typing.List[str]
    divergences: typing.List[Divergence]
    unsupported: typing.List[Unsupported]
    notes: typing.List[str]

    @property
    def complete(self) -> bool:
        """True only if every rule was exported with exact equivalence."""
        return not self.divergences and not self.unsupported

    def summary(self) -> str:
        lines = []
        if self.complete:
            lines.append('Export COMPLETE: every rule is exactly equivalent.')
        else:
            lines.append(
                'Export PARTIAL: %d divergence(s), %d unsupported rule(s).'
                % (len(self.divergences), len(self.unsupported))
            )
        lines.append('Covered columns (%d):' % len(self.covered))
        lines.extend('  %s' % c for c in self.covered)
        for d in self.divergences:
            lines.append('DIVERGENCE at %s [%s]: %s' % (d.path, d.rule, d.reason))
            for w in d.witnesses:
                lines.append(
                    '  witness %r: voluptuous %s / json-schema %s'
                    % (
                        w.document,
                        'accepts' if w.voluptuous_accepts else 'rejects',
                        'accepts' if w.json_schema_accepts else 'rejects',
                    )
                )
        for u in self.unsupported:
            lines.append('UNSUPPORTED at %s [%s]: %s' % (u.path, u.rule, u.reason))
        for n in self.notes:
            lines.append('note: %s' % n)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Mini JSON Schema validator (subset emitted by this module).
# Used to self-verify divergence witnesses and by the tests.
# ---------------------------------------------------------------------------


def _const_equal(const, value) -> bool:
    if isinstance(const, bool) or isinstance(value, bool):
        return const is value
    if isinstance(const, (int, float)) and isinstance(value, (int, float)):
        return const == value
    if type(const) is not type(value):
        return False
    return const == value


def _type_ok(type_name, value) -> bool:
    if isinstance(type_name, list):
        return any(_type_ok(t, value) for t in type_name)
    if type_name == 'null':
        return value is None
    if type_name == 'boolean':
        return isinstance(value, bool)
    if type_name == 'string':
        return isinstance(value, str)
    if type_name == 'integer':
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and value.is_integer()
    if type_name == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == 'object':
        return isinstance(value, dict)
    if type_name == 'array':
        return isinstance(value, list)
    raise ValueError('unknown JSON Schema type %r' % (type_name,))


def json_schema_accepts(schema, value) -> bool:
    """Validate ``value`` against the JSON Schema subset this module emits."""
    if schema is True:
        return True
    if schema is False:
        return False
    if not schema:
        return True
    if 'allOf' in schema:
        if not all(json_schema_accepts(s, value) for s in schema['allOf']):
            return False
    if 'anyOf' in schema:
        if not any(json_schema_accepts(s, value) for s in schema['anyOf']):
            return False
    if 'const' in schema:
        if not _const_equal(schema['const'], value):
            return False
    if 'type' in schema:
        if not _type_ok(schema['type'], value):
            return False
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                return False
        properties = schema.get('properties', {})
        patterns = schema.get('patternProperties', {})
        for key, item in value.items():
            matched = False
            if key in properties:
                matched = True
                if not json_schema_accepts(properties[key], item):
                    return False
            for pattern, subschema in patterns.items():
                if re.search(pattern, key):
                    matched = True
                    if not json_schema_accepts(subschema, item):
                        return False
            if not matched:
                additional = schema.get('additionalProperties', True)
                if additional is False:
                    return False
                if isinstance(additional, dict) and not json_schema_accepts(
                    additional, item
                ):
                    return False
    if isinstance(value, list):
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            return False
        items = schema.get('items')
        if items is not None:
            for element in value:
                if not json_schema_accepts(items, element):
                    return False
    return True


def _synthesize(schema):
    """Build a small JSON value accepted by ``schema`` (best effort)."""
    if isinstance(schema, dict):
        if 'const' in schema:
            return schema['const']
        for combinator in ('allOf', 'anyOf'):
            if schema.get(combinator):
                return _synthesize(schema[combinator][0])
        type_name = schema.get('type')
        if type_name == 'string':
            return 'x'
        if type_name == 'integer':
            return 1
        if type_name == 'number':
            return 1.5
        if type_name == 'boolean':
            return True
        if type_name == 'null':
            return None
        if type_name == 'array':
            return []
        if type_name == 'object':
            return {
                key: _synthesize(subschema)
                for key, subschema in schema.get('properties', {}).items()
                if key in schema.get('required', [])
            }
    return None


def _child_path(path: str, key: str) -> str:
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
        return '%s.%s' % (path, key)
    return '%s[%r]' % (path, key)


@dataclasses.dataclass
class _Context:
    required: bool = False
    extra: int = PREVENT_EXTRA


class _Exporter:
    def __init__(self) -> None:
        self.covered: typing.List[str] = []
        self.divergences: typing.List[Divergence] = []
        self.unsupported: typing.List[Unsupported] = []
        self.notes: typing.List[str] = []

    # -- reporting helpers -------------------------------------------------

    def _problems(self) -> int:
        return len(self.divergences) + len(self.unsupported)

    def _add_unsupported(self, path: str, rule, reason: str) -> None:
        self.unsupported.append(Unsupported(path, repr(rule), reason))

    def _add_divergence(
        self, path: str, rule, exported, reason: str, witnesses
    ) -> None:
        checked = []
        for document, vol_accepts, js_accepts in witnesses:
            if json_schema_accepts(exported, document) != js_accepts:
                # Witness failed self-verification; drop it.
                continue
            checked.append(Witness(document, vol_accepts, js_accepts))
        if checked:
            self.divergences.append(
                Divergence(path, repr(rule), exported, reason, checked)
            )
        else:
            self.unsupported.append(
                Unsupported(
                    path,
                    repr(rule),
                    reason + ' (no verifiable witness could be built)',
                )
            )

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    # -- main dispatch -----------------------------------------------------

    def export(self, node, path: str, ctx: _Context):
        if isinstance(node, Schema):
            return self.export(
                node.schema, path, _Context(required=node.required, extra=node.extra)
            )
        if node is Self:
            self._add_unsupported(path, node, 'recursive Self schema')
            return True
        if node is Extra:
            return True
        if isinstance(node, Msg):
            self._note(
                '%s: Msg() wrapper unwrapped; only the message is lost' % path
            )
            return self.export(node._schema, path, ctx)
        if isinstance(node, Object):
            self._add_unsupported(path, node, 'Object schemas have no JSON form')
            return True
        if isinstance(node, Mapping):
            return self._export_dict(node, path, ctx)
        if isinstance(node, list):
            return self._export_list(node, path, ctx)
        if isinstance(node, (tuple, frozenset, set)):
            self._add_unsupported(
                path, node, 'tuple/set schemas have no JSON counterpart'
            )
            return True
        if isinstance(node, enum.Enum):
            self._add_unsupported(path, node, 'enum members are not JSON values')
            return True

        # Pure type checks.
        if node is str:
            return {'type': 'string'}
        if node is bool:
            return {'type': 'boolean'}
        if node is type(None):
            return {'type': 'null'}
        if node is object:
            return True
        if node is int:
            exported = {'type': 'integer'}
            self._add_divergence(
                path,
                node,
                exported,
                'JSON Schema "integer" also accepts 1.0 and rejects true, '
                'while voluptuous isinstance(v, int) does the opposite',
                [
                    (True, True, False),
                    (1.0, False, True),
                ],
            )
            return exported
        if node is float:
            exported = {'type': 'number'}
            self._add_divergence(
                path,
                node,
                exported,
                'JSON Schema "number" accepts integers, '
                'while voluptuous isinstance(v, float) rejects them',
                [(1, False, True)],
            )
            return exported

        # Conversion-free combinators.
        if isinstance(node, (All, Any, Union)):
            return self._export_combinator(node, path, ctx)
        if isinstance(node, _WithSubValidators):
            self._add_unsupported(
                path, node, '%s is not exportable' % type(node).__name__
            )
            return True

        # Literals.
        if node is None:
            return {'const': None}
        if isinstance(node, bool):
            exported = {'const': node}
            crossover = 1 if node else 0
            self._add_divergence(
                path,
                node,
                exported,
                'voluptuous compares literals with ==, so %r also matches %r'
                % (node, crossover),
                [(crossover, True, False)],
            )
            return exported
        if isinstance(node, str):
            return {'const': node}
        if isinstance(node, (int, float)):
            exported = {'const': node}
            witnesses = []
            if node in (0, 1):
                crossover = bool(node)
                witnesses.append((crossover, True, False))
            if witnesses:
                self._add_divergence(
                    path,
                    node,
                    exported,
                    'voluptuous compares literals with ==, so %r also matches '
                    'the boolean %r' % (node, bool(node)),
                    witnesses,
                )
            return exported
        if isinstance(node, (bytes, complex)):
            self._add_unsupported(path, node, 'not a JSON value')
            return True

        # Anything callable (functions, lambdas, Coerce, Range, Match as a
        # value, arbitrary classes) is refused before it can be executed.
        if callable(node):
            self._add_unsupported(
                path,
                node,
                'Python callable; marked unsupported before any execution',
            )
            return True

        self._add_unsupported(path, node, 'unsupported schema node')
        return True

    # -- combinators ---------------------------------------------------------

    def _export_combinator(self, node, path: str, ctx: _Context):
        if node.discriminant is not None:
            self._add_unsupported(
                path, node, 'Union/Any/All with a discriminant callable'
            )
            return True
        member_ctx = _Context(required=node.required, extra=ctx.extra)
        if isinstance(node, All):
            for member in node.validators:
                if self._has_side_effects(member, member_ctx):
                    self._add_unsupported(
                        path,
                        node,
                        'All() member removes keys or fills defaults; the '
                        'chained semantics are not expressible as allOf',
                    )
                    return True
        before = len(self.unsupported)
        parts = [
            self.export(member, path, member_ctx) for member in node.validators
        ]
        if len(self.unsupported) != before:
            self._add_unsupported(
                path,
                node,
                '%s() contains a rule that cannot be exported; the whole '
                'combination is dropped' % type(node).__name__,
            )
            return True
        keyword = 'allOf' if isinstance(node, All) else 'anyOf'
        if not parts:
            return False if keyword == 'anyOf' else True
        if len(parts) == 1:
            return parts[0]
        return {keyword: parts}

    def _has_side_effects(self, node, ctx: _Context) -> bool:
        """Detect key removal / default filling inside an All() member."""
        if isinstance(node, Schema):
            return self._has_side_effects(
                node.schema, _Context(required=node.required, extra=node.extra)
            )
        if isinstance(node, Msg):
            return self._has_side_effects(node._schema, ctx)
        if isinstance(node, Mapping):
            if ctx.extra == REMOVE_EXTRA:
                return True
            for key, value in node.items():
                if isinstance(key, Remove):
                    return True
                if isinstance(key, (Required, Optional)) and not isinstance(
                    key.default, Undefined
                ):
                    return True
                if self._has_side_effects(value, ctx):
                    return True
            return False
        if isinstance(node, list):
            return any(self._has_side_effects(m, ctx) for m in node)
        if isinstance(node, _WithSubValidators):
            return any(self._has_side_effects(v, ctx) for v in node.validators)
        return False

    # -- lists -----------------------------------------------------------------

    def _export_list(self, node, path: str, ctx: _Context):
        if not node:
            return {'type': 'array', 'maxItems': 0}
        before = len(self.unsupported)
        parts = []
        for member in node:
            if isinstance(member, Remove) or member is Remove:
                self._add_unsupported(
                    path, member, 'Remove in a list drops elements'
                )
                continue
            parts.append(self.export(member, path + '[*]', ctx))
        if len(self.unsupported) != before:
            self._add_unsupported(
                path,
                node,
                'list element union contains a rule that cannot be exported; '
                'the items constraint is dropped',
            )
            return {'type': 'array'}
        items = parts[0] if len(parts) == 1 else {'anyOf': parts}
        return {'type': 'array', 'items': items}

    # -- dicts -----------------------------------------------------------------

    def _export_dict(self, node, path: str, ctx: _Context):
        result: typing.Dict[str, typing.Any] = {'type': 'object'}
        properties: typing.Dict[str, typing.Any] = {}
        required_keys: typing.List[str] = []
        patterns: typing.List[typing.Tuple[Match, typing.Any]] = []
        wildcards: typing.List[typing.Tuple[typing.Any, typing.Any]] = []
        exclusive_groups: typing.Dict[str, typing.List[str]] = {}
        inclusive_groups: typing.Dict[str, typing.List[str]] = {}

        # Voluptuous tries candidates in this exact order.
        for skey, svalue in sorted(node.items(), key=_sort_item):
            marker = skey if isinstance(skey, Marker) else None
            base = marker.schema if marker else skey

            if isinstance(skey, Remove):
                self._add_unsupported(
                    path, skey, 'Remove keys prune output; not exportable'
                )
                continue

            if isinstance(base, str):
                if base in properties:
                    self._note(
                        '%s: duplicate key %r shadowed by an earlier rule; '
                        'dropped' % (path, base)
                    )
                    continue
                before = self._problems()
                exported_value = self.export(
                    svalue, _child_path(path, base), ctx
                )
                properties[base] = exported_value
                if self._problems() == before:
                    self.covered.append(_child_path(path, base))
                if isinstance(skey, Required):
                    if isinstance(skey.default, Undefined):
                        required_keys.append(base)
                    else:
                        self._note(
                            '%s: Required(%r, default=...) accepts a missing '
                            'key (the default is filled); exported as optional'
                            % (path, base)
                        )
                elif skey is base and ctx.required:
                    required_keys.append(base)
                elif isinstance(skey, Optional) and not isinstance(
                    skey.default, Undefined
                ):
                    self._note(
                        '%s: Optional(%r, default=...) fills a default; '
                        'acceptance is unaffected' % (path, base)
                    )
                if isinstance(skey, Exclusive):
                    exclusive_groups.setdefault(
                        skey.group_of_exclusion, []
                    ).append(base)
                elif isinstance(skey, Inclusive):
                    inclusive_groups.setdefault(
                        skey.group_of_inclusion, []
                    ).append(base)
                continue

            if skey is Extra or base is str:
                wildcards.append((skey, svalue))
                continue
            if isinstance(base, Match):
                patterns.append((base, svalue))
                continue
            if marker is not None:
                self._add_unsupported(
                    path, skey, 'non-literal key inside a marker'
                )
                continue
            if base is None or isinstance(base, (int, float, bool, bytes)):
                self._note(
                    '%s: key %r can never appear in a JSON document (object '
                    'keys are strings); rule dropped' % (path, base)
                )
                continue
            if isinstance(base, type):
                self._note(
                    '%s: key type %r can never match a JSON object key; '
                    'rule dropped' % (path, base)
                )
                continue
            if callable(base):
                self._add_unsupported(
                    path,
                    skey,
                    'Python callable used as an object key; marked '
                    'unsupported before any execution',
                )
                continue
            self._add_unsupported(path, skey, 'unsupported key rule')

        if properties:
            result['properties'] = properties
        if required_keys:
            result['required'] = sorted(required_keys)

        self._export_key_rules(
            node, path, ctx, result, properties, patterns, wildcards
        )
        self._export_groups(
            path, result, properties, exclusive_groups, inclusive_groups
        )
        return result

    def _export_key_rules(
        self, node, path, ctx, result, properties, patterns, wildcards
    ) -> None:
        """Assemble patternProperties / additionalProperties.

        Voluptuous semantics for a key not matched by a literal rule: the
        key rules are tried in candidate order and the *first* match wins.
        JSON Schema applies *all* matching patternProperties, then
        additionalProperties to the rest.  The two coincide as long as at
        most one Match pattern can match any given key.
        """
        extra_allows = ctx.extra != PREVENT_EXTRA
        if ctx.extra == REMOVE_EXTRA:
            self._note(
                '%s: extra=REMOVE_EXTRA only prunes output; exported as '
                'additionalProperties: true (identical acceptance)' % path
            )

        ordered_rules: typing.List[typing.Tuple[str, typing.Any, typing.Any]]
        ordered_rules = [('match', m, v) for m, v in patterns]
        ordered_rules += [('wildcard', k, v) for k, v in wildcards]

        if not ordered_rules:
            result['additionalProperties'] = extra_allows
            return

        first_kind = ordered_rules[0][0]
        if first_kind == 'wildcard':
            # A str/Extra wildcard matches every JSON key: it shadows all
            # later key rules, exactly like in voluptuous.
            _, key_node, value_node = ordered_rules[0]
            result['additionalProperties'] = self.export(
                value_node, path + '.*', ctx
            )
            for kind, rule, _ in ordered_rules[1:]:
                self._note(
                    '%s: key rule %r is shadowed by the earlier %r wildcard '
                    'for JSON string keys; dropped' % (path, rule, key_node)
                )
            return

        # One or more Match patterns, optionally followed by wildcards.
        trailing_wildcard = None
        shadowed = []
        for kind, rule, value_node in ordered_rules:
            if kind != 'wildcard':
                continue
            if trailing_wildcard is None:
                trailing_wildcard = (rule, value_node)
            else:
                shadowed.append(rule)
        for rule in shadowed:
            self._note(
                '%s: key rule %r is shadowed by the earlier %r wildcard '
                'for JSON string keys; dropped' % (path, rule, trailing_wildcard[0])
            )
        match_rules = [r for r in ordered_rules if r[0] == 'match']

        literal_exclusion = ''
        if properties:
            literals = '|'.join(re.escape(k) for k in properties)
            literal_exclusion = '(?!(?:%s)$)' % literals

        pattern_properties: typing.Dict[str, typing.Any] = {}
        exported_patterns: typing.List[typing.Tuple[str, typing.Any, typing.Any]] = []
        for _, match_node, value_node in match_rules:
            wrapped = '^%s(?:%s)' % (
                literal_exclusion,
                match_node.pattern.pattern,
            )
            exported_value = self.export(value_node, path + '.*', ctx)
            pattern_properties[wrapped] = exported_value
            exported_patterns.append((wrapped, match_node, exported_value))
        result['patternProperties'] = pattern_properties
        self._note(
            '%s: Match key patterns use Python re.match semantics; exported '
            'as anchored ECMA-262 patterns for patternProperties (regex '
            'dialect differences are possible)' % path
        )

        if trailing_wildcard is not None:
            _, value_node = trailing_wildcard
            result['additionalProperties'] = self.export(
                value_node, path + '.*', ctx
            )
        else:
            result['additionalProperties'] = extra_allows

        if len(match_rules) > 1:
            self._report_overlapping_patterns(
                path, node, result, exported_patterns, set(properties)
            )

    def _report_overlapping_patterns(
        self, path, node, result, exported_patterns, literal_keys
    ) -> None:
        """Two or more Match key rules: first-match-wins vs all-must-pass."""
        witness_key = None
        pair = None
        for i in range(len(exported_patterns)):
            for j in range(i + 1, len(exported_patterns)):
                first = exported_patterns[i][1].pattern
                second = exported_patterns[j][1].pattern
                candidate = _find_common_match(first, second, literal_keys)
                if candidate is not None:
                    witness_key = candidate
                    pair = (exported_patterns[i], exported_patterns[j])
                    break
            if pair:
                break
        if pair is None:
            self._add_unsupported(
                path,
                node,
                'multiple Match key rules: voluptuous applies the first '
                'match, JSON Schema all of them; no overlapping key could '
                'be exhibited automatically',
            )
            return
        (first_wrapped, first_node, first_js), (
            second_wrapped,
            second_node,
            second_js,
        ) = pair
        witness_value = _find_diverging_value(first_js, second_js)
        if witness_value is None:
            self._add_unsupported(
                path,
                node,
                'multiple Match key rules overlap on key %r but no value '
                'separating the two value schemas could be built'
                % witness_key,
            )
            return
        document = {witness_key: witness_value}
        # Fill required siblings so the fragment is otherwise acceptable.
        for key in result.get('required', []):
            if key in result.get('properties', {}):
                document.setdefault(
                    key, _synthesize(result['properties'][key])
                )
        self._add_divergence(
            path,
            node,
            result,
            'overlapping Match key rules: voluptuous validates the value '
            'against the first matching pattern only, JSON Schema against '
            'all matching patterns',
            [(document, True, False)],
        )

    def _export_groups(
        self, path, result, properties, exclusive_groups, inclusive_groups
    ) -> None:
        def base_document():
            document = {}
            for key in result.get('required', []):
                if key in properties:
                    document[key] = _synthesize(properties[key])
            return document

        for group, keys in exclusive_groups.items():
            if len(keys) < 2:
                continue
            document = base_document()
            for key in keys[:2]:
                document[key] = _synthesize(properties.get(key, True))
            self._add_divergence(
                path,
                'Exclusive group %r' % group,
                result,
                'Exclusive(%s) forbids keys %s from appearing together; '
                'not expressible as plain properties'
                % (group, ', '.join(sorted(keys))),
                [(document, False, True)],
            )
        for group, keys in inclusive_groups.items():
            if len(keys) < 2:
                continue
            document = base_document()
            document[keys[0]] = _synthesize(properties.get(keys[0], True))
            self._add_divergence(
                path,
                'Inclusive group %r' % group,
                result,
                'Inclusive(%s) requires keys %s to appear together; '
                'not expressible as plain properties'
                % (group, ', '.join(sorted(keys))),
                [(document, False, True)],
            )


def _find_common_match(first: re.Pattern, second: re.Pattern, exclude=frozenset()):
    """Best-effort bounded search for a string matched by both patterns."""
    alphabet = sorted(set(re.findall(r'[a-zA-Z0-9]', first.pattern + second.pattern)))
    alphabet = (alphabet + [c for c in 'abz _-' if c not in alphabet])[:8]
    frontier = ['']
    visited = 0
    for _depth in range(5):
        next_frontier = []
        for stem in frontier:
            for char in alphabet:
                candidate = stem + char
                visited += 1
                if (
                    candidate not in exclude
                    and first.match(candidate)
                    and second.match(candidate)
                ):
                    return candidate
                next_frontier.append(candidate)
        frontier = next_frontier
        if visited > 20000:
            return None
    return None


def _find_diverging_value(first_js, second_js):
    """A value accepted by first_js but rejected by second_js, or None."""
    candidates = [
        _synthesize(first_js),
        0,
        1,
        1.5,
        True,
        None,
        '',
        'x',
        [],
        {},
        [1],
        {'a': 1},
    ]
    for candidate in candidates:
        if json_schema_accepts(first_js, candidate) and not json_schema_accepts(
            second_js, candidate
        ):
            return candidate
    return None


def to_json_schema(
    schema,
    required: typing.Optional[bool] = None,
    extra: typing.Optional[int] = None,
) -> ExportResult:
    """Export ``schema`` (a :class:`Schema` or a raw schema tree) to JSON Schema.

    Returns an :class:`ExportResult` carrying the JSON Schema document, the
    list of fully covered columns, divergences with witness JSON values, and
    the rules marked unsupported (Python callables are never executed).
    """
    if isinstance(schema, Schema):
        ctx = _Context(
            required=schema.required if required is None else required,
            extra=schema.extra if extra is None else extra,
        )
        node = schema.schema
    else:
        ctx = _Context(
            required=bool(required) if required is not None else False,
            extra=PREVENT_EXTRA if extra is None else extra,
        )
        node = schema
    exporter = _Exporter()
    exported = exporter.export(node, '$', ctx)
    return ExportResult(
        schema=exported,
        covered=exporter.covered,
        divergences=exporter.divergences,
        unsupported=exporter.unsupported,
        notes=exporter.notes,
    )
