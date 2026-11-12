"""Column-level regression ledger for voluptuous dict validation.

For every input column the ledger pins down:

* the original path (``data[<key>]``),
* every key/value transform actually executed for that column,
* its final disposition: kept, removed by ``Remove``, dropped as an extra
  key, overwritten by a normalization collision, or rejected.

The ledger is built from a *single* validation run.  Business transforms
are wrapped in :class:`Recorder`, which observes each real call as it
happens; the ledger never feeds a transform an extra value just for
logging.  The replay below mirrors ``Schema._compile_mapping`` (candidate
ordering, ``Remove`` fall-through, extra-key policy, last-writer-wins on
normalized keys) and cross-checks its reconstructed output against the
real one, so any behaviour change in the library fails loudly here.
"""

import inspect
from collections import deque

import voluptuous as v
from voluptuous import error as er
from voluptuous import schema_builder as sb

# dispositions
KEPT = 'kept'
KEPT_AS_EXTRA = 'kept-as-extra'
REMOVED = 'removed-by-Remove'
DROPPED_AS_EXTRA = 'dropped-as-extra'
OVERWRITTEN = 'overwritten-by-collision'
REJECTED = 'rejected'

_NOLIT = object()


class TransformRecord:
    """One observed call of a business transform (key or value position)."""

    __slots__ = ('label', 'kind', 'input', 'ok', 'output')

    def __init__(self, label, kind, input_, ok, output):
        self.label = label
        self.kind = kind
        self.input = input_
        self.ok = ok
        self.output = output  # result value, or the raised exception

    def render(self):
        if self.ok:
            outcome = repr(self.output)
        else:
            outcome = '%s(%s)' % (type(self.output).__name__, self.output)
        return '%s %s: %r -> %s' % (self.label, self.kind, self.input, outcome)


class Recorder:
    """Wrap a business callable and record each real call exactly once."""

    def __init__(self, func, label, kind, trace):
        self._func = func
        self._label = label
        self._kind = kind
        self._trace = trace

    def __call__(self, value):
        try:
            result = self._func(value)
        except Exception as exc:
            self._trace.append(
                TransformRecord(self._label, self._kind, value, False, exc)
            )
            raise
        self._trace.append(
            TransformRecord(self._label, self._kind, value, True, result)
        )
        return result


class _Rule:
    """One schema entry (key rule -> value rule) with recorder labels."""

    def __init__(self, index, skey, svalue):
        self.index = index
        self.skey = skey
        self.svalue = svalue
        self.key_label = None
        self.value_label = None

    @property
    def label(self):
        return 'rule#%d' % self.index

    def describe(self):
        return '%s: %r -> %r' % (self.label, self.skey, self.svalue)


def _is_literal(x):
    return type(x) in sb.primitive_types


def _recordable(x):
    return (
        callable(x)
        and not inspect.isclass(x)
        and not isinstance(x, sb.Marker)
        and x is not v.Extra
    )


def instrument(schema_dict, trace):
    """Wrap every recordable transform in ``schema_dict`` with a Recorder.

    Classes (``int``, ``str``...) and literals are left untouched: wrapping
    a class would turn voluptuous' isinstance check into a coercion.
    Markers keep their semantics; only a callable *inside* ``Remove`` is
    wrapped.  Returns ``(instrumented_schema, rules)``.
    """
    rules = []
    instrumented = {}
    for index, (skey, svalue) in enumerate(schema_dict.items()):
        rule = _Rule(index, skey, svalue)

        new_key = skey
        if isinstance(skey, sb.Remove):
            inner = skey.schema
            if _recordable(inner):
                rule.key_label = '%s:key' % rule.label
                new_key = sb.Remove(Recorder(inner, rule.key_label, 'key', trace))
            elif not (_is_literal(inner) or inspect.isclass(inner)):
                raise ValueError('unsupported Remove key: %r' % (inner,))
        elif isinstance(skey, sb.Marker):
            if not _is_literal(skey.schema):
                raise ValueError('only literal marker keys supported: %r' % (skey,))
        elif skey is v.Extra or _is_literal(skey) or inspect.isclass(skey):
            pass
        elif _recordable(skey):
            rule.key_label = '%s:key' % rule.label
            new_key = Recorder(skey, rule.key_label, 'key', trace)
        else:
            raise ValueError('unsupported schema key: %r' % (skey,))

        new_value = svalue
        if svalue is v.Extra or _is_literal(svalue) or inspect.isclass(svalue):
            pass
        elif _recordable(svalue):
            rule.value_label = '%s:value' % rule.label
            new_value = Recorder(svalue, rule.value_label, 'value', trace)
        else:
            raise ValueError('unsupported schema value: %r' % (svalue,))

        instrumented[new_key] = new_value
        rules.append(rule)
    return instrumented, rules


def _literal_key(skey):
    """The literal a rule is filed under in ``candidates_by_key``."""
    if isinstance(skey, sb.Marker):
        inner = skey.schema
        return inner if _is_literal(inner) else _NOLIT
    return skey if _is_literal(skey) else _NOLIT


def _priority(skey):
    # Mirrors schema_builder._compile_itemsort, including its quirk that
    # Extra (a plain function) matches is_callable before is_extra.
    if isinstance(skey, sb.Remove):
        return 1
    if isinstance(skey, sb.Marker):
        return 2
    if inspect.isclass(skey):
        return 4
    if callable(skey):
        return 3
    if skey is v.Extra:
        return 5
    return 0


class LedgerEntry:
    """One row of the ledger: the fate of a single input column."""

    def __init__(self, column):
        self.column = column
        self.path = 'data[%r]' % (column,)
        self.transforms = []  # TransformRecords, in execution order
        self.notes = []
        self.claim_rule = None
        self.disposition = None
        self.output_key = None
        self.detail = None


class Ledger:
    """Column-level account of one validation run."""

    def __init__(self, rules, trace, data, extra, output, error):
        self.rules = rules
        self.trace = list(trace)
        self.data = data
        self.extra = extra
        self.output = output
        self.error = error
        self.entries = []
        self._writes = []  # (output_key, column, value) in write order
        self._build()

    @property
    def validation_failed(self):
        return self.error is not None

    def entry(self, column):
        for e in self.entries:
            if e.column == column:
                return e
        raise KeyError(column)

    def render(self):
        lines = []
        for e in self.entries:
            lines.append('%s -> %s' % (e.path, e.disposition))
            for t in e.transforms:
                lines.append('    transform %s' % t.render())
            for n in e.notes:
                lines.append('    note: %s' % n)
            if e.output_key is not None:
                lines.append('    output key: %r' % (e.output_key,))
            if e.detail:
                lines.append('    %s' % e.detail)
        if self.validation_failed:
            lines.append(
                'VALIDATION FAILED: output discarded (%d error(s))'
                % len(self.error.errors)
            )
        return '\n'.join(lines)

    # -- replay of Schema._compile_mapping over the recorded trace -------

    def _build(self):
        queues = {}
        for rec in self.trace:
            queues.setdefault(rec.label, deque()).append(rec)
        for column, value in self.data.items():
            self.entries.append(self._replay_column(queues, column, value))
        self._resolve_collisions()
        if self.error is None:
            implied = {}
            for out_key, _column, val in self._writes:
                implied[out_key] = val
            assert implied == self.output, (implied, self.output)

    def _candidates_for(self, key):
        ordered = sorted(self.rules, key=lambda r: _priority(r.skey))
        direct = [
            r
            for r in ordered
            if _literal_key(r.skey) is not _NOLIT and _literal_key(r.skey) == key
        ]
        additional = [r for r in ordered if _literal_key(r.skey) is _NOLIT]
        return direct + additional

    @staticmethod
    def _consume(queues, label, expected_input):
        q = queues.get(label, ())
        assert q, 'ledger replay diverged: %s was never called' % label
        rec = q[0]
        assert rec.input == expected_input, (
            'ledger replay diverged: %s saw %r, expected %r'
            % (label, rec.input, expected_input)
        )
        return q.popleft()

    def _key_match(self, rule, key, queues, entry):
        skey = rule.skey
        if skey is v.Extra:
            return True, key
        if isinstance(skey, sb.Remove):
            inner = skey.schema
            if _is_literal(inner):
                return key == inner, sb.Remove
            if inspect.isclass(inner):
                return isinstance(key, inner), sb.Remove
            rec = self._consume(queues, rule.key_label, key)
            entry.transforms.append(rec)
            return rec.ok, sb.Remove
        if isinstance(skey, sb.Marker):
            return key == skey.schema, key
        if _is_literal(skey):
            return key == skey, key
        if inspect.isclass(skey):
            return isinstance(key, skey), key
        rec = self._consume(queues, rule.key_label, key)
        entry.transforms.append(rec)
        return (rec.ok, rec.output if rec.ok else None)

    def _value_match(self, rule, value, queues, entry):
        svalue = rule.svalue
        if svalue is v.Extra:
            return True, value
        if _is_literal(svalue):
            return value == svalue, value
        if inspect.isclass(svalue):
            return isinstance(value, svalue), value
        rec = self._consume(queues, rule.value_label, value)
        entry.transforms.append(rec)
        return (rec.ok, rec.output if rec.ok else None)

    def _replay_column(self, queues, column, value):
        entry = LedgerEntry(column)
        remove_matched = False
        for rule in self._candidates_for(column):
            matched, new_key = self._key_match(rule, column, queues, entry)
            if not matched:
                continue
            is_remove = new_key is sb.Remove
            ok, cval = self._value_match(rule, value, queues, entry)
            if ok:
                if is_remove:
                    # the library keeps scanning later candidates even
                    # after a Remove match (remove_key=True; continue)
                    remove_matched = True
                    entry.notes.append(
                        '%s matched; column marked for removal' % rule.label
                    )
                    continue
                entry.claim_rule = rule.label
                entry.disposition = KEPT
                entry.output_key = new_key
                self._writes.append((new_key, column, cval))
                break
            if is_remove or remove_matched:
                continue
            entry.claim_rule = rule.label
            entry.disposition = REJECTED
            entry.detail = 'value rejected by %s' % rule.label
            break
        else:
            if remove_matched:
                entry.disposition = REMOVED
            elif self.extra == v.ALLOW_EXTRA:
                entry.disposition = KEPT_AS_EXTRA
                entry.output_key = column
                self._writes.append((column, column, value))
            elif self.extra == v.REMOVE_EXTRA:
                entry.disposition = DROPPED_AS_EXTRA
            else:
                entry.disposition = REJECTED
                entry.detail = 'extra keys not allowed'
        return entry

    def _resolve_collisions(self):
        last_writer = {}
        for out_key, column, _val in self._writes:
            last_writer[out_key] = column
        by_column = {e.column: e for e in self.entries}
        writers_per_key = {}
        for out_key, column, _val in self._writes:
            writers_per_key.setdefault(out_key, []).append(column)
        for out_key, column, _val in self._writes:
            winner = last_writer[out_key]
            entry = by_column[column]
            if winner != column:
                entry.disposition = OVERWRITTEN
                entry.detail = 'normalized to %r, then overwritten by data[%r]' % (
                    out_key,
                    winner,
                )
            elif len(writers_per_key[out_key]) > 1:
                entry.notes.append('won collision for output key %r' % (out_key,))


def run_ledger(schema_dict, data, extra=v.PREVENT_EXTRA):
    """Validate ``data`` once and return the column-level Ledger."""
    trace = []
    instrumented, rules = instrument(schema_dict, trace)
    schema = v.Schema(instrumented, extra=extra)
    try:
        output = schema(data)
        error = None
    except er.MultipleInvalid as exc:
        output = None
        error = exc
    return Ledger(rules, trace, data, extra, output, error)


# ------------------------------------------------------------------------
# shared fixtures
# ------------------------------------------------------------------------


def normalize_key(key):
    """Business key transform: ints and digit-strings normalize to str."""
    if isinstance(key, bool):
        raise ValueError(key)
    if isinstance(key, int):
        return str(key)
    if isinstance(key, str) and key.isdigit():
        return key
    raise ValueError(key)


def to_a(key):
    return 'A'


def to_b(key):
    return 'B'


def _raise(value):
    raise ValueError('boom')


class BusinessTransform:
    """Stand-in business transform that records its real calls."""

    def __init__(self, func):
        self.func = func
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return self.func(value)


# ------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------


def test_ledger_attributes_three_ways_a_column_disappears():
    schema = {
        'name': str,
        v.Remove('password'): str,
        normalize_key: v.Coerce(int),
    }
    data = {'name': 'n', 'password': 'pw', 1: '5', '1': '7', 'junk': 'x'}
    ledger = run_ledger(schema, data, extra=v.REMOVE_EXTRA)

    assert ledger.output == {'name': 'n', '1': 7}

    assert ledger.entry('name').disposition == KEPT
    assert ledger.entry('name').output_key == 'name'

    # removed by the Remove marker
    password = ledger.entry('password')
    assert password.disposition == REMOVED
    # after the Remove match the library still tried the normalizer
    assert any(t.kind == 'key' and not t.ok for t in password.transforms)

    # normalized to '1', then overwritten by the later column '1'
    one = ledger.entry(1)
    assert one.disposition == OVERWRITTEN
    assert one.output_key == '1'
    assert [t.output for t in one.transforms] == ['1', 5]

    quoted = ledger.entry('1')
    assert quoted.disposition == KEPT
    assert quoted.output_key == '1'

    # matched by no rule, dropped under REMOVE_EXTRA
    assert ledger.entry('junk').disposition == DROPPED_AS_EXTRA

    text = ledger.render()
    assert "data['password'] -> removed-by-Remove" in text
    assert 'data[1] -> overwritten-by-collision' in text
    assert "data['junk'] -> dropped-as-extra" in text


def test_collision_same_output_from_different_histories():
    schema = {normalize_key: v.Coerce(int)}
    history_a = run_ledger(schema, {1: '5', '1': '7'})
    history_b = run_ledger(schema, {'1': '7'})
    history_c = run_ledger(schema, {1: '7'})

    # three different source histories, one identical result dict
    assert history_a.output == history_b.output == history_c.output == {'1': 7}

    # the ledger still tells them apart
    assert history_a.entry(1).disposition == OVERWRITTEN
    assert history_a.entry('1').disposition == KEPT
    assert len(history_b.entries) == 1
    assert history_b.entry('1').disposition == KEPT
    assert history_c.entry(1).disposition == KEPT
    # b kept a literal string key, c kept a normalized int key
    assert history_b.entry('1').transforms[0].input == '1'
    assert history_c.entry(1).transforms[0].input == 1


def test_input_column_order_decides_collision_winner():
    schema = {normalize_key: v.Coerce(int)}
    first = run_ledger(schema, {1: '5', '1': '7'})
    swapped = run_ledger(schema, {'1': '7', 1: '5'})

    # same schema rule order, same columns: only input order changed
    assert first.output == {'1': 7}
    assert swapped.output == {'1': 5}

    assert first.entry(1).disposition == OVERWRITTEN
    assert first.entry('1').disposition == KEPT
    assert swapped.entry('1').disposition == OVERWRITTEN
    assert swapped.entry(1).disposition == KEPT


def test_schema_rule_order_decides_which_rule_claims_a_column():
    data = {'k': 1}
    first = run_ledger({to_a: int, to_b: int}, data)
    swapped = run_ledger({to_b: int, to_a: int}, data)

    # same input column order: only the schema rule order changed
    assert first.output == {'A': 1}
    assert swapped.output == {'B': 1}

    # the first matching rule claims the column; the loser never runs
    assert [t.output for t in first.entry('k').transforms] == ['A']
    assert [t.output for t in swapped.entry('k').transforms] == ['B']


def test_error_on_later_column_after_earlier_transforms_ran():
    ok_transform = BusinessTransform(lambda v: v * 10)
    failing_transform = BusinessTransform(_raise)
    late_transform = BusinessTransform(lambda v: v)

    schema = {'a': ok_transform, 'b': failing_transform, 'c': late_transform}
    ledger = run_ledger(schema, {'a': 1, 'b': 2, 'c': 3})

    assert ledger.validation_failed
    assert ledger.output is None

    # Every business transform ran exactly once, in input column order --
    # including the column after the failure.  Building the ledger did not
    # feed any transform an extra value just for logging.
    assert ok_transform.calls == [1]
    assert failing_transform.calls == [2]
    assert late_transform.calls == [3]

    # earlier transforms already ran (side effects happened) even though
    # the whole output was discarded
    assert ledger.entry('a').disposition == KEPT
    assert ledger.entry('b').disposition == REJECTED
    assert ledger.entry('c').disposition == KEPT
    assert 'VALIDATION FAILED' in ledger.render()


def test_remove_match_still_falls_through_to_later_candidates():
    schema = {v.Remove('a'): str, str: int}

    # Remove matches but its value check fails, so the column falls
    # through to the later `str` rule, which claims and keeps it
    kept = run_ledger(schema, {'a': 5})
    assert kept.output == {'a': 5}
    assert kept.entry('a').disposition == KEPT

    # Remove matches with a passing value check: the column is marked for
    # removal, and the later `str` rule's failing value check is swallowed
    # (remove_key=True), so the column is dropped quietly, not rejected
    removed = run_ledger(schema, {'a': 'x'})
    assert removed.output == {}
    assert removed.entry('a').disposition == REMOVED
    assert any('marked for removal' in n for n in removed.entry('a').notes)


def test_extra_policy_dispositions():
    schema = {'name': str}

    allowed = run_ledger(schema, {'name': 'n', 'junk': 1}, extra=v.ALLOW_EXTRA)
    assert allowed.output == {'name': 'n', 'junk': 1}
    assert allowed.entry('junk').disposition == KEPT_AS_EXTRA

    dropped = run_ledger(schema, {'name': 'n', 'junk': 1}, extra=v.REMOVE_EXTRA)
    assert dropped.output == {'name': 'n'}
    assert dropped.entry('junk').disposition == DROPPED_AS_EXTRA

    prevented = run_ledger(schema, {'name': 'n', 'junk': 1}, extra=v.PREVENT_EXTRA)
    assert prevented.validation_failed
    assert prevented.entry('junk').disposition == REJECTED
    assert prevented.entry('junk').detail == 'extra keys not allowed'
