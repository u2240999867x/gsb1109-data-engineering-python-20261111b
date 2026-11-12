# -*- coding: utf-8 -*-
"""
Pinpoint the FIRST attribution divergence for each of the three mechanisms.

We trace, in real call order:
  * Validator.__init_processing state rebinding
  * _get_child_validator path/crumb assignment
  * every ValidationError object as it is constructed (its document_path)
  * every document_error_tree insertion and handler.emit

The "true origin" of an error is the validator's current document value at the
moment the rule fails. The first constructed error whose document_path points
at a different row/document than its true origin is the divergence point.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import cerberus.validator as cv
from cerberus import Validator
from cerberus.errors import ValidationError, BaseErrorHandler, CUSTOM

seq = []

orig_init_processing = cv.BareValidator._BareValidator__init_processing
orig_get_child = cv.BareValidator._get_child_validator
orig_error = cv.BareValidator._error
orig_ve_init = ValidationError.__init__


def traced_init_processing(self, document, schema=None):
    seq.append(('init_processing ENTER', 'id=%d' % id(self),
                'old _errors id=%s' % id(self._errors),
                'path=%r' % (self.document_path,)))
    orig_init_processing(self, document, schema)
    seq.append(('init_processing EXIT ', 'id=%d' % id(self),
                'NEW _errors id=%s' % id(self._errors),
                'document rebound=%r' % document,
                'path still=%r' % (self.document_path,)))


def traced_get_child(self, document_crumb=None, schema_crumb=None, **kwargs):
    child = orig_get_child(self, document_crumb=document_crumb,
                           schema_crumb=schema_crumb, **kwargs)
    seq.append(('child spawned', 'crumb=%r' % (document_crumb,),
                'parent path=%r' % (self.document_path,),
                'child path=%r' % (child.document_path,)))
    return child


def traced_ve_init(self, document_path, schema_path, code, rule,
                   constraint, value, info):
    seq.append(('ValidationError built', 'doc_path=%r' % (document_path,),
                'rule=%r' % rule, 'value=%r' % value))
    orig_ve_init(self, document_path, schema_path, code, rule,
                 constraint, value, info)


class TracingHandler(BaseErrorHandler):
    def __init__(self, *a, **k):
        super(TracingHandler, self).__init__(*a, **k)
        self.lines = []

    def start(self, validator):
        seq.append(('handler.start', 'reset lines (was %d)' % len(self.lines)))
        self.lines = []

    def emit(self, error):
        seq.append(('handler.emit ', 'doc_path=%r' % (error.document_path,),
                    'tree node exists first: %s' % (
                        cv_validator_holder['v'].document_error_tree
                        .fetch_node_from(error.document_path) is not None)))
        self.lines.append(error.document_path)

    def add(self, error):
        seq.append(('handler.add  ', 'doc_path=%r' % (error.document_path,)))

    def __call__(self, errors):
        return list(self.lines)


cv_validator_holder = {}

cv.BareValidator._BareValidator__init_processing = traced_init_processing
cv.BareValidator._get_child_validator = traced_get_child
ValidationError.__init__ = traced_ve_init


def run(title, build, document):
    print('=' * 84)
    print(title)
    seq.clear()
    v = build()
    cv_validator_holder['v'] = v
    try:
        v.validate(document)
    except RuntimeError as exc:
        seq.append(('EXCEPTION escaped', str(exc)))
    for i, event in enumerate(seq):
        print('  %2d  %-24s %s' % (i, event[0], ' | '.join(event[1:])))
    print()


# ---------------------------------------------------------------- Scenario A
SCHEMA_A = {
    'clients': {'type': 'list', 'schema': {'type': 'dict', 'schema': {
        'name': {'type': 'string'},
        'detail': {'check_with': 'reentrant'}}}}
}


def build_a():
    class V(Validator):
        def _check_with_reentrant(self, field, value):
            self._error(field, CUSTOM, 'legit for %r' % (self.document_path,))
            self.validate({'rogue_field': 'not-an-int'},
                          schema={'rogue_field': {'type': 'integer'}},
                          normalize=False)
    return V(schema=SCHEMA_A, error_handler=TracingHandler)


run('A: re-entrant check_with — first divergence when __init_processing wipes '
    'the legit error and the rogue error inherits the row path',
    build_a, {'clients': [{'name': 'alice', 'detail': 'x'},
                          {'name': 'bob', 'detail': 'y'}]})

# ---------------------------------------------------------------- Scenario B
ROW_SCHEMA = {'detail': {'type': 'integer'}}


def build_b():
    class V(Validator):
        def _validate_row_audit(self, constraint, field, value):
            """{'type': 'boolean'}"""
            child = self._get_child_validator(
                document_crumb=0,                          # hard-coded: wrong for row 1
                schema_crumb=(field, 'row_audit'),
                schema=ROW_SCHEMA)
            child(value, normalize=False)
            if child._errors:
                self._error(child._errors)
    return V(schema={'clients': {'type': 'list',
                                 'schema': {'type': 'dict', 'row_audit': True}}},
             error_handler=TracingHandler)


run('B: wrong parent crumb — first divergence at child spawn for true row 1 '
    'which is assigned path row 0',
    build_b, {'clients': [{'detail': 5000}, {'detail': 'BAD'}]})

# ---------------------------------------------------------------- Scenario C
SCHEMA_C = {'clients': {'type': 'list', 'schema': {'type': 'dict', 'schema': {
    'detail': {'type': 'integer', 'min': 1000}}}}}


class NonResettingHandler(TracingHandler):
    def start(self, validator):
        # deliberately do NOT reset self.lines (the base BasicErrorHandler
        # calls clear() here; this subclass keeps previous-customer state)
        seq.append(('handler.start', 'NO reset, lines kept=%d' % len(self.lines)))


def build_c():
    return Validator(schema=SCHEMA_C, error_handler=NonResettingHandler)


print('=' * 84)
print('C: stale handler across two customers (one validator instance)')
seq.clear()
v = build_c()
cv_validator_holder['v'] = v
v.validate({'clients': [{'detail': 42}]})          # alice, invalid
for i, event in enumerate(seq):
    print('  %2d  %-24s %s' % (i, event[0], ' | '.join(event[1:])))
print('  --- second customer, actually valid ---')
seq.clear()
v.validate({'clients': [{'detail': 5000}]})        # bob, valid
for i, event in enumerate(seq):
    print('  %2d  %-24s %s' % (i, event[0], ' | '.join(event[1:])))
print('  ground-truth v._errors after bob:', v._errors)
print('  displayed v.errors after bob   :', v.errors)
