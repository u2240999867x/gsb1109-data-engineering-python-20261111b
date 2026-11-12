# -*- coding: utf-8 -*-
"""
Scenario A — custom check_with re-enters the SAME Validator instance.

The list `schema` rule spawns a child per row; a check_with running on that
child calls self.validate(...) again while the outer run is still in progress.
__init_processing() then rebinds the SAME object's _errors / trees / document.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cerberus import Validator
from cerberus.errors import BaseErrorHandler, CUSTOM

from common import dump_state


class RecordingHandler(BaseErrorHandler):
    """Streaming handler on the ROOT validator: logs emit() in call order."""

    def __init__(self, *args, **kwargs):
        super(RecordingHandler, self).__init__(*args, **kwargs)
        self.log = []

    def start(self, validator):
        self.log.append(('START', 'path=%r' % (validator.document_path,),
                         'doc=%r' % (validator.document,)))

    def emit(self, error):
        self.log.append(('EMIT ', 'doc_path=%r' % (error.document_path,),
                         'code=%s' % hex(error.code)))

    def add(self, error):
        self.log.append(('ADD  ', 'doc_path=%r' % (error.document_path,)))

    def __call__(self, errors):
        return ['%s %s' % (e.document_path, hex(e.code)) for e in errors]


SCHEMA = {
    'clients': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {
                'name': {'type': 'string'},
                'detail': {'check_with': 'reentrant'},
            },
        },
    }
}

ROGUE_SCHEMA = {'rogue_field': {'type': 'integer'}}
stranded = {}


def build_validator(raise_after_reentry):
    class ReentrantValidator(Validator):
        def _check_with_reentrant(self, field, value):
            base = self.document_path            # ('clients', row_index)
            print('   [check] enter  field=%r value=%r path=%r id(self)=%d'
                  % (field, value, base, id(self)))

            # 1) legitimate error for THIS row, filed before re-entry
            self._error(field, CUSTOM, 'legit failure for %r' % (base,))
            legit = self.recent_error
            print('   [check] pre-reentry  _errors=%d  holds legit error: %r'
                  % (len(self._errors), legit.document_path))

            # 2) re-enter the very same validator instance with another document
            self.validate({'rogue_field': 'not-an-int'},
                          schema=ROGUE_SCHEMA, normalize=False)

            print('   [check] post-reentry _errors=%d  legit still present: %s'
                  % (len(self._errors), any(e is legit for e in self._errors)))
            print('   [check] post-reentry self.document=%r (was the row dict)'
                  % (self.document,))
            for err in self._errors:
                print('   [check]   surviving error doc_path=%r value=%r'
                      % (err.document_path, err.value))

            stranded['v'] = self
            stranded['errors'] = list(self._errors)
            if raise_after_reentry:
                raise RuntimeError('boom at %r' % (base,))

    return ReentrantValidator(schema=SCHEMA, error_handler=RecordingHandler)


DOCUMENT = {
    'clients': [
        {'name': 'alice', 'detail': 'row0-detail'},
        {'name': 'bob', 'detail': 'row1-detail'},
    ]
}

print('### A1: re-entrant check_with, check returns normally')
v = build_validator(raise_after_reentry=False)
print('validate() ->', v.validate(DOCUMENT))
dump_state(v, 'A1 final state', DOCUMENT)

print('### A2: re-entrant check_with, check RAISES after re-entry')
stranded.clear()
v2 = build_validator(raise_after_reentry=True)
try:
    v2.validate(DOCUMENT)
except RuntimeError as exc:
    print('validate() propagated: %s' % exc)
dump_state(v2, 'A2 root state after exception escape', DOCUMENT)
child = stranded.get('v')
print('--- stranded child object (unreachable via the public root API) ---')
print('    child.document_path =', child.document_path)
print('    child.document      =', child.document)
for err in stranded['errors']:
    print('    stranded error doc_path=%r rule=%r value=%r'
          % (err.document_path, err.rule, err.value))
