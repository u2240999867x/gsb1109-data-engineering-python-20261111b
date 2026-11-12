# -*- coding: utf-8 -*-
"""
Scenario B — a child validator is spawned with the WRONG parent document path.

A list's `schema` rule runs ONE sequence-level child over the document
{0: row0, 1: row1}; a custom `row_audit` rule on each row then spawns its own
grandchild. The row index is available there as `field` (0 or 1).

Two flavours of the same attribution bug:
  hard_coded : document_crumb pinned to 0 for every row
  no_crumb   : document_crumb omitted, grandchild keeps the parent path
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cerberus import Validator
from cerberus.errors import BaseErrorHandler

from common import dump_state


class RecordingHandler(BaseErrorHandler):
    def __init__(self, *args, **kwargs):
        super(RecordingHandler, self).__init__(*args, **kwargs)
        self.log = []

    def start(self, validator):
        self.log.append(('START', 'path=%r' % (validator.document_path,)))

    def emit(self, error):
        self.log.append(('EMIT ', 'doc_path=%r' % (error.document_path,),
                         'schema_path=%r' % (error.schema_path,),
                         'value=%r' % (error.value,)))

    def add(self, error):
        self.log.append(('ADD  ', 'doc_path=%r' % (error.document_path,)))

    def __call__(self, errors):
        return ['%s %r' % (e.document_path, e.info) for e in errors]


ROW_SCHEMA = {'detail': {'type': 'integer', 'min': 1000}}
last_child = {}


def build_validator(flavour, raise_inside=False):
    class RowAuditValidator(Validator):
        def _validate_row_audit(self, constraint, field, value):
            """{'type': 'boolean'}"""
            # In the sequence child, `field` is the row index (0 or 1).
            row_index = field
            if flavour == 'hard_coded':
                crumb = 0                       # BUG: always row 0
            elif flavour == 'no_crumb':
                crumb = None                    # BUG: inherit parent path
            else:
                crumb = row_index               # correct

            kwargs = dict(schema_crumb=(field, 'row_audit'), schema=ROW_SCHEMA)
            if crumb is not None:
                kwargs['document_crumb'] = crumb
            child = self._get_child_validator(**kwargs)
            last_child['v'] = child

            if raise_inside:
                # min fires first (MIN_VALUE is filed), then check_with raises
                child.schema = {
                    'detail': {
                        'type': 'integer',
                        'min': 1000,
                        'check_with': lambda f, v, err: (_ for _ in ()).throw(
                            RuntimeError('explosion in row %r' % row_index)),
                    }
                }

            print('   [audit] true row=%r parent path=%r -> child path=%r'
                  % (row_index, self.document_path, child.document_path))
            try:
                child(value, normalize=False)
            except RuntimeError as exc:
                print('   [audit] child raised %r; its _errors stay orphaned' % exc)
                raise
            if child._errors:
                self._error(child._errors)

    outer_schema = {
        'clients': {
            'type': 'list',
            'schema': {'type': 'dict', 'row_audit': True},
        }
    }
    return RowAuditValidator(schema=outer_schema, error_handler=RecordingHandler)


# Different true offending row, identical rendered output.
doc_row1_bad = {'clients': [{'detail': 5000}, {'detail': 'BAD'}]}
doc_row0_bad = {'clients': [{'detail': 'BAD'}, {'detail': 5000}]}

print('### B1a: hard-coded crumb 0 — truth is row 1')
v = build_validator('hard_coded')
v.validate(doc_row1_bad)
dump_state(v, 'B1a (display blames row 0)', doc_row1_bad)

print('### B1b: hard-coded crumb 0 — truth is row 0')
v = build_validator('hard_coded')
v.validate(doc_row0_bad)
dump_state(v, 'B1b (display blames row 0 — this time correctly)', doc_row0_bad)

print('### B2: no crumb — two rows collapse onto one node')
v = build_validator('no_crumb')
v.validate({'clients': [{'detail': 'BAD0'}, {'detail': 'BAD1'}]})
dump_state(v, 'B2 (origin rows no longer separable in display)')

print('### B3: grandchild raises after its priority rule already filed an error')
# row 0: detail=5 fails min 1000 (MIN_VALUE filed), then check_with raises
b3_doc = {'clients': [{'detail': 5}, {'detail': 5000}]}
v = build_validator('hard_coded', raise_inside=True)
try:
    v.validate(b3_doc)
except RuntimeError as exc:
    print('propagated out of validate(): %s' % exc)
dump_state(v, 'B3 parent state after escape')
child = last_child['v']
print('--- orphaned grandchild (never bulk-imported into the parent) ---')
print('    child.document_path =', child.document_path)
print('    child._errors       =')
for err in child._errors:
    print('       doc_path=%r value=%r rule=%r'
          % (err.document_path, err.value, err.rule))
print()
