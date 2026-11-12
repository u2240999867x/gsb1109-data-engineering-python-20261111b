# -*- coding: utf-8 -*-
"""
Scenario C — the SAME error-handler instance keeps state from the previous row.

The validator's own containers (_errors, both error trees, document, paths) are
reset in __init_processing(), but the error handler is a long-lived object.
A handler that accumulates emit()ed lines and does not reset them in start()
keeps the previous customer's formatted line alive in the display.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cerberus import Validator
from cerberus.errors import BasicErrorHandler, BaseErrorHandler

from common import dump_state


class LeakyHandler(BaseErrorHandler):
    """Accumulates emitted formatted lines; start() fails to reset them."""

    def __init__(self, *args, **kwargs):
        super(LeakyHandler, self).__init__(*args, **kwargs)
        self.lines = []
        self.events = []

    def start(self, validator):
        # BUG: no reset of self.lines here
        self.events.append(('START', validator.document_path))

    def emit(self, error):
        line = BasicErrorHandler.messages[error.code].format(
            *error.info, constraint=error.constraint,
            field=error.field, value=error.value)
        self.lines.append((error.document_path, line))
        self.events.append(('EMIT', error.document_path, line))

    def add(self, error):
        self.events.append(('ADD', error.document_path))

    def __call__(self, errors):
        return list(self.lines)


SCHEMA = {
    'clients': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {'detail': {'type': 'integer', 'min': 1000}},
        },
    }
}

cust_alice = {'clients': [{'detail': 42}]}          # alice's row 0 invalid
cust_bob = {'clients': [{'detail': 5000}]}          # bob perfectly valid

handler = LeakyHandler()
v = Validator(schema=SCHEMA, error_handler=handler)

print('### C1: two customers through ONE validator / ONE handler instance')
print('-- run 1: alice (fails) --')
print('   validate ->', v.validate(cust_alice))
dump_state(v, 'C1 after alice')

print('-- run 2: bob (actually valid) --')
print('   validate ->', v.validate(cust_bob))
print('   bob raw _errors        :', v._errors)
print('   bob document_error_tree:', dict(v.document_error_tree.descendants))
print('   bob v.errors (LEAKED display, alice\'s line survives):')
for line in v.errors:
    print('      ', line)
print()

print('### C2: remediate at format time? Re-render each run\'s GROUND-TRUTH raw')
print('         errors through a fresh, correctly-resetting handler.')
v2 = Validator(schema=SCHEMA)
v2.validate(cust_alice)
alice_raw = list(v2._errors)
v2.validate(cust_bob)
bob_raw = list(v2._errors)
print('   alice re-rendered:', BasicErrorHandler()(alice_raw))
print('   bob   re-rendered:', BasicErrorHandler()(bob_raw))
print()

print('### C3: callback ordering — in _error(), the tree is populated BEFORE')
print('         error_handler.emit() runs (validator.py _error, len(args)==1).')


class OrderProbe(BaseErrorHandler):
    def __init__(self, *a, **k):
        super(OrderProbe, self).__init__(*a, **k)
        self.events = []

    def start(self, validator):
        self.events.append(('start',))

    def emit(self, error):
        node = holder['v'].document_error_tree.fetch_node_from(error.document_path)
        already_in_tree = node is not None and any(e is error for e in node.errors)
        self.events.append(('emit', error.document_path,
                            'tree already contains this error: %s' % already_in_tree))

    def add(self, error):
        self.events.append(('add', error.document_path))

    def __call__(self, errors):
        return self.events


probe = OrderProbe()
holder = {}
v3 = Validator(schema=SCHEMA, error_handler=probe)
holder['v'] = v3
v3.validate(cust_alice)
for e in probe.events:
    print('   ', e)
