# -*- coding: utf-8 -*-
"""Shared dumping helpers: show document, raw errors, trees and emit timeline."""
from pprint import pformat

from cerberus.errors import BasicErrorHandler


def dump_state(validator, banner, document=None):
    line = '=' * 80
    print(line)
    print(banner)
    if document is not None:
        print('input document :', document)
    print('validator.document now =', validator.document)
    print('document_path  =', validator.document_path)
    print('schema_path    =', validator.schema_path)
    print('--- emit timeline (streaming handler callbacks, in call order) ---')
    log = getattr(validator.error_handler, 'log', None)
    if log is None:
        print('     (handler has no log)')
    else:
        for i, entry in enumerate(log):
            print('  %2d  %s' % (i, entry))
    print('--- raw v._errors: ground-truth ValidationError objects ---')
    if not validator._errors:
        print('     (empty)')
    for err in validator._errors:
        print('     doc_path=%-30r schema_path=%-34r code=%s rule=%r info=%r'
              % (err.document_path, err.schema_path, hex(err.code), err.rule, err.info))
        if err.is_group_error:
            for child in err.child_errors:
                print('        └ child doc_path=%-24r code=%s info=%r'
                      % (child.document_path, hex(child.code), child.info))
    print('--- document_error_tree (structure follows the *document*) ---')
    print('   ', pformat(_tree_to_dict(validator.document_error_tree)))
    print('--- rendered v.errors (what a consumer actually displays) ---')
    print('   ', BasicErrorHandler()(validator._errors))
    print()


def _tree_to_dict(node):
    out = {}
    for key, child in node.descendants.items():
        entry = {}
        if child.errors:
            entry['<errors>'] = [
                (hex(e.code), e.info) for e in child.errors
            ]
        entry.update(_tree_to_dict(child))
        out[key] = entry
    return out
