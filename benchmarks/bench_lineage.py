"""Lineage benchmark for verbatim region resolution and source mapping.

The offset-tracked pipeline must stay linear in the size of the input:
marker scanning (:func:`sqlparse.lineage.iter_code_spans`) is a single
state-machine pass and each output chunk appends one segment in constant
time.  A regression that re-scanned the text per marker or per output
position would show up here as super-linear growth, the same way the
lexer delimiter benchmark guards GHSA-prg7-hcfm-mfcr.

Run with:  python benchmarks/bench_lineage.py
"""

import sys

from _harness import Vector, main

import sqlparse
from sqlparse.lineage import resolve_directives

SIZES = (250, 500, 1000, 2000)
RESOLVE_SIZES = (250, 500, 1000, 2000, 4000)


def wide_verbatim_select(n):
    """n columns, every other one wrapped in its own verbatim region."""
    columns = []
    for index in range(n):
        col = f'col_{index}'
        if index % 2:
            col = f'@@VBEGIN@@{col}@@VEND@@'
        columns.append(col)
    return 'SELECT ' + ', '.join(columns) + ' FROM t'


def run_lineage(sql):
    return sqlparse.format_lineage(sql, reindent=True)


def run_resolve(sql):
    return resolve_directives(sql)


VECTORS = [
    Vector('format_lineage, many regions', wide_verbatim_select,
           SIZES, run_lineage),
    Vector('marker resolution only', wide_verbatim_select,
           RESOLVE_SIZES, run_resolve),
]

if __name__ == '__main__':
    sys.exit(main('lineage', VECTORS,
                  note='verbatim regions + output-to-source mapping',
                  argv=sys.argv[1:]))
