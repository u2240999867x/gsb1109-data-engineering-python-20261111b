# column_import

Per-batch parsing sessions for a shared pyparsing column-mapping grammar.

Two data-import streams share one grammar.  The same short code resolves to
different output columns depending on the batch's semantic version, so every
parse runs inside an explicit `ParseSession` that binds the run to a source, a
semantic version and the batch's code tables, and owns a private packrat
cache.

## Layout

- `column_import/__init__.py` — `ParseSession`, the session-scoped packrat
  cache, code-table lookup helpers and `nested_parse`.
- `column_import/grammar.py` — the single shared grammar (never copied):
  parent rows resolve codes from the `columns` table; the braced attached
  table is parsed recursively from the separate `attachment_columns` table.
- `tests/test_column_import_sessions.py` — coverage for cache scoping,
  interleaved versions, recursive failure isolation and cache release.

## Usage

```python
from column_import import ParseSession
from column_import.grammar import parse_document

code_tables = {
    "columns": {"A": "alpha_v1", "B": "beta_v1"},
    "attachment_columns": {"X": "att_x_v1"},
}

with ParseSession(source="stream-a", semantic_version="v1",
                  code_tables=code_tables) as session:
    rows = parse_document("A,B\nA,{X}")
    # [{'columns': ['alpha_v1', 'beta_v1'], 'attachment': None},
    #  {'columns': ['alpha_v1'],
    #   'attachment': {'raw': 'X', 'items': ['att_x_v1'], 'error': None}}]
```

## Guarantees

- A packrat entry is only reused within one session, whose cache keys already
  include the source string and input position; different sources or semantic
  versions never share entries.
- A failure while recursively parsing an attached table is captured on that
  row's `attachment` field.  The outer parse continues and already resolved
  parent columns are retained.
- The grammar object is shared, never deep-copied.  Leaving the `with` block
  (or calling `close()`) releases the batch's cache and restores pyparsing's
  prior packrat state.
