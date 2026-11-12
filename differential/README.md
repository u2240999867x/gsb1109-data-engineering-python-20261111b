# Differential projection regression

Two dpath projection plans map glob patterns to output columns:

- **flat** — `dpath.search(obj, glob, yielded=True)` → `[(path_string, value), ...]`
- **tree** — `dpath.search(obj, glob, yielded=False)` → a nested dict/list rebuilt
  from the same matches

An independent oracle (`ref_match` / `ref_walk` / `ref_project` in
`diff_regression.py`) re-implements the documented glob semantics without
calling `dpath.segments.match`. It keeps both the canonical string path and a
type-aware identity, so a list index and a text-numeric dict key stay
distinguishable even when their string forms coincide.

## Comparison convention

For every `(document, pattern)` pair, compare in this order:

1. path (multi)set,
2. row count,
3. raw values.

Patterns the library **rejects** (raises) and legal patterns that merely
project to an **empty** result are reported in separate buckets.

## Bounded, enumerated input contract

- depth 2 is exhausted completely (11,028 documents);
- depth 3 is covered by curated spine documents;
- an explicit collision suite covers int-vs-text keys, shared values,
  leading-zero / negative-index globs, bytes keys and empty-string keys;
- key alphabet `("a", "0", "1")`, leaves `(0, 7, "S")`, list length `<= 2`,
  dict keys `<= 2`; pattern alphabet includes `*`, `**`, `?`, `"0"`, `"00"`.

Each failing combination is shrunk by `minimize()` to a smallest document that
still triggers the same class.

## Run

```
.venv/bin/python -m differential.diff_regression
.venv/bin/python -m pytest tests/test_differential_projection.py -q
```

## Findings (minimal document + pattern)

Rejected vs. empty are kept apart:

- *rejected*: any glob with two `**` segments raises `InvalidGlob`,
  e.g. `**/**/a`.
- *legal but empty*: a well-formed glob that matches nothing returns `[]` /
  `{}`, e.g. `x/**` on `{"a": 1}`. No exception.

Plan crashes (legal pattern, one projection raises):

- **C1** `[0]` + `*` — the **tree** plan raises
  `TypeError: unhashable type: 'ListIndex'`; the flat plan returns
  `[("0", 0)]`. Root-level (and nested) list positions cannot be re-keyed into
  the result tree.
- **C2** `{"": 1}` + `*` — an empty-string key raises `InvalidKeyName` in both
  plans unless `dpath.options.ALLOW_EMPTY_STRING_KEYS` is set.
- **C3** bytes handling splits by glob form. A list-form glob `[b"k"]` makes
  the **flat** plan raise when joining the bytes path with a str separator
  (`sequence item 0: expected str instance, bytes found`), while the tree plan
  returns `{b"k": 1}`. A bytes-object glob `b"k"` raises `TypeError` in both
  plans.

Plan-vs-plan defects:

- **E1** `{0: "x", "0": "y"}` + `*` — the flat plan emits two rows with the
  same column name `"0"`; the tree plan keeps both typed keys. A column-keyed
  consumer such as `dict(rows)` overwrites one source value. With a shared
  value `{0: "S", "0": "S"}` the two flat rows are character-identical, so the
  merge is invisible without tracking source positions.
- **E2** `{"a": [10, 20]}` + `a/-1` — the flat plan yields one row
  `("a/1", 20)`; the tree plan rebuilds `{"a": [None, 20]}`, injecting a
  phantom `None` padding row at `a/0` that exists at no source position.

Shared deviations (both plans agree with each other, both differ from the
independent reference semantics):

- **E3** `{b"k": 1}` + `*` — bytes keys are silently invisible to a str glob
  (both plans return empty). A bytes *glob* segment instead raises `TypeError`.
- **E4** `{"a": [7]}` + `a/00` — the leading-zero text glob `"00"` is coerced
  through `int()` and matches list index `0`, producing column `a/0`. This is
  the numeric-coercion rule in `segments.match`; the reference fnmatch
  semantics treat `"00"` as a literal that does not match `"0"`.

Consistent boundaries worth pinning: `**` zero-segment collapse (`**/x`
matches `x`), and negative-index resolution (`a/-1` matches the last element)
behave identically in both plans.
