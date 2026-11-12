"""Tests for the incremental lexical diagnostics in
:mod:`sqlparse.incremental`."""

import random

import pytest

from sqlparse import tokens as T
from sqlparse.incremental import (
    LexState,
    StatementSpan,
    build_lex_state,
    build_statement_catalog,
    incremental_relex,
)

TOKENS = [
    'select', '1', ' ', ';', 'foo', 'bar', "'", "''", '/*', '*/',
    '$body$', '$x$', '-- c\n', '(', ')', 'begin', 'end', 'x;', '  ', '"',
]


def _bounds(catalog):
    return [(entry.start, entry.end) for entry in catalog]


def _normalized(state):
    spans = sorted((span.ttype, span.start, span.end, span.kind,
                    span.tag, span.pending) for span in state.spans)
    return LexState(spans, tuple(sorted(state.errors)), state.incomplete)


def _run(text, start, old_end, replacement):
    state = build_lex_state(text)
    catalog = build_statement_catalog(text)
    result = incremental_relex(text, state, catalog, start, old_end,
                               replacement)
    new_text = text[:start] + replacement + text[old_end:]
    expected_state = build_lex_state(new_text)
    expected_catalog = build_statement_catalog(new_text)
    return result, new_text, expected_state, expected_catalog


def test_build_state_closed_spans():
    state = build_lex_state(
        "select 'a'; /* c */ select $b$x$b$;")
    kinds = {(span.kind, span.pending) for span in state.spans}
    assert ('string', False) in kinds
    assert ('comment', False) in kinds
    assert ('dollar', False) in kinds
    assert not state.incomplete
    assert state.errors == ()


def test_build_state_pending_spans_and_errors():
    for sql, kind in (
        ("select 1; /* unclosed", 'comment'),
        ("select $b$ unclosed", 'dollar'),
        ("select 'unclosed", 'string'),
    ):
        state = build_lex_state(sql)
        assert state.incomplete
        pending = {span.kind for span in state.spans if span.pending}
        assert kind in pending
    # The unmatched quote is also reported as a lexical error interval.
    string_state = build_lex_state("select 'unclosed")
    assert string_state.errors
    assert all(lo < hi for lo, hi in string_state.errors)


def test_catalog_matches_full_split():
    sql = "select 1; select 'a;b'; begin select 2; end; select 3;"
    catalog = build_statement_catalog(sql)
    assert ''.join(entry.slice(sql) for entry in catalog) == sql
    assert all(isinstance(entry, StatementSpan) for entry in catalog)
    assert _bounds(catalog)[0][0] == 0


def test_repair_unclosed_quote_restores_string():
    sql = "select 'abc; select 1; select 2;"
    pos = sql.index('abc') + 3
    result, new_text, exp_state, exp_catalog = _run(sql, pos, pos, "'")

    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    assert result.recovered
    assert not result.state.incomplete
    assert result.state.errors == ()
    assert any(span.kind == 'string' and not span.pending
               for span in result.state.spans)
    assert result.earliest_affected == 0
    assert ''.join(entry.slice(new_text)
                   for entry in result.catalog) == new_text


def test_opening_quote_records_pending_string():
    sql = "select 'abc'; select 1; select 2;"
    pos = sql.index('abc') + 3
    result, _new, exp_state, exp_catalog = _run(sql, pos, pos + 1, '')

    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    assert result.state.incomplete
    assert not result.recovered
    assert any(span.kind == 'string' and span.pending
               for span in result.state.spans)


def test_closing_block_comment_merges_statements():
    # Adding the missing closer turns the trailing text into one comment
    # span, so the semicolons inside it stop splitting statements.
    sql = "select 1; /* unclosed select 2; select 3;"
    old_count = len(build_statement_catalog(sql))
    result, _new, exp_state, exp_catalog = _run(
        sql, len(sql), len(sql), ' */')

    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    assert len(result.catalog) == old_count - 1
    comment = next(span for span in result.state.spans
                   if span.kind == 'comment')
    assert not comment.pending


def test_removing_block_comment_closer_splits_statements():
    repaired = "select 1; /* unclosed select 2; select 3; */"
    old_count = len(build_statement_catalog(repaired))
    closer = repaired.index('*/')
    result, _new, exp_state, exp_catalog = _run(
        repaired, closer, closer + 2, '')

    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    assert len(result.catalog) == old_count + 1
    assert any(span.kind == 'comment' and span.pending
               for span in result.state.spans)


def test_dollar_quote_pairing_and_pending_state():
    sql = "select $body$ x; yyy select 2; select 3;"
    pos = sql.index(' select 2')
    result, _new, exp_state, exp_catalog = _run(
        sql, pos, pos, ' $body$')
    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    dollar = next(span for span in result.state.spans
                  if span.kind == 'dollar')
    assert dollar.tag == '$body$'
    assert not dollar.pending


def test_suffix_directory_is_reused_with_offset_shift():
    tail = ''.join(f'select {i}; ' for i in range(200))
    sql = "select aaa; " + tail
    start = sql.index('aaa')
    result, new_text, exp_state, exp_catalog = _run(
        sql, start, start + 3, 'bbbbb')

    assert result.realigned
    assert result.suffix_reused
    # Only the head is re-tokenized; the long suffix is never lexed again.
    assert result.relex_end < len(new_text) // 2
    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)

    delta = 2
    assert result.shifted_from is not None
    old_catalog = build_statement_catalog(sql)
    # Reused entries are pure translations and keep their statement text.
    for old_index in range(result.shifted_from, len(old_catalog)):
        entry = result.catalog[old_index]
        assert entry.start == old_catalog[old_index].start + delta
        assert entry.end == old_catalog[old_index].end + delta
        assert entry.slice(new_text) == old_catalog[old_index].slice(sql)


def test_prefix_statements_are_untouched():
    tail = ''.join(f'select {i}; ' for i in range(100))
    sql = "select 1; select 2; " + tail
    start = sql.index('2')
    result, _new, _exp_state, _exp_catalog = _run(sql, start, start + 1, '9')
    old_catalog = build_statement_catalog(sql)
    for index in range(result.changed_from):
        assert result.catalog[index] == old_catalog[index]


def test_no_realignment_reports_earliest_affected_statement():
    # An unclosed opener at end of input can hide later delimiters, so the
    # suffix cannot be proven reusable.
    tail = ''.join(f'select {i}; ' for i in range(100))
    sql = "select 1; " + tail
    start = sql.index('1')
    result, new_text, exp_state, exp_catalog = _run(sql, start, start, "'")

    assert not result.realigned
    assert not result.suffix_reused
    assert result.relex_end == len(new_text)
    assert result.earliest_affected is not None
    earliest = result.catalog[result.earliest_affected]
    assert earliest.start <= start < earliest.end
    # The full rebuild remains the reconciliation reference.
    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)


def test_semicolon_inside_string_is_not_a_sync_point():
    sql = "select 'a;b'; select 2; select 3; select 4;"
    start = sql.index('a')
    result, _new, _exp_state, _exp_catalog = _run(sql, start, start + 1, 'z')
    assert result.realigned
    # The first boundary must be the end of the whole string statement,
    # never the in-string semicolon.
    assert result.relex_end >= sql.index("';") + 2


def test_trailing_edit_reuses_realignment_at_eof():
    sql = "select 1; select 2"
    result, new_text, exp_state, exp_catalog = _run(
        sql, len(sql), len(sql), ';')
    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)
    assert result.earliest_affected == 1


def test_invalid_replacement_range():
    sql = "select 1;"
    state = build_lex_state(sql)
    catalog = build_statement_catalog(sql)
    with pytest.raises(ValueError):
        incremental_relex(sql, state, catalog, 0, len(sql) + 1, 'x')
    with pytest.raises(ValueError):
        incremental_relex(sql, state, catalog, 2, 1, 'x')


def test_empty_document():
    result, new_text, exp_state, exp_catalog = _run('', 0, 0, 'select 1;')
    assert new_text == 'select 1;'
    assert _bounds(result.catalog) == _bounds(exp_catalog)
    assert _normalized(result.state) == _normalized(exp_state)


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_incremental_matches_full_rebuild_randomly(seed):
    rng = random.Random(seed)
    for _ in range(120):
        sql = ''.join(rng.choice(TOKENS)
                      for _ in range(rng.randint(1, 130)))
        start = rng.randint(0, len(sql))
        old_end = rng.randint(start, min(len(sql), start + 12))
        replacement = ''.join(rng.choice(TOKENS)
                              for _ in range(rng.randint(0, 10)))
        result, _new, exp_state, exp_catalog = _run(
            sql, start, old_end, replacement)
        assert _bounds(result.catalog) == _bounds(exp_catalog)
        assert _normalized(result.state) == _normalized(exp_state)
        # A reused suffix is always a pure offset translation.
        if result.realigned:
            delta = len(replacement) - (old_end - start)
            old_catalog = build_statement_catalog(sql)
            for old_index in range(result.shifted_from, len(old_catalog)):
                entry = result.catalog[
                    old_index + len(result.catalog) - len(old_catalog)]
                assert entry.start == old_catalog[old_index].start + delta
                assert entry.end == old_catalog[old_index].end + delta


def test_pending_span_is_exposed_on_state():
    state = build_lex_state("select 'unclosed")
    pending = state.pending_span()
    assert pending is not None
    assert pending.kind == 'string'
    assert pending.ttype in T.String
    assert pending.pending
