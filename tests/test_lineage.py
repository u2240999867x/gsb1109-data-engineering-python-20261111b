"""Tests for verbatim regions and output-to-source range mapping."""

import re

import pytest

import sqlparse
from sqlparse.exceptions import SQLParseError
from sqlparse.lineage import (
    DEFAULT_CLOSE_MARKER,
    DEFAULT_OPEN_MARKER,
    SourceMap,
    line_col,
    resolve_directives,
)


def _word_position(text, word):
    """Return the start of *word* as a standalone SQL word in *text*."""
    match = re.search(rf'(?<![\w]){re.escape(word)}(?![\w])', text)
    assert match is not None
    return match.start()


class TestSourceMapping:
    def test_basic_offsets_point_back_to_source(self):
        sql = 'select a, b from t'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        for name in ('a', 'b', 't'):
            pos = _word_position(out, name)
            start, end = source_map.source_for(pos)
            assert source_map.fragment_for(pos) == name
            assert sql[start:end] == name

    def test_duplicate_column_text_is_disambiguated(self):
        sql = 'select foo, foo from t'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        positions = [i for i in range(len(out)) if out.startswith('foo', i)]
        assert len(positions) == 2
        spans = [source_map.source_for(pos) for pos in positions]
        assert spans == [(7, 10), (12, 15)]
        assert [sql[s:e] for s, e in spans] == ['foo', 'foo']

    def test_full_range_round_trip_on_a_single_token(self):
        sql = 'select amount from ledger'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        pos = out.index('amount')
        start, end = source_map.source_for(pos, pos + len('amount'))
        assert sql[start:end] == 'amount'

    def test_multi_token_column_maps_as_one_contiguous_range(self):
        sql = 'select schema.table_col, a + b from t'
        out, source_map = sqlparse.format_lineage(sql)
        qualified = 'schema.table_col'
        pos = out.index(qualified)
        assert source_map.fragment_for(pos, pos + len(qualified)) == qualified
        expression = 'a + b'
        pos_expr = out.index(expression)
        assert (source_map.fragment_for(pos_expr, pos_expr + len(expression))
                == expression)

    def test_bytes_input_uses_character_offsets(self):
        sql = 'select 金额, b from t'.encode('utf-8')
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        pos = out.index('金额')
        assert source_map.fragment_for(pos, pos + 2) == '金额'

    def test_newline_conversion_points_back_to_original_break(self):
        sql = 'select a\r\nfrom t'
        out, source_map = sqlparse.format_lineage(sql)
        newline = out.index('\n')
        start, end = source_map.source_for(newline)
        assert sql[start:end] == '\r\n'

        sql_cr = 'select a\rfrom t'
        out_cr, map_cr = sqlparse.format_lineage(sql_cr)
        start, end = map_cr.source_for(out_cr.index('\n'))
        assert sql_cr[start:end] == '\r'

    def test_line_break_inside_keyword_token_is_mapped(self):
        sql = 'order\r\nby , a\r\n'
        out, source_map = sqlparse.format_lineage(
            sql, strip_whitespace=True)
        start, end = source_map.source_for(out.index('\n'))
        assert sql[start:end] == '\r\n'

    def test_line_break_inside_string_literal_is_preserved(self):
        sql = "select 'line1\r\nline2' from t"
        out, source_map = sqlparse.format_lineage(sql)
        literal = "'line1\r\nline2'"
        pos = out.index(literal)
        assert source_map.fragment_for(pos, pos + len(literal)) == literal
        assert '\r\n' in out

    def test_multibyte_alias_keeps_character_offsets(self):
        sql = 'select a as 金额 from t'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        pos = out.index('金额')
        start, end = source_map.source_for(pos, pos + 2)
        assert sql[start:end] == '金额'
        assert source_map.fragment_for(pos, pos + 2) == '金额'

    def test_multiple_statements_offsets_are_shifted(self):
        sql = 'select a; select b'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        first = out.index('a')
        second = out.rindex('b')
        assert source_map.fragment_for(first) == 'a'
        assert source_map.fragment_for(second) == 'b'
        # Each statement resolves to the matching source occurrence even
        # though the formatting inserts a blank line between them.
        assert source_map.source_for(first) == (7, 8)
        assert source_map.source_for(second) == (17, 18)

    def test_inserted_indentation_is_unmappable(self):
        out, source_map = sqlparse.format_lineage(
            'select a, b from t', reindent=True)
        newline = out.index('\n')
        indent_pos = newline + 1
        assert out[indent_pos] == ' '
        with pytest.raises(SQLParseError, match='line 2, column 1'):
            source_map.source_for(indent_pos)

    def test_ambiguous_range_across_segments_reports_position(self):
        sql = 'select foo, foo from t'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        first = out.index('foo')
        second = out.rindex('foo') + len('foo')
        with pytest.raises(SQLParseError, match='Ambiguous range'):
            source_map.source_for(first, second)

    def test_reversed_range_is_rejected(self):
        out, source_map = sqlparse.format_lineage('select a from t')
        with pytest.raises(SQLParseError, match='end .* precedes start'):
            source_map.source_for(5, 2)

    def test_line_col_helper(self):
        assert line_col('abc\ndef', 0) == (1, 1)
        assert line_col('abc\ndef', 5) == (2, 2)
        with pytest.raises(SQLParseError):
            line_col('abc', 9)


class TestVerbatimRegions:
    def test_inline_markers_are_consumed(self):
        sql = (f'select a, {DEFAULT_OPEN_MARKER}x + 1{DEFAULT_CLOSE_MARKER} '
               'from t')
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        assert DEFAULT_OPEN_MARKER not in out
        assert DEFAULT_CLOSE_MARKER not in out
        assert 'x + 1' in out
        assert len(source_map.regions) == 1

    def test_region_bytes_are_unchanged_under_keyword_case(self):
        sql = (f'select {DEFAULT_OPEN_MARKER}x + (select y from t2)'
               f'{DEFAULT_CLOSE_MARKER} from t1')
        out, _ = sqlparse.format_lineage(sql, keyword_case='upper')
        assert 'x + (select y from t2)' in out
        assert 'SELECT' in out

    def test_region_crosses_into_subquery_and_keeps_brackets(self):
        sql = (f'select a, {DEFAULT_OPEN_MARKER}x + (select y from t2 '
               f'where z = 1){DEFAULT_CLOSE_MARKER} as 金额 from t1')
        out, source_map = sqlparse.format_lineage(
            sql, reindent=True, keyword_case='upper')
        expected = 'x + (select y from t2 where z = 1)'
        assert expected in out
        out_start, out_end, src_start, src_end = source_map.regions[0]
        assert out[out_start:out_end] == expected
        assert sql[src_start:src_end] == expected

    def test_crlf_and_trailing_spaces_survive_inside_region(self):
        sql = (f'select a,\r\n {DEFAULT_OPEN_MARKER}b\r\n  + c   '
               f'{DEFAULT_CLOSE_MARKER}, d from t')
        out, _ = sqlparse.format_lineage(sql, reindent=True)
        assert 'b\r\n  + c   ' in out
        # Outside the region CRLF is converted to LF.
        assert '\r\n' not in out.split('b\r\n  + c   ')[1]

    def test_region_is_immune_to_identifier_and_string_filters(self):
        sql = (f'select {DEFAULT_OPEN_MARKER}lower_name{DEFAULT_CLOSE_MARKER} '
               "from t where x = 'keep me long enough'")
        out, _ = sqlparse.format_lineage(
            sql, identifier_case='upper', truncate_strings=5)
        assert 'lower_name' in out
        assert 'LOWER_NAME' not in out

    def test_explicit_ranges_are_preserved(self):
        sql = 'select a + 1 , b from t'
        out, source_map = sqlparse.format_lineage(
            sql, verbatim_ranges=[(7, 12)], reindent=True)
        out_start, out_end, src_start, src_end = source_map.regions[0]
        assert out[out_start:out_end] == 'a + 1'
        assert (src_start, src_end) == (7, 12)

    def test_custom_marker_pair(self):
        open_marker, close_marker = '<<VB>>', '<<ENDVB>>'
        sql = f'select {open_marker}a + 1{close_marker} from t'
        out, source_map = sqlparse.format_lineage(
            sql, verbatim_markers=(open_marker, close_marker),
            reindent=True)
        assert open_marker not in out
        assert 'a + 1' in out
        assert source_map.regions

    def test_markers_inside_strings_and_comments_are_literal(self):
        sql_string = f"select '{DEFAULT_OPEN_MARKER}' as x from t"
        out, source_map = sqlparse.format_lineage(
            sql_string, reindent=True)
        assert DEFAULT_OPEN_MARKER in out
        assert source_map.regions == []

        sql_comment = f'select a, /* {DEFAULT_OPEN_MARKER} */ b from t'
        out, source_map = sqlparse.format_lineage(
            sql_comment, reindent=True)
        assert DEFAULT_OPEN_MARKER in out
        assert source_map.regions == []

    def test_markers_inside_dollar_quote_are_literal(self):
        sql = f'select $body$ {DEFAULT_OPEN_MARKER} x $body$, y from t'
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        assert source_map.regions == []
        assert DEFAULT_OPEN_MARKER in out

    def test_region_with_comments_is_not_stripped(self):
        sql = (f'select {DEFAULT_OPEN_MARKER}a /* keep */ + 1'
               f'{DEFAULT_CLOSE_MARKER} from t')
        out, _ = sqlparse.format_lineage(sql, strip_comments=True)
        assert 'a /* keep */ + 1' in out

    def test_region_brackets_keep_statement_splitting(self):
        sql = (f'select {DEFAULT_OPEN_MARKER}f((1 + 2)){DEFAULT_CLOSE_MARKER} '
               'from t1; select 2')
        out, source_map = sqlparse.format_lineage(sql, reindent=True)
        assert out.count('select') + out.count('SELECT') == 2
        # Every region still maps back to a balanced source fragment.
        for _, _, src_start, src_end in source_map.regions:
            fragment = sql[src_start:src_end]
            assert fragment.count('(') == fragment.count(')')


class TestMarkerValidation:
    def test_unclosed_open_marker_reports_position(self):
        sql = f'select {DEFAULT_OPEN_MARKER} x from t'
        with pytest.raises(SQLParseError, match='Unclosed verbatim'):
            sqlparse.format_lineage(sql)

    def test_stray_close_marker_reports_position(self):
        sql = f'select a {DEFAULT_CLOSE_MARKER} from t'
        with pytest.raises(SQLParseError, match='Unmatched verbatim close'):
            sqlparse.format_lineage(sql)

    def test_empty_region_reports_position(self):
        sql = f'select {DEFAULT_OPEN_MARKER}{DEFAULT_CLOSE_MARKER} from t'
        with pytest.raises(SQLParseError, match='Empty verbatim region'):
            sqlparse.format_lineage(sql)

    def test_unbalanced_opening_bracket_reports_position(self):
        sql = (f'select {DEFAULT_OPEN_MARKER}(select 1{DEFAULT_CLOSE_MARKER} '
               'from t')
        with pytest.raises(SQLParseError, match="Unclosed '\\('"):
            sqlparse.format_lineage(sql)

    def test_unbalanced_closing_bracket_reports_position(self):
        sql = f'select {DEFAULT_OPEN_MARKER}1){DEFAULT_CLOSE_MARKER} from t'
        with pytest.raises(SQLParseError, match="Unbalanced '\\)'"):
            sqlparse.format_lineage(sql)

    def test_bracket_inside_string_does_not_count(self):
        sql = (f"select {DEFAULT_OPEN_MARKER}x || ')'{DEFAULT_CLOSE_MARKER} "
               'from t')
        directives = resolve_directives(sql)
        assert len(directives) == 1

    def test_overlapping_explicit_ranges_reported(self):
        with pytest.raises(SQLParseError, match='Overlapping verbatim'):
            resolve_directives('select a, b from t', ranges=[(7, 12), (10, 14)])

    def test_bad_range_shape_and_bounds(self):
        with pytest.raises(SQLParseError, match='must be a'):
            resolve_directives('select a', ranges=[(1,)])
        with pytest.raises(SQLParseError, match='out of bounds'):
            resolve_directives('select a', ranges=[(0, 99)])
        with pytest.raises(SQLParseError, match='integers'):
            resolve_directives('select a', ranges=[('0', '3')])

    def test_identical_markers_rejected(self):
        with pytest.raises(SQLParseError, match='must differ'):
            resolve_directives(
                'select a', markers=('@@X@@', '@@X@@'))

    def test_output_format_is_rejected(self):
        with pytest.raises(SQLParseError, match='output_format'):
            sqlparse.format_lineage('select a', output_format='python')


class TestLegacyCompatibility:
    def test_format_output_matches_without_lineage(self):
        sql = 'select a,  b from t where x = 1'
        assert sqlparse.format_lineage(sql)[0] == sqlparse.format(sql)

    def test_format_with_reindent_matches(self):
        sql = 'select a, b from t join u on t.id = u.id'
        assert (sqlparse.format_lineage(sql, reindent=True)[0]
                == sqlparse.format(sql, reindent=True))

    def test_plain_parse_and_split_unchanged(self):
        sql = 'select a; select b'
        assert len(sqlparse.parse(sql)) == 2
        assert len(sqlparse.split(sql)) == 2

    def test_source_map_repr(self):
        _, source_map = sqlparse.format_lineage('select a')
        assert repr(source_map).startswith('<SourceMap')
        assert isinstance(source_map, SourceMap)
