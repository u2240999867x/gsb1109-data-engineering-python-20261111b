"""Tests for quoting profiles, ingestion manifests and streaming."""

import io

import pytest

import sqlparse
from sqlparse import ingestion, lexer, profiles, quotes
from sqlparse import tokens as T

BACKSLASH_DUMP = (
    "INSERT INTO shop.orders (id, note) VALUES (1, 'a\\';b');\n"
    "INSERT INTO shop.orders (id, note) VALUES (2, 'plain');\n"
)

ANSI_DUMP = (
    "INSERT INTO shop.orders (id, note) VALUES (1, 'it''s open');\n"
    "INSERT INTO shop.orders (id, note) VALUES (2, 'plain');\n"
)


class TestProfiles:
    def test_get_profile_aliases(self):
        assert profiles.get_profile('sqlite') is profiles.ANSI
        assert profiles.get_profile('standard') is profiles.ANSI
        assert profiles.get_profile('MYSQL') is profiles.MYSQL
        assert profiles.get_profile(None) is profiles.DEFAULT
        custom = profiles.QuoteProfile('custom')
        assert profiles.get_profile(custom) is custom

    def test_unknown_profile(self):
        with pytest.raises(ValueError):
            profiles.get_profile('nope')

    def test_default_profile_is_legacy_behaviour(self):
        text = "select '\\'; select '\\''; select 1;"
        default = sqlparse.split(text)
        assert sqlparse.split(text, profile='default') == default
        assert sqlparse.split(text, profile=None) == default

    def test_ansi_and_backslash_differ(self):
        text = "select '\\'; select '\\'';"
        assert sqlparse.split(text, profile='ansi') == [
            "select '\\';", "select '\\'';"]
        assert sqlparse.split(text, profile='mysql') == [
            "select '\\'; select '\\'';"]

    def test_ansi_keeps_backslash_verbatim(self):
        [statement] = sqlparse.parse("SELECT 'a\\\\b'", profile='ansi')
        string = list(statement.flatten())[-1]
        assert string.ttype in T.String
        assert profiles.decode_string(string.value, 'ansi') == 'a\\\\b'

    def test_doubled_quotes(self):
        assert profiles.decode_string("'it''s'", 'ansi') == "it's"
        assert profiles.decode_string("'it''s'", 'mysql') == "it's"

    def test_backslash_escapes(self):
        assert profiles.decode_string("'a\\'b'", 'mysql') == "a'b"
        assert profiles.decode_string(r"'a\nb'", 'mysql') == 'a\nb'
        assert profiles.decode_string(r"'a\nb'", 'ansi') == r'a\nb'

    def test_decode_identifier(self):
        assert profiles.unquote_identifier('"a""b"') == 'a"b'
        assert profiles.unquote_identifier('`a``b`') == 'a`b'
        assert profiles.unquote_identifier('[a b]') == 'a b'

    def test_lexer_instances_are_independent(self):
        ansi = lexer.Lexer.get_profile_instance('ansi')
        mysql = lexer.Lexer.get_profile_instance('mysql')
        assert ansi is not mysql
        assert ansi.profile is profiles.ANSI
        assert mysql.profile is profiles.MYSQL
        default = lexer.Lexer.get_default_instance()
        assert default is not ansi

    def test_quote_state_copy_is_independent(self):
        state = quotes.QuoteState('ansi')
        state.feed("'abc")
        clone = state.copy()
        assert clone.in_string
        clone.feed("def'")
        assert not clone.in_string
        assert state.in_string
        state.feed("ghi'")
        assert not state.in_string

    def test_quote_state_double_close(self):
        state = quotes.QuoteState('ansi')
        state.feed("'a''b'")
        state.close_pending()
        assert not state.in_string
        state.feed("'c")
        assert state.in_string
        assert state.quote == "'"

    def test_quote_state_backslash_profile(self):
        state = quotes.QuoteState('mysql')
        # The escaped quote does not close the literal.
        state.feed("'a\\'")
        assert state.in_string
        state.feed("b'")
        assert not state.in_string

    def test_split_unquoted_newlines_profile(self):
        from sqlparse.utils import split_unquoted_newlines
        text = "'a\\'\nb'\nc"
        assert split_unquoted_newlines(text, 'ansi') == ["'a\\'", "b'\nc"]
        assert split_unquoted_newlines(text, 'mysql') == ["'a\\'\nb'", 'c']


class TestIngestManifest:
    def test_target_table_and_columns(self):
        manifest = ingestion.ingest(
            "INSERT INTO shop.orders (id, note) VALUES (1, 'first');",
            profile='ansi')
        assert manifest.row_list() == (
            ('shop.orders', (1, 'first')),
        )
        row = manifest.rows[0]
        assert row.schema == 'shop'
        assert row.table == 'shop.orders'
        assert [column.name for column in row.columns] == ['id', 'note']
        assert row.columns[1].raw == "'first'"

    def test_qualified_and_quoted_targets(self):
        manifest = ingestion.ingest(
            'INSERT INTO "shop"."orders" (id) VALUES (1);', profile='ansi')
        assert manifest.rows[0].table == 'shop.orders'
        manifest = ingestion.ingest(
            'INSERT INTO `shop`.`orders` (id) VALUES (1);', profile='mysql')
        assert manifest.rows[0].table == 'shop.orders'

    def test_raw_span_matches_source(self):
        text = "INSERT INTO t (a, b) VALUES ('x', 2);"
        manifest = ingestion.ingest(text, profile='ansi')
        for column in manifest.rows[0].columns:
            assert text[column.start:column.end] == column.raw
            assert column.value == (
                'x' if column.index == 0 else 2)
        first = manifest.rows[0].columns[0]
        assert (first.line, first.column) == (1, text.index("'x'") + 1)
        assert manifest.rows[0].columns[1].value == 2

    def test_multiline_raw_span(self):
        text = "INSERT INTO t (a) VALUES ('one\ntwo');"
        manifest = ingestion.ingest(text, profile='ansi')
        column = manifest.rows[0].columns[0]
        assert column.raw == "'one\ntwo'"
        assert column.value == 'one\ntwo'
        assert text[column.start:column.end] == column.raw

    def test_multiple_rows(self):
        text = ("INSERT INTO t (a) VALUES (1), (2), (3);\n"
                "INSERT INTO t (a) VALUES (4);")
        manifest = ingestion.ingest(text, profile='ansi')
        assert [row.values for row in manifest.rows] == [(1,), (2,), (3,), (4,)]
        assert [row.line for row in manifest.rows] == [1, 1, 1, 2]

    def test_residual_line_quarantined(self):
        text = ("INSERT INTO t (a) VALUES (1);\n"
                "INSERT INTO t (a) VALUES (1, 2);\n"
                "not sql\n")
        manifest = ingestion.ingest(text, profile='ansi')
        assert [row.values for row in manifest.rows] == [(1,)]
        reasons = [(line.line, line.reason) for line in manifest.residual]
        assert reasons == [
            (2, 'column_count_mismatch'),
            (3, 'unsupported_statement'),
        ]
        assert manifest.residual[0].text == "INSERT INTO t (a) VALUES (1, 2);"
        assert manifest.quarantined == manifest.residual

    def test_lexical_error_quarantined_under_profile(self):
        manifest = ingestion.ingest(BACKSLASH_DUMP, profile='ansi')
        assert manifest.rows == ()
        assert [item.reason for item in manifest.residual] == [
            'lexical_error', 'lexical_error']
        manifest = ingestion.ingest(BACKSLASH_DUMP, profile='mysql')
        assert [row.values for row in manifest.rows] == [
            (1, "a';b"), (2, 'plain')]
        assert manifest.residual == ()

    def test_to_dict_is_serialisable(self):
        import json
        manifest = ingestion.ingest(
            "INSERT INTO t (a) VALUES ('x');\ngarbage", profile='ansi')
        data = json.loads(json.dumps(manifest.to_dict()))
        assert data['profile'] == 'ansi'
        assert data['rows'][0]['values'] == ['x']
        assert data['residual'][0]['line'] == 2

    def test_stream_like_input(self):
        manifest = ingestion.ingest(
            io.StringIO("INSERT INTO t VALUES (1);"), profile='ansi')
        assert manifest.row_list() == (('t', (1,)),)


class TestConsistency:
    def test_split_then_ingest_matches_direct(self):
        for profile in ('ansi', 'mysql'):
            direct = ingestion.ingest(BACKSLASH_DUMP, profile=profile)
            pieces = sqlparse.split(BACKSLASH_DUMP, profile=profile)
            rows = []
            for piece in pieces:
                rows.extend(ingestion.ingest(piece, profile=profile).rows)
            assert tuple((row.table, row.values) for row in rows) \
                == direct.row_list()

    def test_format_then_ingest_matches_direct(self):
        for profile in ('ansi', 'mysql'):
            direct = ingestion.ingest(BACKSLASH_DUMP, profile=profile)
            formatted = sqlparse.format(BACKSLASH_DUMP, profile=profile)
            again = ingestion.ingest(formatted, profile=profile)
            assert again.row_list() == direct.row_list()

    def test_parse_uses_profile(self):
        assert len(sqlparse.parse(BACKSLASH_DUMP, profile='mysql')) == 2
        assert len(sqlparse.parse(BACKSLASH_DUMP, profile='ansi')) == 1
        statements_a = sqlparse.parse(
            "SELECT 'a\\';b'; SELECT 2;", profile='ansi')
        statements_b = sqlparse.parse(
            "SELECT 'a\\';b'; SELECT 2;", profile='mysql')
        # ANSI ends the literal at the backslash-quote, MySQL does not.
        assert len(statements_a) == 3
        assert len(statements_b) == 2


class TestBoundaryComparison:
    def test_identical_profiles(self):
        diff = ingestion.compare_profiles(BACKSLASH_DUMP, 'ansi', 'ansi')
        assert diff.same
        assert diff.first_differing_line is None

    def test_semicolon_moves_into_value(self):
        text = ("INSERT INTO t VALUES (1, 'a\\');\n"
                "INSERT INTO t VALUES (2, 'b');\n"
                "INSERT INTO t VALUES (3, 'c');\n")
        diff = ingestion.compare_profiles(text, 'ansi', 'mysql')
        assert diff.differs
        assert diff.first_differing_line == 1
        assert diff.statement_count_a == 3
        assert diff.statement_count_b == 1

    def test_semicolon_moves_out_of_value(self):
        text = ("INSERT INTO t VALUES ('a\\';x');\n"
                "INSERT INTO t VALUES ('b');\n")
        diff = ingestion.compare_profiles(text, 'ansi', 'mysql')
        assert diff.differs
        assert diff.first_differing_line == 1
        assert diff.statement_count_a == 1
        assert diff.statement_count_b == 2

    def test_boundaries_report_lines(self):
        boundaries = ingestion.statement_boundaries(
            "SELECT 1;\n\nSELECT 2;\n", profile='ansi')
        assert [(item.line, item.end_line) for item in boundaries] == [
            (1, 1), (3, 3)]

    def test_divergence_on_later_line(self):
        text = ("INSERT INTO t VALUES (1);\n"
                "INSERT INTO t VALUES (2, 'a\\');\n"
                "INSERT INTO t VALUES (3, 'b');\n")
        diff = ingestion.compare_profiles(text, 'ansi', 'mysql')
        assert diff.first_differing_line == 2
        assert diff.statement_count_a == 3
        assert diff.statement_count_b == 2

    def test_divergence_is_symmetric(self):
        text = ("INSERT INTO t VALUES (1, 'a\\');\n"
                "INSERT INTO t VALUES (2);\n")
        forward = ingestion.compare_profiles(text, 'ansi', 'mysql')
        reverse = ingestion.compare_profiles(text, 'mysql', 'ansi')
        assert forward.first_differing_line == reverse.first_differing_line
        assert forward.statement_count_a == reverse.statement_count_b


class TestInterleavedStreams:
    STREAM_SAMPLES = {
        'simple': (
            "INSERT INTO t VALUES ('a', 1);\n"
            "INSERT INTO s.b VALUES (2, 'c');\n"
            "INSERT INTO t VALUES (3, 'd;e');\n"),
        'doubled': (
            "INSERT INTO t VALUES ('it''s; ok', 1);\n"
            "INSERT INTO t VALUES (2, 'x');\n"),
        'comments': (
            "-- head; comment\n"
            "INSERT INTO t VALUES (1); /* mid; */ INSERT INTO t VALUES (2);\n"),
        'dollar': (
            "CREATE FUNCTION f() RETURNS int AS $$ BEGIN x; "
            "RETURN 1; END; $$ LANGUAGE plpgsql;\n"
            "INSERT INTO t VALUES (1);\n"),
        'dollar_tag': (
            "DO $body$ INSERT INTO t VALUES (1); $body$;\n"
            "INSERT INTO t VALUES (2);\n"),
        'block': "BEGIN;\nINSERT INTO t VALUES (1);\nEND;\n",
        'escaped': (
            "INSERT INTO t VALUES ('a\\';b', 1);\n"
            "INSERT INTO t VALUES ('c', 2);\n"),
        'multiline': (
            "INSERT INTO t (a, b) VALUES ('one\ntwo', 1);\n"
            "INSERT INTO t (a, b) VALUES ('x', 2);\n"),
        'backtick': (
            "INSERT INTO `we;ird` VALUES (1);\n"
            "INSERT INTO t VALUES (2);\n"),
        'spaces': (
            "INSERT INTO t VALUES (1);   \n"
            "INSERT INTO t VALUES (2);\t\n"),
    }

    @pytest.mark.parametrize('profile', ['ansi', 'mysql'])
    @pytest.mark.parametrize('sample', sorted(STREAM_SAMPLES))
    def test_stream_matches_direct_for_every_chunk_cut(self, profile, sample):
        text = self.STREAM_SAMPLES[sample]
        direct = ingestion.ingest(text, profile=profile)
        for cut in range(1, len(text)):
            stream = ingestion.ingest_stream(profile=profile)
            stream.feed(text[:cut])
            stream.feed(text[cut:])
            stream.finish()
            manifest = stream.manifest()
            assert manifest.rows == direct.rows
            assert manifest.residual == direct.residual
            assert manifest.statements == direct.statements

    def test_two_streams_do_not_share_quote_state(self):
        left = ingestion.ingest_stream(profile='ansi', source='left')
        right = ingestion.ingest_stream(profile='mysql', source='right')
        left.feed("INSERT INTO l VALUES ('ab")
        assert left._quote.in_string
        right.feed("INSERT INTO r VALUES ('x\\'', 9);")
        assert right.manifest().row_list() == (('r', ("x'", 9)),)
        left.feed("cd', 1);")
        left.finish()
        assert left.manifest().row_list() == (('l', ('abcd', 1)),)
        assert left.manifest().source == 'left'
        assert right.manifest().source == 'right'

    def test_interleaved_by_character(self):
        left_text = "INSERT INTO a VALUES ('one;A', 1);\n"
        right_text = "INSERT INTO b VALUES ('x\\';y', 2);\n"
        left = ingestion.ingest_stream(profile='ansi')
        right = ingestion.ingest_stream(profile='mysql')
        for index in range(max(len(left_text), len(right_text))):
            if index < len(left_text):
                left.feed(left_text[index])
            if index < len(right_text):
                right.feed(right_text[index])
        left.finish()
        right.finish()
        assert left.manifest().row_list() == (('a', ('one;A', 1)),)
        assert right.manifest().row_list() == (('b', ("x';y", 2)),)

    def test_stream_handles_blocks_and_dollar_quotes(self):
        text = ("CREATE FUNCTION f() RETURNS int AS $$ BEGIN x; "
                "RETURN 1; END; $$ LANGUAGE plpgsql;\n"
                "BEGIN;\n"
                "INSERT INTO t VALUES (1);\n"
                "END;\n")
        stream = ingestion.ingest_stream(profile='ansi')
        rows = []
        for char in text:
            rows.extend(stream.feed(char))
        rows.extend(stream.finish())
        assert [row.values for row in rows] == [(1,)]
        assert stream.manifest().row_list() == (('t', (1,)),)

    def test_finished_stream_rejects_feed(self):
        stream = ingestion.ingest_stream(profile='ansi')
        stream.finish()
        with pytest.raises(ValueError):
            stream.feed('SELECT 1')

    def test_feed_returns_completed_rows(self):
        stream = ingestion.ingest_stream(profile='ansi')
        assert [row.values for row in
                stream.feed("INSERT INTO t VALUES (1);")] == [(1,)]
        assert [row.values for row in
                stream.feed("INSERT INTO t VALUES (2);")] == [(2,)]
        assert stream.finish() == []

    def test_finish_flushes_unterminated_statement(self):
        stream = ingestion.ingest_stream(profile='ansi')
        assert stream.feed("INSERT INTO t VALUES (1);\nINSERT INTO t VALUES (2)") \
            == [stream.manifest().rows[0]]
        rows = stream.finish()
        assert [row.values for row in rows] == [(2,)]
        assert stream.manifest().row_list() == (('t', (1,)), ('t', (2,)))

    def test_finish_returns_only_rows_not_yet_reported(self):
        stream = ingestion.ingest_stream(profile='ansi')
        emitted = stream.feed("INSERT INTO t VALUES (1);\n")
        emitted += stream.feed("INSERT INTO t VALUES (2)")
        emitted += stream.finish()
        assert [row.values for row in emitted] == [(1,), (2,)]
        assert len(stream.manifest().rows) == 2

    def test_chunked_whitespace_after_semicolon(self):
        text = "INSERT INTO t VALUES (1);   \nINSERT INTO t VALUES (2);\n"
        direct = ingestion.ingest(text, profile='ansi')
        stream = ingestion.ingest_stream(profile='ansi')
        position = 0
        for size in (24, 3, 3, 3):
            stream.feed(text[position:position + size])
            position += size
        stream.feed(text[position:])
        stream.finish()
        assert stream.manifest().statements == direct.statements
        assert stream.manifest().rows == direct.rows

    def test_declarations_round_trip(self):
        manifest = ingestion.ingest(
            "INSERT INTO t (a, b) VALUES (1, 'x');", profile='ansi')
        row = manifest.rows[0]
        assert isinstance(row, ingestion.IngestedRow)
        assert isinstance(row.columns[0], ingestion.ColumnSpan)
        assert isinstance(
            ingestion.ingest('garbage').residual[0], ingestion.ResidualLine)
