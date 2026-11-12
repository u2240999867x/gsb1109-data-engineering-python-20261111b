import pytest

from sqlparse import versioned
from sqlparse.versioned import (
    OK,
    UNDETERMINED,
    VersionView,
    format_version,
    normalize_version,
)


def test_normalize_version():
    assert normalize_version(50744) == 50744
    assert normalize_version("5.7.44") == 50744
    assert normalize_version("8.0") == 80000
    assert normalize_version((5, 7, 44)) == 50744
    assert format_version(50744) == "5.7.44"


def test_render_strips_plain_comments():
    view = VersionView(
        "SELECT a, -- trailing\n"
        "b /* block */ FROM t # hash\n"
    )
    sql = view.render("5.7.44").sql
    assert "--" not in sql
    assert "/* block */" not in sql
    assert "#" not in sql
    assert "SELECT a," in sql
    assert "b" in sql and "FROM t" in sql


def test_render_keeps_optimizer_hints_in_position():
    view = VersionView("SELECT /*+ MAX_EXECUTION_TIME(1000) */ a FROM t")
    sql = view.render(50744).sql
    assert sql.index("/*+ MAX_EXECUTION_TIME(1000) */") < sql.index("a FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("a",)


def test_render_conditional_version_gate():
    view = VersionView("SELECT a /*!50744 , b */ FROM t")
    assert ", b" not in view.render(50743).sql
    assert ", b" in view.render(50744).sql
    assert ", b" in view.render(80000).sql
    # no target version: versioned conditionals stay comments
    assert ", b" not in view.render().sql


def test_render_unversioned_conditional_always_inline():
    view = VersionView("SELECT a /*! , b */ FROM t")
    assert ", b" in view.render().sql


def test_conditional_may_close_statement():
    # enabled content may contain a semicolon: it is spliced before
    # parsing, not treated as an independent statement upfront
    view = VersionView(
        "/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE */;\n"
        "SELECT a FROM t"
    )
    result = view.projection(40101)
    assert result.status == OK
    assert result.names == ("a",)
    result = view.projection(40100)
    assert result.status == OK
    assert result.names == ("a",)


def test_projection_simple():
    view = VersionView("SELECT a, b AS bee, t.c FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("a", "bee", "c")
    assert result.fields[1].alias == "bee"


def test_projection_wildcard():
    view = VersionView("SELECT * FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("*",)


def test_projection_conditional_comma():
    view = VersionView("SELECT a /*!50744 , b */ FROM t")
    assert view.projection(50743).names == ("a",)
    assert view.projection(50744).names == ("a", "b")


def test_projection_conditional_alias():
    view = VersionView("SELECT a /*!50744 AS x */ FROM t")
    assert view.projection(50743).names == ("a",)
    assert view.projection(50744).names == ("x",)


def test_projection_conditional_half_expression():
    view = VersionView("SELECT a /*!50744 + b */ FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("a + b",)
    assert view.projection(50743).names == ("a",)


def test_projection_maps_back_to_original_offsets():
    sql = "SELECT a, /* note */ b AS bee /*!50744 , c */ FROM t"
    view = VersionView(sql)
    result = view.projection(50744)
    assert result.status == OK
    for field in result.fields:
        snippet = sql[field.orig_start:field.orig_end]
        assert field.name.split(".")[-1] in snippet or "c */" in snippet
    # the conditional field maps to its exact characters, which live
    # inside the conditional comment of the original text
    conditional = result.fields[2]
    assert sql[conditional.orig_start:conditional.orig_end] == "c"
    comment_start = sql.index("/*!50744")
    comment_end = sql.index("*/", comment_start)
    assert comment_start < conditional.orig_start < comment_end
    plain = result.fields[0]
    assert sql[plain.orig_start:plain.orig_end] == "a"


def test_projection_byte_ranges():
    sql = "SELECT 'é', ä_col FROM t"
    view = VersionView(sql)
    result = view.projection(50744)
    assert result.status == OK
    raw = sql.encode("utf-8")
    for field in result.fields:
        assert raw[field.byte_start:field.byte_end] == \
            sql[field.orig_start:field.orig_end].encode("utf-8")
    field = result.fields[1]
    assert sql[field.orig_start:field.orig_end] == "ä_col"
    # byte range is wider than the character range for non-ASCII
    assert field.byte_end - field.byte_start > \
        field.orig_end - field.orig_start


def test_comment_openers_inside_strings_are_ignored():
    view = VersionView("SELECT '/* not a comment */' AS x, a FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("x", "a")


def test_undetermined_for_non_select():
    view = VersionView("INSERT INTO t VALUES (1)")
    result = view.projection(50744)
    assert result.status == UNDETERMINED
    assert result.reason


def test_undetermined_for_multiple_selects():
    view = VersionView("SELECT a FROM t; SELECT b FROM u")
    assert view.projection(50744).status == UNDETERMINED


def test_undetermined_without_select():
    view = VersionView("SET sql_mode = 'ansi'")
    assert view.projection(50744).status == UNDETERMINED


def test_ancillary_set_statement_is_allowed():
    view = VersionView("SET sql_mode = 'ansi'; SELECT a FROM t")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("a",)


def test_projection_with_cte():
    view = VersionView("WITH x AS (SELECT 1) SELECT a, b FROM x")
    result = view.projection(50744)
    assert result.status == OK
    assert result.names == ("a", "b")


def test_conditional_versions_property():
    view = VersionView("SELECT a /*!50744 , b */ /*!80000 , c */ FROM t")
    assert view.conditional_versions == [50744, 80000]


def test_projection_segments_single_change():
    view = VersionView("SELECT a /*!50744 , b */ FROM t")
    report = view.projection_segments("5.7.0", "8.0.0")
    assert [(s.start, s.end) for s in report.segments] == [
        (50700, 50743), (50744, 80000)]
    assert report.segments[0].names == ("a",)
    assert report.segments[1].names == ("a", "b")
    assert len(report.changes) == 1
    change = report.changes[0]
    assert change.version == 50744
    assert change.added == ("b",)
    assert change.removed == ()
    assert not change.undetermined


def test_projection_segments_multiple_thresholds():
    view = VersionView("SELECT a /*!50744 , b */ /*!80000 , c */ FROM t")
    report = view.projection_segments(50000, 90000)
    assert [s.names for s in report.segments] == [
        ("a",), ("a", "b"), ("a", "b", "c")]
    assert [(c.version, c.added) for c in report.changes] == [
        (50744, ("b",)), (80000, ("c",))]


def test_projection_segments_constant_outside_thresholds():
    view = VersionView("SELECT a /*!50744 , b */ FROM t")
    report = view.projection_segments(60000, 90000)
    assert len(report.segments) == 1
    assert report.segments[0].names == ("a", "b")
    assert report.changes == ()


def test_projection_segments_undetermined_side():
    # enabling the conditional turns the script into two SELECTs,
    # which is beyond the declared syntax
    view = VersionView("SELECT a FROM t /*!50744 ; SELECT b FROM u */")
    report = view.projection_segments(50700, 50800)
    assert report.segments[0].result.status == OK
    assert report.segments[1].result.status == UNDETERMINED
    assert len(report.changes) == 1
    assert report.changes[0].undetermined


def test_projection_segments_invalid_range():
    view = VersionView("SELECT a FROM t")
    with pytest.raises(ValueError):
        view.projection_segments(50800, 50700)


def test_public_api_exposed():
    assert versioned.VersionView is VersionView
