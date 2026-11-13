import pytest

from sqlite_utils import Database
from sqlite_utils.db import DescIndex
from sqlite_utils.index_slimming import parse_predicate


@pytest.fixture
def features_db():
    db = Database(memory=True)
    db["features"].create(
        {
            "entity_id": int,
            "sample_date": str,
            "batch_id": int,
            "feature": str,
            "value": float,
        }
    )
    db["features"].create_index(["entity_id"], index_name="idx_entity")
    db["features"].create_index(
        ["entity_id", "sample_date"], index_name="idx_entity_date"
    )
    db["features"].create_index(
        ["entity_id", "batch_id", "sample_date"],
        index_name="idx_dedupe",
        unique=True,
    )
    db["features"].create_index(["value"], index_name="idx_value")
    return db


SAMPLING_QUERY = {
    "name": "sample_by_entity_date",
    "equality": ["entity_id"],
    "range": "sample_date",
    "order_by": [("sample_date", "asc")],
}

DEDUPE_RULE = {
    "name": "no_duplicate_samples",
    "columns": ["entity_id", "batch_id", "sample_date"],
}


def test_minimal_retained_set(features_db):
    preview = features_db["features"].index_slimming_preview(
        queries=[SAMPLING_QUERY], uniques=[DEDUPE_RULE]
    )
    kept = {entry["index"] for entry in preview["keep"]}
    dropped = {entry["index"] for entry in preview["drop"]}
    assert kept == {"idx_entity_date", "idx_dedupe"}
    assert dropped == {"idx_entity", "idx_value"}
    assert preview["unknown"] == []
    assert preview["uncovered"] == []


def test_dropped_indexes_have_successors(features_db):
    preview = features_db["features"].index_slimming_preview(
        queries=[SAMPLING_QUERY], uniques=[DEDUPE_RULE]
    )
    drop = {entry["index"]: entry for entry in preview["drop"]}
    # idx_entity is a left-prefix of idx_entity_date
    assert drop["idx_entity"]["successor"] == "idx_entity_date"
    # idx_value serves nothing in the declared workload
    assert drop["idx_value"]["successor"] is None
    assert "not used" in drop["idx_value"]["reason"]


def test_kept_indexes_explain_what_would_be_lost(features_db):
    preview = features_db["features"].index_slimming_preview(
        queries=[SAMPLING_QUERY], uniques=[DEDUPE_RULE]
    )
    keep = {entry["index"]: entry for entry in preview["keep"]}
    assert keep["idx_entity_date"]["covers"] == ["sample_by_entity_date"]
    assert keep["idx_entity_date"]["if_dropped"]["lost_orders"] == [
        "entity_id = ? AND sample_date BETWEEN ? AND ? ORDER BY sample_date ASC"
    ]
    assert keep["idx_entity_date"]["if_dropped"]["writable_rows"] == []
    assert keep["idx_dedupe"]["covers"] == ["no_duplicate_samples"]
    assert keep["idx_dedupe"]["if_dropped"]["lost_orders"] == []
    rows = keep["idx_dedupe"]["if_dropped"]["writable_rows"]
    assert len(rows) == 2
    assert rows[0] == rows[1]
    assert set(rows[0]) == {"entity_id", "batch_id", "sample_date"}


def test_desc_index_covers_both_directions():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str})
    db["t"].create_index(["a", DescIndex("b")], index_name="idx_a_b_desc")
    for direction in ("asc", "desc"):
        preview = db["t"].index_slimming_preview(
            queries=[
                {
                    "name": "q",
                    "equality": ["a"],
                    "range": "b",
                    "order_by": [("b", direction)],
                }
            ]
        )
        assert preview["uncovered"] == []
        assert [e["index"] for e in preview["keep"]] == ["idx_a_b_desc"]


def test_mixed_direction_order_not_covered():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str, "c": str})
    db["t"].create_index(["a", "b", "c"], index_name="idx_a_b_c")
    preview = db["t"].index_slimming_preview(
        queries=[
            {
                "name": "q",
                "equality": ["a"],
                "order_by": [("b", "asc"), ("c", "desc")],
            }
        ]
    )
    assert [e["name"] for e in preview["uncovered"]] == ["q"]


def test_collation_must_match():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str})
    db.execute("create index idx_nocase on t(a, b collate nocase)")
    # Query declaring the nocase collation is covered
    preview = db["t"].index_slimming_preview(
        queries=[
            {
                "name": "q",
                "equality": ["a"],
                "range": "b",
                "order_by": ["b"],
                "collate": {"b": "nocase"},
            }
        ]
    )
    assert preview["uncovered"] == []
    # Same shape without the collation cannot use that index for ordering
    preview = db["t"].index_slimming_preview(
        queries=[
            {
                "name": "q",
                "equality": ["a"],
                "range": "b",
                "order_by": ["b"],
            }
        ]
    )
    assert [e["name"] for e in preview["uncovered"]] == ["q"]


def test_partial_index_with_constant_predicate():
    db = Database(memory=True)
    db["features"].create({"entity_id": int, "sample_date": str, "kind": str})
    db.execute(
        "create index idx_train on features(entity_id, sample_date) "
        "where kind = 'train'"
    )
    query = {
        "name": "q",
        "equality": {"entity_id": None, "kind": "train"},
        "range": "sample_date",
        "order_by": ["sample_date"],
    }
    preview = db["features"].index_slimming_preview(queries=[query])
    assert preview["uncovered"] == []
    assert preview["keep"][0]["predicate"] == {"kind": "train"}
    # A query that does not pin kind = 'train' cannot use the partial index
    query_without_constant = {
        "name": "q",
        "equality": ["entity_id"],
        "range": "sample_date",
        "order_by": ["sample_date"],
    }
    preview = db["features"].index_slimming_preview(
        queries=[query_without_constant]
    )
    assert [e["name"] for e in preview["uncovered"]] == ["q"]


def test_unknown_predicate_is_kept_but_not_a_successor():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int})
    db.execute("create index idx_weird on t(a) where b > 10")
    db["t"].create_index(["a", "b"], index_name="idx_a_b")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"], "order_by": ["b"]}]
    )
    assert preview["unknown"][0]["index"] == "idx_weird"
    assert preview["unknown"][0]["kept"] is True
    assert "b > 10" in preview["unknown"][0]["reason"]
    # The unknown index cannot be suggested for removal...
    assert "idx_weird" not in {e["index"] for e in preview["drop"]}
    # ...and cannot act as a successor either
    assert all(
        entry["successor"] != "idx_weird" for entry in preview["drop"]
    )


def test_expression_index_is_unknown():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str})
    db.execute("create index idx_expr on t(lower(b))")
    preview = db["t"].index_slimming_preview(queries=[])
    assert preview["unknown"][0]["index"] == "idx_expr"
    assert "expression" in preview["unknown"][0]["reason"]


def test_unique_rule_requires_exact_column_set():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int, "c": int})
    db["t"].create_index(["a", "b", "c"], index_name="idx_abc", unique=True)
    preview = db["t"].index_slimming_preview(
        uniques=[{"name": "u", "columns": ["a", "b"]}]
    )
    # unique (a, b, c) does not prevent duplicate (a, b) pairs
    assert [e["name"] for e in preview["uncovered"]] == ["u"]


def test_partial_unique_index():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int, "kind": str})
    db.execute(
        "create unique index idx_partial_unique on t(a, b) where kind = 'x'"
    )
    # Matches a rule declared with the same predicate constants
    preview = db["t"].index_slimming_preview(
        uniques=[{"name": "u", "columns": ["a", "b"], "where": {"kind": "x"}}]
    )
    assert preview["uncovered"] == []
    assert [e["index"] for e in preview["keep"]] == ["idx_partial_unique"]
    # ...but not a global rule
    preview = db["t"].index_slimming_preview(
        uniques=[{"name": "u", "columns": ["a", "b"]}]
    )
    assert [e["name"] for e in preview["uncovered"]] == ["u"]


def test_duplicate_unique_index_gets_unique_successor():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int})
    db["t"].create_index(["a", "b"], index_name="idx_u1", unique=True)
    db["t"].create_index(["a", "b"], index_name="idx_u2", unique=True)
    preview = db["t"].index_slimming_preview(
        uniques=[{"name": "u", "columns": ["a", "b"]}]
    )
    assert len(preview["keep"]) == 1
    assert len(preview["drop"]) == 1
    dropped = preview["drop"][0]
    kept = preview["keep"][0]
    assert dropped["successor"] == kept["index"]
    assert dropped["index"] != kept["index"]


def test_primary_key_index_is_never_dropped():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str}, pk="a")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"]}]
    )
    assert preview["drop"] == []
    # INTEGER PRIMARY KEY is a rowid alias - no index to drop
    keep = preview["keep"][0]
    assert keep["index"] == "rowid"
    assert keep["origin"] == "pk"
    assert "cannot be dropped" in keep["reason"]


def test_text_primary_key_autoindex_is_never_dropped():
    db = Database(memory=True)
    db["t"].create({"a": str, "b": str}, pk="a")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"]}]
    )
    assert preview["drop"] == []
    keep = preview["keep"][0]
    assert keep["origin"] == "pk"
    assert "cannot be dropped" in keep["reason"]


def test_uncovered_query_is_reported():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int})
    db["t"].create_index(["a"], index_name="idx_a")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"], "order_by": ["b"]}]
    )
    assert preview["uncovered"][0]["type"] == "query"
    assert preview["uncovered"][0]["name"] == "q"
    assert "no existing index provides" in preview["uncovered"][0]["detail"]


def test_order_by_column_index_avoids_sort():
    # An index on the ORDER BY column alone lets SQLite scan in order
    # and apply the equality filter as it goes
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int})
    db["t"].create_index(["b"], index_name="idx_b")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"], "order_by": ["b"]}]
    )
    assert preview["uncovered"] == []
    assert [e["index"] for e in preview["keep"]] == ["idx_b"]


def test_equality_constant_in_query_description():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": str})
    db["t"].create_index(["a", "b"], index_name="idx_a_b")
    preview = db["t"].index_slimming_preview(
        queries=[
            {
                "name": "q",
                "equality": {"a": 5},
                "order_by": [{"column": "b", "direction": "desc"}],
            }
        ]
    )
    assert preview["keep"][0]["if_dropped"]["lost_orders"] == [
        "a = 5 ORDER BY b DESC"
    ]


def test_order_by_on_equality_column_is_a_noop():
    db = Database(memory=True)
    db["t"].create({"a": int, "b": int})
    db["t"].create_index(["a", "b"], index_name="idx_a_b")
    preview = db["t"].index_slimming_preview(
        queries=[{"name": "q", "equality": ["a"], "order_by": ["a", "b"]}]
    )
    assert preview["uncovered"] == []


def test_invalid_direction_rejected():
    db = Database(memory=True)
    db["t"].create({"a": int})
    with pytest.raises(ValueError):
        db["t"].index_slimming_preview(
            queries=[{"equality": ["a"], "order_by": [("a", "sideways")]}]
        )


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("create index i on t(a) where b = 1", {"b": 1}),
        ("create index i on t(a) where b = 'x' and c = 2.5", {"b": "x", "c": 2.5}),
        ("create index i on t(a) where b = 'it''s'", {"b": "it's"}),
        ("create index i on t(a) where 1 = b", {"b": 1}),
        ("create index i on t(a) where t.b = 'y'", {"b": "y"}),
        ('create index i on t(a) where "weird col" = 3', {"weird col": 3}),
        ("create index i on t(a) where b = 'x' AND (c = 1)", {"b": "x", "c": 1}),
        ("create index i on t(a) where b > 1", None),
        ("create index i on t(a) where b = 1 or c = 2", None),
        ("create index i on t(a) where b = c", None),
        ("create index i on t(a) where b = null", None),
        ("create index i on t(a) where lower(b) = 'x'", None),
        ("create index i on t(a)", None),
    ],
)
def test_parse_predicate(sql, expected):
    assert parse_predicate(sql) == expected
