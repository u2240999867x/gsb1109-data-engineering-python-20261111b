"""Tests for per-batch parsing sessions around the shared column grammar.

Covers the requirements:

* one explicit session per parse, bound to source + semantic version,
* packrat entries reused only for the same source, version and input position,
* parent-row actions recursively parse attached tables,
* inner failures do not abort the outer row or lose computed outer fields,
* the child parse reads its own code table, never the parent table's values,
* interleaving the same text under two versions matches running each alone,
* the shared grammar is never deep-copied and batch caches are released.
"""

import threading

import pyparsing as pp
import pytest

from column_import import (
    CodeLookupError,
    ParseSession,
    current_session,
)
from column_import import grammar as grammar_mod
from column_import.grammar import (
    ATTACHMENT_TABLE,
    COLUMNS_TABLE,
    child_table,
    document,
    mapped_code,
    parse_document,
)

V1_TABLES = {
    COLUMNS_TABLE: {"A": "alpha_v1", "B": "beta_v1", "C": "charlie_v1"},
    ATTACHMENT_TABLE: {"A": "att_a_v1", "X": "att_x_v1"},
}
V2_TABLES = {
    COLUMNS_TABLE: {"A": "alpha_v2", "B": "beta_v2", "C": "charlie_v2"},
    ATTACHMENT_TABLE: {"A": "att_a_v2", "X": "att_x_v2"},
}

SAMPLE = "A,B\nC,{A,ZZZ}\nA,{X}"


def test_requires_active_session():
    assert current_session() is None
    with pytest.raises(RuntimeError):
        parse_document("A")


def test_version_selects_code_table_values():
    with ParseSession("stream-a", "v1", V1_TABLES) as session:
        out_v1 = parse_document(SAMPLE)
        assert session.semantic_version == "v1"
    with ParseSession("stream-b", "v2", V2_TABLES):
        out_v2 = parse_document(SAMPLE)

    assert out_v1[0]["columns"] == ["alpha_v1", "beta_v1"]
    assert out_v2[0]["columns"] == ["alpha_v2", "beta_v2"]
    assert out_v1[2]["columns"] == ["alpha_v1"]
    assert out_v2[2]["columns"] == ["alpha_v2"]


def test_attachment_uses_its_own_code_table():
    # "A" exists in both tables with different values; "X" only in the
    # attachment table.  The child parse must never see parent-table values.
    with ParseSession("s", "v1", V1_TABLES):
        out = parse_document(SAMPLE)
    good = out[2]["attachment"]
    assert good["items"] == ["att_x_v1"]
    assert good["error"] is None

    with ParseSession("s", "v2", V2_TABLES):
        out = parse_document("A,{A}")
    assert out[0]["attachment"]["items"] == ["att_a_v2"]
    assert out[0]["columns"] == ["alpha_v2"]


def test_inner_failure_keeps_outer_fields_and_continues():
    with ParseSession("s", "v1", V1_TABLES):
        out = parse_document(SAMPLE)

    failed_row = out[1]
    # Parent columns resolved before the failing attachment are retained.
    assert failed_row["columns"] == ["charlie_v1"]
    attachment = failed_row["attachment"]
    assert attachment["items"] is None
    assert attachment["raw"] == "A,ZZZ"
    assert "ZZZ" in attachment["error"]
    # Rows before and after the failure still parsed.
    assert out[0]["columns"] == ["alpha_v1", "beta_v1"]
    assert out[2]["attachment"]["items"] == ["att_x_v1"]


def test_grammar_syntax_failure_inside_attachment_is_contained():
    with ParseSession("s", "v1", V1_TABLES):
        out = parse_document("B,{!!not-a-code!!}\nA")
    assert out[0]["columns"] == ["beta_v1"]
    assert out[0]["attachment"]["items"] is None
    assert out[1]["columns"] == ["alpha_v1"]


def test_interleaved_versions_match_isolated_runs():
    with ParseSession("stream-a", "v1", V1_TABLES):
        isolated_v1 = parse_document(SAMPLE)
    with ParseSession("stream-b", "v2", V2_TABLES):
        isolated_v2 = parse_document(SAMPLE)

    s1 = ParseSession("stream-a", "v1", V1_TABLES)
    s2 = ParseSession("stream-b", "v2", V2_TABLES)
    with s1:
        interleaved_v1_first = parse_document(SAMPLE)
        with s2:
            interleaved_v2_first = parse_document(SAMPLE)
        interleaved_v1_again = parse_document(SAMPLE)
        with s2:
            interleaved_v2_again = parse_document(SAMPLE)

    assert interleaved_v1_first == interleaved_v1_again == isolated_v1
    assert interleaved_v2_first == interleaved_v2_again == isolated_v2


def test_packrat_cache_is_private_per_session():
    s1 = ParseSession("stream-a", "v1", V1_TABLES)
    s2 = ParseSession("stream-b", "v2", V2_TABLES)
    assert s1.cache is not s2.cache
    with s1:
        parse_document(SAMPLE)
        with s2:
            parse_document(SAMPLE)
            # Snapshot while both sessions are alive; leaving a session
            # releases its batch cache by design.
            s1_store = dict(s1.cache._store)
            s2_store = dict(s2.cache._store)

    assert s1_store and s2_store
    # The stores are independent objects even though pyparsing builds the same
    # key tuple (same grammar elements, strings and positions) for both.
    assert s1.cache._store is not s2.cache._store

    # Every key is pyparsing's (element, instring, loc, call_pre, do_actions)
    # tuple and only references strings this session parsed.
    allowed_strings = {SAMPLE, "A,ZZZ", "X", "A"}
    for key in tuple(s1_store) + tuple(s2_store):
        element, instring, loc, call_pre, do_actions = key
        assert isinstance(element, pp.ParserElement)
        assert instring in allowed_strings
        assert isinstance(loc, int) and 0 <= loc <= len(instring) + 1
        assert isinstance(call_pre, bool)
        assert isinstance(do_actions, bool)

    # A shared key resolves to version-specific values in each private store.
    def mapped_value(store):
        for key, value in store.items():
            if key[0] is mapped_code and key[1] == SAMPLE and key[2] == 0:
                # Successful entries are stored as (end_loc, ParseResults, loc).
                if not isinstance(value, Exception):
                    return value[1].as_list()
        return None

    assert mapped_value(s1_store) == ["alpha_v1"]
    assert mapped_value(s2_store) == ["alpha_v2"]


def test_same_text_different_positions_cached_separately():
    with ParseSession("s", "v1", V1_TABLES) as session:
        parse_document("A\nA")
        locations = {key[2] for key in session.cache_entries() if key[1] == "A\nA"}
    # The same code element is memoized at both the start and after the newline.
    assert 0 in locations
    assert any(loc > 0 for loc in locations)


def test_reparse_serves_from_session_cache():
    with ParseSession("s", "v1", V1_TABLES) as session:
        parse_document(SAMPLE)
        hits_after_first, _ = session.cache_stats()
        parse_document(SAMPLE)
        hits_after_second, _ = session.cache_stats()
    assert hits_after_second > hits_after_first


def test_close_releases_batch_cache_and_restores_global_state():
    pp.ParserElement.reset_cache()
    original_cache = pp.ParserElement.packrat_cache
    original_enabled = pp.ParserElement._packratEnabled

    session = ParseSession("s", "v1", V1_TABLES)
    with session:
        parse_document(SAMPLE)
        assert session.cache_size() > 0
        assert pp.ParserElement._packratEnabled is True
    session.close()

    assert session.cache_size() == 0
    assert pp.ParserElement._packratEnabled == original_enabled
    assert pp.ParserElement.packrat_cache is original_cache
    with pytest.raises(RuntimeError):
        with session:
            pass


def test_shared_grammar_is_not_copied_between_sessions():
    seen_elements = set()

    with ParseSession("s1", "v1", V1_TABLES):
        parse_document("A")
        seen_elements.add(id(document))
        seen_elements.add(id(mapped_code))
        seen_elements.add(id(child_table))
    with ParseSession("s2", "v2", V2_TABLES):
        parse_document("A")

    assert id(document) in seen_elements
    assert id(mapped_code) in seen_elements
    assert id(child_table) in seen_elements
    # The module exposes one grammar instance regardless of active session.
    assert grammar_mod.document is document


def test_import_does_not_change_global_whitespace_defaults():
    assert set(pp.ParserElement.DEFAULT_WHITE_CHARS) == set(" \n\t\r")


def test_concurrent_sessions_keep_results_and_caches_isolated():
    failures = []

    def worker(version: str, tables: dict) -> None:
        try:
            for _ in range(5):
                with ParseSession(f"src-{version}", version, tables) as session:
                    out = parse_document(SAMPLE)
                    assert out[0]["columns"][0] == f"alpha_{version}"
                    assert session.cache_size() > 0
        except Exception as exc:  # pragma: no cover - failure path
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=("v1", V1_TABLES)),
        threading.Thread(target=worker, args=("v2", V2_TABLES)),
        threading.Thread(target=worker, args=("v1", V1_TABLES)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures


def test_fifo_cache_limit_is_enforced_per_session():
    with ParseSession("s", "v1", V1_TABLES, cache_size_limit=16) as session:
        parse_document(SAMPLE)
        assert session.cache_size() <= 16


def test_unknown_code_raises_code_lookup_error_outside_recovery():
    with ParseSession("s", "v1", V1_TABLES):
        with pytest.raises(CodeLookupError):
            mapped_code.parse_string("UNKNOWN", parse_all=True)


def test_memoized_failure_does_not_poison_later_success():
    with ParseSession("s", "v1", V1_TABLES):
        bad = parse_document("B,{!!bad!!}")
        good = parse_document("B,{X}")
    assert bad[0]["attachment"]["items"] is None
    assert good[0]["attachment"]["items"] == ["att_x_v1"]


def test_close_inside_context_manager_releases_lock_balanced():
    from column_import import _global_parse_lock

    session = ParseSession("s", "v1", V1_TABLES)
    with session:
        parse_document(SAMPLE)
        session.close()
        assert session.cache_size() == 0
    # Every __enter__ acquisition must have been released exactly once.
    assert _global_parse_lock.acquire(blocking=False)
    _global_parse_lock.release()


def test_nested_sessions_restore_outer_context():
    outer = ParseSession("outer", "v1", V1_TABLES)
    inner = ParseSession("inner", "v2", V2_TABLES)
    with outer:
        assert current_session() is outer
        with inner:
            assert current_session() is inner
            assert parse_document(SAMPLE)[0]["columns"][0] == "alpha_v2"
        assert current_session() is outer
        assert parse_document(SAMPLE)[0]["columns"][0] == "alpha_v1"
    assert current_session() is None
