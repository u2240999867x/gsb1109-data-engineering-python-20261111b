"""
Tests for the read-only memo claim preview (dbfread.memo_claim).
"""
import os

from pytest import raises

from dbfread import DBF
from dbfread.memo_claim import (
    MemoClaimError,
    claim_memo_files,
    discover_candidates,
    probe_memofile,
)

from memo_claim_fixtures import (
    CANONICAL_NOTES,
    VARIANT_NOTES,
    WRONG_FINGERPRINT_NOTES,
    build_fpt_candidates,
    copy_table,
    write_fpt,
    write_db3_dbt,
    write_db4_dbt,
    VFP_TEXT,
)


def _open_table(tmp_path):
    return DBF(copy_table(tmp_path), ignore_missing_memofile=True)


def _paths_by_name(verdicts, names_to_paths):
    inverted = {path: name for name, path in names_to_paths.items()}
    return {inverted[v.path]: v for v in verdicts}


def test_probe_reads_fpt_header(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    probe = probe_memofile(paths['canonical'], table.header.dbversion)
    assert probe.kind == 'fpt'
    assert probe.block_size == 512
    assert probe.file_size == os.path.getsize(paths['canonical'])
    assert probe.block_count == probe.file_size // 512


def test_probe_rejects_wrong_extension(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    with raises(MemoClaimError):
        probe_memofile(paths['wrong_format'], table.header.dbversion)


def test_discovery_is_format_and_stem_scoped(tmp_path):
    table = _open_table(tmp_path)
    directory = str(tmp_path)
    write_fpt(os.path.join(directory, 'memotest.FPT'), 512,
              {index: (VFP_TEXT, data)
               for index, data in CANONICAL_NOTES.items()})
    write_db4_dbt(os.path.join(directory, 'memotest.dbt'), CANONICAL_NOTES)

    found = discover_candidates(table.filename, table.header.dbversion)
    assert found == [os.path.join(directory, 'memotest.FPT')]


def test_incompatible_candidates_are_rejected(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(table, candidates=paths.values())

    by_name = _paths_by_name(report.verdicts, paths)
    assert by_name['truncated'].compatible is False
    assert any('block 2' in reason and 'truncated' in reason
               for reason in by_name['truncated'].reasons)

    assert by_name['bad_length'].compatible is False
    assert any('block 2' in reason for reason
               in by_name['bad_length'].reasons)

    assert by_name['wrong_format'].compatible is False
    assert any('format mismatch' in reason for reason
               in by_name['wrong_format'].reasons)

    assert set(report.compatible_paths) == {
        paths['canonical'], paths['copy'], paths['variant'],
        paths['wrong_fingerprint']}


def test_fingerprints_exclude_mismatched_candidates(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table, candidates=paths.values(),
        fingerprints={1: 'Alice memo', 2: 'Bob memo'})

    by_name = _paths_by_name(report.verdicts, paths)
    assert by_name['canonical'].compatible is True
    assert by_name['copy'].compatible is True

    assert by_name['variant'].compatible is False
    assert 2 in by_name['variant'].fingerprint_mismatches

    assert by_name['wrong_fingerprint'].compatible is False
    assert 1 in by_name['wrong_fingerprint'].fingerprint_mismatches

    assert set(report.compatible_paths) == {
        paths['canonical'], paths['copy']}


def test_unknown_fingerprint_record_raises(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    with raises(MemoClaimError):
        claim_memo_files(table, candidates=[paths['canonical']],
                         fingerprints={99: 'ghost'})


def test_consensus_and_disputed_rows_keep_record_numbers(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table,
        candidates=[paths['canonical'], paths['copy'], paths['variant']])

    rows = report.compare_rows()
    assert [row.record_number for row in rows] == [1, 2]

    first, second = rows
    assert first.consensus is True
    assert first.disputed is False
    assert set(first.values_by_candidate.values()) == {'Alice memo'}

    assert second.consensus is False
    assert second.disputed is True
    assert second.record_number == 2
    assert second.value(paths['canonical']) == 'Bob memo'
    assert second.value(paths['variant']) == 'Bob CHANGED'

    assert [row.record_number for row in report.consensus_rows] == [1]
    assert [row.record_number for row in report.disputed_rows] == [2]


def test_equivalent_copies_form_one_class_independent_of_name(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table,
        candidates=[paths['canonical'], paths['copy'], paths['variant']])

    classes = report.equivalence_classes()
    assert len(classes) == 2

    members = {frozenset(cls.members) for cls in classes}
    assert frozenset((paths['canonical'], paths['copy'])) in members
    assert frozenset((paths['variant'],)) in members

    # The lower-case-named byte copy must not be chosen or split by name.
    joined = sorted(path for cls in classes for path in cls.members)
    assert paths['copy'] in joined and paths['canonical'] in joined


def test_forensic_rows_are_minimum_for_two_classes(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table,
        candidates=[paths['canonical'], paths['copy'], paths['variant']])

    chosen = report.forensic_rows()
    assert [row.record_number for row in chosen] == [2]
    # Record 1 cannot distinguish them, so a single row is the true minimum.
    assert chosen[0].disputed is True


def test_forensic_rows_need_both_rows_for_three_classes(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    third = os.path.join(str(tmp_path), 'memotest_third.FPT')
    write_fpt(third, 512,
              {1: (VFP_TEXT, b'Alice EDITED'),
               2: (VFP_TEXT, CANONICAL_NOTES[2])})

    table = _open_table(tmp_path)
    report = claim_memo_files(
        table,
        candidates=[paths['canonical'], paths['variant'], third])

    assert len(report.equivalence_classes()) == 3
    chosen = report.forensic_rows()
    assert sorted(row.record_number for row in chosen) == [1, 2]


def test_forensic_selection_respects_verifiable_records(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table,
        candidates=[paths['canonical'], paths['variant']])

    with raises(MemoClaimError):
        report.forensic_rows(verifiable_records=[1])

    chosen = report.forensic_rows(verifiable_records=[1, 2])
    assert [row.record_number for row in chosen] == [2]


def test_equivalent_candidates_yield_no_forensic_rows(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table, candidates=[paths['canonical'], paths['copy']])
    assert len(report.equivalence_classes()) == 1
    assert report.forensic_rows() == []


def test_unique_claim_flag(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(table, candidates=[paths['canonical']])
    assert report.has_unique_claim is True


def test_no_compatible_candidate_has_nothing_to_compare(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table, candidates=[paths['truncated'], paths['wrong_format']])
    assert report.compatible_paths == []
    assert report.has_unique_claim is False
    assert report.compare_rows() == []
    assert report.consensus_rows == []
    assert report.disputed_rows == []
    assert report.equivalence_classes() == []
    assert report.forensic_rows() == []


class _PlainHeader(object):
    dbversion = 0x30


class _PlainField(object):
    type = 'C'


class _PlainTable(object):
    char_decode_errors = 'strict'
    header = _PlainHeader()
    fields = [_PlainField()]
    filename = 'plain.dbf'


def test_table_without_memo_fields_needs_no_memo_file():
    report = claim_memo_files(_PlainTable())
    assert report.verdicts == []
    assert report.compare_rows() == []
    assert report.equivalence_classes() == []


def test_fingerprint_on_table_without_memo_fields_raises():
    with raises(MemoClaimError):
        claim_memo_files(_PlainTable(), fingerprints={1: 'note'})


def test_dbase_four_dbt_candidate(tmp_path):
    table = _open_table(tmp_path)
    table.header.dbversion = 0x8B
    dbt_path = os.path.join(str(tmp_path), 'memotest.dbt')
    write_db4_dbt(dbt_path, CANONICAL_NOTES)

    report = claim_memo_files(table, candidates=[dbt_path])
    assert report.compatible_paths == [dbt_path]
    rows = report.compare_rows()
    assert [row.values_by_candidate[dbt_path] for row in rows] == [
        'Alice memo', 'Bob memo']


def test_dbase_three_dbt_candidate(tmp_path):
    table = _open_table(tmp_path)
    table.header.dbversion = 0x83
    dbt_path = os.path.join(str(tmp_path), 'memotest.dbt')
    write_db3_dbt(dbt_path, CANONICAL_NOTES)

    report = claim_memo_files(table, candidates=[dbt_path])
    assert report.compatible_paths == [dbt_path]
    rows = report.compare_rows()
    assert [row.values_by_candidate[dbt_path] for row in rows] == [
        'Alice memo', 'Bob memo']


def test_preview_is_read_only(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    candidates = [paths['canonical'], paths['copy'], paths['variant']]
    before = {path: open(path, 'rb').read() for path in candidates}

    report = claim_memo_files(table, candidates=candidates,
                              fingerprints={1: 'Alice memo'})
    report.compare_rows()
    report.equivalence_classes()
    report.forensic_rows()

    after = {path: open(path, 'rb').read() for path in candidates}
    assert before == after


def test_block_size_is_taken_from_fpt_header(tmp_path):
    table = _open_table(tmp_path)
    custom = os.path.join(str(tmp_path), 'blocksize.FPT')
    with open(custom, 'wb') as handle:
        handle.write(b'\x00\x00\x00\x00\x00\x00\x04\x00')
        handle.write(b'\x00' * (512 - 8))
    probe = probe_memofile(custom, table.header.dbversion)
    assert probe.block_size == 1024


def test_wrong_fingerprint_fits_without_fingerprints(tmp_path):
    paths = build_fpt_candidates(tmp_path)
    table = _open_table(tmp_path)
    report = claim_memo_files(
        table, candidates=[paths['wrong_fingerprint']])
    assert report.compatible_paths == [paths['wrong_fingerprint']]
    assert WRONG_FINGERPRINT_NOTES[1] != CANONICAL_NOTES[1]
    assert VARIANT_NOTES[2] != CANONICAL_NOTES[2]
