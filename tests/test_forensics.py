"""Tests for the forensic replay/enumeration/rollback machinery."""

from __future__ import annotations

import pytest

from tinydb.table import Document

from forensics.model import (
    CondChange,
    DocumentChange,
    LogEntry,
    candidate_old_values,
    enumerate_seed_states,
    inverse_patches,
    matches_log,
    minimal_discriminating_observation,
    repair_scope,
    replay,
    rollback,
)
from forensics.model import EXT_KEY, _fresh_table
from forensics.scenarios import (
    CHANGES,
    DOMAINS,
    FIXED_FIELDS,
    LEGACY_LOG,
    OBSERVED_FINAL,
    ROW_IDS,
    SEED_A,
    SEED_B,
    run_full_enumeration,
    run_pair,
)


def test_three_upsert_paths_are_classified():
    result_a, _ = run_pair()
    paths = [trace.path for trace in result_a.traces]
    assert paths == ["cond-update", "doc-id-update", "insert"]

    step0 = result_a.traces[0]
    assert step0.returned_ids == (1, 2)
    assert set(step0.overwritten) == {1, 2}

    step2 = result_a.traces[2]
    assert step2.path == "insert"
    assert step2.returned_ids == (3,)
    assert step2.overwritten == {}


def test_cond_upsert_fanout_overwrites_two_master_rows():
    result_a, result_b = run_pair()
    assert result_a.traces[0].overwritten[1] == {"status": "off", "cat": "A"}
    # History B: row 1 already had cat=B (only status changes), row 2 already
    # had status=on (only cat changes) -- same end state, different old values.
    assert result_b.traces[0].overwritten[1] == {"status": "off"}
    assert result_b.traces[0].overwritten[2] == {"cat": "A"}


def test_histories_share_final_table_and_returned_ids():
    result_a, result_b = run_pair()
    assert result_a.final_table == result_b.final_table == OBSERVED_FINAL
    assert (
        result_a.returned_sequences
        == result_b.returned_sequences
        == [entry.returned_ids for entry in LEGACY_LOG]
    )
    assert matches_log(result_a, LEGACY_LOG, OBSERVED_FINAL)
    assert matches_log(result_b, LEGACY_LOG, OBSERVED_FINAL)


def test_overwritten_fields_differ_between_histories():
    result_a, result_b = run_pair()
    assert result_a.traces[0].overwritten != result_b.traces[0].overwritten
    assert result_a.traces[1].overwritten != result_b.traces[1].overwritten


def test_enumeration_covers_every_domain_combination():
    raw = enumerate_seed_states(ROW_IDS, DOMAINS, FIXED_FIELDS)
    per_row = 1
    for values in DOMAINS.values():
        per_row *= len(values)
    expected = per_row ** len(ROW_IDS)
    assert len(raw) == expected  # 27 per row ** 2 rows = 729 seed states
    assert per_row == 27


def test_all_enumerated_histories_match_log_and_pair_is_included():
    compatible = run_full_enumeration()
    assert len(compatible) >= 2
    seeds = {
        tuple(
            sorted((doc_id, tuple(sorted(row.items()))) for doc_id, row in h.traces[0].before.items())
        )
        for h in compatible
    }
    marker_a = tuple(
        sorted((doc_id, tuple(sorted(row.items()))) for doc_id, row in SEED_A.items())
    )
    marker_b = tuple(
        sorted((doc_id, tuple(sorted(row.items()))) for doc_id, row in SEED_B.items())
    )
    assert marker_a in seeds
    assert marker_b in seeds
    for history in compatible:
        assert matches_log(history, LEGACY_LOG, OBSERVED_FINAL)


def test_candidate_old_values_are_exhaustive_and_split_determined():
    compatible = run_full_enumeration()

    step0 = candidate_old_values(compatible, 0)
    row1 = step0["by_doc_id"][1]
    # Fields written by the step-0 payload are fully masked by the merge:
    # every domain value (including "key was absent") stays possible.
    assert set(row1["status"]["possible_values"]) == {"off", "on", "<absent>"}
    assert row1["status"]["unrecoverable"] is True
    assert set(row1["cat"]["possible_values"]) == {"A", "B", "<absent>"}
    assert row1["cat"]["unrecoverable"] is True
    # sku never appears in any payload, so the final value equals the old one.
    assert row1["sku"]["possible_values"] == ["s1"]
    assert row1["sku"]["determined"] is True
    # ext_id is the condition key and is identical across every seed.
    assert row1[EXT_KEY]["possible_values"] == ["X-1"]
    assert row1[EXT_KEY]["determined"] is True

    row2 = step0["by_doc_id"][2]
    assert set(row2["status"]["possible_values"]) == {"off", "on", "<absent>"}
    assert set(row2["cat"]["possible_values"]) == {"A", "B", "<absent>"}
    # note belongs to the enumerated domains, so it may already exist at step 0.
    assert set(row2["note"]["possible_values"]) == {"m", "n", "<absent>"}

    step1 = candidate_old_values(compatible, 1)
    note = step1["by_doc_id"][2]["note"]
    assert set(note["possible_values"]) == {"m", "n", "<absent>"}
    assert note["unrecoverable"] is True
    # By step 1, step 0 has already pinned status/cat to their final values.
    assert step1["by_doc_id"][2]["status"]["possible_values"] == ["on"]
    assert step1["by_doc_id"][2]["cat"]["possible_values"] == ["B"]


def test_narrowing_scenario_pins_old_value_via_upsert_path():
    from forensics.scenarios import run_narrowing_scenario

    narrow = run_narrowing_scenario()
    report = narrow["old_value_report"]
    status = report["by_doc_id"][1]["status"]
    assert status["possible_values"] == ["on"]
    assert status["determined"] is True
    cat = report["by_doc_id"][1]["cat"]
    assert set(cat["possible_values"]) == {"A", "B", "<absent>"}
    assert narrow["counterfactual_path"] == "insert"
    assert narrow["counterfactual_returned"] != (1,)


def test_minimal_observation_is_one_cell_and_discriminates():
    result_a, result_b = run_pair()
    probes = minimal_discriminating_observation(result_a, result_b, 0)
    assert probes, "histories must differ in at least one pre-op cell"
    probe = probes[0]
    assert probe["history_a_old"] != probe["history_b_old"]

    def observe(history):
        return history.traces[0].before[probe["doc_id"]].get(
            probe["field"], "<absent>"
        )

    assert observe(result_a) != observe(result_b)


def test_inverse_patches_roll_back_to_seed():
    for seed, result in ((SEED_A, replay(SEED_A, CHANGES)),
                         (SEED_B, replay(SEED_B, CHANGES))):
        table = _fresh_table()
        for doc_id, row in result.final_table.items():
            table.insert(Document(row, doc_id=doc_id))
        rollback(table, inverse_patches(result))
        restored = {doc.doc_id: dict(doc) for doc in table.all()}
        assert restored == seed


def test_repair_scope_matches_trace_footprint():
    result_a, _ = run_pair()
    scope0 = repair_scope(result_a, 0)
    assert set(scope0["updated_rows"]) == {1, 2}
    assert scope0["updated_rows"][1]["overwritten_fields"] == {
        "status": "off",
        "cat": "A",
    }
    assert scope0["inserted_doc_ids"] == []
    scope2 = repair_scope(result_a, 2)
    assert scope2["inserted_doc_ids"] == [3]
    assert scope2["updated_rows"] == {}


def test_rollback_withdraws_fields_added_to_previously_absent_keys():
    seed = {1: {EXT_KEY: "X-1", "sku": "s1", "status": "off"}}
    changes = [CondChange("X-1", {EXT_KEY: "X-1", "cat": "B"})]
    result = replay(seed, changes)
    assert result.traces[0].added_fields[1] == ["cat"]

    table = _fresh_table()
    table.insert(Document(result.final_table[1], doc_id=1))
    rollback(table, inverse_patches(result))
    assert table.get(doc_id=1) == {EXT_KEY: "X-1", "sku": "s1", "status": "off"}


def test_unknown_doc_id_upsert_falls_through_to_insert():
    result = replay(
        {},
        [DocumentChange(7, {EXT_KEY: "Z", "status": "on"})],
    )
    assert result.traces[0].path == "insert"
    assert result.traces[0].returned_ids == (7,)
