#!/usr/bin/env python3
"""Executable verification of a recovery criterion for sqlite-utils `convert`
backfilling a dimension table, resumed by source row number.

Setup under test
----------------
- dim(id, raw, converted): `converted` starts as a copy of `raw` and is
  backfilled IN PLACE in batches (id ranges) via
  Table.convert("converted", fn, where=...).
  Each convert() call is ONE transaction (sqlite-utils wraps the single
  UPDATE in db.atomic()).
- fn is NON-idempotent: fn(v) = v + "#". Re-applying to a committed row
  yields "##", which is detectable corruption. (Converting raw -> output
  column would be naturally idempotent because raw never changes; the
  in-place form is the one where a checkpoint-only resume can corrupt.)
- A counting trigger AFTER UPDATE OF converted writes a companion row into
  `audit` and increments `counters.n` -- trigger effects commit/roll back
  atomically with the batch UPDATE.
- The caller's acknowledgement (checkpoint) is a file OUTSIDE the database:
  ack = k means "batches 1..k reported done". It is written AFTER commit.

Three failure histories are replayed, then the DB is REOPENED (process
restart) before any resume decision:
  A. last-row conversion failure: fn raises on the last row of batch k
     -> statement aborts, transaction rolls back.
  B. explicit rollback: batch k converts cleanly, caller rolls back the
     outer transaction on purpose.
  C. lost acknowledgement: batch k commits, process dies before the ack
     file is written.

Findings asserted by this script
--------------------------------
1. A and B are indistinguishable on disk and need the SAME action
   (re-run batch k): nothing of batch k persisted, in any channel.
2. A and C leave the SAME checkpoint (ack = k-1) but need OPPOSITE actions
   for batch k. A checkpoint-only ("naive") resume is therefore not a safe
   criterion: in history C it re-applies fn to committed rows (## values,
   counter overshoot).
3. Minimal persistent disambiguating evidence: ONE durable integer written
   in the SAME transaction as the batch -- a committed high-water mark
   (hwm = last committed batch). Because batches are atomic, ordered and
   fixed-boundary, hwm alone separates A/B (hwm = k-1) from C (hwm = k).
   The already-existing trigger counter is an equivalent witness: the
   unique batch prefix whose row-count sum equals counters.n.
4. Recovery criterion: resume from hwm + 1 (never from the ack alone);
   treat the ack as a cache and repair it from hwm. After resume, business
   values reconcile: every row converted exactly once, counters.n ==
   audit rows == number of source rows, for both batch sizes.

Run: .venv/bin/python verify_convert_recovery.py
"""

import json
import os
import sqlite3
import sys
import tempfile

import sqlite_utils

N_ROWS = 10
BATCH_SIZES = [3, 7]  # small multi-batch resume / resume crossing batch boundary
CRASH_BATCH = 2


class ExplicitRollback(Exception):
    pass


def make_fn(fail_value=None, name="convert_fn"):
    def convert_fn(value):
        if value == fail_value:
            raise RuntimeError(f"injected conversion failure on {value!r}")
        return value + "#"

    # NOTE: Database.register_function skips re-registration of an already
    # seen (name, arity) unless replace=True, and convert() does not pass
    # replace=True -- so on one connection every batch needs a distinct
    # function name, otherwise the FIRST batch's closure is silently reused.
    convert_fn.__name__ = name
    return convert_fn


def build_batches(batch_size):
    batches = []
    lo = 1
    while lo <= N_ROWS:
        hi = min(lo + batch_size - 1, N_ROWS)
        batches.append((lo, hi))
        lo = hi + 1
    return batches


def prefix_sums(batches):
    sums, total = [], 0
    for lo, hi in batches:
        total += hi - lo + 1
        sums.append(total)
    return sums


def fresh_db(path):
    db = sqlite_utils.Database(path)
    db["dim"].create({"id": int, "raw": str, "converted": str}, pk="id")
    db["dim"].insert_all(
        {"id": i, "raw": f"v{i}", "converted": f"v{i}"} for i in range(1, N_ROWS + 1)
    )
    db["audit"].create({"row_id": int, "new_value": str})
    db["counters"].create({"name": str, "n": int}, pk="name")
    db["counters"].insert({"name": "convert_hits", "n": 0})
    # committed high-water mark: the minimal disambiguating evidence,
    # written in the SAME transaction as the batch it certifies
    db["hwm"].create({"id": int, "committed_batch": int}, pk="id")
    db["hwm"].insert({"id": 1, "committed_batch": 0})
    db.execute(
        """
        CREATE TRIGGER dim_convert_audit AFTER UPDATE OF converted ON dim
        BEGIN
            INSERT INTO audit (row_id, new_value) VALUES (NEW.id, NEW.converted);
            UPDATE counters SET n = n + 1 WHERE name = 'convert_hits';
        END
        """
    )
    return db


def write_ack(ack_path, k):
    with open(ack_path, "w") as f:
        json.dump({"acked_batch": k}, f)


def read_ack(ack_path):
    if not os.path.exists(ack_path):
        return 0
    with open(ack_path) as f:
        return json.load(f)["acked_batch"]


def run_batches(db, batches, start, ack_path, scenario=None, crash_batch=None):
    """Run batches start..len(batches). Returns batch number where the
    process 'crashed', or None if all batches completed and were acked."""
    for k in range(start, len(batches) + 1):
        lo, hi = batches[k - 1]
        # scenario A: conversion fails on the LAST row of the crash batch
        fail_value = f"v{hi}" if (scenario == "A" and k == crash_batch) else None
        try:
            with db.atomic():
                db["dim"].convert(
                    "converted",
                    make_fn(fail_value, name=f"convert_fn_b{k}"),
                    where="id >= ? AND id <= ?",
                    where_args=[lo, hi],
                )
                db.execute("UPDATE hwm SET committed_batch = ? WHERE id = 1", [k])
                if scenario == "B" and k == crash_batch:
                    raise ExplicitRollback()  # scenario B: caller rolls back
        except (ExplicitRollback, sqlite3.OperationalError):
            return k  # transaction rolled back; nothing of batch k persisted
        if scenario == "C" and k == crash_batch:
            return k  # committed, but process dies before writing the ack
        write_ack(ack_path, k)
    return None


def snapshot(db, ack_path):
    return {
        "converted_rows": db.execute(
            "SELECT COUNT(*) FROM dim WHERE converted LIKE '%#'"
        ).fetchone()[0],
        "double_converted": db.execute(
            "SELECT COUNT(*) FROM dim WHERE converted LIKE '%##%'"
        ).fetchone()[0],
        "audit_rows": db.execute("SELECT COUNT(*) FROM audit").fetchone()[0],
        "counter": db.execute(
            "SELECT n FROM counters WHERE name = 'convert_hits'"
        ).fetchone()[0],
        "hwm": db.execute("SELECT committed_batch FROM hwm WHERE id = 1").fetchone()[
            0
        ],
        "ack": read_ack(ack_path),
    }


def resume(db, batches, ack_path):
    """Recovery criterion: trust the durable hwm, repair the ack cache.
    The ack may lag hwm (lost acknowledgement); the converse is impossible
    because acks are written strictly after commit."""
    hwm = db.execute("SELECT committed_batch FROM hwm WHERE id = 1").fetchone()[0]
    ack = read_ack(ack_path)
    assert ack <= hwm, f"ack ({ack}) ahead of hwm ({hwm}): invariant violated"
    if ack < hwm:
        write_ack(ack_path, hwm)  # repair lost acknowledgement(s)
    run_batches(db, batches, hwm + 1, ack_path)


def infer_committed_from_counter(counter, batches):
    """The trigger counter as witness: batches are ordered, atomic and
    non-empty, so prefix sums are strictly increasing and at most one
    prefix can match."""
    sums = prefix_sums(batches)
    if counter in sums:
        return sums.index(counter) + 1
    if counter == 0:
        return 0
    return None  # counter does not match any committed prefix -> corruption


def reopen(db, path):
    db.close()
    return sqlite_utils.Database(path)


def reconcile_business_values(db, batches, label):
    rows = db.execute("SELECT id, raw, converted FROM dim ORDER BY id").fetchall()
    assert all(c == r + "#" for _, r, c in rows), f"{label}: not converted once"
    snap = snapshot(db, ack_path=current_ack_path)
    assert snap["converted_rows"] == N_ROWS, f"{label}: converted count"
    assert snap["counter"] == N_ROWS, f"{label}: trigger counter"
    assert snap["audit_rows"] == N_ROWS, f"{label}: audit rows"
    audit_ids = {r[0] for r in db.execute("SELECT row_id FROM audit")}
    assert audit_ids == set(range(1, N_ROWS + 1)), f"{label}: audit ids"
    assert snap["hwm"] == len(batches), f"{label}: hwm"
    assert snap["ack"] == len(batches), f"{label}: ack"
    print(f"    reconcile OK: {N_ROWS} rows converted exactly once, "
          f"counter == audit == {N_ROWS}, hwm == ack == {len(batches)}")


current_ack_path = None


def expect(snap, label, **expected):
    for key, value in expected.items():
        assert snap[key] == value, f"{label}: {key} == {snap[key]}, expected {value}"


def scenario_run(batch_size, scenario):
    """Replay one failure history, reopen, check persisted state, resume
    via hwm, reconcile. Returns the post-crash snapshot."""
    global current_ack_path
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "dim.db")
    current_ack_path = db_path + ".ack"
    batches = build_batches(batch_size)
    sums = prefix_sums(batches)
    k = CRASH_BATCH
    db = fresh_db(db_path)

    crashed_at = run_batches(
        db, batches, 1, current_ack_path, scenario=scenario, crash_batch=k
    )
    assert crashed_at == k
    db = reopen(db, db_path)  # process restart: only durable state survives
    snap = snapshot(db, current_ack_path)
    label = f"scenario {scenario} batch_size={batch_size}"
    print(f"  {label}: post-crash state {snap}")

    if scenario in ("A", "B"):
        # nothing of batch k persisted, in any of the three channels
        expect(
            snap,
            label,
            converted_rows=sums[k - 2],
            audit_rows=sums[k - 2],
            counter=sums[k - 2],
            hwm=k - 1,
            ack=k - 1,
        )
    else:  # C: batch k fully committed, only the ack is missing
        expect(
            snap,
            label,
            converted_rows=sums[k - 1],
            audit_rows=sums[k - 1],
            counter=sums[k - 1],
            hwm=k,
            ack=k - 1,  # same checkpoint as A/B, different required action
        )
    # the trigger counter alone already witnesses the committed prefix
    assert infer_committed_from_counter(snap["counter"], batches) == snap["hwm"], label

    # recovery criterion: resume from the durable hwm, not from the ack
    resume(db, batches, current_ack_path)
    db = reopen(db, db_path)
    reconcile_business_values(db, batches, label)
    db.close()
    return snap


def demo_naive_resume_corrupts(batch_size):
    """History C resumed from the checkpoint alone: same ack as A/B, but
    re-running batch k double-applies the non-idempotent fn."""
    global current_ack_path
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "dim.db")
    current_ack_path = db_path + ".ack"
    batches = build_batches(batch_size)
    sums = prefix_sums(batches)
    k = CRASH_BATCH
    db = fresh_db(db_path)
    run_batches(db, batches, 1, current_ack_path, scenario="C", crash_batch=k)
    db = reopen(db, db_path)
    ack = read_ack(current_ack_path)
    assert ack == k - 1
    # naive resume: trust the checkpoint, re-run batch k
    run_batches(db, batches, ack + 1, current_ack_path)
    db = reopen(db, db_path)
    snap = snapshot(db, current_ack_path)
    batch_k_size = sums[k - 1] - sums[k - 2]
    label = f"naive resume batch_size={batch_size}"
    print(f"  {label}: {snap}")
    expect(
        snap,
        label,
        double_converted=batch_k_size,  # '##' rows: fn applied twice
        counter=N_ROWS + batch_k_size,  # counting trigger overshoots
        audit_rows=N_ROWS + batch_k_size,
    )
    assert infer_committed_from_counter(snap["counter"], batches) is None, label
    print(f"    corruption detected: {batch_k_size} rows converted twice, "
          f"counter overshoot +{batch_k_size}")
    db.close()


def main():
    print("sqlite-utils convert recovery-criterion verification")
    print(f"N_ROWS={N_ROWS} CRASH_BATCH={CRASH_BATCH} fn(v)=v+'#' (non-idempotent)")
    for batch_size in BATCH_SIZES:
        print(f"\nbatch_size={batch_size} batches={build_batches(batch_size)}")
        snaps = {}
        for scenario in ("A", "B", "C"):
            snaps[scenario] = scenario_run(batch_size, scenario)
        # the crux: A and C share one checkpoint but need opposite actions
        assert snaps["A"]["ack"] == snaps["C"]["ack"]
        assert snaps["A"]["hwm"] != snaps["C"]["hwm"]
        print(f"  disambiguation: ack identical ({snaps['A']['ack']}) for A and C; "
              f"hwm differs ({snaps['A']['hwm']} vs {snaps['C']['hwm']}) "
              f"-> 1 durable integer separates the histories")
        demo_naive_resume_corrupts(batch_size)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
