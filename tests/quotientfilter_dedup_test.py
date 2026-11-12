#!/usr/bin/env python3
"""Regression tests for quotient-filter based event de-duplication.

The fingerprint space is driven by a controllable hash function so that runs
can be forced to wrap across the end of the slot array. An independent ledger
keeps the full 32-bit hashes of every event that should be observable, so a
structural false negative (a stored fingerprint that the filter can no longer
report) is distinguishable from a genuine 32-bit hash collision between two
business keys.
"""

import itertools
import sys
import unittest
from pathlib import Path

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import QuotientFilter  # noqa: E402

MASK32 = 0xFFFFFFFF


def fingerprint(q_bits: int, quotient: int, remainder: int) -> int:
    """Build a full 32-bit hash from a chosen quotient and remainder."""
    r = 32 - q_bits
    return ((quotient << r) | (remainder & ((1 << r) - 1))) & MASK32


class ControllableHash:
    """Hash strategy mapping named events to pre-selected 32-bit fingerprints."""

    def __init__(self, mapping: dict[str, int]):
        self._mapping = mapping

    def __call__(self, key, seed=0) -> int:
        return self._mapping[key]


# Fingerprints whose run wraps past the final slot on a q=3 (8 slot) table:
# quotient 7 has three remainders whose run spills into slots 0 and 1, while
# quotients 0 and 1 anchor homes inside the same wrapped cluster.
WRAP_SPECS = {
    "evt-7a": fingerprint(3, 7, 1),
    "evt-7b": fingerprint(3, 7, 2),
    "evt-7c": fingerprint(3, 7, 3),
    "evt-0": fingerprint(3, 0, 5),
    "evt-1": fingerprint(3, 1, 7),
}


class TestDedupRegression(unittest.TestCase):
    """De-duplication behaviour with an independent event ledger."""

    def _build(self, specs, quotient=3, auto_expand=False):
        qf = QuotientFilter(
            quotient=quotient,
            auto_expand=auto_expand,
            hash_function=ControllableHash(dict(specs)),
        )
        return qf

    def _audit(self, qf, ledger, step_log, step):
        """Every ledger hash must be observable and the stored set must match."""
        stored = set(qf.get_hashes())
        self.assertEqual(
            stored,
            ledger,
            msg=f"after step {step!r}: stored={sorted(stored)} ledger={sorted(ledger)} log={step_log}",
        )
        for full_hash in ledger:
            self.assertTrue(
                qf.check_alt(full_hash),
                msg=f"structural false negative after {step!r}: {full_hash:#010x} log={step_log}",
            )
        self.assertEqual(qf.elements_added, len(ledger), msg=f"counter drift at {step!r}: {step_log}")

    def test_wrapped_run_remove_and_readd_matches_ledger(self):
        """Delete then re-insert events inside a run that crosses the slot end."""
        qf = self._build(WRAP_SPECS)
        ledger: set[int] = set()
        step_log: list[str] = []

        for name in WRAP_SPECS:
            qf.add(name)
            ledger.add(WRAP_SPECS[name])
            step_log.append(f"+{name}")
            self._audit(qf, ledger, step_log, step_log[-1])

        # every removal order over the same wrapped run must keep the ledger exact
        for order in itertools.permutations(WRAP_SPECS.keys()):
            qf = self._build(WRAP_SPECS)
            ledger = set(WRAP_SPECS.values())
            for name in WRAP_SPECS:
                qf.add(name)
            step_log = []
            for name in order:
                qf.remove(name)
                ledger.discard(WRAP_SPECS[name])
                step_log.append(f"-{name}")
                self._audit(qf, ledger, step_log, step_log[-1])

                # re-adding the same event must not be judged a duplicate-drop
                qf.add(name)
                ledger.add(WRAP_SPECS[name])
                step_log.append(f"+{name}")
                self._audit(qf, ledger, step_log, step_log[-1])

    def test_remove_decrements_counter_and_frees_capacity(self):
        """Minimal failing sequence: fill, remove one, re-add into the freed slot."""
        specs = {f"evt-{i}": fingerprint(3, i, 10 + i) for i in range(8)}
        qf = self._build(specs)
        for name in specs:
            qf.add(name)
        self.assertEqual(qf.elements_added, 8)

        qf.remove("evt-3")
        self.assertEqual(qf.elements_added, 7)
        self.assertEqual(qf.load_factor, 7 / 8)

        # a free slot exists, so the previously removed event must be accepted again
        qf.add("evt-3")
        self.assertEqual(qf.elements_added, 8)
        self.assertTrue(qf.check("evt-3"))

    def test_no_premature_auto_expand_after_remove_readd_cycle(self):
        """Inflated counters must not trigger a resize while real load is below threshold."""
        specs = {f"evt-{i}": fingerprint(3, i, 10 + i) for i in range(8)}
        qf = self._build(specs, auto_expand=True)
        for i in range(6):
            qf.add(f"evt-{i}")
        starting_quotient = qf.quotient

        qf.remove("evt-0")
        qf.remove("evt-1")
        qf.add("evt-0")
        qf.add("evt-1")
        # six distinct fingerprints still live: 6/8 < 0.85, no resize expected
        self.assertEqual(qf.quotient, starting_quotient)
        self.assertEqual(qf.elements_added, 6)

    def test_resize_rechecks_members_under_new_quotient_split(self):
        """After growth, the same hashes must map through the new q/r boundary."""
        qf = self._build(WRAP_SPECS)
        for name in WRAP_SPECS:
            qf.add(name)
        before = set(qf.get_hashes())

        qf.resize()  # quotient 3 -> 4, remainder 29 -> 28
        self.assertEqual((qf.quotient, qf.remainder), (4, 28))
        self.assertEqual(set(qf.get_hashes()), before)
        self.assertEqual(qf.elements_added, len(before))

        for full_hash in before:
            new_quotient = full_hash >> qf.remainder
            new_remainder = full_hash & ((1 << qf.remainder) - 1)
            self.assertTrue(
                qf.check_alt(full_hash),
                msg=f"missing {full_hash:#010x} after resize (q={new_quotient}, r={new_remainder})",
            )

    def test_resize_on_full_table_does_not_crash(self):
        """A table without an empty slot must still be able to grow."""
        specs = {f"evt-{i}": fingerprint(3, i, 10 + i) for i in range(8)}
        qf = self._build(specs)
        for name in specs:
            qf.add(name)
        qf.resize()
        self.assertEqual(qf.quotient, 4)
        self.assertEqual(qf.elements_added, 8)
        for name in specs:
            self.assertTrue(qf.check(name))

    def test_true_hash_collision_is_set_semantics_not_structural_bug(self):
        """Two business keys sharing one 32-bit hash collapse to one fingerprint.

        The structural tests above use distinct hashes and demand exact ledger
        equality; here the collision is inherent to the 32-bit representation,
        so set semantics (one slot, removal clears the shared fingerprint) are
        expected and must not be confused with a metadata false negative.
        """
        shared = fingerprint(3, 5, 42)
        specs = {"order-1": shared, "order-2": shared}
        qf = self._build(specs)
        qf.add("order-1")
        qf.add("order-2")  # identical full hash: no new fingerprint
        self.assertEqual(qf.elements_added, 1)
        self.assertTrue(qf.check("order-1"))
        self.assertTrue(qf.check("order-2"))

        qf.remove("order-1")
        # the shared fingerprint is gone even though order-2 was never removed:
        # this is a true hash collision, not corrupted run metadata
        self.assertFalse(qf.check("order-2"))
        self.assertEqual(qf.elements_added, 0)
        # distinct-hash filters remain exact, proving the structure itself is sound
        distinct = self._build(WRAP_SPECS)
        for name in WRAP_SPECS:
            distinct.add(name)
        self.assertEqual(set(distinct.get_hashes()), set(WRAP_SPECS.values()))

    def test_validate_metadata_flags_invalid_bit_combos(self):
        """The validator must actually inspect per-slot bits."""
        specs = {"evt-0": fingerprint(3, 0, 9)}
        qf = self._build(specs)
        qf.add("evt-0")
        self.assertTrue(qf.validate_metadata())

        qf._is_continuation[0] = 1  # O=1, C=1, S=0 is an illegal combination
        self.assertFalse(qf.validate_metadata())

    def test_shortest_failing_sequence_search_stays_clean_with_log(self):
        """Breadth-first search for the shortest breaking op sequence.

        Each transition is logged with the event and the resulting observable
        membership, so a regression points at the exact step that changed the
        next event's de-duplication verdict. All sequences must stay consistent
        with the independent ledger.
        """
        names = list(WRAP_SPECS.keys())
        hashes = [WRAP_SPECS[name] for name in names]
        shortest_failure = None
        checked_transitions = 0

        # bounded BFS over add/remove masks on the wrapped-run fingerprints
        for depth in range(1, 5):
            for seq in itertools.product(range(len(names)), repeat=depth):
                qf = self._build(WRAP_SPECS)
                live = [False] * len(names)
                ledger: set[int] = set()
                log: list[str] = []
                broken = False
                for idx in seq:
                    name, full_hash = names[idx], hashes[idx]
                    if live[idx]:
                        qf.remove(name)
                        live[idx] = False
                        ledger.discard(full_hash)
                        log.append(f"-{name}")
                    else:
                        qf.add(name)
                        live[idx] = True
                        ledger.add(full_hash)
                        log.append(f"+{name}")
                    checked_transitions += 1
                    if set(qf.get_hashes()) != ledger or any(not qf.check_alt(h) for h in ledger):
                        shortest_failure = list(log)
                        broken = True
                        break
                if broken:
                    break
            if shortest_failure is not None:
                break

        self.assertIsNone(shortest_failure, msg=f"breaking sequence found: {shortest_failure}")
        self.assertGreater(checked_transitions, 0)


if __name__ == "__main__":
    unittest.main()
