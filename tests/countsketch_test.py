#!/usr/bin/env python3
"""Unittest class"""

import sys
import unittest
from pathlib import Path

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import CountMinSketch, CountSketch  # noqa: E402
from probables.exceptions import CountMinSketchError, InitializationError  # noqa: E402
from probables.hashes import fnv_1a  # noqa: E402
from tests.utilities import different_hash  # noqa: E402

# A controlled, colliding hash and sign convention used to make the sketch
# error visible: SKU-CHERRY collides with SKU-APPLE in row 0, SKU-DATE
# collides with SKU-BANANA in row 1 and with SKU-CHERRY in row 2
SKUS = ("SKU-APPLE", "SKU-BANANA", "SKU-CHERRY", "SKU-DATE")
BUCKETS = {
    "SKU-APPLE": [0, 0, 0],
    "SKU-BANANA": [1, 1, 1],
    "SKU-CHERRY": [0, 2, 3],
    "SKU-DATE": [2, 1, 3],
}
SIGNS = {
    "SKU-APPLE": [1, 1, 1],
    "SKU-BANANA": [1, -1, 1],
    "SKU-CHERRY": [-1, 1, 1],
    "SKU-DATE": [1, -1, 1],
}
DEMO_WIDTH = 4
DEMO_DEPTH = 3


def colliding_hash(key, depth):
    """bucket hash with known collisions"""
    if key in BUCKETS:
        return BUCKETS[key][:depth]
    return [fnv_1a(key, seed=idx) % DEMO_WIDTH for idx in range(depth)]


def colliding_sign_hash(key, depth):
    """sign hash paired with the colliding bucket hash"""
    if key in SIGNS:
        return SIGNS[key][:depth]
    return [1 if fnv_1a(key, seed=1000 + idx) & 1 == 0 else -1 for idx in range(depth)]


def other_sign_hash(key, depth):
    """a different sign convention"""
    return [-1 * sign for sign in colliding_sign_hash(key, depth)]


# sales are positive, refunds are negative, revisions are signed corrections;
# (date, table, sku, signed amount)
EVENTS = [
    ("2026-11-10", "online", "SKU-APPLE", 8),
    ("2026-11-10", "online", "SKU-BANANA", 6),
    ("2026-11-10", "online", "SKU-CHERRY", 2),
    ("2026-11-10", "online", "SKU-CHERRY", -6),  # refund
    ("2026-11-10", "online", "SKU-DATE", 3),
    ("2026-11-11", "online", "SKU-APPLE", 5),
    ("2026-11-11", "online", "SKU-APPLE", -2),  # refund
    ("2026-11-11", "online", "SKU-APPLE", -1),  # refund revision
    ("2026-11-10", "retail", "SKU-APPLE", 5),
    ("2026-11-10", "retail", "SKU-BANANA", -2),  # refund with no sale: negative net
    ("2026-11-10", "retail", "SKU-CHERRY", 9),
    ("2026-11-10", "retail", "SKU-DATE", 1),
    ("2026-11-11", "retail", "SKU-CHERRY", -2),  # refund
]

NETS = {"online": {}, "retail": {}}
for _, table, sku, amount in EVENTS:
    NETS[table][sku] = NETS[table].get(sku, 0) + amount
EXACT_INNER_PRODUCT = sum(NETS["online"][sku] * NETS["retail"][sku] for sku in SKUS)


def build_sketch(events, **kwargs):
    """build a demo count-sketch from (date, table, sku, amount) events"""
    prms = {"width": DEMO_WIDTH, "depth": DEMO_DEPTH}
    prms.update(kwargs)
    cs = CountSketch(hash_function=colliding_hash, sign_hash_function=colliding_sign_hash, **prms)
    for _, _, sku, amount in events:
        cs.add(sku, amount)
    return cs


def build_table_sketch(table, dates=None):
    """build the sketch of one report (table), optionally limited to dates"""
    events = [e for e in EVENTS if e[1] == table and (dates is None or e[0] in dates)]
    return build_sketch(events)


class TestCountSketch(unittest.TestCase):
    """Test the signed count-sketch implementation"""

    def test_cs_init_wd(self):
        """Test count-sketch initialization using depth and width"""
        cs = CountSketch(width=1000, depth=5)
        self.assertEqual(cs.width, 1000)
        self.assertEqual(cs.depth, 5)
        self.assertEqual(cs.confidence, 0.96875)
        self.assertEqual(cs.error_rate, 0.002)
        self.assertEqual(cs.elements_added, 0)

    def test_cs_init_ce(self):
        """Test count-sketch initialization using confidence and error rate"""
        cs = CountSketch(confidence=0.96875, error_rate=0.002)
        self.assertEqual(cs.width, 1000)
        self.assertEqual(cs.depth, 5)
        self.assertEqual(cs.confidence, 0.96875)
        self.assertEqual(cs.error_rate, 0.002)
        self.assertEqual(cs.elements_added, 0)

    def test_cs_init_error(self):
        """Test count-sketch initialization without enough params"""
        self.assertRaises(InitializationError, lambda: CountSketch(width=1000))
        self.assertRaises(InitializationError, lambda: CountSketch(width=0, depth=5))
        self.assertRaises(InitializationError, lambda: CountSketch(confidence=0.0, error_rate=0.002))

    def test_cs_init_error_msg(self):
        """Test count-sketch initialization error message"""
        try:
            CountSketch(width=1000)
        except InitializationError as ex:
            msg = (
                "Must provide one of the following to initialize the "
                "Count-Sketch:\n"
                "    The width and depth,\n"
                "    OR confidence and error rate"
            )
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cs_str(self):
        """Test the string representation of the count-sketch"""
        cs = CountSketch(width=100, depth=4)
        self.assertTrue(str(cs).startswith("Count-Sketch:"))

    def test_cs_add_check_signed(self):
        """Test that signed updates and negative nets are supported"""
        cs = CountSketch(width=1000, depth=5)
        cs.add("sku-1", 10)
        cs.add("sku-1", -4)  # refund
        cs.add("sku-2", -7)  # refund only sku: negative net
        self.assertEqual(cs.check("sku-1"), 6)
        self.assertEqual(cs.check("sku-2"), -7)
        self.assertEqual(cs.elements_added, -1)
        self.assertTrue("sku-1" in cs)
        cs.remove("sku-1", 6)
        self.assertEqual(cs.check("sku-1"), 0)
        self.assertFalse("sku-1" in cs)

    def test_cs_remove_alt(self):
        """Test adding and removing via the hash representation"""
        cs = CountSketch(width=1000, depth=5)
        hashes = cs.hashes("sku-1")
        signs = cs.sign_hashes("sku-1")
        cs.add_alt(hashes, signs, 12)
        self.assertEqual(cs.check_alt(hashes, signs), 12)
        cs.remove_alt(hashes, signs, 20)
        self.assertEqual(cs.check_alt(hashes, signs), -8)

    def test_cs_clear(self):
        """Test clearing the count-sketch"""
        cs = CountSketch(width=100, depth=4)
        cs.add("sku-1", 3)
        cs.clear()
        self.assertEqual(cs.check("sku-1"), 0)
        self.assertEqual(cs.elements_added, 0)

    def test_cs_refund_revision_final_state(self):
        """Test that refund revisions leave the sketch at the final effective
        input under the same hash and sign convention"""
        revised = CountSketch(width=1000, depth=5)
        revised.add("SKU-APPLE", 8)
        revised.add("SKU-APPLE", -2)  # refund
        revised.add("SKU-APPLE", -1)  # revision of the refund
        final_only = CountSketch(width=1000, depth=5)
        final_only.add("SKU-APPLE", 5)
        self.assertEqual(list(revised._bins), list(final_only._bins))
        self.assertEqual(revised.check("SKU-APPLE"), final_only.check("SKU-APPLE"))
        self.assertEqual(revised.elements_added, final_only.elements_added)

    def test_cs_join_matches_single_stream(self):
        """Test that joining per-date partitions equals sketching the final
        effective input directly"""
        joined = build_table_sketch("online", dates={"2026-11-10"})
        joined.join(build_table_sketch("online", dates={"2026-11-11"}))
        single = build_table_sketch("online")
        self.assertEqual(list(joined._bins), list(single._bins))
        self.assertEqual(joined.elements_added, single.elements_added)
        for sku in SKUS:
            self.assertEqual(joined.check(sku), single.check(sku))

    def test_cs_join_incompatible(self):
        """Test that joining mismatched sketches is refused"""
        cs = build_table_sketch("online")
        self.assertRaises(CountMinSketchError, cs.join, build_sketch([], width=8))
        self.assertRaises(CountMinSketchError, cs.join, build_sketch([], depth=4))
        self.assertRaises(
            CountMinSketchError,
            cs.join,
            CountSketch(width=DEMO_WIDTH, depth=DEMO_DEPTH, hash_function=different_hash),
        )
        self.assertRaises(
            CountMinSketchError,
            cs.join,
            CountSketch(
                width=DEMO_WIDTH,
                depth=DEMO_DEPTH,
                hash_function=colliding_hash,
                sign_hash_function=other_sign_hash,
            ),
        )
        self.assertRaises(TypeError, cs.join, CountMinSketch(width=DEMO_WIDTH, depth=DEMO_DEPTH))

    def test_cs_inner_product_exact_without_collisions(self):
        """Test the inner product on a wide, collision free sketch"""
        online = CountSketch(width=1000, depth=5)
        retail = CountSketch(width=1000, depth=5)
        for _, table, sku, amount in EVENTS:
            (online if table == "online" else retail).add(sku, amount)
        res = online.inner_product(retail)
        self.assertEqual(res.estimate, EXACT_INNER_PRODUCT)
        self.assertEqual(res.row_estimates, [EXACT_INNER_PRODUCT] * 5)
        self.assertEqual(res.group_estimates, [float(EXACT_INNER_PRODUCT)] * 5)

    def test_cs_inner_product_collisions(self):
        """Test that per-row estimates, group estimates, and the aggregate
        are all exposed when collisions occur"""
        online = build_table_sketch("online")
        retail = build_table_sketch("retail")
        res = online.inner_product(retail)
        # hand-computed from the known collision pattern
        self.assertEqual(res.row_estimates, [-37, 13, 30])
        self.assertEqual(res.group_estimates, [-37.0, 13.0, 30.0])
        self.assertEqual(res.estimate, 13)
        self.assertEqual(res.estimate, EXACT_INNER_PRODUCT)
        self.assertNotEqual(res.row_estimates[0], EXACT_INNER_PRODUCT)

    def test_cs_inner_product_row_groups(self):
        """Test that the caller-declared independent hash-row groups each
        contribute a single observation"""
        online = build_table_sketch("online")
        retail = build_table_sketch("retail")
        res = online.inner_product(retail, row_groups=[[0, 1], [2]])
        self.assertEqual(res.row_estimates, [-37, 13, 30])
        self.assertEqual(res.group_estimates, [-12.0, 30.0])
        self.assertEqual(res.estimate, 9.0)

    def test_cs_inner_product_row_groups_errors(self):
        """Test that malformed hash-row groups are rejected"""
        online = build_table_sketch("online")
        retail = build_table_sketch("retail")
        self.assertRaises(InitializationError, online.inner_product, retail, [[0, 1]])  # missing row
        self.assertRaises(InitializationError, online.inner_product, retail, [[0, 0, 1, 2]])  # duplicate
        self.assertRaises(InitializationError, online.inner_product, retail, [[0, 1, 3]])  # out of range
        self.assertRaises(InitializationError, online.inner_product, retail, [[], [0, 1, 2]])  # empty

    def test_cs_inner_product_incompatible(self):
        """Test that comparing mismatched sketches is refused since the
        original data is not retained to reconcile them"""
        online = build_table_sketch("online")
        self.assertRaises(CountMinSketchError, online.inner_product, build_sketch([], width=8))
        self.assertRaises(CountMinSketchError, online.inner_product, build_sketch([], depth=4))
        self.assertRaises(
            CountMinSketchError,
            online.inner_product,
            CountSketch(width=DEMO_WIDTH, depth=DEMO_DEPTH, hash_function=different_hash),
        )
        self.assertRaises(
            CountMinSketchError,
            online.inner_product,
            CountSketch(
                width=DEMO_WIDTH,
                depth=DEMO_DEPTH,
                hash_function=colliding_hash,
                sign_hash_function=other_sign_hash,
            ),
        )
        self.assertRaises(TypeError, online.inner_product, "not a count-sketch")

    def test_cs_partition_aggregation(self):
        """Test that cross-date aggregation joins the sketches first; taking
        a median per partition and summing is not the inner product of the
        final effective input"""
        dates = sorted({e[0] for e in EVENTS})
        per_partition = []
        for date in dates:
            res = build_table_sketch("online", dates={date}).inner_product(build_table_sketch("retail", dates={date}))
            per_partition.append(res.estimate)
        median_then_sum = sum(per_partition)

        online = build_table_sketch("online", dates={dates[0]})
        online.join(build_table_sketch("online", dates={dates[1]}))
        retail = build_table_sketch("retail", dates={dates[0]})
        retail.join(build_table_sketch("retail", dates={dates[1]}))
        joined = online.inner_product(retail)

        self.assertEqual(joined.estimate, EXACT_INNER_PRODUCT)
        self.assertNotEqual(median_then_sum, EXACT_INNER_PRODUCT)

    def test_cs_business_report(self):
        """Show the business the exact per-key inner product, the per-row
        estimates, and the aggregate difference on colliding signed data"""
        online = build_table_sketch("online")
        retail = build_table_sketch("retail")
        res = online.inner_product(retail)

        lines = ["", "sku            net(online)  est(online)  net(retail)  est(retail)  exact product"]
        for sku in SKUS:
            lines.append(
                "{:14} {:11d} {:11} {:11d} {:11} {:13d}".format(
                    sku,
                    NETS["online"][sku],
                    online.check(sku),
                    NETS["retail"][sku],
                    retail.check(sku),
                    NETS["online"][sku] * NETS["retail"][sku],
                )
            )
        lines.append(f"exact inner product:   {EXACT_INNER_PRODUCT}")
        lines.append(f"row estimates:         {res.row_estimates}")
        lines.append(f"group estimates:       {res.group_estimates}")
        lines.append(f"aggregate estimate:    {res.estimate}")
        lines.append(f"aggregate difference:  {res.estimate - EXACT_INNER_PRODUCT}")
        report = "\n".join(lines)
        print(report)
        self.assertIn("aggregate difference:  0", report)


if __name__ == "__main__":
    unittest.main()
