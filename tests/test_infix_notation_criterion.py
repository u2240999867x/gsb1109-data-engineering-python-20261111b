#
# test_infix_notation_criterion.py
#
# Self-contained criterion for a cleansing "computed column" whose formula is
# declared with pyparsing.infix_notation. The grammar uses an explicit finite
# field domain and explicit prefix/suffix/left/right associativity
# declarations. Every assertion checks one of:
#   - the expression tree that is finally adopted,
#   - the source columns that are actually fetched,
#   - the row value computed from a concrete field assignment,
#   - the parse action trace, split into delivered work and attempts that
#     were tried and then backtracked.
#
# The same checks are run in ordinary mode and in packrat mode.
#
import contextlib
import itertools
import math
import unittest

import pyparsing as pp
from pyparsing import ParserElement

# finite, declared value domain for the fields a, b, c
DOMAIN = (0, 1, 2)

# one concrete row assignment used for delivered-value checks
ROW = {"a": 10, "b": 3, "c": 2}

MODES = ("normal", "packrat")


@contextlib.contextmanager
def memoization(mode):
    if mode == "packrat":
        ParserElement.enable_packrat(force=True)
    else:
        ParserElement.disable_memoization()
    try:
        yield
    finally:
        ParserElement.disable_memoization()


def build_subtraction_grammar(assoc, read_log, fail_log):
    """Prefix '-', suffix '!', and binary '-' with the associativity under
    test. Column fetches and failed attempts are recorded separately."""
    number = pp.Word(pp.nums).set_parse_action(pp.token_map(int))
    column = pp.Word(pp.alphas)
    column.set_parse_action(lambda t: read_log.append(t[0]) or t[0])
    column.set_fail_action(
        lambda s, loc, expr, exc: fail_log.append(("column", loc))
    )
    atom = column | number

    minus = pp.Literal("-")
    minus.set_fail_action(
        lambda s, loc, expr, exc: fail_log.append(("'-'", loc))
    )
    bang = pp.Literal("!")
    bang.set_fail_action(
        lambda s, loc, expr, exc: fail_log.append(("'!'", loc))
    )

    return pp.infix_notation(
        atom,
        [
            (bang, 1, pp.OpAssoc.LEFT),    # suffix postfix operator
            (minus, 1, pp.OpAssoc.RIGHT),  # prefix unary operator
            (minus, 2, assoc),             # binary operator, assoc under test
        ],
    )


def build_binary_grammar(assoc, symbol, read_log):
    number = pp.Word(pp.nums).set_parse_action(pp.token_map(int))
    column = pp.Word(pp.alphas)
    column.set_parse_action(lambda t: read_log.append(t[0]) or t[0])
    op = pp.Literal(symbol)
    atom = column | number
    return pp.infix_notation(atom, [(op, 2, assoc)])


def evaluate(node, row):
    """Reference evaluation of the tree returned by the grammars above."""
    if isinstance(node, int):
        return node
    if isinstance(node, str):
        return row[node]
    if isinstance(node, list):
        if len(node) == 2 and node[0] == "-":
            return -evaluate(node[1], row)
        if len(node) == 2 and node[1] == "!":
            return math.factorial(evaluate(node[0], row))
        value = evaluate(node[0], row)
        for index in range(1, len(node), 2):
            operand = evaluate(node[index + 1], row)
            if node[index] == "-":
                value -= operand
            else:
                value += operand
        return value
    raise AssertionError(node)


def parse(mode, assoc, text, symbol="-"):
    reads = []
    fails = []
    with memoization(mode):
        if symbol == "-":
            grammar = build_subtraction_grammar(assoc, reads, fails)
        else:
            grammar = build_binary_grammar(assoc, symbol, reads)
        ParserElement.packrat_cache_stats[:] = [0, 0]
        tree = grammar.parse_string(text, parse_all=True).as_list()
        stats = tuple(ParserElement.packrat_cache_stats)
    return tree, reads, fails, stats


class TestInfixNotationComputedColumnCriterion(unittest.TestCase):
    def test_declared_associativity_selects_the_adopted_tree(self):
        expected_tree = {
            pp.OpAssoc.LEFT: [["a", "-", "b", "-", "c"]],
            pp.OpAssoc.RIGHT: [["a", "-", ["b", "-", "c"]]],
        }
        expected_value = {pp.OpAssoc.LEFT: 5, pp.OpAssoc.RIGHT: 9}
        for mode in MODES:
            for assoc in (pp.OpAssoc.LEFT, pp.OpAssoc.RIGHT):
                with self.subTest(mode=mode, assoc=assoc.name):
                    tree, reads, _, stats = parse(mode, assoc, "a-b-c")
                    self.assertEqual(tree, expected_tree[assoc])
                    self.assertEqual(evaluate(tree[0], ROW), expected_value[assoc])
                    # the same three source columns are fetched, in order,
                    # regardless of how they are grouped
                    self.assertEqual(reads, ["a", "b", "c"])
                    if mode == "normal":
                        self.assertEqual(stats, (0, 0))
                    else:
                        self.assertGreater(stats[0], 0)

    def test_field_assignment_distinguishes_the_two_trees(self):
        # parse once per associativity: the tree itself is static, only the
        # field values change across rows
        trees = {}
        for assoc in (pp.OpAssoc.LEFT, pp.OpAssoc.RIGHT):
            tree, _, _, _ = parse("normal", assoc, "a-b-c")
            trees[assoc] = tree[0]

        diverging = []
        always_equal = True
        for values in itertools.product(DOMAIN, repeat=3):
            row = dict(zip("abc", values))
            left_value = evaluate(trees[pp.OpAssoc.LEFT], row)
            right_value = evaluate(trees[pp.OpAssoc.RIGHT], row)
            if left_value != right_value:
                diverging.append((values, left_value, right_value))
                always_equal = False

        # subtraction is NOT equivalent under the two groupings: 18 of the
        # 27 finite-domain assignments distinguish the trees
        self.assertFalse(always_equal)
        self.assertEqual(len(diverging), 18)
        # explicit distinguishing assignment: (a, b, c) = (1, 0, 1)
        self.assertIn(((1, 0, 1), 0, 2), diverging)
        # within the declared domain the values coincide exactly when c == 0,
        # i.e. a-b-c == a-(b-0) only when the moved parenthesis is harmless
        for values in itertools.product(DOMAIN, repeat=3):
            row = dict(zip("abc", values))
            same = (
                evaluate(trees[pp.OpAssoc.LEFT], row)
                == evaluate(trees[pp.OpAssoc.RIGHT], row)
            )
            self.assertEqual(same, row["c"] == 0)

    def test_plus_is_always_equivalent_over_the_domain(self):
        trees = {}
        for assoc in (pp.OpAssoc.LEFT, pp.OpAssoc.RIGHT):
            tree, reads, _, _ = parse("normal", assoc, "a+b+c", symbol="+")
            trees[assoc] = tree[0]
            self.assertEqual(reads, ["a", "b", "c"])
        self.assertEqual(trees[pp.OpAssoc.LEFT], ["a", "+", "b", "+", "c"])
        self.assertEqual(trees[pp.OpAssoc.RIGHT], ["a", "+", ["b", "+", "c"]])

        # even though the trees differ, no assignment in the declared domain
        # can tell them apart: '+' is associative, so flag the equivalence
        for values in itertools.product(DOMAIN, repeat=3):
            row = dict(zip("abc", values))
            self.assertEqual(
                evaluate(trees[pp.OpAssoc.LEFT], row),
                evaluate(trees[pp.OpAssoc.RIGHT], row),
            )
        self.assertEqual(
            evaluate(trees[pp.OpAssoc.LEFT], ROW),
            evaluate(trees[pp.OpAssoc.RIGHT], ROW),
        )
        self.assertEqual(evaluate(trees[pp.OpAssoc.LEFT], ROW), 15)

    def test_prefix_suffix_and_parentheses(self):
        cases = [
            ("-a", [["-", "a"]], -10, ["a"]),
            ("a!", [["a", "!"]], math.factorial(10), ["a"]),
            ("--a", [["-", ["-", "a"]]], 10, ["a"]),
            ("-(a-b)", [["-", ["a", "-", "b"]]], -7, ["a", "b"]),
            ("a!-b", [[["a", "!"], "-", "b"]], math.factorial(10) - 3, ["a", "b"]),
            ("(a-b)-c", [[["a", "-", "b"], "-", "c"]], 5, ["a", "b", "c"]),
            ("a-(b-c)", [["a", "-", ["b", "-", "c"]]], 9, ["a", "b", "c"]),
        ]
        for mode in MODES:
            for text, expected_tree, expected_value, expected_reads in cases:
                with self.subTest(mode=mode, text=text):
                    tree, reads, _, _ = parse(mode, pp.OpAssoc.LEFT, text)
                    self.assertEqual(tree, expected_tree)
                    self.assertEqual(evaluate(tree[0], ROW), expected_value)
                    self.assertEqual(reads, expected_reads)

    def test_backtracked_attempt_is_not_a_delivered_row(self):
        # the leading '123' is a numeric atom, so the column alternative is
        # tried at loc 0, fails, and is backtracked; only 'a' is delivered
        expected = {
            "normal": {"column@0": 4, "bang@3": 2, "fails": 13},
            "packrat": {"column@0": 2, "bang@3": 1, "fails": 7},
        }
        for mode in MODES:
            with self.subTest(mode=mode):
                tree, reads, fails, stats = parse(
                    mode, pp.OpAssoc.LEFT, "123-a"
                )
                self.assertEqual(tree, [[123, "-", "a"]])
                self.assertEqual(evaluate(tree[0], ROW), 113)
                # delivered column fetches: the failed column attempt at
                # loc 0 must never appear here, only the real 'a'
                self.assertEqual(reads, ["a"])
                # the attempt itself is visible in the failure trace
                self.assertIn(("column", 0), fails)
                self.assertEqual(fails.count(("column", 0)), expected[mode]["column@0"])
                self.assertEqual(fails.count(("'!'", 3)), expected[mode]["bang@3"])
                self.assertEqual(len(fails), expected[mode]["fails"])
                if mode == "normal":
                    self.assertEqual(stats, (0, 0))
                else:
                    # packrat replays the memoized failure instead of
                    # re-executing it, so fewer attempts are recorded
                    self.assertGreater(stats[0], 0)

    def test_bare_atom_lookahead_attempts_are_backtracked(self):
        expected_fail_count = {"normal": 5, "packrat": 3}
        for mode in MODES:
            with self.subTest(mode=mode):
                tree, reads, fails, _ = parse(mode, pp.OpAssoc.LEFT, "a")
                self.assertEqual(tree, ["a"])
                self.assertEqual(evaluate(tree[0], ROW), 10)
                self.assertEqual(reads, ["a"])
                # operator lookaheads were attempted, found nothing, and
                # backtracked; they do not alter the delivered row
                self.assertTrue(fails)
                self.assertEqual(len(fails), expected_fail_count[mode])

    def test_packrat_memoizes_within_a_parse_but_resets_between_parses(self):
        with memoization("packrat"):
            reads, fails = [], []
            grammar = build_subtraction_grammar(
                pp.OpAssoc.LEFT, reads, fails
            )
            text = "123-a"

            # two raw parse entries without resetting: the second reuses the
            # memoized result, no extra column is fetched
            ParserElement.reset_cache()
            ParserElement.packrat_cache_stats[:] = [0, 0]
            first_loc, first_tokens = grammar._parse(text, 0)
            stats_after_first = tuple(ParserElement.packrat_cache_stats)
            reads_after_first = list(reads)
            second_loc, second_tokens = grammar._parse(text, 0)
            stats_after_second = tuple(ParserElement.packrat_cache_stats)
            self.assertEqual(first_loc, second_loc)
            self.assertEqual(first_tokens.as_list(), second_tokens.as_list())
            self.assertGreater(
                stats_after_second[0], stats_after_first[0]
            )
            self.assertEqual(reads, reads_after_first)

            # parse_string resets the cache on every call, so each call
            # re-runs the grammar from scratch and performs the same fetches
            reads_before = len(reads)
            first_call = grammar.parse_string(text, parse_all=True)
            reads_from_first = list(reads[reads_before:])
            stats_after_first_call = tuple(ParserElement.packrat_cache_stats)

            reads_before = len(reads)
            second_call = grammar.parse_string(text, parse_all=True)
            reads_from_second = list(reads[reads_before:])
            stats_after_second_call = tuple(ParserElement.packrat_cache_stats)

            self.assertEqual(first_call.as_list(), second_call.as_list())
            # the column is fetched again: nothing is reused across the reset
            self.assertEqual(reads_from_first, reads_from_second)
            # each call starts from an empty cache, so it ends with the same
            # miss/hit counts instead of accumulating further hits
            self.assertEqual(
                stats_after_first_call, stats_after_second_call
            )


if __name__ == "__main__":
    unittest.main()
