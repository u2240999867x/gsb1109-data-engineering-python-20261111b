"""Independent differential regression for two dpath projection plans.

Plan A (flat rows):   dpath.search(obj, glob, yielded=True)
                      -> [(path-string, value), ...]
Plan B (nested tree): dpath.search(obj, glob, yielded=False)
                      -> nested dict/list rebuilt from the same matches

An independent oracle (never calls dpath.segments.match) implements the
documented glob semantics: fnmatch per canonical segment string, '**'
matches zero or more segments. It keeps the raw segment tuple alongside
the canonical path string so that distinct source positions sharing one
string identity stay distinguishable.

Comparison convention: path (multi)set first, then row count, then raw
values. Patterns the library rejects and legal patterns that project to an
empty result are reported in separate buckets.

The input contract is bounded (depth, finite key alphabet, list length) and
enumerated in three tiers:

  Tier 1  exhaustive depth-2 documents over a small alphabet that includes
          text-numeric keys ('0', '1'), crossed with all short patterns.
  Tier 2  depth-3 spine documents crossed with 3-segment '**' patterns.
  Tier 3  explicit collision documents (int vs text keys, shared values,
          leading-zero and negative-index globs, bytes, empty-string keys).

Run:  .venv/bin/python -m differential.diff_regression
"""
from __future__ import annotations

import itertools
from collections import Counter
from fnmatch import fnmatchcase
from typing import Any, Dict, List, Sequence, Tuple

import dpath
from dpath.exceptions import InvalidGlob

# ----------------------------------------------------------------------
# Bounded input contract
# ----------------------------------------------------------------------
KEYS = ("a", "0", "1")                 # text-numeric keys participate
LEAVES = (0, 7, "S")
MAX_DEPTH_EXHAUSTIVE = 2
MAX_LIST_LEN = 2
MAX_DICT_KEYS = 2

PAT_SEGMENTS = ("a", "*", "**", "0", "00", "?")
MAX_FLAT_PATTERN_LEN = 2
CURATED_PATTERNS = [
    ("a", "**", "b"), ("**", "a", "b"), ("a", "**", "*"),
    ("*", "**", "0"), ("a", "*", "**"), ("**", "**", "a"),
]

Path = Tuple[Any, ...]
# Row = (canonical string path, type-aware identity (or None for flat), value)
Row = Tuple[str, Any, Any]

# Mirror dpath.segments.leaf independently: only primitive scalars are leaves.
# Note: empty containers are NOT leaves under this rule, so dirs=False drops
# them even though walk() visits them (the library's own leaves() uses the
# looser leafy() rule instead; that internal inconsistency is reported
# separately rather than counted as a plan-vs-plan divergence).
_LEAF_TYPES = (bytes, str, int, float, bool, type(None))


def is_leaf(value: Any) -> bool:
    return isinstance(value, _LEAF_TYPES)


# ----------------------------------------------------------------------
# Independent oracle (no use of dpath.segments.match)
# ----------------------------------------------------------------------
def canon(seg: Any) -> str:
    return str(seg)


class ListIndex(int):
    """Independent equivalent of dpath's list-index marker: knows its
    container length so that negative indices resolve correctly. Written
    here rather than imported, to keep the oracle independent."""

    def __new__(cls, value: int, length: int):
        obj = int.__new__(cls, value)
        obj.length = length
        return obj

    def __eq__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return int(other) == int(self) or self.length + int(other) == int(self)

    __hash__ = int.__hash__


def hashable_path(raw_path: Path) -> Tuple[Any, ...]:
    """Collapse ListIndex to a plain int (it carries length only for match
    semantics); int and text segments stay distinct source positions."""
    return tuple(int(s) if isinstance(s, int) else s for s in raw_path)


def seg_match(seg: Any, g: Any) -> bool:
    """Single-segment match; type-incompatible pairs never match (mirrors
    dpath's convention that a fnmatch exception means no match)."""
    # Negative numeric glob against a list position: resolve relative length.
    if isinstance(seg, ListIndex) and isinstance(g, str) and g.lstrip("-").isdecimal() and g.startswith("-"):
        return seg == int(g)
    try:
        return fnmatchcase(canon(seg), g)
    except TypeError:
        return False


def ref_match(path: Path, glob: Sequence[str]) -> bool:
    glob = tuple(glob)
    if glob.count("**") > 1:
        raise InvalidGlob(f"only one ** permitted: {glob}")
    if "**" not in glob:
        if len(path) != len(glob):
            return False
        return all(seg_match(s, g) for s, g in zip(path, glob))

    ss = glob.index("**")
    head, tail = glob[:ss], glob[ss + 1:]
    if len(path) < len(head) + len(tail):
        return False
    if not all(seg_match(s, g) for s, g in zip(path[:ss], head)):
        return False
    tail_start = len(path) - len(tail)
    return all(
        seg_match(s, g)
        for s, g in zip(path[tail_start:], tail)
    )


def ref_walk(obj: Any, location: Path = ()) -> List[Tuple[Path, Any]]:
    out: List[Tuple[Path, Any]] = []
    if is_leaf(obj):
        return out
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = [(ListIndex(i, len(obj)), v) for i, v in enumerate(obj)]
    else:
        return out
    for k, v in items:
        out.append((location + (k,), v))
    for k, v in items:
        out.extend(ref_walk(v, location + (k,)))
    return out


def ref_project(walked: List[Tuple[Path, Any]], glob: Sequence[str], dirs: bool) -> List[Row]:
    rows = []
    for raw_path, value in walked:
        if not dirs and not is_leaf(value):
            continue
        if ref_match(raw_path, glob):
            string_path = "/".join(canon(s) for s in raw_path)
            rows.append((string_path, hashable_path(raw_path), value))
    return rows


# ----------------------------------------------------------------------
# Library projection plans
# ----------------------------------------------------------------------
def _as_glob(pattern: Tuple[Any, ...]):
    # Keep bytes segments typed (list-form glob); a str glob cannot hold them.
    if any(isinstance(seg, bytes) for seg in pattern):
        return list(pattern)
    return "/".join(pattern)


def lib_flat(obj: Any, pattern: Tuple[Any, ...], dirs: bool) -> List[Row]:
    # The flat plan only ever emits string column names; it has no way to
    # carry a type-aware identity, so identity is the string itself.
    rows = list(dpath.search(obj, _as_glob(pattern), yielded=True, dirs=dirs))
    return [(path_str, ("str-col", path_str), value) for path_str, value in rows]


def lib_tree(obj: Any, pattern: Tuple[Any, ...], dirs: bool) -> List[Row]:
    tree = dpath.search(obj, _as_glob(pattern), yielded=False, dirs=dirs)
    return [
        ("/".join(canon(s) for s in raw_path), hashable_path(raw_path), value)
        for raw_path, value in ref_walk(tree)
        if dirs or is_leaf(value)
    ]


# ----------------------------------------------------------------------
# Comparison: paths first, then row count, then raw values
# ----------------------------------------------------------------------
def compare_plan(plan: str, lib_rows: List[Row], ref_rows: List[Row],
                 ref_string_to_identities: Dict[str, set]) -> Dict[str, Any]:
    reasons: List[str] = []

    if plan == "flat":
        # Column identity is the canonical string. Compare path multisets.
        lib_paths = Counter(r[0] for r in lib_rows)
        ref_paths = Counter(r[0] for r in ref_rows)
        missing = sorted(set(ref_paths) - set(lib_paths))
        extra = sorted(set(lib_paths) - set(ref_paths))
        dup_strings = sorted(p for p, n in lib_paths.items() if n > 1)
        count_mismatch = sorted(
            p for p in set(lib_paths) & set(ref_paths) if lib_paths[p] != ref_paths[p]
        )
        if missing:
            reasons.append("dropped_columns=%r" % missing)
        if extra:
            reasons.append("phantom_columns=%r" % extra)
        if dup_strings:
            reasons.append("repeated_output_path=%r" % dup_strings)
        if count_mismatch:
            reasons.append("row_count_mismatch=%r" % count_mismatch)
        # Distinct source positions that the flat plan forces into one column.
        collided = sorted(
            p for p, ids in ref_string_to_identities.items()
            if len(ids) > 1 and p in lib_paths
        )
        if collided:
            reasons.append("siblings_share_column_identity=%r" % collided)
        compare_key = 0  # compare on string path
    else:
        # The tree keeps typed keys, so compare on type-aware identities.
        lib_ids = Counter(r[1] for r in lib_rows)
        ref_ids = Counter(r[1] for r in ref_rows)
        missing = sorted(map(_id_to_str, set(ref_ids) - set(lib_ids)))
        extra = sorted(map(_id_to_str, set(lib_ids) - set(ref_ids)))
        count_mismatch = sorted(
            _id_to_str(p) for p in (set(lib_ids) & set(ref_ids))
            if lib_ids[p] != ref_ids[p]
        )
        if missing:
            reasons.append("dropped_positions=%r" % missing)
        if extra:
            reasons.append("phantom_positions=%r" % extra)
        if count_mismatch:
            reasons.append("row_count_mismatch=%r" % count_mismatch)
        compare_key = 1  # compare on typed identity

    if not reasons:
        lib_vals = sorted(((str(r[compare_key]), r[2]) for r in lib_rows), key=lambda x: x[0])
        ref_vals = sorted(((str(r[compare_key]), r[2]) for r in ref_rows), key=lambda x: x[0])
        if lib_vals != ref_vals:
            reasons.append("value_mismatch lib=%r ref=%r" % (lib_vals, ref_vals))
    return {"verdict": "diverge" if reasons else "consistent", "reasons": reasons}


def _id_to_str(identity: Tuple[Any, ...]) -> str:
    return "/".join(map(str, identity))


def classify(doc: Any, pattern: Tuple[str, ...], dirs: bool) -> Tuple[str, Dict[str, Any]]:
    """Return (kind, detail).

    kind in:
      rejected   pattern itself is refused (e.g. two '**' segments)
      plan_error a legal pattern makes one projection plan raise (library bug)
      empty      both plans run but project zero rows (distinct from rejected)
      diverge    path/count/value mismatch against the independent oracle
      consistent both plans agree with the oracle
    """
    # Oracle first: an illegal pattern is a pattern-level rejection.
    try:
        walked = ref_walk(doc)
        ref_rows = ref_project(walked, pattern, dirs)
    except InvalidGlob as exc:
        return "rejected", {"error": f"{type(exc).__name__}: {exc}"}

    # Each plan is exercised independently so one plan's crash is not blamed
    # on the other.
    plan_outcome: Dict[str, Any] = {}
    for plan_name, runner in (("flat", lib_flat), ("tree", lib_tree)):
        try:
            plan_outcome[plan_name] = runner(doc, pattern, dirs)
        except Exception as exc:
            plan_outcome[plan_name] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    errors = {name: o["error"] for name, o in plan_outcome.items() if isinstance(o, dict) and "error" in o}
    if errors:
        return "plan_error", {"errors": errors}

    flat, tree = plan_outcome["flat"], plan_outcome["tree"]
    if not ref_rows and not flat and not tree:
        return "empty", {}

    ref_string_to_identities: Dict[str, set] = {}
    for raw_path, value in walked:
        if not dirs and not is_leaf(value):
            continue
        ref_string_to_identities.setdefault(
            "/".join(canon(s) for s in raw_path), set()
        ).add(hashable_path(raw_path))

    flat_cmp = compare_plan("flat", flat, ref_rows, ref_string_to_identities)
    tree_cmp = compare_plan("tree", tree, ref_rows, ref_string_to_identities)
    if flat_cmp["verdict"] == "diverge" or tree_cmp["verdict"] == "diverge":
        return "diverge", {"flat": flat_cmp, "tree": tree_cmp}
    return "consistent", {}


# ----------------------------------------------------------------------
# Document / pattern enumeration
# ----------------------------------------------------------------------
def gen_documents(depth: int) -> List[Any]:
    if depth <= 0:
        return list(LEAVES)
    docs: List[Any] = list(LEAVES)
    smaller = gen_documents(depth - 1)
    for r in range(1, MAX_DICT_KEYS + 1):
        for keys in itertools.combinations(KEYS, r):
            for vals in itertools.product(smaller, repeat=r):
                docs.append(dict(zip(keys, vals)))
    for length in range(0, MAX_LIST_LEN + 1):
        for vals in itertools.product(smaller, repeat=length):
            docs.append(list(vals))
    seen, unique = set(), []
    for d in docs:
        marker = repr(d)
        if marker not in seen:
            seen.add(marker)
            unique.append(d)
    return unique


def gen_patterns() -> List[Tuple[str, ...]]:
    out = []
    for length in range(1, MAX_FLAT_PATTERN_LEN + 1):
        for combo in itertools.product(PAT_SEGMENTS, repeat=length):
            if combo.count("**") <= 1:
                out.append(combo)
    out.extend(CURATED_PATTERNS)
    return sorted(set(out))


def spine_docs() -> List[Any]:
    """Bounded depth-3 shapes that exercise '**' in the middle."""
    return [
        {"a": {"b": {"c": 1}}},
        {"a": {"x": {"b": 2}}},
        {"a": [{"b": 3}, {"b": 4}]},
        {"a": {"0": {"b": 5}}},
        {"x": {"a": {"b": 6}}},
        {"a": {"b": [7, 8]}},
        {"a": {"b": {"0": 9}}},
    ]


def collision_docs() -> List[Tuple[str, Any]]:
    """Explicit minimal documents targeting identity collisions and quirks."""
    return [
        ("int-vs-text key, distinct values", {0: "x", "0": "y"}),
        ("int-vs-text key, shared value", {0: "S", "0": "S"}),
        ("nested int-vs-text key", {"a": {0: "x", "0": "y"}}),
        ("list index vs sibling text key", {"a": [{"k": 1}], "0": [{"k": 2}]}),
        ("leading-zero glob target (list)", {"a": [10]}),
        ("leading-zero glob target (dict)", {"00": 1, "0": 2}),
        ("negative index target", {"a": [10, 20]}),
        ("bytes key document", {b"k": 1}),
        ("empty-string key document", {"": 1, "a": 2}),
        ("shared-value siblings", {"a": {"k": 7}, "b": {"k": 7}}),
    ]


COLLISION_PATTERNS = [
    ("*",), ("0",), ("00",), ("a", "00"), ("a", "-1"), ("**", "k"),
    ("a", "**", "k"), ("k",), (b"k",), ("a", "*"),
]


# ----------------------------------------------------------------------
# Minimization
# ----------------------------------------------------------------------
def still_bad(doc: Any, pattern: Tuple[str, ...], dirs: bool, kinds: set) -> bool:
    kind, _ = classify(doc, pattern, dirs)
    return kind in kinds


def minimize(doc: Any, pattern: Tuple[str, ...], dirs: bool, kinds: set) -> Any:
    current = doc
    changed = True
    while changed:
        changed = False
        if isinstance(current, dict):
            for key in list(current):
                trial = {k: v for k, v in current.items() if k != key}
                if trial and still_bad(trial, pattern, dirs, kinds):
                    current, changed = trial, True
                    break
        if isinstance(current, list):
            for i in range(len(current)):
                trial = current[:i] + current[i + 1:]
                if still_bad(trial, pattern, dirs, kinds):
                    current, changed = trial, True
                    break
        if changed:
            continue
        if isinstance(current, dict):
            for key in list(current):
                child = current[key]
                if isinstance(child, (dict, list)) and child:
                    shrunk = minimize(child, pattern, dirs, kinds)
                    if shrunk is not child:
                        trial = dict(current)
                        trial[key] = shrunk
                        if still_bad(trial, pattern, dirs, kinds):
                            current, changed = trial, True
                            break
    return current


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_suite(name: str, docs: List[Any], patterns: List[Tuple[str, ...]],
              dirs: bool, buckets: Dict[str, list]) -> None:
    for doc in docs:
        for pattern in patterns:
            kind, detail = classify(doc, pattern, dirs)
            buckets[kind].append((name, doc, pattern, detail))


def main() -> None:
    dirs = False
    print("Tier 1: exhaustive depth-%d documents ..." % MAX_DEPTH_EXHAUSTIVE)
    docs = gen_documents(MAX_DEPTH_EXHAUSTIVE)
    patterns = gen_patterns()
    print("  docs=%d patterns=%d pairs=%d" %
          (len(docs), len(patterns), len(docs) * len(patterns)))

    buckets: Dict[str, list] = {
        "diverge": [], "rejected": [], "plan_error": [],
        "empty": [], "consistent": [],
    }
    run_suite("tier1-exhaustive", docs, patterns, dirs, buckets)
    run_suite("tier2-spine", spine_docs(),
              sorted(set(patterns + CURATED_PATTERNS)), dirs, buckets)
    for label, doc in collision_docs():
        for pattern in COLLISION_PATTERNS:
            kind, detail = classify(doc, pattern, dirs)
            buckets[kind].append(("tier3:" + label, doc, pattern, detail))

    total = sum(len(v) for v in buckets.values())
    print("\ntotal pairs      : %d" % total)
    print("consistent pairs : %d" % len(buckets["consistent"]))
    print("legal-empty pairs: %d" % len(buckets["empty"]))
    print("rejected pairs   : %d" % len(buckets["rejected"]))
    print("plan-error pairs : %d" % len(buckets["plan_error"]))
    print("divergent pairs  : %d" % len(buckets["diverge"]))

    # --- rejected patterns, separate from empty ---
    print("\n=== Rejected patterns (library refuses; distinct from empty output) ===")
    rej: Counter = Counter()
    for _, _, pattern, detail in buckets["rejected"]:
        rej[("/".join(map(str, pattern)), detail["error"])] += 1
    for (pattern, error), n in sorted(rej.items(), key=lambda kv: -kv[1]):
        print("  %-14s x%-6d %s" % (pattern, n, error))

    print("\n=== Legal patterns projecting to empty (distinct from rejections) ===")
    emp = Counter("/".join(map(str, p)) for _, _, p, _ in buckets["empty"])
    for pattern, n in emp.most_common(10):
        print("  %-12s x%d" % (pattern, n))

    # --- plan crashes, collapsed to defect classes ---
    print("\n=== Plan crashes (legal pattern, a projection raises) ===")
    crash_classes: Dict[str, Dict[str, Any]] = {}
    for suite, doc, pattern, detail in buckets["plan_error"]:
        for plan, error in detail["errors"].items():
            if "ListIndex" in error:
                name = "C1 tree rebuild cannot key a root-level list (unhashable ListIndex)"
            elif "Empty string keys" in error:
                name = "C2 empty-string key rejected unless options flag is set"
            elif "bytes-like" in error or "bytes found" in error:
                name = "C3 bytes glob: flat plan cannot join a bytes path; bytes-object glob breaks both plans"
            else:
                name = "Cx other: " + error
            entry = crash_classes.setdefault(name, {"count": 0, "examples": {}})
            entry["count"] += 1
            minimal = minimize(doc, pattern, dirs, {"plan_error"})
            entry["examples"].setdefault(
                repr(minimal), (minimal, "/".join(map(str, pattern)), plan, suite))
    for name, entry in sorted(crash_classes.items()):
        print("\n%s  (%d plan/pair occurrences)" % (name, entry["count"]))
        for rep, (doc, pattern, plan, suite) in list(entry["examples"].items())[:2]:
            print("  min: doc=%s pattern=%r failing_plan=%s [%s]" % (rep, pattern, plan, suite))

    # --- cross-plan equivalence + shared deviation ---
    print("\n=== Cross-plan equivalence (the two 'equivalent' projections) ===")
    print("Each running pair is normalized to a multiset of (string-path, value).")
    cross_classes: Dict[str, Dict[str, Any]] = {}

    def cross_bucket(name: str, suite: str, doc: Any, pattern: Tuple[str, ...],
                     note: str, kind: str) -> None:
        entry = cross_classes.setdefault(name, {"count": 0, "kind": kind, "examples": {}})
        entry["count"] += 1
        minimal = minimize(doc, pattern, dirs, {"diverge"})
        entry["examples"].setdefault(
            repr(minimal), (minimal, "/".join(map(str, pattern)), note, suite))

    for suite, doc, pattern, detail in buckets["diverge"]:
        flat_r = detail["flat"]["reasons"]
        tree_r = detail["tree"]["reasons"]
        flat_bad, tree_bad = bool(flat_r), bool(tree_r)

        # Recompute normalized rows for a direct plan-vs-plan comparison.
        flat_norm = Counter((p, v) for p, _, v in lib_flat(doc, pattern, dirs))
        tree_norm = Counter((p, v) for p, _, v in lib_tree(doc, pattern, dirs))
        plans_disagree = flat_norm != tree_norm

        if any("siblings_share_column_identity" in r for r in flat_r):
            cross_bucket(
                "E1 flat plan maps distinct int/text source positions to one column; "
                "tree keeps typed keys (dict(flat) overwrites one source value)",
                suite, doc, pattern,
                "flat emits repeated identical path strings; a column-keyed consumer "
                "e.g. dict(rows) silently drops one position",
                "plan-vs-plan",
            )
        elif plans_disagree and any("phantom_positions" in r for r in tree_r):
            cross_bucket(
                "E2 tree plan injects None padding rows absent from the flat plan",
                suite, doc, pattern,
                "tree=" + " | ".join(tree_r),
                "plan-vs-plan",
            )
        elif any("dropped_columns" in r for r in flat_r) and "b'k'" in " ".join(flat_r):
            cross_bucket(
                "E3 bytes keys invisible to str globs (both plans silently drop)",
                suite, doc, pattern,
                "flat=" + " | ".join(flat_r),
                "shared-deviation",
            )
        elif not plans_disagree and (flat_bad or tree_bad):
            cross_bucket(
                "E4 both plans agree with each other but deviate from reference "
                "(numeric text glob coerced to list index)",
                suite, doc, pattern,
                "flat=" + " | ".join(flat_r) + " | tree=" + " | ".join(tree_r),
                "shared-deviation",
            )
        elif plans_disagree:
            cross_bucket(
                "E9 other plan-vs-plan disagreement",
                suite, doc, pattern,
                "flat=" + " | ".join(flat_r) + " | tree=" + " | ".join(tree_r),
                "plan-vs-plan",
            )

    for name, entry in sorted(cross_classes.items()):
        print("\n[%s] %s  (%d pairs)" % (entry["kind"].upper(), name, entry["count"]))
        for rep, (doc, pattern, note, suite) in list(entry["examples"].items())[:2]:
            print("  min: doc=%s pattern=%r [%s]" % (rep, pattern, suite))
            print("       %s" % note)


if __name__ == "__main__":
    main()
