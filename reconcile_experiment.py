"""对照 pydash.difference_with 的实际匹配与全局最优一对一配对。

读数用 {"line": 来源行号, "v": 读数} 表示；容差 TOL 内视为可配对。
全局口径： bipartite 匹配，先最大化配对数，再最小化总偏差 sum(|a-b|)。
"""
from itertools import product

import pydash as pyd

TOL = 6.0


def within(a, b):
    return abs(a["v"] - b["v"]) <= TOL


def pydash_residual(A, B):
    """pydash.difference_with 返回的剩余 A 侧行号（运营口径的“漏收”）。"""
    return [x["line"] for x in pyd.difference_with(A, B, comparator=within)]


def pydash_first_fit(A, B):
    """复刻 iterdifference 的逐条扫描：每个 A 独立在完整 B 中找第一个可行项。

    注意 B 不会被消耗，同一 B 行可被多个 A 行命中。
    """
    pairs = []
    for a in A:
        for b in B:
            if within(a, b):
                pairs.append((a["line"], b["line"], round(abs(a["v"] - b["v"]), 6)))
                break
    return pairs


def global_optima(A, B):
    """暴力枚举所有可行匹配，返回 (边集, 最大配对数, 最小总偏差, 全部最优匹配)。"""
    nA, nB = len(A), len(B)
    edges = [
        (i, j, round(abs(A[i]["v"] - B[j]["v"]), 6))
        for i, j in product(range(nA), range(nB))
        if within(A[i], B[j])
    ]
    results = []

    def rec(i, used, pairs, cost):
        if i == nA:
            results.append((list(pairs), round(cost, 6)))
            return
        rec(i + 1, used, pairs, cost)  # A[i] 不配对
        for j in range(nB):
            if j in used:
                continue
            d = abs(A[i]["v"] - B[j]["v"])
            if d <= TOL:
                used.add(j)
                pairs.append((i, j, round(d, 6)))
                rec(i + 1, used, pairs, cost + d)
                pairs.pop()
                used.remove(j)

    rec(0, set(), [], 0.0)
    max_card = max(len(p) for p, _ in results)
    candidates = [(p, c) for p, c in results if len(p) == max_card]
    min_cost = min(c for _, c in candidates)
    opts = [(p, c) for p, c in candidates if c == min_cost]
    return edges, max_card, min_cost, opts


def summarize(name, A, B):
    print(f"==== {name} ====")
    print("A 设备:", [(a["line"], a["v"]) for a in A])
    print("B 设备:", [(b["line"], b["v"]) for b in B], f" 容差={TOL}")

    residual = pydash_residual(A, B)
    ff = pydash_first_fit(A, B)
    print("[pydash difference_with] 剩余行:", residual, " 漏收数=", len(residual))
    print("[pydash 逐条首中配对]", ff)
    reuse = {}
    for _, bline, _ in ff:
        reuse[bline] = reuse.get(bline, 0) + 1
    print("  B 行被抵扣次数:", reuse, "-> 超过 1 即违反“一条最多抵扣一条”")

    edges, max_card, min_cost, opts = global_optima(A, B)
    print("[全局] 可行边:", [(A[i]["line"], B[j]["line"], d) for i, j, d in edges])
    print(f"[全局] 最大配对数={max_card}  最小总偏差={min_cost}  最优解个数={len(opts)}")
    unmatched_per_opt = []
    for k, (pairs, cost) in enumerate(opts, 1):
        matched_a = {i for i, _, _ in pairs}
        unmatched = [A[i]["line"] for i in range(len(A)) if i not in matched_a]
        unmatched_per_opt.append(set(unmatched))
        pretty = [(A[i]["line"], B[j]["line"], d) for i, j, d in pairs]
        print(f"  最优解{k}: 配对={pretty} 总偏差={cost} 未配A行={unmatched}")

    always = set.intersection(*unmatched_per_opt) if unmatched_per_opt else set()
    ever = set.union(*unmatched_per_opt) if unmatched_per_opt else set()
    print("  必然未配(每个最优解都未配):", sorted(always))
    print("  可能未配(部分最优解未配):", sorted(ever - always))
    print("  pydash漏收数 vs 全局漏收数:", len(residual), "vs", len(A) - max_card)
    print()


# 例 1：局部首中（B 可重复抵扣）与全局一对一给出不同漏收数
A1 = [
    {"line": "A-L1", "v": 15.0},
    {"line": "A-L2", "v": 13.0},
    {"line": "A-L3", "v": 12.0},
]
B1 = [
    {"line": "B-R1", "v": 10.0},
    {"line": "B-R2", "v": 20.0},
]
summarize("例1 局部 vs 全局，漏收数不同", A1, B1)

# 例 2：最大配对数与最小总偏差都相同，但归属（哪条未配/谁抵谁）不同
A2 = [
    {"line": "A-L1", "v": 10.0},
    {"line": "A-L2", "v": 20.0},
    {"line": "A-L3", "v": 10.0},
]
B2 = [
    {"line": "B-R1", "v": 15.0},
    {"line": "B-R2", "v": 15.0},
]
summarize("例2 总偏差同为最小但归属不同", A2, B2)

# 例 3：存在孤立行（容差内无任何候选），两种口径一致
A3 = [
    {"line": "A-L1", "v": 100.0},
    {"line": "A-L2", "v": 15.0},
]
B3 = [
    {"line": "B-R1", "v": 10.0},
]
summarize("例3 孤立行必然未配", A3, B3)

# 方向不对称性验证：交换两套设备，difference_with 的漏收数会变
print("==== 方向性检查 ====")
Ad = [{"line": "A-L1", "v": 10.0}, {"line": "A-L2", "v": 100.0}]
Bd = [{"line": "B-R1", "v": 10.0}]
print("A 侧剩余:", pydash_residual(Ad, Bd), " 反向 B 侧剩余:", pydash_residual(Bd, Ad))
