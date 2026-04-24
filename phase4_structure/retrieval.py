"""P4.3 — Structural retrieval + equivalence-class clustering.

Operates on the StructuralSignature objects from P4.2 (signature.py).

Given a library {domain_name → StructuralSignature}, we support:

  signature_distance(a, b)            — Ruzicka (weighted Jaccard) on the
                                        op multiset, ∈ [0, 1]. 0 iff op
                                        multisets are identical.
  nearest_neighbor(target, library)   — argmin distance (excluding self)
  cluster_signatures(sigs, tau=0.5)   — connected-component clustering on
                                        the threshold graph {(i,j) :
                                        distance < tau}
  adjusted_rand_index(y_true, y_pred) — ARI as per Hubert & Arabie 1985

Plan §Phase 5 task 2 target: ARI > 0.8 for equivalence-class recovery
on the law-zoo. We expect 1.0 here (7/7 domains correctly clustered
into their 2 classes) — the MVP op-multiset signature is exact for
this fixture. Richer representations (GNN / Markov-category) would be
needed to handle softer isomorphism classes and are deferred.
"""

from __future__ import annotations

from collections import Counter
from math import comb

from phase4_structure.signature import StructuralSignature


def signature_distance(a: StructuralSignature, b: StructuralSignature) -> float:
    """Ruzicka (weighted-Jaccard) distance on op multisets.

    d = 1 - Σ min(c_a(op), c_b(op)) / Σ max(c_a(op), c_b(op))
    """
    ca, cb = Counter(a.ops), Counter(b.ops)
    ops = set(ca) | set(cb)
    if not ops:
        return 0.0
    num = sum(min(ca[op], cb[op]) for op in ops)
    den = sum(max(ca[op], cb[op]) for op in ops)
    return 1.0 - num / den if den else 0.0


def nearest_neighbor(target_name: str,
                     library: dict[str, StructuralSignature]) -> tuple[str, float]:
    """Argmin distance over library, excluding target itself."""
    if target_name not in library:
        raise KeyError(target_name)
    target = library[target_name]
    best_name, best_d = None, float("inf")
    for name, sig in library.items():
        if name == target_name:
            continue
        d = signature_distance(target, sig)
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


def cluster_signatures(library: dict[str, StructuralSignature],
                       mode: str = "fingerprint",
                       tau: float = 0.05) -> dict[str, int]:
    """Cluster a signature library into equivalence classes.

    mode='fingerprint' (default): bucket by StructuralSignature.fingerprint
        — the canonical equivalence signal from P4.2. Gives 0.0 intra /
        non-zero inter separation by construction.

    mode='threshold': connected-component clustering on the tau-threshold
        distance graph {(i,j) : distance(i,j) < tau}. Useful if the
        signature library contains *near-isomorphs* we want to lump
        together; the law-zoo's cross-class Ruzicka distance is ≈0.15
        (only the Sub op differs), so tau in (0, 0.15) keeps classes
        separate, tau > 0.15 merges them.

    Returns: {domain_name → cluster_id (0-indexed, stable order)}.
    """
    names = list(library.keys())

    if mode == "fingerprint":
        fp_to_id: dict[str, int] = {}
        labels = {}
        for name in names:
            fp = library[name].fingerprint
            if fp not in fp_to_id:
                fp_to_id[fp] = len(fp_to_id)
            labels[name] = fp_to_id[fp]
        return labels

    if mode != "threshold":
        raise ValueError(f"unknown mode {mode!r}")

    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if signature_distance(library[names[i]], library[names[j]]) < tau:
                union(i, j)

    roots = {}
    labels = {}
    next_id = 0
    for i, name in enumerate(names):
        r = find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        labels[name] = roots[r]
    return labels


def adjusted_rand_index(y_true: list, y_pred: list) -> float:
    """ARI per Hubert & Arabie 1985. Returns 1.0 for identical partitions
    (up to relabelling), 0.0 for random chance."""
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    n = len(y_true)
    if n == 0:
        return 1.0

    pairs_t = sorted(set(y_true))
    pairs_p = sorted(set(y_pred))
    contingency = {(t, p): 0 for t in pairs_t for p in pairs_p}
    for t, p in zip(y_true, y_pred):
        contingency[(t, p)] += 1

    sum_ij = sum(comb(v, 2) for v in contingency.values())
    row_sums = {t: sum(contingency[(t, p)] for p in pairs_p) for t in pairs_t}
    col_sums = {p: sum(contingency[(t, p)] for t in pairs_t) for p in pairs_p}
    sum_a = sum(comb(v, 2) for v in row_sums.values())
    sum_b = sum(comb(v, 2) for v in col_sums.values())
    total = comb(n, 2)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    maxv = 0.5 * (sum_a + sum_b)
    if maxv - expected == 0:
        return 1.0 if sum_ij == expected else 0.0
    return (sum_ij - expected) / (maxv - expected)
