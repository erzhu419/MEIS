"""Validation for P4.3 retrieval + clustering.

Acceptance:
    1. signature_distance is 0 iff op multisets are identical; positive
       across classes.
    2. For every exp_decay domain, nearest_neighbor is another exp_decay.
    3. For every saturation domain, nearest_neighbor is another saturation.
    4. cluster_signatures on the full 7-domain library yields 2 clusters
       matching CLASS_OF. ARI = 1.0 (Plan §Phase 5 task 2 target > 0.8).
    5. ARI sanity: random labels on a 7-sample 2-class problem gives
       ARI near 0; identical labels give 1.0.

Run:
    python -m phase4_structure.tests.test_retrieval
"""

from __future__ import annotations

import numpy as np

from phase4_structure.law_zoo import (
    DOMAINS, CLASS_OF, exp_decay, saturation, damped_oscillation,
)
from phase4_structure.signature import signature_for_domain
from phase4_structure.retrieval import (
    signature_distance, nearest_neighbor, cluster_signatures,
    adjusted_rand_index,
)


def _build_library():
    lib = {}
    rng = np.random.default_rng(0)
    for name, cls in CLASS_OF.items():
        if cls == "exp_decay":
            t = exp_decay.default_t_grid(name, n=10)
        elif cls == "saturation":
            t = saturation.default_t_grid(name, n=10)
        else:
            t = damped_oscillation.default_t_grid(name, n=10)
        y = DOMAINS[name].simulate(t, rng)
        lib[name] = signature_for_domain(DOMAINS[name], t, y)
    return lib


def test_signature_distance_zero_within_class_positive_across():
    lib = _build_library()
    for cls in set(CLASS_OF.values()):
        names = [n for n, c in CLASS_OF.items() if c == cls]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = signature_distance(lib[names[i]], lib[names[j]])
                assert d == 0.0, f"{names[i]} vs {names[j]}: d={d}"
    # Across pairs
    d_ed_sat = signature_distance(lib["rc_circuit"], lib["capacitor_charging"])
    d_ed_dmp = signature_distance(lib["rc_circuit"], lib["rlc_circuit"])
    d_sat_dmp = signature_distance(lib["capacitor_charging"], lib["rlc_circuit"])
    for name, d in [("ed/sat", d_ed_sat), ("ed/dmp", d_ed_dmp), ("sat/dmp", d_sat_dmp)]:
        assert d > 0, f"{name} cross-class distance should be > 0, got {d}"
    print(f"[PASS] signature_distance intra-class 0.0, inter-class "
          f"ed/sat={d_ed_sat:.3f}, ed/dmp={d_ed_dmp:.3f}, sat/dmp={d_sat_dmp:.3f}")


def test_nearest_neighbor_is_always_same_class():
    lib = _build_library()
    for name in lib:
        nn_name, nn_d = nearest_neighbor(name, lib)
        assert CLASS_OF[nn_name] == CLASS_OF[name], \
            f"{name} (class {CLASS_OF[name]}) nearest is {nn_name} (class {CLASS_OF[nn_name]})"
    print(f"[PASS] nearest_neighbor preserves equivalence class on {len(lib)}/{len(lib)} domains")


def test_cluster_recovers_ground_truth_ari_1():
    lib = _build_library()
    names = list(lib.keys())
    cls_order = {cls: i for i, cls in enumerate(sorted(set(CLASS_OF.values())))}
    y_true = [cls_order[CLASS_OF[n]] for n in names]

    # Fingerprint mode
    labels_fp = cluster_signatures(lib, mode="fingerprint")
    y_pred_fp = [labels_fp[n] for n in names]
    ari_fp = adjusted_rand_index(y_true, y_pred_fp)
    assert ari_fp == 1.0, f"fingerprint-mode ARI={ari_fp}"
    assert len(set(y_pred_fp)) == 3

    # Threshold mode with tau small enough to keep all 3 classes separate
    labels_th = cluster_signatures(lib, mode="threshold", tau=0.05)
    y_pred_th = [labels_th[n] for n in names]
    ari_th = adjusted_rand_index(y_true, y_pred_th)
    assert ari_th == 1.0, f"threshold-mode ARI={ari_th}"

    print(f"[PASS] cluster_signatures recovers 3 ground-truth classes (n={len(names)}):"
          f" ARI(fingerprint)={ari_fp:.3f}, ARI(threshold,τ=0.05)={ari_th:.3f} "
          f"(Plan §Phase 5 task 2 target > 0.8)")


def test_ari_sanity():
    # Perfect agreement
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    # Perfect agreement with label swap
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    # Worst-case anti-correlated on small sample
    bad = adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1])
    assert bad < 0.5
    print(f"[PASS] ARI sanity: perfect→1.0, label-swap→1.0, anti→{bad:.3f}")


if __name__ == "__main__":
    print("=== P4.3 retrieval + clustering validation ===\n")
    test_signature_distance_zero_within_class_positive_across()
    test_nearest_neighbor_is_always_same_class()
    test_cluster_recovers_ground_truth_ari_1()
    test_ari_sanity()
    print("\nAll P4.3 retrieval/clustering checks passed.")
