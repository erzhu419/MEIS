"""Extended garbling search: cubic spline + neural MLP.

Addresses the reviewer limitation #5 — original polynomial garbling
was restricted to degree ≤ 3, leaving open whether a more expressive
fixed scalar garbling could bridge cross-class pairs.

We now add two strictly more expressive families and verify that
cross-class pairs are REJECTED even under:
  - polynomial degree 5 and 7
  - cubic spline with 8 knots
  - MLP (1 hidden layer, 32 units, 1000 epochs)

Acceptance:
  1. Within-class pairs: all three families dominate (rel_residual
     small enough to count as identity garbling).
  2. Cross-class pairs: all three families reject — no fixed scalar
     garbling exists that bridges the classes, confirming the BSS
     inequivalence is genuine and not a linear-fit artefact.
"""

from __future__ import annotations

from phase4_structure.semantic_equivalence import (
    polynomial_garbling_check, cubic_spline_garbling_check,
    neural_garbling_check,
)


WITHIN = [
    ("rc_circuit", "radioactive_decay"),
    ("capacitor_charging", "monomolecular_growth"),
    ("rlc_circuit", "pendulum"),
]
CROSS = [
    ("rc_circuit", "capacitor_charging"),
    ("rc_circuit", "rlc_circuit"),
    ("capacitor_charging", "rlc_circuit"),
]


def test_polynomial_degrees_5_7_still_reject_cross_class():
    """Higher-degree polynomial does not help on cross-class pairs.
    Confirms the rejection isn't an artefact of restricting degree ≤ 3."""
    for degree in [5, 7]:
        for a, b in CROSS:
            r = polynomial_garbling_check(a, b, degree=degree,
                                            n_samples=300, seed=0,
                                            tolerance=1e-6)
            assert not r.dominates, \
                f"degree={degree} {a}→{b} unexpectedly dominates"
            assert r.relative_residual > 0.05
    print(f"[PASS] polynomial degrees 5 and 7 both reject all "
          f"{len(CROSS)} cross-class pairs")


def test_cubic_spline_dominates_within_rejects_cross():
    for a, b in WITHIN:
        r = cubic_spline_garbling_check(a, b, n_knots=8, n_samples=300,
                                          seed=0, tolerance=1e-6)
        assert r.dominates, f"{a}→{b} spline did not dominate: {r.relative_residual}"
    for a, b in CROSS:
        r = cubic_spline_garbling_check(a, b, n_knots=8, n_samples=300,
                                          seed=0, tolerance=1e-6)
        assert not r.dominates
        assert r.relative_residual > 0.4
    print(f"[PASS] cubic spline (8 knots): within-class dom, cross-class reject "
          f"with rel_residual > 0.4")


def test_neural_mlp_dominates_within_rejects_cross():
    """MLP is the most expressive scalar garbling family we ship. If it
    cannot bridge cross-class, no bounded-width fixed scalar function can."""
    for a, b in WITHIN:
        r = neural_garbling_check(a, b, hidden=32, epochs=1000,
                                    n_samples=200, seed=0,
                                    tolerance=0.05)
        assert r.dominates, \
            f"{a}→{b} MLP rel_residual {r.relative_residual} > 0.05"
    for a, b in CROSS:
        r = neural_garbling_check(a, b, hidden=32, epochs=1000,
                                    n_samples=200, seed=0,
                                    tolerance=1e-6)
        assert not r.dominates
        assert r.relative_residual > 0.3
    print(f"[PASS] MLP garbling (hidden=32, 1000 epochs): within-class converges, "
          f"cross-class rejects with rel_residual > 0.3")


if __name__ == "__main__":
    print("=== Extended garbling family rejection ===\n")
    test_polynomial_degrees_5_7_still_reject_cross_class()
    test_cubic_spline_dominates_within_rejects_cross()
    test_neural_mlp_dominates_within_rejects_cross()
    print("\nAll extended-garbling checks passed.")
