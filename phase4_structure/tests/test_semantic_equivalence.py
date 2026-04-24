"""Validation for P4.8 — BSS operational equivalence + Perrone kernel KL.

Acceptance:

  === BSS ===
  1. For every within-class pair, max|Δlog p| == 0 exactly over 100
     random (θ, t, y, σ) samples — same class_mu function ⇒ identical
     likelihood.
  2. For every cross-class pair, max|Δlog p| > 1.0 — disjoint class_mu
     functions disagree substantially on generic inputs.
  3. The `bss_equivalent` flag is True for all within-class and False
     for all cross-class pairs (tolerance 1e-10).

  === Perrone KL ===
  4. Within-class D = 0 exactly (mu functions identical).
  5. Cross-class D > 0 with Monte Carlo standard error << estimate.
  6. KL satisfies D(a,b) = D(b,a) by construction (symmetric mean
     difference) — act as a pseudometric sanity check.

  === Gaussian log-likelihood correctness ===
  7. Hand-computed value at a known point matches the helper.

Run:
    python -m phase4_structure.tests.test_semantic_equivalence
"""

from __future__ import annotations

import itertools
import numpy as np

from phase4_structure.law_zoo import CLASS_OF
from phase4_structure.semantic_equivalence import (
    bss_likelihood_equivalent, perrone_kernel_kl,
    gaussian_log_likelihood,
)


def _within_class_pairs():
    pairs = []
    for cls in set(CLASS_OF.values()):
        members = [n for n, c in CLASS_OF.items() if c == cls]
        for a, b in itertools.combinations(members, 2):
            pairs.append((a, b))
    return pairs


def _cross_class_pairs():
    pairs = []
    classes = sorted(set(CLASS_OF.values()))
    for cls_a, cls_b in itertools.combinations(classes, 2):
        a = [n for n, c in CLASS_OF.items() if c == cls_a][0]
        b = [n for n, c in CLASS_OF.items() if c == cls_b][0]
        pairs.append((a, b))
    return pairs


def test_bss_within_class_exact_equivalence():
    pairs = _within_class_pairs()
    for a, b in pairs:
        r = bss_likelihood_equivalent(a, b, n_samples=100, seed=0)
        assert r.bss_equivalent, f"{a} vs {b}: max|Δlog p| = {r.max_abs_log_lik_diff}"
        assert r.max_abs_log_lik_diff == 0.0, \
            f"{a} vs {b}: expected exact 0, got {r.max_abs_log_lik_diff}"
    print(f"[PASS] BSS within-class: {len(pairs)} pairs, max|Δlog p| = 0 on all")


def test_bss_cross_class_is_detectably_inequivalent():
    pairs = _cross_class_pairs()
    for a, b in pairs:
        r = bss_likelihood_equivalent(a, b, n_samples=100, seed=0)
        assert not r.bss_equivalent, \
            f"{a} vs {b} should NOT be BSS-equivalent: {r.max_abs_log_lik_diff}"
        assert r.max_abs_log_lik_diff > 1.0, \
            f"{a} vs {b}: log-lik diff only {r.max_abs_log_lik_diff}"
    print(f"[PASS] BSS cross-class: {len(pairs)} pairs, all "
          f"max|Δlog p| > 1.0 (disagreement readily detectable)")


def test_perrone_kl_within_class_is_zero():
    pairs = _within_class_pairs()
    for a, b in pairs:
        r = perrone_kernel_kl(a, b, n_samples=300, seed=0)
        assert r.kl_estimate == 0.0, \
            f"{a} vs {b}: within-class KL should be 0, got {r.kl_estimate}"
    print(f"[PASS] Perrone KL within-class: {len(pairs)} pairs, D = 0 exactly")


def test_perrone_kl_cross_class_is_positive_with_small_se():
    pairs = _cross_class_pairs()
    for a, b in pairs:
        r = perrone_kernel_kl(a, b, n_samples=500, seed=0)
        assert r.kl_estimate > 0.0, f"{a} vs {b}: KL not positive: {r.kl_estimate}"
        # SE should be << estimate for meaningful signal
        snr = r.kl_estimate / max(r.kl_stderr, 1e-12)
        assert snr > 3.0, \
            f"{a} vs {b}: KL SNR too low: estimate {r.kl_estimate} ± {r.kl_stderr}"
    print(f"[PASS] Perrone KL cross-class: {len(pairs)} pairs, "
          f"D > 0 with SNR > 3 in every case")


def test_perrone_kl_symmetry_in_squared_mean_formulation():
    """The squared-mean form KL(a‖b) = E[(μ_a - μ_b)²/(2σ²)] is symmetric
    under a↔b. (The general kernel KL is not symmetric, but when both
    kernels share σ and we take the squared-mean surrogate, it is.)
    Verify on one cross-class pair."""
    r_ab = perrone_kernel_kl("rc_circuit", "capacitor_charging",
                              n_samples=500, seed=42)
    r_ba = perrone_kernel_kl("capacitor_charging", "rc_circuit",
                              n_samples=500, seed=42)
    assert np.isclose(r_ab.kl_estimate, r_ba.kl_estimate, rtol=1e-10), \
        f"symmetry broken: {r_ab.kl_estimate} vs {r_ba.kl_estimate}"
    print(f"[PASS] Perrone KL symmetric under squared-mean formulation: "
          f"D(rc_circuit ‖ cap) = D(cap ‖ rc_circuit) = {r_ab.kl_estimate:.4f}")


def test_gaussian_log_likelihood_closed_form():
    # At y = mu, sigma = 1: log N(0|0, 1) = -0.5*log(2π) ≈ -0.9189385
    ll = float(gaussian_log_likelihood(np.array([0.0]),
                                       np.array([0.0]),
                                       sigma=1.0)[0])
    expected = -0.5 * np.log(2 * np.pi)
    assert np.isclose(ll, expected, atol=1e-12), ll
    # At y - mu = 2 sigma: should be expected - 2
    ll2 = float(gaussian_log_likelihood(np.array([2.0]),
                                         np.array([0.0]),
                                         sigma=1.0)[0])
    assert np.isclose(ll2, expected - 2.0, atol=1e-12), ll2
    print(f"[PASS] gaussian_log_likelihood matches closed form at 2 points")


if __name__ == "__main__":
    print("=== P4.8 semantic equivalence validation ===\n")
    test_gaussian_log_likelihood_closed_form()
    test_bss_within_class_exact_equivalence()
    test_bss_cross_class_is_detectably_inequivalent()
    test_perrone_kl_within_class_is_zero()
    test_perrone_kl_cross_class_is_positive_with_small_se()
    test_perrone_kl_symmetry_in_squared_mean_formulation()
    print("\nAll P4.8 semantic-equivalence checks passed.")
