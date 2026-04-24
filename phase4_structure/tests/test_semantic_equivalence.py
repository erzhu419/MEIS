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
    linear_gaussian_bss_check, mc_kernel_kl_gaussian,
    polynomial_garbling_fit, polynomial_garbling_check,
    mc_kernel_kl_general,
    poisson_sample_y, poisson_log_pdf, poisson_log_pdf_scaled,
    closed_form_poisson_kl,
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


def test_fritz_garbling_dominates_within_class():
    """P4.9a: within-class, the best-fit linear-Gaussian garbling is
    the identity map (A = 1, b = 0) with residual at machine epsilon.
    This is the genuine 'there exists a post-processor' statement in
    BSS-dominance terms (not just likelihood equality)."""
    pairs = _within_class_pairs()
    for a, b in pairs:
        r = linear_gaussian_bss_check(a, b, n_samples=300, seed=0, tolerance=1e-10)
        assert r.dominates, f"{a} → {b}: rel_residual {r.relative_residual}"
        assert abs(r.A - 1.0) < 1e-10, f"{a} → {b}: A = {r.A} not 1"
        assert abs(r.b) < 1e-10, f"{a} → {b}: b = {r.b} not 0"
        assert r.relative_residual < 1e-10, \
            f"{a} → {b}: rel_residual {r.relative_residual}"
    print(f"[PASS] Fritz linear-Gaussian garbling within-class: {len(pairs)} pairs, "
          f"identity garbling (A≈1, b≈0, residual ~1e-16)")


def test_fritz_garbling_rejects_cross_class():
    """P4.9a: cross-class, no linear-Gaussian garbling works. The
    best-fit (A, b) still has a significant residual (>5% of μ_b
    scale)."""
    pairs = _cross_class_pairs()
    for a, b in pairs:
        r = linear_gaussian_bss_check(a, b, n_samples=300, seed=0, tolerance=1e-6)
        assert not r.dominates, \
            f"{a} → {b} unexpectedly dominates with residual {r.relative_residual}"
        assert r.relative_residual > 0.05, \
            f"{a} → {b}: residual too small to reject: {r.relative_residual}"
    print(f"[PASS] Fritz linear-Gaussian garbling rejects cross-class: "
          f"{len(pairs)} pairs, all relative residual > 0.05")


def test_mc_kernel_kl_within_class_is_zero():
    """P4.9b: general MC KL = 0 within-class (identical μ functions)."""
    pairs = _within_class_pairs()
    for a, b in pairs:
        r = mc_kernel_kl_gaussian(a, b, n_theta_samples=200,
                                    n_y_per_theta=30, seed=0)
        assert r.kl_estimate == 0.0, \
            f"{a} vs {b}: within-class MC KL should be 0, got {r.kl_estimate}"
    print(f"[PASS] general MC kernel KL within-class: {len(pairs)} pairs, "
          f"D_MC = 0 exactly")


def test_mc_kernel_kl_matches_closed_form_on_gaussian():
    """P4.9b validation: the general MC estimator must agree with
    Perrone's closed-form squared-mean formula on every cross-class
    Gaussian pair, within 3σ MC error. This certifies the general
    machinery is correct before we apply it to future non-Gaussian
    fixtures."""
    pairs = _cross_class_pairs()
    for a, b in pairs:
        r_mc = mc_kernel_kl_gaussian(a, b, n_theta_samples=400,
                                       n_y_per_theta=60, seed=0)
        r_closed = perrone_kernel_kl(a, b, n_samples=400, seed=0)
        # The two estimators use the SAME θ distribution and SAME σ,
        # so they target the same quantity. MC should be within 3σ
        # of closed-form (which is itself a closed-form Monte Carlo).
        diff = abs(r_mc.kl_estimate - r_closed.kl_estimate)
        combined_se = (r_mc.kl_stderr ** 2 + r_closed.kl_stderr ** 2) ** 0.5
        assert diff < 3.0 * max(combined_se, 1e-9), \
            (f"{a} vs {b}: MC {r_mc.kl_estimate:.4f}±{r_mc.kl_stderr:.4f} "
             f"vs closed {r_closed.kl_estimate:.4f}±{r_closed.kl_stderr:.4f}, "
             f"diff {diff:.4f} > 3σ {3.0 * combined_se:.4f}")
    print(f"[PASS] general MC kernel KL agrees with Perrone closed form on "
          f"{len(pairs)} cross-class pairs (within 3σ — validates non-"
          f"Gaussian-ready MC machinery)")


def test_polynomial_garbling_dominates_within_class():
    """Within-class: polynomial of ANY degree ≥ 1 gives identity-like fit
    (residual at machine epsilon). Check degrees 1, 2, 3."""
    pairs = _within_class_pairs()
    for a, b in pairs:
        for degree in [1, 2, 3]:
            r = polynomial_garbling_check(a, b, degree=degree,
                                            n_samples=200, seed=0,
                                            tolerance=1e-10)
            assert r.dominates, \
                f"{a} → {b} degree {degree}: rel_residual {r.relative_residual}"
    print(f"[PASS] polynomial garbling degrees 1/2/3 all dominate within-class "
          f"on {len(pairs)} pairs (residual ~1e-15)")


def test_polynomial_garbling_rejects_cross_class_even_at_degree_3():
    """Crucial: even cubic polynomial cannot find a fixed garbling
    cross-class. BSS-rejection is not an artefact of linear restriction."""
    pairs = _cross_class_pairs()
    for a, b in pairs:
        r = polynomial_garbling_check(a, b, degree=3,
                                        n_samples=300, seed=0,
                                        tolerance=1e-6)
        assert not r.dominates, \
            f"{a} → {b} unexpectedly dominates at degree 3: {r.relative_residual}"
        assert r.relative_residual > 0.05, \
            f"{a} → {b}: residual too small to reject: {r.relative_residual}"
    print(f"[PASS] polynomial degree 3 REJECTS cross-class on {len(pairs)} pairs "
          f"(rel_residual > 0.05) — confirms structural inequivalence, not "
          f"linear-fit artefact")


def test_polynomial_recovers_synthetic_quadratic_garbling():
    """Synthetic demo: μ_b = μ_a² requires polynomial degree ≥ 2.
    Linear fit must fail; degree-2 must recover with coef ≈ [0, 0, 1]."""
    rng = np.random.default_rng(0)
    theta = rng.lognormal(mean=0.0, sigma=1.0, size=500)
    mu_a = theta.copy()
    mu_b = theta ** 2

    # Linear should fail
    _, _, _, rel_lin = polynomial_garbling_fit(mu_a, mu_b, degree=1)
    assert rel_lin > 0.1, f"linear unexpectedly succeeded: rel={rel_lin}"

    # Degree-2 should recover exactly
    coef2, _, _, rel_quad = polynomial_garbling_fit(mu_a, mu_b, degree=2)
    assert rel_quad < 1e-10, f"quadratic residual {rel_quad} not ~0"
    # coefficients should be [0, 0, 1] up to numerical precision
    assert abs(coef2[0]) < 1e-10 and abs(coef2[1]) < 1e-10 and abs(coef2[2] - 1.0) < 1e-10, \
        f"coef {coef2}"
    print(f"[PASS] synthetic pair (μ_a=θ, μ_b=θ²): linear fails (rel={rel_lin:.3f}), "
          f"degree-2 recovers perfectly (coef ≈ [0, 0, 1], rel={rel_quad:.1e})")


def test_general_mc_kl_on_poisson_matches_closed_form():
    """P4.12: non-Gaussian demo. K_a = Poisson(θ), K_b = Poisson(2θ),
    θ ~ Uniform(1, 5). Closed-form E_θ[KL] = E[θ · (1 - log 2)]
    = 3 · (1 - log 2) ≈ 0.9207. MC should match within 3σ."""
    def _sample_theta(rng):
        return float(rng.uniform(1.0, 5.0))

    r = mc_kernel_kl_general(
        sample_y_given_theta_a=poisson_sample_y,
        log_pdf_a=poisson_log_pdf,
        log_pdf_b=poisson_log_pdf_scaled(alpha=2.0),
        sample_theta=_sample_theta,
        n_theta_samples=800, n_y_per_theta=200, seed=0,
        label_a="Poisson(θ)", label_b="Poisson(2θ)",
    )
    expected = 3.0 * (1.0 - np.log(2.0))
    diff = abs(r.kl_estimate - expected)
    assert diff < 3.0 * r.kl_stderr, \
        (f"Poisson MC KL {r.kl_estimate:.4f} ± {r.kl_stderr:.4f} "
         f"vs expected {expected:.4f}, diff {diff:.4f} > 3σ "
         f"{3.0 * r.kl_stderr:.4f}")
    print(f"[PASS] Poisson(θ)‖Poisson(2θ): MC D = {r.kl_estimate:.4f} "
          f"± {r.kl_stderr:.4f}, analytical = {expected:.4f} "
          f"(diff {diff:.4f}, within {diff/r.kl_stderr:.2f}σ) — "
          f"non-Gaussian kernel KL works")


def test_general_mc_kl_equals_zero_for_identical_kernels():
    """Sanity: if K_a == K_b, MC estimator should return ~0 (exactly 0
    up to finite-sample noise)."""
    def _sample_theta(rng):
        return float(rng.uniform(1.0, 5.0))
    r = mc_kernel_kl_general(
        sample_y_given_theta_a=poisson_sample_y,
        log_pdf_a=poisson_log_pdf,
        log_pdf_b=poisson_log_pdf,        # same!
        sample_theta=_sample_theta,
        n_theta_samples=400, n_y_per_theta=100, seed=0,
        label_a="Poisson(θ)", label_b="Poisson(θ)",
    )
    # Exactly identical log-pdfs on same y → KL is exactly 0
    assert r.kl_estimate == 0.0, \
        f"identical-kernel MC KL should be 0, got {r.kl_estimate}"
    print(f"[PASS] identical-kernel MC KL = 0 exactly "
          f"(K_a = K_b = Poisson(θ))")


if __name__ == "__main__":
    print("=== P4.8 + P4.9 + P4.11 + P4.12 semantic equivalence validation ===\n")
    test_gaussian_log_likelihood_closed_form()
    test_bss_within_class_exact_equivalence()
    test_bss_cross_class_is_detectably_inequivalent()
    test_perrone_kl_within_class_is_zero()
    test_perrone_kl_cross_class_is_positive_with_small_se()
    test_perrone_kl_symmetry_in_squared_mean_formulation()
    print()
    test_fritz_garbling_dominates_within_class()
    test_fritz_garbling_rejects_cross_class()
    test_mc_kernel_kl_within_class_is_zero()
    test_mc_kernel_kl_matches_closed_form_on_gaussian()
    print()
    test_polynomial_garbling_dominates_within_class()
    test_polynomial_garbling_rejects_cross_class_even_at_degree_3()
    test_polynomial_recovers_synthetic_quadratic_garbling()
    test_general_mc_kl_on_poisson_matches_closed_form()
    test_general_mc_kl_equals_zero_for_identical_kernels()
    print("\nAll semantic-equivalence checks passed.")
