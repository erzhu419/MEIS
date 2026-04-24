"""Validation for P2.4 Fisher information module.

Three checks:
  1. Normal-linear: EFI(x) ∝ x² (observed-fisher formula matches plan §Phase 2)
  2. Poisson-lograte: EFI(x) = x² · E[exp(theta*x)] — confirm vs analytic
  3. Ranking of candidate observations matches expected direction:
     - alice_charlie-like: large x (tall people) gives most info about theta
     - peregrines-like: moderate x (around peak log-rate) gives most info

Run:
    python -m phase3_embedding.tests.test_fisher_info
"""

from __future__ import annotations

import numpy as np

from phase3_embedding.fisher_info import (
    observed_fisher_normal, observed_fisher_poisson_lograte,
    expected_fisher_information, rank_observation_candidates,
    ObservationCandidate, pretty_print_ranking,
    expected_fisher_via_jax,
)


def test_normal_fisher_closed_form():
    # Plan §Phase 2: Fisher for y ~ N(theta * x, sigma^2) is x^2 / sigma^2.
    obs_sigma = 2.0
    for x in [1.0, 5.0, 10.0]:
        want = x ** 2 / obs_sigma ** 2
        got = observed_fisher_normal(theta=0.0, x=x, obs_sigma=obs_sigma)
        assert abs(got - want) < 1e-12, (x, got, want)

    # Expected Fisher over a posterior on theta: since I does not depend on
    # theta, the expectation equals the constant.
    theta_samples = np.random.default_rng(0).normal(1.414e-5, 4e-7, 500)
    efi = expected_fisher_information([1.0, 5.0, 10.0], theta_samples,
                                      likelihood="normal", obs_sigma=obs_sigma)
    assert np.allclose(efi, [0.25, 6.25, 25.0])
    print(f"[PASS] normal fisher closed-form matches I(x) = x²/σ² over all x")


def test_poisson_fisher_closed_form():
    theta = 0.5
    for x in [1.0, 2.0, 3.0]:
        want = x ** 2 * np.exp(theta * x)
        got = observed_fisher_poisson_lograte(theta=theta, x=x)
        assert abs(got - want) < 1e-10

    # Expected over posterior samples
    rng = np.random.default_rng(1)
    theta_samples = rng.normal(0.5, 0.1, 500)
    xs = [1.0, 2.0, 3.0]
    efi = expected_fisher_information(xs, theta_samples, likelihood="poisson")
    # Manual: sum over theta_samples of x^2 * exp(theta*x), then divide
    manual = np.array([np.mean(x ** 2 * np.exp(theta_samples * x)) for x in xs])
    assert np.allclose(efi, manual)
    # Monotonically increasing in x for positive theta
    assert efi[0] < efi[1] < efi[2], efi
    print(f"[PASS] poisson fisher matches analytic closed form; "
          f"EFI monotone in x (positive theta): {efi}")


def test_ranking_alice_charlie_like():
    """For Normal-linear model with observable height h, posterior on theta,
    Fisher says: the LARGER h you observe, the more information about theta
    (since I(theta, h) = h^6 / σ²; the relevant covariate is h^3)."""
    # theta ~ N(1.4e-5, 4e-7); obs_sigma = 2 (kg)
    rng = np.random.default_rng(0)
    theta_samples = rng.normal(1.414e-5, 4e-7, 500)
    candidates = [
        ObservationCandidate("h=150", 150.0 ** 3),
        ObservationCandidate("h=170", 170.0 ** 3),
        ObservationCandidate("h=190", 190.0 ** 3),
    ]
    scored = rank_observation_candidates(candidates, theta_samples,
                                          likelihood="normal", obs_sigma=2.0)
    names = [c.name for c, _ in scored]
    # Tallest should be most informative
    assert names[0] == "h=190", names
    assert names[-1] == "h=150", names
    # EFIs should differ substantially
    efi_max = scored[0][1]
    efi_min = scored[-1][1]
    assert efi_max / efi_min > 2.0
    print(f"[PASS] alice-charlie ranking: most informative = {names[0]}, "
          f"ratio max/min EFI = {efi_max/efi_min:.2f}")


def test_ranking_peregrines_like():
    """For Poisson(exp(theta * x)) with theta_samples around 0.5,
    EFI(x) = x^2 * E[exp(theta*x)] grows fast with x."""
    rng = np.random.default_rng(2)
    theta_samples = rng.normal(0.5, 0.1, 300)
    candidates = [
        ObservationCandidate("t=0",  0.0),
        ObservationCandidate("t=1",  1.0),
        ObservationCandidate("t=3",  3.0),
        ObservationCandidate("t=5",  5.0),
    ]
    scored = rank_observation_candidates(candidates, theta_samples,
                                          likelihood="poisson")
    names = [c.name for c, _ in scored]
    assert names[0] == "t=5", names
    assert names[-1] == "t=0", names
    print(f"[PASS] peregrines-like ranking: most informative = {names[0]}, "
          f"EFI values = {[round(s, 3) for _, s in scored]}")


def test_jax_fisher_agrees_with_closed_form():
    """The jax.hessian route must give the same answer as the closed-form
    branch for the Normal case."""
    import jax.numpy as jnp

    def log_lik_normal(theta, x, y, sigma=2.0):
        mu = theta * x
        return -0.5 * ((y - mu) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

    def expected_y_normal(theta, x):
        return theta * x

    rng = np.random.default_rng(3)
    theta_samples = rng.normal(1.414e-5, 4e-7, 100)
    xs = [150.0 ** 3, 170.0 ** 3, 190.0 ** 3]

    efi_closed = expected_fisher_information(xs, theta_samples,
                                              likelihood="normal", obs_sigma=2.0)
    efi_jax = expected_fisher_via_jax(xs, theta_samples,
                                       log_lik_normal, expected_y_normal)
    assert np.allclose(efi_closed, efi_jax, rtol=1e-6), (efi_closed, efi_jax)
    print(f"[PASS] jax.hessian route matches closed-form: {efi_closed}")


if __name__ == "__main__":
    print("=== P2.4 Fisher information validation ===\n")
    test_normal_fisher_closed_form()
    test_poisson_fisher_closed_form()
    test_ranking_alice_charlie_like()
    test_ranking_peregrines_like()
    test_jax_fisher_agrees_with_closed_form()
    print("\nAll P2.4 Fisher info checks passed.")
