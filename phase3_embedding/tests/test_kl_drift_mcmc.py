"""Validation for P2.2 MCMC-based KL drift ranker.

Three checks:
  1. Gaussian-moment KL estimator matches closed-form on known Gaussians.
  2. MCMC ranker on alice_charlie (conjugate Normal-Normal) reproduces
     the ordering that the closed-form ranker produced in Step 6.
  3. MCMC ranker on a NON-conjugate case (Poisson count model mimicking
     peregrines) produces a sensible KL ordering: harder-to-fit
     hypotheses get bigger KLs.

Run:
    python -m phase3_embedding.tests.test_kl_drift_mcmc
"""

from __future__ import annotations

import math

import numpy as np

from phase3_embedding.kl_drift_mcmc import (
    kl_gaussian_moment, kl_gaussian_fullcov, kl_kde,
    MCMCHypothesis, rank_hypotheses_mcmc, pretty_print_mcmc,
)


def test_gaussian_moment_estimator_accuracy():
    """Feed samples from two known Gaussians; moment KL should match analytic."""
    rng = np.random.default_rng(0)
    n = 5000
    mu_p, sigma_p = 0.0, 1.0
    mu_q, sigma_q = 2.0, 1.0
    sp = rng.normal(mu_p, sigma_p, n)
    sq = rng.normal(mu_q, sigma_q, n)
    kl = kl_gaussian_moment(sp, sq)
    # Analytic KL N(0,1)||N(2,1) = 0.5 * (0 + (0-2)^2 - 0) = 2.0
    analytic = 2.0
    assert abs(kl - analytic) < 0.1, f"got {kl}, want ~{analytic}"
    print(f"[PASS] Gaussian-moment KL on n=5000 samples: {kl:.4f} ≈ {analytic:.4f}")


def test_gaussian_fullcov_matches_moment_in_1d():
    """Full-cov should reduce to the univariate case for 1-D inputs."""
    rng = np.random.default_rng(1)
    n = 3000
    sp = rng.normal(0.0, 1.5, n).reshape(-1, 1)
    sq = rng.normal(0.5, 1.0, n).reshape(-1, 1)
    a = kl_gaussian_fullcov(sp, sq)
    b = kl_gaussian_moment(sp.squeeze(), sq.squeeze())
    assert abs(a - b) < 1e-6, f"{a} vs {b}"
    print(f"[PASS] full-cov reduces to moment in 1-D: KL={a:.4f}")


def test_mcmc_ranking_alice_charlie_conjugate():
    """On alice_charlie (Normal-Normal conjugate), MCMC ranker should
    reproduce the closed-form ordering from Step 6."""
    import pymc as pm
    from phase1_mvp.envs.alice_charlie import THETA_MU, THETA_SIGMA, OBS_NOISE

    rng = np.random.default_rng(42)
    heights = rng.uniform(150, 190, size=10)
    theta_true = THETA_MU
    weights = theta_true * heights ** 3 + rng.normal(0.0, OBS_NOISE, size=10)

    def build_base():
        theta = pm.Normal("theta", mu=THETA_MU, sigma=THETA_SIGMA)
        pm.Normal("w_obs", mu=theta * heights ** 3, sigma=OBS_NOISE, observed=weights)

    def _make_hyp(name, summary, h_cm, w_kg):
        def apply():
            theta = pm.Model.get_context()["theta"]
            pm.Normal(f"h_{name}",
                      mu=theta * h_cm ** 3, sigma=OBS_NOISE,
                      observed=np.array([w_kg]))
        return MCMCHypothesis(name=name, summary=summary,
                              apply_fn=apply, latent_var="theta")

    hypotheses = [
        _make_hyp("H1", "170 cm, 70 kg (central)", 170.0, 70.0),
        _make_hyp("H2", "170 cm, 80 kg (upper)",   170.0, 80.0),
        _make_hyp("H3", "170 cm, 55 kg (lower)",   170.0, 55.0),
        _make_hyp("H4", "170 cm, 120 kg (obese)",  170.0, 120.0),
        _make_hyp("H5", "170 cm, 200 kg (absurd)", 170.0, 200.0),
    ]

    scores = rank_hypotheses_mcmc(
        build_base_model=build_base,
        hypotheses=hypotheses,
        latent_var="theta",
        draws=1000, tune=500, chains=2, random_seed=0,
    )
    pretty_print_mcmc(scores)

    names = [s.hypothesis.name for s in scores]
    # H5 must be last; H1 must be first
    assert names[-1] == "H5", f"H5 not last: {names}"
    assert names[0] == "H1", f"H1 not first: {names}"
    # KLs must be monotone non-decreasing
    kls = [s.kl_from_base for s in scores]
    for i in range(len(kls) - 1):
        assert kls[i+1] >= kls[i] - 1e-6
    print(f"[PASS] MCMC ranker reproduces closed-form ordering: {names}")


def test_mcmc_ranking_poisson_log_cubic():
    """Non-conjugate case (peregrines-like Poisson with log-cubic mean).
    Compare three hypotheses about a future count at t=1.0 — one plausible,
    two absurd. Expect KL: plausible < far < absurd."""
    import pymc as pm

    rng = np.random.default_rng(7)
    # Simulate ground-truth trajectory
    alpha_true, b1, b2, b3 = 4.5, 1.2, 0.07, -0.24
    ts = np.linspace(0, 3, 10)
    log_lambdas = alpha_true + b1 * ts + b2 * ts ** 2 + b3 * ts ** 3
    counts = rng.poisson(np.exp(log_lambdas))

    def build_base():
        alpha = pm.Normal("alpha", mu=4.5, sigma=0.5)
        beta1 = pm.Normal("beta1", mu=1.2, sigma=0.3)
        beta2 = pm.Normal("beta2", mu=0.07, sigma=0.1)
        beta3 = pm.Normal("beta3", mu=-0.24, sigma=0.1)
        log_mu = (alpha + beta1 * ts + beta2 * ts ** 2 + beta3 * ts ** 3)
        pm.Poisson("y_obs", mu=pm.math.exp(log_mu), observed=counts)

    def _hyp(name, summary, t_query, count_claim):
        def apply():
            alpha = pm.Model.get_context()["alpha"]
            beta1 = pm.Model.get_context()["beta1"]
            beta2 = pm.Model.get_context()["beta2"]
            beta3 = pm.Model.get_context()["beta3"]
            log_mu_q = alpha + beta1 * t_query + beta2 * t_query ** 2 + beta3 * t_query ** 3
            pm.Poisson(f"y_{name}", mu=pm.math.exp(log_mu_q),
                       observed=np.array([count_claim]))
        return MCMCHypothesis(name=name, summary=summary,
                              apply_fn=apply, latent_var="alpha")

    # Expected count at t=1: exp(4.5 + 1.2 + 0.07 - 0.24) = exp(5.53) ≈ 251
    hypotheses = [
        _hyp("H_plausible", "count 250 at t=1 (central)", 1.0, 250),
        _hyp("H_far",       "count 1000 at t=1 (4x high)", 1.0, 1000),
        _hyp("H_absurd",    "count 10000 at t=1 (absurd)", 1.0, 10000),
    ]

    scores = rank_hypotheses_mcmc(
        build_base_model=build_base,
        hypotheses=hypotheses,
        latent_var="alpha",
        draws=800, tune=500, chains=2, random_seed=0,
    )
    pretty_print_mcmc(scores)
    names = [s.hypothesis.name for s in scores]
    kls = [s.kl_from_base for s in scores]
    # Plausible should be least disturbing; absurd most
    assert names[0] == "H_plausible", f"got {names}"
    assert names[-1] == "H_absurd", f"got {names}"
    # Monotone
    for i in range(len(kls) - 1):
        assert kls[i+1] >= kls[i] - 1e-4, f"non-monotone: {kls}"
    print(f"[PASS] Non-conjugate Poisson case orders correctly: {names}")


if __name__ == "__main__":
    print("=== P2.2 validation: MCMC KL drift ===\n")
    test_gaussian_moment_estimator_accuracy()
    test_gaussian_fullcov_matches_moment_in_1d()
    print()
    print("--- MCMC on conjugate alice_charlie (should match Step 6 ordering) ---")
    test_mcmc_ranking_alice_charlie_conjugate()
    print()
    print("--- MCMC on non-conjugate Poisson (peregrines-like) ---")
    test_mcmc_ranking_poisson_log_cubic()
    print("\nAll P2.2 checks passed.")
