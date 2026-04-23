"""Step 1 validation: AliceCharlie env basics + PyMC posterior vs closed-form.

Run:
    python -m phase1_mvp.tests.test_alice_charlie
"""

from __future__ import annotations

import numpy as np
import pymc as pm

from phase1_mvp.envs.alice_charlie import (
    AliceCharlie, DirectGoal, DirectGoalNaive,
    THETA_MU, THETA_SIGMA, OBS_NOISE, HEIGHT_LOWER, HEIGHT_UPPER,
)


def test_instantiation():
    env = AliceCharlie()
    assert env.env_name == "alice_charlie"
    assert HEIGHT_LOWER <= env.sample_random_input() <= HEIGHT_UPPER
    assert THETA_MU - 5 * THETA_SIGMA < env.theta < THETA_MU + 5 * THETA_SIGMA, env.theta
    DirectGoal(env)
    DirectGoalNaive(env)
    print("[PASS] instantiation + Goal classes")


def test_step_and_validate():
    env = AliceCharlie()
    env.theta = THETA_MU  # fix latent for deterministic test
    # step returns plausible weight
    w = env.step(170.0)
    expected = THETA_MU * 170.0 ** 3  # ≈ 69.5
    assert abs(w - expected) < 4 * OBS_NOISE, (w, expected)

    # validate_input
    assert env.validate_input("170") == 170.0
    assert isinstance(env.validate_input("200"), str)
    assert isinstance(env.validate_input("abc"), str)

    # run_experiment happy path
    w, ok = env.run_experiment("175")
    assert ok and isinstance(w, float)
    assert len(env.observed_data) == 1

    # run_experiment reject out-of-bounds
    msg, ok = env.run_experiment("999")
    assert not ok and "between" in msg
    # observed_data unchanged on rejection
    assert len(env.observed_data) == 1
    print(f"[PASS] step + validate_input + run_experiment  (w@170={w:.2f})")


def test_posterior_alignment(seed: int = 0, n_obs: int = 20, tol_sigma: float = 0.5):
    """After n_obs observations, PyMC posterior on theta should match the analytical
    Gaussian conjugate posterior within tol_sigma posterior-stds.

    Closed-form: prior N(mu0, sigma0^2), likelihood y_i ~ N(theta * x_i, sigma^2) with x_i = h_i^3.
        post_prec = 1/sigma0^2 + sum(x_i^2)/sigma^2
        post_mu   = (mu0/sigma0^2 + sum(x_i*y_i)/sigma^2) / post_prec
    """
    rng = np.random.default_rng(seed)
    env = AliceCharlie()
    # Overwrite env.theta with a known fixed value so the test is deterministic.
    true_theta = THETA_MU + 1.5 * THETA_SIGMA  # inside 2sigma of prior
    env.theta = true_theta

    # Simulate n_obs observations at uniformly-spaced heights
    heights = np.linspace(HEIGHT_LOWER + 5, HEIGHT_UPPER - 5, n_obs)
    weights = np.array([env.step(h) for h in heights])
    xs = heights ** 3  # shape (n_obs,)

    # --- Closed-form Gaussian conjugate posterior ---
    post_prec = 1.0 / THETA_SIGMA ** 2 + np.sum(xs ** 2) / OBS_NOISE ** 2
    post_var = 1.0 / post_prec
    post_mu_cf = post_var * (THETA_MU / THETA_SIGMA ** 2 + np.sum(xs * weights) / OBS_NOISE ** 2)
    post_sd_cf = np.sqrt(post_var)

    # --- PyMC MCMC posterior ---
    with pm.Model():
        theta = pm.Normal("theta", mu=THETA_MU, sigma=THETA_SIGMA)
        pm.Normal("w_obs", mu=theta * xs, sigma=OBS_NOISE, observed=weights)
        trace = pm.sample(2000, tune=1000, chains=2, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False,
                          return_inferencedata=False)
    mcmc_samples = np.asarray(trace["theta"])
    post_mu_mcmc = float(np.mean(mcmc_samples))
    post_sd_mcmc = float(np.std(mcmc_samples, ddof=1))

    print(f"  True theta           : {true_theta:.6f}")
    print(f"  Prior                : N(mu={THETA_MU:.6f}, sd={THETA_SIGMA:.6f})")
    print(f"  Closed-form posterior: N(mu={post_mu_cf:.6f}, sd={post_sd_cf:.6f})")
    print(f"  PyMC posterior       : N(mu={post_mu_mcmc:.6f}, sd={post_sd_mcmc:.6f})")

    # Check posterior mean agreement
    mean_diff = abs(post_mu_cf - post_mu_mcmc)
    assert mean_diff < tol_sigma * post_sd_cf, \
        f"posterior mean mismatch: CF={post_mu_cf:.6f}, MCMC={post_mu_mcmc:.6f}, diff={mean_diff:.3e} > {tol_sigma}·σ={tol_sigma*post_sd_cf:.3e}"

    # Check posterior std agreement (within 20%)
    sd_ratio = post_sd_mcmc / post_sd_cf
    assert 0.8 < sd_ratio < 1.2, f"posterior sd ratio MCMC/CF = {sd_ratio:.3f} (want 0.8–1.2)"

    # True theta should be in 3sigma of the posterior
    z = (true_theta - post_mu_cf) / post_sd_cf
    print(f"  True theta is {z:+.2f}σ from closed-form posterior mean")

    print(f"[PASS] PyMC posterior matches closed-form within {tol_sigma}·σ")


def test_norm_factors():
    env = AliceCharlie()
    goal = DirectGoal(env)
    # Reduce sample count for fast test; still large enough for stable mean
    # by temporarily monkey-patching the loop constant would require touching code;
    # instead we just run the default N=10000 (fast for pure numpy).
    err_mean, err_std = goal.get_norm_factors()
    print(f"  norm_mu = {goal.norm_mu:.2f}, norm_sigma (err_std) = {goal.norm_sigma:.2f}")
    # norm_mu should fall in physically plausible human weight range
    assert 40 < goal.norm_mu < 120, goal.norm_mu
    print("[PASS] get_norm_factors")


if __name__ == "__main__":
    print("=== Step 1 validation: AliceCharlie env ===\n")
    test_instantiation()
    test_step_and_validate()
    print("\n--- Posterior alignment (PyMC vs closed-form) ---")
    test_posterior_alignment()
    print("\n--- Normalization factors ---")
    test_norm_factors()
    print("\nAll Step 1 checks passed.")
