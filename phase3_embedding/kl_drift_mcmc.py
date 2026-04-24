"""MEIS Phase 2 P2.2 — MCMC-based KL drift for non-Gaussian posteriors.

`kl_drift.py` (Phase 1 Step 6) handles the Gaussian conjugate case
exactly via closed-form posterior updates. This module extends the
minimum-embedding-distance ranker to arbitrary PyMC generative models
by estimating KL via Monte Carlo on samples from the two posteriors.

Two KL estimators supported:
  - **Gaussian-moment estimator** (default): fit a Normal to each
    posterior's samples, use closed-form N-N KL. Fast, adequate when
    the posteriors are unimodal and roughly symmetric.
  - **KDE estimator** (optional): fit a Gaussian KDE to both posteriors,
    estimate KL via KL(P||Q) = E_P[log p(x) - log q(x)] with expectation
    over P's samples. Handles heavy tails / skewness better but slower.

Phase 1 `kl_drift.py` still handles the pure Gaussian-conjugate case
when available. This module is invoked when the hypothesis cannot be
represented as a clean (x_i, y_i) observation sequence or when the
generative model has non-conjugate structure (e.g., Poisson count
likelihoods on peregrines, Lotka-Volterra coupled ODEs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# PyMC is imported lazily in functions to avoid top-level import cost.


# =============================================================================
# Non-Gaussian KL estimators (over posterior samples)
# =============================================================================
def kl_gaussian_moment(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """Fit a Normal to each sample set, return closed-form KL(N_p || N_q).

    Accepts 1-D arrays (scalar latent) or 2-D arrays (vector latent of dim d).
    For vector latent: treats components as independent (product of univariate
    KLs). This is a reasonable first approximation when components are
    weakly correlated in the posterior; use `kl_gaussian_fullcov` for
    correlated latents.
    """
    p = np.asarray(samples_p, dtype=float)
    q = np.asarray(samples_q, dtype=float)
    if p.ndim == 1:
        mu_p, mu_q = float(p.mean()), float(q.mean())
        var_p = float(p.var(ddof=1))
        var_q = float(q.var(ddof=1))
        return (
            math.log(math.sqrt(var_q / var_p))
            + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
            - 0.5
        )
    assert p.ndim == 2 and q.ndim == 2 and p.shape[1] == q.shape[1]
    # Sum of per-dim univariate KLs (diag-Gaussian approximation)
    total = 0.0
    for d in range(p.shape[1]):
        total += kl_gaussian_moment(p[:, d], q[:, d])
    return total


def kl_gaussian_fullcov(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """KL between two multivariate Gaussians fit to the samples with full
    covariance. Handles correlated latents correctly."""
    p = np.atleast_2d(samples_p).astype(float)
    q = np.atleast_2d(samples_q).astype(float)
    if p.shape[0] < p.shape[1]:
        p = p.T
    if q.shape[0] < q.shape[1]:
        q = q.T
    mu_p = p.mean(axis=0)
    mu_q = q.mean(axis=0)
    cov_p = np.cov(p, rowvar=False)
    cov_q = np.cov(q, rowvar=False)
    if np.ndim(cov_p) == 0:
        cov_p = np.array([[float(cov_p)]])
        cov_q = np.array([[float(cov_q)]])
    d = mu_p.shape[0]
    inv_q = np.linalg.inv(cov_q)
    diff = mu_p - mu_q
    trace_term = np.trace(inv_q @ cov_p)
    quad_term = float(diff.T @ inv_q @ diff)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    return 0.5 * (trace_term + quad_term - d + (logdet_q - logdet_p))


def kl_kde(samples_p: np.ndarray, samples_q: np.ndarray,
           bandwidth: str = "scott") -> float:
    """KL(P||Q) via kernel-density fits. More robust to non-Gaussianity
    but higher variance. Uses E_P[log p(x) - log q(x)] estimator."""
    from scipy.stats import gaussian_kde
    p = np.asarray(samples_p, dtype=float)
    q = np.asarray(samples_q, dtype=float)
    if p.ndim == 1:
        kde_p = gaussian_kde(p, bw_method=bandwidth)
        kde_q = gaussian_kde(q, bw_method=bandwidth)
    else:
        kde_p = gaussian_kde(p.T, bw_method=bandwidth)
        kde_q = gaussian_kde(q.T, bw_method=bandwidth)
    log_p = kde_p.logpdf(p.T if p.ndim > 1 else p)
    log_q = kde_q.logpdf(p.T if p.ndim > 1 else p)  # eval q at P's samples
    return float(np.mean(log_p - log_q))


# =============================================================================
# Generic PyMC-based ranker
# =============================================================================
@dataclass
class MCMCHypothesis:
    """A hypothesis instantiated as a function that adds evidence to a PyMC model.

    `apply` receives the model-building callable (which creates the base
    belief network + prior observations inside a `pm.Model` context) and
    returns a modified model (or calls side-effects inside the same
    pm.Model context). The ranker will sample from both the base model
    and the model-with-hypothesis, estimate each posterior, and compute KL.

    The `latent_var` field names the variable of interest (e.g. "theta")
    whose posterior we compare before vs after adding the hypothesis.
    """
    name: str
    summary: str
    apply_fn: Callable[[], None]   # run inside a pm.Model() context to add hypothesis
    latent_var: str


@dataclass
class MCMCEmbeddingScore:
    hypothesis: MCMCHypothesis
    kl_from_base: float
    kl_method: str   # "gaussian_moment" | "kde"
    base_post_mean: float | list
    base_post_std: float | list
    hyp_post_mean: float | list
    hyp_post_std: float | list


def rank_hypotheses_mcmc(
    build_base_model: Callable[[], None],
    hypotheses: list[MCMCHypothesis],
    latent_var: str,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    kl_method: str = "gaussian_moment",
    progressbar: bool = False,
    random_seed: int = 0,
) -> list[MCMCEmbeddingScore]:
    """Rank hypotheses by KL drift under a generic PyMC belief network.

    `build_base_model` is a zero-arg callable that, when invoked inside
    `with pm.Model():`, defines the full base belief network including
    all observed evidence. Each hypothesis's `apply_fn` is run IN ADDITION
    to the base; together they must leave the model in a well-defined,
    samplable state.

    Strategy:
      1. Build base model, MCMC-sample the posterior of `latent_var`.
      2. For each hypothesis h, build base model + apply_fn, MCMC-sample.
      3. KL estimate between the two sample sets.
      4. Sort ascending by KL.
    """
    import pymc as pm

    # --- base posterior ---
    with pm.Model() as base:
        build_base_model()
        base_trace = pm.sample(draws=draws, tune=tune, chains=chains,
                               random_seed=random_seed, progressbar=progressbar,
                               return_inferencedata=False,
                               compute_convergence_checks=False)
    base_samples = np.asarray(base_trace[latent_var])

    # --- per-hypothesis posterior ---
    scores = []
    for i, h in enumerate(hypotheses):
        with pm.Model() as hyp_model:
            build_base_model()
            h.apply_fn()
            hyp_trace = pm.sample(draws=draws, tune=tune, chains=chains,
                                  random_seed=random_seed + 1 + i,
                                  progressbar=progressbar,
                                  return_inferencedata=False,
                                  compute_convergence_checks=False)
        hyp_samples = np.asarray(hyp_trace[latent_var])

        if kl_method == "gaussian_moment":
            kl = kl_gaussian_moment(base_samples, hyp_samples)
        elif kl_method == "kde":
            kl = kl_kde(base_samples, hyp_samples)
        else:
            raise ValueError(f"unknown kl_method: {kl_method}")

        def _mean_std(arr):
            if arr.ndim == 1:
                return float(arr.mean()), float(arr.std(ddof=1))
            return arr.mean(axis=0).tolist(), arr.std(ddof=1, axis=0).tolist()

        bm, bs = _mean_std(base_samples)
        hm, hs = _mean_std(hyp_samples)
        scores.append(MCMCEmbeddingScore(
            hypothesis=h, kl_from_base=kl, kl_method=kl_method,
            base_post_mean=bm, base_post_std=bs,
            hyp_post_mean=hm, hyp_post_std=hs,
        ))

    scores.sort(key=lambda s: s.kl_from_base)
    return scores


def pretty_print_mcmc(scores: list[MCMCEmbeddingScore]) -> None:
    print(f'{"rank":>4}  {"KL":>10}  method              hypothesis')
    print('-' * 90)
    for i, s in enumerate(scores):
        print(f'{i+1:>4}  {s.kl_from_base:10.4f}  {s.kl_method:<18}  {s.hypothesis.name}: {s.hypothesis.summary}')
