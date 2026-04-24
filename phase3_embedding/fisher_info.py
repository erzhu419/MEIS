"""MEIS Phase 2 P2.4 — Fisher information for observation selection.

Plan §Phase 2 §5: "Fisher 信息矩阵能正确排序'哪个下一步观测最值钱'".

Expected Fisher Information (EFI) at a candidate query point `x_q`
tells you how much information a future observation `y_q ~ p(y | theta, x_q)`
would contribute to tightening the posterior on theta, under the current
posterior. Higher EFI = more informative next observation.

This is a **conjugate-to-EIG** metric that can substitute for the
nested-MC Expected Information Gain (EIG) that boxing-gym uses on its
own envs. The two agree in the Gaussian-linear limit and diverge on
non-Gaussian likelihoods.

Supported likelihoods in this module (Phase 2):
  - Normal(theta * x, obs_sigma)
  - Poisson(exp(theta * x))

Uses jax for automatic differentiation of the log-likelihood to get
the expected observed-info matrix I(theta) = -E_y[∂²/∂theta² log p(y|theta, x)].
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Observed Fisher Information for two canonical likelihoods
# =============================================================================
def observed_fisher_normal(theta: float, x: float, obs_sigma: float) -> float:
    """For y ~ Normal(theta * x, obs_sigma^2), the Fisher info w.r.t. theta
    is I(theta) = x^2 / obs_sigma^2 (constant in theta, hence observed
    fisher equals expected fisher)."""
    return float(x ** 2 / obs_sigma ** 2)


def observed_fisher_poisson_lograte(theta: float, x: float) -> float:
    """For y ~ Poisson(exp(theta * x)), d²/dtheta² log p(y|theta,x)
    = -x^2 * exp(theta * x). I(theta) = E_y[-d²...] = x^2 * lambda
    where lambda = exp(theta * x)."""
    return float((x ** 2) * np.exp(theta * x))


# =============================================================================
# Expected Fisher Information (over current posterior on theta)
# =============================================================================
def expected_fisher_information(
    query_points: np.ndarray | list[float],
    theta_samples: np.ndarray,
    likelihood: str,
    obs_sigma: float | None = None,
) -> np.ndarray:
    """Expected Fisher information at each query point, averaged over the
    posterior samples of theta.

    Returns an array of shape (len(query_points),) where entry i is
        EFI(x_i) = E_{theta ~ posterior} [ I(theta, x_i) ].

    Higher EFI = more informative observation at that query point.
    Rank ascending for "least informative" (same cost as a nested-MC EIG
    minimum).
    """
    query_points = np.asarray(query_points, dtype=float).reshape(-1)
    theta_samples = np.asarray(theta_samples, dtype=float).reshape(-1)

    out = np.zeros(query_points.shape, dtype=float)
    for i, x in enumerate(query_points):
        if likelihood == "normal":
            assert obs_sigma is not None
            # I(theta) doesn't depend on theta for linear-Gaussian, so the
            # average just returns the constant.
            out[i] = observed_fisher_normal(0.0, float(x), obs_sigma)
        elif likelihood == "poisson":
            vals = (x ** 2) * np.exp(theta_samples * x)
            out[i] = float(np.mean(vals))
        else:
            raise ValueError(f"unsupported likelihood: {likelihood}")
    return out


# =============================================================================
# Automatic-differentiation version (jax) — for more general likelihoods later
# =============================================================================
def expected_fisher_via_jax(
    query_points: np.ndarray | list[float],
    theta_samples: np.ndarray,
    log_lik_fn,            # (theta, x, y) -> scalar log p
    expected_y_fn,         # (theta, x) -> expected y  (for plug-in estimator)
) -> np.ndarray:
    """Expected Fisher via jax.hessian of the log-likelihood, averaged over
    theta ~ posterior samples, with the observation `y` at its expected
    value (plug-in MLE approximation, adequate when likelihood is
    well-behaved).

    This is more general than the closed-form branches above: pass in any
    differentiable `log_lik_fn(theta, x, y)` and `expected_y_fn(theta, x)`.
    Used mainly as a sanity check and for extensibility to new likelihoods.
    """
    hess = jax.hessian(log_lik_fn, argnums=0)  # d²/dtheta²

    query_points = jnp.asarray(query_points, dtype=float).reshape(-1)
    theta_samples = jnp.asarray(theta_samples, dtype=float).reshape(-1)

    out = np.zeros(query_points.shape, dtype=float)
    for i, x in enumerate(query_points):
        vals = []
        for theta in theta_samples:
            y_exp = expected_y_fn(theta, x)
            # Observed info = -hess of log lik at expected y
            info = -float(hess(theta, x, y_exp))
            vals.append(info)
        out[i] = float(np.mean(vals))
    return out


# =============================================================================
# Ranking
# =============================================================================
@dataclass
class ObservationCandidate:
    name: str
    x: float


def rank_observation_candidates(
    candidates: list[ObservationCandidate],
    theta_samples: np.ndarray,
    likelihood: str,
    obs_sigma: float | None = None,
) -> list[tuple[ObservationCandidate, float]]:
    """Rank candidate observations by expected Fisher info, descending
    (most-informative first)."""
    xs = [c.x for c in candidates]
    efi = expected_fisher_information(xs, theta_samples, likelihood,
                                      obs_sigma=obs_sigma)
    scored = list(zip(candidates, efi))
    scored.sort(key=lambda p: -p[1])  # descending
    return scored


def pretty_print_ranking(scored: list[tuple[ObservationCandidate, float]]) -> None:
    print(f'{"rank":>4}  {"candidate":<20}  {"x":>8}  {"EFI":>14}')
    print('-' * 60)
    for i, (c, efi) in enumerate(scored):
        print(f'{i+1:>4}  {c.name:<20}  {c.x:8.3f}  {efi:14.6f}')
