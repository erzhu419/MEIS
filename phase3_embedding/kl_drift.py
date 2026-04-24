"""MEIS Phase 3 L3 — minimum embedding distance via KL drift.

Given a belief network (parameterised by continuous latent θ with a current
posterior P(θ | D) over observed data D), and a set of candidate hypotheses
{h_1, ..., h_k} each of which adds new evidence to the network, rank the
hypotheses by how little they perturb the existing posterior:

    D(h) = KL( P(θ | D) || P(θ | D, h) )

Smaller D(h) = the new hypothesis fits the current belief network with less
revision = more coherent / "best explanation" under Lakatos / Quine style
minimum-disturbance abduction.

This is the first concrete instantiation of MEIS blueprint §3 "最小扰动
嵌入评分". Deliberately restricted to the Gaussian conjugate case for
Phase 1 (alice_charlie has one Normal latent θ with Normal likelihood),
which admits an exact closed-form posterior and hence exact KL. Phase 2+
will generalise to MCMC posteriors with KL estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import math

import numpy as np


# -----------------------------------------------------------------------------
# Gaussian conjugate update
# -----------------------------------------------------------------------------
@dataclass
class GaussianPosterior:
    """Normal(mu, sigma^2) distribution over a single scalar latent θ."""
    mu: float
    sigma: float

    @property
    def var(self) -> float:
        return self.sigma ** 2

    @property
    def prec(self) -> float:
        return 1.0 / self.var


def kl_normal(p: GaussianPosterior, q: GaussianPosterior) -> float:
    """KL( N(μp, σp²) || N(μq, σq²) ) closed form, nats.

        KL = log(σq/σp) + (σp² + (μp - μq)²) / (2 σq²) - 1/2
    """
    return (
        math.log(q.sigma / p.sigma)
        + (p.var + (p.mu - q.mu) ** 2) / (2.0 * q.var)
        - 0.5
    )


def condition_normal(prior: GaussianPosterior, x: np.ndarray, y: np.ndarray,
                     obs_sigma: float) -> GaussianPosterior:
    """Conjugate update for the model y_i ~ Normal(θ · x_i, obs_sigma^2).

    If y is a value and x is a coefficient, one observation contributes
    precision x^2/σ² to θ and mean shift x·y/σ². Full posterior:

        post_prec = prior_prec + Σ x_i² / σ²
        post_mean = (prior_prec·prior_mean + Σ x_i·y_i / σ²) / post_prec
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape
    inv_noise = 1.0 / (obs_sigma ** 2)
    post_prec = prior.prec + float(np.sum(x ** 2) * inv_noise)
    post_var = 1.0 / post_prec
    post_mean = post_var * (prior.mu * prior.prec +
                            float(np.sum(x * y) * inv_noise))
    return GaussianPosterior(mu=post_mean, sigma=math.sqrt(post_var))


# -----------------------------------------------------------------------------
# Hypothesis abstraction
# -----------------------------------------------------------------------------
@dataclass
class Hypothesis:
    """A candidate claim about the system that can be represented as one or
    more synthetic observations added to the belief network.

    For the alice_charlie env's single-latent model `w = θ · h³ + N(0, σ_obs)`:
    a hypothesis is a list of (height_cm, weight_kg) pairs that the claim
    implies.
    """
    name: str
    summary: str
    synthetic_obs: list[tuple[float, float]]
    obs_sigma: float = 2.0   # inherits alice_charlie OBS_NOISE default

    def xy_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.array([h ** 3 for h, _ in self.synthetic_obs], dtype=float)
        ys = np.array([w for _, w in self.synthetic_obs], dtype=float)
        return xs, ys


# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------
@dataclass
class EmbeddingScore:
    hypothesis: Hypothesis
    kl_from_base: float
    structural_edit_count: int = 0
    @property
    def composite(self) -> float:
        """D(h, B) = KL + λ · |Δ structure|.  λ default 0 for Phase 1
        (pure Gaussian case has no structural edits)."""
        return self.kl_from_base + 0.0 * self.structural_edit_count


def rank_hypotheses(
    posterior_D: GaussianPosterior,
    hypotheses: list[Hypothesis],
) -> list[EmbeddingScore]:
    """Rank hypotheses by KL( P(θ|D) || P(θ|D, h) ), ascending."""
    scores = []
    for h in hypotheses:
        xs, ys = h.xy_vectors()
        post_Dh = condition_normal(posterior_D, xs, ys, h.obs_sigma)
        scores.append(EmbeddingScore(
            hypothesis=h,
            kl_from_base=kl_normal(posterior_D, post_Dh),
        ))
    scores.sort(key=lambda s: s.composite)
    return scores


def pretty_print(scores: list[EmbeddingScore]) -> None:
    print(f'{"rank":>4}  {"KL":>8}  hypothesis')
    print('-' * 72)
    for i, s in enumerate(scores):
        print(f'{i+1:>4}  {s.kl_from_base:8.4f}  {s.hypothesis.name}: {s.hypothesis.summary}')


# -----------------------------------------------------------------------------
# Convenience: build a posterior starting from a prior + observation list
# -----------------------------------------------------------------------------
def posterior_from_observations(
    prior: GaussianPosterior,
    observations: list[tuple[float, float]],
    obs_sigma: float,
) -> GaussianPosterior:
    xs = np.array([h ** 3 for h, _ in observations], dtype=float)
    ys = np.array([w for _, w in observations], dtype=float)
    return condition_normal(prior, xs, ys, obs_sigma)
