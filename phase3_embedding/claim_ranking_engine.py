"""MEIS Phase 3 P3.1 — generic claim-ranking engine.

Implements Plan §Phase 3's formulation:

    D(h, B) = D_KL(P(B) || P(B | h)) + λ · |Δ structure|

as a reusable engine over any PyMC belief network and any list of
candidate claims. The single-domain demo in `demo_claim_ranking.py`
is now a thin wrapper over this engine.

Pluggable axes (selected by string argument for cheap ablation studies):
  - kl_estimator:      "gaussian_moment" | "kde" | "gaussian_fullcov"
  - structural_formula: "bic"  (λ = log(N_observations) / 2 per additional parameter)
                        "count" (λ = 1 per additional parameter; coarser)
                        "none"  (λ = 0; pure KL, used as ablation control)
  - kl_direction:      "base_to_hyp" (default, matches plan text)
                       "hyp_to_base" (reverse; generally different numerical value)

Usage:
    engine = ClaimRankingEngine(
        build_base_model=fn_returning_pm_Model,
        latent_var="weight_A",
        bic_n=30,
        kl_estimator="gaussian_moment",
        structural_formula="bic",
    )
    scores = engine.rank([ClaimSpec(...), ClaimSpec(...)])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from phase3_embedding.kl_drift_mcmc import (
    kl_gaussian_moment, kl_gaussian_fullcov, kl_kde,
)


# =============================================================================
# Spec + score dataclasses
# =============================================================================
@dataclass
class ClaimSpec:
    """A candidate claim: a name, a short description, a PyMC-model builder
    that adds the claim's implied evidence on top of the base network, and
    an integer count of structural additions (new nodes + new edges that
    the claim requires but the base network doesn't have).

    For claims that fully fit the existing vocabulary (e.g. "Alice is taller"
    when heights are already in the network), `structural_additions=0`.
    For claims requiring new orphan concepts (e.g. zodiac), set it to the
    number of new parameters the claim introduces — this drives the BIC
    penalty."""
    name: str
    summary: str
    build_model: Callable[[], object] | None = None  # returns pm.Model
    structural_additions: int = 0
    structural_cost_override: float | None = None


@dataclass
class ClaimScore:
    claim: ClaimSpec
    kl_drift: float
    structural_term: float
    composite: float
    # Audit fields — useful for paper figures / debugging
    base_post_mean: float | None = None
    base_post_std: float | None = None
    hyp_post_mean: float | None = None
    hyp_post_std: float | None = None


# =============================================================================
# Engine
# =============================================================================
@dataclass
class ClaimRankingEngine:
    build_base_model: Callable[[], object]         # returns pm.Model
    latent_var: str                                 # name of the latent to rank KL over
    bic_n: int = 30                                 # "effective sample size" for BIC λ
    kl_estimator: str = "gaussian_moment"
    structural_formula: str = "bic"
    kl_direction: str = "base_to_hyp"

    # MCMC knobs (applied to both base + each hypothesis model)
    draws: int = 1500
    tune: int = 1000
    chains: int = 2
    random_seed: int = 0

    # Internal caches
    _base_samples: np.ndarray | None = field(default=None, init=False, repr=False)

    # -- Lambda coefficient for the structural term --
    def _structural_lambda(self) -> float:
        if self.structural_formula == "bic":
            return math.log(max(self.bic_n, 1)) / 2.0
        if self.structural_formula == "count":
            return 1.0
        if self.structural_formula == "none":
            return 0.0
        raise ValueError(f"unknown structural_formula: {self.structural_formula}")

    # -- KL estimator dispatcher --
    def _compute_kl(self, samples_base: np.ndarray,
                    samples_hyp: np.ndarray) -> float:
        p, q = (samples_base, samples_hyp) if self.kl_direction == "base_to_hyp" \
               else (samples_hyp, samples_base)
        if self.kl_estimator == "gaussian_moment":
            return kl_gaussian_moment(p, q)
        if self.kl_estimator == "gaussian_fullcov":
            return kl_gaussian_fullcov(p, q)
        if self.kl_estimator == "kde":
            return kl_kde(p, q)
        raise ValueError(f"unknown kl_estimator: {self.kl_estimator}")

    # -- Base posterior sampling (cached) --
    def _sample_base(self) -> np.ndarray:
        if self._base_samples is not None:
            return self._base_samples
        import pymc as pm
        base_model = self.build_base_model()
        with base_model:
            trace = pm.sample(draws=self.draws, tune=self.tune,
                              chains=self.chains, random_seed=self.random_seed,
                              progressbar=False, return_inferencedata=False,
                              compute_convergence_checks=False)
        self._base_samples = np.asarray(trace[self.latent_var])
        return self._base_samples

    # -- Main API --
    def rank(self, claims: list[ClaimSpec]) -> list[ClaimScore]:
        import pymc as pm
        lam = self._structural_lambda()
        base_samples = self._sample_base()

        scores: list[ClaimScore] = []
        for i, claim in enumerate(claims):
            if claim.build_model is None:
                # No model: the claim can't be tested; assign a large
                # fallback KL (honest claim specs should always supply
                # a model; this is only to prevent crashes).
                kl = 100.0
                hyp_mean = float("nan")
                hyp_std = float("nan")
            else:
                hyp_model = claim.build_model()
                with hyp_model:
                    trace = pm.sample(
                        draws=self.draws, tune=self.tune, chains=self.chains,
                        random_seed=self.random_seed + 1 + i,
                        progressbar=False, return_inferencedata=False,
                        compute_convergence_checks=False,
                    )
                hyp_samples = np.asarray(trace[self.latent_var])
                kl = self._compute_kl(base_samples, hyp_samples)
                hyp_mean = float(hyp_samples.mean())
                hyp_std = float(hyp_samples.std(ddof=1))

            if claim.structural_cost_override is not None:
                structural = claim.structural_cost_override
            else:
                structural = lam * claim.structural_additions

            composite = kl + structural
            scores.append(ClaimScore(
                claim=claim,
                kl_drift=kl,
                structural_term=structural,
                composite=composite,
                base_post_mean=float(base_samples.mean()),
                base_post_std=float(base_samples.std(ddof=1)),
                hyp_post_mean=hyp_mean,
                hyp_post_std=hyp_std,
            ))

        scores.sort(key=lambda s: s.composite)
        return scores


def pretty_print_engine_scores(scores: list[ClaimScore]) -> None:
    print(f"{'rank':>4}  {'claim':<16}  {'KL':>9}  {'Δ struct':>9}  "
          f"{'composite':>11}  description")
    print('-' * 108)
    for i, s in enumerate(scores):
        print(f"{i+1:>4}  {s.claim.name:<16}  "
              f"{s.kl_drift:9.4f}  {s.structural_term:9.3f}  "
              f"{s.composite:11.3f}  {s.claim.summary[:58]}")
