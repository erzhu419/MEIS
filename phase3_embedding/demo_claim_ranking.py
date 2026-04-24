"""MEIS Phase 2 — claim-ranking demo (Plan §0.2 元评价系统).

Plan §0.2 frames MEIS as a META-EVALUATION system: given a set of
natural-language claims explaining an observed fact, rank them by
"coherence with existing belief network". The operationalization
(Plan §Phase 3) is:

    D(h, B) = KL(P(B) || P(B | h)) + λ · |Δ structure|

  - KL term  — how much the claim's implied evidence perturbs the
                existing posterior.
  - Δstructure — how many new nodes / edges the claim requires that
                 aren't supported by data.

This demo ranks 4 candidate explanations for "Alice is heavier than
Charlie" in the 3-person Alice-Charlie belief network built in
demo_alice_charlie_chain.py.

Key acceptance: the **structural** penalty from embedding "Alice is a
Tiger-year zodiac" (requires +1 new node + edge, neither supported by
data) dominates the composite score, so H_zodiac correctly ranks LAST
with D(h) orders of magnitude above any claim that fits existing
vocabulary.

Pure math, 0 API.

Note on what KL measures:
  KL(base || hyp) over the latent weight_A measures the POSTERIOR SHIFT
  the claim induces, which is "information gain" rather than "coherence".
  A claim that directly sharpens weight_A's posterior (e.g., "Alice is
  5 cm taller") has larger KL than a claim that merely adds a side
  constraint (e.g., "Alice has larger feet"). Both are coherent with
  the network. Only the zodiac claim is INCOHERENT (requires orphan
  structure) — that's the main signal we want to detect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pymc as pm

from phase3_embedding.demo_alice_charlie_chain import build_model, _sample


# =============================================================================
# Claim abstraction
# =============================================================================
@dataclass
class Claim:
    name: str
    summary: str
    # A function run inside a PyMC model context that adds the claim's
    # implied observation. If None, the claim is purely structural.
    apply_evidence: Callable[[], None] | None = None
    # Count of new graph elements (nodes + edges) the claim requires that
    # the existing belief network doesn't have. Zero means the claim fits
    # into existing vocabulary.
    structural_additions: int = 0


@dataclass
class ClaimScore:
    claim: Claim
    kl_drift: float
    structural_term: float
    composite: float


# =============================================================================
# KL-drift estimator over posterior samples (reuse P2.2 machinery)
# =============================================================================
def _kl_normal_approx(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """Fit a univariate Normal to each sample set, return closed-form KL(p||q)."""
    mu_p, mu_q = float(samples_p.mean()), float(samples_q.mean())
    var_p = float(samples_p.var(ddof=1))
    var_q = float(samples_q.var(ddof=1))
    # Guard against near-degenerate posteriors (e.g., when a claim collapses
    # the posterior to a near-point).
    var_p = max(var_p, 1e-20)
    var_q = max(var_q, 1e-20)
    return (
        math.log(math.sqrt(var_q / var_p))
        + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
        - 0.5
    )


# =============================================================================
# Build the base belief network (same as the multi-obs demo's "stage 1")
# =============================================================================
def _sample_base(draws: int = 1500, tune: int = 1000, random_seed: int = 0):
    """Base belief network: weak Alice-taller-than-Bob evidence (no other obs)."""
    model = build_model(alice_taller_than_bob=True,
                        height_diff_mean=2.0, height_diff_sigma=5.0,
                        equal_shoes=False, alice_footprint_deeper_by=None)
    return model, _sample(model, draws=draws, tune=tune, random_seed=random_seed)


# =============================================================================
# Claim builders (each adds a distinct piece of evidence to the base model)
# =============================================================================
def _claim_taller():
    model = build_model(alice_taller_than_bob=True,
                        height_diff_mean=5.0, height_diff_sigma=2.0,
                        equal_shoes=False, alice_footprint_deeper_by=None)
    return model


def _claim_denser():
    """Encode 'Alice is noticeably denser than Charlie' by adding a soft
    inequality on the per-person theta."""
    with pm.Model() as model:
        # rebuild the base network
        from phase3_embedding.demo_alice_charlie_chain import (
            THETA_MU, THETA_SIGMA, HEIGHT_MU, HEIGHT_SIGMA,
            SHOE_MU, SHOE_SIGMA,
        )
        theta_A = pm.Normal("theta_A", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_B = pm.Normal("theta_B", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_C = pm.Normal("theta_C", mu=THETA_MU, sigma=THETA_SIGMA)
        height_B = pm.Normal("height_B", mu=HEIGHT_MU, sigma=HEIGHT_SIGMA)
        height_C = pm.Deterministic("height_C", height_B)
        height_A = pm.Deterministic(
            "height_A",
            height_B + pm.Normal("dh_alice", mu=2.0, sigma=5.0))
        pm.Deterministic("weight_A", theta_A * height_A ** 3)
        pm.Deterministic("weight_B", theta_B * height_B ** 3)
        pm.Deterministic("weight_C", theta_C * height_C ** 3)
        pm.Normal("shoe_A", mu=SHOE_MU, sigma=SHOE_SIGMA)
        pm.Normal("shoe_B", mu=SHOE_MU, sigma=SHOE_SIGMA)
        pm.Normal("shoe_C", mu=SHOE_MU, sigma=SHOE_SIGMA)
        # Claim evidence: theta_A - theta_C ≈ +8e-7 (i.e. ~5% denser)
        pm.Normal("theta_gap", mu=theta_A - theta_C, sigma=2e-7,
                  observed=8e-7)
    return model


def _claim_larger_feet():
    """Encode 'Alice has larger feet than Charlie' as shoe_A - shoe_C = +3."""
    model = build_model(alice_taller_than_bob=True,
                        height_diff_mean=2.0, height_diff_sigma=5.0,
                        equal_shoes=False, alice_footprint_deeper_by=None)
    with model:
        shoe_A = model["shoe_A"]
        shoe_C = model["shoe_C"]
        pm.Normal("feet_gap", mu=shoe_A - shoe_C, sigma=0.3, observed=3.0)
    return model


def _claim_zodiac():
    """Honest encoding of 'Alice is Year-of-Tiger, Charlie isn't' — adds
    two DISCONNECTED Categorical nodes (no causal link to weight). Observing
    zodiac values therefore does NOT affect theta's posterior; KL ≈ 0. The
    structural term (BIC for 2 new parameters without data-support) is
    what makes this claim disruptive."""
    import pymc as pm
    from phase3_embedding.demo_alice_charlie_chain import (
        THETA_MU, THETA_SIGMA, HEIGHT_MU, HEIGHT_SIGMA, SHOE_MU, SHOE_SIGMA,
    )
    with pm.Model() as model:
        theta_A = pm.Normal("theta_A", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_B = pm.Normal("theta_B", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_C = pm.Normal("theta_C", mu=THETA_MU, sigma=THETA_SIGMA)
        height_B = pm.Normal("height_B", mu=HEIGHT_MU, sigma=HEIGHT_SIGMA)
        height_C = pm.Deterministic("height_C", height_B)
        dh = pm.Normal("dh_alice", mu=2.0, sigma=5.0)
        height_A = pm.Deterministic("height_A", height_B + dh)
        pm.Deterministic("weight_A", theta_A * height_A ** 3)
        pm.Deterministic("weight_B", theta_B * height_B ** 3)
        pm.Deterministic("weight_C", theta_C * height_C ** 3)
        pm.Normal("shoe_A", mu=SHOE_MU, sigma=SHOE_SIGMA)
        pm.Normal("shoe_B", mu=SHOE_MU, sigma=SHOE_SIGMA)
        pm.Normal("shoe_C", mu=SHOE_MU, sigma=SHOE_SIGMA)
        # Orphan zodiac nodes — uniform prior over 12 values
        zodiac_A = pm.Categorical("zodiac_A", p=[1/12] * 12)
        zodiac_C = pm.Categorical("zodiac_C", p=[1/12] * 12)
        # Observe: zodiac_A == tiger (idx 2), zodiac_C != tiger
        pm.Potential("zA_obs", pm.math.switch(pm.math.eq(zodiac_A, 2), 0.0, -1e10))
        pm.Potential("zC_obs", pm.math.switch(pm.math.eq(zodiac_C, 2), -1e10, 0.0))
    return model


def _build_claims() -> list[Claim]:
    return [
        Claim(
            name="H_taller",
            summary="Alice is 5 cm taller than Charlie (fits existing height structure)",
            apply_evidence=_claim_taller,
            structural_additions=0,
        ),
        Claim(
            name="H_denser",
            summary="Alice is ~5% denser than Charlie (uses existing theta node)",
            apply_evidence=_claim_denser,
            structural_additions=0,
        ),
        Claim(
            name="H_larger_feet",
            summary="Alice has larger feet than Charlie (uses existing shoe node)",
            apply_evidence=_claim_larger_feet,
            structural_additions=0,
        ),
        Claim(
            name="H_zodiac",
            summary=("Alice was born in the Year of the Tiger; Charlie was not "
                     "(requires 2 new orphan Categorical nodes; disconnected "
                     "from weight in the existing belief network)"),
            apply_evidence=_claim_zodiac,  # real PyMC model; KL will be ~0
            structural_additions=2,        # 2 orphan nodes; BIC penalty log(N)/2 · 2
        ),
    ]


# =============================================================================
# Ranking
# =============================================================================
def rank_claims(
    claims: list[Claim] | None = None,
    latent_var: str = "weight_A",
    *,
    bic_n: int = 30,
    draws: int = 1500, tune: int = 1000,
    random_seed: int = 0,
    verbose: bool = True,
) -> list[ClaimScore]:
    """Rank claims ascending by D(h) = KL(P(B)||P(B|h)) + lambda·|Δstructure|.

    lambda ≈ log(N)/2 per BIC; with N=30 observations in the belief network,
    lambda ≈ 1.7. Each additional structural element (new node or edge
    without data support) thus contributes ~1.7 nats to D(h).
    """
    claims = claims or _build_claims()
    bic_lambda = math.log(max(bic_n, 1)) / 2.0

    # Base posterior samples for the latent we care about (weight_A here).
    # We also need weight_C to compute the difference for interpretability.
    base_model, base_trace = _sample_base(draws=draws, tune=tune,
                                          random_seed=random_seed)
    base_samples = np.asarray(base_trace[latent_var])

    scores: list[ClaimScore] = []
    for claim in claims:
        if claim.apply_evidence is None:
            # No model provided: assume the claim cannot be tested against
            # the belief network numerically. Use a large KL floor only as
            # a last resort — honest claims below provide their own model.
            kl = 100.0
        else:
            hyp_model = claim.apply_evidence()
            hyp_trace = _sample(hyp_model, draws=draws, tune=tune,
                                random_seed=random_seed + 1)
            hyp_samples = np.asarray(hyp_trace[latent_var])
            kl = _kl_normal_approx(base_samples, hyp_samples)
        structural = bic_lambda * claim.structural_additions
        composite = kl + structural
        scores.append(ClaimScore(claim=claim, kl_drift=kl,
                                 structural_term=structural,
                                 composite=composite))

    scores.sort(key=lambda s: s.composite)

    if verbose:
        print(f"\n{'rank':>4}  {'claim':<14}  "
              f"{'KL drift':>9}  {'Δ struct':>9}  {'composite':>11}  description")
        print('-' * 110)
        for i, s in enumerate(scores):
            print(f"{i+1:>4}  {s.claim.name:<14}  "
                  f"{s.kl_drift:9.4f}  {s.structural_term:9.3f}  "
                  f"{s.composite:11.3f}  {s.claim.summary[:60]}")

    return scores


if __name__ == "__main__":
    print("MEIS Phase 2 — claim-ranking demo (Plan §0.2 元评价系统)\n")
    print("Base belief: Alice plausibly taller than Bob; Bob == Charlie.")
    print("Ranking 4 claims explaining 'Alice is heavier than Charlie'.")
    scores = rank_claims(verbose=True)
    print()
    print("Acceptance criterion: H_zodiac must rank LAST with composite D(h)")
    print("  at least an order of magnitude above any in-vocabulary claim.")
    print("  (Non-zodiac KL ordering depends on how directly the claim")
    print("   constrains weight_A and is not the primary signal.)")
