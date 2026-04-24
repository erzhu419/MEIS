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
# Legacy shim — the engine (P3.1) is the new home for these abstractions
# =============================================================================
from phase3_embedding.claim_ranking_engine import (
    ClaimSpec, ClaimScore, ClaimRankingEngine, pretty_print_engine_scores,
)

# Backward-compatible alias so existing tests can still import `Claim`.
Claim = ClaimSpec


# =============================================================================
# Base belief network builder (used by the engine)
# =============================================================================
def _build_base_alice_charlie():
    """Base belief network: weak Alice-taller-than-Bob evidence."""
    return build_model(alice_taller_than_bob=True,
                       height_diff_mean=2.0, height_diff_sigma=5.0,
                       equal_shoes=False, alice_footprint_deeper_by=None)


# Legacy tests may still import these helpers
def _kl_normal_approx(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    mu_p, mu_q = float(samples_p.mean()), float(samples_q.mean())
    var_p = max(float(samples_p.var(ddof=1)), 1e-20)
    var_q = max(float(samples_q.var(ddof=1)), 1e-20)
    return (math.log(math.sqrt(var_q / var_p))
            + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
            - 0.5)


def _sample_base(draws: int = 1500, tune: int = 1000, random_seed: int = 0):
    model = _build_base_alice_charlie()
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
            build_model=_claim_taller,
            structural_additions=0,
        ),
        Claim(
            name="H_denser",
            summary="Alice is ~5% denser than Charlie (uses existing theta node)",
            build_model=_claim_denser,
            structural_additions=0,
        ),
        Claim(
            name="H_larger_feet",
            summary="Alice has larger feet than Charlie (uses existing shoe node)",
            build_model=_claim_larger_feet,
            structural_additions=0,
        ),
        Claim(
            name="H_zodiac",
            summary=("Alice was born in the Year of the Tiger; Charlie was not "
                     "(requires 2 new orphan Categorical nodes; disconnected "
                     "from weight in the existing belief network)"),
            build_model=_claim_zodiac,  # real PyMC model; KL will be ~0
            structural_additions=2,        # 2 orphan nodes; BIC penalty log(N)/2 · 2
        ),
    ]


# =============================================================================
# Ranking
# =============================================================================
def rank_claims(
    claims: list[ClaimSpec] | None = None,
    latent_var: str = "weight_A",
    *,
    bic_n: int = 30,
    draws: int = 1500, tune: int = 1000,
    random_seed: int = 0,
    verbose: bool = True,
    kl_estimator: str = "gaussian_moment",
    structural_formula: str = "bic",
) -> list[ClaimScore]:
    """Rank claims ascending by D(h) = KL + lambda · |Δstructure|.

    Thin wrapper over ClaimRankingEngine — all new work should go through
    the engine directly. This function exists for backwards compatibility
    with code written during Phase 2 realignment.

    `kl_estimator` and `structural_formula` are passed through so simple
    ablations can be done at the demo level too.
    """
    claims = claims or _build_claims()
    # ClaimSpec uses `build_model`; legacy Claim used `apply_evidence` (same
    # field on the alias). Map forward if needed.
    normalized: list[ClaimSpec] = []
    for c in claims:
        build_model_fn = getattr(c, "build_model", None) or getattr(c, "apply_evidence", None)
        normalized.append(ClaimSpec(
            name=c.name, summary=c.summary,
            build_model=build_model_fn,
            structural_additions=c.structural_additions,
            structural_cost_override=getattr(c, "structural_cost_override", None),
        ))

    engine = ClaimRankingEngine(
        build_base_model=_build_base_alice_charlie,
        latent_var=latent_var,
        bic_n=bic_n,
        kl_estimator=kl_estimator,
        structural_formula=structural_formula,
        draws=draws, tune=tune, random_seed=random_seed,
    )
    scores = engine.rank(normalized)

    if verbose:
        pretty_print_engine_scores(scores)
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
