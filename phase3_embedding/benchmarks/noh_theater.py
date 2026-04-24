"""Noh-theater gender monopoly benchmark (Plan §Phase 3 success criterion).

Historical fact to be explained:
    In pre-modern Japan, women were excluded from performing Noh (能).
    The monopoly persisted roughly 14th–19th centuries.

We encode this as a belief network over latent cultural/religious
quantities and rank four candidate explanations by D(h, B) = KL + λ·|Δstruct|.

Key design choice: social-science concepts that don't have a natural
continuous scale are modeled as *proxies* on [0, 1] with soft priors:

    ritual_role           ∈ [0, 1]  — how "ceremonial-religious" Noh is
                                      (vs pure secular entertainment)
    female_blood_taboo    ∈ [0, 1]  — Shinto/Buddhist "blood pollution" belief strength
    shogunate_ban_effect  ∈ [0, 1]  — 1629 Kabuki ban's spillover onto Noh
    natural_voice_fit     ∈ [0, 1]  — "women can't do Noh singing" (orphan claim)

Observed proxy facts (soft-weighted into the posterior on P(women_banned=1)):
    - Okina ceremony required purification & celibacy before performance
    - Some Shinto shrines historically forbade women entry ("nyonin-kekkai")
    - Women DID perform Kabuki but were banned in 1629 by shogunate
    - Female Noh performers DID exist in some temple contexts (Kumano-Bikuni)

Candidate explanations for "why women were banned":
    H_priestly         : ritual_role strongly implies exclusion (in-network)
    H_blood_taboo      : female_blood_taboo drives exclusion (in-network)
    H_shogunate_ban    : shogunate_ban_effect explains (in-network, but weaker
                        causal link in base model)
    H_natural_voice_fit: voice-based explanation — encoded as a claim with a
                        new orphan node (no historical/structural evidence
                        supports "women's voices inherently unsuitable for
                        Noh vocalization"). Expected to rank LAST.

Pure PyMC, 0 API.
"""

from __future__ import annotations

import numpy as np
import pymc as pm


# =============================================================================
# Priors on latent cultural quantities
# =============================================================================
RITUAL_ROLE_MU, RITUAL_ROLE_SIGMA = 0.75, 0.15          # Noh is heavily ritual
BLOOD_TABOO_MU, BLOOD_TABOO_SIGMA = 0.55, 0.20          # strong but not absolute
SHOGUNATE_MU, SHOGUNATE_SIGMA = 0.25, 0.15              # 1629 ban was Kabuki-specific; spillover weak

# Latent event strengths (each in [0,1] via logistic transform of a Normal)
def _bounded_01(name: str, mu: float, sigma: float):
    raw = pm.Normal(f"{name}_raw", mu=mu, sigma=sigma)
    return pm.Deterministic(name, pm.math.sigmoid(raw * 4.0 - 2.0))


# =============================================================================
# Base belief network
# =============================================================================
def build_base_model():
    """Base belief network with observed proxy facts but no specific
    explanation for the ban yet."""
    with pm.Model() as model:
        ritual_role = pm.Beta("ritual_role", alpha=6.0, beta=2.0)          # ~0.75
        blood_taboo = pm.Beta("blood_taboo", alpha=3.0, beta=2.5)          # ~0.55
        shogunate = pm.Beta("shogunate_ban_effect", alpha=2.0, beta=6.0)   # ~0.25

        # Proxy observations, each weighting the relevant latent upward.
        # "Some shrines had nyonin-kekkai" supports blood_taboo mid-high
        pm.Beta("obs_nyonin_kekkai",
                alpha=1.0 + 5.0 * blood_taboo,
                beta=1.0 + 5.0 * (1.0 - blood_taboo),
                observed=0.6)   # historians report yes-but-not-universal

        # "Okina ceremony requires purification" supports ritual_role mid-high
        pm.Beta("obs_okina_purification",
                alpha=1.0 + 5.0 * ritual_role,
                beta=1.0 + 5.0 * (1.0 - ritual_role),
                observed=0.85)  # strong evidence (primary source)

        # "1629 ban applied to Kabuki, not Noh directly" — shogunate spillover weak
        pm.Beta("obs_shogunate_scope",
                alpha=1.0 + 5.0 * shogunate,
                beta=1.0 + 5.0 * (1.0 - shogunate),
                observed=0.15)

        # Probability that women were banned from Noh, as a function of the
        # latent explanatory factors. We use a logistic-like combination.
        # P(banned) rises with any of the three; max out around 0.95.
        # Weights encode relative strength: ritual_role dominates (ceremony),
        # blood_taboo secondary, shogunate weak.
        logit = (2.5 * ritual_role + 1.8 * blood_taboo
                 + 0.8 * shogunate - 2.0)
        p_banned = pm.Deterministic("p_banned", pm.math.sigmoid(logit))

        # Observe that women WERE banned (historical fact).
        pm.Bernoulli("women_banned_historical", p=p_banned, observed=1)

    return model


# =============================================================================
# Claim models — each one adds a soft-evidence constraint that elevates the
# corresponding explanatory factor
# =============================================================================
def claim_priestly():
    """'Noh was banned because of its priestly/ritual role.' Evidence: add
    a soft observation that ritual_role is very high (say 0.9 ± 0.05)."""
    model = build_base_model()
    with model:
        rr = model["ritual_role"]
        pm.Normal("claim_priestly_obs",
                  mu=rr - 0.9, sigma=0.05, observed=0.0)
    return model


def claim_blood_taboo():
    """'Banned because of Shinto/Buddhist blood-pollution taboo.'"""
    model = build_base_model()
    with model:
        bt = model["blood_taboo"]
        pm.Normal("claim_blood_obs",
                  mu=bt - 0.85, sigma=0.05, observed=0.0)
    return model


def claim_shogunate():
    """'Banned because of the 1629 shogunate spillover effect.'"""
    model = build_base_model()
    with model:
        sg = model["shogunate_ban_effect"]
        pm.Normal("claim_sg_obs",
                  mu=sg - 0.85, sigma=0.05, observed=0.0)
    return model


def claim_natural_voice():
    """Orphan claim: 'Women's voices are inherently unsuitable for Noh.'
    This introduces a new latent node `voice_fit` that is DISCONNECTED from
    all existing factors and from p_banned — no causal chain in the base
    belief network links voice properties to a historical policy decision
    about who performs."""
    model = build_base_model()
    with model:
        # Orphan latent: Beta prior, observed extreme value,
        # but never enters the p_banned logit. Hence KL on p_banned ≈ 0.
        voice_fit_a = pm.Beta("voice_fit_alice", alpha=2.0, beta=2.0)
        voice_fit_b = pm.Beta("voice_fit_b", alpha=2.0, beta=2.0)
        pm.Normal("claim_voice_obs",
                  mu=voice_fit_a - 0.1, sigma=0.05, observed=0.0)
        pm.Normal("claim_voice_obs2",
                  mu=voice_fit_b - 0.1, sigma=0.05, observed=0.0)
    return model


# =============================================================================
# Public entry point — a list of canonical claim specs for the engine
# =============================================================================
def get_claims():
    """Return the 4 canonical Noh claims in the format expected by
    ClaimRankingEngine."""
    from phase3_embedding.claim_ranking_engine import ClaimSpec
    return [
        ClaimSpec(
            name="H_priestly",
            summary="Noh banned women because of its priestly/ritual role (in-network)",
            build_model=claim_priestly,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_blood_taboo",
            summary="Banned because of Shinto/Buddhist blood-pollution taboo (in-network)",
            build_model=claim_blood_taboo,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_shogunate_ban",
            summary="Banned because of 1629 shogunate spillover (in-network, weak link)",
            build_model=claim_shogunate,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_natural_voice",
            summary="Banned because female voices are unsuitable for Noh "
                    "(2 orphan voice-fit nodes, disconnected from p_banned)",
            build_model=claim_natural_voice,
            structural_additions=2,
        ),
    ]


if __name__ == "__main__":
    from phase3_embedding.claim_ranking_engine import (
        ClaimRankingEngine, pretty_print_engine_scores,
    )
    print("Noh-theater gender monopoly — D(h, B) ranking\n")
    engine = ClaimRankingEngine(
        build_base_model=build_base_model,
        latent_var="p_banned",
        bic_n=10,   # ~10 historical proxy observations
        kl_estimator="gaussian_moment",
        structural_formula="bic",
        draws=1200, tune=800, random_seed=0,
    )
    scores = engine.rank(get_claims())
    pretty_print_engine_scores(scores)
    print()
    print("Expected ordering (most coherent → least): 3 in-vocab explanations")
    print("then H_natural_voice (orphan nodes → BIC penalty dominates).")
