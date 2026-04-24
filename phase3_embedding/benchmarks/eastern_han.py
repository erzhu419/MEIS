"""Eastern Han emperor early-death benchmark (Plan §Phase 3 success criterion).

Historical fact to be explained:
    The median age at death for Eastern Han (25-220 CE) emperors is
    strikingly low (~21 years). Candidates for cause have been debated
    by historians — here we rank 4 with MEIS's D(h, B).

Key design choice: each candidate cause gets a latent `strength ∈ [0, 1]`
proxy. Observed soft facts (from primary sources) pull the latents.
P(emperor_young_at_death | causes) aggregates via a logistic.

Candidate explanations:
    H_lead_poisoning       — lead-based cosmetics/wine vessels common at court
    H_political_stress     — eunuch/consort-kin factional violence shortened lives
    H_incest_bottleneck    — intra-clan marriage concentrated genetic defects
    H_orphan_geomancy      — "palace built on bad fengshui" — no causal chain in
                             the network, expected to rank LAST via structural penalty
"""

from __future__ import annotations

import numpy as np
import pymc as pm


# =============================================================================
# Base belief network
# =============================================================================
def build_base_model():
    with pm.Model() as model:
        # Latent cause strengths, each in [0, 1] via Beta(α, β)
        lead = pm.Beta("lead_poisoning_strength", alpha=4.0, beta=3.0)       # 0.57
        stress = pm.Beta("political_stress_strength", alpha=5.0, beta=2.0)    # 0.71
        incest = pm.Beta("incest_bottleneck_strength", alpha=3.0, beta=4.0)   # 0.43

        # Proxy observations:
        # 1. "Han archaeological lead content in imperial vessels is
        #    substantial" — moderate support for lead.
        pm.Beta("obs_lead_vessels",
                alpha=1.0 + 5.0 * lead,
                beta=1.0 + 5.0 * (1.0 - lead),
                observed=0.55)
        # 2. "Half of Eastern Han emperors died during political purges" —
        #    strong support for political_stress.
        pm.Beta("obs_purge_deaths",
                alpha=1.0 + 5.0 * stress,
                beta=1.0 + 5.0 * (1.0 - stress),
                observed=0.8)
        # 3. "Court genealogies show repeated cousin-level intra-clan
        #    marriage" — moderate support for incest bottleneck.
        pm.Beta("obs_genealogy",
                alpha=1.0 + 5.0 * incest,
                beta=1.0 + 5.0 * (1.0 - incest),
                observed=0.50)

        # P(emperor young-at-death) increases with each cause.
        logit = (1.6 * lead + 1.9 * stress + 1.5 * incest - 2.5)
        p_young_death = pm.Deterministic("p_young_death", pm.math.sigmoid(logit))

        # Observe: Eastern Han emperors WERE short-lived on average.
        pm.Bernoulli("young_death_historical", p=p_young_death, observed=1)

    return model


# =============================================================================
# Claim models
# =============================================================================
def claim_lead_poisoning():
    model = build_base_model()
    with model:
        pm.Normal("claim_lead_obs",
                  mu=model["lead_poisoning_strength"] - 0.9,
                  sigma=0.05, observed=0.0)
    return model


def claim_political_stress():
    model = build_base_model()
    with model:
        pm.Normal("claim_stress_obs",
                  mu=model["political_stress_strength"] - 0.9,
                  sigma=0.05, observed=0.0)
    return model


def claim_incest_bottleneck():
    model = build_base_model()
    with model:
        pm.Normal("claim_incest_obs",
                  mu=model["incest_bottleneck_strength"] - 0.9,
                  sigma=0.05, observed=0.0)
    return model


def claim_orphan_geomancy():
    """Orphan claim: palace fengshui. Adds 2 disconnected latents (palace
    orientation + unfortunate-direction frequency) that do NOT enter the
    p_young_death logit. Observing extreme values for them contributes
    no information about emperor lifespan — KL will be ~0 — but BIC
    penalises the data-free parameters."""
    model = build_base_model()
    with model:
        palace_orient = pm.Beta("palace_orientation", alpha=2.0, beta=2.0)
        unfortunate_days = pm.Beta("unfortunate_day_count", alpha=2.0, beta=2.0)
        pm.Normal("claim_orient_obs",
                  mu=palace_orient - 0.9, sigma=0.05, observed=0.0)
        pm.Normal("claim_days_obs",
                  mu=unfortunate_days - 0.9, sigma=0.05, observed=0.0)
    return model


def get_claims():
    from phase3_embedding.claim_ranking_engine import ClaimSpec
    return [
        ClaimSpec(
            name="H_lead_poisoning",
            summary="Lead poisoning from cosmetics/wine vessels (in-network)",
            build_model=claim_lead_poisoning,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_political_stress",
            summary="Factional political violence shortened lives (in-network)",
            build_model=claim_political_stress,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_incest_bottleneck",
            summary="Repeated intra-clan marriage → genetic bottleneck (in-network)",
            build_model=claim_incest_bottleneck,
            structural_additions=0,
        ),
        ClaimSpec(
            name="H_orphan_geomancy",
            summary="Palace fengshui was bad (2 orphan latents, no causal link)",
            build_model=claim_orphan_geomancy,
            structural_additions=2,
        ),
    ]


if __name__ == "__main__":
    from phase3_embedding.claim_ranking_engine import (
        ClaimRankingEngine, pretty_print_engine_scores,
    )
    print("Eastern Han early-death benchmark — D(h, B) ranking\n")
    engine = ClaimRankingEngine(
        build_base_model=build_base_model,
        latent_var="p_young_death",
        bic_n=10,
        draws=1200, tune=800, random_seed=0,
    )
    scores = engine.rank(get_claims())
    pretty_print_engine_scores(scores)
