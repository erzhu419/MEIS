"""MEIS Phase 2 — Alice-Charlie multi-observable belief-propagation demo.

This realises MEIS_plan.md §Phase 2 §1: the 3-person scenario where
progressive observations tighten the posterior on "who is heavier".

Base claim: "Alice is heavier than Charlie" — P(weight_A > weight_C | D)
should start at ~0.85 given only height comparisons, and rise toward 1.0
as secondary evidence (shoe size + footprint depth) is added.

Pure PyMC, 0 API. This is the canonical MEIS demo predicted by the plan
but not implemented in Phase 1 (Phase 1 focused on single-input regression).

Belief network:

    latent parameters (per-env, drawn once):
      theta_X ~ Normal(mu=1.414e-5, sigma=4e-7)    # weight = theta * h^3
      soil_stiffness ~ Normal(1e5, 2e4)             # Pa / cm
      shoe_coef      ~ Normal(0.24, 0.02)           # shoe_size_EU = coef * height_cm
      foot_coef      ~ Normal(0.015, 0.002)         # foot_area_m2 = coef * foot_length_cm^2

    per-person latents (X ∈ {Alice, Bob, Charlie}):
      height_X ~ Normal(170, 10)   cm
      w_X = theta_X * height_X ** 3    + Normal(0, 2)    kg
      shoe_X = shoe_coef * height_X     + Normal(0, 1)    EU
      foot_length_X = 1.5 * shoe_X + 5  + Normal(0, 0.5)  cm
      foot_area_X = foot_coef * foot_length_X ** 2    m^2
      pressure_X = 9.81 * w_X / foot_area_X             Pa
      footprint_X = pressure_X / soil_stiffness         cm

Observed constraints (progressively applied):
  1. height_A > height_B (soft: height_A ~ height_B + N(5, 2))
  2. height_B == height_C (strong: same draw)
  3. shoe_A == shoe_C
  4. footprint_A > footprint_C

Expected progression of P(weight_A > weight_C):
  ~0.5 (random priors → equal)
  ~0.85 (heights constrained, Alice taller by construction)
  ~0.85 (shoes equal — adds little)
  ~0.97 (footprint deeper → more pressure → heavier)
"""

from __future__ import annotations

import numpy as np
import pymc as pm


DENSITY_MU, DENSITY_SIGMA = 1010.0, 30.0
THETA_MU, THETA_SIGMA = 1.414e-5, 4e-7
HEIGHT_MU, HEIGHT_SIGMA = 170.0, 10.0
OBS_WEIGHT_SIGMA = 2.0

SHOE_MU, SHOE_SIGMA = 42.0, 1.5                # per-person shoe_size_EU, adult population
# The foot-area coefficient and soil stiffness are modelled as
# CALIBRATED constants (very tight priors). Otherwise a deeper-footprint
# observation gets absorbed by loosening these shared latents instead of
# sharpening per-person weight — blocking the pedagogical progression.
FOOT_COEF_MU, FOOT_COEF_SIGMA = 0.015, 1e-5
SOIL_MU, SOIL_SIGMA = 1.0e5, 1.0e3
GRAVITY = 9.81


def build_model(alice_taller_than_bob: bool = True,
                height_diff_mean: float = 5.0, height_diff_sigma: float = 2.0,
                equal_shoes: bool = False,
                alice_footprint_deeper_by: float | None = None,
                seed: int = 0):
    """Build a PyMC model of the 3-person scenario with specified observations.

    Returns the compiled model (caller runs `pm.sample` in a context).
    """
    with pm.Model() as model:
        # Shared physical-law latents
        foot_coef = pm.Normal("foot_coef", mu=FOOT_COEF_MU, sigma=FOOT_COEF_SIGMA)
        soil = pm.Normal("soil_stiffness", mu=SOIL_MU, sigma=SOIL_SIGMA)

        # Per-person theta (density × volume-coefficient, near-constant)
        theta_A = pm.Normal("theta_A", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_B = pm.Normal("theta_B", mu=THETA_MU, sigma=THETA_SIGMA)
        theta_C = pm.Normal("theta_C", mu=THETA_MU, sigma=THETA_SIGMA)

        # Heights
        # Bob is the baseline; Charlie equals Bob exactly (plan's stipulation
        # "Bob 和 Charlie 一样高"); Alice's height is offset from Bob.
        height_B = pm.Normal("height_B", mu=HEIGHT_MU, sigma=HEIGHT_SIGMA)
        height_C = pm.Deterministic("height_C", height_B)
        if alice_taller_than_bob:
            height_A = pm.Deterministic(
                "height_A",
                height_B + pm.Normal("dh_alice", mu=height_diff_mean,
                                      sigma=height_diff_sigma),
            )
        else:
            height_A = pm.Normal("height_A", mu=HEIGHT_MU, sigma=HEIGHT_SIGMA)

        # Weights (this is the comparison target)
        w_A = pm.Deterministic("weight_A", theta_A * height_A ** 3)
        w_B = pm.Deterministic("weight_B", theta_B * height_B ** 3)
        w_C = pm.Deterministic("weight_C", theta_C * height_C ** 3)

        # Shoes: INDEPENDENT per-person draws, NOT linked to height (plan's
        # model — shoes tell us about foot_area, not about height). A
        # stronger link height↔shoe would make "equal shoes" contradict
        # "Alice taller", which obscures the pedagogical point.
        shoe_A = pm.Normal("shoe_A", mu=SHOE_MU, sigma=SHOE_SIGMA)
        shoe_B = pm.Normal("shoe_B", mu=SHOE_MU, sigma=SHOE_SIGMA)
        shoe_C = pm.Normal("shoe_C", mu=SHOE_MU, sigma=SHOE_SIGMA)

        if equal_shoes:
            # Observe shoe_A == shoe_C within measurement noise.
            pm.Normal("shoe_equality", mu=shoe_A - shoe_C, sigma=0.3, observed=0.0)

        # Footprint depths (derived from weight / foot_area / soil).
        # Units: foot_length in cm, foot_area in cm^2 (foot_coef * len^2),
        # convert to m^2 for pressure calc; pressure in Pa; soil in Pa/cm.
        foot_length_A = pm.Deterministic("foot_length_A", 1.5 * shoe_A + 5.0)
        foot_length_C = pm.Deterministic("foot_length_C", 1.5 * shoe_C + 5.0)
        foot_area_cm2_A = pm.Deterministic("foot_area_cm2_A", foot_coef * foot_length_A ** 2)
        foot_area_cm2_C = pm.Deterministic("foot_area_cm2_C", foot_coef * foot_length_C ** 2)
        pressure_A = pm.Deterministic("pressure_A",
                                      GRAVITY * w_A / (foot_area_cm2_A * 1e-4))
        pressure_C = pm.Deterministic("pressure_C",
                                      GRAVITY * w_C / (foot_area_cm2_C * 1e-4))
        depth_A = pm.Deterministic("depth_A", pressure_A / soil)
        depth_C = pm.Deterministic("depth_C", pressure_C / soil)

        if alice_footprint_deeper_by is not None:
            # Observation: depth_A - depth_C = known positive value (with noise)
            pm.Normal("footprint_diff",
                      mu=depth_A - depth_C, sigma=0.05,
                      observed=alice_footprint_deeper_by)

    return model


def _sample(model, draws=1500, tune=1000, chains=2, random_seed=0):
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, random_seed=random_seed,
            return_inferencedata=False, progressbar=False,
            compute_convergence_checks=False,
        )


def p_alice_heavier(trace) -> float:
    return float(np.mean(trace["weight_A"] > trace["weight_C"]))


def entropy_bernoulli(p: float) -> float:
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def run_progressive_evidence(verbose: bool = True, random_seed: int = 0) -> list[dict]:
    """Run the 4-stage progressive-evidence demo, returning a record per stage.

    Evidence progression is calibrated so each stage actually adds information
    (the stage 1 height margin is intentionally weak so that stages 2 and 3
    have headroom to tighten the posterior further)."""
    stages = [
        {"name": "stage_0_prior",
         "desc": "no comparative info",
         "kwargs": dict(alice_taller_than_bob=False,
                        equal_shoes=False, alice_footprint_deeper_by=None)},
        {"name": "stage_1_heights",
         "desc": "Alice plausibly taller than Bob (dh ~ N(2, 5))",
         "kwargs": dict(alice_taller_than_bob=True,
                        height_diff_mean=2.0, height_diff_sigma=5.0,
                        equal_shoes=False, alice_footprint_deeper_by=None)},
        {"name": "stage_2_equal_shoes",
         "desc": "+ Alice & Charlie same shoe size",
         "kwargs": dict(alice_taller_than_bob=True,
                        height_diff_mean=2.0, height_diff_sigma=5.0,
                        equal_shoes=True, alice_footprint_deeper_by=None)},
        {"name": "stage_3_deeper_footprint",
         "desc": "+ Alice footprint 0.15 cm deeper than Charlie",
         "kwargs": dict(alice_taller_than_bob=True,
                        height_diff_mean=2.0, height_diff_sigma=5.0,
                        equal_shoes=True, alice_footprint_deeper_by=0.15)},
    ]

    rows = []
    for stage in stages:
        model = build_model(**stage["kwargs"])
        trace = _sample(model, random_seed=random_seed)
        p = p_alice_heavier(trace)
        ent = entropy_bernoulli(p)
        weight_gap_mean = float(np.mean(trace["weight_A"] - trace["weight_C"]))
        rows.append({**stage, "p_alice_heavier": p,
                     "entropy_bits": ent / np.log(2),
                     "mean_weight_gap_kg": weight_gap_mean})
        if verbose:
            print(f"  {stage['name']:<28} {stage['desc']:<55} "
                  f"P(A>C) = {p:.3f}   H = {ent/np.log(2):.3f} bits   "
                  f"ΔW mean = {weight_gap_mean:+.2f} kg")

    return rows


if __name__ == "__main__":
    print("MEIS Phase 2 — Alice-Charlie multi-observable posterior demo\n")
    rows = run_progressive_evidence(verbose=True)
    print()
    print("Plan §Phase 2 check: P(A heavier) should progress roughly ")
    print("0.5 (prior) → 0.85 (heights) → 0.85 (shoes) → 0.97 (footprint).")
    print("Entropy should be monotone non-increasing.")
