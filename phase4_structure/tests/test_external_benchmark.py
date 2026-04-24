"""External benchmark validation — signature discovers equivalence to
law-zoo without being told.

This test is the honest answer to the "construction-sensitive evaluation"
reviewer critique: we build 8 fresh PyMC models (linear regression,
logistic regression, Poisson regression, exp-decay, saturating growth,
Gamma regression, power law, sin-decay) that are NOT members of the
author-constructed law-zoo, fingerprint them, and check:

  1. Fresh exp-decay model matches the law-zoo exp_decay class fingerprint
     exactly (discovered, not declared).
  2. Fresh saturating growth matches the law-zoo saturation class
     fingerprint exactly (discovered, not declared).
  3. Sin-decay (A·exp(-γt)·sin(ωt)) does NOT match the law-zoo
     damped_oscillation (A·exp(-γt)·cos(ωt+φ)) because cos ≠ sin in the
     op graph — correctly distinguishes structurally different dynamics.
  4. All 8 models produce distinct op-multiset fingerprints (no
     collisions), showing the signature is discriminative on
     independent PyMC models.
"""

from __future__ import annotations

import numpy as np

from phase4_structure.external_benchmark import MODELS, run_external
from phase4_structure.law_zoo import exp_decay as zoo_exp, saturation as zoo_sat, damped_oscillation as zoo_dmp
from phase4_structure.signature import signature_for_domain, extract_signature
from phase4_structure.external_benchmark import LATENT_ROLES_GENERIC


def _zoo_fp(mod, name):
    t = mod.default_t_grid(name, n=10)
    y = mod.DOMAIN_REGISTRY[name].simulate(t, np.random.default_rng(0))
    return signature_for_domain(mod.DOMAIN_REGISTRY[name], t, y).fingerprint


def test_external_exp_decay_matches_law_zoo_discovered():
    """m4 (fresh PyMC exp-decay with totally different priors) should
    hash to the same op-multiset fingerprint as any law-zoo exp_decay
    member. This is the key 'discovered, not declared' result."""
    summaries = run_external({
        "exp_decay_external": MODELS["exp_decay_external"],
    })
    external_fp = summaries[0].op_multiset_fp
    zoo_fp = _zoo_fp(zoo_exp, "rc_circuit")
    assert external_fp == zoo_fp, \
        f"external exp-decay fp {external_fp} != law-zoo fp {zoo_fp}"
    print(f"[PASS] external exp-decay fingerprint = law-zoo exp_decay fp "
          f"({external_fp}) — discovered, not declared")


def test_external_saturation_matches_law_zoo_discovered():
    """m5 (fresh saturating growth) should match law-zoo saturation."""
    summaries = run_external({"saturating_growth": MODELS["saturating_growth"]})
    external_fp = summaries[0].op_multiset_fp
    zoo_fp = _zoo_fp(zoo_sat, "capacitor_charging")
    assert external_fp == zoo_fp, \
        f"external saturation fp {external_fp} != law-zoo fp {zoo_fp}"
    print(f"[PASS] external saturating-growth fp = law-zoo saturation fp "
          f"({external_fp}) — discovered, not declared")


def test_sin_decay_does_not_match_cosine_damped_oscillation():
    """m8 uses sin(ωt) while law-zoo damped uses cos(ωt+φ). The op-
    graph distinguishes them: Sin vs Cos are different scalar ops.
    This is the correct discriminative behaviour."""
    summaries = run_external({"sinusoid_decay": MODELS["sinusoid_decay"]})
    external_fp = summaries[0].op_multiset_fp
    zoo_fp = _zoo_fp(zoo_dmp, "rlc_circuit")
    assert external_fp != zoo_fp, \
        f"sin-decay fp {external_fp} unexpectedly matches cos-damped {zoo_fp}"
    print(f"[PASS] sin-decay ({external_fp}) ≠ law-zoo cos-damped "
          f"({zoo_fp}) — functional distinction preserved")


def test_no_cross_model_fingerprint_collisions():
    """All 8 external models should have distinct op-multiset
    fingerprints (no accidental collisions among unrelated models)."""
    summaries = run_external()
    fps = [s.op_multiset_fp for s in summaries]
    assert len(set(fps)) == len(fps), \
        f"fingerprint collision among external models: {fps}"
    print(f"[PASS] {len(summaries)} external models produce "
          f"{len(set(fps))} distinct op-multiset fingerprints "
          f"(no collisions)")


def test_wl_also_agrees_on_discovered_matches():
    """WL signature should make the same calls as op-multiset on the
    same-family discovered matches."""
    summaries = run_external({
        "exp_decay_external": MODELS["exp_decay_external"],
        "saturating_growth": MODELS["saturating_growth"],
    })
    from phase4_structure.wl_signature import wl_signature_for_domain
    t = zoo_exp.default_t_grid("rc_circuit", n=10)
    y = zoo_exp.DOMAIN_REGISTRY["rc_circuit"].simulate(t, np.random.default_rng(0))
    zoo_ed_wl = wl_signature_for_domain(
        zoo_exp.DOMAIN_REGISTRY["rc_circuit"], t, y).fingerprint
    t = zoo_sat.default_t_grid("capacitor_charging", n=10)
    y = zoo_sat.DOMAIN_REGISTRY["capacitor_charging"].simulate(
        t, np.random.default_rng(0))
    zoo_sat_wl = wl_signature_for_domain(
        zoo_sat.DOMAIN_REGISTRY["capacitor_charging"], t, y).fingerprint
    assert summaries[0].wl_fp == zoo_ed_wl, \
        f"WL external ed {summaries[0].wl_fp} != zoo ed {zoo_ed_wl}"
    assert summaries[1].wl_fp == zoo_sat_wl, \
        f"WL external sat {summaries[1].wl_fp} != zoo sat {zoo_sat_wl}"
    print(f"[PASS] WL signature agrees with op-multiset on discovered matches")


if __name__ == "__main__":
    print("=== External benchmark validation ===\n")
    test_external_exp_decay_matches_law_zoo_discovered()
    test_external_saturation_matches_law_zoo_discovered()
    test_sin_decay_does_not_match_cosine_damped_oscillation()
    test_no_cross_model_fingerprint_collisions()
    test_wl_also_agrees_on_discovered_matches()
    print("\nAll external-benchmark checks passed.")
