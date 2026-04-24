"""Validation for Phase 4.1 law-zoo v1.

Acceptance criteria:
    1. All 7 registered domains expose the canonical API:
         {DOMAIN_ID, CLASS_ID, LATENT_ROLES, build_model, simulate, true_params}.
    2. simulate() produces monotone behaviour consistent with the class:
         - exp_decay: y decreases monotonically (up to noise)
         - saturation: y increases monotonically and asymptotes near ymax
    3. MCMC recovers true parameters within tolerance on a representative
       domain per class (1 domain each, fast smoke).

Run:
    python -m phase4_structure.tests.test_law_zoo
"""

from __future__ import annotations

import numpy as np
import pymc as pm

from phase4_structure.law_zoo import DOMAINS, CLASS_OF
from phase4_structure.law_zoo import exp_decay, saturation


REQUIRED_ATTRS = {"DOMAIN_ID", "CLASS_ID", "LATENT_ROLES",
                  "build_model", "simulate", "true_params"}


def test_registry_has_expected_domains():
    expected = {
        "rc_circuit", "radioactive_decay", "first_order_reaction", "forgetting_curve",
        "capacitor_charging", "monomolecular_growth", "light_adaptation",
    }
    assert set(DOMAINS.keys()) == expected, DOMAINS.keys()
    assert set(CLASS_OF.values()) == {"exp_decay", "saturation"}
    print(f"[PASS] 7 domains across 2 classes registered: "
          f"{sum(v == 'exp_decay' for v in CLASS_OF.values())} exp_decay + "
          f"{sum(v == 'saturation' for v in CLASS_OF.values())} saturation")


def test_each_domain_has_canonical_api():
    for name, mod in DOMAINS.items():
        missing = REQUIRED_ATTRS - set(dir(mod))
        assert not missing, f"{name} missing attrs: {missing}"
    print(f"[PASS] all 7 domains expose canonical API")


def test_exp_decay_simulates_monotone_decrease():
    rng = np.random.default_rng(0)
    for name in exp_decay.DOMAIN_REGISTRY:
        t = exp_decay.default_t_grid(name, n=30)
        y = exp_decay.DOMAIN_REGISTRY[name].simulate(t, rng)
        # Compare first vs last third of the trajectory (average over noise)
        assert y[:10].mean() > y[-10:].mean(), \
            f"{name}: not decreasing ({y[:10].mean():.3f} → {y[-10:].mean():.3f})"
        # y should remain positive (exp-decay goes to 0 asymptotically)
        assert y.min() > -3 * exp_decay.DOMAIN_REGISTRY[name].spec.true_sigma, \
            f"{name}: unphysically negative"
    print(f"[PASS] 4 exp_decay domains all monotone decreasing")


def test_saturation_simulates_monotone_increase_and_plateau():
    rng = np.random.default_rng(0)
    for name in saturation.DOMAIN_REGISTRY:
        t = saturation.default_t_grid(name, n=30)
        y = saturation.DOMAIN_REGISTRY[name].simulate(t, rng)
        assert y[:10].mean() < y[-10:].mean(), \
            f"{name}: not increasing ({y[:10].mean():.3f} → {y[-10:].mean():.3f})"
        spec = saturation.DOMAIN_REGISTRY[name].spec
        # Late plateau should be within a few sigma of ymax
        plateau = y[-10:].mean()
        assert abs(plateau - spec.true_ymax) < 5 * spec.true_sigma + 0.15 * spec.true_ymax, \
            f"{name}: plateau {plateau:.3f} far from ymax {spec.true_ymax:.3f}"
    print(f"[PASS] 3 saturation domains all monotone increasing toward ymax")


def test_mcmc_recovers_exp_decay_params():
    """Smoke test: fit first_order_reaction, check posterior means near truth."""
    rng = np.random.default_rng(0)
    mod = exp_decay.DOMAIN_REGISTRY["first_order_reaction"]
    t = exp_decay.default_t_grid("first_order_reaction", n=20)
    y = mod.simulate(t, rng)

    with mod.build_model(t, y):
        idata = pm.sample(draws=800, tune=500, chains=2, random_seed=0,
                          progressbar=False, compute_convergence_checks=False)

    post = idata.posterior
    y0_mean = float(post["y0"].mean())
    k_mean = float(post["k"].mean())
    truth = mod.true_params()
    assert abs(y0_mean - truth["y0"]) / truth["y0"] < 0.10, f"y0 off: {y0_mean} vs {truth['y0']}"
    assert abs(k_mean - truth["k"]) / truth["k"] < 0.15, f"k off: {k_mean} vs {truth['k']}"
    print(f"[PASS] exp_decay MCMC recovers first_order_reaction: "
          f"y0 {y0_mean:.3f} (truth {truth['y0']:.3f}), "
          f"k {k_mean:.4f} (truth {truth['k']:.4f})")


def test_mcmc_recovers_saturation_params():
    """Smoke test: fit capacitor_charging, check posterior means near truth."""
    rng = np.random.default_rng(0)
    mod = saturation.DOMAIN_REGISTRY["capacitor_charging"]
    t = saturation.default_t_grid("capacitor_charging", n=20)
    y = mod.simulate(t, rng)

    with mod.build_model(t, y):
        idata = pm.sample(draws=800, tune=500, chains=2, random_seed=0,
                          progressbar=False, compute_convergence_checks=False)

    post = idata.posterior
    ymax_mean = float(post["ymax"].mean())
    k_mean = float(post["k"].mean())
    truth = mod.true_params()
    assert abs(ymax_mean - truth["ymax"]) / truth["ymax"] < 0.10, \
        f"ymax off: {ymax_mean} vs {truth['ymax']}"
    assert abs(k_mean - truth["k"]) / truth["k"] < 0.20, \
        f"k off: {k_mean} vs {truth['k']}"
    print(f"[PASS] saturation MCMC recovers capacitor_charging: "
          f"ymax {ymax_mean:.3f} (truth {truth['ymax']:.3f}), "
          f"k {k_mean:.4f} (truth {truth['k']:.4f})")


if __name__ == "__main__":
    print("=== P4.1 law-zoo validation ===\n")
    test_registry_has_expected_domains()
    test_each_domain_has_canonical_api()
    test_exp_decay_simulates_monotone_decrease()
    test_saturation_simulates_monotone_increase_and_plateau()
    test_mcmc_recovers_exp_decay_params()
    test_mcmc_recovers_saturation_params()
    print("\nAll P4.1 law-zoo checks passed.")
