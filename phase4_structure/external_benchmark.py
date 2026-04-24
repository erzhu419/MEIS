"""External benchmark: signature fingerprint on third-party PyMC models.

Addresses the "construction-sensitive evaluation" limitation. Instead
of the author-constructed law-zoo (where each class shares a canonical
class_mu by design), we build 8 PyMC models drawn from textbook
examples the authors did not design into equivalence classes.

Models (all standard PyMC gallery / textbook forms):
  1. Linear regression with Gaussian noise
  2. Logistic regression (Bernoulli outcome)
  3. Poisson regression with log-link
  4. Exponential decay (y0 · exp(-k·t))
  5. Saturating growth (ymax · (1 - exp(-k·t)))
  6. Gamma rate-parameter regression (light-tailed positive counts)
  7. Power law (y = a · x^b)
  8. Sinusoid with decay (A · exp(-γt) · sin(ωt))

Expected partition under ODE-family equivalence (if the signature
layer is doing its job):
  {4}  exp_decay
  {5}  saturation
  {8}  damped oscillation (sinusoid-with-decay, different op
       multiset from our law-zoo's cos form — genuine distractor)
  {1}  linear regression (no time structure)
  {2}  Bernoulli (different likelihood family)
  {3, 6}  count-family regressions (both have Poisson-like count
          structure; Gamma rate may hash to a different signature
          because distribution family differs)
  {7}  power law (pow op)

Hypothesis we test: the op-multiset signature should produce
distinct fingerprints for models with distinct op-families; the WL
signature should maintain these distinctions and additionally
separate same-ops-different-wiring pairs; BSS should agree
where class_mu matches.

This is a genuine external test: nothing about the signature
pipeline was tuned against these models.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pymc as pm
from dataclasses import dataclass

from phase4_structure.signature import extract_signature, signature_for_domain
from phase4_structure.wl_signature import extract_wl_signature


LATENT_ROLES_GENERIC = {
    # include the common names our models use; unknown names are
    # ignored by extract_signature
    "intercept": "scale",
    "slope": "rate",
    "y0": "scale",
    "ymax": "scale",
    "k": "rate",
    "A": "scale",
    "gamma": "rate",
    "omega": "frequency",
    "a": "scale",
    "b": "rate",
    "lam": "rate",
    "p": "probability",
    "mu_theta": "scale",
    "sigma": "noise",
    "y_obs": "obs",
    "count_obs": "obs",
    "bin_obs": "obs",
}


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def m1_linear_regression():
    rng = _rng(0)
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.shape)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0.0, 5.0)
        slope = pm.Normal("slope", 0.0, 5.0)
        sigma = pm.HalfNormal("sigma", 2.0)
        mu = intercept + slope * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model


def m2_logistic_regression():
    rng = _rng(0)
    x = np.linspace(-3, 3, 20)
    p_true = 1.0 / (1.0 + np.exp(-(0.5 + 1.2 * x)))
    y = rng.binomial(1, p_true)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0.0, 5.0)
        slope = pm.Normal("slope", 0.0, 5.0)
        p = pm.Deterministic("p", pm.math.sigmoid(intercept + slope * x))
        pm.Bernoulli("bin_obs", p=p, observed=y)
    return model


def m3_poisson_regression():
    rng = _rng(0)
    x = np.linspace(0, 2, 20)
    lam_true = np.exp(0.5 + 1.0 * x)
    y = rng.poisson(lam_true)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0.0, 3.0)
        slope = pm.Normal("slope", 0.0, 3.0)
        lam = pm.Deterministic("lam", pm.math.exp(intercept + slope * x))
        pm.Poisson("count_obs", mu=lam, observed=y)
    return model


def m4_exp_decay():
    rng = _rng(0)
    t = np.linspace(0, 10, 20)
    y = 5.0 * np.exp(-0.3 * t) + rng.normal(0, 0.1, t.shape)
    with pm.Model() as model:
        y0 = pm.LogNormal("y0", np.log(5.0), 0.4)
        k = pm.LogNormal("k", np.log(0.3), 0.4)
        sigma = pm.HalfNormal("sigma", 0.2)
        mu = y0 * pm.math.exp(-k * t)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model


def m5_saturating_growth():
    rng = _rng(0)
    t = np.linspace(0, 10, 20)
    y = 5.0 * (1.0 - np.exp(-0.5 * t)) + rng.normal(0, 0.1, t.shape)
    with pm.Model() as model:
        ymax = pm.LogNormal("ymax", np.log(5.0), 0.3)
        k = pm.LogNormal("k", np.log(0.5), 0.4)
        sigma = pm.HalfNormal("sigma", 0.2)
        mu = ymax * (1.0 - pm.math.exp(-k * t))
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model


def m6_gamma_rate_regression():
    rng = _rng(0)
    x = np.linspace(0, 2, 20)
    rate_true = np.exp(1.0 + 0.5 * x)
    y = rng.gamma(shape=2.0, scale=1.0 / rate_true)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0.0, 3.0)
        slope = pm.Normal("slope", 0.0, 3.0)
        rate = pm.Deterministic("rate", pm.math.exp(intercept + slope * x))
        pm.Gamma("count_obs", alpha=2.0, beta=rate, observed=y)
    return model


def m7_power_law():
    rng = _rng(0)
    x = np.linspace(1, 10, 20)
    y = 2.0 * x ** 1.5 + rng.normal(0, 1.0, x.shape)
    with pm.Model() as model:
        a = pm.LogNormal("a", np.log(2.0), 0.4)
        b = pm.LogNormal("b", np.log(1.5), 0.3)
        sigma = pm.HalfNormal("sigma", 2.0)
        mu = a * x ** b
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model


def m8_sinusoid_with_decay():
    rng = _rng(0)
    t = np.linspace(0, 10, 20)
    y = 3.0 * np.exp(-0.2 * t) * np.sin(2.0 * t) + rng.normal(0, 0.1, t.shape)
    with pm.Model() as model:
        A = pm.LogNormal("A", np.log(3.0), 0.3)
        gamma = pm.LogNormal("gamma", np.log(0.2), 0.4)
        omega = pm.LogNormal("omega", np.log(2.0), 0.3)
        sigma = pm.HalfNormal("sigma", 0.2)
        mu = A * pm.math.exp(-gamma * t) * pm.math.sin(omega * t)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model


MODELS = {
    "linear_regression":       m1_linear_regression,
    "logistic_regression":     m2_logistic_regression,
    "poisson_regression":      m3_poisson_regression,
    "exp_decay_external":      m4_exp_decay,
    "saturating_growth":       m5_saturating_growth,
    "gamma_rate_regression":   m6_gamma_rate_regression,
    "power_law":               m7_power_law,
    "sinusoid_decay":          m8_sinusoid_with_decay,
}


@dataclass
class SignatureSummary:
    name: str
    op_multiset_fp: str
    wl_fp: str
    n_ops: int
    n_nodes_wl: int


def run_external(models=None):
    if models is None:
        models = MODELS
    summaries = []
    for name, builder in models.items():
        model = builder()
        sig = extract_signature(model, LATENT_ROLES_GENERIC)
        wl = extract_wl_signature(model, num_iterations=3)
        summaries.append(SignatureSummary(
            name=name,
            op_multiset_fp=sig.fingerprint,
            wl_fp=wl.fingerprint,
            n_ops=len(sig.ops),
            n_nodes_wl=wl.num_nodes,
        ))
    return summaries


def partition_by_fingerprint(summaries, key="op_multiset_fp"):
    groups = {}
    for s in summaries:
        fp = getattr(s, key)
        groups.setdefault(fp, []).append(s.name)
    return groups


if __name__ == "__main__":
    print("External benchmark: signature fingerprints on 8 third-party models\n")
    summaries = run_external()
    print(f"{'model':<24}  {'op-multiset fp':<18}  {'WL fp':<18}  {'nOps':>5}  {'nNodes':>7}")
    for s in summaries:
        print(f"{s.name:<24}  {s.op_multiset_fp}  {s.wl_fp}  "
              f"{s.n_ops:>5}  {s.n_nodes_wl:>7}")

    print("\nOp-multiset partition (same fp = same class):")
    gom = partition_by_fingerprint(summaries, "op_multiset_fp")
    for fp, members in gom.items():
        print(f"  {fp}: {members}")

    print("\nWL partition:")
    gwl = partition_by_fingerprint(summaries, "wl_fp")
    for fp, members in gwl.items():
        print(f"  {fp}: {members}")

    print(f"\nOp-multiset: {len(gom)} classes among {len(summaries)} models")
    print(f"WL:          {len(gwl)} classes among {len(summaries)} models")
