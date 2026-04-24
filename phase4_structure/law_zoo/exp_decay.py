"""Exponential decay equivalence class (Plan §Phase 5 定律动物园).

Shared ODE: dy/dt = -k·y,  y(0) = y0  →  y(t) = y0 · exp(-k·t)

Four physical domains, each with its own semantics (and priors calibrated
accordingly) but identical structural skeleton:

  Domain              y0 role               k role
  ─────────────────  ────────────────────  ──────────────────────────
  rc_circuit         initial voltage V0    1/(R·C) time constant
  radioactive_decay  initial atom count    decay constant λ = ln2/T½
  first_order_rxn    initial concentration reaction rate k
  forgetting_curve   initial retention 1   forgetting rate 1/S (Ebbinghaus)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pymc as pm


CLASS_ID = "exp_decay"

# Shared node-role vocabulary used by P4.2 structural signature.
LATENT_ROLES = {
    "y0": "scale",
    "k": "rate",
    "sigma": "noise",
    "y_obs": "obs",
}


@dataclass
class DomainSpec:
    name: str
    y0_prior: tuple  # (mean, sigma) for log-Normal on y0
    k_prior: tuple   # (mean, sigma) for log-Normal on k
    sigma_prior: float
    true_y0: float
    true_k: float
    true_sigma: float
    t_range: tuple   # (t_min, t_max)


_SPECS = {
    "rc_circuit": DomainSpec(
        name="rc_circuit",
        y0_prior=(np.log(5.0), 0.4),   # ~5 V initial
        k_prior=(np.log(0.5), 0.4),    # τ = 1/k = 2 s
        sigma_prior=0.2,
        true_y0=5.0, true_k=0.5, true_sigma=0.1,
        t_range=(0.0, 10.0),
    ),
    "radioactive_decay": DomainSpec(
        name="radioactive_decay",
        y0_prior=(np.log(1000.0), 0.4),
        k_prior=(np.log(0.1), 0.4),     # half-life ~7 yr
        sigma_prior=5.0,
        true_y0=1000.0, true_k=0.1, true_sigma=8.0,
        t_range=(0.0, 50.0),
    ),
    "first_order_reaction": DomainSpec(
        name="first_order_reaction",
        y0_prior=(np.log(2.0), 0.4),    # 2 mol/L
        k_prior=(np.log(0.2), 0.4),
        sigma_prior=0.05,
        true_y0=2.0, true_k=0.2, true_sigma=0.03,
        t_range=(0.0, 25.0),
    ),
    "forgetting_curve": DomainSpec(
        name="forgetting_curve",
        y0_prior=(np.log(1.0), 0.2),    # full retention at t=0
        k_prior=(np.log(0.3), 0.4),     # 1/S where S~3 days
        sigma_prior=0.05,
        true_y0=1.0, true_k=0.3, true_sigma=0.04,
        t_range=(0.0, 14.0),
    ),
}


def _decay(t: np.ndarray, y0: float, k: float) -> np.ndarray:
    return y0 * np.exp(-k * t)


def _make_domain_module(spec: DomainSpec):
    """Factory — returns a namespace with build_model / simulate / true_params
    bound to this DomainSpec."""

    def simulate(t: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        mu = _decay(t, spec.true_y0, spec.true_k)
        return mu + rng.normal(0.0, spec.true_sigma, size=mu.shape)

    def true_params() -> dict:
        return dict(y0=spec.true_y0, k=spec.true_k, sigma=spec.true_sigma)

    def build_model(t_obs: np.ndarray, y_obs: np.ndarray) -> pm.Model:
        with pm.Model() as model:
            y0 = pm.LogNormal("y0", mu=spec.y0_prior[0], sigma=spec.y0_prior[1])
            k = pm.LogNormal("k", mu=spec.k_prior[0], sigma=spec.k_prior[1])
            sigma = pm.HalfNormal("sigma", sigma=spec.sigma_prior)
            mu = y0 * pm.math.exp(-k * t_obs)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
        return model

    # namespace object
    ns = type("Domain", (), dict(
        DOMAIN_ID=spec.name,
        CLASS_ID=CLASS_ID,
        LATENT_ROLES=LATENT_ROLES,
        spec=spec,
        simulate=staticmethod(simulate),
        true_params=staticmethod(true_params),
        build_model=staticmethod(build_model),
    ))
    return ns


DOMAIN_REGISTRY = {name: _make_domain_module(spec) for name, spec in _SPECS.items()}
globals().update(DOMAIN_REGISTRY)   # allow `exp_decay.rc_circuit` attribute access


def default_t_grid(domain_name: str, n: int = 15) -> np.ndarray:
    spec = _SPECS[domain_name]
    return np.linspace(spec.t_range[0], spec.t_range[1], n)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in DOMAIN_REGISTRY:
        t = default_t_grid(name, n=10)
        y = DOMAIN_REGISTRY[name].simulate(t, rng)
        print(f"{name:>22}  t=[{t[0]:.1f}, {t[-1]:.1f}]  y=[{y[0]:.3f} → {y[-1]:.3f}]")
