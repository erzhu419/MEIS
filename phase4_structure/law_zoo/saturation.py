"""Saturation growth equivalence class (Plan §Phase 5 定律动物园).

Shared ODE: dy/dt = k·(ymax - y),  y(0) = 0  →  y(t) = ymax·(1 - exp(-k·t))

Three physical domains, identical structural skeleton:

  Domain                  ymax role              k role
  ──────────────────────  ─────────────────────  ──────────────────────────
  capacitor_charging      supply voltage V_s     1/(R·C) charging rate
  monomolecular_growth    carrying capacity K    intrinsic rate
  light_adaptation        steady-state response  adaptation rate
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pymc as pm


CLASS_ID = "saturation"

LATENT_ROLES = {
    "ymax": "scale",
    "k": "rate",
    "sigma": "noise",
    "y_obs": "obs",
}


@dataclass
class DomainSpec:
    name: str
    ymax_prior: tuple
    k_prior: tuple
    sigma_prior: float
    true_ymax: float
    true_k: float
    true_sigma: float
    t_range: tuple


_SPECS = {
    "capacitor_charging": DomainSpec(
        name="capacitor_charging",
        ymax_prior=(np.log(5.0), 0.3),
        k_prior=(np.log(0.5), 0.4),
        sigma_prior=0.2,
        true_ymax=5.0, true_k=0.5, true_sigma=0.1,
        t_range=(0.0, 10.0),
    ),
    "monomolecular_growth": DomainSpec(
        name="monomolecular_growth",
        ymax_prior=(np.log(100.0), 0.4),
        k_prior=(np.log(0.15), 0.4),
        sigma_prior=2.0,
        true_ymax=100.0, true_k=0.15, true_sigma=1.5,
        t_range=(0.0, 40.0),
    ),
    "light_adaptation": DomainSpec(
        name="light_adaptation",
        ymax_prior=(np.log(1.0), 0.2),
        k_prior=(np.log(1.2), 0.4),
        sigma_prior=0.05,
        true_ymax=1.0, true_k=1.2, true_sigma=0.03,
        t_range=(0.0, 5.0),
    ),
}


def _saturate(t: np.ndarray, ymax: float, k: float) -> np.ndarray:
    return ymax * (1.0 - np.exp(-k * t))


def _make_domain_module(spec: DomainSpec):
    def simulate(t: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        mu = _saturate(t, spec.true_ymax, spec.true_k)
        return mu + rng.normal(0.0, spec.true_sigma, size=mu.shape)

    def true_params() -> dict:
        return dict(ymax=spec.true_ymax, k=spec.true_k, sigma=spec.true_sigma)

    def build_model(t_obs: np.ndarray, y_obs: np.ndarray,
                    prior_overrides: dict | None = None) -> pm.Model:
        po = prior_overrides or {}
        ymax_sigma = po.get("ymax_sigma", spec.ymax_prior[1])
        k_sigma = po.get("k_sigma", spec.k_prior[1])
        with pm.Model() as model:
            ymax = pm.LogNormal("ymax", mu=spec.ymax_prior[0], sigma=ymax_sigma)
            k = pm.LogNormal("k", mu=spec.k_prior[0], sigma=k_sigma)
            sigma = pm.HalfNormal("sigma", sigma=spec.sigma_prior)
            mu = ymax * (1.0 - pm.math.exp(-k * t_obs))
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
        return model

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
globals().update(DOMAIN_REGISTRY)


def default_t_grid(domain_name: str, n: int = 15) -> np.ndarray:
    spec = _SPECS[domain_name]
    return np.linspace(spec.t_range[0], spec.t_range[1], n)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in DOMAIN_REGISTRY:
        t = default_t_grid(name, n=10)
        y = DOMAIN_REGISTRY[name].simulate(t, rng)
        print(f"{name:>24}  t=[{t[0]:.1f}, {t[-1]:.1f}]  y=[{y[0]:.3f} → {y[-1]:.3f}]")
