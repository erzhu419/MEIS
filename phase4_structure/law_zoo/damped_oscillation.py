"""Damped oscillation equivalence class (Plan §Phase 5 定律动物园).

Shared ODE:  ÿ + 2γ ẏ + ω₀² y = 0,  underdamped (γ < ω₀)
           y(t) = A · exp(-γ·t) · cos(ω·t + φ),  ω = sqrt(ω₀² - γ²)

Three physical domains, identical structural skeleton:

  Domain        A role                  γ role              ω role
  ───────────  ──────────────────────  ──────────────────  ─────────────────
  rlc_circuit  initial charge/current  R/(2L)              LC resonance
  pendulum     initial angle θ₀         air friction coef   g/L
  mass_spring  initial displacement    damping coef c/2m   k_spring/m

Each domain exposes the canonical law-zoo API (DOMAIN_ID, CLASS_ID,
LATENT_ROLES, build_model, simulate, true_params) plus one extra
kwarg in build_model: prior_overrides={'A_sigma', 'gamma_sigma',
'omega_sigma', 'phi_sigma'} for P4.4-style transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pymc as pm


CLASS_ID = "damped_oscillation"

LATENT_ROLES = {
    "A": "scale",
    "gamma": "rate",
    "omega": "frequency",
    "phi": "phase",
    "sigma": "noise",
    "y_obs": "obs",
}


@dataclass
class DomainSpec:
    name: str
    A_prior: tuple
    gamma_prior: tuple
    omega_prior: tuple
    phi_prior: tuple
    sigma_prior: float
    true_A: float
    true_gamma: float
    true_omega: float
    true_phi: float
    true_sigma: float
    t_range: tuple


_SPECS = {
    "rlc_circuit": DomainSpec(
        name="rlc_circuit",
        A_prior=(np.log(5.0), 0.3),
        gamma_prior=(np.log(0.25), 0.4),
        omega_prior=(np.log(2.0), 0.3),
        phi_prior=(0.0, 0.5),
        sigma_prior=0.2,
        true_A=5.0, true_gamma=0.25, true_omega=2.0, true_phi=0.0,
        true_sigma=0.1,
        t_range=(0.0, 12.0),
    ),
    "pendulum": DomainSpec(
        name="pendulum",
        A_prior=(np.log(0.3), 0.3),              # ~0.3 rad initial angle
        gamma_prior=(np.log(0.15), 0.4),
        omega_prior=(np.log(3.0), 0.3),          # ω₀ = sqrt(g/L), L=1.09 m
        phi_prior=(0.0, 0.5),
        sigma_prior=0.05,
        true_A=0.3, true_gamma=0.15, true_omega=3.0, true_phi=0.0,
        true_sigma=0.02,
        t_range=(0.0, 8.0),
    ),
    "mass_spring": DomainSpec(
        name="mass_spring",
        A_prior=(np.log(1.0), 0.3),              # ~1 m initial displacement
        gamma_prior=(np.log(0.4), 0.4),
        omega_prior=(np.log(1.5), 0.3),
        phi_prior=(0.0, 0.5),
        sigma_prior=0.1,
        true_A=1.0, true_gamma=0.4, true_omega=1.5, true_phi=0.0,
        true_sigma=0.05,
        t_range=(0.0, 10.0),
    ),
}


def _damped(t: np.ndarray, A: float, gamma: float, omega: float, phi: float) -> np.ndarray:
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)


def _make_domain_module(spec: DomainSpec):
    def simulate(t: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        mu = _damped(t, spec.true_A, spec.true_gamma, spec.true_omega, spec.true_phi)
        return mu + rng.normal(0.0, spec.true_sigma, size=mu.shape)

    def true_params() -> dict:
        return dict(A=spec.true_A, gamma=spec.true_gamma, omega=spec.true_omega,
                    phi=spec.true_phi, sigma=spec.true_sigma)

    def build_model(t_obs: np.ndarray, y_obs: np.ndarray,
                    prior_overrides: dict | None = None) -> pm.Model:
        po = prior_overrides or {}
        A_sigma = po.get("A_sigma", spec.A_prior[1])
        gamma_sigma = po.get("gamma_sigma", spec.gamma_prior[1])
        omega_sigma = po.get("omega_sigma", spec.omega_prior[1])
        phi_sigma = po.get("phi_sigma", spec.phi_prior[1])
        with pm.Model() as model:
            A = pm.LogNormal("A", mu=spec.A_prior[0], sigma=A_sigma)
            gamma = pm.LogNormal("gamma", mu=spec.gamma_prior[0], sigma=gamma_sigma)
            omega = pm.LogNormal("omega", mu=spec.omega_prior[0], sigma=omega_sigma)
            phi = pm.Normal("phi", mu=spec.phi_prior[0], sigma=phi_sigma)
            sigma = pm.HalfNormal("sigma", sigma=spec.sigma_prior)
            mu = A * pm.math.exp(-gamma * t_obs) * pm.math.cos(omega * t_obs + phi)
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


def default_t_grid(domain_name: str, n: int = 30) -> np.ndarray:
    spec = _SPECS[domain_name]
    return np.linspace(spec.t_range[0], spec.t_range[1], n)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in DOMAIN_REGISTRY:
        t = default_t_grid(name, n=20)
        y = DOMAIN_REGISTRY[name].simulate(t, rng)
        print(f"{name:>14}  t=[{t[0]:.1f}, {t[-1]:.1f}]  "
              f"y=[{y[0]:+.3f}, ..., {y[-1]:+.3f}]  "
              f"|y|max={np.abs(y).max():.3f}")
