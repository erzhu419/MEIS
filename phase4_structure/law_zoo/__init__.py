"""Law-zoo v2 — MEIS Phase 4.1 + 4.v2 fixtures.

Three equivalence classes, each instantiated as several physical
domains sharing the same ODE / functional form:

  exp_decay            y(t) = y0·exp(-k·t)
      rc_circuit, radioactive_decay, first_order_reaction, forgetting_curve
  saturation           y(t) = ymax·(1 - exp(-k·t))
      capacitor_charging, monomolecular_growth, light_adaptation
  damped_oscillation   y(t) = A·exp(-γ·t)·cos(ω·t + φ)
      rlc_circuit, pendulum, mass_spring

Each domain exposes:
    build_model(t_obs, y_obs, prior_overrides=None) -> pm.Model
    true_params()            -> dict of truth values
    simulate(t, rng=None)    -> y array
    DOMAIN_ID                -> str
    CLASS_ID                 -> one of the three above
    LATENT_ROLES            -> dict node_name -> role label
"""

from . import exp_decay, saturation, damped_oscillation

__all__ = ["exp_decay", "saturation", "damped_oscillation", "DOMAINS", "CLASS_OF"]

DOMAINS = {
    **exp_decay.DOMAIN_REGISTRY,
    **saturation.DOMAIN_REGISTRY,
    **damped_oscillation.DOMAIN_REGISTRY,
}

CLASS_OF = {name: mod.CLASS_ID for name, mod in DOMAINS.items()}
