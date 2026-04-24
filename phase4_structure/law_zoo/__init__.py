"""Law-zoo v1 — MEIS Phase 4.1 fixtures.

Two equivalence classes, each instantiated as several physical domains
sharing the same underlying ODE / functional form. Cross-domain transfer
between domains *within* a class should succeed; across classes should fail.

  exp_decay     dy/dt = -k·y,  y(0)=y0                        → y(t) = y0·exp(-k·t)
      rc_circuit, radioactive_decay, first_order_reaction, forgetting_curve
  saturation    dy/dt = k·(ymax - y),  y(0)=0                 → y(t) = ymax·(1 - exp(-k·t))
      capacitor_charging, monomolecular_growth, light_adaptation

Each domain exposes:
    build_model(t_obs, y_obs) -> pm.Model
    true_params()            -> dict with {y0 or ymax, k, sigma}
    simulate(t, rng=None)    -> y array
    DOMAIN_ID                -> str
    CLASS_ID                 -> str ('exp_decay' | 'saturation')
    LATENT_ROLES            -> dict node_name -> role ('scale'|'rate'|'obs'|'noise')
"""

from . import exp_decay, saturation

__all__ = ["exp_decay", "saturation", "DOMAINS", "CLASS_OF"]

DOMAINS = {
    **exp_decay.DOMAIN_REGISTRY,
    **saturation.DOMAIN_REGISTRY,
}

CLASS_OF = {name: mod.CLASS_ID for name, mod in DOMAINS.items()}
