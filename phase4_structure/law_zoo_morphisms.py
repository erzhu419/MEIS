"""P4.7 — Law-zoo belief networks expressed as string diagrams.

Each of the three equivalence classes in phase4_structure/law_zoo is
mapped to an ABSTRACT morphism in the Markov category defined in
markov_category.py. The mapping ignores domain-specific parameter
values and prior distributions; it captures only the categorical
skeleton: which priors feed into which deterministic transforms which
feed into which observation.

Expected shape fingerprints (verified in tests/test_markov_category.py):

  exp_decay          (prior_scale ⊗ prior_rate) ; decay_kernel ; normal_obs
  saturation         (prior_scale ⊗ prior_rate) ; saturation_kernel ; normal_obs
  damped_oscillation (prior_A ⊗ prior_gamma ⊗ prior_omega ⊗ prior_phi)
                     ; damped_kernel ; normal_obs

Crucially, the *kind* of the deterministic kernel is class-specific
(decay / saturation / damped) so the shape signatures differ across
classes by design — matching what the PyTensor-level op-multiset and
WL signatures gave us.
"""

from __future__ import annotations

from phase4_structure.markov_category import (
    Obj, Morph, Atom, Compose, Tensor,
    prior, deterministic, observation,
    shape_signature, StringDiagramSignature,
)


# ---------------------------------------------------------------------------
# Shared object types (the "wire kinds" we use across the zoo)
# ---------------------------------------------------------------------------

_PARAM = Obj("Θ", kind="parameter")        # scalar latent
_SERIES_MU = Obj("μ(t)", kind="time-series")  # deterministic mean trajectory
_SERIES_Y = Obj("y(t)", kind="observation-series")


# ---------------------------------------------------------------------------
# Class diagrams
# ---------------------------------------------------------------------------


def exp_decay_diagram(scale_name: str = "y0", rate_name: str = "k") -> Morph:
    """y(t) = y0 · exp(-k·t)

    String diagram:  (prior ⊗ prior) ; decay_kernel ; normal_obs
    """
    p_scale = prior(f"prior_{scale_name}", _PARAM)
    p_rate = prior(f"prior_{rate_name}", _PARAM)
    kernel = Atom(dom=(_PARAM, _PARAM), cod=(_SERIES_MU,),
                  name="decay_kernel", kind="decay_kernel")
    obs = Atom(dom=(_SERIES_MU,), cod=(_SERIES_Y,),
               name="normal_obs", kind="normal_observation")
    return Tensor(p_scale, p_rate) >> kernel >> obs


def saturation_diagram(scale_name: str = "ymax", rate_name: str = "k") -> Morph:
    """y(t) = ymax · (1 - exp(-k·t))"""
    p_scale = prior(f"prior_{scale_name}", _PARAM)
    p_rate = prior(f"prior_{rate_name}", _PARAM)
    kernel = Atom(dom=(_PARAM, _PARAM), cod=(_SERIES_MU,),
                  name="saturation_kernel", kind="saturation_kernel")
    obs = Atom(dom=(_SERIES_MU,), cod=(_SERIES_Y,),
               name="normal_obs", kind="normal_observation")
    return Tensor(p_scale, p_rate) >> kernel >> obs


def damped_oscillation_diagram(scale_name: str = "A",
                                damping_name: str = "gamma",
                                freq_name: str = "omega",
                                phase_name: str = "phi") -> Morph:
    """y(t) = A · exp(-γ·t) · cos(ω·t + φ)"""
    p_A = prior(f"prior_{scale_name}", _PARAM)
    p_gamma = prior(f"prior_{damping_name}", _PARAM)
    p_omega = prior(f"prior_{freq_name}", _PARAM)
    p_phi = prior(f"prior_{phase_name}", _PARAM)
    kernel = Atom(dom=(_PARAM, _PARAM, _PARAM, _PARAM), cod=(_SERIES_MU,),
                  name="damped_kernel", kind="damped_kernel")
    obs = Atom(dom=(_SERIES_MU,), cod=(_SERIES_Y,),
               name="normal_obs", kind="normal_observation")
    priors = Tensor(Tensor(Tensor(p_A, p_gamma), p_omega), p_phi)
    return priors >> kernel >> obs


CLASS_DIAGRAMS = {
    "exp_decay": exp_decay_diagram,
    "saturation": saturation_diagram,
    "damped_oscillation": damped_oscillation_diagram,
}


def diagram_signature_for_class(class_id: str) -> StringDiagramSignature:
    return shape_signature(CLASS_DIAGRAMS[class_id]())


if __name__ == "__main__":
    print("Categorical string-diagram signatures for the law-zoo classes:\n")
    for cls, builder in CLASS_DIAGRAMS.items():
        sig = shape_signature(builder())
        print(f"  {cls:<22}  fp={sig.fingerprint}")
        print(f"  {'':<22}  shape={sig.shape}")
        print()
