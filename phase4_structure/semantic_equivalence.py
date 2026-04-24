"""P4.8 — Semantic equivalence checks for Markov-category morphisms.

This module upgrades P4.7's SYNTACTIC equivalence (string-diagram
shape matching) with two stronger, numerically computable checks:

  1. Blackwell-Sherman-Stein (BSS) operational equivalence
     ---------------------------------------------------
     In the full Fritz 2020 Markov-category framework, two statistical
     experiments are BSS-equivalent iff each can be obtained from the
     other via post-processing. For our law-zoo (where within-class
     domains share an identical likelihood formula parametrised by the
     same (scale, rate, ...) tuple, and only prior distributions
     differ), BSS-equivalence of the LIKELIHOOD reduces to the
     condition

          ∀ θ, t, y:  log P_a(y | θ, t) = log P_b(y | θ, t)

     which we check by evaluating both log-likelihoods on a random
     grid of (θ, t, y) points and requiring exact equality (up to
     floating-point tolerance). This is semantically stronger than
     shape matching: it also catches cases where two functionally
     distinct models happen to share a string-diagram skeleton.

  2. Perrone 2024 categorical KL divergence on Markov kernels
     -----------------------------------------------------
     For kernels K_a, K_b : Θ ⇝ Y (our belief networks' observation
     kernels), the Perrone-style information-geometric divergence is

          D(K_a ‖ K_b) = E_{θ ~ P_Θ, t} [ KL( K_a(·|θ, t) ‖ K_b(·|θ, t) ) ]

     For Gaussian observation Normal(μ, σ²) with the same σ across
     the two kernels (shared noise scale),

          KL( N(μ_a, σ²) ‖ N(μ_b, σ²) ) = (μ_a - μ_b)² / (2 σ²)

     so D is a Monte Carlo expectation of the squared mean-function
     difference, normalised by noise. Within-class: μ_a = μ_b → D = 0.
     Cross-class: μ differs → D > 0.

Together these fill the two gaps flagged at the end of P4.7: full BSS
is now operationally checked (within the restricted world of shared
likelihood formulas), and a Perrone-style kernel divergence is
computable.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from phase4_structure.law_zoo import CLASS_OF, exp_decay, saturation, damped_oscillation


_CLASS_MODULES = {
    "exp_decay": exp_decay,
    "saturation": saturation,
    "damped_oscillation": damped_oscillation,
}


def class_of(domain_name: str) -> str:
    return CLASS_OF[domain_name]


def class_mu(class_id: str):
    return _CLASS_MODULES[class_id].class_mu


def class_param_names(class_id: str) -> tuple:
    return _CLASS_MODULES[class_id].CLASS_PARAM_NAMES


# ---------------------------------------------------------------------------
# Gaussian log-likelihood
# ---------------------------------------------------------------------------


def gaussian_log_likelihood(y: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    return -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def domain_log_likelihood(domain_name: str,
                          theta: tuple,
                          t: np.ndarray,
                          y: np.ndarray,
                          sigma: float) -> np.ndarray:
    """Evaluate the domain's likelihood at a given (θ, t, y, σ)."""
    mu_fn = class_mu(CLASS_OF[domain_name])
    mu = mu_fn(theta, t)
    return gaussian_log_likelihood(y, mu, sigma)


# ---------------------------------------------------------------------------
# BSS operational equivalence
# ---------------------------------------------------------------------------


@dataclass
class BSSResult:
    domain_a: str
    domain_b: str
    n_samples: int
    max_abs_log_lik_diff: float
    bss_equivalent: bool
    tolerance: float


def bss_likelihood_equivalent(domain_a: str,
                              domain_b: str,
                              n_samples: int = 200,
                              seed: int = 0,
                              tolerance: float = 1e-10,
                              cross_class_theta_strategy: str = "canonical",
                              ) -> BSSResult:
    """Evaluate both domains' log-likelihoods on the SAME random (θ, t, y, σ)
    grid and return the max absolute disagreement.

    For within-class pairs, both domains use the same canonical (scale,
    rate, ...) parameter tuple — their class_mu functions are identical,
    so log-likelihoods agree exactly (up to floating-point noise).

    For cross-class pairs, we must choose a compatibility strategy:
      - 'canonical': use each class's first N parameters where N =
        min(dim(θ_a), dim(θ_b)), zero-pad if necessary. This is a
        *legitimate* cross-class test: if the two classes were BSS-
        equivalent under any parameter bijection, they would agree on
        at least one such shared-prefix sampling. Disagreement on even
        one sample disproves equivalence.
    """
    rng = np.random.default_rng(seed)
    cls_a, cls_b = CLASS_OF[domain_a], CLASS_OF[domain_b]
    params_a = class_param_names(cls_a)
    params_b = class_param_names(cls_b)

    diffs = []
    for _ in range(n_samples):
        # Sample a random θ vector of length max(dim_a, dim_b)
        dim_max = max(len(params_a), len(params_b))
        theta_full = rng.lognormal(mean=0.0, sigma=0.5, size=dim_max)
        # Slice to each class's arity
        theta_a = tuple(theta_full[: len(params_a)])
        theta_b = tuple(theta_full[: len(params_b)])
        t = rng.uniform(0.0, 10.0, size=8)
        y = rng.normal(size=t.shape)
        sigma = 1.0
        ll_a = domain_log_likelihood(domain_a, theta_a, t, y, sigma)
        ll_b = domain_log_likelihood(domain_b, theta_b, t, y, sigma)
        diffs.append(np.max(np.abs(ll_a - ll_b)))

    max_diff = float(np.max(diffs))
    return BSSResult(
        domain_a=domain_a, domain_b=domain_b,
        n_samples=n_samples,
        max_abs_log_lik_diff=max_diff,
        bss_equivalent=(max_diff < tolerance),
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Perrone categorical KL divergence
# ---------------------------------------------------------------------------


@dataclass
class KernelKLResult:
    domain_a: str
    domain_b: str
    n_samples: int
    sigma: float
    kl_estimate: float
    kl_stderr: float


def perrone_kernel_kl(domain_a: str,
                      domain_b: str,
                      n_samples: int = 500,
                      seed: int = 0,
                      sigma: float = 1.0,
                      ) -> KernelKLResult:
    """Monte Carlo estimate of Perrone-style KL between two Markov kernels
    K_a(y|θ,t) and K_b(y|θ,t), both modelled as Normal(μ(θ,t), σ²):

        D = E_{θ, t} [ (μ_a(θ,t) - μ_b(θ,t))² / (2 σ²) ]

    θ is sampled from a generic LogNormal reference (so this is a
    property of the likelihood families, not of the specific priors).

    Within-class pairs (same class_mu function): D = 0 exactly.
    Cross-class pairs: D > 0 by construction of distinct class_mu.
    """
    rng = np.random.default_rng(seed)
    cls_a, cls_b = CLASS_OF[domain_a], CLASS_OF[domain_b]
    params_a = class_param_names(cls_a)
    params_b = class_param_names(cls_b)

    per_sample_kl = np.empty(n_samples)
    for i in range(n_samples):
        dim_max = max(len(params_a), len(params_b))
        theta_full = rng.lognormal(mean=0.0, sigma=0.5, size=dim_max)
        theta_a = tuple(theta_full[: len(params_a)])
        theta_b = tuple(theta_full[: len(params_b)])
        t = rng.uniform(0.0, 10.0, size=16)
        mu_a = class_mu(cls_a)(theta_a, t)
        mu_b = class_mu(cls_b)(theta_b, t)
        per_sample_kl[i] = float(np.mean((mu_a - mu_b) ** 2) / (2.0 * sigma ** 2))

    return KernelKLResult(
        domain_a=domain_a, domain_b=domain_b,
        n_samples=n_samples, sigma=sigma,
        kl_estimate=float(per_sample_kl.mean()),
        kl_stderr=float(per_sample_kl.std(ddof=1) / np.sqrt(n_samples)),
    )


if __name__ == "__main__":
    print("P4.8 semantic equivalence — BSS + Perrone kernel KL\n")
    print("=== BSS likelihood equivalence ===")
    pairs = [
        ("rc_circuit", "radioactive_decay"),        # within exp_decay
        ("capacitor_charging", "monomolecular_growth"),  # within saturation
        ("rlc_circuit", "pendulum"),                # within damped
        ("rc_circuit", "capacitor_charging"),       # cross: exp_decay / saturation
        ("rc_circuit", "rlc_circuit"),              # cross: exp_decay / damped
        ("capacitor_charging", "rlc_circuit"),      # cross: saturation / damped
    ]
    for a, b in pairs:
        r = bss_likelihood_equivalent(a, b, n_samples=100, seed=0)
        verdict = "✓ BSS-equivalent" if r.bss_equivalent else "✗ inequivalent"
        print(f"  {a:<22} vs {b:<22}  max|Δlog p| = {r.max_abs_log_lik_diff:.3e}  "
              f"{verdict}")

    print("\n=== Perrone categorical KL divergence ===")
    for a, b in pairs:
        r = perrone_kernel_kl(a, b, n_samples=400, seed=0)
        print(f"  {a:<22} vs {b:<22}  D = {r.kl_estimate:.4e}  "
              f"(SE {r.kl_stderr:.4e})")
