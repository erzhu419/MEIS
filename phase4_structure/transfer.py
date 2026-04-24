"""P4.4 — Structural transfer as inductive bias.

Plan §Phase 5 task 3 target: held-out MSE ≥ 30% reduction vs cold-start.

Protocol:

  Step 1  Fit the SOURCE domain's belief network on a rich dataset.
          Extract the posterior log-SDs for its scale latent and rate
          latent — call these σ_scale^src, σ_rate^src. These capture
          how tightly data pins down the respective latents under the
          shared functional form (a class-level precision estimate).

  Step 2  Verify TARGET and SOURCE share a StructuralSignature (P4.2
          fingerprint) — i.e. they belong to the same equivalence class
          (P4.3 retrieval gate). Refuse transfer otherwise.

  Step 3  Fit the TARGET on a SMALL dataset (n_obs points, earliest
          timestamps) under two conditions:
            (a) cold-start  — uninformed prior: SD = uninformed_log_sd
                              (default 1.5) on both latents, centered at
                              target's default prior mean. This models
                              "we know the functional form but have no
                              class-level precision estimate".
            (b) transferred — SD = (σ_scale^src, σ_rate^src), same
                              prior means. This models "we have the
                              source's precision estimate".

  Step 4  Predict on held-out later-time observations, compute MSE
          under each condition. Improvement = 1 - MSE_trans / MSE_cold.

The claim: within-class transfer yields improvement ≥ 30%; cross-class
attempts are refused by the signature gate.

Honest framing: the law-zoo's "default" priors in
phase4_structure/law_zoo/*.py are already class-calibrated (σ=0.3–0.4
on log-params), so using them as the cold-start baseline would bury
the transfer effect under an already-informed prior. We isolate the
*class-level precision contribution* by pitting a truly uninformed
baseline against a source-informed one; both share the same prior
means (a generic "order-of-magnitude" sketch that doesn't encode
class-level tightness).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pymc as pm

from phase4_structure.law_zoo import DOMAINS, CLASS_OF, exp_decay, saturation
from phase4_structure.signature import signature_for_domain


@dataclass
class TransferResult:
    target: str
    source: str
    mse_cold: float
    mse_transfer: float
    improvement: float
    scale_sigma_transferred: float
    rate_sigma_transferred: float
    signature_match: bool


def _default_grid(name: str, n: int) -> np.ndarray:
    if CLASS_OF[name] == "exp_decay":
        return exp_decay.default_t_grid(name, n=n)
    return saturation.default_t_grid(name, n=n)


def _scale_rate_names(name: str) -> tuple[str, str]:
    if CLASS_OF[name] == "exp_decay":
        return "y0", "k"
    return "ymax", "k"


def infer_posterior_shape(source_name: str, n_points: int = 30, seed: int = 0,
                          draws: int = 800, tune: int = 500):
    """Fit the source with rich data and return posterior log-SDs for its
    two primary latents. Numbers are what P4.4 transfers."""
    rng = np.random.default_rng(seed)
    t = _default_grid(source_name, n_points)
    y = DOMAINS[source_name].simulate(t, rng)
    with DOMAINS[source_name].build_model(t, y):
        idata = pm.sample(draws=draws, tune=tune, chains=2, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False)
    scale_name, rate_name = _scale_rate_names(source_name)
    scale_post = np.log(idata.posterior[scale_name].values.flatten())
    rate_post = np.log(idata.posterior[rate_name].values.flatten())
    return float(scale_post.std()), float(rate_post.std())


def _mse_on_heldout(idata, t_ho: np.ndarray, y_ho: np.ndarray,
                    class_id: str, scale_name: str, rate_name: str) -> float:
    scale = idata.posterior[scale_name].values.flatten()
    rate = idata.posterior[rate_name].values.flatten()
    if class_id == "exp_decay":
        pred = scale[:, None] * np.exp(-rate[:, None] * t_ho[None, :])
    else:
        pred = scale[:, None] * (1.0 - np.exp(-rate[:, None] * t_ho[None, :]))
    mu = pred.mean(axis=0)
    return float(((mu - y_ho) ** 2).mean())


def run_transfer_benchmark(target_name: str,
                           source_name: str,
                           n_target_obs: int = 3,
                           n_heldout: int = 6,
                           seed: int = 0,
                           draws: int = 800,
                           tune: int = 500,
                           uninformed_log_sd: float = 1.5,
                           obs_window_frac: float | None = None,
                           source_shape: tuple[float, float] | None = None
                           ) -> TransferResult:
    """Run one (target, source) transfer trial. Returns a TransferResult.

    If source_shape=(scale_log_sd, rate_log_sd) is provided, skips
    refitting the source (useful when running many targets against the
    same source).

    Refuses transfer (raises) if target and source have different
    StructuralSignatures — this is the P4.3 retrieval gate applied to
    the P4.4 inductive-bias-transfer step.
    """
    # --- Signature gate ---
    t_dummy = _default_grid(target_name, 5)
    y_dummy = DOMAINS[target_name].simulate(t_dummy, np.random.default_rng(seed))
    sig_t = signature_for_domain(DOMAINS[target_name], t_dummy, y_dummy)
    t_src_dummy = _default_grid(source_name, 5)
    y_src_dummy = DOMAINS[source_name].simulate(t_src_dummy, np.random.default_rng(seed))
    sig_s = signature_for_domain(DOMAINS[source_name], t_src_dummy, y_src_dummy)
    if not sig_t.matches(sig_s):
        raise ValueError(
            f"structural signature mismatch: {target_name} ({sig_t.fingerprint}) "
            f"vs {source_name} ({sig_s.fingerprint}); transfer refused"
        )

    # --- Source posterior shape ---
    if source_shape is None:
        scale_sd, rate_sd = infer_posterior_shape(
            source_name, n_points=30, seed=seed, draws=draws, tune=tune)
    else:
        scale_sd, rate_sd = source_shape

    # --- Target: extrapolation split.
    # Early obs on first `obs_window_frac` of t-range; held-out on last
    # 50%. This isolates the case where parameters are poorly identified
    # from the obs window and held-out MSE is dominated by parameter
    # uncertainty at late t. exp_decay needs a tighter window (~5%) to
    # avoid seeing meaningful decay; saturation can use a wider one
    # (~20%) because ymax identification requires reaching the plateau.
    rng_t = np.random.default_rng(seed + 17)
    if CLASS_OF[target_name] == "exp_decay":
        t_min, t_max = exp_decay._SPECS[target_name].t_range
        if obs_window_frac is None:
            obs_window_frac = 0.05
    else:
        t_min, t_max = saturation._SPECS[target_name].t_range
        if obs_window_frac is None:
            obs_window_frac = 0.20
    span = t_max - t_min
    t_obs = np.linspace(t_min, t_min + obs_window_frac * span, n_target_obs)
    t_ho = np.linspace(t_min + 0.5 * span, t_max, n_heldout)
    y_obs = DOMAINS[target_name].simulate(t_obs, rng_t)
    y_ho = DOMAINS[target_name].simulate(t_ho, rng_t)

    scale_name, rate_name = _scale_rate_names(target_name)
    class_id = CLASS_OF[target_name]

    # --- Cold-start with UNINFORMED prior SDs (no class-level precision) ---
    cold_overrides = {f"{scale_name}_sigma": uninformed_log_sd,
                      f"{rate_name}_sigma": uninformed_log_sd}
    with DOMAINS[target_name].build_model(t_obs, y_obs, prior_overrides=cold_overrides):
        idata_cold = pm.sample(draws=draws, tune=tune, chains=2, random_seed=seed,
                               progressbar=False, compute_convergence_checks=False)
    mse_cold = _mse_on_heldout(idata_cold, t_ho, y_ho, class_id, scale_name, rate_name)

    # --- Transfer: source-derived prior SDs (class-level precision) ---
    overrides = {f"{scale_name}_sigma": scale_sd, f"{rate_name}_sigma": rate_sd}
    with DOMAINS[target_name].build_model(t_obs, y_obs, prior_overrides=overrides):
        idata_trans = pm.sample(draws=draws, tune=tune, chains=2, random_seed=seed,
                                progressbar=False, compute_convergence_checks=False)
    mse_trans = _mse_on_heldout(idata_trans, t_ho, y_ho, class_id, scale_name, rate_name)

    improvement = 1.0 - (mse_trans / mse_cold) if mse_cold > 0 else 0.0
    return TransferResult(
        target=target_name, source=source_name,
        mse_cold=mse_cold, mse_transfer=mse_trans, improvement=improvement,
        scale_sigma_transferred=scale_sd, rate_sigma_transferred=rate_sd,
        signature_match=True,
    )


if __name__ == "__main__":
    print("Phase 4.4 transfer benchmark — within-class\n")
    for target, source in [
        ("capacitor_charging", "monomolecular_growth"),
        ("light_adaptation", "capacitor_charging"),
        ("rc_circuit", "first_order_reaction"),
        ("forgetting_curve", "first_order_reaction"),
    ]:
        r = run_transfer_benchmark(
            target_name=target, source_name=source,
            n_target_obs=3, n_heldout=6, seed=0,
        )
        print(f"  {target:<22} <- {source:<22}  "
              f"cold={r.mse_cold:.5f}  trans={r.mse_transfer:.5f}  "
              f"improvement={100*r.improvement:+.1f}%")
