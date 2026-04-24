"""MEIS P2.3 commit 5 — sequential-experiment demo (offline, 0 API).

Demonstrates that a BeliefStore with cross-run evidence accumulation
produces tighter posterior-predictive distributions than a stateless
run, using pure math (no LLM involved). This validates that the
persistence mechanism itself delivers a real, measurable benefit
before spending API tokens on an end-to-end LLM test.

Setup:
  - Fix a true population parameter `theta_true = 1.414e-5`.
  - Each "round" produces 10 noisy observations
      (h_ij, w_ij) where w = theta_true * h^3 + Normal(0, 2 kg).
  - Condition A (STATELESS): round k fits a posterior from its own 10
    observations only, starting from the library prior.
  - Condition B (PERSIST):   round k fits a posterior from cumulative
    (1..k)*10 observations — each round inherits previous round's
    posterior as its prior.

Metric: posterior-predictive MAE on 10 held-out random heights.

Expected: B's predictive-MAE decreases monotonically with round;
A's stays noisy around a baseline. At round 10, B should be
substantially tighter than A.

Run:
    python -m phase3_embedding.demo_sequential
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from phase2_prior_library.retrieval import PriorLibrary
from phase3_embedding.belief_store import BeliefStore, Evidence
from phase3_embedding.kl_drift import GaussianPosterior


THETA_TRUE = 1.414e-5
OBS_SIGMA = 2.0
HEIGHT_LOW, HEIGHT_HIGH = 150.0, 190.0
N_OBS_PER_ROUND = 10
N_EVAL_PER_ROUND = 10
N_ROUNDS = 10
LATENT_ID = "weight_from_height_cube_law::theta"


def _simulate_round_observations(rng: np.random.Generator,
                                 n: int = N_OBS_PER_ROUND
                                 ) -> list[tuple[float, float]]:
    """Return n (height_cm, weight_kg) pairs sampled from ground-truth."""
    heights = rng.uniform(HEIGHT_LOW, HEIGHT_HIGH, n)
    weights = THETA_TRUE * heights ** 3 + rng.normal(0.0, OBS_SIGMA, n)
    return list(zip(heights.tolist(), weights.tolist()))


def _predictive_mae(posterior: GaussianPosterior,
                    eval_heights: np.ndarray,
                    rng: np.random.Generator) -> float:
    """MAE of the posterior-predictive mean against noisy ground-truth
    weights at `eval_heights`. Marginal over both posterior uncertainty
    and observation noise, but we use the posterior MEAN as the point
    prediction (decision-theoretic MMSE under squared loss, adequate
    under MAE too when posterior is near-symmetric)."""
    gt = THETA_TRUE * eval_heights ** 3 + rng.normal(0.0, OBS_SIGMA, len(eval_heights))
    preds = posterior.mu * eval_heights ** 3
    return float(np.mean(np.abs(preds - gt)))


def run_stateless_vs_persist(verbose: bool = True) -> dict:
    """Run both conditions and collect per-round predictive MAE."""
    lib = PriorLibrary.load_default()

    rng_data = np.random.default_rng(42)      # shared across conditions: same observations
    rng_eval = np.random.default_rng(4242)    # same eval sets too

    rounds_data = [_simulate_round_observations(rng_data) for _ in range(N_ROUNDS)]
    rounds_eval_heights = [rng_eval.uniform(HEIGHT_LOW, HEIGHT_HIGH, N_EVAL_PER_ROUND)
                           for _ in range(N_ROUNDS)]

    # -- Condition A: stateless --
    A_maes: list[float] = []
    A_sigmas: list[float] = []
    for k in range(N_ROUNDS):
        store = BeliefStore.from_library(lib)
        for j, (h, w) in enumerate(rounds_data[k]):
            store.add_evidence(
                Evidence(id=f"A_r{k}_obs_{j}", kind="observation",
                         target_nodes=[LATENT_ID], value=w, x=h ** 3,
                         provenance="stateless_A"),
                obs_sigma=OBS_SIGMA,
            )
        post = store.get_node(LATENT_ID).posterior.as_gaussian()
        A_maes.append(_predictive_mae(
            post, rounds_eval_heights[k], np.random.default_rng(777 + k)))
        A_sigmas.append(post.sigma)

    # -- Condition B: persist --
    with tempfile.TemporaryDirectory() as tmp:
        persist_dir = Path(tmp)
        B_maes: list[float] = []
        B_sigmas: list[float] = []
        B_n_evidence: list[int] = []
        for k in range(N_ROUNDS):
            if persist_dir.exists() and any(persist_dir.iterdir()):
                store = BeliefStore.load(persist_dir)
            else:
                store = BeliefStore.from_library(lib)
            for j, (h, w) in enumerate(rounds_data[k]):
                store.add_evidence(
                    Evidence(id=f"B_r{k}_obs_{j}", kind="observation",
                             target_nodes=[LATENT_ID], value=w, x=h ** 3,
                             provenance="persist_B"),
                    obs_sigma=OBS_SIGMA,
                )
            store.save(persist_dir)
            post = store.get_node(LATENT_ID).posterior.as_gaussian()
            B_maes.append(_predictive_mae(
                post, rounds_eval_heights[k], np.random.default_rng(777 + k)))
            B_sigmas.append(post.sigma)
            B_n_evidence.append(len(store.evidence))

    results = dict(
        A_maes=A_maes, A_sigmas=A_sigmas,
        B_maes=B_maes, B_sigmas=B_sigmas,
        B_n_evidence=B_n_evidence,
    )

    if verbose:
        print(f"\n{'round':>5}  {'A (stateless)':>20}  {'B (persist)':>20}   "
              f"{'n_evidence_B':>14}")
        print(f"{'':>5}  {'MAE      σ':>20}  {'MAE      σ':>20}")
        print('-' * 72)
        for k in range(N_ROUNDS):
            print(
                f"{k+1:>5}  "
                f"{A_maes[k]:7.3f}  {A_sigmas[k]:7.2e}   "
                f"{B_maes[k]:7.3f}  {B_sigmas[k]:7.2e}   "
                f"{B_n_evidence[k]:>14}"
            )
        print()
        print(f"Mean MAE  : A={np.mean(A_maes):.3f}   B={np.mean(B_maes):.3f}   "
              f"ratio B/A = {np.mean(B_maes) / np.mean(A_maes):.3f}")
        print(f"Round-10  : A={A_maes[-1]:.3f}   B={B_maes[-1]:.3f}   "
              f"reduction = {(A_maes[-1] - B_maes[-1]) / A_maes[-1] * 100:+.1f}%")
        print(f"σ round-10: A={A_sigmas[-1]:.2e}  B={B_sigmas[-1]:.2e}   "
              f"ratio B/A = {B_sigmas[-1] / A_sigmas[-1]:.3f}")

    return results


if __name__ == "__main__":
    print("MEIS P2.3 commit-5 demo — sequential experiment (stateless vs persist)")
    results = run_stateless_vs_persist(verbose=True)
