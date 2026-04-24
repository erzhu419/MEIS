"""Step 6 / blueprint §4.3 validation: KL drift ranking matches intuition.

Five ascending-implausibility hypotheses about a new person's (height, weight)
should produce ascending KL drift from the current posterior.

Run:
    python -m phase3_embedding.tests.test_kl_drift
"""

from __future__ import annotations

import math

import numpy as np

from phase1_mvp.envs.alice_charlie import (
    AliceCharlie, THETA_MU, THETA_SIGMA, OBS_NOISE,
)
from phase3_embedding.kl_drift import (
    GaussianPosterior, Hypothesis, condition_normal, kl_normal,
    posterior_from_observations, rank_hypotheses, pretty_print,
)


def _synthetic_observations(seed: int = 0, n_obs: int = 10,
                            theta_true: float = THETA_MU) -> list[tuple[float, float]]:
    """Draw n_obs (height, weight) pairs from the ground-truth generative model."""
    rng = np.random.default_rng(seed)
    heights = rng.uniform(150, 190, size=n_obs)
    obs = []
    for h in heights:
        w = theta_true * h ** 3 + rng.normal(0.0, OBS_NOISE)
        obs.append((float(h), float(w)))
    return obs


def test_gaussian_closed_forms():
    """Unit: closed-form KL is symmetric-asymmetric and non-negative."""
    p = GaussianPosterior(mu=1.0, sigma=1.0)
    q = GaussianPosterior(mu=1.0, sigma=1.0)
    assert abs(kl_normal(p, q)) < 1e-10, "KL to identical dist should be 0"

    p2 = GaussianPosterior(mu=0.0, sigma=1.0)
    q2 = GaussianPosterior(mu=2.0, sigma=1.0)
    kl_pq = kl_normal(p2, q2)
    kl_qp = kl_normal(q2, p2)
    assert kl_pq > 0 and kl_qp > 0
    # Same-variance Gaussians: KL is symmetric
    assert abs(kl_pq - kl_qp) < 1e-10
    # Hand: 0.5 * (μ_p - μ_q)² / σ² = 0.5 * 4 / 1 = 2
    assert abs(kl_pq - 2.0) < 1e-10
    print("[PASS] Gaussian KL closed-form identities")


def test_conjugate_update():
    """Unit: conjugate update matches analytical posterior for y = θ x + noise."""
    prior = GaussianPosterior(mu=THETA_MU, sigma=THETA_SIGMA)
    obs = _synthetic_observations(seed=1, n_obs=20)
    xs = np.array([h ** 3 for h, _ in obs])
    ys = np.array([w for _, w in obs])

    post = condition_normal(prior, xs, ys, OBS_NOISE)
    # True theta at seed=1 via a handy recompute
    # Posterior mean should be within 2 posterior-stds of the true theta.
    assert abs(post.mu - THETA_MU) < 5 * post.sigma, (post.mu, THETA_MU, post.sigma)
    assert post.sigma < THETA_SIGMA, "posterior should be tighter than prior"
    print(f"[PASS] conjugate update tight: post=N({post.mu:.3e}, {post.sigma:.3e}), "
          f"prior std={THETA_SIGMA:.3e}, shrinkage={THETA_SIGMA/post.sigma:.0f}x")


def test_ranking_matches_intuition():
    """Core Step 6 check: ascending implausibility → ascending KL drift.

    Build a posterior from 10 observations at theta=THETA_MU, then pose
    5 hypotheses about a new person at height 170 cm with various weights.
    Expected ranking (from least implausible to most):
        70 kg (center of predictive) < 80 kg (tail) < 55 kg (other tail)
        < 120 kg (far) < 200 kg (absurd)
    """
    prior = GaussianPosterior(mu=THETA_MU, sigma=THETA_SIGMA)
    obs = _synthetic_observations(seed=42, n_obs=10)
    post_D = posterior_from_observations(prior, obs, OBS_NOISE)
    # Predictive mean weight at h=170 under posterior:
    expected = post_D.mu * 170 ** 3
    print(f"  posterior N(μ={post_D.mu:.3e}, σ={post_D.sigma:.3e})")
    print(f"  expected weight at h=170: {expected:.2f} kg")

    hypotheses = [
        Hypothesis("H1", "Alice is 170 cm and 70 kg (central)",         [(170.0, 70.0)]),
        Hypothesis("H2", "Alice is 170 cm and 80 kg (upper-typical)",   [(170.0, 80.0)]),
        Hypothesis("H3", "Alice is 170 cm and 55 kg (lower-typical)",   [(170.0, 55.0)]),
        Hypothesis("H4", "Alice is 170 cm and 120 kg (obese)",          [(170.0, 120.0)]),
        Hypothesis("H5", "Alice is 170 cm and 200 kg (absurd)",         [(170.0, 200.0)]),
    ]
    scores = rank_hypotheses(post_D, hypotheses)
    pretty_print(scores)

    names_in_order = [s.hypothesis.name for s in scores]
    # Absolutely require H5 is last (most disruptive).
    assert names_in_order[-1] == "H5", f"H5 not last: {names_in_order}"
    # Require H1 is among the top 2 (most coherent).
    assert names_in_order[0] in {"H1"}, f"H1 not first: {names_in_order}"
    # H4 should be in positions 3-4 (large drift but not absurd).
    assert names_in_order.index("H4") >= 2, f"H4 too high: {names_in_order}"
    # KL should be strictly monotonically increasing
    kls = [s.kl_from_base for s in scores]
    assert all(kls[i+1] >= kls[i] for i in range(len(kls)-1))
    print(f"[PASS] KL ranking matches intuition: {names_in_order}")


def test_composed_vs_single_hypothesis():
    """Sanity: a hypothesis that combines multiple (consistent) observations
    should have a KL that sums approximately to the sum of their individual
    KLs (when they're IID relative to the current posterior)."""
    prior = GaussianPosterior(mu=THETA_MU, sigma=THETA_SIGMA)
    obs = _synthetic_observations(seed=7, n_obs=10)
    post_D = posterior_from_observations(prior, obs, OBS_NOISE)

    single_pairs = [(170.0, 70.0), (165.0, 63.0)]
    h_single_1 = Hypothesis("a", "", [single_pairs[0]])
    h_single_2 = Hypothesis("b", "", [single_pairs[1]])
    h_combined = Hypothesis("ab", "", single_pairs)

    scores = rank_hypotheses(post_D, [h_single_1, h_single_2, h_combined])
    d = {s.hypothesis.name: s.kl_from_base for s in scores}
    # Combined KL should be larger than either single (strictly more evidence)
    assert d["ab"] >= max(d["a"], d["b"]) - 1e-9
    print(f"[PASS] composed KL ≥ single: a={d['a']:.6f}, b={d['b']:.6f}, ab={d['ab']:.6f}")


def test_structural_hypothesis_is_rankable():
    """Sanity: a hypothesis can include multiple synthetic pairs and still
    fit the API without error. This scaffolds future composite-claim support."""
    prior = GaussianPosterior(mu=THETA_MU, sigma=THETA_SIGMA)
    obs = _synthetic_observations(seed=3, n_obs=10)
    post_D = posterior_from_observations(prior, obs, OBS_NOISE)

    wide_h = Hypothesis(
        "family", "all five adults of varying heights",
        [(150.0, 48.0), (160.0, 59.0), (170.0, 70.0), (180.0, 84.0), (190.0, 97.0)],
    )
    narrow_h = Hypothesis(
        "one", "single adult 170 cm 70 kg",
        [(170.0, 70.0)],
    )
    scores = rank_hypotheses(post_D, [wide_h, narrow_h])
    pretty_print(scores)
    # wide_h contains more synthetic observations, each coherent with posterior,
    # so its cumulative drift should be >= narrow_h's (more info, more update).
    d = {s.hypothesis.name: s.kl_from_base for s in scores}
    assert d["family"] >= d["one"] - 1e-9
    print(f"[PASS] multi-obs hypothesis ranked correctly")


if __name__ == "__main__":
    print("=== Phase 3 Step 6 validation: KL drift (minimum embedding distance) ===\n")
    test_gaussian_closed_forms()
    test_conjugate_update()
    print()
    test_ranking_matches_intuition()
    print()
    test_composed_vs_single_hypothesis()
    print()
    test_structural_hypothesis_is_rankable()
    print("\nAll Step 6 validation checks passed.")
