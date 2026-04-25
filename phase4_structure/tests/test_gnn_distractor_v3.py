"""GNN trained-vs-untrained gap on law-zoo v3 with within-class variation.

Closes the original §5 honest null in Paper 3: on the previous law-zoo
v2 (one canonical graph per class) random-init MPNN already achieved
ARI = 1.00, so we couldn't show a training advantage. The v3 fixture
introduces within-class variation:
  - 5-dim one-hot type label perturbed by Gaussian noise (sigma=0.15)
  - 0..3 random "decoration" prior leaves attached to obs
  - 20% edge dropout

Acceptance:
  1. Random-init MPNN ARI on v3 fixture is significantly below 1.0
     (mean across 5 seeds < 0.5).
  2. Trained MPNN (300 epochs NT-Xent contrastive) achieves ARI = 1.0
     on every seed.
  3. The gap (trained mean − random mean) is > 0.5.

This is the experiment that gives Paper 3's GNN section non-null
empirical content.
"""

from __future__ import annotations

from phase4_structure.gnn_distractor_fixture import run_gnn_comparison


def test_random_init_does_not_solve_v3():
    r = run_gnn_comparison(k_per_class=8, seeds=(0, 1, 2, 3, 4),
                            verbose=False)
    assert r["random_init_mean"] < 0.5, \
        f"random-init mean ARI {r['random_init_mean']} >= 0.5; fixture too easy"
    print(f"[PASS] random-init mean ARI = {r['random_init_mean']:.3f} "
          f"(individual: {[f'{x:.2f}' for x in r['random_init_aris']]}) — "
          f"v3 fixture is non-trivial")


def test_trained_solves_v3():
    r = run_gnn_comparison(k_per_class=8, seeds=(0, 1, 2, 3, 4),
                            verbose=False)
    assert r["trained_mean"] == 1.0, \
        f"trained mean ARI {r['trained_mean']} < 1.0"
    for ari in r["trained_aris"]:
        assert ari == 1.0, f"trained seed ARI {ari} < 1.0"
    print(f"[PASS] trained MPNN reaches ARI = 1.0 on all 5 seeds")


def test_gap_is_substantial():
    r = run_gnn_comparison(k_per_class=8, seeds=(0, 1, 2, 3, 4),
                            verbose=False)
    assert r["gap"] > 0.5, f"trained-vs-random gap {r['gap']} <= 0.5"
    print(f"[PASS] trained-vs-random gap = +{r['gap']:.3f} on law-zoo v3 "
          f"(random {r['random_init_mean']:.3f} → trained {r['trained_mean']:.3f})")


if __name__ == "__main__":
    print("=== GNN trained-vs-untrained on law-zoo v3 ===\n")
    test_random_init_does_not_solve_v3()
    test_trained_solves_v3()
    test_gap_is_substantial()
    print("\nAll GNN v3 distractor checks passed.")
