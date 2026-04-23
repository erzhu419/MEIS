"""Phase 1 smoke test: port d1 Bayesian Tug-of-War from Church to Python.

Source model: world-models/domains/d1-probabilistic-reasoning/world-model.scm

Uses pure-numpy rejection sampling to match Church's `rejection-query` semantics
1:1 (no sigmoid relaxation, no CLT approximation). For a 4-player / 2-match
setup this converges in seconds.

Observations:
    Match 1: Tom won against John.
    Match 2: {John, Mary} won against {Tom, Sue}.
Query: posterior distribution over Mary's strength.
Expected: posterior mean noticeably > 50 (Mary is above average).
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLAYERS = ("tom", "john", "mary", "sue")
N_POSTERIOR_SAMPLES = 5000
MAX_ATTEMPTS = 5_000_000

rng = np.random.default_rng(0)


def sample_world():
    strength = {p: rng.normal(50.0, 20.0) for p in PLAYERS}
    laziness = {p: rng.uniform(0.0, 1.0) for p in PLAYERS}
    return strength, laziness


def team_strength(team, strength, laziness):
    total = 0.0
    for p in team:
        if rng.uniform() < laziness[p]:
            total += strength[p] / 2.0
        else:
            total += strength[p]
    return total


def won_against(t1, t2, strength, laziness):
    return team_strength(t1, strength, laziness) > team_strength(t2, strength, laziness)


def main():
    mary_samples = []
    tom_samples = []
    attempts = 0
    while len(mary_samples) < N_POSTERIOR_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1
        strength, laziness = sample_world()
        if not won_against(["tom"], ["john"], strength, laziness):
            continue
        if not won_against(["john", "mary"], ["tom", "sue"], strength, laziness):
            continue
        mary_samples.append(strength["mary"])
        tom_samples.append(strength["tom"])

    mary_samples = np.array(mary_samples)
    tom_samples = np.array(tom_samples)
    accept_rate = len(mary_samples) / attempts

    print(f"Accepted {len(mary_samples)} / {attempts} (rate={accept_rate:.4f})")
    print(f"Mary strength posterior: mean={mary_samples.mean():.2f}, std={mary_samples.std():.2f}")
    print(f"Tom  strength posterior: mean={tom_samples.mean():.2f}, std={tom_samples.std():.2f}")
    print(f"Prior            : mean=50.00, std=20.00")

    out_dir = Path(__file__).parent
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mary_samples, bins=40, density=True, alpha=0.7, label="Mary (posterior)")
    ax.hist(tom_samples, bins=40, density=True, alpha=0.4, label="Tom (posterior)")
    xs = np.linspace(-20, 120, 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - 50) / 20) ** 2) / (20 * np.sqrt(2 * np.pi)),
            "k--", alpha=0.6, label="Prior N(50, 20)")
    ax.axvline(50, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("strength")
    ax.set_ylabel("density")
    ax.set_title("Tug-of-War posterior (rejection sampling, N=5000)")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "tug_of_war_posterior.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    if mary_samples.mean() > 55:
        print("[PASS] Mary's posterior mean > 55, consistent with paper's qualitative result.")
    else:
        print("[WARN] Mary's posterior mean not clearly above prior; check model.")


if __name__ == "__main__":
    main()
