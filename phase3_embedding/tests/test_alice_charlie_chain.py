"""Validation for the Plan §Phase 2 multi-observable belief-propagation demo.

Run:
    python -m phase3_embedding.tests.test_alice_charlie_chain
"""

from __future__ import annotations

import numpy as np

from phase3_embedding.demo_alice_charlie_chain import run_progressive_evidence


def test_progressive_evidence_tightens_posterior():
    rows = run_progressive_evidence(verbose=False, random_seed=0)
    assert len(rows) == 4
    stage_by_name = {r["name"]: r for r in rows}

    # Stage 0: symmetric prior → P ≈ 0.5
    p0 = stage_by_name["stage_0_prior"]["p_alice_heavier"]
    assert 0.4 < p0 < 0.6, f"prior P should be near 0.5, got {p0:.3f}"

    # Stage 1: height evidence should push P > 0.5 substantially
    p1 = stage_by_name["stage_1_heights"]["p_alice_heavier"]
    assert p1 > p0 + 0.05, f"heights should raise P above prior: {p0:.3f} -> {p1:.3f}"

    # Stage 2: shoes are independent of height in the model; P should be close
    # to stage 1 (within MCMC noise window of ~0.05).
    p2 = stage_by_name["stage_2_equal_shoes"]["p_alice_heavier"]
    assert abs(p2 - p1) < 0.08, \
        f"equal shoes should not swing P much (shoes⊥height): {p1:.3f} -> {p2:.3f}"

    # Stage 3: footprint-depth evidence should pull P strongly toward 1
    p3 = stage_by_name["stage_3_deeper_footprint"]["p_alice_heavier"]
    assert p3 > 0.85, f"footprint evidence should give P > 0.85, got {p3:.3f}"
    assert p3 > p2, f"adding footprint should increase P further: {p2:.3f} -> {p3:.3f}"

    # Entropy should be monotone non-increasing from stage 1 onward,
    # within an MCMC-noise tolerance.
    h = [r["entropy_bits"] for r in rows]
    for i in range(1, len(h)):
        assert h[i] <= h[i-1] + 0.08, \
            f"entropy must not increase beyond MCMC noise: stage {i-1}={h[i-1]:.3f} -> stage {i}={h[i]:.3f}"

    print(f"[PASS] progressive evidence: P(A heavier than C) "
          f"{p0:.3f} → {p1:.3f} → {p2:.3f} → {p3:.3f}  "
          f"(Plan §Phase 2 expected ~0.5 → ~0.85 → ~0.85 → ~0.97)")


def test_weight_gap_mean_increases_with_footprint():
    """A deeper footprint forces the inferred weight gap to widen."""
    rows = run_progressive_evidence(verbose=False, random_seed=0)
    gap_heights = [r for r in rows if r["name"] == "stage_1_heights"][0]["mean_weight_gap_kg"]
    gap_foot = [r for r in rows if r["name"] == "stage_3_deeper_footprint"][0]["mean_weight_gap_kg"]
    # Footprint of 0.15 cm with calibrated foot_area/soil ≈ 10.5 kg weight gap;
    # stage-1-only height gap ≈ 3 kg. At minimum, footprint should >1.5x widen.
    assert gap_foot > gap_heights * 1.5, \
        f"mean weight gap should widen with footprint evidence: "\
        f"{gap_heights:.2f} -> {gap_foot:.2f} kg"
    print(f"[PASS] weight-gap widens under footprint evidence: "
          f"heights-only {gap_heights:.2f} kg → + footprint {gap_foot:.2f} kg")


if __name__ == "__main__":
    print("=== Plan §Phase 2 multi-observable demo validation ===\n")
    test_progressive_evidence_tightens_posterior()
    test_weight_gap_mean_increases_with_footprint()
    print("\nAll Alice-Charlie chain demo checks passed.")
