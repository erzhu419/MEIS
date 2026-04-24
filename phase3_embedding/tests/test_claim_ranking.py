"""Validation for the Plan §0.2 claim-ranking demo.

Run:
    python -m phase3_embedding.tests.test_claim_ranking
"""

from __future__ import annotations

import math

from phase3_embedding.demo_claim_ranking import rank_claims


def test_zodiac_ranks_last_with_large_structural_penalty():
    """Honest version: zodiac is encoded as 2 orphan Categorical nodes in a
    real PyMC model, so its KL on weight_A posterior is ≈ 0 (disconnected).
    The BIC structural penalty (log(30)/2 · 2 ≈ 3.4) is what makes it
    dominate — this tests the MECHANISM of the composite score, not a
    hardcoded 'huge number for inaccessible claim' fallback."""
    scores = rank_claims(verbose=False)
    assert len(scores) == 4, len(scores)

    ids = [s.claim.name for s in scores]
    assert ids[-1] == "H_zodiac", \
        f"H_zodiac should rank LAST (least coherent), got {ids}"

    zodiac = scores[-1]
    rest = scores[:-1]

    # Zodiac KL on weight_A should be small (< 0.5) — zodiac is disconnected
    # from weight so observing zodiac values doesn't move theta's posterior.
    assert zodiac.kl_drift < 0.5, \
        f"zodiac KL should be ~0 (disconnected from weight), got {zodiac.kl_drift}"

    # Zodiac's DOMINANCE comes from the structural term (BIC for 2 orphan
    # vars ≈ 3.4), not from KL. Test that structural_term is the main driver.
    assert zodiac.structural_term > 3.0, zodiac.structural_term
    assert zodiac.claim.structural_additions >= 2
    # Structural term > 5× KL term, i.e. structure dominates
    assert zodiac.structural_term > 5 * zodiac.kl_drift

    # Composite dominance: at least 20× any in-vocabulary claim
    max_other_D = max(s.composite for s in rest)
    assert zodiac.composite > 20 * max_other_D, \
        f"zodiac D should dominate by 20×: zodiac={zodiac.composite:.3f}, "\
        f"max_other={max_other_D:.3f}"

    # All in-vocabulary claims have zero structural term
    for s in rest:
        assert s.structural_term == 0.0, \
            f"{s.claim.name} should have no structural additions"

    print(f"[PASS] zodiac ranks last with honest BIC-derived penalty. "
          f"KL={zodiac.kl_drift:.4f} (disconnected → ~0), "
          f"structural={zodiac.structural_term:.3f}, "
          f"composite={zodiac.composite:.3f}, ratio {zodiac.composite/max(max_other_D, 1e-9):.1f}x")


def test_in_vocabulary_claims_embed_with_low_perturbation():
    """All three in-vocabulary claims (taller / denser / larger_feet) should
    produce finite, small KL drift — they each embed coherently into the
    belief network."""
    scores = rank_claims(verbose=False)
    vocab = [s for s in scores if s.claim.structural_additions == 0]
    assert len(vocab) == 3
    for s in vocab:
        assert 0.0 <= s.kl_drift < 1.0, \
            f"{s.claim.name} KL should be small/moderate but finite; "\
            f"got {s.kl_drift}"
    print(f"[PASS] 3 in-vocabulary claims all have KL < 1.0: "
          f"{[(s.claim.name, round(s.kl_drift, 4)) for s in vocab]}")


if __name__ == "__main__":
    print("=== Plan §0.2 claim-ranking demo validation ===\n")
    test_zodiac_ranks_last_with_large_structural_penalty()
    test_in_vocabulary_claims_embed_with_low_perturbation()
    print("\nAll claim-ranking checks passed.")
