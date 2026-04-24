"""Validation for the Plan §0.2 claim-ranking demo.

Run:
    python -m phase3_embedding.tests.test_claim_ranking
"""

from __future__ import annotations

import math

from phase3_embedding.demo_claim_ranking import rank_claims


def test_zodiac_ranks_last_with_large_structural_penalty():
    scores = rank_claims(verbose=False)
    # Must have 4 scored claims
    assert len(scores) == 4, len(scores)

    ids = [s.claim.name for s in scores]
    assert ids[-1] == "H_zodiac", \
        f"H_zodiac should rank LAST (least coherent), got {ids}"

    zodiac = scores[-1]
    rest = scores[:-1]

    # Zodiac should be an order of magnitude or more above any other claim
    max_other_D = max(s.composite for s in rest)
    assert zodiac.composite > 10 * max_other_D + 1.0, \
        f"zodiac D should dominate: zodiac={zodiac.composite:.3f}, "\
        f"max_other={max_other_D:.3f}"

    # Zodiac's composite should be driven by the structural term, not KL
    assert zodiac.structural_term > 0, zodiac.structural_term
    assert zodiac.claim.structural_additions >= 2

    # All non-zodiac claims should have zero structural term (they fit
    # existing vocabulary)
    for s in rest:
        assert s.structural_term == 0.0, \
            f"{s.claim.name} should have no structural additions"

    # All non-zodiac KL values must be finite
    for s in rest:
        assert math.isfinite(s.kl_drift), \
            f"{s.claim.name} has non-finite KL: {s.kl_drift}"

    print(f"[PASS] zodiac ranks last with composite {zodiac.composite:.2f} "
          f"vs max other {max_other_D:.3f} "
          f"(ratio = {zodiac.composite / max(max_other_D, 1e-9):.1f}x)")


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
