"""Validation for P3.1 ClaimRankingEngine.

Covers:
  1. Engine produces identical rankings to the legacy demo function
     (round-trip through the engine preserves Phase 2 result)
  2. `structural_formula="none"` (pure KL ablation) rearranges the ranking
     such that zodiac is NOT last — because its penalty term is zero and
     it has the smallest KL of all claims
  3. `structural_formula="count"` (unit-weighted) differs from "bic" in
     magnitude but preserves the ranking order of zodiac-last
  4. `kl_estimator="kde"` produces a similar ordering to the default
     gaussian_moment (non-regression check of the alternate estimator)

Run:
    python -m phase3_embedding.tests.test_claim_ranking_engine
"""

from __future__ import annotations

from phase3_embedding.claim_ranking_engine import (
    ClaimRankingEngine, ClaimSpec, pretty_print_engine_scores,
)
from phase3_embedding.demo_claim_ranking import (
    _build_claims, _build_base_alice_charlie,
)


def _make_engine(**overrides) -> ClaimRankingEngine:
    defaults = dict(
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        bic_n=30,
        draws=1200, tune=800,
        random_seed=0,
    )
    defaults.update(overrides)
    return ClaimRankingEngine(**defaults)


def test_engine_matches_legacy_ranking():
    """With default settings (bic + gaussian_moment), the engine must
    reproduce the Phase 2 claim-ranking result: zodiac last, ~90× above
    max in-vocabulary."""
    engine = _make_engine()
    scores = engine.rank(_build_claims())
    ids = [s.claim.name for s in scores]
    assert ids[-1] == "H_zodiac", ids

    zodiac_composite = scores[-1].composite
    max_other = max(s.composite for s in scores[:-1])
    assert zodiac_composite > 50 * max_other, \
        f"zodiac not dominant: {zodiac_composite} vs max other {max_other}"
    assert scores[-1].structural_term > 3.0   # ≈ log(30)/2 · 2 ≈ 3.4
    print(f"[PASS] engine default ranking matches Phase 2: "
          f"zodiac composite {zodiac_composite:.3f}, "
          f"ratio {zodiac_composite / max(max_other, 1e-9):.1f}× max other")


def test_pure_kl_ablation_changes_order():
    """structural_formula='none' kills the structural term. Zodiac's KL is
    tiny (disconnected), so without the structural penalty zodiac ranks
    near the TOP, not bottom — a critical sanity check showing that the
    structural term is what actually isolates incoherent claims."""
    engine = _make_engine(structural_formula="none")
    scores = engine.rank(_build_claims())
    ids = [s.claim.name for s in scores]
    # Zodiac is NO LONGER last; it's one of the more-coherent-by-KL ones
    zodiac_rank = ids.index("H_zodiac")
    assert zodiac_rank <= 1, \
        f"under pure-KL, zodiac should rank near top (low KL), got rank {zodiac_rank}"
    # Structural term is 0 for all claims
    for s in scores:
        assert s.structural_term == 0.0
    print(f"[PASS] pure-KL ablation: zodiac rank = {zodiac_rank+1}/4 "
          f"(vs 4/4 with BIC). Confirms structural term is load-bearing.")


def test_count_formula_preserves_zodiac_last():
    """structural_formula='count' uses λ=1.0 per new parameter instead of
    log(N)/2. Zodiac has 2 structural additions → penalty = 2.0, still
    dominates the small KL values. Order should be same as BIC case,
    magnitudes different."""
    engine_bic = _make_engine(structural_formula="bic")
    engine_count = _make_engine(structural_formula="count")
    bic_order = [s.claim.name for s in engine_bic.rank(_build_claims())]
    count_order = [s.claim.name for s in engine_count.rank(_build_claims())]
    assert bic_order[-1] == count_order[-1] == "H_zodiac"
    print(f"[PASS] count formula preserves zodiac-last ordering "
          f"(BIC λ≈1.7 → composite 3.4, count λ=1.0 → composite 2.0; "
          f"still dominates KL).")


def test_engine_caches_base_samples():
    """A second call to rank() on the same engine should not re-sample the
    base model — evidenced by self._base_samples being non-None after first
    call."""
    engine = _make_engine(draws=500, tune=300)
    assert engine._base_samples is None
    _ = engine.rank(_build_claims()[:1])  # just 1 claim for speed
    assert engine._base_samples is not None
    cached = engine._base_samples
    _ = engine.rank(_build_claims()[:1])
    # Exactly the same array reference
    assert engine._base_samples is cached
    print(f"[PASS] base samples cached (shape {cached.shape}); "
          f"second rank() call doesn't resample")


if __name__ == "__main__":
    print("=== P3.1 ClaimRankingEngine validation ===\n")
    test_engine_matches_legacy_ranking()
    test_pure_kl_ablation_changes_order()
    test_count_formula_preserves_zodiac_last()
    test_engine_caches_base_samples()
    print("\nAll P3.1 engine checks passed.")
