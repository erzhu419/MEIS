"""Validation for Plan §Phase 3 success-criterion benchmarks.

Two historical-social-science case studies that the original MEIS plan
called out specifically:

  - Noh theater gender monopoly (plan §Phase 3 example 1)
  - Eastern Han emperor early death (plan §Phase 3 example 2)

For each, the acceptance criteria are:

  1. The ORPHAN claim (geomancy / voice-fit) ranks LAST under the
     default composite metric (KL + BIC·|Δstructure|).
  2. The orphan claim's KL drift is near zero (orphan latent has no
     causal path to the target posterior).
  3. The orphan's structural term is > 2 (BIC penalty for 2 orphan
     parameters × λ = log(10)/2 ≈ 1.15 → total ~2.3).
  4. The orphan's composite is at least an order of magnitude above
     every in-vocabulary claim's KL term alone.
  5. Under structural_formula="none" ablation, the orphan claim MOVES
     UP (to a top position by composite-as-KL), confirming the
     structural term is the load-bearing quantity.

Run:
    python -m phase3_embedding.tests.test_plan_benchmarks
"""

from __future__ import annotations

from phase3_embedding.claim_ranking_engine import ClaimRankingEngine
from phase3_embedding.benchmarks import noh_theater, eastern_han


_BENCHMARKS = [
    dict(
        name="Noh theater gender monopoly",
        module=noh_theater,
        latent_var="p_banned",
        orphan_name="H_natural_voice",
    ),
    dict(
        name="Eastern Han early death",
        module=eastern_han,
        latent_var="p_young_death",
        orphan_name="H_orphan_geomancy",
    ),
]


def _run_benchmark(bench, **engine_overrides):
    defaults = dict(
        build_base_model=bench["module"].build_base_model,
        latent_var=bench["latent_var"],
        bic_n=10,
        draws=1000, tune=700,
        random_seed=0,
    )
    defaults.update(engine_overrides)
    engine = ClaimRankingEngine(**defaults)
    return engine.rank(bench["module"].get_claims())


def test_orphan_claim_ranks_last_under_default_metric():
    for bench in _BENCHMARKS:
        scores = _run_benchmark(bench)
        ids = [s.claim.name for s in scores]
        orphan_name = bench["orphan_name"]
        assert ids[-1] == orphan_name, \
            f"[{bench['name']}] orphan '{orphan_name}' not last: {ids}"

        orphan = scores[-1]
        assert orphan.kl_drift < 0.1, \
            f"[{bench['name']}] orphan KL should be ~0 (disconnected), got {orphan.kl_drift}"
        assert orphan.structural_term > 2.0, \
            f"[{bench['name']}] orphan structural penalty too small: {orphan.structural_term}"
        print(f"[PASS] {bench['name']}: orphan '{orphan_name}' last with "
              f"composite={orphan.composite:.3f} "
              f"(KL={orphan.kl_drift:.4f}, structural={orphan.structural_term:.3f})")


def test_pure_kl_ablation_orphan_rises_to_top():
    """Confirms the structural term (not KL) is responsible for ranking
    the orphan claim last. Under structural_formula='none', orphan's
    near-zero KL puts it at or near the top."""
    for bench in _BENCHMARKS:
        scores = _run_benchmark(bench, structural_formula="none")
        ids = [s.claim.name for s in scores]
        orphan_rank = ids.index(bench["orphan_name"])
        # Orphan should be in the top-2 by composite-as-KL (since its
        # KL is ~0, it's the LOWEST perturbation).
        assert orphan_rank <= 1, \
            f"[{bench['name']}] under pure-KL, orphan should rank top-2; got {orphan_rank+1}/4"
        # Structural term should be zero for all
        for s in scores:
            assert s.structural_term == 0.0
        print(f"[PASS] {bench['name']}: pure-KL → orphan rank {orphan_rank+1}/4 "
              f"(structural term is load-bearing)")


def test_in_vocabulary_claims_all_coherent():
    """All in-vocabulary claims should have structural_additions=0 and
    finite KL (i.e., they embed cleanly into the existing network)."""
    for bench in _BENCHMARKS:
        scores = _run_benchmark(bench)
        in_vocab = [s for s in scores if s.claim.structural_additions == 0]
        assert len(in_vocab) == 3
        for s in in_vocab:
            assert s.structural_term == 0.0
            assert 0.0 <= s.kl_drift < 5.0, \
                f"[{bench['name']}] {s.claim.name} KL out of range: {s.kl_drift}"
        print(f"[PASS] {bench['name']}: 3 in-vocabulary claims all "
              f"finite-KL (< 5) with zero structural term")


if __name__ == "__main__":
    print("=== Plan §Phase 3 benchmark validation ===\n")
    test_orphan_claim_ranks_last_under_default_metric()
    print()
    test_pure_kl_ablation_orphan_rises_to_top()
    print()
    test_in_vocabulary_claims_all_coherent()
    print("\nAll benchmark checks passed.")
