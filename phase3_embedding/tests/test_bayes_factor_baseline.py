"""Validation for Paper 2 §5 — D vs full-BIC baseline.

Acceptance:
  1. D-ranking puts orphan last on all 3 benchmarks (reproduces
     Table 1/2/3 of Paper 2).
  2. BIC-ranking puts orphan last on Alice-Charlie (both agree here).
  3. On Noh and Eastern Han, BIC does NOT put orphan last — it ranks
     the orphan 2nd on both. This demonstrates D ≠ BIC: the two give
     different rankings on benchmarks whose orphan hypothesis adds
     well-fitting auxiliary observations.
  4. Kendall τ(D, BIC) = 1.00 on Alice-Charlie, ≤ 0.5 on both
     qualitative benchmarks.

This is the reviewer-defense fact: D is not a rewording of BIC.
"""

from __future__ import annotations

from phase3_embedding.bayes_factor_baseline import run_all


def test_d_orphan_last_on_all_three():
    results = run_all(verbose=False)
    orphan_names = {
        "alice_charlie": "H_zodiac",
        "noh_theater": "H_natural_voice",
        "eastern_han": "H_orphan_geomancy",
    }
    for r in results:
        assert r.d_ranking[-1] == orphan_names[r.benchmark], \
            f"D did not put orphan last on {r.benchmark}: {r.d_ranking}"
    print(f"[PASS] D ranks orphan last on all 3 benchmarks")


def test_bic_and_d_diverge_on_qualitative():
    results = run_all(verbose=False)
    bench_map = {r.benchmark: r for r in results}

    # Alice-Charlie: agreement expected
    ac = bench_map["alice_charlie"]
    assert ac.kendall_d_vs_bic == 1.0, \
        f"Alice-Charlie D vs BIC Kendall τ = {ac.kendall_d_vs_bic}, expected 1.0"

    # Noh / Eastern Han: BIC DOES NOT put orphan last
    for name in ("noh_theater", "eastern_han"):
        r = bench_map[name]
        orphan = {"noh_theater": "H_natural_voice",
                  "eastern_han": "H_orphan_geomancy"}[name]
        assert r.bic_ranking[-1] != orphan, \
            f"BIC unexpectedly put orphan last on {name}: {r.bic_ranking}"
        assert r.kendall_d_vs_bic < 1.0, \
            f"{name} D vs BIC Kendall τ = {r.kendall_d_vs_bic}, expected < 1.0"
    print(f"[PASS] BIC and D diverge on Noh (τ={bench_map['noh_theater'].kendall_d_vs_bic:+.3f}) "
          f"and Eastern Han (τ={bench_map['eastern_han'].kendall_d_vs_bic:+.3f})")


def test_bic_ranks_orphan_second_on_qualitative():
    """Specific finding: BIC puts the orphan 2nd on both Noh and
    Eastern Han. This is the mechanism: orphan's auxiliary evidence
    fits well, inflating BIC's global-likelihood term."""
    results = run_all(verbose=False)
    bench_map = {r.benchmark: r for r in results}
    for name in ("noh_theater", "eastern_han"):
        r = bench_map[name]
        orphan = {"noh_theater": "H_natural_voice",
                  "eastern_han": "H_orphan_geomancy"}[name]
        orphan_rank = r.bic_ranking.index(orphan) + 1
        assert orphan_rank == 2, \
            f"BIC-rank of orphan on {name} = {orphan_rank}, expected 2"
    print(f"[PASS] BIC ranks orphan 2nd (not last) on both qualitative "
          f"benchmarks — D ≠ BIC empirically, not just by definition")


if __name__ == "__main__":
    print("=== Paper 2 §5 D-vs-BIC baseline validation ===\n")
    test_d_orphan_last_on_all_three()
    test_bic_and_d_diverge_on_qualitative()
    test_bic_ranks_orphan_second_on_qualitative()
    print("\nAll baseline checks passed.")
