"""Validation for P5.2 — Plan §Phase 5 task 4.

Acceptance:
    1. pairwise_concordance is 1.0 for identical orders, 0.5 for
       random-pair single-swap, 0.0 for full reverse.
    2. Each of the 3 Phase 3 benchmarks yields pairwise concordance
       ≥ 0.70 (Plan target) against its authorial expert ranking.
    3. Mean concordance across all 3 benchmarks ≥ 0.70.
    4. Orphan claim ranks last in all 3 benchmarks (sanity — this is
       already covered by phase3_embedding tests but re-asserted here
       for the task 4 deliverable).

Run:
    python -m phase5_evaluation.tests.test_task4_expert_agreement
"""

from __future__ import annotations

from phase5_evaluation.task4_expert_agreement import (
    pairwise_concordance, run_all_benchmarks, BENCHMARKS,
)


def test_pairwise_concordance_sanity():
    a = ["x", "y", "z", "w"]
    assert pairwise_concordance(a, a) == 1.0
    # full reverse → all 6 pairs discordant
    assert pairwise_concordance(a, list(reversed(a))) == 0.0
    # one adjacent swap → 5 concordant, 1 discordant (adjacent swap
    # only flips 1 pair)
    swapped = ["y", "x", "z", "w"]
    assert pairwise_concordance(a, swapped) == 5.0 / 6.0
    print(f"[PASS] pairwise_concordance: identical→1.0, reverse→0.0, "
          f"one adjacent swap→{5.0/6.0:.3f}")


def test_each_benchmark_hits_plan_target():
    results, _ = run_all_benchmarks(draws=800, tune=500, seed=0)
    for r in results:
        assert r.pairwise_concordance >= 0.70, \
            f"{r.benchmark}: concordance {r.pairwise_concordance} < 0.70"
        print(f"[PASS] {r.benchmark:<14} concordance={r.pairwise_concordance:.3f}  "
              f"{'(exact match)' if r.exact_match else ''}")


def test_mean_agreement_hits_plan_target():
    _, mean = run_all_benchmarks(draws=800, tune=500, seed=0)
    assert mean >= 0.70, f"mean concordance {mean} < 0.70"
    print(f"[PASS] mean pairwise concordance across 3 benchmarks: {mean:.3f}  "
          f"(Plan §Phase 5 task 4 target ≥ 0.70)")


def test_orphan_last_all_benchmarks():
    results, _ = run_all_benchmarks(draws=800, tune=500, seed=0)
    for r in results:
        assert r.orphan_last, f"{r.benchmark}: orphan not last in {r.system_order}"
    print(f"[PASS] orphan claim ranks last in all {len(results)} Phase 3 benchmarks")


if __name__ == "__main__":
    print("=== P5.2 expert-agreement validation ===\n")
    test_pairwise_concordance_sanity()
    test_each_benchmark_hits_plan_target()
    test_mean_agreement_hits_plan_target()
    test_orphan_last_all_benchmarks()
    print("\nAll P5.2 expert-agreement checks passed.")
