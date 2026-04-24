"""Validation for Paper 2 §4.4 — cross-family LLM expert panel.

Acceptance:
  1. All 4 raters (2 OpenAI + 2 Google) return valid rankings on
     all 3 benchmarks.
  2. Inter-rater mean Kendall τ ≥ 0.5 on every benchmark
     (reasonable expert-panel agreement proxy).
  3. Orphan-last fraction = 1.00 on every benchmark (12/12
     rater-benchmark pairs place the orphan last).
  4. D-ranking agrees with panel direction: orphan is placed LAST
     in both panel and D on all benchmarks.

Network-dependent: calls ruoli.dev with cached responses. If cached
JSON exists, uses it; otherwise calls the live API.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase3_embedding.llm_expert_panel import (
    run_all, compare_to_D, BENCHMARKS,
)


def test_all_raters_succeed_on_all_benchmarks():
    results = run_all(verbose=False)
    assert len(results) == 3
    for r in results:
        assert len(r.per_rater_rankings) == 4, \
            f"{r.benchmark}: only {len(r.per_rater_rankings)} raters"
    print(f"[PASS] all 4 raters succeeded on all 3 benchmarks ({4*3}/12)")


def test_inter_rater_agreement():
    results = run_all(verbose=False)
    for r in results:
        assert r.inter_rater_tau_mean >= 0.5, \
            f"{r.benchmark}: mean inter-rater τ = {r.inter_rater_tau_mean}"
    mean_across = sum(r.inter_rater_tau_mean for r in results) / len(results)
    print(f"[PASS] inter-rater mean τ ≥ 0.5 on every benchmark; "
          f"cross-benchmark mean = {mean_across:+.3f}")


def test_orphan_last_across_panel():
    results = run_all(verbose=False)
    bench_to_orphan = {b["name"]: b["orphan"] for b in BENCHMARKS}
    total = 0
    orphan_last = 0
    for r in results:
        for rater_name, ranking in r.per_rater_rankings.items():
            total += 1
            if ranking[-1] == bench_to_orphan[r.benchmark]:
                orphan_last += 1
    frac = orphan_last / total
    assert frac == 1.0, f"orphan-last fraction {frac} < 1.00 "
    print(f"[PASS] orphan-last across 4 raters × 3 benchmarks: "
          f"{orphan_last}/{total} ({100*frac:.0f}%)")


def test_D_and_panel_both_orphan_last():
    results = run_all(verbose=False)
    comparisons = compare_to_D(results, verbose=False)
    bench_to_orphan = {b["name"]: b["orphan"] for b in BENCHMARKS}
    for c in comparisons:
        assert c["d_ranking"][-1] == bench_to_orphan[c["benchmark"]]
        assert c["panel_ranking"][-1] == bench_to_orphan[c["benchmark"]]
    mean_d_panel = sum(c["tau_d_panel"] for c in comparisons) / len(comparisons)
    print(f"[PASS] D and panel BOTH put orphan last on 3/3 benchmarks; "
          f"mean D-vs-panel τ = {mean_d_panel:+.3f}")


if __name__ == "__main__":
    print("=== Paper 2 §4.4 cross-family LLM panel validation ===\n")
    test_all_raters_succeed_on_all_benchmarks()
    test_inter_rater_agreement()
    test_orphan_last_across_panel()
    test_D_and_panel_both_orphan_last()
    print("\nAll LLM panel checks passed.")
