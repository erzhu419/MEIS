"""Validation for P3.3 cross-metric ablation.

Plan §3 risk mitigation: "并列实现 3 种 (KL/BIC/edit-distance), 实证对比".

This test runs all 18 cells (3 benchmarks × 3 structural formulas ×
2 KL directions) and asserts:

  1. bic-formula orphan-last rate: 100%
  2. count-formula orphan-last rate: 100%
  3. none-formula (pure KL) orphan-last rate: 0%
  4. Composite ratio orphan / max-in-vocabulary under bic > 1 in every
     cell (meaningful separation, not borderline)

Together these establish that |Δstructure|-style structural penalty
is BOTH necessary (fail without it) AND sufficient (succeeds with
either BIC or count) to isolate orphan-node claims.

Run:
    python -m phase3_embedding.tests.test_metric_ablation
"""

from __future__ import annotations

from phase3_embedding.demo_metric_ablation import run_ablation, summarize_cells


def test_structural_penalty_is_load_bearing():
    cells = run_ablation(verbose=False)
    summary = summarize_cells(cells)
    assert summary["total_cells"] == 18, summary

    # BIC must rank orphan last in all cells
    bic_rate = summary["by_structural_formula"]["bic"]
    assert bic_rate == 1.0, f"BIC failed on some cell: rate {bic_rate}"

    # Count must rank orphan last in all cells
    count_rate = summary["by_structural_formula"]["count"]
    assert count_rate == 1.0, f"count failed on some cell: rate {count_rate}"

    # Pure-KL must FAIL to rank orphan last in every cell
    none_rate = summary["by_structural_formula"]["none"]
    assert none_rate == 0.0, f"pure-KL unexpectedly isolated orphan: rate {none_rate}"

    print(f"[PASS] structural-penalty dichotomy on 3 benchmarks × 2 KL directions:")
    print(f"       bic   → 100% orphan-last")
    print(f"       count → 100% orphan-last")
    print(f"       none  →   0% orphan-last")


def test_bic_ratio_separation():
    """Under bic, the orphan's composite must be strictly larger than every
    in-vocab claim in every cell (ratio > 1). The ratio's magnitude varies
    with benchmark: Alice-Charlie has very low in-vocab KLs (~0.04 max) so
    ratio ~75-90; Noh/Eastern-Han have in-vocab KLs up to ~1.8 so ratio is
    more modest (~1.2-3.7). All still > 1."""
    cells = run_ablation(verbose=False)
    bic_cells = [c for c in cells if c["structural_formula"] == "bic"]
    for c in bic_cells:
        assert c["ratio"] > 1.0, \
            f"BIC cell {c['benchmark']}/{c['kl_direction']} has ratio {c['ratio']}"
    # At least half of BIC cells should have ratio > 2 (large separation is
    # env-dependent; Alice-Charlie gets 70-90×, social-science ones get
    # 1.2-3.7×).
    high_ratio = sum(1 for c in bic_cells if c["ratio"] > 2.0)
    assert high_ratio >= 3, f"only {high_ratio}/6 BIC cells have ratio > 2"
    ratios = [round(c["ratio"], 2) for c in bic_cells]
    print(f"[PASS] BIC cells: all 6 have orphan-composite > max-in-vocab; "
          f"{high_ratio}/6 have ratio > 2. Ratios: {ratios}")


def test_kl_direction_does_not_flip_ordering():
    """Ranking of orphan should be the same under base_to_hyp and
    hyp_to_base for the SAME benchmark × SAME structural formula. (This
    sanity-checks the engine's KL-direction plumbing.)"""
    cells = run_ablation(verbose=False)
    pairs: dict[tuple, list] = {}
    for c in cells:
        key = (c["benchmark"], c["structural_formula"])
        pairs.setdefault(key, []).append(c)
    for key, two in pairs.items():
        assert len(two) == 2
        a, b = two
        assert a["orphan_last"] == b["orphan_last"], \
            f"KL direction flipped orphan_last on {key}: {a['orphan_last']} vs {b['orphan_last']}"
    print(f"[PASS] KL direction (base→hyp vs hyp→base) preserves orphan_last "
          f"for all 9 (benchmark × formula) combos")


if __name__ == "__main__":
    print("=== P3.3 cross-metric ablation validation ===\n")
    test_structural_penalty_is_load_bearing()
    print()
    test_bic_ratio_separation()
    print()
    test_kl_direction_does_not_flip_ordering()
    print("\nAll ablation checks passed.")
