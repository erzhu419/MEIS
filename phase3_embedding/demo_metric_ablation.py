"""MEIS Phase 3 P3.3 — cross-metric ablation study.

Plan §3 risk-mitigation: "最小扰动度量选型分歧 … 并列实现 3 种 (KL / BIC /
图编辑距离)，实证对比".

This demo runs each of 3 benchmarks (Alice-Charlie, Noh, Eastern-Han)
under 3 structural-formula settings (bic / count / none) and 2
KL-direction choices (base→hyp, hyp→base), producing a 3×6 table.

Headline quantity: does the orphan claim rank LAST in every
metric configuration? If yes, the result is robust to metric choice;
if no, we've identified which metrics actually isolate incoherent
claims vs which don't.

Pure PyMC, 0 API.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from phase3_embedding.claim_ranking_engine import ClaimRankingEngine
from phase3_embedding.benchmarks import noh_theater, eastern_han


@dataclass
class BenchmarkSpec:
    name: str
    build_base_model: object
    latent_var: str
    get_claims: object
    orphan_name: str
    bic_n: int


def _alice_charlie_spec() -> BenchmarkSpec:
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie, _build_claims,
    )
    return BenchmarkSpec(
        name="alice_charlie",
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        get_claims=_build_claims,
        orphan_name="H_zodiac",
        bic_n=30,
    )


def _noh_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="noh_theater",
        build_base_model=noh_theater.build_base_model,
        latent_var="p_banned",
        get_claims=noh_theater.get_claims,
        orphan_name="H_natural_voice",
        bic_n=10,
    )


def _eastern_han_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="eastern_han",
        build_base_model=eastern_han.build_base_model,
        latent_var="p_young_death",
        get_claims=eastern_han.get_claims,
        orphan_name="H_orphan_geomancy",
        bic_n=10,
    )


BENCHMARKS = [_alice_charlie_spec(), _noh_spec(), _eastern_han_spec()]
STRUCTURAL_FORMULAS = ["bic", "count", "none"]
KL_DIRECTIONS = ["base_to_hyp", "hyp_to_base"]


def run_ablation(verbose: bool = True):
    """Run each (benchmark, structural_formula, kl_direction) cell,
    record whether the orphan ranks last and the composite ratio
    vs max in-vocabulary claim."""
    cells = []
    for bench in BENCHMARKS:
        for sf in STRUCTURAL_FORMULAS:
            for kd in KL_DIRECTIONS:
                engine = ClaimRankingEngine(
                    build_base_model=bench.build_base_model,
                    latent_var=bench.latent_var,
                    bic_n=bench.bic_n,
                    structural_formula=sf,
                    kl_direction=kd,
                    draws=1000, tune=600, random_seed=0,
                )
                scores = engine.rank(bench.get_claims())
                ids = [s.claim.name for s in scores]
                orphan = next(s for s in scores if s.claim.name == bench.orphan_name)
                orphan_rank = ids.index(bench.orphan_name) + 1  # 1-indexed
                rest_max = max(s.composite for s in scores if s.claim.name != bench.orphan_name)
                cells.append({
                    "benchmark": bench.name,
                    "structural_formula": sf,
                    "kl_direction": kd,
                    "orphan_rank": orphan_rank,
                    "orphan_composite": orphan.composite,
                    "orphan_kl": orphan.kl_drift,
                    "orphan_struct": orphan.structural_term,
                    "max_rest": rest_max,
                    "ratio": orphan.composite / max(rest_max, 1e-9),
                    "orphan_last": orphan_rank == len(scores),
                })

    if verbose:
        print(f"{'benchmark':<15}  {'structural':>12}  {'kl dir':>14}  "
              f"{'orphan_rank':>11}  {'composite':>10}  {'ratio vs rest':>14}")
        print("-" * 92)
        for c in cells:
            marker = "✓ last" if c["orphan_last"] else f"rank {c['orphan_rank']}"
            print(f"{c['benchmark']:<15}  {c['structural_formula']:>12}  "
                  f"{c['kl_direction']:>14}  {marker:>11}  "
                  f"{c['orphan_composite']:10.3f}  {c['ratio']:14.2f}")

    return cells


def summarize_cells(cells):
    """Summary statistics across all cells."""
    total = len(cells)
    orphan_last = sum(c["orphan_last"] for c in cells)
    by_formula = {}
    for c in cells:
        key = c["structural_formula"]
        by_formula.setdefault(key, []).append(c["orphan_last"])
    return dict(
        total_cells=total,
        orphan_last_count=orphan_last,
        orphan_last_rate=orphan_last / total,
        by_structural_formula={
            k: sum(v) / len(v) for k, v in by_formula.items()
        },
    )


if __name__ == "__main__":
    print("MEIS Phase 3 P3.3 — metric ablation (3 benchmarks × 3 formulas × 2 directions)\n")
    cells = run_ablation(verbose=True)
    summary = summarize_cells(cells)
    print()
    print(f"Total cells: {summary['total_cells']}")
    print(f"Orphan ranks LAST in: {summary['orphan_last_count']} / {summary['total_cells']} "
          f"({summary['orphan_last_rate']*100:.0f}%)")
    print()
    print("Rate by structural formula:")
    for k, v in summary["by_structural_formula"].items():
        print(f"  {k:>10}: {v*100:.0f}%")
    print()
    print("Plan §3 mitigation ('实证对比'): the structural formula is the load-")
    print("bearing quantity for correct orphan-isolation. 'none' (pure-KL) should")
    print("fail in every cell (0%); 'bic' and 'count' should succeed in every cell.")
