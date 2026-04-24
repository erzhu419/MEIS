"""Paper 2 Phase C — λ sensitivity of the structural penalty.

For each benchmark, sweep λ across a range and check whether the
orphan-last ordering persists. The theoretical boundary is
λ > κ* / |Δ_h|, where κ* is the maximum in-vocabulary KL drift.
Empirically we show the ordering is stable for λ ≥ κ*/2.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from phase3_embedding.claim_ranking_engine import ClaimRankingEngine


@dataclass
class SensitivityRow:
    benchmark: str
    lambda_value: float
    orphan_last: bool
    orphan_composite: float
    max_in_vocab_composite: float
    ratio: float   # orphan / max in-vocab


def _alice_spec():
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie, _build_claims,
    )
    return dict(
        name="alice_charlie",
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        claims=_build_claims(),
        orphan="H_zodiac",
    )


def _noh_spec():
    from phase3_embedding.benchmarks import noh_theater
    return dict(
        name="noh_theater",
        build_base_model=noh_theater.build_base_model,
        latent_var="p_banned",
        claims=noh_theater.get_claims(),
        orphan="H_natural_voice",
    )


def _eastern_han_spec():
    from phase3_embedding.benchmarks import eastern_han
    return dict(
        name="eastern_han",
        build_base_model=eastern_han.build_base_model,
        latent_var="p_young_death",
        claims=eastern_han.get_claims(),
        orphan="H_orphan_geomancy",
    )


def run_lambda_sweep(lambda_grid: list[float] | None = None,
                      seed: int = 0,
                      draws: int = 500, tune: int = 400):
    """Sweep λ across a geometric grid and record orphan-last status."""
    if lambda_grid is None:
        lambda_grid = [0.0, 0.25, 0.5, 1.0, 1.7, 3.0, 5.0, 10.0]
    specs = [_alice_spec(), _noh_spec(), _eastern_han_spec()]

    rows = []
    for spec in specs:
        for lam in lambda_grid:
            # Rebuild engine per λ: we override structural_cost_override
            # on each ClaimSpec to use lam · structural_additions.
            scored_claims = []
            for c in spec["claims"]:
                c_copy = type(c)(
                    name=c.name, summary=c.summary,
                    build_model=c.build_model,
                    structural_additions=c.structural_additions,
                    structural_cost_override=lam * c.structural_additions,
                )
                scored_claims.append(c_copy)
            # Use structural_formula='none' so the engine uses our
            # overrides as-is (the override bypasses BIC/count logic).
            engine = ClaimRankingEngine(
                build_base_model=spec["build_base_model"],
                latent_var=spec["latent_var"],
                bic_n=10,   # unused since formula=none + override
                structural_formula="none",
                draws=draws, tune=tune, random_seed=seed,
            )
            scores = engine.rank(scored_claims)
            names_ordered = [s.claim.name for s in scores]
            orphan_idx = names_ordered.index(spec["orphan"])
            orphan_score = scores[orphan_idx].composite
            in_vocab_scores = [s.composite for s in scores
                                if s.claim.name != spec["orphan"]]
            max_in_vocab = max(in_vocab_scores)
            ratio = orphan_score / max(max_in_vocab, 1e-9)
            rows.append(SensitivityRow(
                benchmark=spec["name"],
                lambda_value=lam,
                orphan_last=(names_ordered[-1] == spec["orphan"]),
                orphan_composite=orphan_score,
                max_in_vocab_composite=max_in_vocab,
                ratio=ratio,
            ))
    return rows


def kappa_star_per_benchmark(seed: int = 0, draws: int = 600, tune: int = 500):
    """For each benchmark compute κ* = max_{h': Δ=0} D_KL(θ* | h'),
    which sets the theoretical threshold λ > κ* / |Δ_orphan| for
    orphan-last under Proposition 1."""
    specs = [_alice_spec(), _noh_spec(), _eastern_han_spec()]
    out = {}
    for spec in specs:
        engine = ClaimRankingEngine(
            build_base_model=spec["build_base_model"],
            latent_var=spec["latent_var"],
            bic_n=10,
            structural_formula="none",
            draws=draws, tune=tune, random_seed=seed,
        )
        scores = engine.rank(spec["claims"])
        in_vocab_kls = [s.kl_drift for s in scores
                         if s.claim.name != spec["orphan"]]
        kappa_star = max(in_vocab_kls)
        orphan_delta = next(s.claim.structural_additions for s in scores
                             if s.claim.name == spec["orphan"])
        out[spec["name"]] = dict(
            kappa_star=kappa_star,
            orphan_delta=orphan_delta,
            threshold_lambda=kappa_star / max(orphan_delta, 1),
        )
    return out


if __name__ == "__main__":
    print("Paper 2 Phase C — λ sensitivity sweep\n")
    print("=== κ* thresholds per benchmark (λ > κ*/|Δ| ⇒ orphan-last) ===")
    thr = kappa_star_per_benchmark()
    for name, d in thr.items():
        print(f"  {name:<15}  κ* = {d['kappa_star']:.3f},  "
              f"|Δ| = {d['orphan_delta']},  "
              f"λ-threshold = {d['threshold_lambda']:.3f}")

    print("\n=== orphan-last outcome across λ grid ===")
    rows = run_lambda_sweep()
    header = "  {:<15}  {:>7}  {:>9}  {:>10}  {:>10}  {:>6}".format(
        "benchmark", "λ", "orphan-D", "in-vocab", "ratio", "last?")
    print(header)
    print("  " + "-" * len(header))
    for r in rows:
        flag = "✓" if r.orphan_last else "✗"
        print("  {:<15}  {:>7.2f}  {:>9.3f}  {:>10.3f}  {:>10.2f}  {:>6}".format(
            r.benchmark, r.lambda_value, r.orphan_composite,
            r.max_in_vocab_composite, r.ratio, flag))
