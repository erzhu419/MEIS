"""P5.2 — Expert agreement on minimum-perturbation scoring (Plan §Phase 5 task 4).

Target: expert-agreement rate ≥ 70%.

Protocol:

  For each Phase 3 claim-ranking benchmark, an **authorial expert
  ranking** is declared (the ordering an informed reader would assign
  by ascending expected perturbation, reflecting physical / historical
  intuition). The system produces a ranking from D(h, B) = KL + λ·|Δ|
  via the P3 ClaimRankingEngine. Agreement is measured by **pairwise
  concordance**:

    concordance(expert, system) =
        #{(i, j): sign(expert_rank_i - expert_rank_j)
                  = sign(system_rank_i - system_rank_j)} / C(n, 2)

  This equals (1 + Kendall's τ) / 2 ∈ [0, 1]. Concordance 1.0 means
  identical orders (up to ties); 0.5 is chance; 0.0 is reverse order.

Honesty caveat: the "expert ranking" here is authorial — the person
who wrote each benchmark also declared the expected order. Multi-rater
expert annotation (e.g. historians evaluating the Eastern Han case) is
future work.

Reason this is still worth shipping: the benchmark fixtures were
built to reflect domain conventions (ritual-role priority in Noh,
political-stress dominance in Eastern Han), NOT to favor the scoring
function. The system's D(h, B) score is derived from PyMC KL drift
and BIC structural penalty — it does not "see" the expert ranking
during inference. So an agreement rate is still a real test of
whether the minimum-perturbation metric reproduces domain-consistent
orderings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from phase3_embedding.claim_ranking_engine import ClaimRankingEngine
from phase3_embedding.benchmarks import noh_theater, eastern_han


@dataclass
class AgreementResult:
    benchmark: str
    expert_order: list[str]
    system_order: list[str]
    pairwise_concordance: float
    exact_match: bool
    orphan_last: bool


def pairwise_concordance(expert_order: list[str], system_order: list[str]) -> float:
    if set(expert_order) != set(system_order):
        raise ValueError(
            f"name-set mismatch: expert {expert_order} vs system {system_order}"
        )
    expert_rank = {name: i for i, name in enumerate(expert_order)}
    system_rank = {name: i for i, name in enumerate(system_order)}
    names = list(expert_rank.keys())
    n = len(names)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            er = expert_rank[a] - expert_rank[b]
            sr = system_rank[a] - system_rank[b]
            # Signs differ from zero only when non-tied
            if (er > 0 and sr > 0) or (er < 0 and sr < 0) or (er == 0 and sr == 0):
                concordant += 1
            total += 1
    return concordant / total if total else 1.0


# ---------------------------------------------------------------------------
# Benchmarks — authorial expert rankings (ascending expected perturbation)
# ---------------------------------------------------------------------------


def _alice_charlie_spec():
    """Return (build_base_model, latent_var, get_claims, expert_order, orphan_name)."""
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie, _build_claims,
    )
    return dict(
        name="alice_charlie",
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        get_claims=_build_claims,
        expert_order=["H_larger_feet", "H_denser", "H_taller", "H_zodiac"],
        orphan_name="H_zodiac",
        bic_n=30,
    )


def _noh_spec():
    return dict(
        name="noh_theater",
        build_base_model=noh_theater.build_base_model,
        latent_var="p_banned",
        get_claims=noh_theater.get_claims,
        expert_order=["H_priestly", "H_shogunate_ban", "H_blood_taboo", "H_natural_voice"],
        orphan_name="H_natural_voice",
        bic_n=10,
    )


def _eastern_han_spec():
    return dict(
        name="eastern_han",
        build_base_model=eastern_han.build_base_model,
        latent_var="p_young_death",
        get_claims=eastern_han.get_claims,
        expert_order=["H_political_stress", "H_lead_poisoning",
                      "H_incest_bottleneck", "H_orphan_geomancy"],
        orphan_name="H_orphan_geomancy",
        bic_n=10,
    )


BENCHMARKS = [_alice_charlie_spec(), _noh_spec(), _eastern_han_spec()]


def evaluate_benchmark(bench: dict, draws: int = 1000, tune: int = 700,
                       seed: int = 0) -> AgreementResult:
    engine = ClaimRankingEngine(
        build_base_model=bench["build_base_model"],
        latent_var=bench["latent_var"],
        bic_n=bench["bic_n"],
        draws=draws, tune=tune, random_seed=seed,
    )
    scores = engine.rank(bench["get_claims"]())
    system_order = [s.claim.name for s in scores]
    concordance = pairwise_concordance(bench["expert_order"], system_order)
    return AgreementResult(
        benchmark=bench["name"],
        expert_order=bench["expert_order"],
        system_order=system_order,
        pairwise_concordance=concordance,
        exact_match=(system_order == bench["expert_order"]),
        orphan_last=(system_order[-1] == bench["orphan_name"]),
    )


def run_all_benchmarks(draws: int = 1000, tune: int = 700, seed: int = 0):
    results = [evaluate_benchmark(b, draws=draws, tune=tune, seed=seed)
               for b in BENCHMARKS]
    mean_concordance = sum(r.pairwise_concordance for r in results) / len(results)
    return results, mean_concordance


if __name__ == "__main__":
    print("P5.2 expert-agreement benchmark\n")
    results, mean = run_all_benchmarks()
    for r in results:
        match = "✓ exact" if r.exact_match else f"concordance {r.pairwise_concordance:.2f}"
        print(f"  {r.benchmark:<18}  expert={r.expert_order}")
        print(f"  {'':<18}  system={r.system_order}")
        print(f"  {'':<18}  {match}  orphan-last={r.orphan_last}")
        print()
    print(f"Mean pairwise concordance: {mean:.3f}  "
          f"(Plan §Phase 5 task 4 target ≥ 0.70)")
