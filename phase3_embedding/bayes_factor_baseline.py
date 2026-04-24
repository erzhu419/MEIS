"""Full-BIC baseline for D(h, B) — Paper 2 §5.

Runs the classical Bayesian Information Criterion (via MAP + free-RV
count + observed-data count) on each of the Phase 3 benchmarks
(Alice-Charlie, Noh theater, Eastern Han) and produces a ranking of
the 4 candidate hypotheses per benchmark, then compares that ranking
to D's ranking.

Purpose: honestly answer the reviewer question "how does D differ
from standard Bayesian model selection?" The answer turns out to be:
the qualitative ranking agrees on the orphan case (both put orphan
last) but D decomposes the penalty into two interpretable additive
terms, whereas BIC hides the structural cost inside the parameter
count and the fit cost inside -2 log L̂.

Note on Bayes-factor estimation via SMC: we attempted
`pm.sample_smc` for log marginal likelihood, but it fails with
singular-covariance errors on the multi-latent Alice-Charlie model.
BIC is the more robust baseline for heterogeneous model families
and is what we report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings
import numpy as np
import pymc as pm

from phase3_embedding.claim_ranking_engine import ClaimRankingEngine
from phase3_embedding.benchmarks import noh_theater, eastern_han


@dataclass
class ModelSelectionScore:
    claim_name: str
    bic: float                          # lower = better
    num_params: int
    num_observations: int
    map_loglik: float                   # MAP log likelihood of observations


@dataclass
class BenchmarkComparison:
    benchmark: str
    candidates: list                     # list of ClaimSpec names in order
    d_ranking: list                      # names by ascending D
    bic_ranking: list                    # names by ascending BIC
    d_scores: list[float]
    bic_scores: list[float]
    kendall_d_vs_bic: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_free_params(model) -> int:
    """Count free random variables as scalar latents (multi-d variables
    counted by their shape product)."""
    n = 0
    for rv in model.free_RVs:
        shape = rv.shape.eval() if hasattr(rv.shape, "eval") else tuple(rv.shape)
        if len(shape) == 0:
            n += 1
        else:
            n += int(np.prod(shape))
    return n


def _count_observations(model) -> int:
    """Total number of scalar observations across all observed RVs."""
    n = 0
    for rv in model.observed_RVs:
        data = rv.observations if hasattr(rv, "observations") else rv.owner.inputs[-1]
        try:
            arr = np.asarray(data.eval())
        except Exception:
            arr = np.atleast_1d(np.asarray(data))
        n += int(np.prod(arr.shape)) if arr.size > 1 else 1
    return n


def _bic_from_map(build_model, seed: int = 0) -> tuple[float, int, int, float]:
    """Compute BIC = -2 log p(D|θ_MAP) + k log N via MAP + introspection.

    Returns (bic, k, N, map_loglik).
    """
    model = build_model()
    with model:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            map_est = pm.find_MAP(progressbar=False, seed=seed)
    k = _count_free_params(model)
    N = max(_count_observations(model), 1)
    # Filter map_est down to only the value-var names the compiled fn
    # accepts; PyMC's find_MAP returns both transformed and untransformed
    # keys but point_logps' compile_fn wants exactly value_vars.
    accepted_keys = {v.name for v in model.value_vars}
    filtered = {k: v for k, v in map_est.items() if k in accepted_keys}
    with model:
        per_var_logps = model.point_logps(point=filtered)
    obs_names = [rv.name for rv in model.observed_RVs]
    ll = float(sum(per_var_logps.get(n, 0.0) for n in obs_names))
    bic = -2.0 * ll + k * math.log(N)
    return bic, k, N, ll


def score_model(build_model, claim_name: str,
                seed: int = 0) -> ModelSelectionScore:
    bic, k, N, ll = _bic_from_map(build_model, seed=seed)
    return ModelSelectionScore(
        claim_name=claim_name,
        bic=bic,
        num_params=k,
        num_observations=N,
        map_loglik=ll,
    )


def _kendall_tau(a: list, b: list) -> float:
    """Kendall-τ between two rankings of the same item set."""
    n = len(a)
    if n < 2:
        return 1.0
    rank_a = {x: i for i, x in enumerate(a)}
    rank_b = {x: i for i, x in enumerate(b)}
    items = list(rank_a.keys())
    concordant = discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (rank_a[items[i]] - rank_a[items[j]]) * \
                (rank_b[items[i]] - rank_b[items[j]])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def compare_on_benchmark(bench_name: str,
                          build_base_model, latent_var: str,
                          claim_specs: list,
                          bic_n: int = 10,
                          seed: int = 0) -> BenchmarkComparison:
    """Compute D and BIC rankings on one benchmark."""
    engine = ClaimRankingEngine(
        build_base_model=build_base_model,
        latent_var=latent_var,
        bic_n=bic_n,
        draws=600, tune=400, random_seed=seed,
    )
    d_scores_list = engine.rank(claim_specs)
    d_ranking = [s.claim.name for s in d_scores_list]
    d_score_values = [s.composite for s in d_scores_list]

    ms_scores = []
    for spec in claim_specs:
        ms_scores.append(score_model(spec.build_model, spec.name, seed=seed))

    # BIC: lower = better → ascending
    bic_ranking = sorted(ms_scores, key=lambda s: s.bic)
    bic_names = [s.claim_name for s in bic_ranking]
    bic_values = [s.bic for s in bic_ranking]

    return BenchmarkComparison(
        benchmark=bench_name,
        candidates=[c.name for c in claim_specs],
        d_ranking=d_ranking,
        bic_ranking=bic_names,
        d_scores=d_score_values,
        bic_scores=bic_values,
        kendall_d_vs_bic=_kendall_tau(d_ranking, bic_names),
    )


# ---------------------------------------------------------------------------
# Benchmark specs (mirror those in task4_expert_agreement)
# ---------------------------------------------------------------------------


def _alice_charlie_spec():
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie, _build_claims,
    )
    return dict(
        name="alice_charlie",
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        claim_specs=_build_claims(),
        bic_n=30,
    )


def _noh_spec():
    return dict(
        name="noh_theater",
        build_base_model=noh_theater.build_base_model,
        latent_var="p_banned",
        claim_specs=noh_theater.get_claims(),
        bic_n=10,
    )


def _eastern_han_spec():
    return dict(
        name="eastern_han",
        build_base_model=eastern_han.build_base_model,
        latent_var="p_young_death",
        claim_specs=eastern_han.get_claims(),
        bic_n=10,
    )


BENCHMARKS = [_alice_charlie_spec, _noh_spec, _eastern_han_spec]


def run_all(seed: int = 0, verbose: bool = True):
    results = []
    for spec_fn in BENCHMARKS:
        spec = spec_fn()
        if verbose:
            print(f"\n=== {spec['name']} ===")
        r = compare_on_benchmark(spec["name"],
                                  spec["build_base_model"],
                                  spec["latent_var"],
                                  spec["claim_specs"],
                                  bic_n=spec["bic_n"],
                                  seed=seed)
        results.append(r)
        if verbose:
            print(f"  D-ranking   : {r.d_ranking}")
            print(f"  BIC-ranking : {r.bic_ranking}")
            print(f"  D scores    : {[f'{v:.3f}' for v in r.d_scores]}")
            print(f"  BIC scores  : {[f'{v:.2f}' for v in r.bic_scores]}")
            print(f"  Kendall τ(D, BIC) = {r.kendall_d_vs_bic:+.3f}")
    return results


if __name__ == "__main__":
    run_all()
