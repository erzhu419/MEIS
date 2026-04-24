"""Paper 2 §4.4 — LLM expert panel (cross-family, 4 raters).

Replaces the single "authorial ranking" (which is a self-consistency
check, not external evaluation) with a cross-family LLM-as-judge
panel. Four raters, 2 families × 2 capability tiers:

  gpt-5.5               (OpenAI family, strong)
  gpt-5.4-mini          (OpenAI family, cheap)
  gemini-3-pro-preview  (Google family, strong)
  gemini-3-flash        (Google family, cheap)

Each rater receives the benchmark problem statement and the set of 4
candidate hypotheses in JSON, and returns a strict-JSON ranking from
"most coherent / minimum perturbation" to "least coherent / most
disruptive". The panel's aggregate ranking is the mean Borda count.

Metrics reported per benchmark:
  - Inter-rater pairwise Kendall τ (4x4 matrix); mean off-diagonal
    gives a reliability proxy (analogous to multi-rater κ).
  - Kendall τ between D-ranking (from ClaimRankingEngine) and the
    panel-mean ranking.
  - Fraction of raters that placed the orphan last.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from _meis_keys import GPT_KEY
from _meis_keys import GEMINI_KEY
from _meis_keys import RUOLI_BASE_URL
API_URL = f"{RUOLI_BASE_URL}/chat/completions"

RATERS = [
    {"name": "gpt-5.5",              "model": "gpt-5.5",                  "api_key": GPT_KEY,    "family": "openai"},
    {"name": "gpt-5.4",              "model": "gpt-5.4",                  "api_key": GPT_KEY,    "family": "openai"},
    {"name": "gpt-5.4-mini",         "model": "gpt-5.4-mini",             "api_key": GPT_KEY,    "family": "openai"},
    {"name": "gpt-5.2",              "model": "gpt-5.2",                  "api_key": GPT_KEY,    "family": "openai"},
    {"name": "gemini-3.1-pro",       "model": "gemini-3.1-pro-preview",   "api_key": GEMINI_KEY, "family": "google"},
    {"name": "gemini-3-flash",       "model": "gemini-3-flash",           "api_key": GEMINI_KEY, "family": "google"},
    {"name": "gemini-3-flash-prev",  "model": "gemini-3-flash-preview",   "api_key": GEMINI_KEY, "family": "google"},
    {"name": "gemini-3.1-flash-lite","model": "gemini-3.1-flash-lite",    "api_key": GEMINI_KEY, "family": "google"},
]


# ---------------------------------------------------------------------------
# Benchmark specs (problem statement + candidate hypothesis names)
# ---------------------------------------------------------------------------

BENCHMARKS = [
    {
        "name": "alice_charlie",
        "problem": (
            "Three adults: Alice, Bob, Charlie. Alice is ~170 cm tall. "
            "Bob and Charlie are both ~175 cm. All three wear the same "
            "European shoe size. Alice's footprint in sand is 0.15 cm "
            "deeper than Charlie's. Human body density is ~1010 kg/m^3, "
            "approximately constant across adults. Weight approximately "
            "scales with height^3 * density. Footprint depth is "
            "proportional to pressure = weight / foot_area. "
            "Question to explain: why is Alice's footprint deeper than "
            "Charlie's, i.e., why might Alice be heavier?"
        ),
        "candidates": [
            {"name": "H_taller",      "summary": "Alice is actually 5 cm taller than reported"},
            {"name": "H_denser",      "summary": "Alice's body density is 5 percent higher than average"},
            {"name": "H_larger_feet", "summary": "Alice's foot area is smaller (shoe size is coarse-grained)"},
            {"name": "H_zodiac",      "summary": "Alice was born in the Year of the Tiger and Charlie in the Year of the Rat"},
        ],
        "orphan": "H_zodiac",
    },
    {
        "name": "noh_theater",
        "problem": (
            "Historical fact: women were excluded from performing Noh "
            "theatre in Japan from roughly the 14th through 19th "
            "centuries. Primary-source context: the Okina purification "
            "rituals emphasised priestly roles; the nyonin-kekkai "
            "tradition excluded women from certain shrine precincts; "
            "the 1629 Tokugawa shogunate ban on Kabuki performers was "
            "partly motivated by gender-mixing concerns. "
            "Question: why were women banned from Noh?"
        ),
        "candidates": [
            {"name": "H_priestly",        "summary": "Priestly/ritual-role hypothesis: Noh was religious, women excluded from priestly roles"},
            {"name": "H_shogunate_ban",   "summary": "Shogunate-ban spillover from the 1629 Kabuki ban"},
            {"name": "H_blood_taboo",     "summary": "Blood-pollution taboo: menstrual impurity concerns"},
            {"name": "H_natural_voice",   "summary": "Female vocal range and aesthetic preference were unsuitable for Noh's musical style"},
        ],
        "orphan": "H_natural_voice",
    },
    {
        "name": "eastern_han",
        "problem": (
            "Historical fact: Eastern Han dynasty (25-220 CE) emperors "
            "had a median age at death around 21 years — strikingly "
            "young. Primary-source context: archaeological evidence of "
            "lead in imperial vessels and cosmetics; court records of "
            "repeated eunuch/consort-kin factional purges; court "
            "genealogies showing cousin-level intra-clan marriage. "
            "Question: why did Eastern Han emperors die so young?"
        ),
        "candidates": [
            {"name": "H_lead_poisoning",    "summary": "Lead poisoning from imperial cosmetics and drinking vessels"},
            {"name": "H_political_stress",  "summary": "Factional eunuch/consort-kin political violence shortened emperor lifespans"},
            {"name": "H_incest_bottleneck", "summary": "Genetic bottleneck from repeated intra-clan consanguineous marriage"},
            {"name": "H_orphan_geomancy",   "summary": "The palace was built on unfortunate fengshui (geomantic) orientation"},
        ],
        "orphan": "H_orphan_geomancy",
    },
]


# ---------------------------------------------------------------------------
# LLM rater call
# ---------------------------------------------------------------------------


RATER_PROMPT = """\
You are a domain-expert reviewer. Given a problem and 4 candidate hypotheses,
rank them from MOST COHERENT (fits the existing evidence with minimum
disruption of what we already know) to LEAST COHERENT (most disrupts or
contradicts existing knowledge).

Problem:
{problem}

Candidates (JSON):
{candidates}

Respond in STRICT JSON only, exactly this format:
{{
  "ranking": ["name_most_coherent", "name_second", "name_third", "name_least_coherent"],
  "reasoning": "one paragraph explaining the ranking"
}}
"""


def _extract_json(text: str) -> dict:
    """Parse a JSON object from a (possibly markdown-wrapped) response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    # Also tolerate leading prose before the JSON block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def call_rater(rater: dict, problem: str, candidates: list,
                seed: int = 0, timeout: int = 120) -> list:
    prompt = RATER_PROMPT.format(
        problem=problem,
        candidates=json.dumps(candidates, indent=2),
    )
    payload = {
        "model": rater["model"],
        "messages": [
            {"role": "system",
             "content": "You are a careful expert rater. Always respond in strict JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "seed": seed,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {rater['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return _extract_json(content)["ranking"]
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"rater {rater['name']} failed after 3 attempts: {e}")


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def kendall_tau(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    ra = {x: i for i, x in enumerate(a)}
    rb = {x: i for i, x in enumerate(b)}
    items = list(ra.keys())
    concordant = discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (ra[items[i]] - ra[items[j]]) * (rb[items[i]] - rb[items[j]])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def borda_mean_ranking(rankings: list[list[str]]) -> list[str]:
    """Borda count: for each candidate sum position across rankings
    (lower = more coherent), order by ascending sum."""
    all_names = set()
    for r in rankings:
        all_names.update(r)
    scores = {n: 0 for n in all_names}
    for r in rankings:
        for pos, name in enumerate(r):
            scores[name] += pos
    return sorted(all_names, key=lambda n: scores[n])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PanelResult:
    benchmark: str
    per_rater_rankings: dict[str, list[str]]
    panel_ranking: list[str]
    inter_rater_tau_mean: float
    inter_rater_tau_min: float
    inter_rater_tau_matrix: np.ndarray
    orphan_last_fraction: float


def run_panel_on_benchmark(bench: dict, seed: int = 0,
                            cache_path: Path | None = None,
                            verbose: bool = True) -> PanelResult:
    per_rater = {}
    raw_responses = {}
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        per_rater = cached.get("per_rater", {})
        raw_responses = cached.get("raw", {})

    for rater in RATERS:
        name = rater["name"]
        if name in per_rater:
            if verbose:
                print(f"  (cached)   {name:<18} {per_rater[name]}")
            continue
        if verbose:
            print(f"  calling    {name:<18} ...", flush=True)
        try:
            ranking = call_rater(rater, bench["problem"], bench["candidates"], seed=seed)
            per_rater[name] = ranking
            raw_responses[name] = ranking
            if verbose:
                print(f"  got        {name:<18} {ranking}")
        except Exception as e:
            print(f"  FAILED     {name:<18} {e}")
            per_rater[name] = None
        time.sleep(2)  # small pacing

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"per_rater": per_rater, "raw": raw_responses}, indent=2))

    # Filter out failed raters
    valid = {k: v for k, v in per_rater.items() if v is not None}
    names = list(valid.keys())
    n_raters = len(names)
    tau_mat = np.eye(n_raters)
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            t = kendall_tau(valid[names[i]], valid[names[j]])
            tau_mat[i, j] = tau_mat[j, i] = t
    off = tau_mat[np.triu_indices(n_raters, k=1)]
    tau_mean = float(off.mean()) if off.size else 1.0
    tau_min = float(off.min()) if off.size else 1.0

    panel = borda_mean_ranking(list(valid.values()))
    orphan_last = sum(1 for r in valid.values() if r[-1] == bench["orphan"])
    return PanelResult(
        benchmark=bench["name"],
        per_rater_rankings=valid,
        panel_ranking=panel,
        inter_rater_tau_mean=tau_mean,
        inter_rater_tau_min=tau_min,
        inter_rater_tau_matrix=tau_mat,
        orphan_last_fraction=orphan_last / max(n_raters, 1),
    )


def run_all(cache_dir: Path | None = None, seed: int = 0,
             verbose: bool = True) -> list[PanelResult]:
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "runs" / "llm_expert_panel"
    results = []
    for bench in BENCHMARKS:
        if verbose:
            print(f"\n=== Benchmark: {bench['name']} ===")
        r = run_panel_on_benchmark(
            bench, seed=seed,
            cache_path=cache_dir / f"{bench['name']}.json",
            verbose=verbose,
        )
        results.append(r)
        if verbose:
            print(f"  panel ranking     : {r.panel_ranking}")
            print(f"  inter-rater τ mean: {r.inter_rater_tau_mean:+.3f}")
            print(f"  inter-rater τ min : {r.inter_rater_tau_min:+.3f}")
            print(f"  orphan-last frac  : {r.orphan_last_fraction:.2f} "
                  f"({sum(1 for x in r.per_rater_rankings.values() if x[-1] == bench['orphan'])}"
                  f"/{len(r.per_rater_rankings)})")
    return results


def compare_to_D(results: list[PanelResult], seed: int = 0,
                  verbose: bool = True):
    """For each benchmark, compute Kendall τ between D-ranking and the
    LLM panel's Borda-mean ranking."""
    from phase3_embedding.claim_ranking_engine import ClaimRankingEngine
    from phase3_embedding.benchmarks import noh_theater, eastern_han
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie, _build_claims,
    )

    bench_specs = {
        "alice_charlie": dict(
            build_base_model=_build_base_alice_charlie,
            latent_var="weight_A",
            claims=_build_claims(),
            bic_n=30,
        ),
        "noh_theater": dict(
            build_base_model=noh_theater.build_base_model,
            latent_var="p_banned",
            claims=noh_theater.get_claims(),
            bic_n=10,
        ),
        "eastern_han": dict(
            build_base_model=eastern_han.build_base_model,
            latent_var="p_young_death",
            claims=eastern_han.get_claims(),
            bic_n=10,
        ),
    }

    out = []
    for r in results:
        spec = bench_specs[r.benchmark]
        engine = ClaimRankingEngine(
            build_base_model=spec["build_base_model"],
            latent_var=spec["latent_var"],
            bic_n=spec["bic_n"],
            draws=500, tune=400, random_seed=seed,
        )
        scored = engine.rank(spec["claims"])
        d_ranking = [s.claim.name for s in scored]
        tau_d_panel = kendall_tau(d_ranking, r.panel_ranking)
        out.append(dict(
            benchmark=r.benchmark,
            d_ranking=d_ranking,
            panel_ranking=r.panel_ranking,
            tau_d_panel=tau_d_panel,
            inter_rater_tau_mean=r.inter_rater_tau_mean,
            orphan_last_frac=r.orphan_last_fraction,
        ))
        if verbose:
            print(f"  {r.benchmark:<15}  τ(D, panel)={tau_d_panel:+.3f}  "
                  f"inter-rater={r.inter_rater_tau_mean:+.3f}  "
                  f"orphan-last={r.orphan_last_fraction:.2f}")
    return out


if __name__ == "__main__":
    print("Paper 2 §4.4 — cross-family LLM expert panel "
          "(4 raters: gpt-5.5/5.4-mini + gemini-3-pro/flash)")
    results = run_all()
    print("\n=== D-vs-Panel Kendall τ ===")
    compare_to_D(results)
