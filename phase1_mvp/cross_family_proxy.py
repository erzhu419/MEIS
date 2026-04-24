"""Paper 1 cross-family proxy for the Peregrines MEIS effect.

The full P2.1 experiment (32 seeds × full boxing-gym pipeline) is
GPT-family only. A proper cross-family replication would require
re-wiring the scientist/novice harness to route through the Gemini
API, which is a larger engineering effort than this paper's scope.

This module provides a *prompt-only proxy*: we feed each of four
models (gpt-5.5, gpt-5.4-mini, gemini-3.1-pro, gemini-3-flash) the
Peregrines problem description under two conditions:

  (BASELINE)  Problem statement only, ask for the functional form.
  (MEIS)      Problem statement + a Poisson log-rate cross-domain
              prior from our library.

We then grep each response for mentions of Poisson / log-rate /
lambda. The headline P2.1 claim was that baseline scientists never
mention Poisson unassisted, while MEIS-primed scientists do, on the
\texttt{count\_or\_poisson} regex tier (0% vs 56%, MW-U p<0.001).

The proxy's role: check that the direction of this effect is
consistent across families. It cannot reproduce the 32-seed
p-value. Honest scope: direction-check only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

GPT_KEY = "sk-w4kL9fnlcUWqMvo97OcTjKTiU6waq2EDWIXWl8KdE3fILFyf"
GEMINI_KEY = "sk-sWaEbAnYG0z8e9oQJaWZmXDiZOIoyrM2bxm7ZR2AxfsYjG2A"
API_URL = "https://ruoli.dev/v1/chat/completions"


MODELS = [
    {"name": "gpt-5.5",          "model": "gpt-5.5",                "key": GPT_KEY},
    {"name": "gpt-5.4-mini",     "model": "gpt-5.4-mini",           "key": GPT_KEY},
    {"name": "gemini-3.1-pro",   "model": "gemini-3.1-pro-preview", "key": GEMINI_KEY},
    {"name": "gemini-3-flash",   "model": "gemini-3-flash",         "key": GEMINI_KEY},
]


PROBLEM = """\
You are a scientist analysing an ecological dataset.

Setting: A Peregrine falcon population is monitored over five decades.
You observe annual counts: c_t = number of Peregrines observed in year t.
The counts rise, peak, and fall over time. Counts are non-negative
integers; some years have very low counts (< 5), some have very high
counts (> 200).

Task: Propose a generative model that would give these counts, and
describe the likelihood family and the functional form of the mean.
Be specific about distributional assumptions.
"""


MEIS_HINT = """\

Cross-domain prior (provided by the system):
  For non-negative count data with an overdispersed distribution and
  a possibly non-monotone mean, the canonical generative family is
  Poisson with a log-rate mean:
    c_t ~ Poisson(exp(alpha + beta_1 * t + beta_2 * t^2 + beta_3 * t^3))
  where the rise-peak-fall pattern suggests a cubic log-rate term.
"""


# Regex tiers from the P2.1 evaluation (BoxingGym scientist-side)
TIERS = {
    "rise_and_fall":      re.compile(r"rise\s+and\s+fall|peak.*fall|unimodal", re.I),
    "polynomial_form":    re.compile(r"polynomial|cubic|quadratic|\bt\^2|\bt\^3", re.I),
    "count_or_poisson":   re.compile(r"poisson|log[\s\-]?rate|\blambda\b|exp\(alpha|log[\s\-]?linear", re.I),
}


def call(model: dict, prompt: str, seed: int = 0, temperature: float = 0.5,
          timeout: int = 120) -> str:
    payload = {
        "model": model["model"],
        "messages": [
            {"role": "system",
             "content": "You are a careful statistician. Respond with a clear generative model description."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "seed": seed,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {model['key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return f"ERROR: {e}"


def score_tiers(text: str) -> dict:
    return {tier: bool(rx.search(text or "")) for tier, rx in TIERS.items()}


@dataclass
class ProbeResult:
    model: str
    condition: str          # baseline | meis
    seed: int
    response: str
    tiers: dict


def run_probe(seeds: list[int] | None = None,
               cache_path: Path | None = None,
               verbose: bool = True) -> list[ProbeResult]:
    if seeds is None:
        seeds = [0, 1, 2]
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "runs" / "cross_family_proxy.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    out = []
    for model in MODELS:
        for cond in ["baseline", "meis"]:
            for seed in seeds:
                key = f"{model['name']}::{cond}::{seed}"
                if key in cache:
                    resp = cache[key]
                else:
                    if verbose:
                        print(f"  calling {key}...", flush=True)
                    prompt = PROBLEM + (MEIS_HINT if cond == "meis" else "")
                    resp = call(model, prompt, seed=seed)
                    cache[key] = resp
                    cache_path.write_text(json.dumps(cache, indent=2))
                    time.sleep(2)
                out.append(ProbeResult(
                    model=model["name"],
                    condition=cond,
                    seed=seed,
                    response=resp,
                    tiers=score_tiers(resp),
                ))
    return out


def summarize_by_model(results: list[ProbeResult]) -> dict:
    """Return dict model -> {condition -> {tier -> rate (0..1)}}."""
    grouped = {}
    for r in results:
        grouped.setdefault(r.model, {}).setdefault(r.condition, []).append(r)
    out = {}
    for m, conds in grouped.items():
        out[m] = {}
        for cond, lst in conds.items():
            out[m][cond] = {
                tier: sum(r.tiers[tier] for r in lst) / max(len(lst), 1)
                for tier in TIERS
            }
            out[m][cond]["n"] = len(lst)
    return out


if __name__ == "__main__":
    print("Paper 1 cross-family proxy on Peregrines (4 models × 2 conds × 3 seeds)\n")
    results = run_probe(seeds=[0, 1, 2])
    summary = summarize_by_model(results)
    print("Per-model tier rates (fraction of seeds with the tier present):\n")
    print(f"{'model':<18}  {'cond':<9}  {'rise_and_fall':>13}  {'polynomial':>11}  {'poisson':>8}  {'n':>2}")
    for m, conds in summary.items():
        for cond in ("baseline", "meis"):
            s = conds.get(cond, {})
            print(f"{m:<18}  {cond:<9}  "
                  f"{s.get('rise_and_fall', 0):>13.2f}  "
                  f"{s.get('polynomial_form', 0):>11.2f}  "
                  f"{s.get('count_or_poisson', 0):>8.2f}  "
                  f"{s.get('n', 0):>2}")
    print()
    # Headline cross-family claim: in BASELINE, count_or_poisson rate
    # across models; in MEIS, does it rise?
    b_rates = [summary[m]["baseline"]["count_or_poisson"] for m in summary]
    m_rates = [summary[m]["meis"]["count_or_poisson"] for m in summary]
    print(f"count_or_poisson baseline rate (across 4 models): {sum(b_rates)/len(b_rates):.2f} "
          f"(range {min(b_rates):.2f}-{max(b_rates):.2f})")
    print(f"count_or_poisson meis     rate (across 4 models): {sum(m_rates)/len(m_rates):.2f} "
          f"(range {min(m_rates):.2f}-{max(m_rates):.2f})")
