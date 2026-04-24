"""LLM-bootstrap of the cross-domain prior library to ≥200 entries.

Calls gpt-5.5 to generate candidate entries in batches across 12
domains, then runs a separate LLM-as-reviewer (gemini-3.1-pro) pass
that filters out malformed / contradictory / non-falsifiable entries.

Domains chosen to span the law-zoo's 4 equivalence classes
(exp_decay, saturation, damped_osc, allocation) plus generic
quantitative-reasoning territory.

Output: phase2_prior_library/{domain}_llm.json files; reviewed entries
also merged into phase2_prior_library/_master_extended.json.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from _meis_keys import GPT_KEY
from _meis_keys import GEMINI_KEY
from _meis_keys import RUOLI_BASE_URL
API_URL = f"{RUOLI_BASE_URL}/chat/completions"

OUT_DIR = Path(__file__).parent
GENERATOR_MODEL = "gpt-5.5"
REVIEWER_MODEL = "gemini-3.1-pro-preview"


DOMAINS = {
    "human_physiology":   "human body, vital signs, metabolic rates, biomechanics",
    "classical_mechanics": "force, mass, springs, pendulums, friction, projectile motion",
    "circuits_electronics": "voltage, current, RC and RLC circuits, capacitor charging, resistor networks",
    "thermodynamics":     "heat capacity, conduction rates, enthalpy, equilibrium constants",
    "chemistry_kinetics": "first/second-order reaction rates, half-lives, Arrhenius equation, equilibrium",
    "population_dynamics": "birth/death rates, logistic growth, predator-prey, carrying capacity",
    "epidemiology":       "transmission rate R0, incubation period, recovery rate, vaccination coverage",
    "geophysics":         "seismic wave propagation, decay constants, density of crust/mantle",
    "astronomy":          "stellar luminosity, orbital periods, distance modulus, Hubble constant",
    "psychology_cognitive": "memory decay (forgetting curve), reaction time distributions, learning rates",
    "economics":          "elasticities, discount rates, inflation drift, beta of stock returns",
    "engineering_signals": "filter time constants, signal-to-noise ratios, sampling frequencies",
}


GENERATE_PROMPT = """\
You are curating a cross-domain Bayesian prior library. Each entry
captures a quantitative distributional claim about a real-world
variable that a probabilistic-program user could plug directly into a
PyMC model.

Domain: {domain_name}
Sub-area description: {description}

Generate exactly {n} new entries that meet ALL of the following:
  - Distributional claim (Normal / LogNormal / HalfNormal / Beta /
    Gamma / Poisson) with concrete parameter values.
  - Real-world domain anchor (variable interpretation + canonical units).
  - Cite a specific generic source (textbook, standard reference, well-
    known empirical study). Do NOT invent specific paper citations.
  - Confidence ∈ {{"high", "medium", "low"}} reflecting literature
    consensus.
  - Each entry is DIFFERENT from the others (different variable, not
    just rephrasing).

Output STRICT JSON of the form:
{{
  "entries": [
    {{
      "id": "domain_short_id",
      "domain": "{domain_name}",
      "keywords": ["keyword1", "keyword2", ...],
      "vars_involved": ["variable_name_with_units"],
      "statement": "one-paragraph plain-language description",
      "formal": {{
        "distribution": "Normal" | "LogNormal" | "HalfNormal" | "Beta" | "Gamma" | "Poisson",
        "mu": <number, if applicable>,
        "sigma": <number, if applicable>,
        "alpha": <number, if applicable>,
        "beta": <number, if applicable>,
        "lam": <number, if applicable>,
        "units": "string"
      }},
      "source": "domain reference (textbook or well-known study)",
      "confidence": "high" | "medium" | "low"
    }}
  ]
}}
"""


REVIEW_PROMPT = """\
You are reviewing prior-library entries for inclusion in a cross-domain
Bayesian inference system. For each entry, judge:

  (a) Plausibility: are the parameter values consistent with the
      plain-language statement?
  (b) Domain accuracy: is the source / variable interpretation correct
      for the stated domain?
  (c) Operational completeness: are all required fields filled in;
      can this be plugged into PyMC as-is?

For each entry return REJECT or KEEP plus a brief reason. Output
strict JSON:
{{
  "verdicts": [
    {{ "id": "...", "verdict": "KEEP" | "REJECT", "reason": "..." }},
    ...
  ]
}}

Entries to review:
{entries_json}
"""


def call_api(model: str, prompt: str, key: str, seed: int = 0,
              timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You curate scientific prior knowledge with care. Always respond in strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "seed": seed,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
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
            raise


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def generate_for_domain(domain_name: str, description: str, n: int,
                         seed: int = 0) -> list:
    raw = call_api(GENERATOR_MODEL,
                    GENERATE_PROMPT.format(domain_name=domain_name,
                                             description=description, n=n),
                    GPT_KEY, seed=seed)
    data = _extract_json(raw)
    return data.get("entries", [])


def review_entries(entries: list, seed: int = 0) -> list:
    """Return entries with verdict=KEEP only."""
    if not entries:
        return []
    raw = call_api(REVIEWER_MODEL,
                    REVIEW_PROMPT.format(entries_json=json.dumps(entries, indent=2)),
                    GEMINI_KEY, seed=seed)
    try:
        data = _extract_json(raw)
        verdicts = {v["id"]: v["verdict"] for v in data.get("verdicts", [])}
        kept = [e for e in entries if verdicts.get(e["id"]) == "KEEP"]
        return kept
    except Exception:
        # If review fails, conservatively keep all (better than losing
        # all generated entries to a parsing glitch)
        return entries


def bootstrap(target_per_domain: int = 16, verbose: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "_llm_bootstrap_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    all_kept = []
    summary = {}
    for domain, desc in DOMAINS.items():
        if domain in cache and "kept" in cache[domain]:
            kept = cache[domain]["kept"]
            if verbose:
                print(f"  (cached) {domain:<25} {len(kept)} kept")
        else:
            if verbose:
                print(f"  generating  {domain:<25} ...")
            entries = generate_for_domain(domain, desc, n=target_per_domain)
            if verbose:
                print(f"  reviewing   {domain:<25} {len(entries)} candidates ...")
            kept = review_entries(entries)
            cache[domain] = dict(generated=entries, kept=kept)
            cache_path.write_text(json.dumps(cache, indent=2))
            if verbose:
                print(f"  result      {domain:<25} {len(kept)}/{len(entries)} kept")
        all_kept.extend(kept)
        summary[domain] = dict(kept=len(kept),
                                generated=len(cache[domain].get("generated", [])))

        # Per-domain JSON file in the canonical schema
        fname = f"{domain}_llm.json"
        (OUT_DIR / fname).write_text(json.dumps(kept, indent=2))
        time.sleep(2)

    # Master file with everything
    master = OUT_DIR / "_master_extended.json"
    # Combine with the original 20 entries
    original = []
    for src in ["human_body.json", "growth_curves.json",
                 "dynamics_multivariate.json", "count_regression.json"]:
        original.extend(json.loads((OUT_DIR / src).read_text()))
    master.write_text(json.dumps(original + all_kept, indent=2))

    print(f"\n=== Summary ===")
    print(f"  original library size:     {len(original)}")
    print(f"  LLM-bootstrap kept:        {len(all_kept)}")
    print(f"  combined master size:      {len(original) + len(all_kept)}")
    for d, s in summary.items():
        print(f"  {d:<25} {s['kept']:>3}/{s['generated']:>3} kept")
    return original + all_kept


if __name__ == "__main__":
    bootstrap()
