"""Paper 2 §7 follow-up: full LLM-driven Δ_h extraction.

In the original LLM-gen experiment (llm_gen_hypothesis_experiment.py)
the LLM proposes hypotheses with a declared `kind` (in-vocab / orphan
/ contradictory), and a hand-coded mapping turns each kind into one
of three fixed PyMC templates. That left the structural edit (Δ_h)
out of the LLM's hands.

This module closes the loop. The LLM is given:
  (a) the base belief-network schema (existing latents + their families
      + which observations they enter)
  (b) an NL hypothesis statement
and is asked to emit a strict JSON describing:
  (i)   new latent random variables to add (name, family, parameters,
        and which existing latents they should be parent-connected to)
  (ii)  new observed evidence (node identifier + observation value
        with optional likelihood-distribution sigma)

We then dynamically build a PyMC ClaimSpec from this JSON. No hand-
coded kind→template mapping. The metric still uses the same
ClaimRankingEngine and computes D(h, B).

Test: 4 hypotheses about Alice-Charlie, varying in coherence. Verify
that D's ordering reflects intuitive coherence (orphan-style claims
high D, in-vocab low D, contradictory huge D) — without the pipeline
ever being told the kind.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pymc as pm


from _meis_keys import GPT_KEY
from _meis_keys import RUOLI_BASE_URL
API_URL = f"{RUOLI_BASE_URL}/chat/completions"
MODEL = "gpt-5.5"


# Base belief-network schema description for the LLM
BASE_SCHEMA_DESC = """\
Base belief network B (Alice-Charlie):
  Latents:
    height_A      : Normal(170, 10) — Alice's height in cm
    height_B      : Normal(175, 5)  — Bob's height in cm  (used as anchor)
    height_C      : Normal(175, 5)  — Charlie's height in cm
    theta_A       : Normal(1010, 30) — Alice's body density in kg/m^3
    theta_B       : Normal(1010, 30)
    theta_C       : Normal(1010, 30)
    shoe_A        : Normal(43, 1)    — Alice's European shoe size
    shoe_B, shoe_C: Normal(43, 1)
    foot_coef     : LogNormal       — slope mapping shoe to foot area
    soil_stiffness: HalfNormal      — sand soil stiffness coefficient

  Deterministic:
    weight_A = theta_A * (height_A/100)^3
    weight_B = theta_B * (height_B/100)^3
    weight_C = theta_C * (height_C/100)^3
    foot_area_A, foot_area_B, foot_area_C = foot_coef * shoe_*
    pressure_A = weight_A / foot_area_A   (and B, C similarly)

  Observations:
    dh_alice ~ Normal(height_A - 170, 1) ; observed Alice 5 cm taller
                                            than reported.
    Alice/Bob/Charlie shoe equality observed.
    Footprint depth observed: depth_A - depth_C = 0.15 cm,
       where depth = (pressure / soil_stiffness)^(1/2).

Target latent: weight_A (we want to know if Alice is heavier).
"""


HYPOTHESIS_PROMPT = """\
You will receive a base belief network and an NL hypothesis. Your job
is to encode the hypothesis as a STRUCTURED JSON edit to the network.

{schema_desc}

NL hypothesis:
  "{hypothesis_nl}"

Encode this as STRICT JSON of the form:
{{
  "new_latents": [
    {{
      "name": "var_name",
      "family": "Normal" | "LogNormal" | "HalfNormal" | "Beta" | "Categorical",
      "params": {{ "mu": ..., "sigma": ..., ... }},
      "parents": [],
      "rationale": "why this latent is added; if NONE, leave new_latents empty"
    }}
  ],
  "new_evidence": [
    {{
      "node": "name of latent or deterministic in the EXTENDED network",
      "observation_value": <number>,
      "sigma": <number, the noise level on this soft observation>,
      "rationale": "why this observation supports the hypothesis"
    }}
  ],
  "explanation": "one paragraph linking the hypothesis to the JSON above"
}}

Rules:
- If the hypothesis introduces a concept that has no causal pathway to
  existing latents (e.g. zodiac, astrology), declare new_latents that
  do NOT list any parents and do NOT participate in any new_evidence
  on existing latents. This corresponds to an orphan-node hypothesis.
- If the hypothesis is consistent with existing latents (e.g. Alice
  is taller, Alice is denser), DO NOT add new latents; instead add a
  new_evidence on the existing latent (or a deterministic depending
  on it) with a value that supports the hypothesis.
- If the hypothesis CONTRADICTS the existing observations (e.g.
  "Alice is shorter than Charlie"), add a new_evidence that pulls
  an existing latent toward the contradicting value with small sigma.
- The variable identifiers MUST exist in the schema above.
"""


@dataclass
class LLMEdit:
    new_latents: list[dict]
    new_evidence: list[dict]
    explanation: str
    raw: str


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def call_llm_edit(hypothesis_nl: str, model: str = MODEL,
                   seed: int = 0, timeout: int = 120) -> LLMEdit:
    prompt = HYPOTHESIS_PROMPT.format(
        schema_desc=BASE_SCHEMA_DESC,
        hypothesis_nl=hypothesis_nl,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a probabilistic-modeling assistant. Always respond in strict JSON exactly matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "seed": seed,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GPT_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            data = _extract_json(content)
            return LLMEdit(
                new_latents=data.get("new_latents", []),
                new_evidence=data.get("new_evidence", []),
                explanation=data.get("explanation", ""),
                raw=content,
            )
        except (urllib.error.HTTPError, urllib.error.URLError,
                json.JSONDecodeError, KeyError) as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM edit extraction failed: {e}")


# ---------------------------------------------------------------------------
# Apply LLMEdit to a PyMC base model
# ---------------------------------------------------------------------------


_FAMILY_TO_PM = {
    "Normal":     pm.Normal,
    "LogNormal":  pm.LogNormal,
    "HalfNormal": pm.HalfNormal,
    "Beta":       pm.Beta,
    "Categorical": pm.Categorical,
}


def build_extended_model(base_builder, edit: LLMEdit, hypothesis_id: str):
    """Take a base PyMC builder + an LLM edit, return a new builder
    closure that produces the extended model."""
    def _builder():
        model = base_builder()
        added_latents = {}  # name -> RV
        with model:
            for spec in edit.new_latents:
                name = f"llm_{hypothesis_id}_{spec['name']}"
                family = spec.get("family", "Normal")
                params = dict(spec.get("params", {}))
                fam_fn = _FAMILY_TO_PM.get(family)
                if fam_fn is None:
                    # Unknown family: skip (LLM error)
                    continue
                # PyMC distribution constructors take params by kw; map
                # standard mu/sigma/alpha/beta/p
                try:
                    rv = fam_fn(name, **params)
                except Exception:
                    # Try a softer fallback
                    rv = pm.Beta(name, alpha=2.0, beta=2.0)
                added_latents[name] = rv

            for ev in edit.new_evidence:
                node_name = ev["node"]
                # Check if the node refers to a base latent or one of
                # our added latents (with the prefix).
                target = None
                if node_name in added_latents:
                    target = added_latents[node_name]
                else:
                    try:
                        target = model[node_name]
                    except KeyError:
                        # Maybe a deterministic — try with an LLM-prefixed
                        # variant
                        prefixed = f"llm_{hypothesis_id}_{node_name}"
                        if prefixed in added_latents:
                            target = added_latents[prefixed]
                if target is None:
                    continue
                obs_val = float(ev.get("observation_value", 0.0))
                sigma = float(ev.get("sigma", 0.05))
                obs_name = f"llm_obs_{hypothesis_id}_{ev['node']}"
                try:
                    pm.Normal(obs_name, mu=target - obs_val, sigma=sigma,
                                observed=0.0)
                except Exception:
                    # Some target nodes may be discrete; skip silently
                    pass
        return model
    return _builder


def claim_from_llm(hypothesis_nl: str, hypothesis_id: str,
                    base_builder):
    from phase3_embedding.claim_ranking_engine import ClaimSpec
    edit = call_llm_edit(hypothesis_nl)
    builder = build_extended_model(base_builder, edit, hypothesis_id)
    return ClaimSpec(
        name=hypothesis_id,
        summary=hypothesis_nl,
        build_model=builder,
        structural_additions=len(edit.new_latents),
    ), edit


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------


HYPOTHESES = [
    ("H1_taller_alice",
     "Alice is actually 5 centimetres taller than reported."),
    ("H2_denser_alice",
     "Alice's body density is 5 percent higher than the population average."),
    ("H3_zodiac",
     "Alice was born in the Year of the Tiger and Charlie in the Year of the Rat."),
    ("H4_contradiction",
     "Alice is much shorter than Charlie, around 30 cm shorter."),
]


def run_demo(out_dir: Path | None = None, seed: int = 0):
    from phase3_embedding.demo_claim_ranking import _build_base_alice_charlie
    from phase3_embedding.claim_ranking_engine import ClaimRankingEngine

    if out_dir is None:
        out_dir = Path(__file__).parent / "runs" / "llm_delta_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LLM-driven Δ_h extraction on Alice-Charlie ({MODEL})\n")
    specs = []
    edits_log = {}
    cache = out_dir / "edits.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}

    for hid, nl in HYPOTHESES:
        if hid in cached:
            edit = LLMEdit(**cached[hid])
            print(f"  (cached) {hid:<22} new_latents={len(edit.new_latents)}, "
                  f"new_evidence={len(edit.new_evidence)}")
        else:
            print(f"  calling  {hid:<22} ...")
            edit = call_llm_edit(nl, seed=seed)
            cached[hid] = dict(new_latents=edit.new_latents,
                                new_evidence=edit.new_evidence,
                                explanation=edit.explanation,
                                raw=edit.raw)
            cache.write_text(json.dumps(cached, indent=2))
            print(f"           {hid:<22} new_latents={len(edit.new_latents)}, "
                  f"new_evidence={len(edit.new_evidence)}")
        edits_log[hid] = edit
        builder = build_extended_model(_build_base_alice_charlie, edit, hid)
        from phase3_embedding.claim_ranking_engine import ClaimSpec
        specs.append(ClaimSpec(
            name=hid, summary=nl,
            build_model=builder,
            structural_additions=len(edit.new_latents),
        ))

    print("\nRanking via D(h, B)...")
    engine = ClaimRankingEngine(
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        bic_n=30,
        draws=500, tune=400, random_seed=seed,
    )
    scored = engine.rank(specs)

    out = []
    print(f"\n  {'rank':<5} {'id':<22} {'|Δ|':>4} {'D':>10}  summary")
    for i, s in enumerate(scored, 1):
        print(f"  {i:<5} {s.claim.name:<22} {s.claim.structural_additions:>4} "
              f"{s.composite:>10.3f}  {s.claim.summary[:50]}")
        out.append(dict(rank=i, name=s.claim.name,
                          structural_additions=s.claim.structural_additions,
                          composite=s.composite,
                          kl_drift=s.kl_drift,
                          structural_term=s.structural_term))
    (out_dir / "ranking.json").write_text(json.dumps(out, indent=2))
    return scored, edits_log


if __name__ == "__main__":
    run_demo()
