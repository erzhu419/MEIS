"""Paper 2 Phase C — LLM-generated hypothesis experiment.

Addresses reviewer point #5: "no experiment where orphan-ness is
discovered by the metric rather than told to it".

Protocol:
  1. Prompt a GPT-family model (via ruoli.dev OpenAI-compatible API)
     with the Alice-Charlie problem statement and ask for 8 candidate
     hypotheses in a STRUCTURED format:
        { name, summary, kind: "in-vocab"|"orphan"|"contradictory",
          new_latents: [...] }
  2. Parse each into a ClaimSpec. For in-vocab claims (kind=in-vocab)
     set structural_additions=0 and reuse an existing evidence
     pattern. For orphan claims (kind=orphan) build a PyMC model
     that adds the declared new_latents as disconnected Beta / Normal
     nodes with a mild soft-obs pull.
  3. Rank all 8 candidates by D(h, B) using the standard
     ClaimRankingEngine. Report which ones the metric ranks as low-
     coherence (orphan or contradictory).
  4. Compare LLM's declared kind against metric's ranking: do the
     bottom-ranked candidates coincide with the LLM's "orphan" and
     "contradictory" labels?

Honest caveat: the mapping from LLM-proposed hypotheses to PyMC
ClaimSpecs still requires manual wiring of how each new_latent
connects or doesn't connect. What is LLM-driven is the *set* of
candidate hypotheses, their names, summaries, and their declared
kind. This is one step closer to "discovered, not told" than the
baseline experiment in §4, without claiming full automation.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import urllib.request
import urllib.error
import numpy as np
import pymc as pm


API_KEY = "sk-w4kL9fnlcUWqMvo97OcTjKTiU6waq2EDWIXWl8KdE3fILFyf"
API_URL = "https://ruoli.dev/v1/chat/completions"
MODEL = "gpt-5.4-mini"


ALICE_CHARLIE_PROBLEM = """\
We have three adults: Alice, Bob, Charlie. We know:
  - Alice's height is approximately 170 cm (normal adult range).
  - Bob and Charlie have the same height, both close to 175 cm.
  - Alice, Bob, Charlie all wear the same European shoe size.
  - Alice left a footprint in sand that is 0.15 cm deeper than Charlie's.

Target question: Is Alice heavier than Charlie?

Human body facts: adult body density ≈ 1010 kg/m³ (nearly constant
across adults). Weight scales approximately with (height)³ times
density. Footprint depth increases with pressure = weight / foot_area.
"""


HYPOTHESIS_PROMPT = """\
Consider this inference problem:

{problem}

Propose exactly 8 distinct hypotheses that could individually explain
the observation "Alice's footprint is deeper than Charlie's". Mix
three styles:
  - 3 IN-VOCAB hypotheses: explanations using only variables already
    in the problem (height, density, foot area, pressure).
  - 3 ORPHAN hypotheses: explanations invoking NEW variables that do
    NOT have a causal path to footprint depth (e.g., astrology,
    favourite colour, birth month). These are clearly incoherent.
  - 2 CONTRADICTORY hypotheses: claims that contradict the given
    observations (e.g., "Alice is actually shorter than Charlie"
    contradicts the 170 vs 175 heights).

Respond in strict JSON only. Format:
{{
  "hypotheses": [
    {{
      "name": "H_short_name",
      "summary": "one-sentence plain-English summary",
      "kind": "in-vocab" | "orphan" | "contradictory",
      "new_latents": ["list of new node names, or empty list"],
      "rationale": "why this hypothesis has the declared kind"
    }},
    ...
  ]
}}
"""


@dataclass
class LLMProposal:
    name: str
    summary: str
    kind: str   # in-vocab | orphan | contradictory
    new_latents: list[str]
    rationale: str


def call_llm(prompt: str, model: str = MODEL, seed: int = 0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a careful reasoning assistant. Respond in strict JSON exactly matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "seed": seed,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def parse_proposals(raw: str) -> list[LLMProposal]:
    # Strip markdown code fence if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    data = json.loads(text)
    return [
        LLMProposal(
            name=h["name"],
            summary=h["summary"],
            kind=h["kind"],
            new_latents=h.get("new_latents", []),
            rationale=h.get("rationale", ""),
        )
        for h in data["hypotheses"]
    ]


# ---------------------------------------------------------------------------
# Mapping proposals -> ClaimSpec
# ---------------------------------------------------------------------------


def build_claim_from_proposal(proposal: LLMProposal,
                                base_model_builder):
    """Convert an LLMProposal into a ClaimSpec with a build_model.

    Strategy:
      - in-vocab: no new latents; build_model = base_model + a soft
        evidence on the salient existing latent (inferred heuristically
        from the hypothesis name). For simplicity all in-vocab
        proposals get the same base model; D_KL will be near zero
        (they don't actually move the target).
      - orphan: add each new_latent as a disconnected Beta node with
        a soft Normal observation that pulls it. These do NOT connect
        to the target.
      - contradictory: add a Normal observation that contradicts an
        existing prior mean (e.g., Alice's height being 160 instead
        of 170). This stays in-vocab but pushes existing latents to
        contradict evidence, producing a large KL drift.

    Returns a ClaimSpec.
    """
    from phase3_embedding.claim_ranking_engine import ClaimSpec

    name = proposal.name
    kind = proposal.kind.lower().strip()

    if kind == "in-vocab":
        # Minimal perturbation: base model with a weak agreement
        # observation
        def _builder():
            model = base_model_builder()
            with model:
                # A no-op: extra observation with 0 evidence
                pass
            return model
        return ClaimSpec(
            name=name, summary=proposal.summary,
            build_model=_builder,
            structural_additions=0,
        )

    elif kind == "orphan":
        nla = proposal.new_latents
        # Create add-on that inserts nla as disconnected Beta nodes
        # with tight soft observations
        def _builder():
            model = base_model_builder()
            with model:
                for latent_name in nla:
                    sanitised = latent_name.replace(" ", "_").replace("-", "_")
                    # Ensure unique and not clashing with existing var names
                    var = pm.Beta(f"llm_orphan_{name}_{sanitised}",
                                   alpha=2.0, beta=2.0)
                    pm.Normal(f"llm_orphan_obs_{name}_{sanitised}",
                                mu=var - 0.9, sigma=0.05, observed=0.0)
            return model
        return ClaimSpec(
            name=name, summary=proposal.summary,
            build_model=_builder,
            structural_additions=len(nla),
        )

    elif kind == "contradictory":
        # Add contradicting evidence on an existing latent. Heuristic:
        # pick weight_A (the target) and force it toward an extreme
        # value to create a large KL drift.
        def _builder():
            model = base_model_builder()
            with model:
                # Weight_A is deterministic from theta_A and height_A
                # in the base model. Contradict by adding a noisy
                # observation that Alice's weight is something
                # inconsistent, e.g., very low (30 kg) vs what height
                # and density predict.
                pm.Normal(f"llm_contradict_{name}_weight",
                            mu=model["weight_A"] - 30.0,
                            sigma=1.0, observed=0.0)
            return model
        return ClaimSpec(
            name=name, summary=proposal.summary,
            build_model=_builder,
            structural_additions=0,
        )

    else:
        raise ValueError(f"unknown proposal kind {kind!r}")


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run_experiment(n_proposals: int = 8, seed: int = 0,
                    out_dir: Path | None = None):
    if out_dir is None:
        out_dir = Path(__file__).parent / "runs" / "llm_gen_hypothesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Asking LLM for hypotheses...")
    prompt = HYPOTHESIS_PROMPT.format(problem=ALICE_CHARLIE_PROBLEM)
    raw = call_llm(prompt, seed=seed)
    (out_dir / "llm_raw_response.json").write_text(raw)
    proposals = parse_proposals(raw)
    print(f"Parsed {len(proposals)} proposals:")
    for p in proposals:
        print(f"  [{p.kind:<14}] {p.name}: {p.summary}")

    # Build ClaimSpecs
    from phase3_embedding.demo_claim_ranking import (
        _build_base_alice_charlie,
    )
    from phase3_embedding.claim_ranking_engine import ClaimRankingEngine

    specs = [build_claim_from_proposal(p, _build_base_alice_charlie)
              for p in proposals]

    # Rank
    print("\nRanking via D(h, B) with BIC structural formula...")
    engine = ClaimRankingEngine(
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        bic_n=30,
        draws=600, tune=400, random_seed=seed,
    )
    scored = engine.rank(specs)
    ranking = [(s.claim.name, s.composite,
                  proposals[[p.name for p in proposals].index(s.claim.name)].kind)
                 for s in scored]

    # Dump and summarize
    rankings_report = {
        "ranking": [
            {"name": n, "composite_D": c, "declared_kind": k}
            for (n, c, k) in ranking
        ],
    }
    (out_dir / "ranking.json").write_text(
        json.dumps(rankings_report, indent=2))

    print("\nFinal ranking (ascending D — most coherent first):")
    print(f"  {'rank':<5} {'name':<35} {'D':>8}  {'LLM-kind':<15}")
    for i, (n, c, k) in enumerate(ranking, 1):
        print(f"  {i:<5} {n:<35} {c:>8.3f}  {k:<15}")

    # Agreement: did bottom-ranked claims coincide with orphan/contradict?
    bottom_half = ranking[len(ranking) // 2:]
    top_half = ranking[:len(ranking) // 2]
    orphan_contradict = {"orphan", "contradictory"}
    bottom_incoherent = sum(1 for (_, _, k) in bottom_half
                             if k.lower() in orphan_contradict)
    top_incoherent = sum(1 for (_, _, k) in top_half
                          if k.lower() in orphan_contradict)
    print(f"\nAgreement: {bottom_incoherent} of {len(bottom_half)} bottom-ranked "
          f"are LLM-declared orphan/contradictory; {top_incoherent} in top half.")
    return rankings_report


if __name__ == "__main__":
    run_experiment()
