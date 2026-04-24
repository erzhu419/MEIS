"""Validation for the LLM-generated hypothesis experiment (Paper 2 §7).

Acceptance:
  1. Pipeline ranks 8 LLM-proposed hypotheses via D(h, B).
  2. The LLM-declared 'contradictory' hypotheses rank LAST (largest D).
  3. The LLM-declared 'orphan' hypotheses rank strictly above
     in-vocab but below contradictory.
  4. All three kind-boundaries are cleanly separated (>10x ratio
     between adjacent kind groups).

Network-dependent: calls ruoli.dev's LLM API. If the API is
unreachable, the test uses a cached canonical response file
(runs/llm_gen_hypothesis/llm_raw_response.json) so subsequent runs
are deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase3_embedding.llm_gen_hypothesis_experiment import (
    parse_proposals, build_claim_from_proposal, call_llm,
    HYPOTHESIS_PROMPT, ALICE_CHARLIE_PROBLEM,
)


_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs" / "llm_gen_hypothesis"
_CACHED_RESPONSE = _RUNS_DIR / "llm_raw_response.json"


def _get_proposals():
    """Load cached LLM response if present; otherwise call API."""
    if _CACHED_RESPONSE.exists():
        raw = _CACHED_RESPONSE.read_text()
    else:
        raw = call_llm(HYPOTHESIS_PROMPT.format(problem=ALICE_CHARLIE_PROBLEM))
        _CACHED_RESPONSE.parent.mkdir(parents=True, exist_ok=True)
        _CACHED_RESPONSE.write_text(raw)
    return parse_proposals(raw)


def test_kind_ordering_is_respected_by_D():
    proposals = _get_proposals()
    from phase3_embedding.demo_claim_ranking import _build_base_alice_charlie
    from phase3_embedding.claim_ranking_engine import ClaimRankingEngine

    specs = [build_claim_from_proposal(p, _build_base_alice_charlie)
              for p in proposals]
    engine = ClaimRankingEngine(
        build_base_model=_build_base_alice_charlie,
        latent_var="weight_A",
        bic_n=30,
        draws=500, tune=400, random_seed=0,
    )
    scored = engine.rank(specs)
    kind_by_name = {p.name: p.kind.lower() for p in proposals}

    # Group by kind; take min/max D per group
    groups = {"in-vocab": [], "orphan": [], "contradictory": []}
    for s in scored:
        k = kind_by_name.get(s.claim.name, "unknown").lower()
        if k in groups:
            groups[k].append(s.composite)

    # Every contradictory D must exceed every orphan D
    assert all(c > o for c in groups["contradictory"]
                for o in groups["orphan"]), \
        (f"kind boundary broken: contradictory {groups['contradictory']} "
         f"not > orphan {groups['orphan']}")
    # Every orphan D must exceed every in-vocab D
    assert all(o > i for o in groups["orphan"]
                for i in groups["in-vocab"]), \
        (f"kind boundary broken: orphan {groups['orphan']} "
         f"not > in-vocab {groups['in-vocab']}")

    # Quantitative separation: orphan/in-vocab ≥ 100; contradictory/orphan ≥ 100
    sep_orphan_invocab = min(groups["orphan"]) / max(groups["in-vocab"])
    sep_contra_orphan = min(groups["contradictory"]) / max(groups["orphan"])
    assert sep_orphan_invocab > 100, (f"orphan/in-vocab separation only "
                                         f"{sep_orphan_invocab:.1f}x")
    assert sep_contra_orphan > 100, (f"contradictory/orphan separation only "
                                       f"{sep_contra_orphan:.1f}x")
    print(f"[PASS] LLM-gen kind ordering: in-vocab < orphan < contradictory,"
          f" orphan/invoc={sep_orphan_invocab:.0f}x, "
          f"contra/orphan={sep_contra_orphan:.0f}x")


if __name__ == "__main__":
    print("=== Paper 2 §7 LLM-gen hypothesis validation ===\n")
    test_kind_ordering_is_respected_by_D()
    print("\nAll LLM-gen checks passed.")
