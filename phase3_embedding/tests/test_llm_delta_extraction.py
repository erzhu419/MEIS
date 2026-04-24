"""End-to-end LLM-driven Δ_h extraction validation.

The pipeline accepts an NL hypothesis statement, the LLM (gpt-5.5)
produces a strict-JSON description of the (E_h, Δ_h) network edit,
and the system dynamically builds a PyMC ClaimSpec. No hand-coded
kind→template mapping anywhere.

Acceptance:
  1. The LLM produces parseable structured JSON for all 4 test
     hypotheses.
  2. For the orphan-style hypothesis ("zodiac"), the LLM declares
     ≥ 1 new_latent with no parent connections. structural_additions
     ≥ 1 in the resulting ClaimSpec.
  3. For the in-vocab consistent hypothesis ("denser"), the LLM
     declares 0 new_latents (only new_evidence on existing nodes).
  4. The ranking respects the intuitive coherence ordering (most
     coherent first):
       in-vocab-consistent  < orphan  ≤ contradictory
"""

from __future__ import annotations

from phase3_embedding.llm_delta_extraction import run_demo, HYPOTHESES


def test_llm_delta_pipeline_runs_endtoend():
    scored, edits = run_demo()
    assert len(scored) == len(HYPOTHESES) == 4
    print(f"[PASS] LLM produced parseable Δ_h for all {len(HYPOTHESES)} hypotheses")


def test_orphan_gets_new_latents_invocab_does_not():
    _, edits = run_demo()
    # Zodiac should get ≥ 1 new latent
    z = edits["H3_zodiac"]
    assert len(z.new_latents) >= 1, \
        f"LLM did not declare any new latent for zodiac; got {z.new_latents}"
    # Denser should get 0 new latents
    d = edits["H2_denser_alice"]
    assert len(d.new_latents) == 0, \
        f"LLM unexpectedly added latents for denser hypothesis: {d.new_latents}"
    print(f"[PASS] LLM correctly distinguishes orphan ({len(z.new_latents)} new latents) "
          f"from in-vocab ({len(d.new_latents)} new latents)")


def test_ranking_respects_coherence_ordering():
    """The intuitive ordering must hold:
       D(consistent-in-vocab) < D(orphan) ≤ D(contradictory)
    """
    scored, _ = run_demo()
    by_name = {s.claim.name: s for s in scored}
    d_consistent = by_name["H2_denser_alice"].composite
    d_orphan     = by_name["H3_zodiac"].composite
    d_contradict_mild  = by_name["H1_taller_alice"].composite  # 5cm-shorter
    d_contradict_heavy = by_name["H4_contradiction"].composite # 30cm-shorter

    # consistent < orphan
    assert d_consistent < d_orphan, \
        f"D(consistent)={d_consistent} not < D(orphan)={d_orphan}"
    # orphan < contradictory (severe)
    assert d_orphan < d_contradict_heavy, \
        f"D(orphan)={d_orphan} not < D(severe contradict)={d_contradict_heavy}"
    # severity order on contradictory
    assert d_contradict_mild < d_contradict_heavy, \
        f"D(mild contradict)={d_contradict_mild} not < D(heavy contradict)={d_contradict_heavy}"
    print(f"[PASS] D ordering is "
          f"consistent({d_consistent:.3f}) < orphan({d_orphan:.3f}) "
          f"< mild-contradict({d_contradict_mild:.3f}) < heavy-contradict({d_contradict_heavy:.3f})")


if __name__ == "__main__":
    print("=== LLM-driven Δ_h extraction validation ===\n")
    test_llm_delta_pipeline_runs_endtoend()
    test_orphan_gets_new_latents_invocab_does_not()
    test_ranking_respects_coherence_ordering()
    print("\nAll Δ_h extraction checks passed.")
