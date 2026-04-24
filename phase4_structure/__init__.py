"""MEIS Phase 4 — structural isomorphism recognition across domains.

Plan §Phase 4: cross-domain transfer. After learning "height → density →
weight → pressure → footprint", the system should recognize that "voltage
→ resistance → current → power → heat" is an isomorphic reasoning
structure and directly transfer the prior.

Deliverables (see md/MEIS_plan.md §Phase 4 + §Phase 5):
    P4.1  law_zoo/           Fixtures: 2-3 equivalence classes × 3-4 domains each
    P4.2  signature.py       PyMC model → canonical DAG → Weisfeiler-Lehman hash
    P4.3  retrieval.py       Structural nearest-neighbor + ARI clustering eval
    P4.4  transfer.py        Retrieved neighbor posterior → new-domain prior
    P4.5  md/P4_5_paper_3_draft.md
"""
