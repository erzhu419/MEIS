"""Validation for P2.3 commits 2 + 3: in-memory + on-disk belief store.

Covers 6 of the 7 acceptance tests from md/P2_3_persistent_belief_network_design.md §6:
  (1) test_fresh_store_from_library            — build from PriorLibrary
  (2) test_single_conjugate_update_matches_phase1 — parity with kl_drift
  (4) test_hypothesis_ranking_persists         — deterministic rankings
  (5) test_evidence_is_append_only             — disk evidence never rewritten (commit 3)
  (6) test_rollback_to_snapshot                — restore to prior state
  (7) test_cross_domain_query                  — tag-overlap retrieval

Plus a round-trip test (save → load → same state) covering commit-3 I/O.

Skipped (deferred to later commit):
  (3) test_non_conjugate_update_via_mcmc       — needs MCMC dispatch

Run:
    python -m phase3_embedding.tests.test_belief_store
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from phase2_prior_library.retrieval import PriorLibrary
from phase3_embedding.belief_store import (
    BeliefStore, Node, PosteriorHandle, Evidence,
)
from phase3_embedding.kl_drift import (
    GaussianPosterior, Hypothesis, condition_normal,
)


# -----------------------------------------------------------------------------
# (1) from_library
# -----------------------------------------------------------------------------
def test_fresh_store_from_library():
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)

    # Every library entry becomes either a node or a relation (qualitative
    # entries land as nodes with deterministic posterior).
    # Nodes may ALSO include auto-spawned scalar parameter nodes from
    # relations with {param_mu, param_sigma} — so node count can exceed
    # entry count.
    assert len(store.nodes) + len(store.relations) >= len(lib.entries), \
        f"coverage gap: {len(store.nodes)=} + {len(store.relations)=} < {len(lib.entries)=}"

    # Spot-check a known node (distribution-only entry)
    assert "human_density_adult" in store.nodes
    dens = store.get_node("human_density_adult")
    assert dens.posterior.kind == "gaussian"
    assert abs(dens.posterior.mu - 1010.0) < 1e-6
    assert abs(dens.posterior.sigma - 30.0) < 1e-6

    # Spot-check a known relation
    assert "weight_from_height_cube_law" in store.relations
    rel = store.relations["weight_from_height_cube_law"]
    assert "theta * height_cm" in rel.relation_expr

    # Auto-spawned scalar parameter node for cube-law theta
    auto_theta_id = "weight_from_height_cube_law::theta"
    assert auto_theta_id in store.nodes, "theta_mu/theta_sigma should spawn a scalar node"
    theta_node = store.get_node(auto_theta_id)
    assert abs(theta_node.posterior.mu - 1.414e-5) < 1e-8

    print(f"[PASS] from_library built {len(store.nodes)} nodes + "
          f"{len(store.relations)} relations from {len(lib.entries)} entries "
          f"({store.summary()})")


# -----------------------------------------------------------------------------
# (2) conjugate update matches Phase 1 closed-form
# -----------------------------------------------------------------------------
def test_single_conjugate_update_matches_phase1():
    """Adding a (height, weight) observation through BeliefStore.add_evidence
    should produce the same posterior as directly calling kl_drift.condition_normal
    on the prior."""
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)

    theta_id = "weight_from_height_cube_law::theta"
    prior = store.get_node(theta_id).posterior.as_gaussian()

    # Observation: height=170 cm → weight=70 kg  (central)
    height, weight = 170.0, 70.0
    obs_sigma = 2.0

    # Reference path: kl_drift closed-form
    xs = np.array([height ** 3])
    ys = np.array([weight])
    ref = condition_normal(prior, xs, ys, obs_sigma)

    # BeliefStore path
    ev = Evidence(id="obs_1", kind="observation",
                  target_nodes=[theta_id],
                  value=weight, x=height ** 3,
                  provenance="test_single_conjugate_update")
    store.add_evidence(ev, obs_sigma=obs_sigma)
    post = store.get_node(theta_id).posterior.as_gaussian()

    assert abs(post.mu - ref.mu) < 1e-10, (post.mu, ref.mu)
    assert abs(post.sigma - ref.sigma) < 1e-10, (post.sigma, ref.sigma)
    assert len(store.evidence) == 1
    assert "obs_1" in store.get_node(theta_id).sources

    print(f"[PASS] conjugate update parity: "
          f"post_mean={post.mu:.6e} (ref {ref.mu:.6e}), "
          f"post_std={post.sigma:.6e}")


# -----------------------------------------------------------------------------
# (4) hypothesis ranking is deterministic
# -----------------------------------------------------------------------------
def test_hypothesis_ranking_persists():
    """Ranking the same hypotheses twice on the same store must produce
    identical KL values (closed-form = fully deterministic)."""
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)

    theta_id = "weight_from_height_cube_law::theta"
    # Inject 10 synthetic observations to get an informative posterior
    rng = np.random.default_rng(42)
    true_theta = store.get_node(theta_id).posterior.mu
    for i, h in enumerate(rng.uniform(150, 190, 10)):
        w = true_theta * h ** 3 + rng.normal(0, 2.0)
        store.add_evidence(
            Evidence(id=f"obs_{i}", kind="observation",
                     target_nodes=[theta_id], value=w, x=float(h) ** 3),
            obs_sigma=2.0,
        )

    hypotheses = [
        Hypothesis("H_central", "170 cm, 70 kg", [(170.0, 70.0)]),
        Hypothesis("H_obese",   "170 cm, 120 kg", [(170.0, 120.0)]),
        Hypothesis("H_absurd",  "170 cm, 200 kg", [(170.0, 200.0)]),
    ]

    scores_1 = store.rank_hypotheses(hypotheses, theta_id)
    scores_2 = store.rank_hypotheses(hypotheses, theta_id)

    kls_1 = [s.kl_from_base for s in scores_1]
    kls_2 = [s.kl_from_base for s in scores_2]
    names_1 = [s.hypothesis.name for s in scores_1]
    names_2 = [s.hypothesis.name for s in scores_2]

    assert names_1 == names_2, (names_1, names_2)
    for a, b in zip(kls_1, kls_2):
        assert abs(a - b) < 1e-12
    # And the ordering should match intuition
    assert names_1 == ["H_central", "H_obese", "H_absurd"], names_1
    print(f"[PASS] rankings deterministic across calls. Final order: {names_1}, "
          f"KLs {[f'{k:.3f}' for k in kls_1]}")


# -----------------------------------------------------------------------------
# (6) snapshot / rollback restores state
# -----------------------------------------------------------------------------
def test_rollback_to_snapshot():
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)
    theta_id = "weight_from_height_cube_law::theta"

    prior_mu = store.get_node(theta_id).posterior.mu
    prior_sigma = store.get_node(theta_id).posterior.sigma
    n_ev_before = len(store.evidence)

    snap = store.snapshot()

    # Mutate: add 5 observations
    for i in range(5):
        store.add_evidence(
            Evidence(id=f"mut_{i}", kind="observation",
                     target_nodes=[theta_id],
                     value=70.0 + i, x=170.0 ** 3),
            obs_sigma=2.0,
        )
    post = store.get_node(theta_id).posterior
    assert post.mu != prior_mu or post.sigma != prior_sigma
    assert len(store.evidence) == n_ev_before + 5

    # Rollback should fully restore
    store.rollback(snap)
    restored = store.get_node(theta_id).posterior
    assert abs(restored.mu - prior_mu) < 1e-12
    assert abs(restored.sigma - prior_sigma) < 1e-12
    assert len(store.evidence) == n_ev_before

    print(f"[PASS] rollback restored posterior to "
          f"N({prior_mu:.3e}, {prior_sigma:.3e}), evidence count {n_ev_before}")


# -----------------------------------------------------------------------------
# (7) cross-domain retrieval by tag overlap
# -----------------------------------------------------------------------------
def test_cross_domain_query():
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)

    # "density" should surface the human_body density node as top-1
    hits = store.search_nodes("density body", k=3)
    ids = [h.id for h in hits]
    assert "human_density_adult" in ids, ids
    assert hits[0].id == "human_density_adult", \
        f"expected top-1 = human_density_adult, got {hits[0].id}"

    # Domain filter
    hb_hits = store.search_nodes("density", k=5, domain="human_body")
    assert all(h.domain == "human_body" for h in hb_hits)

    # Empty / novel query
    assert store.search_nodes("") == []
    assert store.search_nodes("zzz_nonexistent_xyz") == []

    print(f"[PASS] cross-domain query: top-3 for 'density body' = {ids}")


# -----------------------------------------------------------------------------
# (5) evidence files are append-only across save() calls (commit 3)
# -----------------------------------------------------------------------------
def test_evidence_is_append_only():
    """Save a store, add more evidence, save again. The original evidence
    files must still exist, byte-for-byte unchanged."""
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)
    theta_id = "weight_from_height_cube_law::theta"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Round 1: add 2 observations, save
        for i in range(2):
            store.add_evidence(
                Evidence(id=f"run1_obs_{i}", kind="observation",
                         target_nodes=[theta_id], value=70.0 + i, x=170.0 ** 3,
                         provenance="round_1"),
                obs_sigma=2.0,
            )
        store.save(root)
        assert (root / "evidence" / "run1_obs_0.json").exists()
        original_contents = {
            fname: (root / "evidence" / fname).read_bytes()
            for fname in ("run1_obs_0.json", "run1_obs_1.json")
        }

        # Round 2: add 3 more observations, save again
        for i in range(3):
            store.add_evidence(
                Evidence(id=f"run2_obs_{i}", kind="observation",
                         target_nodes=[theta_id], value=60.0 + i, x=165.0 ** 3,
                         provenance="round_2"),
                obs_sigma=2.0,
            )
        store.save(root)

        # All 5 files must exist
        on_disk = {p.name for p in (root / "evidence").glob("*.json")}
        assert on_disk == {"run1_obs_0.json", "run1_obs_1.json",
                           "run2_obs_0.json", "run2_obs_1.json", "run2_obs_2.json"}, \
            f"unexpected evidence files on disk: {on_disk}"

        # Round-1 files must be byte-identical to their original content
        for fname, original in original_contents.items():
            current = (root / "evidence" / fname).read_bytes()
            assert current == original, f"{fname} mutated between save() calls"

        print(f"[PASS] 5 evidence files persisted, 2 round-1 files unchanged across 2 save() calls")


# -----------------------------------------------------------------------------
# Round-trip: save → load produces equivalent state
# -----------------------------------------------------------------------------
def test_disk_roundtrip():
    lib = PriorLibrary.load_default()
    store = BeliefStore.from_library(lib)
    theta_id = "weight_from_height_cube_law::theta"

    # Mutate with a handful of observations so there's posterior drift
    for i in range(5):
        store.add_evidence(
            Evidence(id=f"obs_{i}", kind="observation",
                     target_nodes=[theta_id], value=70.0 + i * 0.5,
                     x=(168 + i) ** 3, provenance="roundtrip_test"),
            obs_sigma=2.0,
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store.save(root)

        # Verify directory layout
        assert (root / "nodes").is_dir()
        assert (root / "relations").is_dir()
        assert (root / "evidence").is_dir()
        assert len(list((root / "nodes").glob("*.json"))) > 0

        reloaded = BeliefStore.load(root)

    # Same counts
    assert len(reloaded.nodes) == len(store.nodes)
    assert len(reloaded.relations) == len(store.relations)
    assert len(reloaded.evidence) == len(store.evidence)

    # Same posterior on the mutated node
    a = store.get_node(theta_id).posterior
    b = reloaded.get_node(theta_id).posterior
    assert a.kind == b.kind == "gaussian"
    assert abs(a.mu - b.mu) < 1e-12
    assert abs(a.sigma - b.sigma) < 1e-12

    # Sources preserved (tracks that updates were applied)
    assert set(store.get_node(theta_id).sources) == set(
        reloaded.get_node(theta_id).sources)

    print(f"[PASS] save→load round-trip preserves state: "
          f"{len(reloaded.nodes)} nodes, "
          f"{len(reloaded.relations)} relations, "
          f"{len(reloaded.evidence)} evidence records")


# -----------------------------------------------------------------------------
# Sample-based posterior disk support
# -----------------------------------------------------------------------------
def test_disk_sample_posterior_roundtrip():
    """Nodes with kind='samples' should save samples to .npz and restore them."""
    store = BeliefStore()
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 2.0, size=500)
    store.nodes["sample_node"] = Node(
        id="sample_node", domain="test", name="test", type="continuous",
        posterior=PosteriorHandle(kind="samples", samples=samples),
        tags=["test"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store.save(root)
        # Confirm .npz exists
        assert (root / "nodes" / "sample_node.samples.npz").exists()
        reloaded = BeliefStore.load(root)
    restored = reloaded.get_node("sample_node").posterior
    assert restored.kind == "samples"
    assert restored.samples.shape == samples.shape
    assert np.allclose(restored.samples, samples)
    print(f"[PASS] sample-based posterior ({len(samples)} draws) survives disk roundtrip")


if __name__ == "__main__":
    print("=== P2.3 commits 2+3 validation: in-memory + on-disk BeliefStore ===\n")
    test_fresh_store_from_library()
    test_single_conjugate_update_matches_phase1()
    test_hypothesis_ranking_persists()
    test_rollback_to_snapshot()
    test_cross_domain_query()
    test_evidence_is_append_only()
    test_disk_roundtrip()
    test_disk_sample_posterior_roundtrip()
    print("\nAll 8 checks passed (6/7 acceptance + 2 disk round-trips). "
          "Test 3 (non-conjugate MCMC update) still deferred.")
