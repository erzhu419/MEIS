"""Step 2 validation: prior library loads + retrieval gives right hits for
the specific queries Step 3's agent will make on alice_charlie.

Run:
    python -m phase2_prior_library.tests.test_retrieval
"""

from __future__ import annotations

from phase2_prior_library.retrieval import PriorLibrary


def test_load():
    lib = PriorLibrary.load_default()
    assert len(lib.entries) == 10, f"expected 10 entries, got {len(lib.entries)}"
    domains = {e.get("domain") for e in lib.entries}
    assert "human_body" in domains
    assert "classical_mechanics" in domains
    assert "soil_mechanics" in domains
    # schema check
    for e in lib.entries:
        assert "id" in e and "keywords" in e and "statement" in e
    print(f"[PASS] loaded {len(lib.entries)} entries across domains: {sorted(domains)}")


def test_retrieval_alice_charlie_critical_path():
    """The retrieval must surface the ONE entry that makes alice_charlie
    solvable: weight_from_height_cube_law. This is the core Step 3 gate.
    """
    lib = PriorLibrary.load_default()
    hits = lib.retrieve("predict weight given height", k=3)
    ids = [h["id"] for h in hits]
    assert "weight_from_height_cube_law" in ids, f"cube law not in top-3: got {ids}"
    # Ranked first or second, not buried
    assert ids.index("weight_from_height_cube_law") <= 1, f"cube law ranked {ids.index('weight_from_height_cube_law')+1}, want top-2; got {ids}"
    print(f"[PASS] query 'predict weight given height' → top-3: {ids}")


def test_retrieval_component_priors():
    """Checks the density + volume building-block priors surface independently."""
    lib = PriorLibrary.load_default()

    hits = lib.retrieve("body density kg", k=2)
    assert hits[0]["id"] == "human_density_adult", f"got {hits[0]['id']}"
    print(f"[PASS] query 'body density kg' → {hits[0]['id']}")

    hits = lib.retrieve("volume height cube", k=2)
    ids = [h["id"] for h in hits]
    assert "human_body_volume_from_height" in ids, f"got {ids}"
    print(f"[PASS] query 'volume height cube' → top-2: {ids}")


def test_retrieval_cross_domain():
    """Checks Step 4 future use: physics relations surface for 'pressure' / 'footprint'."""
    lib = PriorLibrary.load_default()

    hits = lib.retrieve("footprint depth sand pressure", k=3)
    ids = [h["id"] for h in hits]
    assert "footprint_depth_from_pressure" in ids, f"got {ids}"

    hits = lib.retrieve("pressure force area mechanics", k=3)
    ids = [h["id"] for h in hits]
    assert "pressure_force_over_area" in ids, f"got {ids}"
    print("[PASS] cross-domain queries (footprint, pressure)")


def test_domain_filter():
    lib = PriorLibrary.load_default()
    hits = lib.retrieve("pressure", k=5, domain="classical_mechanics")
    assert all(h.get("domain") == "classical_mechanics" for h in hits), f"got {[h.get('domain') for h in hits]}"
    assert len(hits) >= 1
    print(f"[PASS] domain filter restricts to classical_mechanics ({len(hits)} hits)")


def test_empty_and_novel_queries():
    lib = PriorLibrary.load_default()
    assert lib.retrieve("") == []
    assert lib.retrieve("quantum chromodynamics neutrino") == []
    print("[PASS] empty / novel queries return []")


def test_format_for_prompt():
    lib = PriorLibrary.load_default()
    hits = lib.retrieve("weight height", k=2)
    blob = PriorLibrary.format_for_prompt(hits)
    assert isinstance(blob, str)
    assert "weight_from_height_cube_law" in blob
    assert "Formal:" in blob or "Distribution:" in blob
    # Cap check: shouldn't blow past max_chars
    blob_short = PriorLibrary.format_for_prompt(hits, max_chars=200)
    assert len(blob_short) <= 200
    print(f"[PASS] format_for_prompt produces readable chunk ({len(blob)} chars)")


if __name__ == "__main__":
    print("=== Step 2 validation: prior library + retrieval ===\n")
    test_load()
    test_retrieval_alice_charlie_critical_path()
    test_retrieval_component_priors()
    test_retrieval_cross_domain()
    test_domain_filter()
    test_empty_and_novel_queries()
    test_format_for_prompt()
    print("\nAll Step 2 checks passed.")
