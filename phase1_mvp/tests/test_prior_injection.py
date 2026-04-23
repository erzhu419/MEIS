"""Step 3 validation: prior injection produces the right system prompt.

Run:
    python -m phase1_mvp.tests.test_prior_injection
"""

from __future__ import annotations

import os

# LMExperimenter constructs an openai.OpenAI() client on __init__, which
# requires an API key even though this test never calls the LLM.
os.environ.setdefault("OPENAI_API_KEY", "unit-test-dummy")
os.environ.setdefault("OPENAI_BASE_URL", "https://dummy-not-called.test/v1")

from phase1_mvp.envs.alice_charlie import AliceCharlie, DirectGoalNaive
from phase1_mvp.agents.prior_injecting_experimenter import (
    PriorInjectingExperimenter, DEFAULT_PRIOR_HEADER,
)
from phase2_prior_library.retrieval import PriorLibrary


ALICE_CHARLIE_QUERY = "predict weight given height"


def _build_agent():
    lib = PriorLibrary.load_default()
    # Use a model_name that does NOT match "claude" — routes to OpenAI branch.
    # No API call is made inside set_system_message (LLM is not invoked), so a
    # fake model name is fine for this unit test.
    agent = PriorInjectingExperimenter(
        model_name="gpt-5.4-unit-test",
        library=lib,
        prior_query=ALICE_CHARLIE_QUERY,
        prior_k=5,
    )
    return agent, lib


def test_prior_block_contains_cube_law():
    agent, _ = _build_agent()
    env = AliceCharlie()
    goal = DirectGoalNaive(env)
    agent.set_system_message(goal.get_system_message(include_prior=True))

    assert "weight_from_height_cube_law" in agent.system, \
        "cube-law prior missing from injected system message"
    assert DEFAULT_PRIOR_HEADER.split(".")[0] in agent.system, "header missing"
    assert "Adult humans of varying heights" in agent.system, "base system msg lost"
    print(f"[PASS] cube-law prior + header + base-env prompt all present in system ({len(agent.system)} chars)")


def test_retrieved_block_recorded_for_audit():
    agent, _ = _build_agent()
    agent.set_system_message("base prompt")
    assert len(agent.last_prior_hits) == 5, f"want 5 retrieved hits, got {len(agent.last_prior_hits)}"
    assert len(agent.last_prior_block) > 0
    # Top hit MUST be the cube-law on the alice_charlie query.
    assert agent.last_prior_hits[0]["id"] == "weight_from_height_cube_law", \
        f"top-1 hit was {agent.last_prior_hits[0]['id']}, expected weight_from_height_cube_law"
    ids = [h["id"] for h in agent.last_prior_hits]
    print(f"[PASS] retrieval audit recorded on agent: top-5 ids = {ids}")


def test_prior_k_controls_size():
    lib = PriorLibrary.load_default()
    a1 = PriorInjectingExperimenter(model_name="gpt-5.4", library=lib,
                                    prior_query=ALICE_CHARLIE_QUERY, prior_k=1)
    a3 = PriorInjectingExperimenter(model_name="gpt-5.4", library=lib,
                                    prior_query=ALICE_CHARLIE_QUERY, prior_k=3)
    a1.set_system_message("hi")
    a3.set_system_message("hi")
    assert len(a1.system) < len(a3.system), "k=1 should give shorter prompt than k=3"
    print(f"[PASS] k controls block size (k=1: {len(a1.system)} chars vs k=3: {len(a3.system)} chars)")


def test_empty_query_degrades_gracefully():
    lib = PriorLibrary.load_default()
    agent = PriorInjectingExperimenter(model_name="gpt-5.4", library=lib,
                                       prior_query="zzz_nonexistent_query_xyz", prior_k=5)
    agent.set_system_message("base")
    assert agent.last_prior_block == ""
    # System should just equal the base message (or base + nothing appended)
    assert agent.system.startswith("base")
    print("[PASS] novel query → empty prior block, base prompt unchanged")


def test_inheritance_preserves_message_format():
    """Sanity: the injected agent still maintains OpenAI-style content blocks
    in self.messages, so downstream run_experiment logic isn't broken."""
    agent, _ = _build_agent()
    agent.set_system_message("base")
    # OpenAI branch (non-claude model name) should populate self.messages[0]
    assert len(agent.messages) == 1
    msg0 = agent.messages[0]
    assert msg0["role"] == "system"
    assert isinstance(msg0["content"], list)
    assert msg0["content"][0]["type"] == "text"
    print("[PASS] self.messages still has OpenAI content-block structure")


if __name__ == "__main__":
    print("=== Step 3 validation: prior injection ===\n")
    test_prior_block_contains_cube_law()
    test_retrieved_block_recorded_for_audit()
    test_prior_k_controls_size()
    test_empty_query_degrades_gracefully()
    test_inheritance_preserves_message_format()
    print("\nAll Step 3 unit checks passed.")
