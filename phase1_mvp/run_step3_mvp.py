"""Step 3 MVP runner — alice_charlie with prior-injected scientist.

Same Scientist→Novice loop as run_step1_baseline.py, but the Scientist is a
PriorInjectingExperimenter that gets 5 retrieved priors in its system message.
The Novice stays as a plain LMExperimenter (no priors) — it only sees the
Scientist's natural-language explanation, like in Step 1.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=... \\
        python -m phase1_mvp.run_step3_mvp --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from boxing_gym.agents.agent import LMExperimenter

from phase1_mvp.envs.alice_charlie import AliceCharlie, DirectGoalNaive
from phase1_mvp.agents.prior_injecting_experimenter import (
    PriorInjectingExperimenter, DEFAULT_PRIOR_HEADER,
)
from phase2_prior_library.retrieval import PriorLibrary


MAX_TRIES = 3
OUT_DIR = Path(__file__).parent / "runs_step3"
OUT_DIR.mkdir(exist_ok=True)

# Narrow query — Step 2 test confirmed this puts cube law at top-1.
ALICE_CHARLIE_QUERY = "predict weight given height"


def run_mvp(seed: int, model_name: str = "gpt-5.4",
            prior_k: int = 5,
            num_experiments: int = 10, num_evals: int = 10,
            com_limit: int = 200, include_prior: bool = True,
            temperature: float = 0.0, max_tokens: int = 512):

    random.seed(seed)
    np.random.seed(seed)

    env = AliceCharlie()
    env.include_prior = include_prior
    goal = DirectGoalNaive(env)

    # --- Prior-injected scientist + plain novice ---
    library = PriorLibrary.load_default()
    scientist = PriorInjectingExperimenter(
        model_name=model_name, library=library,
        prior_query=ALICE_CHARLIE_QUERY, prior_k=prior_k,
        temperature=temperature, max_tokens=max_tokens,
    )
    novice = LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    scientist.set_system_message(goal.get_system_message(include_prior))

    assert "weight_from_height_cube_law" in scientist.system, \
        "cube-law prior must be in scientist's system message before any LLM call"

    # --- 10 observation rounds ---
    queries, observations, successes = [], [], []
    observation = None
    for _ in range(num_experiments):
        observe = scientist.generate_actions(observation)
        queries.append(observe)
        observation, ok = env.run_experiment(observe)
        observations.append(observation)
        successes.append(ok)
        tries = 1
        while not ok and tries < MAX_TRIES:
            observe, _ = scientist.prompt_llm_and_parse(str(observation), True)
            queries.append(observe)
            observation, ok = env.run_experiment(observe)
            observations.append(observation)
            successes.append(ok)
            if not ok:
                tries += 1

    final_results = f"The final result is {observation}."

    # --- Explanation + novice predictions ---
    comm_prompt = goal.get_comm_prompt(com_limit=com_limit, include_prior=include_prior)
    explanation = scientist.prompt_llm(comm_prompt)

    naive_system_message = goal.get_naive_system_message(include_prior) + explanation
    novice.set_system_message(naive_system_message)

    predictions, gts, questions = [], [], []
    goal.eval_pointer = 0
    for _ in range(num_evals):
        question, gt = goal.get_goal_eval_question(include_prior)
        question = final_results + "\n" + question
        pred = novice.generate_predictions(question)
        predictions.append(pred)
        gts.append(gt)
        questions.append(question)

    err_mean, err_std = goal.evaluate_predictions(predictions, gts)

    # --- Save result JSON — same layout as Step 0/1 baseline, plus MEIS metadata ---
    store = {
        "config": {
            "seed": seed, "include_prior": include_prior, "use_ppl": False,
            "llms": {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens},
            "envs": {"env_name": "alice_charlie", "goal_name": "direct_discovery", "num_evals": num_evals},
            "exp": {"num_experiments": [num_experiments], "experiment_type": "discovery"},
            "meis": {
                "prior_query": ALICE_CHARLIE_QUERY,
                "prior_k": prior_k,
                "prior_header": DEFAULT_PRIOR_HEADER,
                "retrieved_prior_ids": [h["id"] for h in scientist.last_prior_hits],
            },
        },
        "data": {
            "results": [[[err_mean, err_std], questions, gts, predictions]],
            "queries": queries,
            "observations": observations,
            "successes": successes,
            "explanations": [explanation],
            "eigs": [],
            "programs": [],
        },
        "scientist_messages": scientist.all_messages,
        "naive_messages": novice.all_messages,
    }
    fname = OUT_DIR / f"step3_mvp_alice_charlie_{model_name}_{seed}.json"
    with open(fname, "w") as f:
        json.dump(store, f, indent=2)

    print(f"seed={seed}  err_mean={err_mean:.4f}  err_std={err_std:.4f}  "
          f"retrieved={[h['id'] for h in scientist.last_prior_hits]}")
    print(f"  -> {fname.name}")
    return store


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument("--prior-k", type=int, default=5)
    args = p.parse_args()
    run_mvp(args.seed, args.model, prior_k=args.prior_k)
