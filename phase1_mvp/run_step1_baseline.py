"""Step 1 baseline runner — boxing-gym Scientist→Novice loop on alice_charlie.

Reimplements the no-PPL, discovery-mode path of boxing-gym/run_experiment.py
minimally, so that we can import MEIS's own env without modifying boxing-gym
trunk. Produces JSON output compatible with Step 0 baseline format.

Usage:
    python -m phase1_mvp.run_step1_baseline --seed 1

Env vars required:
    OPENAI_API_KEY, OPENAI_BASE_URL  (via boxing_gym.agents.LMExperimenter)
"""

from __future__ import annotations

import argparse
import json
import os
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


MAX_TRIES = 3
OUT_DIR = Path(__file__).parent / "baselines"
OUT_DIR.mkdir(exist_ok=True)


def run_baseline(seed: int, model_name: str = "gpt-5.4",
                 num_experiments: int = 10, num_evals: int = 10,
                 com_limit: int = 200, include_prior: bool = True,
                 temperature: float = 0.0, max_tokens: int = 512):

    random.seed(seed)
    np.random.seed(seed)

    env = AliceCharlie()
    env.include_prior = include_prior
    goal = DirectGoalNaive(env)

    scientist = LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    novice = LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    scientist.set_system_message(goal.get_system_message(include_prior))

    # --- 10 observation rounds (Scientist phase) ---
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

    # --- Scientist writes explanation for Novice ---
    comm_prompt = goal.get_comm_prompt(com_limit=com_limit, include_prior=include_prior)
    explanation = scientist.prompt_llm(comm_prompt)

    # --- Novice reads explanation + predicts on 10 held-out heights ---
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

    # --- Save in Step-0-compatible JSON layout ---
    store = {
        "config": {
            "seed": seed, "include_prior": include_prior, "use_ppl": False,
            "llms": {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens},
            "envs": {"env_name": "alice_charlie", "goal_name": "direct_discovery", "num_evals": num_evals},
            "exp": {"num_experiments": [num_experiments], "experiment_type": "discovery"},
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
    fname = OUT_DIR / f"step1_alice_charlie_{model_name}_{seed}.json"
    with open(fname, "w") as f:
        json.dump(store, f, indent=2)
    print(f"seed={seed}  err_mean={err_mean:.4f}  err_std={err_std:.4f}  -> {fname.name}")
    return store


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", type=str, default="gpt-5.4")
    args = p.parse_args()
    run_baseline(args.seed, args.model)
