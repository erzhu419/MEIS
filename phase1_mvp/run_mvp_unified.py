"""Unified alice_charlie runner: single source of truth for all three conditions.

Conditions controlled by three flags:
  --scientist-priors   inject MEIS priors into scientist's system message
  --novice-priors      inject MEIS priors into novice's  system message
  --no-echo-anchor     drop boxing-gym's `"The final result is <obs>"` eval prefix

Three canonical configs used in Step 3.5:
  baseline_noecho       (no priors, no echo anchor)
  meis_sci_noecho       (--scientist-priors --no-echo-anchor)
  meis_full_noecho      (--scientist-priors --novice-priors --no-echo-anchor)

Outputs go to phase1_mvp/runs/<config_tag>/seed_<n>.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from boxing_gym.agents.agent import LMExperimenter

from phase1_mvp.envs.alice_charlie import AliceCharlie, DirectGoalNaive
from phase1_mvp.agents.prior_injecting_experimenter import (
    PriorInjectingExperimenter, DEFAULT_PRIOR_HEADER,
)
from phase2_prior_library.retrieval import PriorLibrary


MAX_TRIES = 3
ALICE_CHARLIE_QUERY = "predict weight given height"
RUNS_ROOT = Path(__file__).parent / "runs"
RUNS_ROOT.mkdir(exist_ok=True)


def _make_agent(kind: str, *, model_name: str, library: PriorLibrary,
                temperature: float, max_tokens: int, prior_k: int):
    if kind == "plain":
        return LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    elif kind == "prior_injected":
        return PriorInjectingExperimenter(
            model_name=model_name, library=library,
            prior_query=ALICE_CHARLIE_QUERY, prior_k=prior_k,
            temperature=temperature, max_tokens=max_tokens,
        )
    raise ValueError(f"unknown agent kind: {kind}")


def config_tag(scientist_priors: bool, novice_priors: bool, echo_anchor: bool) -> str:
    # Canonical three tags used in Step 3.5
    if not scientist_priors and not novice_priors and echo_anchor:
        return "baseline_echo"
    if not scientist_priors and not novice_priors and not echo_anchor:
        return "baseline_noecho"
    if scientist_priors and not novice_priors and not echo_anchor:
        return "meis_sci_noecho"
    if scientist_priors and novice_priors and not echo_anchor:
        return "meis_full_noecho"
    # Fallback label
    return f"sci{int(scientist_priors)}_nov{int(novice_priors)}_echo{int(echo_anchor)}"


def run_mvp(seed: int, model_name: str = "gpt-5.4", *,
            scientist_priors: bool = False, novice_priors: bool = False,
            echo_anchor: bool = True,
            prior_k: int = 5,
            num_experiments: int = 10, num_evals: int = 10,
            com_limit: int = 200, include_prior: bool = True,
            temperature: float = 0.0, max_tokens: int = 512):

    random.seed(seed)
    np.random.seed(seed)

    library = PriorLibrary.load_default()
    env = AliceCharlie()
    env.include_prior = include_prior
    goal = DirectGoalNaive(env)

    scientist = _make_agent("prior_injected" if scientist_priors else "plain",
                            model_name=model_name, library=library,
                            temperature=temperature, max_tokens=max_tokens,
                            prior_k=prior_k)
    novice = _make_agent("prior_injected" if novice_priors else "plain",
                         model_name=model_name, library=library,
                         temperature=temperature, max_tokens=max_tokens,
                         prior_k=prior_k)

    scientist.set_system_message(goal.get_system_message(include_prior))

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

    final_results = f"The final result is {observation}." if echo_anchor else ""

    # --- Scientist writes explanation for novice ---
    comm_prompt = goal.get_comm_prompt(com_limit=com_limit, include_prior=include_prior)
    explanation = scientist.prompt_llm(comm_prompt)

    # --- Novice reads system message + explanation, predicts ---
    naive_system_message = goal.get_naive_system_message(include_prior) + explanation
    novice.set_system_message(naive_system_message)

    predictions, gts, questions = [], [], []
    goal.eval_pointer = 0
    for _ in range(num_evals):
        question, gt = goal.get_goal_eval_question(include_prior)
        if echo_anchor:
            question = final_results + "\n" + question
        pred = novice.generate_predictions(question)
        predictions.append(pred)
        gts.append(gt)
        questions.append(question)

    err_mean, err_std = goal.evaluate_predictions(predictions, gts)

    tag = config_tag(scientist_priors, novice_priors, echo_anchor)
    out_dir = RUNS_ROOT / tag
    out_dir.mkdir(exist_ok=True)

    retrieved_ids_sci = (getattr(scientist, "last_prior_hits", None) or [])
    retrieved_ids_sci = [h["id"] for h in retrieved_ids_sci]
    retrieved_ids_nov = (getattr(novice, "last_prior_hits", None) or [])
    retrieved_ids_nov = [h["id"] for h in retrieved_ids_nov]

    store = {
        "config": {
            "seed": seed, "include_prior": include_prior, "use_ppl": False,
            "llms": {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens},
            "envs": {"env_name": "alice_charlie", "goal_name": "direct_discovery", "num_evals": num_evals},
            "exp": {"num_experiments": [num_experiments], "experiment_type": "discovery"},
            "meis": {
                "config_tag": tag,
                "scientist_priors": scientist_priors,
                "novice_priors": novice_priors,
                "echo_anchor": echo_anchor,
                "prior_query": ALICE_CHARLIE_QUERY,
                "prior_k": prior_k,
                "prior_header": DEFAULT_PRIOR_HEADER,
                "scientist_retrieved_ids": retrieved_ids_sci,
                "novice_retrieved_ids": retrieved_ids_nov,
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
    fname = out_dir / f"seed_{seed}.json"
    with open(fname, "w") as f:
        json.dump(store, f, indent=2)
    print(f"[{tag}] seed={seed}  err_mean={err_mean:.4f}  err_std={err_std:.4f}  -> {fname}")
    return store


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument("--scientist-priors", action="store_true")
    p.add_argument("--novice-priors", action="store_true")
    p.add_argument("--no-echo-anchor", action="store_true")
    p.add_argument("--prior-k", type=int, default=5)
    args = p.parse_args()
    run_mvp(
        seed=args.seed, model_name=args.model,
        scientist_priors=args.scientist_priors,
        novice_priors=args.novice_priors,
        echo_anchor=not args.no_echo_anchor,
        prior_k=args.prior_k,
    )
