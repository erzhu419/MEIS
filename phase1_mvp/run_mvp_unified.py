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

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from boxing_gym.agents.agent import LMExperimenter
from boxing_gym.envs.dugongs import Dugongs, DirectGoalNaive as DugongsDirectGoalNaive

from phase1_mvp.envs.alice_charlie import AliceCharlie, DirectGoalNaive as AliceCharlieDirectGoalNaive
from phase1_mvp.agents.prior_injecting_experimenter import (
    PriorInjectingExperimenter, DEFAULT_PRIOR_HEADER,
)
from phase2_prior_library.retrieval import PriorLibrary


MAX_TRIES = 3
RUNS_ROOT = Path(__file__).parent / "runs"
RUNS_ROOT.mkdir(exist_ok=True)

# Per-env retrieval queries — should land the critical-path prior in top-3.
PRIOR_QUERIES = {
    "alice_charlie": "predict weight given height",
    "dugongs":       "predict length given age",
}

ENV_REGISTRY = {
    "alice_charlie": (AliceCharlie, AliceCharlieDirectGoalNaive),
    "dugongs":       (Dugongs,      DugongsDirectGoalNaive),
}

# Plausible output ranges per env. Predictions outside this range trigger
# a retry with an out-of-range warning. Applied uniformly to all configs
# (baseline AND MEIS) so the comparison stays fair — it's a runner-level
# sanity guard, not a MEIS mechanism.
SANITY_RANGES = {
    "alice_charlie": (30.0, 120.0),   # adult weight in kg
    "dugongs":       (0.0,   3.0),    # sea cow length in meters
}


def _predict_with_sanity_retry(novice, question: str, env_name: str,
                               enabled: bool, max_retries: int = 3
                               ) -> tuple[str, int]:
    """Ask novice for a prediction; if `enabled` and it parses to a float outside
    the env's plausible range, re-prompt with an out-of-range warning up to
    max_retries times. Returns (final_pred_str, retries_used)."""
    pred_str = novice.generate_predictions(question)
    if not enabled:
        return pred_str, 0
    rng = SANITY_RANGES.get(env_name)
    if rng is None:
        return pred_str, 0
    lo, hi = rng
    retries = 0
    while retries < max_retries:
        try:
            pred_f = float(str(pred_str).strip())
        except (ValueError, TypeError):
            break  # parse failure — let downstream evaluate_predictions handle
        if lo <= pred_f <= hi:
            break
        retries += 1
        retry_q = (f"{question}\n\n(Your previous answer was {pred_f}, which is "
                   f"outside the plausible range [{lo}, {hi}] for this quantity. "
                   f"Please provide a realistic value within that range.)")
        pred_str = novice.generate_predictions(retry_q)
    return pred_str, retries


def _make_agent(kind: str, *, model_name: str, library: PriorLibrary,
                prior_query: str,
                temperature: float, max_tokens: int, prior_k: int):
    if kind == "plain":
        return LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    elif kind == "prior_injected":
        return PriorInjectingExperimenter(
            model_name=model_name, library=library,
            prior_query=prior_query, prior_k=prior_k,
            temperature=temperature, max_tokens=max_tokens,
        )
    raise ValueError(f"unknown agent kind: {kind}")


def config_tag(scientist_priors: bool, novice_priors: bool, echo_anchor: bool,
               prior_k: int = 5, sanity_retry: bool = False) -> str:
    # Canonical four tags used in Step 3.5 (implicit k=5, no sanity)
    if not scientist_priors and not novice_priors and echo_anchor:
        base = "baseline_echo"
    elif not scientist_priors and not novice_priors and not echo_anchor:
        base = "baseline_noecho"
    elif scientist_priors and not novice_priors and not echo_anchor:
        base = "meis_sci_noecho"
    elif scientist_priors and novice_priors and not echo_anchor:
        base = "meis_full_noecho"
    else:
        base = f"sci{int(scientist_priors)}_nov{int(novice_priors)}_echo{int(echo_anchor)}"
    # Only suffix prior_k when it differs from the default used in Step 3.5.
    if prior_k != 5 and (scientist_priors or novice_priors):
        base = f"{base}_k{prior_k}"
    if sanity_retry:
        base = f"{base}_sanity"
    return base


def run_mvp(seed: int, model_name: str = "gpt-5.4", *,
            env_name: str = "alice_charlie",
            scientist_priors: bool = False, novice_priors: bool = False,
            echo_anchor: bool = True,
            prior_k: int = 5,
            sanity_retry: bool = False,
            num_experiments: int = 10, num_evals: int = 10,
            com_limit: int = 200, include_prior: bool = True,
            temperature: float = 0.0, max_tokens: int = 512):

    random.seed(seed)
    np.random.seed(seed)

    env_cls, goal_cls = ENV_REGISTRY[env_name]
    prior_query = PRIOR_QUERIES[env_name]

    library = PriorLibrary.load_default()
    env = env_cls()
    env.include_prior = include_prior
    goal = goal_cls(env)

    scientist = _make_agent("prior_injected" if scientist_priors else "plain",
                            model_name=model_name, library=library,
                            prior_query=prior_query,
                            temperature=temperature, max_tokens=max_tokens,
                            prior_k=prior_k)
    novice = _make_agent("prior_injected" if novice_priors else "plain",
                         model_name=model_name, library=library,
                         prior_query=prior_query,
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

    predictions, gts, questions, sanity_retries = [], [], [], []
    goal.eval_pointer = 0
    for _ in range(num_evals):
        question, gt = goal.get_goal_eval_question(include_prior)
        if echo_anchor:
            question = final_results + "\n" + question
        pred, n_retry = _predict_with_sanity_retry(
            novice, question, env_name, enabled=sanity_retry,
        )
        predictions.append(pred)
        sanity_retries.append(n_retry)
        gts.append(gt)
        questions.append(question)

    err_mean, err_std = goal.evaluate_predictions(predictions, gts)

    tag = config_tag(scientist_priors, novice_priors, echo_anchor,
                     prior_k=prior_k, sanity_retry=sanity_retry)
    out_dir = RUNS_ROOT / env_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieved_ids_sci = (getattr(scientist, "last_prior_hits", None) or [])
    retrieved_ids_sci = [h["id"] for h in retrieved_ids_sci]
    retrieved_ids_nov = (getattr(novice, "last_prior_hits", None) or [])
    retrieved_ids_nov = [h["id"] for h in retrieved_ids_nov]

    store = {
        "config": {
            "seed": seed, "include_prior": include_prior, "use_ppl": False,
            "llms": {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens},
            "envs": {"env_name": env_name, "goal_name": "direct_discovery", "num_evals": num_evals},
            "exp": {"num_experiments": [num_experiments], "experiment_type": "discovery"},
            "meis": {
                "config_tag": tag,
                "scientist_priors": scientist_priors,
                "novice_priors": novice_priors,
                "echo_anchor": echo_anchor,
                "sanity_retry": sanity_retry,
                "sanity_range": SANITY_RANGES.get(env_name),
                "sanity_retries_per_eval": sanity_retries,
                "prior_query": prior_query,
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
    p.add_argument("--env", type=str, default="alice_charlie",
                   choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--scientist-priors", action="store_true")
    p.add_argument("--novice-priors", action="store_true")
    p.add_argument("--no-echo-anchor", action="store_true")
    p.add_argument("--prior-k", type=int, default=5)
    p.add_argument("--sanity-retry", action="store_true",
                   help="If predictions fall outside SANITY_RANGES, retry with warning.")
    args = p.parse_args()
    run_mvp(
        seed=args.seed, model_name=args.model, env_name=args.env,
        scientist_priors=args.scientist_priors,
        novice_priors=args.novice_priors,
        echo_anchor=not args.no_echo_anchor,
        prior_k=args.prior_k,
        sanity_retry=args.sanity_retry,
    )
