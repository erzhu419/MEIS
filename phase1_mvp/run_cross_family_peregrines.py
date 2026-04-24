"""Cross-family replication of P2.1 Peregrines MEIS effect.

Reuses run_mvp_unified.run_mvp() but injects the Gemini endpoint via
OPENAI_API_KEY / OPENAI_BASE_URL so the OpenAI-compatible client in
boxing_gym/agents/agent.py routes through ruoli.dev.

Two conditions × N seeds per condition:
  baseline  : no --scientist-priors, no --novice-priors
  meis_full : both priors on, --no-echo-anchor

Output: phase1_mvp/runs/peregrines_crossfamily/{condition}/seed_{n}.json

Post-processing: apply same regex tiers as P2.1 and run Mann-Whitney U
against the existing GPT-side results stored in
runs/peregrines/meis_full_noecho/ and runs/peregrines/baseline_noecho/.

Honest scope: we use one Gemini model (gemini-3-flash, the cheap tier).
A second Gemini model or Claude would strengthen the claim; those are
Appendix-scope.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Must set env vars BEFORE run_mvp_unified imports openai downstream
# (openai.OpenAI() captures env at construction time).
from _meis_keys import GPT_KEY, GEMINI_KEY, RUOLI_BASE_URL

MODEL_NAME = os.environ.get("MEIS_CROSS_FAMILY_MODEL", "gemini-3-flash")
API_KEY = GEMINI_KEY if MODEL_NAME.startswith("gemini") else GPT_KEY

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = RUOLI_BASE_URL
# Some boxing_gym code expects OPENAI_API_KEY only; BASE_URL is the
# addition ruoli.dev needs.

from phase1_mvp.run_mvp_unified import run_mvp, config_tag


OUT_ROOT = Path(__file__).parent / "runs" / "peregrines_crossfamily"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(seed: int, condition: str):
    """condition ∈ {'baseline', 'meis_full'}; both use --no-echo-anchor
    to match the original P2.1 Peregrines protocol exactly."""
    if condition == "baseline":
        kwargs = dict(scientist_priors=False, novice_priors=False,
                       echo_anchor=False)
    elif condition == "meis_full":
        kwargs = dict(scientist_priors=True, novice_priors=True,
                       echo_anchor=False)
    else:
        raise ValueError(condition)

    out_dir = OUT_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.json"
    if out_path.exists():
        print(f"  (cached)  {out_path.name}")
        return out_path

    print(f"  running   {condition} seed={seed} model={MODEL_NAME} ...")
    t0 = time.time()
    try:
        result = run_mvp(
            seed=seed, model_name=MODEL_NAME, env_name="peregrines",
            scientist_priors=kwargs["scientist_priors"],
            novice_priors=kwargs["novice_priors"],
            echo_anchor=kwargs["echo_anchor"],
            prior_k=5,
            sanity_retry=True,
            structured_channel=False,
            persist_belief_path=None,
        )
    except Exception as e:
        print(f"  FAILED    seed={seed}: {e}")
        out_path.write_text(json.dumps({"error": str(e)}))
        return out_path
    elapsed = time.time() - t0
    # run_mvp returns a dict-like result and also writes its own files;
    # we explicitly save a copy with our naming so downstream analysis
    # doesn't depend on run_mvp's side-channel.
    try:
        out_path.write_text(json.dumps(result, indent=2, default=str))
    except Exception as e:
        out_path.write_text(json.dumps({"raw_repr": repr(result),
                                          "write_error": str(e)}))
    print(f"  done      seed={seed} ({elapsed:.0f}s)")
    return out_path


def main(n_seeds: int = 16):
    seeds = list(range(n_seeds))
    for condition in ("baseline", "meis_full"):
        print(f"\n=== Condition: {condition} (n={n_seeds}) ===")
        for s in seeds:
            run_one(s, condition)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    main(n_seeds=n)
