"""Scientist-side MEIS evaluation: does the Scientist USE priors?

Rather than measure end-to-end MAE (dominated by Novice NL-bottleneck noise),
this analyzes whether the Scientist's natural-language explanation contains
the CORRECT functional form for each env. If MEIS priors are doing anything,
MEIS runs should mention the right form more often and more specifically
than baseline runs.

Zero-API: parses saved JSON files only.

Usage:
    python -m phase1_mvp.analysis.eval_scientist
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"


# Per-env: regex patterns for "correct functional form" content.
# Each pattern should match a specific semantic claim, not a generic word.
# Group into tiers so we can score "depth of correctness".
FORM_PATTERNS = {
    "alice_charlie": {
        "cube_law": [
            r"\bcube\b",
            r"\bcubic\b",
            r"h\^\s*3\b",
            r"height\s*\^\s*3\b",
            r"\(\s*height[^\)]*\)\s*\^\s*3",
            r"third\s+power",
            r"1\.4(?:\d+)?\s*(?:×|x|\*)\s*10\s*\^?\s*-\s*5",   # 1.414 × 10^-5
            r"0\.00001[45]",
        ],
        "density_concept": [
            r"\bdensity\b",
            r"body\s+density",
            r"1010\s*kg",
        ],
        "volume_concept": [
            r"\bvolume\b",
            r"k\s*[·*]\s*h",
        ],
    },
    "dugongs": {
        "saturating_shape": [
            r"\bsaturat",
            r"\basymptot",
            r"\bplateau",
            r"converg",
            r"limit(?:ing|s)?",
            r"levels?\s+off",
            r"diminish",
        ],
        "exponential_or_decay": [
            r"\bexponent",
            r"\bdecay",
            r"\bgamma\b",
            r"\blambda\b",
            r"von\s+bertalanffy",
            r"1\s*-\s*e\s*\^",
            r"alpha\s*-\s*beta",
            r"α\s*-\s*β",
        ],
    },
}


@dataclass
class RunSummary:
    env_name: str
    config_tag: str
    seed: int
    explanation: str
    has_scientist_priors: bool
    has_novice_priors: bool

    def match_counts(self) -> dict[str, int]:
        """Count regex-pattern matches per tier for this explanation."""
        patterns = FORM_PATTERNS.get(self.env_name, {})
        text = self.explanation.lower()
        out = {}
        for tier, regs in patterns.items():
            count = 0
            for rx in regs:
                count += len(re.findall(rx, text, re.IGNORECASE))
            out[tier] = count
        return out

    def has_any(self, tier: str) -> bool:
        return self.match_counts().get(tier, 0) > 0


def load_all_runs() -> list[RunSummary]:
    runs: list[RunSummary] = []

    # 1. runs/<env>/<config>/seed_*.json  — Step 3.5 and later
    for env_dir in sorted(RUNS_DIR.iterdir()):
        if not env_dir.is_dir() or env_dir.name in {"dugongs_noise_test"}:
            continue
        if env_dir.is_file():
            continue
        # Either runs/alice_charlie/<cfg>/seed_*.json or runs/dugongs/<cfg>/seed_*.json
        for cfg_dir in env_dir.iterdir():
            if not cfg_dir.is_dir():
                continue
            for jf in sorted(cfg_dir.glob("seed_*.json")):
                try:
                    d = json.load(open(jf))
                except Exception:
                    continue
                meis = d.get("config", {}).get("meis", {})
                env_name = d.get("config", {}).get("envs", {}).get("env_name", env_dir.name)
                expls = d.get("data", {}).get("explanations", [])
                if not expls:
                    continue
                seed = int(jf.stem.split("_")[1])
                runs.append(RunSummary(
                    env_name=env_name,
                    config_tag=cfg_dir.name,
                    seed=seed,
                    explanation=expls[0],
                    has_scientist_priors=bool(meis.get("scientist_priors", False)),
                    has_novice_priors=bool(meis.get("novice_priors", False)),
                ))

    # 2. baselines/step1_alice_charlie_*.json  (Step 1 baseline WITH echo)
    for jf in sorted(BASELINES_DIR.glob("step1_alice_charlie_gpt-5.4_*.json")):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        seed = int(jf.stem.split("_")[-1])
        if "_raw" in jf.stem:
            seed = int(jf.stem.split("seed")[1].split("_")[0])
        expls = d.get("data", {}).get("explanations", [])
        if not expls:
            continue
        runs.append(RunSummary(
            env_name="alice_charlie",
            config_tag="baseline_echo_step1",
            seed=seed,
            explanation=expls[0],
            has_scientist_priors=False,
            has_novice_priors=False,
        ))

    # 3. runs_step3/*.json (Step 3, MEIS scientist-only, WITH echo)
    rs3 = Path(__file__).resolve().parents[1] / "runs_step3"
    for jf in sorted(rs3.glob("step3_mvp_*.json")):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        seed = int(jf.stem.split("_")[-1])
        expls = d.get("data", {}).get("explanations", [])
        if not expls:
            continue
        runs.append(RunSummary(
            env_name="alice_charlie",
            config_tag="meis_sci_echo_step3",
            seed=seed,
            explanation=expls[0],
            has_scientist_priors=True,
            has_novice_priors=False,
        ))

    return runs


def group_stats(runs: list[RunSummary]) -> dict:
    """Aggregate per (env, config_tag)."""
    groups: dict[tuple[str, str], list[RunSummary]] = {}
    for r in runs:
        groups.setdefault((r.env_name, r.config_tag), []).append(r)

    out = {}
    for key, members in sorted(groups.items()):
        env_name, cfg = key
        tiers = list(FORM_PATTERNS.get(env_name, {}).keys())
        tier_match_rate = {}
        tier_mean_count = {}
        for t in tiers:
            matches = [r.has_any(t) for r in members]
            counts = [r.match_counts().get(t, 0) for r in members]
            tier_match_rate[t] = sum(matches) / len(members) if members else 0.0
            tier_mean_count[t] = sum(counts) / len(members) if members else 0.0
        expl_lens = [len(r.explanation) for r in members]
        out[key] = dict(
            n=len(members),
            avg_explanation_chars=(sum(expl_lens) / len(expl_lens)) if expl_lens else 0,
            tier_match_rate=tier_match_rate,
            tier_mean_count=tier_mean_count,
        )
    return out


def compare_two(a_runs: list[RunSummary], b_runs: list[RunSummary],
                env_name: str, tier: str, a_lbl: str, b_lbl: str) -> None:
    """Compare tier match counts between two configs via Mann-Whitney U."""
    a_counts = [r.match_counts().get(tier, 0) for r in a_runs]
    b_counts = [r.match_counts().get(tier, 0) for r in b_runs]
    if not a_counts or not b_counts:
        return
    # One-sided MW-U: H1 = b_counts > a_counts (MEIS mentions form MORE)
    try:
        u, p = stats.mannwhitneyu(a_counts, b_counts, alternative="less")
    except ValueError:
        u, p = 0, 1.0
    a_rate = sum(c > 0 for c in a_counts) / len(a_counts)
    b_rate = sum(c > 0 for c in b_counts) / len(b_counts)
    print(f"    {tier:>25}  {a_lbl:>22} ({a_rate*100:4.0f}%, mean={sum(a_counts)/len(a_counts):.2f})  "
          f"vs  {b_lbl:>22} ({b_rate*100:4.0f}%, mean={sum(b_counts)/len(b_counts):.2f})  "
          f"MW-U p={p:.3f} (1-sided)")


def main():
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs with explanations\n")

    stats_by_group = group_stats(runs)

    # Top summary table
    print(f'{"env":>16}  {"config":>28}  {"n":>3}  {"avg chars":>9}  tier match-rate (%)')
    print('-' * 104)
    for (env_name, cfg), st in sorted(stats_by_group.items()):
        tiers = list(FORM_PATTERNS.get(env_name, {}).keys())
        tier_summary = "  ".join(f'{t}={int(100*st["tier_match_rate"][t]):>3}%' for t in tiers)
        print(f'{env_name:>16}  {cfg:>28}  {st["n"]:>3}  {int(st["avg_explanation_chars"]):>9}  {tier_summary}')

    # Per-env MEIS-signal hunt
    print('\n=== Per-env MEIS-signal hunt: does scientist_priors lift tier-match rate? ===')

    # Group runs by env
    by_env = {}
    for r in runs:
        by_env.setdefault(r.env_name, []).append(r)

    for env_name, env_runs in sorted(by_env.items()):
        print(f'\n--- {env_name} ---')
        # We compare baseline (no scientist priors) vs MEIS (with scientist priors).
        # Both groups can include with- or without-echo / sanity / etc.
        base_runs = [r for r in env_runs if not r.has_scientist_priors]
        meis_runs = [r for r in env_runs if r.has_scientist_priors]
        tiers = list(FORM_PATTERNS.get(env_name, {}).keys())
        for t in tiers:
            compare_two(base_runs, meis_runs, env_name, t,
                        "baseline (no sci prior)", "MEIS (sci prior)")


if __name__ == "__main__":
    main()
