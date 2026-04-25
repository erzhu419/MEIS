"""Analysis of cross-family Peregrines results.

Loads cached per-seed JSONs from runs/peregrines_crossfamily/{baseline,meis_full}/
and applies the same 3 regex tiers used in P2.1 to the scientist's
final explanation. Reports per-tier rates + 1-sided Mann-Whitney U
of (MEIS rate > baseline rate).

Usage: python -m phase1_mvp.analyze_cross_family_peregrines
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from scipy.stats import mannwhitneyu


RUNS_DIR = Path(__file__).parent / "runs" / "peregrines_crossfamily"

# Same tiers as P2.1 (paper §3.2)
TIERS = {
    "rise_and_fall":      re.compile(r"rise\s+and\s+fall|peak.*fall|unimodal", re.I),
    "polynomial_form":    re.compile(r"polynomial|cubic|quadratic|\bt\^2|\bt\^3", re.I),
    "count_or_poisson":   re.compile(r"poisson|log[\s\-]?rate|\blambda\b|exp\(alpha|log[\s\-]?linear", re.I),
}


def _explanations_for_run(path: Path) -> str:
    d = json.loads(path.read_text())
    if "error" in d:
        return ""
    expls = d.get("data", {}).get("explanations", [])
    if not expls:
        return ""
    if isinstance(expls[0], str):
        return expls[0]
    return expls[0].get("text", str(expls[0]))


def score_condition(condition: str) -> dict:
    cdir = RUNS_DIR / condition
    if not cdir.exists():
        return {}
    rows = []
    for p in sorted(cdir.glob("seed_*.json")):
        text = _explanations_for_run(p)
        rows.append({tier: int(bool(rx.search(text))) for tier, rx in TIERS.items()})
    return {tier: [r[tier] for r in rows] for tier in TIERS}


def mwu_1sided(meis_vec, base_vec):
    """1-sided MWU testing meis > baseline."""
    if not meis_vec or not base_vec:
        return float("nan")
    try:
        _, p = mannwhitneyu(meis_vec, base_vec, alternative="greater")
    except ValueError:
        return float("nan")
    return float(p)


def main():
    print("Cross-family Peregrines analysis (gemini-3-flash scientist + novice)")
    print(f"Run dir: {RUNS_DIR}\n")
    base = score_condition("baseline")
    meis = score_condition("meis_full")
    if not base or not meis:
        print("Missing condition data — re-run after batch completes.")
        return

    n_base = len(next(iter(base.values()), []))
    n_meis = len(next(iter(meis.values()), []))
    print(f"n_baseline = {n_base},  n_meis_full = {n_meis}\n")

    print(f"{'tier':<22}  {'baseline':>9}  {'MEIS':>9}  {'1-sided MWU p':>16}")
    print("-" * 64)
    for tier in TIERS:
        b_rate = sum(base[tier]) / max(n_base, 1)
        m_rate = sum(meis[tier]) / max(n_meis, 1)
        p = mwu_1sided(meis[tier], base[tier])
        print(f"  {tier:<20}  {b_rate*100:>7.0f}%  {m_rate*100:>7.0f}%  {p:>16.4f}")

    print("\nReplication of P2.1 (GPT-family, n=32 per condition):")
    print("                      baseline   MEIS   p-value")
    print("  rise_and_fall          88%    100%   <0.001")
    print("  polynomial_form        12%     62%   =0.001")
    print("  count_or_poisson        0%     56%   <0.001")


if __name__ == "__main__":
    main()
