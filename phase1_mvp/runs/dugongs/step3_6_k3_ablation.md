# Step 3.6 — prior_k=3 ablation on MEIS-full dugongs

Follow-up to Step 3.5 which identified a 3.8% outlier rate in MEIS-full
(vs 1.9% baseline) caused by LLM glitches under rich prompts. This
ablation tests whether reducing retrieved priors 5 → 3 shrinks prompt
size enough to eliminate glitches, and what it costs in accuracy.

## Setup

| Key | Value |
|---|---|
| env | `dugongs` |
| config | `meis_full_noecho` (scientist + novice priors, no echo anchor) |
| prior_k | **3** (was 5 in Step 3.5) |
| seeds | 1–16 launched, **12 / 16 succeeded** (4 hit proxy HTTP 402 "insufficient credits") |

Usable seeds: {1, 2, 3, 6, 8, 9, 11, 12, 13, 14, 15, 16}

## Results on the common 12 seeds

| metric | k=5 (Step 3.5 subset) | k=3 | Δ |
|---|---|---|---|
| raw MAE | 7.57 ± 11.77 (outlier-dominated) | 0.63 ± 0.36 | — (raw not comparable) |
| winsorized MAE (clip>5→5) | 0.58 ± 0.52 | **0.63 ± 0.36** | +0.05 (slightly worse) |
| median absolute error | **0.41 ± 0.42** | 0.63 ± 0.36 | +0.22 (k=3 worse) |
| **outlier rate (pred > 5)** | 4.2% (5/120) | **0.0% (0/120)** | **−4.2pp ✓** |

**Paired tests (two-sided, n=12):**

| metric | t-test p | Wilcoxon p |
|---|---|---|
| winsorized MAE | 0.816 | 0.733 |
| median AE | 0.293 | 0.233 |

No significant difference on accuracy; the outlier difference is categorical.

## Per-seed winsorized MAE

| seed | k=5 | k=3 | Δ | k=5 outliers |
|---|---|---|---|---|
| 1 | 0.919 | **0.493** | −0.43 | 1 |
| 2 | 1.875 | **0.272** | −1.60 | 1 |
| 3 | 0.869 | **0.409** | −0.46 | 1 |
| 6 | 1.054 | **0.526** | −0.53 | 2 |
| 8 | 0.520 | **0.362** | −0.16 | 0 |
| 9 | **0.186** | 1.009 | +0.82 | 0 |
| 11 | **0.319** | 0.965 | +0.65 | 0 |
| 12 | **0.155** | 0.749 | +0.59 | 0 |
| 13 | **0.265** | 0.750 | +0.49 | 0 |
| 14 | **0.093** | 1.437 | +1.34 | 0 |
| 15 | **0.285** | 0.239 | −0.05 | 0 |
| 16 | **0.361** | 0.333 | −0.03 | 0 |

## Interpretation: k=3 is a reliability / accuracy trade, not a Pareto win

k=3 **cleanly eliminates** LLM glitches (5 → 0). But on the seeds where k=5
ran glitch-free, k=3 is **noticeably worse** because the novice now lacks
two of the supporting priors (`adult_weight_distribution`,
`human_density_adult` or their dugongs equivalents) and falls back on
weaker interpolation between the observations.

Mean winsorized MAE is statistically indistinguishable (p=0.73), so k=3 is
neither clearly better nor worse than k=5 at the aggregate level.

**Decision**: do NOT adopt k=3 as default. The k=5 outlier rate is
acceptable when robust metrics (median, winsorized) are used for
evaluation, and k=5 retains the per-seed peak performance.

## Better follow-up candidates (not yet run)

- **k=5 + output sanity check**: in the runner, reject predictions
  outside [plausible_min, plausible_max] (per-env bounds) and
  force a retry with an "out-of-range" warning. 1-line runner change,
  would probably drop outlier rate with zero accuracy cost.
- **k=5 + trimmed `formal` blocks**: remove the parametric numbers from
  the retrieved prior chunks (keep only the natural-language statement +
  relation). Reduces token count while preserving structural info.
- **k=4**: middle ground, another 16 runs.

None of these are blocking Step 6 (KL drift / minimum perturbation scoring).
The main Step 3.5 conclusion (MEIS-full beats baseline on dugongs at
p=0.029 median AE) stands.

## Non-run note

Seeds 4/5/7/10 hit `openai.APIStatusError: 402` from the ruoli.dev proxy
mid-run (credits exhausted). Topping up and retrying those 4 would bring
us to n=16 and strengthen the comparison slightly, but with n=12 the
ablation is already conclusive: k=3 doesn't win.
