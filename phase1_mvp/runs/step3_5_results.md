# Step 3.5 — Echo-fix + MEIS-full on alice_charlie (16 seeds × 3 configs)

Decision context (from prior Step 3 negative result): echo-anchor in eval
prompts was dominating noise, hiding MEIS signal. Also unclear whether
scientist-only priors are enough or if the novice also needs them.

This iteration does both:
  (A) drop the `"The final result is <obs_last>"` eval-prompt prefix
  (B) test MEIS-full (both agents get priors) vs MEIS-scientist-only

## Setup

| Key | Value |
|---|---|
| env | `alice_charlie` |
| LLM | gpt-5.4 via ruoli.dev/v1 |
| echo anchor | **OFF for all three configs** |
| seeds | 1–16 |
| Scientist priors | as per config |
| Novice priors | as per config (only in `meis_full_noecho`) |
| Retrieval | `prior_query="predict weight given height"`, `k=5` |

Same unified runner (`phase1_mvp/run_mvp_unified.py`) drives all three —
no code drift between conditions.

## Headline metrics (n=16 each)

| config | MAE mean ± std | MSE | \|bias\| mean | echo rate |
|---|---|---|---|---|
| baseline_WITH_echo (old Step 1) | 4.38 ± 2.02 | 38.5 ± 26.7 | 2.03 | 43.8% |
| **baseline_noecho** | **2.41 ± 1.01** | 9.01 ± 6.25 | 1.72 | 3.1% |
| meis_sci_noecho | 2.73 ± 1.01 | 10.72 ± 7.33 | 2.26 | 4.4% |
| **meis_full_noecho** | **1.95 ± 0.59** | **6.11 ± 3.44** | **1.20** | 6.2% |

## Paired comparisons (per-seed MAE, n=16)

| A → B | mean Δ (kg) | wins-of-B / 16 | t-test p (1-sided) | Wilcoxon p (1-sided) |
|---|---|---|---|---|
| baseline_WITH_echo → baseline_noecho | +1.97 | **14** | **0.0003** | **0.0001** |
| baseline_noecho → meis_sci_noecho | −0.32 | 5 | 0.22 (not sig) | 0.78 (not sig) |
| meis_sci_noecho → meis_full_noecho | +0.79 | **12** | **0.0034** | **0.0026** |
| baseline_noecho → meis_full_noecho | +0.47 | 10 | **0.0649** | 0.0964 |

## Per-seed MAE table

| seed | WITH_echo | noecho | meis_sci | meis_full |
|---|---|---|---|---|
| 1 | 6.07 | 3.14 | 1.99 | **1.14** |
| 2 | 7.37 | 2.47 | 3.24 | **1.69** |
| 3 | 7.03 | 2.40 | 3.12 | 2.54 |
| 4 | 2.82 | **2.68** | 3.73 | 2.59 |
| 5 | 3.80 | **1.54** | 5.07 | 1.68 |
| 6 | 5.85 | 1.53 | 2.59 | **1.35** |
| 7 | 5.05 | 3.62 | **1.71** | 2.33 |
| 8 | 1.45 | **1.36** | 1.95 | 1.82 |
| 9 | 2.85 | **2.31** | 2.52 | 2.61 |
| 10 | 1.62 | **1.69** | 4.34 | 3.24 |
| 11 | 6.23 | 4.54 | 2.23 | **1.31** |
| 12 | 1.73 | **1.74** | 2.06 | 1.74 |
| 13 | 6.32 | 1.94 | 2.15 | **1.75** |
| 14 | 5.09 | **4.30** | 2.26 | 2.29 |
| 15 | 4.29 | 1.56 | 3.42 | **1.32** |
| 16 | 2.43 | **1.74** | 1.35 | 1.73 |

## Three findings

### 1. The echo artifact was the dominant noise source

Removing the `"The final result is X"` prefix from novice eval prompts
(zero MEIS changes) cut baseline MAE from 4.38 → 2.41 (**−45%**) and echo
rate from 43.8% → 3.1%. Highly significant: t-test p=0.0003, Wilcoxon
p=0.0001, 14/16 seed wins.

**This validates the user's "validate baseline is good before continuing"
discipline.** Without fixing echo, we could not have seen MEIS signal.

### 2. Scientist-only prior injection does NOT help

`meis_sci_noecho` (2.73) vs `baseline_noecho` (2.41): scientist-only
priors *marginally hurt* — 5/16 wins, p=0.22 (not significant). The
hypothesis from Step 3 is confirmed: the scientist→novice natural-language
channel is a lossy bottleneck. Rich formulas in the scientist's head don't
reach the novice's predictions.

### 3. MEIS-full significantly beats scientist-only; marginally beats baseline

`meis_full_noecho` (1.95) vs `meis_sci_noecho` (2.73): **p=0.003**, 12/16
wins. Giving priors to the novice directly is where the MEIS value lives.

`meis_full_noecho` vs `baseline_noecho`: **p=0.065**, 10/16 wins. Real
signal but marginal on this env. Importantly, std drops from 1.01 → 0.59
(**42% tighter**), meaning MEIS is not just slightly-better-on-average but
substantially **more stable across seeds**.

## Interpretation — marginal win on alice_charlie is suspect

alice_charlie is **our own env**: we designed the ground-truth generative
model, and then built a prior library that matches it. There is no
mechanism in this pipeline that proves MEIS generalizes beyond priors
we authored. The marginal p=0.065 result on an env where baseline
already sits near the noise floor (MAE 2.41 kg ≈ noise-σ of 2 kg) is
**not strong enough** to call a success.

Also: baseline_noecho's MAE of 2.41 kg on weights averaging 70 kg is
only 3.4% relative error — gpt-5.4 already extracts most of the signal
without MEIS. The ceiling is too low for clean differentiation.

## Next step — pivot to dugongs (per user's fallback rule)

Following the user's discipline "如果这次还不如baseline，那就从纯boxing-gym的项目开始":

| Step | env | config | Notes |
|---|---|---|---|
| 3.5-D0 | **dugongs** | pure boxing-gym, None-guard + noecho | re-run the 16-seed baseline that crashed 1/5 last time |
| 3.5-D1 | dugongs | + scientist priors (growth-curve + allometry entries added to library) | tests that priors transfer to a foreign env |
| 3.5-D2 | dugongs | + scientist + novice priors | MEIS-full on dugongs |

Why dugongs:
- boxing-gym-owned env → no authoring bias risk
- dugongs Step 0 baseline MAE was 0.54 at ground truth ≈ 1.1 (49% relative)
  → plenty of room for MEIS to show an effect
- adding Gompertz/allometry priors to the library tests the extensibility
  part of Phase 2 that has not been exercised yet

## Direction audit (blueprint §4.4)

1. **L-layer attribution** ✓ — echo fix lives in L4 runner; MEIS-full is still L3+L4 only
2. **Trunk-free** ✓ — zero boxing-gym edits in this step (echo fix is in our own `run_mvp_unified.py`, not in boxing-gym's `run_experiment.py`)
3. **Scope** ✓ — did exactly what blueprint §4.3 Step 3 says plus the echo-fix fallback path
