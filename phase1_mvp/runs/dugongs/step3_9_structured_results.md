# Step 3.9 — Structured-JSON scientist→novice channel (bypass NL bottleneck)

Follow-up to Step 3.7 (sanity-retry null) and Step 3.8 (scientist-side p<0.001
regex finding). The hypothesis was: **MEIS priors DO work at the scientist
level; the NL-transmission channel is where the benefit is erased**. So we
test by replacing the novice's NL-explanation input with a strict JSON model
distillation from the scientist.

## Setup

Added `--structured-channel` flag to `run_mvp_unified.py`. When on:
1. Scientist still writes NL explanation (kept for audit)
2. Scientist is then asked: `STRUCTURED_DISTILL_PROMPTS[env_name]` → strict
   JSON with keys {formula, parameters, explanation}
3. Novice's system message gets the JSON instead of the NL explanation
4. Novice predicts as before

Runs: 16 seeds × 2 configs on dugongs = 32. Both sanity-retry on, for fair
comparison with step 3.7 counterparts. JSON parse rate: 32/32 = 100%.

## Full 6-config dugongs comparison (n=16 each)

| config | winsor MAE | median AE | outlier rate |
|---|---|---|---|
| baseline_echo (Step 1) | 4.38 ± 2.02 | 3.12 | 43.8% |
| baseline_noecho | 0.65 ± 0.28 | 0.58 | 1.9% |
| baseline_noecho_sanity | 0.58 ± 0.33 | 0.52 | 0.0% |
| **baseline_noecho_sanity_structured** | **0.244 ± 0.07** | **0.217 ± 0.08** | **0.0%** |
| meis_full_noecho | 0.55 ± 0.47 | 0.41 | 3.8% |
| meis_full_noecho_sanity | 0.64 ± 0.42 | 0.58 | 0.0% |
| **meis_full_noecho_sanity_structured** | **0.239 ± 0.08** | **0.202 ± 0.09** | **0.0%** |

## Paired significance tests (n=16, one-sided Wilcoxon)

| comparison | mean Δ (winsor MAE) | B-wins / 16 | Wilcoxon p |
|---|---|---|---|
| baseline_sanity → baseline_structured | +0.334 | **14** | **0.0003** |
| meis_full_sanity → meis_full_structured | +0.398 | **14** | **0.0005** |
| baseline_structured vs meis_full_structured | +0.005 | 8 | 0.32 |

## Three key findings

### 1. **The bypass-NL hypothesis is confirmed (both configs)**

Replacing NL explanation with structured JSON cuts MAE by **~60% on both
baseline and MEIS-full** (p=0.0003 / p=0.0005, 14/16 wins each). The NL
channel was indeed the bottleneck.

### 2. **Under structured output, MEIS-full ≈ baseline (null result)**

p=0.32 (winsor) / 0.17 (median), 8/16 and 12/16 wins. MEIS priors do
not add **statistically significant** end-task accuracy when the scientist
is asked to emit JSON directly. Mean MAE 0.239 vs 0.244 — effectively tied.

### 3. **Why the tie — baseline catches up via JSON prompting**

When asked for JSON, even the baseline scientist produces reasonable
parametric forms on dugongs:
  - baseline seed=1 distilled: `L_inf - (L_inf - L0) * exp(-k*age)` with
    L_inf=2.25, L0=0.484, k=0.45 (von Bertalanffy form!)

gpt-5.4 has enough domain knowledge for dugong-like saturating growth
that it surfaces cleanly when the elicitation format is explicit. The
MEIS-injected priors don't add marginal knowledge — they reinforce what
gpt-5.4 already knows when asked right.

Interestingly, **MEIS-full seed=1 distilled a piecewise-linear
interpolation** instead (overfitting to the observations), possibly
because the rich prior context led the scientist to "trust" every data
point rather than trust the saturating-form prior. Sample of one, but
suggestive of a prior-vs-data trust tension.

## Updated MEIS value proposition (after Steps 3.5 → 3.9)

The original pitch was: **MEIS priors improve end-task accuracy**. Four
rounds of evidence now show this is ONLY true in a narrow regime:

|  | LLM has inherent domain knowledge | LLM lacks domain knowledge |
|---|---|---|
| **NL channel** | MEIS effects are drowned by NL noise | untested; hypothesized MEIS advantage |
| **Structured channel** | baseline catches up; **MEIS tied** | untested; hypothesized **biggest MEIS win** |

dugongs sits in the "LLM has knowledge + structured channel" quadrant,
which is exactly where MEIS shows no advantage. To see MEIS's
distinctive value, we need an env in the bottom-right: **LLM doesn't
know the functional form AND we use structured elicitation**.

## What this implies for C (cross-env expansion)

The useful cross-env envs are NOT the ones where baseline already
performs well. Candidates:

- `irt` (Item Response Theory): sigmoid response with ability/difficulty
  latents — gpt-5.4 might fake this
- `lotka_volterra`: 2D ODE system with predator-prey dynamics — genuinely
  nonlinear, LLMs may not extract the right ODE form from 10 noisy samples
- `emotion` / `moral_machines`: higher-dimensional, LLM-like latent tasks

Priority: **lotka_volterra** — genuinely novel functional form that
baselines in step 3.5 had more trouble with (saw Step 3.5 aside).

## Direction audit

1. **L-layer attribution** ✓ — structured channel is L4 (runner-level communication change), no L1-L3 drift.
2. **Trunk-free** ✓ — zero boxing-gym edits; lives in run_mvp_unified.py.
3. **Scope** ✓ — Option B from step3_7 proposal.

## Artifacts

- `phase1_mvp/run_mvp_unified.py` — `--structured-channel` flag + JSON distillation
- `phase1_mvp/runs/dugongs/baseline_noecho_sanity_structured/seed_{1..16}.json`
- `phase1_mvp/runs/dugongs/meis_full_noecho_sanity_structured/seed_{1..16}.json`
- 32 new LLM runs total, 0 crashes, 100% JSON parse rate
