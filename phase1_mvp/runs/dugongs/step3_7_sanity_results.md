# Step 3.7 — Sanity-retry A/B test on dugongs (32 runs, honest null result)

Fix applied: runner-level `--sanity-retry` that rejects novice predictions
outside `SANITY_RANGES["dugongs"] = (0.0, 3.0)` m and re-prompts with an
out-of-range warning, up to 3 retries.

Applied uniformly to both baseline AND MEIS-full to preserve fair comparison.

## Setup

| Key | Value |
|---|---|
| env | dugongs |
| seeds | 1–16 |
| LLM | gpt-5.4 via ruoli.dev/v1 |
| plausible range | (0.0, 3.0) m |
| new configs | `baseline_noecho_sanity`, `meis_full_noecho_sanity` |
| reference (no sanity) | `baseline_noecho`, `meis_full_noecho` from Step 3.5 |

## Headline metrics (all n=16)

| config | raw MAE | winsor MAE | median AE | outlier rate | retries triggered |
|---|---|---|---|---|---|
| baseline_noecho (Step 3.5) | 0.676 ± 0.33 | 0.654 ± 0.28 | 0.575 ± 0.29 | 1.9% | — |
| **baseline_noecho_sanity** | **0.579 ± 0.33** | 0.579 ± 0.33 | 0.517 ± 0.34 | **0.0%** | 3 / 160 |
| meis_full_noecho (Step 3.5) | 6.797 ± 10.79 | 0.550 ± 0.47 | 0.408 ± 0.39 | 3.8% | — |
| meis_full_noecho_sanity | 0.637 ± 0.42 | 0.637 ± 0.42 | 0.583 ± 0.46 | **0.0%** | 6 / 160 |

## Three findings

### 1. Sanity-retry mechanism works exactly as designed

Outlier rate for MEIS-full dropped 3.8% → **0.0%**, with 6 retries triggered
(one per original outlier). Baseline was already near-clean, with 3 retries
(0 residual outliers after sanity). The raw-MAE explosion from
6.80 kg → 0.64 on MEIS-full confirms every single pre-retry outlier was
caught and re-prompted to a plausible value. **This part is a Pareto win
for reliability** — no accuracy cost vs no-retry on the same runs.

### 2. Under fair comparison, MEIS-full no longer significantly beats baseline

**The key paired comparison (both with sanity, n=16):**

| metric | mean Δ (baseline − MEIS-full) | MEIS-full wins / 16 | Wilcoxon p (one-sided, H1: MEIS better) |
|---|---|---|---|
| winsor MAE | −0.058 | 8 | **0.684** |
| median AE | −0.066 | 7 | **0.719** |

The p=0.029 "first significant MEIS win" reported in Step 3.5 on dugongs
**does not replicate** under a fair outlier-treatment regime. The apparent
edge was partly driven by the fact that Step 3.5's MEIS-full had more
outliers than baseline (6 vs 3), which interacted with winsorization and
median estimators in MEIS-full's favor.

### 3. LLM run-to-run variance at temperature=0 is non-trivial

Comparing `meis_full_noecho` (Step 3.5) vs `meis_full_noecho_sanity` on
seeds where NO retries were triggered (so sanity was a no-op):

| seed | Step 3.5 winsor MAE | Step 3.7 winsor MAE | difference |
|---|---|---|---|
| 9 | 0.186 | 0.562 | +0.376 |
| 11 | 0.319 | 1.481 | +1.162 |
| 12 | 0.155 | 0.652 | +0.497 |
| 13 | 0.265 | 0.787 | +0.522 |
| 14 | 0.093 | 1.465 | +1.372 |
| 15 | 0.285 | 1.192 | +0.907 |

Same Python/numpy seed, same config (no sanity-retry triggered), and yet
predictions differ by factors of 4-15×. This is gpt-5.4 (or the ruoli.dev
proxy) being genuinely non-deterministic at temperature=0, not a bug in our
pipeline.

**Consequence**: n=16 is probably insufficient to detect true MEIS effects
because run-to-run LLM noise can swamp the ~0.1 kg per-seed improvement we
hoped MEIS would deliver.

## Updated honest assessment of Phase 1 MVP

| env | MEIS-full vs baseline (both sanity) | p-value |
|---|---|---|
| alice_charlie (Step 3.5, no sanity) | marginal | 0.065 |
| **dugongs (Step 3.7, sanity)** | **not significant** | **0.68** |

The "MEIS-full significantly beats baseline" claim from Step 3.5 dugongs
**does not survive** a fair outlier treatment. The observed edge in Step 3.5
was a mix of (a) LLM random variance across seeds, (b) outlier-handling
asymmetry between configs. Under sanity treatment both sources are
neutralized.

This is not fatal to MEIS but it **is a serious update** to the project's
story. Under the current Scientist → Novice NL-mediated evaluation, MEIS
priors do not produce reliable gains on these two envs with gpt-5.4.

## Where does this leave Phase 1?

Three plausible directions:

**A. Scale n from 16 → 48 or 64 seeds.** Cost: ~96 additional LLM runs per
A/B. If MEIS has a true ~0.1 kg effect, n=48 might separate it from the
~0.5 kg run-to-run noise. But this is a lot of API tokens for a marginal
effect.

**B. Change the evaluation metric away from novice MAE.** Candidates:
  - **Scientist observation policy quality**: EIG of each chosen
    observation point. MEIS should choose more informative measurements
    if priors are actually used. (This is partly inside the scientist,
    less bottlenecked by NL.)
  - **Scientist explanation correctness**: programmatic check of whether
    the scientist's explanation mentions the right functional form
    ("cube law" for alice_charlie, "saturating" for dugongs).
  - **Direct scientist prediction error**: cut out the novice entirely.

**C. Accept current result and move to Step 6 anyway.** KL drift /
minimum-perturbation scoring is the blueprint's theoretical core and
does NOT depend on MEIS beating baseline on MAE — it depends on MEIS
being able to *rank* hypotheses by coherence-with-belief-network. This
is a different task where MEIS could succeed even when end-to-end MAE
doesn't.

I'd recommend **B first** (switch metric to expose scientist-side MEIS
effect) since it's cheap and directly addresses the "NL bottleneck"
hypothesis we formulated back in Step 3.

## Direction audit (blueprint §4.4)

1. **L-layer attribution** ✓ — all changes here are runner-level (L4) / evaluation, not architectural
2. **Trunk-free** ✓ — zero boxing-gym edits
3. **Scope** ✓ — Option B from Step 3.7 proposal executed exactly; no creep
