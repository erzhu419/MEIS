# Step 3 MVP — Prior-Injected Scientist on alice_charlie (16 seeds)

**Honest negative result**: MEIS prior injection as-is does not beat baseline,
and the per-seed variance + root-cause analysis show *both* configurations
are ceiling-limited by the same boxing-gym echo-anchoring artifact.
Do NOT proceed to Step 4 without first fixing the echo artifact.

## Setup

| Key | Value |
|---|---|
| env | `alice_charlie` (height→weight, θ·h³ + N(0,2)) |
| LLM | gpt-5.4 via ruoli.dev/v1 |
| Scientist | `PriorInjectingExperimenter`, `prior_k=5`, query=`"predict weight given height"` |
| Novice | plain `LMExperimenter` (no priors) |
| Retrieved priors (same for all 16 seeds) | `[weight_from_height_cube_law, bmi_adult_population, human_body_volume_from_height, adult_weight_distribution, human_density_adult]` |
| seeds | 1–16 (all successful) |

Scientist system message is ~2966 chars: original env prompt + 2000-char prior block with cube-law formula and three component priors.

## Headline numbers (MEIS vs Step 1 baseline, n=16 each)

| metric | baseline | MEIS | Δ | verdict |
|---|---|---|---|---|
| MAE (kg) | 4.38 ± 2.02 | 4.85 ± 2.54 | +0.47 | **worse** |
| MSE | 38.5 ± 26.7 | 56.1 ± 61.2 | +17.6 | **worse** |
| RMSE | 5.74 | 6.70 | +0.96 | worse |
| Skill score (MAE/σ₀=7.45) | 0.59 | 0.65 | +0.06 | worse |
| Mean signed bias | +1.16 | +0.08 | near-zero | neutral-ish |
| Std of bias across seeds | 2.52 | 4.21 | +1.69 | **MEIS more variable** |
| \|bias\| mean | 2.03 | 2.77 | +0.74 | worse |
| Echo rate (preds within ±0.5 kg of obs_last) | 43.8% | **50.0%** | +6.2pp | **worse** |
| Crash rate | 0% | 0% | — | tied |

**Paired per-seed comparison**:
- MEIS improved over baseline: **5 / 16** seeds
- MEIS worsened vs baseline: **11 / 16** seeds
- Median Δ MAE: −0.57 kg (negative = worse)
- Paired t-test (H1: baseline > MEIS): t=−0.61, **p=0.28** (not significant)
- Wilcoxon signed-rank: **p=0.78** (clearly not an improvement)

## Per-seed table

| seed | base MAE | MEIS MAE | Δ | base bias | MEIS bias |
|---|---|---|---|---|---|
| 1 | 6.07 | **3.45** | −2.62 ✓ | +4.34 | +2.04 |
| 2 | 7.37 | **2.37** | −5.00 ✓ | +0.19 | +2.22 |
| 3 | 7.03 | **3.96** | −3.07 ✓ | +3.16 | −0.72 |
| 4 | **2.82** | 3.70 | +0.87 | +1.30 | −0.79 |
| 5 | **3.80** | **11.77** | +7.97 ✗✗ | +0.76 | +11.77 |
| 6 | **5.85** | 7.49 | +1.63 | −0.30 | −7.31 |
| 7 | 5.05 | **1.54** | −3.51 ✓✓ | +0.27 | +0.40 |
| 8 | **1.45** | 4.31 | +2.86 | −0.03 | +0.30 |
| 9 | **2.85** | 4.22 | +1.37 | −1.76 | +2.29 |
| 10 | **1.62** | 4.50 | +2.88 | +1.14 | −0.38 |
| 11 | 6.23 | 6.01 | −0.22 ✓ | +5.36 | −1.85 |
| 12 | **1.73** | 2.53 | +0.80 | +0.35 | −0.51 |
| 13 | 6.32 | 6.33 | +0.01 | −4.90 | −4.70 |
| 14 | 5.09 | 5.23 | +0.14 | +4.32 | −4.00 |
| 15 | **4.29** | 7.44 | +3.15 | +2.58 | +3.77 |
| 16 | **2.43** | 2.77 | +0.34 | +1.72 | −1.30 |

## Root cause: echo-anchoring dominates

Looking at best/worst/typical seeds in detail:

**seed=7 (MEIS wins big, 5.05 → 1.54)**
- Scientist wrote: "weight (kg) ≈ 1.43 × 10⁻⁵ × height(cm)³" + 5 worked examples
- Novice applied the formula; errors mostly within ±4 kg; most predictions unique (not echoes)

**seed=5 (MEIS catastrophic, 3.80 → 11.77)**
- Scientist's explanation: essentially identical cube-law formula
- obs_last = 83.61 kg (height 182.5)
- Novice predictions: `[83.61, 83.61, 83.61, 83.61, 71.80, 83.61, 83.61, 83.61, 54.45, 83.61]`
- **8 out of 10 predictions are exactly 83.61** — the echo-anchoring artifact from Step 0 striking HARDER than in baseline

**seed=1 (MEIS wins moderately, 6.07 → 3.45)**
- Scientist wrote very good cube-law + "70 × (Height/170)³" mental version
- But 5 of 10 novice predictions are still exactly 67.96 (obs_last)

### Why MEIS makes echo WORSE, not better

In the runner (inherited pattern from boxing-gym/run_experiment.py line 129 and [phase1_mvp/run_step1_baseline.py:73](../run_step1_baseline.py#L73)):

```python
final_results = f"The final result is {observation}."
question = final_results + "\n" + question
```

Every one of novice's 10 eval prompts is prefixed with `"The final result is <obs_last>.\n"`. When the scientist's explanation is purely qualitative (baseline case), the novice has no structured fallback and mixes echo with guesswork. When the scientist's explanation is a precise formula (MEIS case), the novice has TWO authoritative-looking sources — the formula AND the prompt-stated "final result". Counterintuitively, gpt-5.4 more often DEFERS to the prompt-stated value when the system message is richer.

Echo rate jumped from 43.8% → 50.0% under MEIS.

## Direction audit (blueprint §4.4)

1. **L-layer attribution** ✓ — Step 3 change lives entirely in L4 (scientist's system prompt augmentation). Nothing leaked into L1/L2/L3.
2. **Trunk-free** ✓ — `PriorInjectingExperimenter` subclasses `LMExperimenter`; zero boxing-gym edits beyond the pre-existing Step 0 model-routing + Step 1 None-guard patches.
3. **Scope** ✓ — only did what blueprint §4.3 Step 3 says.

## Decision: don't proceed to Step 4 until echo artifact is fixed

This is exactly the failure mode the user warned about: **"验证通过才行，之前的代码要优化到把 baseline 好才继续"**. Moving to Step 4 (Fisher-info observation selection, KL drift, etc.) on top of an echo-ceilinged baseline would pile new uncertainty on top of noise we already know is there.

### Two candidate fixes for next iteration

**(A)** Drop the `"The final result is X"` prefix from eval prompts. Replace with a neutral transition like `"Now answer the following prediction question:"`. Faithful to what boxing-gym's `evaluate` function intends but without the numerical anchor.

**(B)** Inject priors into the novice too (MEIS-full). Removes the novice's reliance on the scientist's natural-language explanation as the sole conduit for structured knowledge.

**(A)** is a 1-line fix, doesn't change the MEIS architecture, and gives a clean fair comparison against Step 1 baseline (also echo-limited). **Need to do both baseline AND MEIS re-runs** — so 32 new gpt-5.4 runs.

**(B)** requires a small refactor but tests whether MEIS's value is in the prior itself or the scientist→novice NL bottleneck.

Recommended order: **(A) first** to establish a clean baseline, then decide whether MEIS + (A) is enough or whether **(B)** is also needed.
