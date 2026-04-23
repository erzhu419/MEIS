# Step 1 Baseline — alice_charlie (gpt-5.4, 16 seeds)

**Setup**

| Key | Value |
|---|---|
| env | `alice_charlie` (height→weight, θ · h³ + N(0, 2)) |
| LLM | gpt-5.4 via ruoli.dev/v1 proxy |
| mode | discovery (Scientist→Novice game), `num_experiments=10`, `num_evals=10` |
| `include_prior` | true |
| `use_ppl` | false |
| seeds | 1–16 (all successful; 0 crashes after agent.py None-guard patch) |
| MEIS changes | none beyond the env itself + one robustness patch to boxing-gym/agent.py |

**Result JSONs**: `phase1_mvp/baselines/step1_alice_charlie_gpt-5.4_{1..16}.json`
**Runner**: `phase1_mvp/run_step1_baseline.py` (minimal reimpl of discovery loop; no boxing-gym trunk changes needed)

## Per-seed metrics (n=16)

| seed | MSE | RMSE | MAE (kg) | Bias (kg) | obs_last (kg) | echo (preds within ±0.5kg of obs_last) |
|---|---|---|---|---|---|---|
| 1 | 52.35 | 7.23 | 6.07 | +4.34 | 65.32 | 8/10 |
| 2 | 80.27 | 8.96 | 7.37 | +0.19 | 70.79 | 5/10 |
| 3 | 73.08 | 8.55 | 7.03 | +3.16 | 70.94 | 4/10 |
| 4 | 12.65 | 3.56 | 2.82 | +1.30 | 67.55 | 3/10 |
| 5 | 35.76 | 5.98 | 3.80 | +0.76 | 69.21 | 7/10 |
| 6 | 60.71 | 7.79 | 5.85 | −0.30 | 71.16 | 9/10 |
| 7 | 40.88 | 6.39 | 5.05 | +0.27 | 76.01 | 3/10 |
| 8 | **3.18** | **1.78** | **1.45** | −0.03 | 93.34 | 0/10 |
| 9 | 18.98 | 4.36 | 2.85 | −1.76 | 54.06 | 4/10 |
| 10 | 3.79 | 1.95 | 1.62 | +1.14 | 73.46 | 2/10 |
| 11 | 65.13 | 8.07 | 6.23 | +5.36 | 73.09 | 7/10 |
| 12 | 5.81 | 2.41 | 1.73 | +0.35 | 78.55 | 0/10 |
| 13 | 74.94 | 8.66 | 6.32 | −4.90 | 70.98 | 6/10 |
| 14 | 40.91 | 6.40 | 5.09 | +4.32 | 85.83 | 3/10 |
| 15 | 29.33 | 5.42 | 4.29 | +2.58 | 71.06 | 6/10 |
| 16 | 18.68 | 4.32 | 2.43 | +1.72 | 73.03 | 3/10 |

## Pooled summary (n=16)

| Metric | mean | std | SE | min | max |
|---|---|---|---|---|---|
| MSE (kg²) | **38.53** | 26.75 | 6.69 | 3.18 | 80.27 |
| RMSE (kg) | 5.74 | 2.44 | 0.61 | 1.78 | 8.96 |
| MAE (kg) | **4.38** | 2.02 | 0.51 | 1.45 | 7.37 |
| Bias (kg) | **+1.16** | 2.52 | 0.63 | −4.90 | +5.36 |
| Skill score (MAE / marginal_σ = 7.45) | **0.59** | 0.27 | — | — | — |
| Echo rate (preds within 0.5 kg of obs_last) | — | — | — | — | **70 / 160 = 43.8%** |
| NaN predictions (parse failures) | — | — | — | — | 0 / 160 |
| Crash rate | — | — | — | — | 0 / 16 |

Skill score < 1 means novice beats the marginal-mean predictor; baseline already extracts real signal from the scientist's explanation. But MAE = 4.38 kg at a population mean of ~70 kg = **6.3% relative error**, with echo-anchoring still contaminating nearly half the predictions.

## Side-by-side vs Step 0 (dugongs)

| Aspect | Step 0 (dugongs) | Step 1 (alice_charlie) |
|---|---|---|
| n seeds (usable) | 4 / 5 | 16 / 16 |
| Crash rate | 20% | 0% (after agent.py patch) |
| Relative MAE (MAE / marginal_mean) | ~49% | ~6% |
| Skill score (MAE / marginal_σ) | 0.62 | 0.59 |
| Bias sign stability | unstable (±σ > mean) | unstable (±σ > mean) |
| Echo rate | 65% (6.5/10) | 44% (4.4/10) |

**alice_charlie is an easier env for the raw baseline** than dugongs, for three identifiable reasons:
1. Weight≈cube(height) is intuitive to any LLM — no need to rediscover a Gompertz curve
2. Higher signal-to-noise: SNR ≈ 35 at h=170 (vs SNR ≈ 4 for dugongs at length ~1.1, σ=0.25)
3. Narrower target range: human weights 48–97 kg, so even a constant predictor isn't catastrophic

But skill scores are **nearly identical (0.59 vs 0.62)** — meaning both envs compress the baseline to roughly the same fraction of the marginal predictor's error. The 43.8% echo rate confirms the Step 0 structural failure mode is still present, just at smaller absolute magnitudes.

## Bug caught + fixed during this batch

Rerunning with parallel load, 5 out of first 6 concurrent seeds crashed at [boxing-gym/src/boxing_gym/agents/agent.py:71](../../boxing-gym/src/boxing_gym/agents/agent.py#L71) with:

```
TypeError: expected string or bytes-like object, got 'NoneType'
```

**Root cause**: `openai.chat.completions.create(...).choices[0].message.content` returns `None` when gpt-5 family models emit reasoning tokens but no user-facing content. More common under concurrent load (rate-adjacent requests get truncated responses).

**Fix** (in boxing-gym/agent.py):
- `prompt_llm`: coerce `None` content → `""`
- `parse_response`: explicit `None` guard returns `None` (triggers retry in `prompt_llm_and_parse`)

After fix: 12/12 re-launches completed, 0 crashes.

## Targets for MEIS to beat (updated with real numbers)

| Metric | Step 1 baseline (n=16) | MEIS goal |
|---|---|---|
| MAE (kg) | 4.38 ± 2.02 | **< 2.0** (≈ measurement noise floor σ=2) |
| Bias (kg) | +1.16 ± 2.52 | **\|bias\| < 0.5**, std < 1.0 |
| Skill score (MAE/σ₀) | 0.59 | **< 0.25** |
| Echo rate | 43.8% | **< 10%** |
| Crash rate | 0% | 0% (hold) |

MEIS mechanism (Steps 2-3): inject the curated prior `weight = density · k · height³` with `density ~ N(1010, 30)` and `k ~ N(1.4e-8, 0.7e-9)` from the cross-domain prior library into the scientist's system message. That factors the 1-parameter θ regression into two known human-physics constants, collapsing the effective degrees of freedom and eliminating the need to rediscover the cube law from 10 noisy samples.
