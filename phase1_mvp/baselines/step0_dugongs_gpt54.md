# Step 0 Baseline — dugongs_direct_discovery (gpt-5.4, 5 seeds)

**Setup**

| Key | Value |
|---|---|
| env | `dugongs_direct_discovery` (Gompertz, α=2, β=1.5, λ=0.4) |
| LLM | gpt-5.4 via ruoli.dev/v1 proxy |
| mode | discovery (`num_experiments=[10]`, `num_evals=10`) |
| `include_prior` | true |
| `use_ppl` | false |
| seeds | 1, 2, 3, 4, 5 (→ 4 usable; seed=3 crashed) |
| MEIS changes | none — pristine boxing-gym |

**Result JSONs**: `boxing-gym/results/dugongs/direct_discovery_gpt-5.4_discovery_True_{1,2,4,5}_critic=True.json`
**Raw copies**: `phase1_mvp/baselines/step0_dugongs_gpt54_seed{1,2,4,5}_raw.json`

## Per-seed metrics

| seed | MSE | RMSE | MAE | Bias | Scientist self-err | obs10 (outlier) | echo-artifact (preds == obs10) |
|---|---|---|---|---|---|---|---|
| 1 | 0.4060 | 0.6372 | 0.5693 | **−0.569** | 0.406 | 0.423 | **7 / 10** |
| 2 | 0.6013 | 0.7754 | 0.7103 | **+0.710** | 0.601 | 1.775 | 3 / 10 |
| 4 | 0.3838 | 0.6195 | 0.4845 | **+0.485** | 0.384 | 1.671 | **8 / 10** |
| 5 | 0.2004 | 0.4477 | 0.3887 | +0.257 | 0.200 | 2.062 | **8 / 10** |

## Pooled summary (n=4)

| Metric | mean | std (ddof=1) | min | max |
|---|---|---|---|---|
| MSE | **0.3979** | 0.1639 | 0.2004 | 0.6013 |
| RMSE | 0.6199 | 0.1343 | 0.4477 | 0.7754 |
| MAE | 0.5382 | 0.1364 | 0.3887 | 0.7103 |
| Bias | **+0.2206** | **0.5582** | −0.5693 | +0.7103 |
| Scientist self-err mean | 0.3979 | 0.1639 | 0.2004 | 0.6013 |

**σ/mean (CV)** for MSE ≈ **41%** — baseline is unstable across seeds. Variance dominated by (a) which observation sample lands closest to the 10th query, and (b) which explanation strategy the scientist happens to settle on.

## Three reproducible artifacts of this baseline

1. **Echo-anchoring**: in 3 of 4 runs, the novice's prediction **exactly equals the 10th observation value** (`obs10`) for 7–8 of 10 eval points. Cause: each novice prompt is prefixed with `"The final result is <obs10>."` — so when the scientist's natural-language explanation carries no numerical values, the novice falls back on echoing `obs10`. This is a **boxing-gym prompt-construction issue**, not a gpt-5.4 weakness.

2. **Bias flips sign across seeds**: seed=1 under-predicts by 0.57; seeds 2/4/5 over-predict by +0.26 to +0.71. The bias is whatever `obs10` happened to land on — essentially a noisy anchor. Bias σ (0.56) exceeds its mean (0.22).

3. **Crash rate 20%**: seed=3 raised `TypeError: expected string or bytes-like object, got 'NoneType'` at [agent.py:71 `re.search(pattern, response, re.DOTALL)`](../../boxing-gym/src/boxing_gym/agents/agent.py#L71) — gpt-5.4 returned an empty response and `prompt_llm` returned `None`, which the regex parser can't handle. BoxingGym never null-checks. Noted for later robustness patch (non-blocking for MEIS Phase 1).

## Example scientist "explanation" (seed=1)

> "Length starts at some baseline near age 0. It changes noticeably over early ages. It then **dips through the middle/older range**. After that, it rises somewhat again near the oldest ages."

Real Gompertz is monotonically increasing. Scientist over-fit to the outlier at age=1.25 (length=0.423) and invented a non-existent mid-range dip. Exactly the failure mode BoxingGym paper Sec.4.1–4.2 documents.

## Targets for MEIS to beat

| Metric | Step 0 baseline (mean ± std) | MEIS goal |
|---|---|---|
| MSE | 0.40 ± 0.16 | **< 0.20** (one-sigma below baseline) |
| Bias | +0.22 ± 0.56 | **\|bias\| < 0.10**, std < 0.20 (kill the anchor artifact) |
| Crash rate | 20% | 0% |
| Novice-echo count | 6.5 / 10 avg | < 2 / 10 (genuine reasoning, not prompt echo) |

MEIS mechanism to flip these numbers: inject a **curated Gompertz-like prior** (`length = α − β · λ^age` with documented α≈1.4, β≈1.5, λ≈0.4 ± noise) from the cross-domain prior library (Step 2) into the scientist's system prompt (Step 3). Forces the scientist to build a structured posterior over (α,β,λ) rather than free-form mental modeling, and the novice gets numerical summaries rather than qualitative hand-waving.
