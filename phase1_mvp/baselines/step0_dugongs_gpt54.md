# Step 0 Baseline — dugongs_direct_discovery

**Setup**
| Key | Value |
|---|---|
| env | `dugongs_direct_discovery` (Gompertz growth, lambda=0.4, alpha=2, beta=1.5) |
| LLM | gpt-5.4 via ruoli.dev/v1 proxy |
| mode | discovery (`exp/discovery.yaml`, `num_experiments=[10]`) |
| seed | 1 |
| `include_prior` | true (scientist told it's sea cows, not abstract f(x)) |
| `use_ppl` | false (no PyMC program injection) |
| MEIS changes | none — pristine boxing-gym baseline |

**Result JSON path**: `boxing-gym/results/dugongs/direct_discovery_gpt-5.4_discovery_True_1_critic=True.json`
**Hydra log**: `boxing-gym/outputs/2026-04-23/22-42-25/run_experiment.log`

## Scientist observation policy (10 experiments)
| age | observed length |
|---|---|
| 2.5 | 1.196 |
| 1.0 | 1.310 |
| 4.0 | 1.163 |
| 0.0 | 1.066 |
| 5.0 | 1.210 |
| 3.5 | 1.088 |
| 4.5 | 1.091 |
| 1.75 | 1.054 |
| 0.5 | 0.935 |
| 1.25 | **0.423** ← noisy outlier |

Strategy was reasonable (uniform coverage of [0,5]), but the outlier at age=1.25 dragged the scientist's mental model.

## Novice prediction results (10 eval points)

| pred | gt | err |
|---|---|---|
| 0.680 | 1.256 | −0.576 |
| 0.423 | 0.723 | −0.300 |
| 0.423 | 1.257 | −0.834 |
| 0.764 | 1.156 | −0.392 |
| 0.580 | 1.143 | −0.563 |
| 0.423 | 0.837 | −0.414 |
| 0.423 | 0.995 | −0.572 |
| 0.423 | 1.716 | −1.293 |
| 0.423 | 0.904 | −0.481 |
| 0.423 | 0.693 | −0.270 |

## Metrics (reference for MEIS comparison)

| Metric | Value |
|---|---|
| Novice MSE | **0.4060** |
| Novice RMSE | 0.6372 |
| Novice MAE | 0.5693 |
| Standardized err (MAE/σ₀, σ₀=9.234) | 0.0617 |
| Mean signed err (bias) | **−0.569** (severe under-prediction) |
| Scientist self-error mean | 0.406 |
| Scientist self-error std | 0.454 |

## Observations

1. **Novice collapsed to 0.423 on 7/10 points.** Looking at the prompt sent to the novice, each eval query starts with `"The final result is 0.42311031675710453.\nPredict the length at age ..."` — the previous prediction leaks into the next prompt as context. When the scientist's explanation is purely qualitative (no numbers), the novice falls back on echoing that prompt-prefix value.

2. **Scientist's explanation was wrong** about the env dynamics — it claimed length "dips through the middle/older range" then "rises again near the oldest ages". Real Gompertz curve is monotonically increasing. The scientist over-fit to the single outlier at age=1.25.

3. This matches the BoxingGym paper's core finding (Sec.4.1–4.2): LLMs cannot reliably extract the right functional form from noisy data, and adding prior context does not help.

## What MEIS needs to beat

- **MSE < 0.2** would be a clear improvement
- **Bias within ±0.2** (get rid of systematic under-prediction)
- **No prompt-leakage artifact** (ideally the novice doesn't anchor on 0.423)

The mechanism MEIS plans to use: inject a curated prior ("dugong length follows Gompertz with asymptote ~1.4, starting baseline ~1.0") via the cross-domain prior library. That lets the scientist build a structured posterior instead of free-form speculation.
