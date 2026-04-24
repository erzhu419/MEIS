# Step 3.10 — Cross-env expansion to Lotka-Volterra

Adds a third env (LV predator-prey ODE) to test whether the scientist-side
MEIS effect (p<0.001 on dugongs in Step 3.8) generalises or was domain-specific.

## Setup

| Key | Value |
|---|---|
| env | `lotka_volterra` (coupled predator-prey ODE, output = tuple) |
| LLM | gpt-5.4 via ruoli.dev/v1 |
| configs | baseline_noecho vs meis_full_noecho (NL channel) |
| seeds | 1-16 each |
| sanity-retry | off (output is tuple, current sanity guard is 1-D) |
| structured-channel | off (would require asking scientist for ODE params — complex) |
| prior library expansion | added `dynamics_multivariate.json` with 3 entries: lotka_volterra_predator_prey, periodic_oscillation_shape, ode_two_variable_coupling |
| retrieval query | "predict predator prey population oscillation time ODE" |

## End-task MAE (n=16 each)

| config | MAE | median AE | MAE prey | MAE predator |
|---|---|---|---|---|
| baseline_noecho | 13.03 ± 8.52 | 11.38 | 22.04 | 4.01 |
| meis_full_noecho | 14.88 ± 4.75 | 11.47 | 24.94 | 4.81 |

Paired (n=16, 1-sided): MAE p=0.37 (Wilcoxon), 11/16 MEIS-wins. Median AE
p=0.35, 8/16. **No significant difference either direction.**

Per-seed is bimodal: seeds where eval points sit near initial conditions
(t ≈ 0) baseline trivially wins by guessing initial values; where eval
points drift, both configs fail comparably. The LV env is too noisy
in this config to be a clean MEIS test.

## Scientist-side regex analysis (the reason for Step 3.10)

| env | tier | baseline | MEIS-sci | MW-U p (1-sided) | interpretation |
|---|---|---|---|---|---|
| alice_charlie | cube_law | 100% | 100% | 0.74 | LLM knows h³ |
| alice_charlie | density_concept | 88% | 94% | 0.039 | marginal |
| **dugongs** | saturating_shape | 29% | **83%** | **<0.001** | LLM lacks form, MEIS teaches |
| **dugongs** | exponential_or_decay | 2% | **50%** | **<0.001** | same |
| lotka_volterra | predator_prey_concept | **100%** | **100%** | — | env prompt GIVES the concept |
| lotka_volterra | oscillation_shape | 100% | 100% | — | same |
| lotka_volterra | ode_form | 0% | 6% | 0.17 | null |

## Critical finding — the env prompt seeds the LLM

`boxing-gym/src/boxing_gym/envs/lotka_volterra.py:247`:
```
PRIOR = f"You are observing the populations of predators and prey at different times. ..."
```

The env's own `include_prior=True` system message explicitly tells the LLM
this is a predator-prey system. gpt-5.4 doesn't need MEIS priors to know
to use predator-prey vocabulary — it's handed to it for free. Any regex
match rate on predator/prey/oscillation therefore hits 100% at baseline,
leaving no headroom for MEIS to add a signal.

This means **LV is not a fair test of MEIS's "supply missing domain
knowledge" hypothesis**. It would only be fair with `include_prior=False`,
but then we also lose the baseline LLM's chance to apply its training.

## Synthesis across all three envs

| env | LLM has the critical form in its training? | env prompt seeds the form? | MEIS scientist-side effect |
|---|---|---|---|
| alice_charlie | ✅ (h³ is textbook) | no (just says "adult humans") | null |
| **dugongs** | ❌ (saturating growth is niche) | no (just says "sea cow") | **p < 0.001** |
| lotka_volterra | ✅ (everyone knows LV) | ✅ (env says "predator-prey") | saturated at 100% |

**MEIS's independent marginal value surfaces only in the `(no training, no env-prompt)` quadrant.** Dugongs happens to sit there. Alice_charlie and LV do not.

## Phase 1 MVP final conclusion

After 10 sub-steps (3.5 → 3.10), the clean MEIS evaluation picture:

1. **Scientist-side effect** is **real and statistically robust** (dugongs:
   +48 percentage-points on saturating-shape mentions, +47pp on exponential
   mentions, both p < 0.001, n=16 non-parametric). Requires env where LLM
   doesn't already have the form.

2. **End-task MAE effect** is **drowned by downstream noise**:
   (a) NL transmission + LLM stochasticity (Step 3.7: sanity fixed outliers
        but also erased MEIS MAE advantage)
   (b) When structured output is used as a bypass (Step 3.9), MAE drops
        60% for BOTH configs — baseline catches up via elicitation format
        because gpt-5.4 has the knowledge once asked right.

3. **MEIS's role is best framed as "reasoning-transparency / scientist-explanation
   quality" rather than "end-task accuracy gain"** under current LLM
   capabilities. For genuine accuracy gains, we'd need either (a) envs
   truly outside LLM training distribution, or (b) architectures that
   don't go through a fresh novice LLM.

4. **KL drift** (Step 6) shows MEIS has a second, orthogonal role: ranking
   multiple candidate hypotheses by coherence with the belief network.
   This is the blueprint's theoretical core (L3) and doesn't depend on the
   end-task MAE regime.

## Artifacts (this step)

- `phase2_prior_library/dynamics_multivariate.json` (3 new priors)
- `phase1_mvp/run_mvp_unified.py` (registered LV env + prior query)
- `phase1_mvp/analysis/eval_scientist.py` (LV regex patterns added)
- `phase1_mvp/runs/lotka_volterra/baseline_noecho/seed_{1..16}.json`
- `phase1_mvp/runs/lotka_volterra/meis_full_noecho/seed_{1..16}.json`
- 32 LV runs, 0 crashes, 0 parse failures

## Direction audit

1. **L-layer attribution** ✓ — added L3 (library entries) + L4 (regex rules), no L1/L2 drift.
2. **Trunk-free** ✓ — LV env is imported from boxing-gym, not modified.
3. **Scope** ✓ — exactly what Step 3.9 recommended as C.
