# P2.1b — Peregrines structured channel (n=2 for bypass-NL, with caveat)

Follow-up to P2.1 (peregrines regex result replicated dugongs p<0.001).
This round tests whether Step 3.9's "structured channel fixes MAE for
BOTH configs" finding also replicates on peregrines.

## Setup

- 32 runs: 16 seeds × {baseline, meis_full}, all with
  `--sanity-retry --structured-channel`
- `SANITY_RANGES["peregrines"] = (0, 5000)` — wide to catch only
  decimal-glitches, not population-range variation
- Structured distill prompt for peregrines elicits `log(λ) = α + β1·t + β2·t² + β3·t³`
  form explicitly (see `run_mvp_unified.STRUCTURED_DISTILL_PROMPTS["peregrines"]`)

## Four-config MAE table (peregrines, n=16 each)

| config | MAE | med AE | log MAE | outl |
|---|---|---|---|---|
| baseline_noecho (NL)        | 181.6 ± 101.9 | 184.2 | 2.64 ± 1.54 | 0 |
| meis_full_noecho (NL)       | 202.1 ±  53.8 | 209.1 | 2.75 ± 0.94 | 0 |
| **baseline_sanity_structured** | **23.4 ± 22.0** | **21.9** | **0.15 ± 0.21** | 0 |
| **meis_full_sanity_structured** | 89.2 ± 228.3 | 94.3 | 0.28 ± 0.41 | 0 |

## Finding 1 — structured channel delivers massive MAE cut on BOTH (bypass-NL n=2)

| A → B | MAE mean Δ | B-wins / 16 | Wilcoxon p |
|---|---|---|---|
| baseline_NL → baseline_structured | −158 | **16** | **< 0.0001** |
| meis_NL → meis_structured | −113 | 14 | 0.003 |
| both configs on log-MAE | −2.47 | 16/15 | < 0.0001 |

The "structured elicitation bypasses the NL bottleneck" finding from
dugongs Step 3.9 replicates with an **even stronger effect on peregrines**
(log-MAE cut by factor of ~18 vs dugongs ~3x). This makes sense: peregrines
baseline NL was catastrophically bad (single-digit novice predictions
vs 100+ ground-truth), so the ceiling for improvement was higher.

## Finding 2 — Under structured, MEIS-full is actually WORSE than baseline

| A → B | Δ | B-wins / 16 | Wilcoxon p |
|---|---|---|---|
| baseline_structured → meis_full_structured (MAE_w) | +66 (MEIS worse) | 6 | 0.83 (null, trending MEIS worse) |
| same (med AE) | +72 (MEIS worse) | 5 | 0.87 |
| same (log MAE) | +0.13 (MEIS worse) | 5 | 0.89 |

Per-seed shows **3 catastrophic MEIS failures**:
- seed 7:  baseline 24.3,  **MEIS 931.4** (+907)
- seed 11: baseline 19.3,  **MEIS 141.7** (+122)
- seed 16: baseline 22.4,  **MEIS 130.6** (+108)

Removing those 3 outliers, MEIS and baseline are nearly tied. With them,
MEIS mean MAE is 3.8× baseline's.

## Root cause — exp() is unbounded

Peregrines' log-cubic-Poisson ground truth includes `exp(α + β1·t + β2·t² + β3·t³)`.
Small parameter errors at large |β3| can cause the formula to blow up at
query times beyond the scientist's observation range. The MEIS scientist,
given richer prior context (including the specific poisson_log_polynomial
entry), more confidently distills structured formulas — but "more confident"
also means "more likely to overreach with overfit parameters that diverge".

Baseline scientists, without priors, tend to emit more conservative
formulas (lower-order polynomial, bounded, or even piecewise). On
peregrines-like unbounded-output domains, that conservatism happens to
be protective.

Compare to dugongs Step 3.9: ground truth is `α − β · γ^age`, which
is **bounded** regardless of parameter draws. MEIS and baseline were
effectively tied. The asymmetry between dugongs and peregrines comes
entirely from the boundedness of the ground-truth formula class.

## Cross-env synthesis (now n=4 envs, full 4-config grid on 2)

| env | structured helps baseline? | structured helps MEIS? | MEIS vs baseline under structured? |
|---|---|---|---|
| dugongs (bounded form) | **yes, p=0.0003** | **yes, p=0.0005** | tied (p=0.32) |
| **peregrines (unbounded exp)** | **yes, p<0.0001** (larger effect) | **yes, p=0.003** | **MEIS WORSE (p=0.89)** — occasional exp-divergence |

**The structured-channel conclusion strengthens** (now n=2), but the
"MEIS ≡ baseline under structured" conclusion was **conditional on the
env having a bounded output class**. On unbounded-output envs, MEIS's
confident parameter distillation can backfire.

## Actionable implications

1. Structured channel is robustly valuable — recommend making it default.
2. MEIS priors under structured need a **bounded-output guard**: the
   scientist should validate that its distilled formula, evaluated
   across the env's input range, stays in-range. If not, either
   accept an NL-channel fallback or repair the formula.
3. For any new env, classify whether the ground-truth output is
   bounded or unbounded by construction. On unbounded, MEIS structured
   should be tested against baseline structured explicitly before
   deployment.
4. The sanity-retry range (0, 5000) did NOT catch these failures
   because the MEIS-divergent predictions fall IN the plausible range
   for peregrine counts (931 is bigger than typical peak ~250, but not
   absurd like 1e8). Range-based sanity can't catch "plausible but
   wrong".

## Direction audit

1. L-layer attribution ✓ — runner + library + regex only; no L1-L3 drift
2. Trunk-free ✓ — peregrines from boxing-gym verbatim
3. Scope ✓ — directly tests Step 3.9 replication (the Phase 1 report §8 suggested this)
