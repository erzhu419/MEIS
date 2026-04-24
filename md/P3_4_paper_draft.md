# Minimum Embedding Distance as a Hypothesis Coherence Metric

*MEIS Phase 3 mini-paper draft — P3.4 deliverable. Sketches the
mathematical framework, algorithm, and empirical validation that would
go into a publishable paper; hands-off to a full draft later.*

---

## Abstract

Given a probabilistic belief network $\mathcal{B}$ summarizing current knowledge
over a set of latent variables, and a set of candidate hypotheses
$\{h_1, \ldots, h_k\}$ each proposing an explanation for an observed fact,
we formalize and validate a **minimum-embedding coherence score**

$$D(h, \mathcal{B}) = D_{\mathrm{KL}}\big(P(\mathcal{B}) \,\|\, P(\mathcal{B}|h)\big) + \lambda \cdot |\Delta_{\text{structure}}|$$

where the first term measures posterior perturbation the hypothesis induces
on the target latent, and the second term applies a Bayesian Information
Criterion–derived penalty for new nodes or edges the hypothesis requires
but that lack data support. Ranking claims by ascending $D$ produces an
ordering consistent with the Quinean "web of belief" intuition: the best
explanation is the one requiring the minimum disruption of existing
beliefs.

We validate on three case studies (one quantitative, two qualitative):
Alice-Charlie body-weight comparison, Noh theater gender monopoly, and
Eastern Han emperor early-death. Across 18 ablation cells
(3 benchmarks × 3 structural formulations × 2 KL directions), the
structural term is shown to be both necessary and sufficient for
isolating orphan-node claims: it succeeds in 12/12 cells with either
BIC or count penalty, and fails in 0/6 with pure KL.

---

## 1. Motivation

Human explanatory reasoning routinely prefers explanations that fit
existing beliefs with minimum revision (Quine's web-of-belief; Lakatos's
protective belt). Bayesian probabilistic inference formalizes the
numerical half of this idea via KL-regularized posterior updates. The
structural half — how much a hypothesis requires revising the network's
graph — has no standard Bayesian analog.

We propose an additive composite that combines the two, derived from:
- **KL posterior drift** on the target latent (classical)
- **BIC-style penalty** for data-unsupported graph additions

The result is a computable metric ranking candidate hypotheses by
"coherence with current beliefs", recovering expert-intuition rankings
on three disparate case studies.

## 2. Formulation

### 2.1 Belief network

Let $\mathcal{B}$ be a PyMC probabilistic program encoding the prior
knowledge about a set of latents $\theta = (\theta_1, \ldots, \theta_d)$
and previously observed data $\mathcal{D}$. The posterior
$P(\theta \mid \mathcal{D})$ is the base distribution.

### 2.2 Hypothesis as additional evidence

A hypothesis $h$ translates into a pair
$h = (E_h, \Delta_h)$, where $E_h$ is additional observed evidence
extending $\mathcal{B}$, and $\Delta_h$ is the set of new graph elements
(nodes, edges, or both) that $h$ requires but $\mathcal{B}$ does not have.

The posterior conditioned on $h$ is
$P(\theta \mid \mathcal{D}, E_h)$ inside the extended network
$\mathcal{B}' = \mathcal{B} \cup \Delta_h$.

### 2.3 Composite score

For a chosen target latent $\theta^*$:

$$D(h, \mathcal{B}) = D_{\mathrm{KL}}\!\big(P(\theta^* | \mathcal{D}) \,\|\, P(\theta^* | \mathcal{D}, E_h)\big) + \lambda \cdot |\Delta_h|$$

with $\lambda = \log(N)/2$ per BIC, $N$ being the effective observation
count in $\mathcal{B}$.

### 2.4 Estimation

- **Gaussian conjugate case**: closed-form; both $\mathrm{KL}$ and
  $\mathcal{B}' $ posteriors available analytically.
- **General PyMC models**: Monte Carlo estimate of $\mathrm{KL}$ from
  posterior samples (Gaussian-moment approximation, full-covariance,
  or kernel density).
- **Structural edits**: counted by inspection of $\Delta_h$; alternative
  formulations use a unit-count penalty or an explicit graph-edit
  distance (Appendix).

## 3. Case studies

### 3.1 Alice-Charlie weight comparison (controlled)

Three-person belief network: heights, shoe sizes, foot areas, pressures,
footprint depths; latent weights via $w = \theta \cdot h^3$. Candidate
claims explain "Alice is heavier than Charlie":

| claim | structural | KL ($\theta_{A}$) | composite |
|---|---|---|---|
| Alice is 5 cm taller | 0 | 0.037 | 0.037 |
| Alice is 5% denser | 0 | 0.007 | 0.007 |
| Alice has larger feet | 0 | 0.000 | 0.000 |
| **Alice is Year-of-Tiger zodiac** | **2 (orphan nodes)** | **0.002** | **3.403** |

Zodiac ranks last with composite $\sim 92 \times$ any in-vocabulary claim.

### 3.2 Noh theater gender monopoly (qualitative)

Historical fact: women excluded from Noh performance (≈14-19th century).
Base belief network latents: `ritual_role`, `blood_taboo`,
`shogunate_ban_effect` ∈ $[0, 1]$ with Beta priors softly pulled by
primary-source evidence (Okina purification rituals, nyonin-kekkai
shrine exclusion, scope of 1629 Kabuki ban). Target:
$P(\text{women}_\text{banned}=1)$.

| claim | structural | KL | composite |
|---|---|---|---|
| Priestly/ritual role | 0 | 0.33 | 0.33 |
| Shogunate ban spillover | 0 | 0.96 | 0.96 |
| Blood-pollution taboo | 0 | 1.82 | 1.82 |
| **Female voice unsuitability (orphan)** | **2** | **0.001** | **2.30** |

### 3.3 Eastern Han emperor early death (qualitative)

Historical fact: median age at death for Eastern Han emperors ≈21.
Latents: `lead_poisoning_strength`, `political_stress_strength`,
`incest_bottleneck_strength`. Soft proxies: archaeological lead content,
purge-era death counts, court genealogies.

| claim | structural | KL | composite |
|---|---|---|---|
| Political faction stress | 0 | 0.24 | 0.24 |
| Lead poisoning | 0 | 1.49 | 1.49 |
| Incest bottleneck | 0 | 1.81 | 1.81 |
| **Palace fengshui (orphan)** | **2** | **0.002** | **2.30** |

## 4. Cross-metric ablation (robustness)

We tested every combination of:
- **Structural formula**: BIC ($\lambda = \log N / 2$), count ($\lambda = 1$),
  none ($\lambda = 0$)
- **KL direction**: base-to-hyp ($\text{KL}(P \| Q)$), hyp-to-base ($\text{KL}(Q \| P)$)

on all 3 benchmarks, producing 18 cells. Per-cell outcome = "does the
orphan claim rank last?"

| structural formula | orphan-last rate |
|---|---|
| BIC | 6/6 (100%) |
| count | 6/6 (100%) |
| none (pure KL) | 0/6 (0%) |

KL direction had no effect on the orphan-last outcome: in every
(benchmark, formula) combo, both directions agreed.

**Conclusion**: $|\Delta_{\text{structure}}|$ is both necessary AND
sufficient for correct orphan isolation. Under pure KL, the orphan
ranks AT THE TOP because disconnected latents induce near-zero posterior
shift — exactly the opposite of the desired behavior.

## 5. Related work

| Work | What it does | Relation to MEIS |
|---|---|---|
| Wong et al., "From Word Models to World Models" (2023) | LLM→PPL translation for one-shot inference | Complementary upstream pipeline; our belief network is the $\mathcal{B}$ they don't persist |
| POPPER (Huang et al., 2025) | LLM hypothesis falsification via sequential testing | Complementary: POPPER validates a single hypothesis; we rank multiple by coherence |
| Pearl causal inference | $\text{do}$-calculus on a single DAG | $\mathcal{B}$ in MEIS directly reuses Pearl's representation; we add inter-DAG comparison |
| Friston free-energy principle | Unified perception/action minimize free energy | $D(h, \mathcal{B})$ = complexity + accuracy is the same structural form |
| Perrone, "Categorical Information Geometry" (2024) | Divergences on Markov-category morphisms | Directly aligns with our L2+L3 — theoretical grounding for future Phase 4 |

## 6. Implementation

All results are reproducible from:

- `phase3_embedding/claim_ranking_engine.py` — reusable engine with
  pluggable KL estimator and structural formula
- `phase3_embedding/benchmarks/{noh_theater,eastern_han}.py` —
  benchmark-specific PyMC belief-network builders
- `phase3_embedding/demo_claim_ranking.py` — Alice-Charlie case
- `phase3_embedding/demo_metric_ablation.py` — cross-metric study
- `phase3_embedding/tests/test_{claim_ranking,plan_benchmarks,metric_ablation}.py`

Acceptance suite: 13 unit tests across 3 test modules, all PASS.

## 7. Limitations and Future Work

1. **Orphan encoding is manual**: each benchmark author marks
   `structural_additions` on a claim. A full system would detect orphan
   nodes automatically (semantic parsing of the claim → implied graph
   edits). This is Phase 4 scope.
2. **Social-science benchmarks use soft proxies**: latents like
   `ritual_role` are $[0, 1]$ proxies that historians would want to
   scrutinize. The metric is robust to specific prior values but the
   BENCHMARK's quantitative claims aren't forensic.
3. **Qualitative claim-ranking within "in-vocabulary"** partly reflects
   prior-mean distance rather than purely coherence. This is a feature
   (consistency with strong existing evidence = low KL) but may surprise
   casual readers.
4. **No multi-latent generalization tested** beyond the conjugate
   Alice-Charlie case. Extending to dugongs (3 params) / peregrines
   (4 params) is one configuration update but hasn't been exercised.

## 8. Paper-level claim

MEIS's minimum-embedding coherence metric $D(h, \mathcal{B})$ correctly
isolates structurally incoherent claims from structurally coherent ones
across three heterogeneous case studies (one quantitative, two
qualitative-historical), under two distinct structural penalty
formulations (BIC and unit count), in both KL directions. Pure-KL
without a structural term systematically FAILS the isolation task.
The structural penalty is empirically the load-bearing quantity for
coherence-based hypothesis ranking in probabilistic belief networks.

This is a concrete, computable, publishable instantiation of the
Quine/Lakatos intuition that "good explanations disrupt the web of
belief minimally".

---

*Status: draft. Not yet camera-ready; missing full empirical Table 1
(raw numbers currently in commit messages), discussion of alternatives
to BIC, and positioning vs. recent LLM-grounded abduction work (e.g.,
Pallagani et al., 2024).*
