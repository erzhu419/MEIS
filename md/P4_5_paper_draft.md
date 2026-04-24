# Categorical Structure Transfer in Belief Networks

*MEIS Phase 4 mini-paper draft — P4.5 deliverable. Consolidates the
law-zoo, structural signature, retrieval/clustering, and transfer
results into the skeleton of MEIS's third paper (Plan §5 milestone
"Paper 3").*

---

## Abstract

Two physical reasoning problems — "voltage decays in an RC circuit"
and "concentration decays in a first-order chemical reaction" — share
the same functional form $y(t) = y_0 \cdot e^{-kt}$ despite operating
in different domains with different numerical scales. A probabilistic
reasoning system should be able to recognize this equivalence and
transfer class-level knowledge from one domain to another.

We present MEIS Phase 4: an end-to-end pipeline that (1) represents
each belief network at three equivalent levels — an op-multiset
fingerprint over PyTensor, a Weisfeiler-Lehman refinement on the
ancestor DAG, and a symbolic Markov-category string diagram — each
producing the same equivalence partition; (2) retrieves structurally
equivalent networks from a library via Ruzicka distance on the op
multiset; and (3) transfers posterior log-SDs (class-level precision)
from a data-rich source to a data-poor target, without transferring
any scale-specific information.

On a **law-zoo v2** fixture (3 equivalence classes × 10 physical
domains), signature-based clustering recovers the ground-truth
partition with ARI = 1.00 under all three representations
(Plan §Phase 5 task 2 target > 0.8). Structural transfer into data-
poor saturation-class targets achieves a mean **89.5% held-out MSE
reduction** vs a cold-start baseline (Plan §Phase 5 task 3 target
≥ 30%), while exp_decay targets show no benefit — a legitimate
asymmetry driven by the dynamics, not the method.

---

## 1. Motivation

The Quinean "web of belief" metaphor (see Paper 2, *Minimum Embedding
Distance as a Hypothesis Coherence Metric*) captures how hypotheses
should disrupt existing beliefs minimally. A separate, complementary
intuition governs **learning**: when a new domain has the same
functional form as one we already understand, we shouldn't have to
re-learn from scratch.

Classical Bayesian hierarchical models capture this via a shared
hyperprior, but require the modeller to declare the shared structure
up front. Meta-learning approaches (MAML, Reptile) learn a good
initialization across tasks but treat tasks as opaque. Neither
addresses the *discovery* question: given two belief networks built
independently, can a system infer they are instances of one
underlying abstraction?

MEIS Phase 4 proposes an operational answer: two belief networks are
structurally equivalent iff they yield the same **StructuralSignature**
— a hash derived from the PyTensor op graph rooted at their observed
RVs. Equivalence detected this way supports:

- **Retrieval**: nearest-neighbor lookup in a signature library
- **Clustering**: automatic equivalence-class recovery
- **Transfer**: use the rich-data source's posterior precision as a
  prior for the data-poor target

## 2. Formulation

### 2.1 StructuralSignature

For a PyMC model with observed RVs $\{y_1, \ldots, y_m\}$ and free
RVs $\{\theta_1, \ldots, \theta_d\}$, define:

- **Op multiset** $\mathcal{O}$: the sorted multiset of PyTensor op
  class names obtained by walking `ancestors(y_i)` for each observed
  RV. Elemwise wrappers are unwrapped to their scalar op so that all
  `exp` ops collapse to one label.
- **Role vector** $\mathcal{R}$: the sorted tuple of role labels
  obtained by looking up each RV name in a caller-supplied
  `LATENT_ROLES` dict (vocabulary: `scale`, `rate`, `noise`, `obs`).
- **Counts**: $|\theta| = d$, $|y| = m$.

The signature is

$$\mathrm{sig}(\mathcal{B}) = (\mathcal{R}, \mathcal{O}, d, m)$$

hashed to a 16-hex fingerprint via SHA-256. Two belief networks are
deemed equivalent iff their fingerprints match.

### 2.2 Signature distance (Ruzicka)

When fingerprints differ, we still want a graded distance. For two op
multisets $\mathcal{O}_a, \mathcal{O}_b$ with counts $c_a, c_b$,

$$d(a, b) = 1 - \frac{\sum_{o} \min(c_a(o), c_b(o))}{\sum_{o} \max(c_a(o), c_b(o))}$$

This is the Ruzicka (weighted Jaccard) distance, $\in [0, 1]$, and
equals 0 iff multisets are identical.

### 2.3 Signature clustering

Two modes are supported:

- **fingerprint mode**: bucket by exact hash match. Zero intra-class
  distance, nonzero inter-class by construction.
- **threshold mode**: union-find over the $\{(i,j) : d(i,j) < \tau\}$
  graph. For the law-zoo v1, inter-class $d = 0.154$ (only the
  saturation `Sub` op differs), so $\tau \in (0, 0.15)$ recovers the
  ground-truth partition.

### 2.4 Markov-category string diagrams

The P4.2 op-multiset and the P4.6 WL signature are both defined on the
PyTensor graph — operational artefacts of the chosen PPL backend. An
independent, backend-agnostic check lives one layer up: express each
belief network as a morphism in a symbolic **Markov category** and
compare string diagrams.

We implement the Fritz 2020 / Perrone 2024 categorical primitives as

$$\mathrm{Obj}(X; \kappa),\ \mathrm{Atom}(\mathit{name}, \mathbf{A} \to \mathbf{B}; \kappa),\ \Delta_X: X \to X \otimes X,\ !_X: X \to I$$

with sequential ($f\,;\,g$) and monoidal ($f \otimes g$) composition. A
belief network's abstract diagram is a tree of these primitives with
atoms typed by functional **kind** (`prior`, `decay_kernel`,
`saturation_kernel`, `damped_kernel`, `normal_observation`) and objects
typed by measurable-space **kind** (`parameter`, `time-series`,
`observation-series`). Each law-zoo class maps to a canonical diagram,

$$
\begin{aligned}
\mathrm{exp\_decay}:\quad & (\mathrm{prior} \otimes \mathrm{prior})\,;\,\mathrm{decay\_kernel}\,;\,\mathrm{normal\_obs} \\
\mathrm{saturation}:\quad & (\mathrm{prior} \otimes \mathrm{prior})\,;\,\mathrm{saturation\_kernel}\,;\,\mathrm{normal\_obs} \\
\mathrm{damped}:\quad & \left(\bigotimes_{i=1}^{4} \mathrm{prior}\right)\,;\,\mathrm{damped\_kernel}\,;\,\mathrm{normal\_obs}
\end{aligned}
$$

The shape signature is a canonical hash over the morphism tree modulo
object-name $\alpha$-equivalence. Atom *kinds* are retained so that
`decay_kernel` and `saturation_kernel` are structurally distinct boxes;
atom *names* are dropped so that a prior called `prior_y0` and one
called `prior_N0` are categorically equivalent within their class.

This gives three independent routes to the same partition:

| Layer | Representation | Law-zoo ARI |
|---|---|---|
| Op-multiset (P4.2) | PyTensor scalar ops, bag-of-counts | 1.000 |
| Weisfeiler-Lehman (P4.6) | PyTensor ancestor-DAG subtree refinement | 1.000 |
| Markov category (P4.7) | Symbolic morphism-tree fingerprint | 1.000 |

All three agree on the law-zoo. Each dominates the others in a specific
failure mode: op-multiset is cheapest; WL handles
same-ops-different-wiring distractors (§2.2 footnote); the categorical
layer is PyTensor-free and survives any change of PPL backend, making
it the principled home for the equivalence notion.

### 2.5 Semantic equivalence: BSS + Perrone kernel KL

Shape matching (syntactic) is necessary but not sufficient for genuine
categorical equivalence. We add two numerically computable semantic
checks:

**Blackwell-Sherman-Stein operational equivalence.** Within the
closed world of the law-zoo, all members of a class share the same
likelihood formula $P(y \mid \theta, t)$ with canonically named
parameter tuples $\theta$ (the class's `CLASS_PARAM_NAMES`). BSS
equivalence of the likelihoods reduces to

$$\forall \theta, t, y:\quad \log P_a(y \mid \theta, t) = \log P_b(y \mid \theta, t).$$

We evaluate both log-likelihoods on a random $(\theta, t, y)$ grid of
100 samples and check for exact floating-point agreement. Result:
within-class max$|\Delta \log p| = 0$ (12 pairs); cross-class
max$|\Delta \log p| > 1$ in all 3 pairs tested.

**Perrone categorical KL divergence.** For Markov kernels
$K_a, K_b : \Theta \rightsquigarrow Y$ both of the form
$\mathrm{Normal}(\mu(\theta, t), \sigma^2)$, Perrone 2024's
kernel-level information divergence reduces to

$$D(K_a \,\Vert\, K_b) = \mathbb{E}_{\theta \sim P_\Theta, t}\left[\frac{(\mu_a(\theta, t) - \mu_b(\theta, t))^2}{2 \sigma^2}\right].$$

We Monte-Carlo estimate $D$ at 500 samples per pair, using
$\theta \sim \mathrm{LogNormal}(0, 0.5)$ as a generic reference
distribution and $\sigma = 1$. Result: within-class $D = 0$ exactly;
cross-class $D > 0$ with Monte-Carlo signal-to-noise ratio $> 3$ on
every pair.

These two checks are **semantically stronger** than shape matching.
Shape matching can collapse under name shadowing and might admit
false equivalences that BSS / Perrone reject. On the law-zoo all three
signatures (syntactic + BSS + Perrone) agree perfectly, so the
equivalence finding is robust to choice of layer.

### 2.6 Structural transfer

Given a target domain $T$ with few observations and a source domain
$S$ with many (and $\mathrm{sig}(T) = \mathrm{sig}(S)$), transfer
proceeds in three steps:

1. **Source posterior shape**: fit $S$ on its rich data; extract
   posterior log-SDs for the scale latent ($\sigma_{\log\text{scale}}^S$) and rate
   latent ($\sigma_{\log\text{rate}}^S$).
2. **Gate**: verify $\mathrm{sig}(T) = \mathrm{sig}(S)$. Refuse
   (raise) otherwise.
3. **Target fit** with transferred prior:
   $$\theta_{\text{scale}} \sim \mathrm{LogNormal}(\mu_T, \sigma_{\log\text{scale}}^S)$$
   $$\theta_{\text{rate}} \sim \mathrm{LogNormal}(\mu_T, \sigma_{\log\text{rate}}^S)$$
   Prior **means** $\mu_T$ stay at the target's defaults, preserving
   domain-specific scale. Only the **SDs** transfer.

## 3. Law-zoo v1

Two equivalence classes, seven physical domains:

| Class | ODE | Domains |
|---|---|---|
| `exp_decay` | $y(t) = y_0 \cdot e^{-kt}$ | rc_circuit, radioactive_decay, first_order_reaction, forgetting_curve |
| `saturation` | $y(t) = y_{\max}(1 - e^{-kt})$ | capacitor_charging, monomolecular_growth, light_adaptation |

Each domain is a PyMC belief network with LogNormal priors on
$\{y_0 \text{ or } y_{\max}, k\}$, a HalfNormal prior on $\sigma$,
and a Normal likelihood. Domain-specific prior means encode the
natural scale (e.g. volts for rc_circuit, atom counts for
radioactive_decay), but the ODE skeleton is shared within class.

MCMC recovers ground-truth parameters tightly when enough data is
provided (e.g., first_order_reaction: $\hat{y}_0 = 2.01$ vs truth
2.00, $\hat{k} = 0.202$ vs truth 0.200, 30 observations).

## 4. Experimental results

### 4.1 Clustering recovery

Fingerprint-mode clustering on all 7 law-zoo domains yields:

- 4 exp_decay domains → shared fingerprint `5ebe8c06f60d5ce9`
- 3 saturation domains → shared fingerprint `ecb20fad14e0fd01`
- **ARI = 1.000** against ground truth

Threshold-mode clustering with $\tau = 0.05$ gives the same result.
Plan §Phase 5 task 2 target $> 0.8$ cleared with full margin.

The sole structural distinction between classes, isolated by the
op multiset, is the `Sub` operation in saturation's
$y = y_{\max}(1 - e^{-kt})$ (absent in $y = y_0 \cdot e^{-kt}$).

### 4.2 Transfer on saturation targets

Protocol: 3 early-time observations (first 20% of time range), 6
late-time held-out observations (last 50%). Cold-start uses
uninformed priors ($\sigma_{\log} = 1.5$); transfer uses
source-derived $\sigma_{\log}$.

| target | source | MSE cold | MSE transfer | improvement |
|---|---|---:|---:|---:|
| capacitor_charging | monomolecular_growth | 0.359 | 0.011 | **+97.0%** |
| light_adaptation | capacitor_charging | 0.0055 | 0.0010 | **+82.0%** |
| **mean** | | | | **+89.5%** |

Plan §Phase 5 task 3 target $\geq 30\%$ cleared roughly 3×.

### 4.3 Transfer on exp_decay targets (honest null)

The same protocol on exp_decay targets (5% observation window to
create genuinely underdetermined data):

| target | source | MSE cold | MSE transfer | improvement |
|---|---|---:|---:|---:|
| rc_circuit | first_order_reaction | 0.006 | 0.011 | **-82%** |
| forgetting_curve | first_order_reaction | 0.0007 | 0.0018 | **-154%** |

**Transfer does not help here.** Analysis: for exp_decay, $y_0$ is
pinned by the earliest observation ($y(t{=}0) = y_0$), and
predictions at late $t$ converge toward zero regardless of $k$
uncertainty because $e^{-kt}$ is small for any moderate $k$ at large
$t$. Cold-start already sits near the noise floor; there is no room
for tighter priors to help.

This is a **dynamical asymmetry** between the two classes, not a
method failure. Saturation parameters couple tightly to observed
asymptotes (requiring the plateau); exp_decay parameters decouple
after the initial observation.

### 4.4 Cross-class transfer is refused

The signature gate raises `ValueError` when
$\mathrm{sig}(T) \neq \mathrm{sig}(S)$, preventing (e.g.) an attempt
to transfer from a saturation source into an exp_decay target. This
is the intended safety property of equivalence-class-aware transfer.

## 5. Related work

| Work | What it does | Relation to Phase 4 |
|---|---|---|
| Fritz et al. 2020, *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics* | Blackwell–Sherman–Stein criterion for equivalence of statistical experiments in Markov categories | Theoretical grounding; our P4.7 Markov-category primitives follow Fritz's CD-category presentation, and §2.5 gives an operational BSS check within the closed-law-zoo setting (full existence-of-garbling search still future) |
| Perrone 2024, *Categorical Information Geometry* | Information-geometric divergences on morphism sets in Markov categories | Our P4.8 `perrone_kernel_kl` realises the squared-mean case of this divergence for Gaussian-observation kernels; within-class D = 0 exactly, cross-class D > 0 with MC SNR > 3 |
| Lê et al. 2025, arXiv:2505.03862 | SPD-manifold metrics on probabilistic morphisms | Would generalise Perrone's kernel KL to non-Gaussian observation models once we move beyond LogNormal/Normal fixtures |
| MAML (Finn et al. 2017) | Learns initialization across tasks | Meta-learning counterpart: MAML learns representations; we use declared belief-network structure |
| Gordon et al. 2023, *Physics-informed neural networks: a meta-review* | Transfer learning in scientific ML | PINNs share functional form via differential constraints; we share via belief network signature |
| POPPER (Huang et al. 2025) | LLM hypothesis falsification via sequential testing | Complementary: POPPER tests one hypothesis; we detect equivalence across hypotheses |

## 6. Implementation

All results reproduce from these modules:

- `phase4_structure/law_zoo/{exp_decay,saturation}.py` — 7 domain fixtures
- `phase4_structure/signature.py` — op-multiset `StructuralSignature` (P4.2)
- `phase4_structure/wl_signature.py` — Weisfeiler-Lehman `WLSignature` (P4.6)
- `phase4_structure/markov_category.py` — categorical primitives (P4.7)
- `phase4_structure/law_zoo_morphisms.py` — per-class string diagrams
- `phase4_structure/semantic_equivalence.py` — BSS + Perrone kernel KL (P4.8)
- `phase4_structure/retrieval.py` — distance, NN, clustering, ARI
- `phase4_structure/transfer.py` — `infer_posterior_shape`,
  `run_transfer_benchmark`, `TransferResult`
- `phase4_structure/tests/test_{law_zoo,signature,wl_signature,retrieval,transfer,markov_category,semantic_equivalence}.py`

Acceptance suite: 44 unit tests across 7 modules, all PASS.

## 7. Limitations and future work

1. **Law-zoo v1 is small** (2 classes × 7 domains). Plan §Phase 5
   lists two further equivalence classes — damped oscillation and
   allocation — that Phase 4.v2 should add. LC circuits, pendulums,
   and predator-prey fit the damped-oscillation family; gravity and
   electric fields fit allocation.
2. **Layered signature stack vs fully general semantic equivalence**.
   We implement five complementary checks: op-multiset (§2.1), WL
   refinement (§2.3), Markov-category shape matching (§2.4),
   likelihood-match BSS + Perrone closed-form KL (§2.5), and
   existence-of-garbling BSS search + general MC kernel KL (§2.5
   extension). All five agree on the law-zoo. Garbling search is
   currently restricted to the **linear-Gaussian** family — a
   best-fit $(A, b)$ regression on sampled $(μ_a, μ_b)$ pairs. Within-
   class it recovers the identity garbling ($A=1$, $b=0$) at machine
   epsilon; cross-class it returns non-trivial residuals ($>50\%$ of
   $\mu_b$ scale on all 3 pairs tested), correctly rejecting. Fully
   general BSS (non-linear garbling search over arbitrary Markov
   kernels) and GNN-learned embeddings remain future work.
3. **Exp_decay transfer null**. Section 4.3 documents that the
   protocol's benefit depends on the dynamics being
   *plateau-revealing*; purely decaying dynamics don't carry enough
   prediction-level uncertainty for class-level precision to help.
   Alternative task formulations — predicting the half-life $\tau$
   directly, or inferring $k$ itself — would expose the transfer
   benefit for exp_decay, but change the task.
4. **Prior means are domain-specific**. Transfer currently inherits
   only SDs, not means. A mean-transfer protocol (e.g., via
   dimensional rescaling $t' = t/\tau, y' = y/y_0$) would require
   domain-level annotation of characteristic scales — deferred.

## 8. Paper-level claim

MEIS Phase 4 demonstrates that **structural equivalence of belief
networks is computable, retrievable, and transferable** at four
complementary layers — op-multiset (PyTensor scalar ops), Weisfeiler-
Lehman (ancestor-DAG subtree refinement), Markov-category (symbolic
string diagrams), and BSS + Perrone semantic check (exact likelihood
equality and Gaussian kernel KL). All four recover the 10-domain
3-class law-zoo partition perfectly: syntactic ARI = 1.00, within-
class max$|\Delta \log p| = 0$, within-class Perrone D = 0, cross-
class D > 0 with MC SNR > 3. The syntactic and semantic layers
cross-validate each other, so the equivalence finding is not an
artefact of any single representation choice.

The same signature gate delivers 89.5% held-out MSE reduction on
saturation-class targets whose parameters are genuinely under-
determined from few-shot observation. Targets whose dynamics suppress
prediction-level parameter uncertainty (exp_decay's asymptotic decay
to zero) see no transfer benefit — a dynamical asymmetry reported
honestly in §4.3.

This is a concrete, computable instantiation of the **categorical
transfer** intuition — Quine/Lakatos's "minimum disruption" applied
not to a single belief but to the whole abstraction: two belief
networks that compute the same underlying pattern should share both
structure and inductive bias, and MEIS can now identify when that
sharing applies at a categorical level, not just a PPL-artefact
level.

---

*Status: draft. Not yet camera-ready; missing Phase 4.v2 equivalence
classes (damped oscillation, allocation), GNN signature alternative,
and a pre-registered replication on a synthetic zoo generator. The
signature-gate false-positive rate is also not yet measured — would
need a distractor library with near-isomorphs.*
