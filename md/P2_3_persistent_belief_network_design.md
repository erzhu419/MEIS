# P2.3 — Persistent Belief Network Design Spec

> Design-only document. No code in this commit. Purpose: pin down the
> data structure, storage format, update semantics, and retrieval API for
> a belief network that accumulates **across runs / tasks** (not reset
> on each env restart), so the MEIS system can build up and act on
> durable cross-domain beliefs the way the blueprint §3 describes.

## 1. Why we need it

Phase 1 every run is stateless. Each `env.reset()` discards observations,
each scientist starts from scratch given only the library + env priors +
10 fresh observations. That works for single-task evaluation but throws
away the thing the blueprint §3 explicitly calls out:

> "L1 表示层 — 持久多域信念网络 — 跨任务累积"

The MEIS thesis is that cross-domain priors compose coherently. You
can't test that composition without a substrate that actually persists.

Concrete examples of what persistence unlocks:

1. **Sequential experiments**. Run 1 observes heights/weights of
   10 adults. Run 2 is asked about *the same person* at a different
   input — should condition on Run 1's posterior.
2. **Cross-domain composition**. Dugongs posterior learned in Run 1
   can inform peregrines Run 2's prior on "rise-and-fall lifespan"
   via a shared `growth_curve_shape` abstract node.
3. **Hypothesis re-ranking over time**. Step 6 / P2.2 ranks hypotheses
   at one time point. A persistent BN lets us see how rankings move
   as new evidence arrives.
4. **Minimum-embedding regret analysis**. Given a full history of
   hypotheses posed + resolved, we can check whether low-KL hypotheses
   systematically turn out to be true (the real Lakatos claim).

## 2. Required abstractions

Three types of entities live in the BN:

### 2.1 Nodes

A **node** represents a scalar or vector latent variable with a
maintained posterior.

Required fields:
```
id:             str            # stable identifier, e.g. "dugong_theta_run7"
domain:         str            # "human_body" | "dynamics" | ...
name:           str            # human-readable label
type:           "continuous" | "discrete" | "categorical"
support:        tuple | str    # e.g. (0.0, inf) or "bool"
posterior:      PosteriorHandle  # see §2.3
sources:        list[EvidenceID]  # all pieces of evidence informing this node
tags:           set[str]       # keywords for retrieval overlap with library
```

Optional fields:
```
parent_nodes:   list[NodeID]   # explicit edges (if known)
functional_form: str           # Python expression as string, variables
                               #   referenced by parent NodeID
created_at:     ISO8601
last_updated_at: ISO8601
```

### 2.2 Edges (causal / functional relations)

A **relation** links nodes via a parametric expression.

```
id:           str
from_nodes:   list[NodeID]     # left-hand variables
to_node:      NodeID           # right-hand variable
relation:     str              # Python expr, e.g. "theta * height**3"
noise_model:  str              # "Normal" | "Poisson" | "LogNormal" | "Deterministic"
noise_params: dict             # e.g. {"sigma": 2.0}
source:       EvidenceID       # library entry, or data-driven inference
strength:     float            # confidence in [0, 1]
```

### 2.3 Posterior handles

Posteriors come in two representations — closed-form (for Gaussian
conjugate cases) and sample-based (for general PyMC models):

```python
@dataclass
class PosteriorHandle:
    kind: "gaussian" | "samples"
    # for gaussian:
    mu: float | np.ndarray
    sigma: float | np.ndarray
    cov: np.ndarray | None         # full-cov optional
    # for samples:
    samples: np.ndarray             # (n_draws, dim)
    n_effective: int                # effective sample size for honest variance
```

The handle is the bridge between this persistent structure and both
  `phase3_embedding/kl_drift.py` (Gaussian, Phase 1)
  `phase3_embedding/kl_drift_mcmc.py` (any, Phase 2)
so `rank_hypotheses()` and `rank_hypotheses_mcmc()` become methods on
the BN instead of free functions.

### 2.4 Evidence / observations

An **evidence atom** is an observation that updated one or more nodes'
posteriors. Must be retained for audit, for re-running posterior
updates with a different model, and for the minimum-embedding-regret
longitudinal check.

```
id:           str
kind:         "observation" | "hypothesis_accepted" | "library_prior"
observed_var: NodeID | RelationID
value:        any                  # scalar, list, tuple, ...
timestamp:    ISO8601
provenance:   str                  # which run generated this
```

## 3. Storage layer

Requirements (in order of importance):

1. **Plain-text serialisation** — human-readable diff in git.
2. **Append-only where possible** — minimise concurrency bugs.
3. **Small deps** — no graph DB, no sqlite requirement.
4. **Fast load** — all nodes/edges fit in memory.

Proposed concrete scheme:

```
phase3_belief_store/
├── nodes/
│   ├── dugong_theta.json        # one JSON file per node
│   ├── human_body_density.json  # id==filename-stem
│   └── ...
├── relations/
│   ├── weight_from_height_cube.json
│   └── ...
├── evidence/
│   ├── run_2026-04-23T12-33-05_dugongs_seed1.json
│   ├── run_2026-04-23T12-45-10_alice_charlie_seed7.json
│   └── ...
└── index.json                   # optional: node→relation membership,
                                 #   cross-references, last-modified
```

- Posteriors in `gaussian` kind: store mu / sigma directly as JSON floats.
- Posteriors in `samples` kind: store a compressed `.npz` beside the
  node's JSON, reference it by relative path.

Concurrent writers: at most one runner at a time per evidence file;
`evidence/` is append-only (filename includes timestamp+seed, never
overwritten). Node files are overwritten on each update but with a
`.prev.json` kept for one-step rollback.

Atomic write pattern: `write→temp→fsync→rename` on Linux (`os.rename`
is atomic within a single filesystem).

## 4. Update semantics (the hard part)

Given: existing node `N` with posterior `P_prev`, evidence `E` to add.

Three cases:

### Case A: conjugate Gaussian

`P_prev` is a `GaussianPosterior`, and the evidence is `(x, y)` pairs
with known `Normal(x·θ, σ)` likelihood. Use
`condition_normal(P_prev, x, y, σ)` from `kl_drift.py`. O(k) update,
no MCMC.

### Case B: non-conjugate, PyMC-representable

Rebuild the whole model in a `pm.Model()`:
  - each node's prior = its current `P_prev` (use Normal-approx of samples
    if it's sample-based, or the Gaussian closed form if conjugate)
  - add evidence `E` as observed variables
  - sample posterior, update each node's handle

Tricky: the Normal-approx step loses information. Mitigation: also
retain a **base-level** posterior (the earliest Gaussian prior +
full list of evidence atoms), and periodically re-MCMC from scratch
over the full evidence list to refresh. Never approximate twice in a
row without a re-ground.

### Case C: structural — hypothesis adds a new node or relation

Must explicitly append new graph elements:
  - New node: create JSON, default prior from library if vars_involved match
  - New relation: create JSON, attach to existing nodes' source list

These are the "|Δ structure|" events in the blueprint's composite
score: `D(h, B) = KL(...) + λ · |Δ_structure|`.

## 5. Retrieval API (what callers use)

```python
class BeliefStore:
    def load(path: str) -> "BeliefStore": ...
    def save(path: str) -> None: ...
    def get_node(id: str) -> Node: ...
    def search_nodes(query: str) -> list[Node]: ...  # tag overlap
    def add_evidence(evidence: Evidence, relations_used: list[RelationID]) -> None: ...
    def rank_hypotheses(hypotheses: list[Hypothesis], latent_var: str) -> list[Score]: ...
    def minimum_embedding_distance(hypothesis: Hypothesis) -> float: ...
    def snapshot() -> dict: ...     # for test fixtures
    def rollback(to_snapshot: dict) -> None: ...
```

## 6. Testing surface (before we code)

Before writing a line of implementation, define these tests (they
double as acceptance criteria):

- `test_fresh_store_from_library()` — start with empty store, add
  all `phase2_prior_library` entries as nodes/relations, assert
  count matches 20.
- `test_single_conjugate_update_matches_phase1()` — add Gaussian-
  conjugate evidence, check posterior equals `kl_drift.condition_normal`
  result.
- `test_non_conjugate_update_via_mcmc()` — add a Poisson evidence,
  posterior is sample-based, mean matches within 2·SE of ground truth.
- `test_hypothesis_ranking_persists()` — run KL ranker twice, get
  same order (deterministic given seed).
- `test_evidence_is_append_only()` — add evidence, later add more,
  confirm old file exists with its original contents.
- `test_rollback_to_snapshot()` — snapshot, add evidence, rollback,
  confirm state matches snapshot exactly.
- `test_cross_domain_query()` — query "human body density" returns
  nodes/relations from human_body AND any that list it as a source.

## 7. Migration path

1. **Commit 1 (this doc)** — design spec, acceptance tests named.
2. **Commit 2** — `phase3_embedding/belief_store.py` with dataclasses,
   serialisation, in-memory ops, no actual persistence yet.
3. **Commit 3** — disk I/O, atomic writes, .npz for samples.
4. **Commit 4** — wire into `run_mvp_unified.py` as optional
   `--persist-belief <dir>` flag so old runs keep working.
5. **Commit 5** — sequential-experiment demo: run #1 adds evidence,
   run #2 loads the store, runs with the updated prior.

Each commit has its tests from §6.

## 8. What this does NOT try to solve (scope gate)

- Graph learning (discovering edges from data alone). Relations are
  library-provided or explicitly added by the runner.
- Fast incremental sampling (pymc4 + variational approximations).
  We take the hit of re-running MCMC on each evidence batch.
- Concurrent multi-process writers. Single-runner assumption holds
  until we have evidence it doesn't.
- Ontology-level questions (what counts as the "same" node across
  runs). Handled by id-matching; users who want merging must do it
  explicitly.

## 9. Relation to existing modules

| piece | status | relation to this spec |
|---|---|---|
| `phase2_prior_library/retrieval.py` | done | feeds initial nodes/relations on first-time store creation |
| `phase3_embedding/kl_drift.py` | done | directly usable as `update` for Gaussian case |
| `phase3_embedding/kl_drift_mcmc.py` | done (P2.2) | same for non-Gaussian |
| `phase1_mvp/agents/prior_injecting_experimenter.py` | done | candidate consumer: reads nodes matching env keywords, injects into system prompt |
| `phase1_mvp/run_mvp_unified.py` | done | candidate consumer: writes evidence back after each run |

## 10. Decision log — why these choices

- **Flat JSON files, not sqlite**: human-diff'able, git-friendly, zero
  extra deps. Slower than sqlite at scale but we won't have O(10⁶) nodes
  for a long time.
- **Separate evidence/ dir, append-only**: matches blueprint's Lakatos
  theme — you never erase evidence, only refine beliefs around it.
- **No graph DB (neo4j etc.)**: graph ops we need (parent lookup, tag
  overlap) are O(edges) where edges are at most a few hundred for
  foreseeable Phase 2. Keeping it in-memory + JSON is simpler.
- **Sample-based and closed-form handles both supported**: inevitable
  for cross-domain cases (Gaussian on human_body, Poisson on peregrines).
- **Runner flag, not always-on**: users doing headline evaluations
  (Phase 1 style) shouldn't have to deal with state. Only opt-in.

## 11. Open questions for later

- What's the right `λ` in the composite `D(h) = KL + λ·|Δ_structure|`?
  Probably tuned against the minimum-embedding-regret longitudinal test.
- Evidence expiry: if a node hasn't been queried in N runs, should its
  posterior decay toward the prior? (Inspired by recency-weighted Bayes.)
- Multi-store composition: can two separate BNs be merged? What happens
  to evidence?

All three are Phase 3 territory. Don't design them now.
