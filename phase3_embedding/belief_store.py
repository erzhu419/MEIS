"""MEIS Phase 2 P2.3 — persistent belief network (commit 2: in-memory).

Dataclasses + operations for a belief network that bridges the Phase 1
prior library and the Phase 1-2 KL-drift rankers. This commit is pure
in-memory: `snapshot()` / `rollback()` work via dicts; disk I/O lands in
commit 3. See `md/P2_3_persistent_belief_network_design.md` for the full
design.

Scope of this commit:
  - Dataclasses: PosteriorHandle / Node / Relation / Evidence
  - BeliefStore.from_library: populate nodes/relations from a PriorLibrary
  - BeliefStore.get_node / search_nodes: retrieval
  - BeliefStore.add_evidence: conjugate-Gaussian update path
  - BeliefStore.rank_hypotheses: delegate to kl_drift.GaussianPosterior
  - BeliefStore.snapshot / rollback

Out of scope (commit 3+):
  - Disk persistence (`load` / `save`)
  - Non-conjugate (MCMC) `add_evidence` dispatch
  - Structural hypothesis support (add-node, add-relation)
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from phase3_embedding.kl_drift import (
    GaussianPosterior, condition_normal, kl_normal,
    Hypothesis, EmbeddingScore, rank_hypotheses as kl_rank_hypotheses,
)


# =============================================================================
# Posterior representation
# =============================================================================
@dataclass
class PosteriorHandle:
    """Unified handle for a node's current posterior. Two representations:

      kind='gaussian': closed-form Normal — mu, sigma required.
      kind='samples':  MCMC sample array — samples (N,) or (N, d) required.
      kind='deterministic': constant value (e.g. g=9.81) — mu required, sigma=0.

    Provides `.as_gaussian()` to hand off to phase3_embedding.kl_drift
    (which expects a GaussianPosterior dataclass).
    """
    kind: str
    mu: float | np.ndarray | None = None
    sigma: float | np.ndarray | None = None
    samples: np.ndarray | None = None

    def as_gaussian(self) -> GaussianPosterior:
        if self.kind == "gaussian":
            assert self.mu is not None and self.sigma is not None
            return GaussianPosterior(mu=float(self.mu), sigma=float(self.sigma))
        if self.kind == "deterministic":
            assert self.mu is not None
            # Represent a delta as a tiny-σ Gaussian for rankers that
            # need positive variance.
            return GaussianPosterior(mu=float(self.mu), sigma=1e-12)
        if self.kind == "samples":
            assert self.samples is not None and self.samples.ndim == 1
            return GaussianPosterior(
                mu=float(np.mean(self.samples)),
                sigma=float(np.std(self.samples, ddof=1)),
            )
        raise ValueError(f"unknown posterior kind: {self.kind}")


# =============================================================================
# Graph elements
# =============================================================================
@dataclass
class Node:
    id: str
    domain: str
    name: str
    type: str                            # "continuous" | "discrete" | "categorical"
    posterior: PosteriorHandle
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # EvidenceID list
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Relation:
    id: str
    from_nodes: list[str]
    to_node: str
    relation_expr: str                   # Python-evaluable formula string
    noise_model: str = "Normal"          # "Normal" | "Poisson" | "Deterministic"
    noise_params: dict[str, float] = field(default_factory=dict)
    source: str = ""                     # EvidenceID / LibraryEntryID
    strength: float = 1.0                # confidence in [0, 1]


@dataclass
class Evidence:
    id: str
    kind: str                            # "observation" | "hypothesis_accepted" | "library_prior"
    target_nodes: list[str]
    value: Any                           # scalar, list, tuple, dict — schema depends on `kind`
    x: float | None = None               # covariate for conjugate updates (y = theta*x + noise)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provenance: str = ""                 # run id / human note


# =============================================================================
# The store
# =============================================================================
_TOKEN_RE = re.compile(r"[^a-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.split(text.lower()) if t}


@dataclass
class BeliefStore:
    nodes: dict[str, Node] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    # -- Construction from Phase 1 library ----------------------------------
    @classmethod
    def from_library(cls, library) -> "BeliefStore":
        """Parse each PriorLibrary entry into either a Node (if its `formal`
        block defines a univariate distribution on its own variable) or a
        Relation (if it defines an expression linking vars). Entries with
        neither are skipped with a qualitative_only tag (retrievable via
        tags, just no Bayesian update path)."""
        store = cls()
        for entry in library.entries:
            eid = entry["id"]
            domain = entry.get("domain", "uncategorized")
            formal = entry.get("formal", {}) or {}
            tags = list(entry.get("keywords", []))
            vars_involved = list(entry.get("vars_involved", []))

            # Case A: distribution-over-single-variable → Node.
            # Two schemas accepted:
            #   (i)  formal.distribution = "Normal", formal.mu, formal.sigma
            #        (older entries like human_density_adult)
            #   (ii) formal.distribution = {"type": "Normal", "mu": m, "sigma": s}
            #        (newer entries like adult_height_distribution)
            #
            # Entries that have BOTH a distribution AND a relation (e.g.
            # bmi_adult_population) go into Case B as a relation.
            dist = formal.get("distribution")
            has_relation = isinstance(formal.get("relation"), str)
            is_normal_dist = (
                (isinstance(dist, dict) and dist.get("type") == "Normal")
                or (dist == "Normal" and "mu" in formal and "sigma" in formal)
            )
            if is_normal_dist and not has_relation:
                if isinstance(dist, dict):
                    mu_val, sigma_val = float(dist["mu"]), float(dist["sigma"])
                else:
                    mu_val, sigma_val = float(formal["mu"]), float(formal["sigma"])
                node = Node(
                    id=eid, domain=domain,
                    name=vars_involved[0] if vars_involved else eid,
                    type="continuous",
                    posterior=PosteriorHandle(
                        kind="gaussian", mu=mu_val, sigma=sigma_val,
                    ),
                    tags=tags,
                    sources=[f"library:{eid}"],
                )
                store.nodes[eid] = node
                continue

            # Case B: relation string → Relation (+ scalar parameter nodes)
            relation_expr = formal.get("relation")
            params = formal.get("parameters_hint") or formal.get("parameters") or {}
            if isinstance(relation_expr, str):
                rel = Relation(
                    id=eid,
                    from_nodes=vars_involved[1:] if len(vars_involved) > 1 else [],
                    to_node=vars_involved[0] if vars_involved else "",
                    relation_expr=relation_expr,
                    noise_model=formal.get("noise_model", "Normal"),
                    noise_params=formal.get("noise_params", {}),
                    source=f"library:{eid}",
                    strength=1.0 if entry.get("confidence") in ("high", "exact") else 0.6,
                )
                store.relations[eid] = rel
                # Best-effort: if the entry explicitly states `theta_mu / theta_sigma`
                # in `parameters`, spawn a scalar Node for that latent (needed so
                # hypothesis-ranking can target it).
                for k, v in params.items():
                    if k.endswith("_mu"):
                        base = k[:-3]
                        sigma_key = f"{base}_sigma"
                        if sigma_key in params:
                            node_id = f"{eid}::{base}"
                            store.nodes[node_id] = Node(
                                id=node_id, domain=domain, name=base,
                                type="continuous",
                                posterior=PosteriorHandle(
                                    kind="gaussian",
                                    mu=float(v), sigma=float(params[sigma_key]),
                                ),
                                tags=tags + [base],
                                sources=[f"library:{eid}"],
                            )
                continue

            # Case C: qualitative only — no posterior update path, just
            # retrievable by tags
            store.nodes[eid] = Node(
                id=eid, domain=domain, name=eid,
                type="categorical",
                posterior=PosteriorHandle(kind="deterministic", mu=0.0),
                tags=tags + ["qualitative"],
                sources=[f"library:{eid}"],
            )
        return store

    # -- Retrieval ----------------------------------------------------------
    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def search_nodes(self, query: str, k: int = 5,
                     domain: str | None = None) -> list[Node]:
        """Bag-of-words score over tags + id + name."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, Node]] = []
        for node in self.nodes.values():
            if domain is not None and node.domain != domain:
                continue
            kw = {t for tag in node.tags for t in _tokenize(tag)}
            name_t = _tokenize(node.name) | _tokenize(node.id)
            score = 3.0 * len(q_tokens & kw) + 1.0 * len(q_tokens & name_t)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:k]]

    # -- Conjugate evidence update -----------------------------------------
    def add_evidence(self, ev: Evidence, obs_sigma: float | None = None) -> None:
        """Conjugate Gaussian update path.

        Expected shapes:
          - ev.target_nodes = [theta_node_id]
          - ev.x = covariate value (e.g. height_cm ** 3)
          - ev.value = observation y (e.g. weight_kg)
          - ev.kind = "observation"

        Updates the target node's posterior in-place and records the
        evidence in self.evidence (append-only in memory for now)."""
        assert len(ev.target_nodes) == 1, \
            "commit-2 only supports single-latent conjugate updates"
        tid = ev.target_nodes[0]
        node = self.nodes[tid]
        if node.posterior.kind != "gaussian":
            raise NotImplementedError(
                "commit-2 only supports Gaussian-conjugate updates; "
                "non-conjugate MCMC path is commit-3 scope")
        if obs_sigma is None or ev.x is None:
            raise ValueError(
                "conjugate update requires obs_sigma and ev.x (covariate)")
        prior = node.posterior.as_gaussian()
        post = condition_normal(prior,
                                np.array([ev.x]),
                                np.array([float(ev.value)]),
                                obs_sigma)
        node.posterior = PosteriorHandle(kind="gaussian", mu=post.mu, sigma=post.sigma)
        node.sources.append(ev.id)
        node.last_updated_at = datetime.utcnow().isoformat()
        self.evidence.append(ev)

    # -- Hypothesis ranking (delegates to Phase 1 ranker) ------------------
    def rank_hypotheses(self, hypotheses: list[Hypothesis],
                        latent_var: str) -> list[EmbeddingScore]:
        """Rank hypotheses by KL drift on `latent_var`'s posterior.

        Delegates to phase3_embedding.kl_drift.rank_hypotheses, so this
        gives exact (closed-form) scores for any Gaussian latent.
        Call `BeliefStore.to_rank_mcmc(...)` (to be added commit 3)
        for non-Gaussian latents.
        """
        node = self.nodes[latent_var]
        posterior = node.posterior.as_gaussian()
        return kl_rank_hypotheses(posterior, hypotheses)

    def minimum_embedding_distance(self, h: Hypothesis,
                                   latent_var: str) -> float:
        """Single-hypothesis convenience wrapper: D(h, B) = KL drift."""
        return self.rank_hypotheses([h], latent_var)[0].kl_from_base

    # -- Snapshot / rollback (in-memory state) -----------------------------
    def snapshot(self) -> dict:
        return copy.deepcopy({
            "nodes": {k: v for k, v in self.nodes.items()},
            "relations": {k: v for k, v in self.relations.items()},
            "evidence": list(self.evidence),
        })

    def rollback(self, snap: dict) -> None:
        self.nodes = copy.deepcopy(snap["nodes"])
        self.relations = copy.deepcopy(snap["relations"])
        self.evidence = list(snap["evidence"])

    # -- Summary (for logging / debugging) ---------------------------------
    def summary(self) -> str:
        return (f"BeliefStore(nodes={len(self.nodes)}, "
                f"relations={len(self.relations)}, "
                f"evidence={len(self.evidence)})")
