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
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
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

    # -- Conjugate / non-conjugate evidence update -------------------------
    def add_evidence(self, ev: Evidence, obs_sigma: float | None = None,
                     likelihood: str = "normal",
                     mcmc_draws: int = 600, mcmc_tune: int = 400,
                     mcmc_chains: int = 2,
                     mcmc_seed: int = 0) -> None:
        """Update the target node's posterior with a new observation.

        Dispatch:
          - likelihood="normal"  AND node.posterior.kind == "gaussian":
              closed-form conjugate update via kl_drift.condition_normal.
              Requires ev.x (covariate) and obs_sigma.
          - likelihood="poisson":
              MCMC rebuild of a log-rate-linear-in-theta Poisson model:
                  y ~ Poisson(exp(theta * ev.x))
              Starts from the node's current posterior (gaussian → prior;
              samples → Gaussian-moment approximation), samples new
              posterior, stores as samples-based PosteriorHandle.

        Future: other likelihoods (LogNormal, Binomial) via the same
        MCMC-rebuild pattern. The dispatch is deliberately explicit so
        subtle modeling choices (link functions, noise distributions)
        are the caller's responsibility, not magic."""
        assert len(ev.target_nodes) == 1, \
            "single-latent updates only; multi-latent needs explicit model builder"
        tid = ev.target_nodes[0]
        node = self.nodes[tid]

        if likelihood == "normal":
            if node.posterior.kind != "gaussian":
                raise NotImplementedError(
                    "Gaussian-conjugate update requires gaussian posterior; "
                    "call with likelihood='poisson' or similar for sample-based nodes."
                )
            if obs_sigma is None or ev.x is None:
                raise ValueError("conjugate update requires obs_sigma and ev.x")
            prior = node.posterior.as_gaussian()
            post = condition_normal(prior,
                                    np.array([ev.x]),
                                    np.array([float(ev.value)]),
                                    obs_sigma)
            node.posterior = PosteriorHandle(kind="gaussian", mu=post.mu, sigma=post.sigma)

        elif likelihood == "poisson":
            if ev.x is None:
                raise ValueError("Poisson update requires ev.x (covariate)")
            # Extract current posterior as Gaussian prior for MCMC
            current = node.posterior.as_gaussian()
            import pymc as pm
            with pm.Model():
                theta = pm.Normal("theta", mu=current.mu, sigma=current.sigma)
                # Log-rate linear in theta: y ~ Poisson(exp(theta * x))
                rate = pm.math.exp(theta * float(ev.x))
                pm.Poisson("y_obs", mu=rate, observed=np.array([int(round(float(ev.value)))]))
                trace = pm.sample(
                    draws=mcmc_draws, tune=mcmc_tune, chains=mcmc_chains,
                    random_seed=mcmc_seed, progressbar=False,
                    return_inferencedata=False, compute_convergence_checks=False,
                )
            samples = np.asarray(trace["theta"])
            node.posterior = PosteriorHandle(kind="samples", samples=samples)

        else:
            raise ValueError(f"unsupported likelihood: {likelihood}")

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

    # -- Disk I/O (commit 3) ------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise the full store to `path/{nodes,relations,evidence}/*.json`.

        Guarantees:
          - Atomic per-file writes (tempfile + fsync + os.replace)
          - Evidence is **append-only**: existing evidence files are never
            overwritten or deleted — they may only be augmented by new ids.
          - Node/relation files CAN be overwritten (posterior updates), but
            a `.prev.json` backup is kept beside each for one-step rollback.
          - Sample-based posteriors save samples to `.samples.npz` beside
            the JSON; the JSON references the npz file by relative path.
        """
        root = Path(path)
        (root / "nodes").mkdir(parents=True, exist_ok=True)
        (root / "relations").mkdir(parents=True, exist_ok=True)
        (root / "evidence").mkdir(parents=True, exist_ok=True)

        for node in self.nodes.values():
            self._save_node(root, node)
        for rel in self.relations.values():
            _atomic_write_json(root / "relations" / _sanitize(rel.id, ".json"),
                               _relation_to_dict(rel))
        # Evidence append-only: write only files that don't exist yet.
        for ev in self.evidence:
            target = root / "evidence" / _sanitize(ev.id, ".json")
            if not target.exists():
                _atomic_write_json(target, _evidence_to_dict(ev))

    def _save_node(self, root: Path, node: Node) -> None:
        nodes_dir = root / "nodes"
        target = nodes_dir / _sanitize(node.id, ".json")
        # Preserve previous-version copy for one-step rollback (per design §3).
        if target.exists():
            prev = nodes_dir / _sanitize(node.id, ".prev.json")
            prev.write_bytes(target.read_bytes())
        payload, samples_arr = _node_to_dict(node)
        if samples_arr is not None:
            npz_rel = _sanitize(node.id, ".samples.npz")
            np.savez_compressed(nodes_dir / npz_rel, samples=samples_arr)
            payload["posterior"]["samples_ref"] = f"nodes/{npz_rel}"
        _atomic_write_json(target, payload)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefStore":
        """Load the full store from the on-disk layout produced by `.save`."""
        root = Path(path)
        store = cls()
        for p in sorted((root / "nodes").glob("*.json")):
            if p.name.endswith(".prev.json"):
                continue
            payload = json.loads(p.read_text())
            node = _node_from_dict(payload, root)
            store.nodes[node.id] = node
        for p in sorted((root / "relations").glob("*.json")):
            store.relations[p.stem.replace("__", "::")] = _relation_from_dict(
                json.loads(p.read_text()))
        # Evidence ordered by timestamp field when present, else filename.
        ev_paths = sorted((root / "evidence").glob("*.json"))
        for p in ev_paths:
            store.evidence.append(_evidence_from_dict(json.loads(p.read_text())))
        # Stable chronological ordering by timestamp when available.
        store.evidence.sort(key=lambda e: e.timestamp or "")
        return store

    # -- Summary (for logging / debugging) ---------------------------------
    def summary(self) -> str:
        return (f"BeliefStore(nodes={len(self.nodes)}, "
                f"relations={len(self.relations)}, "
                f"evidence={len(self.evidence)})")


# =============================================================================
# Serialisation helpers (module-level, commit 3)
# =============================================================================
_UNSAFE = re.compile(r"[^a-zA-Z0-9_.\-]+")


def _sanitize(node_id: str, suffix: str) -> str:
    """Filesystem-safe filename while keeping the id humanly recognisable.
    `::` becomes `__` (reversible); any other unsafe char becomes `_`."""
    safe = node_id.replace("::", "__")
    safe = _UNSAFE.sub("_", safe)
    return safe + suffix


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tempfile in same directory → fsync → os.replace (atomic on Linux)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, default=_json_default)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        # Clean up tempfile if anything went wrong before the rename.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _node_to_dict(node: Node) -> tuple[dict, np.ndarray | None]:
    """Returns (payload_dict, samples_arr_or_None).

    Samples-based posteriors are kept out of the JSON and returned separately
    so the caller can npz-compress them beside the node file.
    """
    posterior_d: dict[str, Any] = {"kind": node.posterior.kind}
    samples_arr: np.ndarray | None = None
    if node.posterior.kind == "gaussian":
        posterior_d["mu"] = float(node.posterior.mu)  # type: ignore[arg-type]
        posterior_d["sigma"] = float(node.posterior.sigma)  # type: ignore[arg-type]
    elif node.posterior.kind == "deterministic":
        posterior_d["mu"] = float(node.posterior.mu)  # type: ignore[arg-type]
    elif node.posterior.kind == "samples":
        samples_arr = np.asarray(node.posterior.samples, dtype=float)
    return {
        "id": node.id,
        "domain": node.domain,
        "name": node.name,
        "type": node.type,
        "posterior": posterior_d,
        "tags": list(node.tags),
        "sources": list(node.sources),
        "created_at": node.created_at,
        "last_updated_at": node.last_updated_at,
    }, samples_arr


def _node_from_dict(payload: dict, root: Path) -> Node:
    p = payload["posterior"]
    if p["kind"] == "gaussian":
        post = PosteriorHandle(kind="gaussian", mu=float(p["mu"]), sigma=float(p["sigma"]))
    elif p["kind"] == "deterministic":
        post = PosteriorHandle(kind="deterministic", mu=float(p["mu"]))
    elif p["kind"] == "samples":
        npz_path = root / p["samples_ref"]
        with np.load(npz_path) as z:
            samples = z["samples"].astype(float)
        post = PosteriorHandle(kind="samples", samples=samples)
    else:
        raise ValueError(f"unknown posterior kind on disk: {p['kind']}")
    return Node(
        id=payload["id"],
        domain=payload["domain"],
        name=payload["name"],
        type=payload["type"],
        posterior=post,
        tags=list(payload.get("tags", [])),
        sources=list(payload.get("sources", [])),
        created_at=payload.get("created_at", ""),
        last_updated_at=payload.get("last_updated_at", ""),
    )


def _relation_to_dict(rel: Relation) -> dict:
    return {
        "id": rel.id,
        "from_nodes": list(rel.from_nodes),
        "to_node": rel.to_node,
        "relation_expr": rel.relation_expr,
        "noise_model": rel.noise_model,
        "noise_params": dict(rel.noise_params),
        "source": rel.source,
        "strength": float(rel.strength),
    }


def _relation_from_dict(payload: dict) -> Relation:
    return Relation(
        id=payload["id"],
        from_nodes=list(payload.get("from_nodes", [])),
        to_node=payload.get("to_node", ""),
        relation_expr=payload.get("relation_expr", ""),
        noise_model=payload.get("noise_model", "Normal"),
        noise_params=dict(payload.get("noise_params", {})),
        source=payload.get("source", ""),
        strength=float(payload.get("strength", 1.0)),
    )


def _evidence_to_dict(ev: Evidence) -> dict:
    # ev.value can be scalar / list / tuple / dict — JSON handles all with
    # _json_default for numpy scalars.
    val = ev.value
    if isinstance(val, np.ndarray):
        val = val.tolist()
    return {
        "id": ev.id,
        "kind": ev.kind,
        "target_nodes": list(ev.target_nodes),
        "value": val,
        "x": None if ev.x is None else float(ev.x),
        "timestamp": ev.timestamp,
        "provenance": ev.provenance,
    }


def _evidence_from_dict(payload: dict) -> Evidence:
    return Evidence(
        id=payload["id"],
        kind=payload.get("kind", "observation"),
        target_nodes=list(payload.get("target_nodes", [])),
        value=payload.get("value"),
        x=payload.get("x"),
        timestamp=payload.get("timestamp", ""),
        provenance=payload.get("provenance", ""),
    )
