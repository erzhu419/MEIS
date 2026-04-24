"""P4.6 — Weisfeiler-Lehman graph-kernel signature.

Alternative to the op-multiset fingerprint of P4.2: iterate the
classical Weisfeiler-Lehman subtree-refinement procedure on the
PyTensor ancestor DAG rooted at each observed RV.

Each iteration updates a node's colour by hashing its own current
colour together with the multiset of its parents' colours. After K
iterations, the signature is a canonical hash of the final colour
multiset across all nodes.

Why bother over op-multiset?

  Op-multiset collapses the DAG to a bag of ops — it is correct on
  the law-zoo because the three classes already differ on at least
  one op (Sub in saturation, Cos in damped oscillation). WL captures
  LOCAL STRUCTURE: two graphs with the same op counts but different
  wiring get different WL signatures. This is what we want as the
  law zoo grows: same-ops-different-wiring distractors would collapse
  the op-multiset signature but survive WL.

This module provides WL as an alternative, not a replacement: we keep
op-multiset because it is simpler and cheaper. On the current v2
law-zoo (3 classes × 10 domains) both recover ARI = 1.0; test
test_wl_vs_op_multiset explicitly constructs a distractor pair where
they diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

try:
    from pytensor.graph.traversal import ancestors
except ImportError:
    from pytensor.graph.basic import ancestors
from pytensor.tensor.elemwise import Elemwise


def _op_name(op) -> str:
    if isinstance(op, Elemwise):
        return type(op.scalar_op).__name__
    return type(op).__name__


def _initial_label(var) -> str:
    if var.owner is None:
        # constants and RNG state, roughly
        return "Leaf"
    return _op_name(var.owner.op)


@dataclass(frozen=True)
class WLSignature:
    colour_multiset: tuple  # sorted tuple of final colours (8-hex each)
    num_nodes: int
    num_iterations: int
    fingerprint: str        # 16-hex sha256 of colour_multiset

    def matches(self, other: "WLSignature") -> bool:
        return self.fingerprint == other.fingerprint


def _collect_ancestor_graph(model) -> tuple[set, dict]:
    """Return (nodes, parents) where parents[v] is the list of v's
    ancestor tensors that appear in the same set (i.e. the DAG is
    restricted to ancestors of observed RVs)."""
    nodes = set()
    for rv in model.observed_RVs:
        for v in ancestors([rv]):
            nodes.add(v)
        nodes.add(rv)

    parents = {}
    for v in nodes:
        if v.owner is None:
            parents[v] = []
        else:
            parents[v] = [p for p in v.owner.inputs if p in nodes]
    return nodes, parents


def extract_wl_signature(model, num_iterations: int = 3) -> WLSignature:
    nodes, parents = _collect_ancestor_graph(model)

    colours: dict = {v: _initial_label(v) for v in nodes}
    for _ in range(num_iterations):
        new_colours = {}
        for v in nodes:
            nb = tuple(sorted(colours[p] for p in parents[v]))
            payload = (colours[v] + "|" + str(nb)).encode()
            new_colours[v] = hashlib.sha256(payload).hexdigest()[:8]
        colours = new_colours

    multiset = tuple(sorted(colours.values()))
    fp = hashlib.sha256(repr(multiset).encode()).hexdigest()[:16]

    return WLSignature(
        colour_multiset=multiset,
        num_nodes=len(nodes),
        num_iterations=num_iterations,
        fingerprint=fp,
    )


def wl_distance(a: WLSignature, b: WLSignature) -> float:
    """Ruzicka distance on the colour multiset (analogous to
    signature.py's op-multiset distance)."""
    from collections import Counter
    ca = Counter(a.colour_multiset)
    cb = Counter(b.colour_multiset)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.0
    num = sum(min(ca[k], cb[k]) for k in keys)
    den = sum(max(ca[k], cb[k]) for k in keys)
    return 1.0 - num / den if den else 0.0


def wl_signature_for_domain(domain_module, t_obs, y_obs,
                            num_iterations: int = 3) -> WLSignature:
    model = domain_module.build_model(t_obs, y_obs)
    return extract_wl_signature(model, num_iterations=num_iterations)
