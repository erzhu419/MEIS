"""P4.2 — Structural signature for belief networks.

Approach (lean MVP before jumping to GNNs):

  1. Walk the PyTensor op graph rooted at each observed RV and collect a
     sorted multiset of op names (exp, neg, mul, sub, add, ...). This
     captures the *functional-form* skeleton of the likelihood.

  2. Collect role labels for every named RV from the caller-supplied
     `latent_roles` dict (e.g., {'y0': 'scale', 'k': 'rate', ...}).

  3. Combine (roles, ops, num_latents, num_observed) into a 16-hex
     fingerprint via sha256.

Claim under validation (see tests/test_signature.py):
  - All 4 exp_decay domains produce the SAME fingerprint (isomorphism
    within class).
  - All 3 saturation domains produce the SAME fingerprint.
  - The exp_decay fingerprint differs from the saturation fingerprint
    (inter-class separation): saturation's mu involves a Sub op
    (1 - exp(-kt)) that exp_decay's mu (y0·exp(-kt)) lacks.

This is intentionally simpler than the Markov-category / GNN machinery
Plan §Phase 4 hints at — a principled graph-kernel fingerprint that
gives us a working retrieval/clustering endpoint in P4.3 before we
invest in heavier representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

try:
    from pytensor.graph.traversal import ancestors  # pytensor >= 2.30
except ImportError:
    from pytensor.graph.basic import ancestors
from pytensor.tensor.elemwise import Elemwise


@dataclass(frozen=True)
class StructuralSignature:
    roles: tuple            # sorted tuple of role labels, e.g. ('noise','obs','rate','scale')
    ops: tuple              # sorted tuple of op class names on observed-RV ancestor graph
    num_latents: int
    num_observed: int
    fingerprint: str        # 16-hex sha256 digest of the above

    def matches(self, other: "StructuralSignature") -> bool:
        return self.fingerprint == other.fingerprint


def _op_name(op) -> str:
    """Canonical name for a PyTensor op. Elemwise wrappers are unwrapped
    to their underlying scalar op so that (e.g.) all exp-ops share one
    label regardless of dtype."""
    if isinstance(op, Elemwise):
        return type(op.scalar_op).__name__
    return type(op).__name__


def _collect_ops(root) -> list[str]:
    ops: list[str] = []
    for v in ancestors([root]):
        if v.owner is None:
            continue
        ops.append(_op_name(v.owner.op))
    return ops


def extract_signature(model, latent_roles: dict[str, str]) -> StructuralSignature:
    """Build a StructuralSignature from a PyMC model and its role dict.

    Parameters
    ----------
    model : pymc.Model
    latent_roles : dict mapping RV name → role label

    Only RV names that appear in the model are used (unknown keys in
    latent_roles are ignored so a shared role vocabulary can be reused
    across classes).
    """
    free_rv_names = [rv.name for rv in model.free_RVs]
    obs_rv_names = [rv.name for rv in model.observed_RVs]
    all_rv_names = set(free_rv_names) | set(obs_rv_names)

    roles = tuple(sorted(
        role for name, role in latent_roles.items() if name in all_rv_names
    ))

    op_list: list[str] = []
    for rv in model.observed_RVs:
        op_list.extend(_collect_ops(rv))
    ops = tuple(sorted(op_list))

    payload = repr((roles, ops, len(free_rv_names), len(obs_rv_names))).encode()
    fp = hashlib.sha256(payload).hexdigest()[:16]

    return StructuralSignature(
        roles=roles,
        ops=ops,
        num_latents=len(free_rv_names),
        num_observed=len(obs_rv_names),
        fingerprint=fp,
    )


def signature_for_domain(domain_module, t_obs, y_obs) -> StructuralSignature:
    """Convenience: build the PyMC model from a law-zoo domain module and
    return its signature using the domain's LATENT_ROLES."""
    model = domain_module.build_model(t_obs, y_obs)
    return extract_signature(model, domain_module.LATENT_ROLES)
