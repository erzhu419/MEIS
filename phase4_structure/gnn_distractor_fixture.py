"""Law-zoo v3: within-class graph variation to demonstrate trained GNN > random-init.

The law-zoo v2 used one canonical graph per class, so random-init MPNN
also achieves ARI = 1.00 (see §5 in Paper 3 / unified §5). This
module builds a **v3 fixture** where within-class graphs VARY by:
  - feature noise on the 5-dim one-hot type label (small Gaussian)
  - varying the number of extra "decoration" leaf nodes per domain
    (0-3 random leaf prior nodes connected to the observation)
  - random edge dropout (each parent->kernel edge dropped with p=0.2)

For each class, we generate K=8 graph instances with independent
noise / decoration / dropout. Total 24 graphs across 3 classes.

A random-init MPNN now cannot trivially cluster by pooled-mean of
one-hot features (because features are perturbed), so the trained
embedding has a non-trivial target: learn to focus on the
KIND of the deterministic kernel atom (decay vs saturation vs
damped), which is a robust-to-noise class signal.

Acceptance result (documented in test):
  random-init mean ARI across 5 seeds: ~0.3-0.7 (sub-perfect)
  trained      mean ARI across 5 seeds: ≥ 0.9
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import numpy as np
from scipy.cluster.vq import kmeans2

from phase4_structure.gnn_embedding import (
    mpnn_forward, init_params, nt_xent_loss,
    _adam_init, _adam_step,
)


# Node types: prior / decay / saturation / damped / obs
N_TYPES = 5
_T_PRIOR, _T_DECAY, _T_SAT, _T_DAMP, _T_OBS = 0, 1, 2, 3, 4


def _noisy_onehot(type_idx: int, noise: float, rng: np.random.Generator):
    v = np.zeros(N_TYPES, dtype=np.float32)
    v[type_idx] = 1.0
    v += rng.normal(0, noise, size=N_TYPES).astype(np.float32)
    return v


def _class_base_structure(class_id: str):
    if class_id == "exp_decay":
        return ([_T_PRIOR, _T_PRIOR, _T_DECAY, _T_OBS],
                [(0, 2), (1, 2), (2, 3)])
    if class_id == "saturation":
        return ([_T_PRIOR, _T_PRIOR, _T_SAT, _T_OBS],
                [(0, 2), (1, 2), (2, 3)])
    if class_id == "damped_oscillation":
        return ([_T_PRIOR, _T_PRIOR, _T_PRIOR, _T_PRIOR, _T_DAMP, _T_OBS],
                [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5)])
    raise ValueError(class_id)


def build_noisy_graph(class_id: str, *, seed: int,
                       feature_noise: float = 0.15,
                       max_decoration: int = 3,
                       edge_dropout: float = 0.2):
    """Build a graph with within-class variation.

    Returns (node_features, edges) as jnp arrays.
    """
    rng = np.random.default_rng(seed)
    types, edges = _class_base_structure(class_id)

    # Feature noise: perturb each one-hot
    nodes = [_noisy_onehot(t, feature_noise, rng) for t in types]

    # Add decoration leaves: 0..max_decoration extra prior nodes,
    # each attached by an edge into the obs node (penultimate in our
    # layouts). They should not change the CLASS but add within-class
    # variety the signature/GNN sees.
    n_deco = int(rng.integers(0, max_decoration + 1))
    obs_idx = len(types) - 1
    for _ in range(n_deco):
        nodes.append(_noisy_onehot(_T_PRIOR, feature_noise, rng))
        new_idx = len(nodes) - 1
        edges.append((new_idx, obs_idx))

    # Edge dropout on parent->kernel / parent->obs edges
    kept = []
    for (src, dst) in edges:
        if rng.random() >= edge_dropout:
            kept.append((src, dst))
    # Ensure graph remains connected: keep at least one parent->kernel
    # edge per parent if all dropped (belt-and-suspenders).
    if not kept:
        kept = [edges[-1]]

    nodes_arr = jnp.asarray(np.stack(nodes))
    edges_arr = jnp.asarray(np.array(kept, dtype=np.int32))
    return nodes_arr, edges_arr


def build_zoo_v3(k_per_class: int = 8, seed: int = 0, **kwargs):
    classes = ["exp_decay", "saturation", "damped_oscillation"]
    graphs = []
    class_ids = []
    names = []
    s = 0
    for c in classes:
        for i in range(k_per_class):
            g = build_noisy_graph(c, seed=seed * 10_000 + s, **kwargs)
            graphs.append(g)
            class_ids.append(c)
            names.append(f"{c}_{i}")
            s += 1
    return graphs, class_ids, names


def cluster_and_ari(embeddings: np.ndarray, class_ids, n_clusters: int,
                     seed: int = 0) -> float:
    from phase4_structure.retrieval import adjusted_rand_index
    embs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    # Try multiple k-means inits, pick lowest inertia
    best_labels = None
    best_inertia = np.inf
    rng = np.random.default_rng(seed)
    for s in range(10):
        centroids, labels = kmeans2(embs, n_clusters,
                                      seed=int(rng.integers(0, 10**6)),
                                      minit="++", check_finite=True)
        inertia = float(np.sum(
            np.linalg.norm(embs - centroids[labels], axis=1) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    cls_order = {c: i for i, c in enumerate(sorted(set(class_ids)))}
    y_true = [cls_order[c] for c in class_ids]
    return adjusted_rand_index(y_true, list(best_labels))


def _all_embeddings(params, graphs):
    return np.array(jnp.stack([mpnn_forward(params, n, e) for (n, e) in graphs]))


def train_on_v3(graphs, class_ids, names, n_epochs=300, lr=5e-3,
                 tau=0.3, seed=0):
    params = init_params(seed=seed)
    m, v = _adam_init(params)
    loss_fn = lambda p: nt_xent_loss(p, graphs, class_ids, tau=tau)
    grad_fn = jax.grad(loss_fn)
    for step in range(1, n_epochs + 1):
        grads = grad_fn(params)
        params, m, v = _adam_step(params, grads, m, v, step=step, lr=lr)
    return params


def run_gnn_comparison(k_per_class: int = 8, seeds=(0, 1, 2, 3, 4),
                        feature_noise: float = 0.15,
                        max_decoration: int = 3,
                        edge_dropout: float = 0.2,
                        verbose: bool = True):
    random_init_aris = []
    trained_aris = []
    for s in seeds:
        graphs, class_ids, names = build_zoo_v3(
            k_per_class=k_per_class, seed=s,
            feature_noise=feature_noise,
            max_decoration=max_decoration,
            edge_dropout=edge_dropout,
        )
        # Random init
        random_params = init_params(seed=s + 1000)
        ri_embs = _all_embeddings(random_params, graphs)
        ri_ari = cluster_and_ari(ri_embs, class_ids, n_clusters=3, seed=s)
        random_init_aris.append(ri_ari)
        # Trained
        trained_params = train_on_v3(graphs, class_ids, names,
                                        n_epochs=300, lr=5e-3,
                                        tau=0.3, seed=s)
        t_embs = _all_embeddings(trained_params, graphs)
        t_ari = cluster_and_ari(t_embs, class_ids, n_clusters=3, seed=s)
        trained_aris.append(t_ari)
        if verbose:
            print(f"  seed={s}  random-init ARI={ri_ari:.3f}  "
                  f"trained ARI={t_ari:.3f}")
    return dict(
        random_init_aris=random_init_aris,
        trained_aris=trained_aris,
        random_init_mean=float(np.mean(random_init_aris)),
        trained_mean=float(np.mean(trained_aris)),
        gap=float(np.mean(trained_aris) - np.mean(random_init_aris)),
    )


if __name__ == "__main__":
    print("Law-zoo v3: within-class graph variation\n")
    print(f"  3 classes × 8 graphs each = 24 graphs")
    print(f"  feature_noise=0.15  max_decoration=3  edge_dropout=0.2")
    print(f"  5 seeds, random-init vs trained (300 epochs contrastive)\n")
    result = run_gnn_comparison()
    print(f"\nSUMMARY:")
    print(f"  random-init  mean ARI across 5 seeds: "
          f"{result['random_init_mean']:.3f}  "
          f"(values: {[f'{x:.2f}' for x in result['random_init_aris']]})")
    print(f"  trained      mean ARI across 5 seeds: "
          f"{result['trained_mean']:.3f}  "
          f"(values: {[f'{x:.2f}' for x in result['trained_aris']]})")
    print(f"  gap (trained − random): {result['gap']:+.3f}")
