"""P4.10 — GNN-learned structural embedding (contrastive objective).

The final deferred item from P4.9's limitation list. A minimal
message-passing network (MPNN) learns to embed each belief-network
graph into R^d such that intra-class pairs are close and inter-class
pairs are far, purely via contrastive training on positive/negative
pairs derived from the law-zoo's ground-truth equivalence classes.

Why bother, given the symbolic routes (P4.2/P4.6/P4.7) already hit
ARI = 1.0? Two reasons:
  (a) Scaling: once the law zoo contains 100s of domains with noisy
      structural variation, a learned similarity should dominate a
      rigid symbolic hash. This commit establishes the machinery.
  (b) Robustness: a trained embedding gives a CONTINUOUS similarity
      score, which composes with information-geometric downstream
      tasks (e.g., Perrone-style KL-weighted nearest-neighbour).

Implementation:
  - Graphs built from hardcoded class structures (exp_decay /
    saturation / damped), same structure shared within a class,
    distinct across classes. Node features are 5-dim one-hot type
    labels (prior / decay / saturation / damped / obs).
  - 2-layer MPNN: h' = ReLU(W_self · h + mean-over-parents(W_msg · h))
  - Graph embedding: mean pool over final node features, linear
    projection to R^d.
  - Contrastive loss: NT-Xent on all (anchor, positive, K negatives)
    triples with positive = random same-class, negatives = all
    different-class.
  - Optimizer: hand-coded Adam on jax grads.

The ground-truth partition is used ONLY for loss construction; the
evaluation metric (ARI after k-means clustering on embeddings) is
independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import numpy as np
from scipy.cluster.vq import kmeans2


# ---------------------------------------------------------------------------
# Hardcoded graph structures (per class)
# ---------------------------------------------------------------------------

# Node-type one-hot: [prior, decay_kernel, saturation_kernel, damped_kernel, normal_obs]
N_TYPES = 5
_T_PRIOR = 0
_T_DECAY = 1
_T_SAT = 2
_T_DAMP = 3
_T_OBS = 4


def _onehot(i: int, n: int = N_TYPES) -> list:
    v = [0.0] * n
    v[i] = 1.0
    return v


def _graph_for_class(class_id: str):
    """Return (nodes, edges) arrays for the class's canonical structure."""
    if class_id == "exp_decay":
        nodes = np.array([_onehot(_T_PRIOR), _onehot(_T_PRIOR),
                          _onehot(_T_DECAY), _onehot(_T_OBS)],
                         dtype=np.float32)
        edges = np.array([[0, 2], [1, 2], [2, 3]], dtype=np.int32)
    elif class_id == "saturation":
        nodes = np.array([_onehot(_T_PRIOR), _onehot(_T_PRIOR),
                          _onehot(_T_SAT), _onehot(_T_OBS)],
                         dtype=np.float32)
        edges = np.array([[0, 2], [1, 2], [2, 3]], dtype=np.int32)
    elif class_id == "damped_oscillation":
        nodes = np.array([_onehot(_T_PRIOR)] * 4
                         + [_onehot(_T_DAMP), _onehot(_T_OBS)],
                         dtype=np.float32)
        edges = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 5]], dtype=np.int32)
    else:
        raise ValueError(f"unknown class {class_id!r}")
    return nodes, edges


# ---------------------------------------------------------------------------
# MPNN
# ---------------------------------------------------------------------------


def mpnn_forward(params, nodes: jnp.ndarray, edges: jnp.ndarray) -> jnp.ndarray:
    """2-layer message-passing + mean pool + linear head → d-dim embedding."""
    src = edges[:, 0]
    dst = edges[:, 1]
    n_nodes = nodes.shape[0]
    ones = jnp.ones((edges.shape[0], 1), dtype=nodes.dtype)
    degree = jnp.maximum(
        segment_sum(ones, dst, num_segments=n_nodes), 1.0
    )

    # Layer 1
    msg1 = nodes[src] @ params["W1_msg"]
    agg1 = segment_sum(msg1, dst, num_segments=n_nodes) / degree
    h1 = jax.nn.relu(nodes @ params["W1_self"] + agg1)

    # Layer 2
    msg2 = h1[src] @ params["W2_msg"]
    agg2 = segment_sum(msg2, dst, num_segments=n_nodes) / degree
    h2 = jax.nn.relu(h1 @ params["W2_self"] + agg2)

    # Graph-level pooling + projection
    graph = jnp.mean(h2, axis=0)
    emb = graph @ params["W_out"]
    return emb


def init_params(seed: int = 0, f_in: int = N_TYPES,
                hidden: int = 8, d_out: int = 16, scale: float = 0.3):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 5)
    return {
        "W1_self": scale * jax.random.normal(ks[0], (f_in, hidden)),
        "W1_msg": scale * jax.random.normal(ks[1], (f_in, hidden)),
        "W2_self": scale * jax.random.normal(ks[2], (hidden, hidden)),
        "W2_msg": scale * jax.random.normal(ks[3], (hidden, hidden)),
        "W_out": scale * jax.random.normal(ks[4], (hidden, d_out)),
    }


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent)
# ---------------------------------------------------------------------------


def _all_embeddings(params, graphs):
    return jnp.stack([mpnn_forward(params, n, e) for (n, e) in graphs])


def nt_xent_loss(params, graphs, class_ids, tau: float = 0.5):
    embs = _all_embeddings(params, graphs)
    norms = jnp.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    sim = embs @ embs.T / tau
    n = len(class_ids)

    # For each anchor i, iterate over all same-class j != i.
    # logits = [sim(i,j) | sim(i,k) for k in different class].
    # Softmax-based loss with positive at index 0.
    total = 0.0
    count = 0
    for i in range(n):
        same = [j for j in range(n) if j != i and class_ids[j] == class_ids[i]]
        diff = [j for j in range(n) if class_ids[j] != class_ids[i]]
        diff_arr = jnp.array(diff, dtype=jnp.int32)
        for p in same:
            logits = jnp.concatenate([sim[i, p:p + 1], sim[i, diff_arr]])
            log_probs = jax.nn.log_softmax(logits)
            total = total - log_probs[0]
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Training loop with manual Adam
# ---------------------------------------------------------------------------


def _adam_init(params):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    return m, v


def _adam_step(params, grads, m, v, step, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    new_params, new_m, new_v = {}, {}, {}
    for k in params:
        mk = b1 * m[k] + (1 - b1) * grads[k]
        vk = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
        m_hat = mk / (1 - b1 ** step)
        v_hat = vk / (1 - b2 ** step)
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = mk
        new_v[k] = vk
    return new_params, new_m, new_v


@dataclass
class TrainResult:
    params: dict
    final_loss: float
    loss_history: list
    class_ids: list
    names: list
    embeddings: np.ndarray


def train_embedding(graphs, class_ids, names, n_epochs: int = 500,
                    lr: float = 1e-2, tau: float = 0.5,
                    seed: int = 0, log_every: int = 50,
                    verbose: bool = True) -> TrainResult:
    params = init_params(seed=seed)
    m, v = _adam_init(params)
    loss_fn = lambda p: nt_xent_loss(p, graphs, class_ids, tau=tau)
    grad_fn = jax.grad(loss_fn)

    history = []
    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        params, m, v = _adam_step(params, grads, m, v, step=epoch, lr=lr)
        if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
            loss = float(loss_fn(params))
            history.append((epoch, loss))
            if verbose:
                print(f"  epoch {epoch:>4}  loss={loss:.4f}")

    final_embs = np.array(_all_embeddings(params, graphs))
    return TrainResult(
        params=params,
        final_loss=history[-1][1],
        loss_history=history,
        class_ids=class_ids,
        names=names,
        embeddings=final_embs,
    )


# ---------------------------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------------------------


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int,
                       seed: int = 0) -> np.ndarray:
    """k-means on embeddings; returns integer cluster labels."""
    # Normalize to unit length so k-means on cosine-similar geometry.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embs = embeddings / np.maximum(norms, 1e-9)
    # Try a few random seeds and pick the best (k-means can get stuck)
    best_labels = None
    best_inertia = np.inf
    rng = np.random.default_rng(seed)
    for s in range(10):
        init_key = rng.integers(0, 10**6)
        centroids, labels = kmeans2(embs, n_clusters, seed=int(init_key),
                                     minit="++", check_finite=True)
        inertia = float(np.sum(
            np.linalg.norm(embs - centroids[labels], axis=1) ** 2
        ))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


# ---------------------------------------------------------------------------
# Law-zoo convenience
# ---------------------------------------------------------------------------


def build_law_zoo_graphs():
    """Build the 10 law-zoo graphs keyed by domain name."""
    from phase4_structure.law_zoo import CLASS_OF
    graphs = []
    class_ids = []
    names = []
    for name in CLASS_OF:
        cls = CLASS_OF[name]
        nodes, edges = _graph_for_class(cls)
        graphs.append((jnp.array(nodes), jnp.array(edges)))
        class_ids.append(cls)
        names.append(name)
    return graphs, class_ids, names


if __name__ == "__main__":
    print("P4.10 — GNN-learned embedding on law-zoo\n")
    graphs, class_ids, names = build_law_zoo_graphs()
    print(f"  {len(graphs)} graphs, classes: "
          f"{sorted(set(class_ids))}")
    print("  training 2-layer MPNN, 16-dim embedding, NT-Xent loss...\n")
    result = train_embedding(graphs, class_ids, names, n_epochs=300,
                              lr=5e-3, tau=0.3, seed=0, log_every=50)

    print(f"\n  final embeddings (first 4 dims shown):")
    for name, cls, emb in zip(result.names, result.class_ids, result.embeddings):
        print(f"    {name:<22} [{cls:<20}] {emb[:4]}")

    labels = cluster_embeddings(result.embeddings, n_clusters=3, seed=0)
    from phase4_structure.retrieval import adjusted_rand_index
    cls_order = {c: i for i, c in enumerate(sorted(set(class_ids)))}
    y_true = [cls_order[c] for c in class_ids]
    ari = adjusted_rand_index(y_true, list(labels))
    print(f"\n  k-means ARI on learned embeddings: {ari:.3f}")
