"""P5.1 — Transition matrix inference (Plan §Phase 5 task 1).

Target: Frobenius error ||P_true - P_hat||_F < 0.1.

Setup: a discrete-state Markov chain with K states emits a sequence of
observations {s_0, s_1, ..., s_N}. The system must infer the K×K
transition matrix P where P[i,j] = Pr(s_{t+1} = j | s_t = i).

Approach: Dirichlet-Categorical conjugate. The posterior over each row
of P given a uniform Dirichlet(α) prior and transition counts n[i, :]
is Dirichlet(α + n[i, :]); we report its posterior mean as P_hat.

  P_hat[i, j] = (α + n[i, j]) / Σ_k (α + n[i, k])

α = 1 corresponds to Laplace smoothing; α → 0 recovers the MLE.

Sanity: Frobenius error scales ~1/√N for large N when rows are
non-degenerate. With K=3 and N=500 transitions we expect error ~0.05
under iid sampling assumptions — comfortably under the 0.1 target.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TransitionMatrixResult:
    K: int
    n_transitions: int
    P_true: np.ndarray
    P_hat: np.ndarray
    frobenius_error: float
    dirichlet_alpha: float


def simulate_chain(P_true: np.ndarray, n_steps: int,
                   rng: np.random.Generator,
                   start: int = 0) -> np.ndarray:
    """Simulate a K-state discrete Markov chain for n_steps observations
    (i.e. n_steps - 1 transitions)."""
    K = P_true.shape[0]
    if not (0 <= start < K):
        raise ValueError(f"start state {start} out of range [0, {K})")
    states = np.empty(n_steps, dtype=int)
    states[0] = start
    for t in range(1, n_steps):
        states[t] = rng.choice(K, p=P_true[states[t - 1]])
    return states


def infer_transition_matrix(sequence: np.ndarray, K: int,
                            dirichlet_alpha: float = 1.0) -> np.ndarray:
    """Posterior mean of the row-wise Dirichlet-Categorical."""
    counts = np.full((K, K), dirichlet_alpha, dtype=float)
    transitions = np.column_stack([sequence[:-1], sequence[1:]])
    for src, dst in transitions:
        counts[src, dst] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))


def run_task1_benchmark(K: int = 3, n_steps: int = 500,
                        seed: int = 0,
                        P_true: np.ndarray | None = None,
                        dirichlet_alpha: float = 1.0
                        ) -> TransitionMatrixResult:
    rng = np.random.default_rng(seed)
    if P_true is None:
        P_true = rng.dirichlet(np.ones(K), size=K)
    elif P_true.shape != (K, K):
        raise ValueError(f"P_true shape {P_true.shape} != ({K}, {K})")
    seq = simulate_chain(P_true, n_steps, rng)
    P_hat = infer_transition_matrix(seq, K, dirichlet_alpha=dirichlet_alpha)
    err = frobenius_error(P_true, P_hat)
    return TransitionMatrixResult(
        K=K, n_transitions=n_steps - 1,
        P_true=P_true, P_hat=P_hat,
        frobenius_error=err,
        dirichlet_alpha=dirichlet_alpha,
    )


if __name__ == "__main__":
    print("P5.1 transition-matrix benchmark\n")
    for K, N in [(3, 500), (4, 500), (5, 1000), (3, 50)]:
        errs = []
        for seed in range(5):
            r = run_task1_benchmark(K=K, n_steps=N, seed=seed)
            errs.append(r.frobenius_error)
        print(f"  K={K}  N={N:>4}  Frobenius err over 5 seeds: "
              f"mean={np.mean(errs):.4f}  max={np.max(errs):.4f}  "
              f"(target < 0.1)")
