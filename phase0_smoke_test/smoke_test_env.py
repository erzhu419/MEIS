"""Phase 0 smoke test: 3-node Gaussian Bayesian network in NumPyro.

Corresponds to MEIS plan 本周 item #1 ("搭建 NumPyro 环境，跑通一个最简单的贝叶斯网络(3 节点)").

Model: A -> B -> C
    A ~ Normal(0, 1)
    B | A ~ Normal(A, 0.5)
    C | B ~ Normal(B, 0.5)
Observe C = 2.0, infer posterior over A, B.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model(c_obs: float | None = None):
    a = numpyro.sample("A", dist.Normal(0.0, 1.0))
    b = numpyro.sample("B", dist.Normal(a, 0.5))
    numpyro.sample("C", dist.Normal(b, 0.5), obs=c_obs)


def main():
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=4000, num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), c_obs=2.0)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    a_mean, a_std = float(samples["A"].mean()), float(samples["A"].std())
    b_mean, b_std = float(samples["B"].mean()), float(samples["B"].std())

    # Analytic posterior for this linear-Gaussian chain:
    #   P(A | C=c) = Normal(c * var_A / (var_A + var_B + var_C), ...)
    # With var_A=1, var_B=var_C=0.25 -> posterior mean of A = 2.0 * 1 / 1.5 = 1.333...
    print(f"\n[Result] A posterior mean = {a_mean:.3f} (analytic ~ 1.333)")
    print(f"[Result] B posterior mean = {b_mean:.3f} (analytic ~ 1.667)")
    print(f"[Result] A posterior std  = {a_std:.3f}")
    print(f"[Result] B posterior std  = {b_std:.3f}")

    ok_a = abs(a_mean - 4.0 / 3.0) < 0.05
    ok_b = abs(b_mean - 5.0 / 3.0) < 0.05
    print(f"\n[PASS] NumPyro environment works." if (ok_a and ok_b) else "[FAIL] Posterior off.")


if __name__ == "__main__":
    main()
