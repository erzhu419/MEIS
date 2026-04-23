"""MEIS Phase 1 Step 1 — Alice-Charlie height→weight environment.

Design notes
------------
Motivated by the MEIS plan §0.1 example: "Alice 比 Bob 高, Bob 和 Charlie 一样高. Alice
和 Charlie 谁体重重?" The env is a 1D regression (height → weight) structurally
parallel to boxing-gym's dugongs, but the ground-truth functional form encodes a
cross-domain physical prior: weight = density × volume, density ≈ const across
people, volume ≈ k · height³.

Ground-truth generative model (per `env.reset()`):
    theta  ~ Normal(mu=THETA_MU, sigma=THETA_SIGMA)    # one latent per env
    weight ~ Normal(mu=theta * height**3, sigma=OBS_NOISE)   # per observation

Sanity check at mean prior:
    theta = 0.01414  →  h=150cm → 47.7 kg; h=170 → 69.5 kg; h=190 → 96.9 kg
    (reasonable adult weights across the height range)

MEIS angle: an agent equipped with the cross-domain prior
"weight ≈ density · k · height³, density ≈ 1010 kg/m³ ± 3%"
collapses the 1-parameter regression to a near-zero-free-parameter model.
Baseline boxing-gym LLM has no such prior and must rediscover the cubic form
from 10 noisy samples — which it typically fails to do (see Step 0 results).

Import-only from boxing-gym (pip-editable); does NOT modify trunk.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
from scipy.stats import norm

from boxing_gym.envs.goal import Goal
from boxing_gym.agents.box_loop_helper import construct_dataframe


# --- Ground-truth population constants ---------------------------------
# Physical derivation: weight_kg = density_kg/m³ · volume_m³ = density · k · (height_cm)³
# with density ≈ 1010 kg/m³, k ≈ 1.4e-8 m³/cm³ (so that h=170 cm → ~70 kg).
#   k = 1.4e-8  ⇒ theta = density × k ≈ 1010 × 1.4e-8 = 1.414e-5 kg/cm³
# Sanity: theta · 170³ = 1.414e-5 · 4.913e6 ≈ 69.5 kg ✓
THETA_MU = 1.414e-5         # kg / cm³
THETA_SIGMA = 4.0e-7        # per-env std (≈2.8% relative) — tight human-population prior
OBS_NOISE = 2.0             # kg measurement noise
HEIGHT_LOWER = 150.0
HEIGHT_UPPER = 190.0


PRIOR = ("Adult humans of varying heights have weights that scale roughly with "
         "the cube of their height, modulated by a near-constant body density.")
NO_PRIOR = "You are observing a float response to a float input."


# =============================================================================
# Goal classes
# =============================================================================
class DirectGoal(Goal):
    """Regression goal: predict weight given a new height."""

    def __init__(self, env):
        super().__init__(env)
        self.eval_points: list[tuple[float, float]] = []
        self.eval_pointer = 0
        self.update_params_cache: dict = {}
        # Populated by get_norm_factors(); reasonable default for tests.
        self.norm_mu: float = 70.0
        self.norm_sigma: float = 12.0

    def get_system_message(self, include_prior: bool) -> str:
        if include_prior:
            goal_description = ("Your goal is to be able to predict the weight (kg) of an "
                                "adult human given their height (cm). Conduct experiments "
                                "to learn about the environment and make predictions based "
                                "on your observations.")
        else:
            goal_description = ("Your goal is to be able to predict the float response of "
                                "the environment to a given input. Conduct experiments to "
                                "learn about the environment and make predictions based on "
                                "your observations.")
        return self.env.generate_system_message(include_prior, goal_description)

    def get_goal_eval_question(self, include_prior: bool):
        # Draw a held-out eval height; bias sampling toward middle of range via Beta.
        if self.eval_pointer + 1 > len(self.eval_points):
            beta_draw = np.random.beta(2, 2)  # symmetric, middle-biased
            height = HEIGHT_LOWER + beta_draw * (HEIGHT_UPPER - HEIGHT_LOWER)
            weight = self.env.step(height)
            self.eval_points.append((height, weight))
        else:
            height, weight = self.eval_points[self.eval_pointer]
        self.eval_pointer += 1

        if include_prior:
            question = (f"Predict the weight (kg) of an adult human of height: {height:.2f} cm.\n"
                        "When asked to answer a question about the environment, respond in "
                        "the format specified below. Make assumptions about the environment "
                        "and provide your best guess.\n"
                        "Here is an example.\n"
                        "<thought> your thought </thought>\n"
                        "<answer>70</answer>")
        else:
            question = (f"Predict the float response to the following input: {height:.2f}.\n"
                        "When asked to answer a question about the environment, respond in "
                        "the format specified below. Make assumptions about the environment "
                        "and provide your best guess.\n"
                        "Here is an example.\n"
                        "<thought> your thought </thought>\n"
                        "<answer>70</answer>")
        return question, weight

    def evaluate_predictions(self, predictions, measurements):
        assert len(predictions) == len(measurements)
        parsed = []
        for p in predictions:
            try:
                parsed.append(float(str(p).strip()))
            except ValueError:
                parsed.append(float("nan"))
        errs = np.array(parsed) - np.array(measurements)
        return float(np.nanmean(np.abs(errs))), float(np.nanstd(np.abs(errs)))

    def expected_information_gain(self, query_point: float,
                                  num_outer_samples: int = 1000,
                                  num_inner_samples: int = 10) -> float:
        """Nested-MC EIG estimator for a proposed observation at `query_point` (height)."""
        height_query = float(query_point)
        existing = self.env.observed_data
        key = tuple((round(h, 3), round(w, 3)) for h, w in existing)
        if key not in self.update_params_cache:
            with pm.Model():
                theta_i = pm.Normal("theta_i", mu=self.env.theta_mu, sigma=self.env.theta_sigma)
                if len(existing) > 0:
                    heights_obs = pm.Data("h_obs", [h for h, _ in existing])
                    weights_obs = pm.Data("w_obs", [w for _, w in existing])
                    pm.Normal("obs", mu=theta_i * heights_obs ** 3,
                              sigma=OBS_NOISE, observed=weights_obs)
                total = num_outer_samples * num_inner_samples + num_outer_samples
                trace = pm.sample(total, tune=1000, return_inferencedata=False,
                                  progressbar=False, compute_convergence_checks=False)
            thetas = np.asarray(trace["theta_i"])
            rng = np.random.default_rng(0)
            rng.shuffle(thetas)
            self.update_params_cache[key] = thetas

        thetas = self.update_params_cache[key]
        outer = thetas[:num_outer_samples]
        log_ratios = []
        for n, theta_n in enumerate(outer):
            mu_n = theta_n * height_query ** 3
            w_sampled = np.random.normal(mu_n, OBS_NOISE)
            log_lik = norm.logpdf(w_sampled, mu_n, OBS_NOISE)
            inner = thetas[num_outer_samples + n * num_inner_samples:
                           num_outer_samples + (n + 1) * num_inner_samples]
            inner_mus = inner * height_query ** 3
            inner_logliks = norm.logpdf(w_sampled, inner_mus, OBS_NOISE)
            max_l = np.max(inner_logliks)
            log_marginal = max_l + np.log(np.mean(np.exp(inner_logliks - max_l)))
            log_ratios.append(log_lik - log_marginal)
        return float(np.mean(log_ratios))

    def get_norm_factors(self):
        """Calibrate baseline error by predicting marginal mean for many random inputs."""
        N = 10000
        measurements = []
        for i in range(N):
            if i % 10 == 0:
                self.env.reset()
            h = self.env.sample_random_input()
            measurements.append(self.env.step(h))
        mu = float(np.mean(measurements))
        preds = [str(mu)] * N
        err_mean, err_std = self.evaluate_predictions(preds, measurements)
        self.norm_mu = mu
        self.norm_sigma = err_std if err_std > 0 else 1.0
        return err_mean, err_std


class DirectGoalNaive(DirectGoal):
    """Novice variant (Scientist→Novice game): gets only the explanation, no direct data access."""

    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0

    def get_system_message(self, include_prior: bool) -> str:
        goal_description = ("Your goal is to conduct experiments and explain the environment "
                            "to the user so that they can achieve their goal. ")
        if include_prior:
            goal_description += ("The goal of the user is to be able to predict the weight (kg) "
                                 "of an adult human given their height (cm).")
        else:
            goal_description += ("The goal of the user is to be able to predict the float response "
                                 "of the environment to a given input.")
        return self.env.generate_system_message(include_prior, goal_description)

    def get_naive_system_message(self, include_prior: bool) -> str:
        if include_prior:
            goal_description = "Your goal is to predict the weight (kg) of an adult human given their height (cm)."
        else:
            goal_description = "Your goal is to predict the float response of the environment to a given input."
        format_instructions = (
            "You will be provided an input to this environment and will be tasked with "
            "predicting the output for each input.\n"
            "You must respond with a real number. You may also think before providing your predictions.\n"
            "Here is an example:\n"
            "<thought>your thought</thought>\n"
            "<answer>70</answer>"
        )
        return goal_description + "\n" + format_instructions + "\nHere is what you know about the environment:\n"

    def get_comm_prompt(self, include_prior: bool, com_limit: int = 300,
                        use_ppl: bool = False, str_prob_prog=None, params_summary_str=None) -> str:
        description = (
            "Assume that the user has no prior knowledge, and will not be able to run any "
            "experiments before making predictions.\n"
            "They will make predictions based solely on your explanation, so provide as "
            "much detail as possible. You CANNOT provide your own experiments or observations.\n"
            f"Limit your explanation to {com_limit} words."
        )
        if use_ppl:
            description += ("\nTo make your explanation clearer and more informative, look at the "
                            "statistical model (written in pymc) designed by a colleague for the "
                            "experimental data and the inferred parameters.\n"
                            f"Here is the statistical model.\n{str_prob_prog}\n"
                            f"Here are the inferred params.\n{params_summary_str}\n"
                            "Don't literally describe the model verbatim but use it to conceptually "
                            "motivate your explanation.\n"
                            "The agent will not be able to use the model explicitly but having a "
                            "conceptual understanding will be beneficial.")
        return description


# =============================================================================
# Env class
# =============================================================================
class AliceCharlie:
    """Height→weight 1D regression env parallel to boxing_gym.envs.Dugongs."""

    def __init__(self,
                 theta_mu: float = THETA_MU,
                 theta_sigma: float = THETA_SIGMA,
                 obs_noise: float = OBS_NOISE,
                 lower_limit: float = HEIGHT_LOWER,
                 upper_limit: float = HEIGHT_UPPER):
        self.theta_mu = theta_mu
        self.theta_sigma = theta_sigma
        self.obs_noise = obs_noise
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.env_name = "alice_charlie"
        self.include_prior = True  # set externally by run_experiment
        self.model = self._build_model()
        self.reset()

    def _build_model(self):
        with pm.Model() as model:
            pm.Normal("theta", mu=self.theta_mu, sigma=self.theta_sigma)
        return model

    def reset(self):
        with self.model:
            self.theta = float(pm.draw(self.model["theta"]))
        self.observed_data: list[tuple[float, float]] = []

    def generate_system_message(self, include_prior: bool = True, goal: str | None = None) -> str:
        assert goal is not None
        if include_prior:
            message = (f"{PRIOR}\n{goal}\n"
                       "Make observations by specifying a single height (cm) you want to observe "
                       "with a real number. The environment will return the weight of an adult "
                       "human at that height (kg).\n"
                       f"The height values are between {self.lower_limit} and {self.upper_limit}.\n"
                       "You may also think before providing your predictions.\n\n"
                       "Here is an example:\n"
                       "<thought> your thought </thought>\n"
                       "<observe>170</observe>\n"
                       "When asked to answer a question about the environment, respond in the "
                       "format specified in the question.\n"
                       "<thought> your thought </thought>\n"
                       "<answer> your answer </answer>\n")
        else:
            message = (f"{NO_PRIOR}\n{goal}\n"
                       "You may observe the value of the function one input value at a time. "
                       "Make observations by specifying a single value you want to observe with a float.\n"
                       "The environment will return the float response of the function at that input.\n"
                       f"The input values are between {self.lower_limit} and {self.upper_limit}.\n"
                       "You may also think before providing your predictions.\n\n"
                       "Here is an example:\n"
                       "<thought> your thought </thought>\n"
                       "<observe>170</observe>\n"
                       "When asked to answer a question about the environment, respond in the "
                       "format specified in the question.\n"
                       "Example:\n"
                       "<thought> your thought </thought>\n"
                       "<answer> your answer </answer>\n")
        return message

    def sample_random_input(self) -> float:
        return float(np.random.uniform(self.lower_limit, self.upper_limit))

    def step(self, height_cm: float) -> float:
        """Noisy weight observation: w = theta · h³ + Normal(0, obs_noise)."""
        h = float(height_cm)
        mu = self.theta * h ** 3
        return float(mu + np.random.normal(0.0, self.obs_noise))

    def validate_input(self, input_string: str):
        try:
            h = float(str(input_string).strip())
        except (ValueError, TypeError):
            return "Input must be a float."
        if h < self.lower_limit or h > self.upper_limit:
            return f"Input must be between {self.lower_limit} and {self.upper_limit}."
        return h

    def run_experiment(self, input_string: str):
        h = self.validate_input(input_string)
        if isinstance(h, str):
            return h, False
        w = self.step(h)
        self.observed_data.append((h, w))
        return w, True

    def get_data(self):
        return self.observed_data

    def get_df(self):
        self.df = construct_dataframe(self)

    def get_description(self):
        if self.include_prior:
            return "Heights (cm) and measured weights (kg) of adult humans."
        return "x and Y are the input and output values of the environment."

    def describe_data_columns(self):
        return self.format_column_description()

    def get_ordered_column_names(self):
        return ["x", "Y"]

    def get_ordered_features(self):
        return self.get_ordered_column_names()[:-1]

    def format_column_description(self):
        if self.include_prior:
            return ("The observations are:\n -Y: weight (kg) of an adult human\n"
                    "The input values are:\n -x: height (cm) of the person.\n"
                    "Use the input values to help you model the observations.")
        return ""


if __name__ == "__main__":
    env = AliceCharlie()
    goal = DirectGoal(env)
    print("env_name:", env.env_name)
    print("theta (this env instance):", env.theta)
    print("sample w @ h=170:", env.step(170.0))
    print("run_experiment('175'):", env.run_experiment("175"))
    print("run_experiment('200'):", env.run_experiment("200"))
    print("run_experiment('abc'):", env.run_experiment("abc"))
