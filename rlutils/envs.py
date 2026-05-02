"""Bandit environments for educational use."""

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats as _scipy_stats
except ImportError:  # scipy is optional at import time
    _scipy_stats = None


class GaussianBandit:
    """Simple multi-armed Gaussian bandit for educational use.

    The parameters mu and sigma are randomly generated based on the initial
    seed to provide different experiment setups for different students.

    Parameters
    ----------
    seed : int or None
        Random seed used to generate the environment parameters (mu, sigma)
        and to initialise the reward RNG.
    """

    def __init__(self, seed=None):
        param_rng = np.random.default_rng(seed)
        self.n_actions = 5
        self.mu = param_rng.uniform(0.0, 10.0, size=self.n_actions)
        self.sigma = param_rng.uniform(0.5, 2.0, size=self.n_actions)
        self.reset(seed=seed)

    # --- core API ---
    def reset(self, seed=None):
        """Reset the reward RNG."""
        self.rng = np.random.default_rng(seed)
        return None

    def step(self, action):
        """Return a scalar reward drawn from N(mu[action], sigma[action])."""
        return float(self.rng.normal(self.mu[action], self.sigma[action]))

    # --- utilities ---
    @property
    def optimal_action(self):
        """Index of the arm with the highest mean reward."""
        return int(np.argmax(self.mu))

    def report_settings(self):
        """Print the environment settings for a report."""
        print("--- Bandit Environment Report ---")
        print(f"Number of Actions: {self.n_actions}")
        for i in range(self.n_actions):
            print(f"Action {i}: mean = {self.mu[i]:.4f}, "
                  f"std = {self.sigma[i]:.4f}")
        print(f"Optimal Action: {self.optimal_action}")
        print("---------------------------------")

    def plot(self):
        """Plot the reward distributions for all arms."""
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(np.min(self.mu) - 4, np.max(self.mu) + 4, 500)
        for i, mean in enumerate(self.mu):
            if _scipy_stats is None:
                raise ImportError("scipy is required for plot(). "
                                  "Install it with: pip install scipy")
            pdf = _scipy_stats.norm.pdf(x, loc=mean, scale=self.sigma[i])
            ax.plot(x, pdf, label=f"$a{i}$")
        ax.legend()
        ax.set_xlabel("Reward (r)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Reward Distributions per Action")
        ax.grid()
        plt.show()
