"""rlutils — lightweight RL teaching utilities for CPAI_study.

Convenience imports so that ``from rlutils import GaussianBandit`` works.
"""

from .envs import GaussianBandit
from .utils import argmax_random_tie
from .policies import (
    epsilon_greedy_policy,
    naive_epsilon_greedy_policy,
    boltzmann_policy,
    select_ucb_action,
)
from .methods import (
    uniform_random_method,
    value_based_method,
    ucb_method,
    run_experiment,
)
from .plotting import (
    plot_epsilon_greedy_policy,
    plot_boltzmann_policy,
    compare_policies,
)

__all__ = [
    "GaussianBandit",
    "argmax_random_tie",
    "epsilon_greedy_policy",
    "naive_epsilon_greedy_policy",
    "boltzmann_policy",
    "select_ucb_action",
    "uniform_random_method",
    "value_based_method",
    "ucb_method",
    "run_experiment",
    "plot_epsilon_greedy_policy",
    "plot_boltzmann_policy",
    "compare_policies",
]
