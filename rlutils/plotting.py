"""Shared plotting helpers for bandit notebooks."""

import numpy as np
import matplotlib.pyplot as plt

from .policies import epsilon_greedy_policy, naive_epsilon_greedy_policy
from .policies import boltzmann_policy


def plot_epsilon_greedy_policy(Q, epsilon=0.1):
    """Visualise Q-values alongside the epsilon-greedy distribution."""
    policy = epsilon_greedy_policy(Q, epsilon)
    action = np.arange(Q.shape[0])

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].bar(action, Q)
    axarr[0].set_xlabel("action a")
    axarr[0].set_ylabel("action value Q(a)")
    axarr[1].bar(action, policy,
                 label=r"$\epsilon$ = %3.1f" % epsilon)
    axarr[1].set_xlabel("action a")
    axarr[1].set_ylabel(r"policy $\pi$ (a)")
    plt.legend()
    plt.show()


def plot_boltzmann_policy(Q, beta=1.0):
    """Visualise Q-values alongside the Boltzmann distribution."""
    policy = boltzmann_policy(Q, beta)
    action = np.arange(Q.shape[0])

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].bar(action, Q)
    axarr[0].set_xlabel("action a")
    axarr[0].set_ylabel("action value Q(a)")
    axarr[1].bar(action, policy,
                 label=r"$\beta$ = %3.1f" % beta)
    axarr[1].set_xlabel("action a")
    axarr[1].set_ylabel(r"policy $\pi$ (a)")
    plt.legend()
    plt.show()


def compare_policies(Q, epsilon=0.1):
    """Compare random-tie-break vs naive epsilon-greedy side by side."""
    p_random = epsilon_greedy_policy(Q, epsilon)
    p_naive = naive_epsilon_greedy_policy(Q, epsilon)
    action = np.arange(Q.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(action - width / 2, p_random, width, label="Random Tie-break")
    ax.bar(action + width / 2, p_naive, width,
           label="Naive (Smallest Index)")
    ax.set_xlabel("Action a")
    ax.set_ylabel("Probability")
    ax.set_title(f"Policy Comparison (epsilon={epsilon})")
    ax.set_xticks(action)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
