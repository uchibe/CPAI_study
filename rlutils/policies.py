"""Action-selection policies for bandit problems."""

import numpy as np
from .utils import argmax_random_tie


# ------------------------------------------------------------------
# epsilon-greedy
# ------------------------------------------------------------------
def epsilon_greedy_policy(Q, epsilon=0.1):
    """Epsilon-greedy policy with random tie-breaking.

    Returns a probability vector over actions.
    """
    n_actions = Q.shape[0]
    policy = epsilon / n_actions * np.ones(n_actions)
    max_val = np.max(Q)
    best = np.flatnonzero(Q == max_val)
    policy[best] += (1 - epsilon) / len(best)
    return policy


def naive_epsilon_greedy_policy(Q, epsilon=0.1):
    """Naive epsilon-greedy using ``np.argmax`` (biased tie-break).

    Provided for comparison / teaching purposes.
    """
    n_actions = Q.shape[0]
    policy = np.ones(n_actions) * (epsilon / n_actions)
    best_action = np.argmax(Q)
    policy[best_action] += 1.0 - epsilon
    return policy


# ------------------------------------------------------------------
# Boltzmann (softmax)
# ------------------------------------------------------------------
def boltzmann_policy(Q, beta=1.0):
    """Boltzmann (softmax) policy.

    Parameters
    ----------
    Q : ndarray
        Action-value estimates (1-D).
    beta : float
        Inverse temperature (higher = more greedy).
    """
    Qmax = np.max(Q)
    expQ = np.exp(beta * (Q - Qmax))
    return expQ / np.sum(expQ)


# ------------------------------------------------------------------
# UCB
# ------------------------------------------------------------------
def select_ucb_action(Q_values, N_values, t, c=2.0, rng=None):
    """Select an action using the UCB1 rule.

    Parameters
    ----------
    Q_values : ndarray
        Current action-value estimates.
    N_values : ndarray
        Per-action visit counts.
    t : int
        Current total step count (>= 1).
    c : float
        Exploration parameter.
    rng : numpy.random.Generator or None
        For tie-breaking.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_actions = Q_values.shape[0]

    untried = np.where(N_values == 0)[0]
    if len(untried) > 0:
        return int(rng.choice(untried))

    ucb_scores = np.zeros(n_actions)
    for a in range(n_actions):
        ucb_scores[a] = Q_values[a] + c * np.sqrt(np.log(t) / N_values[a])

    return argmax_random_tie(ucb_scores, rng=rng)
